use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use im::{HashMap as PersistentHashMap, HashSet as PersistentHashSet, Vector};

use crate::domain::conversation::{ConversationState, Message, Role, refresh_message_search_text};
use crate::domain::ids::MessageId;
use crate::domain::session_document::{
    ConversationItemId, ConversationItemView, ConversationRole, DisplayOrdinals, ItemStatus,
    ProjectionDelta, SessionDocument, TrajectoryItemId,
};
use crate::domain::trajectory::{TrajectoryProjection, TrajectoryRecord};

/// One immutable, revision-stamped desktop read model. Both visible surfaces
/// are materialized from the same `SessionDocument` snapshot before this value
/// is published, so the UI cannot observe a new conversation with an old
/// trajectory (or vice versa).
#[derive(Clone, Debug, Default)]
pub(crate) struct SessionView {
    pub(crate) revision: u64,
    pub(crate) conversation: ConversationState,
    pub(crate) trajectory: TrajectoryProjection,
    conversation_revision: u64,
    tool_schema_fingerprint: u64,
    conversation_indices: PersistentHashMap<ConversationItemId, usize>,
    claimed_message_keys: PersistentHashSet<MessageId>,
    #[cfg(test)]
    materialized_messages: usize,
}

impl SessionView {
    pub(crate) fn from_document(
        document: &SessionDocument,
        title: &str,
        tool_schemas: &HashMap<String, String>,
        previous: Option<&Self>,
    ) -> Self {
        let revision = document.revisions().document;
        let trajectory = previous.map_or_else(
            || TrajectoryProjection::from_document(document),
            |view| TrajectoryProjection::from_document_reusing(document, Some(&view.trajectory)),
        );
        debug_assert_eq!(revision, trajectory.source_revision());
        let revisions = document.revisions();
        let tool_schema_fingerprint = tool_schema_fingerprint(tool_schemas);
        let reuse_conversation = previous.is_some_and(|previous| {
            previous.conversation_revision == revisions.conversation
                && previous.tool_schema_fingerprint == tool_schema_fingerprint
        });
        let messages = if let Some(previous) = previous
            && previous.conversation_revision == revisions.conversation
            && previous.tool_schema_fingerprint == tool_schema_fingerprint
        {
            previous.conversation.messages.clone()
        } else {
            let ordinals = document.display_ordinals();
            let trajectory_by_id = trajectory
                .records
                .iter()
                .map(|record| (record.id.clone(), record.as_ref()))
                .collect::<HashMap<_, _>>();
            let previous_messages = previous
                .into_iter()
                .flat_map(|view| view.conversation.messages.iter())
                .map(|message| (message.key, message))
                .collect::<HashMap<_, _>>();

            let mut claimed_keys = HashSet::new();
            document
                .conversation()
                .into_iter()
                .map(|item| {
                    let key = unique_message_key(item.id, &mut claimed_keys);
                    let timing = trajectory_id(item.id)
                        .as_ref()
                        .and_then(|id| trajectory_by_id.get(id).copied());
                    let previous = previous_messages.get(&key).copied();
                    materialize_message(item, key, ordinals, timing, tool_schemas, previous)
                })
                .collect::<Vector<_>>()
        };
        let conversation_indices = if reuse_conversation {
            previous
                .expect("reused conversation has a previous view")
                .conversation_indices
                .clone()
        } else {
            document
                .conversation_ids()
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, id)| (id, index))
                .collect()
        };
        let claimed_message_keys = if reuse_conversation {
            previous
                .expect("reused conversation has a previous view")
                .claimed_message_keys
                .clone()
        } else {
            messages.iter().map(|message| message.key).collect()
        };
        #[cfg(test)]
        let materialized_messages = if reuse_conversation {
            previous
                .expect("reused conversation has a previous view")
                .materialized_messages
        } else {
            messages.len()
        };

        let stats = document.stats();
        let conversation = ConversationState {
            tool_calls: trajectory
                .records
                .iter()
                .filter(|record| matches!(&record.id, TrajectoryItemId::Tool(_)))
                .count(),
            messages,
            title: display_title(title),
            turns: stats.turns,
        };

        Self {
            revision,
            conversation,
            trajectory,
            conversation_revision: revisions.conversation,
            tool_schema_fingerprint,
            conversation_indices,
            claimed_message_keys,
            #[cfg(test)]
            materialized_messages,
        }
    }

    pub(crate) fn after_delta(
        document: &SessionDocument,
        delta: &ProjectionDelta,
        title: &str,
        tool_schemas: &HashMap<String, String>,
        previous: &Self,
    ) -> Self {
        let tool_schema_fingerprint = tool_schema_fingerprint(tool_schemas);
        if previous.tool_schema_fingerprint != tool_schema_fingerprint {
            return Self::from_document(document, title, tool_schemas, Some(previous));
        }

        let revisions = document.revisions();
        let trajectory = TrajectoryProjection::after_delta(document, delta, &previous.trajectory);
        let ordinals = document.display_ordinals();
        let mut messages = previous.conversation.messages.clone();
        let mut conversation_indices = previous.conversation_indices.clone();
        let mut claimed_message_keys = previous.claimed_message_keys.clone();
        let appended = if delta.conversation_order_changed {
            appended_conversation_ids(document, delta, previous)
        } else {
            Some(&[][..])
        };
        let Some(appended) = appended else {
            return Self::from_document(document, title, tool_schemas, Some(previous));
        };
        let mut materialized = 0_usize;
        for id in appended {
            let Some(item) = document.conversation_by_id(id) else {
                return Self::from_document(document, title, tool_schemas, Some(previous));
            };
            let key = unique_persistent_message_key(id, &mut claimed_message_keys);
            let timing = trajectory_id(id)
                .as_ref()
                .and_then(|trajectory_id| trajectory.record_by_id(trajectory_id));
            let message = materialize_message(item, key, ordinals, timing, tool_schemas, None);
            conversation_indices.insert(id.clone(), messages.len());
            messages.push_back(message);
            materialized = materialized.saturating_add(1);
        }
        for id in &delta.changed_conversation {
            if appended.contains(id) {
                continue;
            }
            let Some(index) = previous.conversation_indices.get(id).copied() else {
                return Self::from_document(document, title, tool_schemas, Some(previous));
            };
            let Some(item) = document.conversation_by_id(id) else {
                return Self::from_document(document, title, tool_schemas, Some(previous));
            };
            let timing = trajectory_id(id)
                .as_ref()
                .and_then(|id| trajectory.record_by_id(id));
            let prior = messages.get(index);
            let message = materialize_message(
                item,
                prior.map_or_else(
                    || unique_message_key(id, &mut HashSet::new()),
                    |item| item.key,
                ),
                ordinals,
                timing,
                tool_schemas,
                prior,
            );
            materialized = materialized.saturating_add(1);
            messages.set(index, message);
        }

        let stats = document.stats();
        Self {
            revision: revisions.document,
            conversation: ConversationState {
                messages,
                title: display_title(title),
                turns: stats.turns,
                tool_calls: previous.conversation.tool_calls.saturating_add(
                    appended
                        .iter()
                        .filter(|id| matches!(id, ConversationItemId::Tool(_)))
                        .count(),
                ),
            },
            trajectory,
            conversation_revision: revisions.conversation,
            tool_schema_fingerprint,
            conversation_indices,
            claimed_message_keys,
            #[cfg(test)]
            materialized_messages: previous.materialized_messages.saturating_add(materialized),
        }
    }

    #[cfg(test)]
    pub(crate) fn materialized_messages(&self) -> usize {
        self.materialized_messages
    }
}

fn appended_conversation_ids<'a>(
    document: &'a SessionDocument,
    delta: &ProjectionDelta,
    previous: &SessionView,
) -> Option<&'a [ConversationItemId]> {
    let ids = document.conversation_ids();
    let previous_len = previous.conversation.messages.len();
    let suffix = ids.get(previous_len..)?;
    if suffix.is_empty()
        || suffix
            .iter()
            .any(|id| previous.conversation_indices.contains_key(id))
        || suffix
            .iter()
            .any(|id| !delta.changed_conversation.contains(id))
    {
        return None;
    }
    Some(suffix)
}

fn materialize_message(
    item: ConversationItemView<'_>,
    key: MessageId,
    ordinals: &DisplayOrdinals,
    trajectory: Option<&TrajectoryRecord>,
    tool_schemas: &HashMap<String, String>,
    previous: Option<&Arc<Message>>,
) -> Arc<Message> {
    let turn = item
        .turn_id
        .and_then(|turn_id| ordinals.turn(turn_id))
        .unwrap_or_default() as usize;
    let step = item
        .turn_id
        .zip(item.step_id)
        .and_then(|(turn_id, step_id)| ordinals.step(turn_id, step_id))
        .unwrap_or_default() as usize;
    let role = role_snapshot(item.role);
    let title = item
        .title
        .map(ToOwned::to_owned)
        .or_else(|| match item.role {
            ConversationRole::Steering => Some("Steering".to_owned()),
            ConversationRole::Context => Some("Context".to_owned()),
            _ => None,
        });
    let tool_call_id = match item.id {
        ConversationItemId::Tool(call_id) => Some(call_id.to_string()),
        _ => None,
    };
    let request_id = match item.id {
        ConversationItemId::ResponseSegment { request_id, .. } => Some(request_id.to_string()),
        _ => item.step_id.map(ToString::to_string),
    };
    let schema = (role == Role::Tool)
        .then(|| {
            title
                .as_deref()
                .and_then(|name| tool_schemas.get(name))
                .cloned()
        })
        .flatten();
    let started_at_ms = trajectory
        .and_then(|record| record.timing.started.as_ref())
        .and_then(|time| u128::try_from(time.wall_time_ms()).ok());
    let duration_ms = trajectory
        .and_then(|record| record.timing.duration_ns())
        .map(|duration| u128::from(duration) / 1_000_000);

    let mut message = Message {
        key,
        revision: previous.map_or(0, |message| message.revision),
        role,
        tool_call_id,
        title,
        text: item.text.to_owned(),
        payload: item.payload.map(ToOwned::to_owned),
        schema,
        pending: matches!(item.status, ItemStatus::Pending | ItemStatus::Running),
        failed: matches!(
            item.status,
            ItemStatus::Failed
                | ItemStatus::Aborted
                | ItemStatus::Denied
                | ItemStatus::NotExecuted
                | ItemStatus::Unknown
        ),
        started_at_ms,
        duration_ms,
        turn,
        step,
        request_id,
        search_text: String::new(),
    };
    refresh_message_search_text(&mut message);

    if let Some(previous) = previous {
        if same_canonical_message(previous, &message) {
            return Arc::clone(previous);
        }
        message.revision = previous.revision.saturating_add(1);
    }
    Arc::new(message)
}

fn same_canonical_message(previous: &Message, current: &Message) -> bool {
    previous.key == current.key
        && previous.role == current.role
        && previous.tool_call_id == current.tool_call_id
        && previous.title == current.title
        && previous.text == current.text
        && previous.payload == current.payload
        && previous.schema == current.schema
        && previous.pending == current.pending
        && previous.failed == current.failed
        && previous.started_at_ms == current.started_at_ms
        && previous.duration_ms == current.duration_ms
        && previous.turn == current.turn
        && previous.step == current.step
        && previous.request_id == current.request_id
        && previous.search_text == current.search_text
}

fn role_snapshot(role: ConversationRole) -> Role {
    match role {
        ConversationRole::User | ConversationRole::Steering => Role::User,
        ConversationRole::Context | ConversationRole::Notice => Role::Notice,
        ConversationRole::Reasoning => Role::Reasoning,
        ConversationRole::Assistant => Role::Assistant,
        ConversationRole::Tool => Role::Tool,
    }
}

fn trajectory_id(id: &ConversationItemId) -> Option<TrajectoryItemId> {
    match id {
        ConversationItemId::Input(id) => Some(TrajectoryItemId::Input(id.clone())),
        ConversationItemId::ResponseSegment { request_id, .. } => {
            Some(TrajectoryItemId::Assistant(request_id.clone()))
        }
        ConversationItemId::Tool(id) => Some(TrajectoryItemId::Tool(id.clone())),
        ConversationItemId::Compaction(id) => Some(TrajectoryItemId::Compaction(id.clone())),
    }
}

fn display_title(title: &str) -> String {
    if title == "Untitled session" || title.trim().is_empty() {
        "New chat".to_owned()
    } else {
        title.to_owned()
    }
}

fn unique_message_key(id: &ConversationItemId, claimed: &mut HashSet<MessageId>) -> MessageId {
    let mut hash = semantic_hash(id);
    loop {
        let key = MessageId(hash.max(1));
        if claimed.insert(key) {
            return key;
        }
        hash = hash.wrapping_mul(1_099_511_628_211).wrapping_add(1);
    }
}

fn unique_persistent_message_key(
    id: &ConversationItemId,
    claimed: &mut PersistentHashSet<MessageId>,
) -> MessageId {
    let mut hash = semantic_hash(id);
    loop {
        let key = MessageId(hash.max(1));
        if claimed.insert(key).is_none() {
            return key;
        }
        hash = hash.wrapping_mul(1_099_511_628_211).wrapping_add(1);
    }
}

fn semantic_hash(id: &ConversationItemId) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;

    let mut hash = OFFSET;
    let mut write = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    match id {
        ConversationItemId::Input(id) => {
            write(b"input\0");
            write(id.as_str().as_bytes());
        }
        ConversationItemId::ResponseSegment {
            request_id,
            ordinal,
        } => {
            write(b"response\0");
            write(request_id.as_str().as_bytes());
            write(&ordinal.to_le_bytes());
        }
        ConversationItemId::Tool(id) => {
            write(b"tool\0");
            write(id.as_str().as_bytes());
        }
        ConversationItemId::Compaction(id) => {
            write(b"compaction\0");
            write(id.as_str().as_bytes());
        }
    }
    hash
}

fn tool_schema_fingerprint(tool_schemas: &HashMap<String, String>) -> u64 {
    let mut entries = tool_schemas.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let mut hash = 14_695_981_039_346_656_037_u64;
    for (name, schema) in entries {
        for byte in name.bytes().chain([0]).chain(schema.bytes()).chain([0]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use kcastle_agent::{
        AssistantChunk, EasyInputMessage, EventTime, InputId, InputItem, InputOrigin,
        RecordedEvent, RequestHeaderReason, RequestId, ResponseInfo, RunId, RunOutcome,
        SessionConfig, SessionEvent, StepId, StepOutcome, TokenUsage, TurnEndReason, TurnId, TxId,
    };

    use super::*;

    fn recorded(seq: u64, event: SessionEvent) -> RecordedEvent {
        RecordedEvent {
            seq,
            tx_id: TxId::from_raw(format!("tx-{seq}")),
            time: EventTime {
                wall_time_ms: i64::try_from(seq).unwrap_or(i64::MAX),
                clock_id: "session-view-test".to_owned(),
                monotonic_ns: seq.saturating_mul(1_000_000),
            },
            event,
        }
    }

    fn fixture_events() -> Vec<RecordedEvent> {
        let run_id = RunId::from("run-1");
        let turn_id = TurnId::from("turn-1");
        let step_id = StepId::from("step-1");
        let input_id = InputId::from("input-1");
        let request_id = RequestId::from("request-1");
        let usage = TokenUsage {
            uncached_input_tokens: 10,
            cache_read_input_tokens: 40,
            cache_write_input_tokens: 2,
            output_tokens: 5,
            reasoning_output_tokens: 3,
        };
        let input = |text: &str| InputItem::from(EasyInputMessage::from(text));
        vec![
            recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: run_id.clone(),
                },
            ),
            recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id: run_id.clone(),
                    turn_id: turn_id.clone(),
                },
            ),
            recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id: turn_id.clone(),
                    step_id: step_id.clone(),
                },
            ),
            recorded(
                3,
                SessionEvent::InputSubmitted {
                    input_id: input_id.clone(),
                    input: "question".to_owned(),
                    origin: InputOrigin::Initial,
                },
            ),
            recorded(
                4,
                SessionEvent::InputAttached {
                    input_id,
                    step_id: step_id.clone(),
                    items: vec![input("question")],
                },
            ),
            recorded(
                5,
                SessionEvent::RequestSnapshot {
                    request_id: request_id.clone(),
                    step_id: step_id.clone(),
                    reason: RequestHeaderReason::Initial,
                    model: "fixture".to_owned(),
                    instructions: Some("test".to_owned()),
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: None,
                    session_config: SessionConfig::default(),
                },
            ),
            recorded(
                6,
                SessionEvent::ModelRequestStarted {
                    request_id: request_id.clone(),
                },
            ),
            recorded(
                7,
                SessionEvent::AssistantChunk {
                    request_id: request_id.clone(),
                    chunk: AssistantChunk::OutputTextDelta {
                        delta: "answer".to_owned(),
                    },
                },
            ),
            recorded(
                8,
                SessionEvent::AssistantCompleted {
                    request_id,
                    items: vec![input("answer")],
                    response: ResponseInfo {
                        id: "response-1".to_owned(),
                        model: "fixture".to_owned(),
                        usage: Some(usage),
                    },
                },
            ),
            recorded(
                9,
                SessionEvent::StepTerminated {
                    step_id,
                    outcome: StepOutcome::Completed,
                    error: None,
                },
            ),
            recorded(
                10,
                SessionEvent::TurnTerminated {
                    turn_id,
                    reason: TurnEndReason::Completed,
                },
            ),
            recorded(
                11,
                SessionEvent::RunTerminated {
                    run_id,
                    outcome: RunOutcome::Completed,
                    error: None,
                },
            ),
        ]
    }

    fn fixture() -> SessionDocument {
        SessionDocument::from_events(fixture_events()).unwrap()
    }

    #[test]
    fn conversation_and_trajectory_share_one_revision() {
        let document = fixture();
        let view = SessionView::from_document(&document, "Fixture", &HashMap::new(), None);
        assert_eq!(view.revision, document.revisions().document);
        assert_eq!(view.revision, view.trajectory.source_revision());
    }

    #[test]
    fn stable_keys_reuse_canonical_arcs() {
        let document = fixture();
        let first = SessionView::from_document(&document, "Fixture", &HashMap::new(), None);
        let assistant = first
            .conversation
            .messages
            .iter()
            .position(|message| message.role == Role::Assistant)
            .unwrap();
        let original_key = first.conversation.messages[assistant].key;
        let assistant_record = first
            .trajectory
            .records
            .iter()
            .position(|record| record.kind == crate::domain::TrajectoryKind::Assistant)
            .unwrap();
        assert_eq!(
            first.trajectory.records[assistant_record].id,
            TrajectoryItemId::Assistant(RequestId::from("request-1"))
        );
        let second =
            SessionView::from_document(&document, "Renamed", &HashMap::new(), Some(&first));
        assert_eq!(second.conversation.messages[assistant].key, original_key);
        assert!(Arc::ptr_eq(
            &first.conversation.messages[assistant],
            &second.conversation.messages[assistant]
        ));
        assert!(Arc::ptr_eq(
            &first.trajectory.records[assistant_record],
            &second.trajectory.records[assistant_record]
        ));
    }

    #[test]
    fn every_incremental_prefix_matches_a_full_rebuild() {
        let schemas = HashMap::new();
        let mut document = SessionDocument::default();
        let mut incremental = SessionView::from_document(&document, "Fixture", &schemas, None);
        for event in fixture_events() {
            let delta = document.apply_batch(vec![event]).unwrap();
            incremental =
                SessionView::after_delta(&document, &delta, "Fixture", &schemas, &incremental);
            let rebuilt = SessionView::from_document(&document, "Fixture", &schemas, None);
            assert_eq!(incremental.revision, rebuilt.revision);
            assert_eq!(incremental.trajectory.records, rebuilt.trajectory.records);
            assert_eq!(
                incremental.conversation.messages.len(),
                rebuilt.conversation.messages.len()
            );
            for (actual, expected) in incremental
                .conversation
                .messages
                .iter()
                .zip(rebuilt.conversation.messages.iter())
            {
                assert!(same_canonical_message(actual, expected));
            }
        }
    }

    #[test]
    fn title_and_stats_come_from_the_same_document_snapshot() {
        let document = fixture();
        let view = SessionView::from_document(&document, "Fixture", &HashMap::new(), None);
        assert_eq!(view.conversation.title, "Fixture");
        assert_eq!(view.conversation.turns, 1);
        assert_eq!(view.conversation.tool_calls, 0);
        assert_eq!(view.trajectory.stats().turns, 1);
        assert_eq!(view.trajectory.stats().input_tokens, 52);

        let untitled =
            SessionView::from_document(&document, "Untitled session", &HashMap::new(), None);
        assert_eq!(untitled.conversation.title, "New chat");
    }

    #[test]
    fn ten_thousand_item_stream_updates_touch_only_the_changed_stable_id() {
        const HISTORY: usize = 10_000;
        const DELTAS: usize = 1_000;

        let run_id = RunId::from("large-run");
        let turn_id = TurnId::from("large-turn");
        let step_id = StepId::from("large-step");
        let request_id = RequestId::from("large-request");
        let mut events = vec![
            recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: run_id.clone(),
                },
            ),
            recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id,
                    turn_id: turn_id.clone(),
                },
            ),
            recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id,
                    step_id: step_id.clone(),
                },
            ),
        ];
        let mut seq = 3_u64;
        for index in 0..HISTORY {
            events.push(recorded(
                seq,
                SessionEvent::InputSubmitted {
                    input_id: InputId::from_raw(format!("history-{index}")),
                    input: format!("history item {index}"),
                    origin: InputOrigin::Queue,
                },
            ));
            seq = seq.saturating_add(1);
        }
        events.push(recorded(
            seq,
            SessionEvent::RequestSnapshot {
                request_id: request_id.clone(),
                step_id,
                reason: RequestHeaderReason::Initial,
                model: "fixture".to_owned(),
                instructions: None,
                tools: Vec::new(),
                reasoning_effort: None,
                max_output_tokens: None,
                session_config: SessionConfig::default(),
            },
        ));
        seq = seq.saturating_add(1);
        events.push(recorded(
            seq,
            SessionEvent::ModelRequestStarted {
                request_id: request_id.clone(),
            },
        ));
        seq = seq.saturating_add(1);
        events.push(recorded(
            seq,
            SessionEvent::AssistantChunk {
                request_id: request_id.clone(),
                chunk: AssistantChunk::OutputTextDelta {
                    delta: "x".to_owned(),
                },
            },
        ));

        let mut document = SessionDocument::from_events(events).unwrap();
        let schemas = HashMap::new();
        let mut view = SessionView::from_document(&document, "Large", &schemas, None);
        let untouched_message = Arc::clone(&view.conversation.messages[HISTORY / 2]);
        let untouched_record = Arc::clone(&view.trajectory.records[HISTORY / 2]);
        let messages_before = view.materialized_messages();
        let records_before = view.trajectory.materialized_records();
        let ordinal_observations = document.display_ordinals().observations();

        for _ in 0..DELTAS {
            let seq = document.cursor().next_seq;
            let delta = document
                .apply_batch(vec![recorded(
                    seq,
                    SessionEvent::AssistantChunk {
                        request_id: request_id.clone(),
                        chunk: AssistantChunk::OutputTextDelta {
                            delta: "x".to_owned(),
                        },
                    },
                )])
                .unwrap();
            view = SessionView::after_delta(&document, &delta, "Large", &schemas, &view);
        }

        assert_eq!(
            view.materialized_messages() - messages_before,
            DELTAS,
            "each receipt may materialize only its changed conversation ID"
        );
        assert_eq!(
            view.trajectory.materialized_records() - records_before,
            DELTAS,
            "each receipt may materialize only its changed trajectory ID"
        );
        assert_eq!(
            document.display_ordinals().observations(),
            ordinal_observations,
            "text-only receipts must not revisit ordinal assignment"
        );
        assert!(Arc::ptr_eq(
            &untouched_message,
            &view.conversation.messages[HISTORY / 2]
        ));
        assert!(Arc::ptr_eq(
            &untouched_record,
            &view.trajectory.records[HISTORY / 2]
        ));

        let rebuilt = SessionView::from_document(&document, "Large", &schemas, None);
        assert_eq!(view.revision, rebuilt.revision);
        assert_eq!(view.trajectory.records, rebuilt.trajectory.records);
        assert_eq!(
            view.conversation.messages.len(),
            rebuilt.conversation.messages.len()
        );
        for (incremental, full) in view
            .conversation
            .messages
            .iter()
            .zip(rebuilt.conversation.messages.iter())
        {
            assert!(same_canonical_message(incremental, full));
        }
    }
}
