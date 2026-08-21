use std::collections::{HashMap, HashSet};

use kcastle_agent::{
    AssistantChunk, EventTime, RecordedEvent, SessionEvent, StepOutcome, ToolResultStatus,
    UserMessageMode,
};

use crate::domain::MessageId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryLane {
    Input,
    Model,
    Tools,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryKind {
    System,
    User,
    Context,
    Steering,
    Assistant,
    Tool,
    Compaction,
    RequestFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryStatus {
    Running,
    Completed,
    Failed,
    Denied,
    NotExecuted,
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryUsage {
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) cached_tokens: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryStats {
    pub(crate) turns: usize,
    pub(crate) steps: usize,
    pub(crate) llm_ns: u64,
    pub(crate) tool_ns: u64,
    pub(crate) ttft_ns: u64,
    pub(crate) ttft_steps: usize,
    pub(crate) decode_ns: u64,
    pub(crate) decode_tokens: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) cached_tokens: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct RecordTiming {
    pub(crate) started: Option<EventTime>,
    pub(crate) first_token: Option<EventTime>,
    pub(crate) execution_started: Option<EventTime>,
    pub(crate) execution_finished: Option<EventTime>,
    pub(crate) completed: Option<EventTime>,
}

impl RecordTiming {
    pub(crate) fn duration_ns(&self) -> Option<u64> {
        self.completed
            .as_ref()?
            .duration_since(self.started.as_ref()?)
    }

    pub(crate) fn ttft_ns(&self) -> Option<u64> {
        self.first_token
            .as_ref()?
            .duration_since(self.started.as_ref()?)
    }

    pub(crate) fn generation_ns(&self) -> Option<u64> {
        self.completed
            .as_ref()?
            .duration_since(self.first_token.as_ref()?)
    }

    pub(crate) fn execution_ns(&self) -> Option<u64> {
        self.execution_finished
            .as_ref()?
            .duration_since(self.execution_started.as_ref()?)
    }

    pub(crate) fn pre_execution_ns(&self) -> Option<u64> {
        self.execution_started
            .as_ref()?
            .duration_since(self.started.as_ref()?)
    }

    pub(crate) fn post_execution_ns(&self) -> Option<u64> {
        self.completed
            .as_ref()?
            .duration_since(self.execution_finished.as_ref()?)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrajectoryRecord {
    pub(crate) id: MessageId,
    pub(crate) source_seq: u64,
    pub(crate) kind: TrajectoryKind,
    pub(crate) lane: TrajectoryLane,
    pub(crate) title: String,
    pub(crate) text: String,
    pub(crate) payload: Option<String>,
    pub(crate) raw: String,
    pub(crate) turn: Option<u32>,
    pub(crate) step: Option<u32>,
    pub(crate) call_id: Option<String>,
    pub(crate) status: TrajectoryStatus,
    pub(crate) timing: RecordTiming,
    pub(crate) usage: Option<TrajectoryUsage>,
}

impl TrajectoryRecord {
    pub(crate) fn matches(&self, query: &str) -> bool {
        query.is_empty()
            || self.title.to_lowercase().contains(query)
            || self.text.to_lowercase().contains(query)
            || self.raw.to_lowercase().contains(query)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryProjection {
    pub(crate) records: Vec<TrajectoryRecord>,
    stats: TrajectoryStats,
    revision: u64,
    step_starts: HashMap<(u32, u32), (u64, EventTime)>,
    request_starts: HashMap<(u32, u32), (u64, EventTime)>,
    completed_steps: HashSet<(u32, u32)>,
    completed_turns: HashSet<u32>,
    assistants: HashMap<(u32, u32), usize>,
    tools: HashMap<String, usize>,
    compactions: HashMap<String, usize>,
}

impl TrajectoryProjection {
    pub(crate) fn from_events(events: &[RecordedEvent]) -> Self {
        let mut projection = Self::default();
        for event in events {
            projection.apply(event);
        }
        projection
    }

    pub(crate) fn apply(&mut self, recorded: &RecordedEvent) {
        self.revision = self.revision.saturating_add(1);
        match &recorded.event {
            SessionEvent::StepStart { turn, step } => {
                self.step_starts
                    .insert((*turn, *step), (recorded.seq, recorded.time.clone()));
            }
            SessionEvent::ModelRequestStart { turn, step } => {
                self.request_starts
                    .insert((*turn, *step), (recorded.seq, recorded.time.clone()));
            }
            SessionEvent::RequestHeader {
                turn,
                step,
                model,
                instructions,
                tools,
                ..
            } => self.push_record(TrajectoryRecord {
                id: record_id(recorded.seq),
                source_seq: recorded.seq,
                kind: TrajectoryKind::System,
                lane: TrajectoryLane::Input,
                title: "System".into(),
                text: format!("{model} · {} tools", tools.len()),
                payload: Some(instructions.clone()),
                raw: pretty(recorded),
                turn: Some(*turn),
                step: Some(*step),
                call_id: None,
                status: TrajectoryStatus::Completed,
                timing: point_timing(&recorded.time),
                usage: None,
            }),
            SessionEvent::UserMessage {
                turn,
                step,
                mode,
                items,
                ..
            } => {
                let (kind, title) = match mode {
                    UserMessageMode::Initial | UserMessageMode::Queue => {
                        (TrajectoryKind::User, "User")
                    }
                    UserMessageMode::Steer => (TrajectoryKind::Steering, "Steering"),
                    UserMessageMode::Context => (TrajectoryKind::Context, "Context"),
                };
                self.push_record(TrajectoryRecord {
                    id: record_id(recorded.seq),
                    source_seq: recorded.seq,
                    kind,
                    lane: TrajectoryLane::Input,
                    title: title.into(),
                    text: items_text(items),
                    payload: None,
                    raw: pretty(recorded),
                    turn: Some(*turn),
                    step: Some(*step),
                    call_id: None,
                    status: TrajectoryStatus::Completed,
                    timing: point_timing(&recorded.time),
                    usage: None,
                });
            }
            SessionEvent::AssistantChunk { turn, step, chunk } => {
                let index = self.ensure_assistant(*turn, *step, recorded);
                self.update_record(index, |record| {
                    if record.timing.first_token.is_none() && chunk.is_non_empty_token() {
                        record.timing.first_token = Some(recorded.time.clone());
                    }
                    match chunk {
                        AssistantChunk::OutputTextDelta { delta }
                        | AssistantChunk::ReasoningTextDelta { delta } => {
                            record.text.push_str(delta)
                        }
                        AssistantChunk::ToolCallArgumentsDelta { delta, .. } => {
                            record
                                .payload
                                .get_or_insert_with(String::new)
                                .push_str(delta);
                        }
                        AssistantChunk::Usage { usage } => {
                            record.usage = Some(usage_snapshot(usage));
                        }
                    }
                    record.raw = pretty(recorded);
                });
            }
            SessionEvent::AssistantMessage {
                turn,
                step,
                items,
                response,
            } => {
                let index = self.ensure_assistant(*turn, *step, recorded);
                self.update_record(index, |record| {
                    if record.text.is_empty() {
                        record.text = items_text(items);
                    }
                    record.status = TrajectoryStatus::Completed;
                    record.timing.completed = Some(recorded.time.clone());
                    record.usage = response.usage.as_ref().map(usage_snapshot);
                    record.raw = pretty(recorded);
                });
            }
            SessionEvent::ToolCall {
                turn,
                step,
                call_id,
                name,
                arguments,
                ..
            } => {
                let index = self.records.len();
                self.tools.insert(call_id.clone(), index);
                self.push_record(TrajectoryRecord {
                    id: record_id(recorded.seq),
                    source_seq: recorded.seq,
                    kind: TrajectoryKind::Tool,
                    lane: TrajectoryLane::Tools,
                    title: name.clone(),
                    text: arguments.clone(),
                    payload: Some(arguments.clone()),
                    raw: pretty(recorded),
                    turn: Some(*turn),
                    step: Some(*step),
                    call_id: Some(call_id.clone()),
                    status: TrajectoryStatus::Running,
                    timing: RecordTiming {
                        started: Some(recorded.time.clone()),
                        ..RecordTiming::default()
                    },
                    usage: None,
                });
            }
            SessionEvent::ToolExecutionStart { call_id } => {
                if let Some(index) = self.tool_index(call_id) {
                    self.update_record(index, |record| {
                        record.timing.execution_started = Some(recorded.time.clone());
                    });
                }
            }
            SessionEvent::ToolExecutionFinish { call_id, .. } => {
                if let Some(index) = self.tool_index(call_id) {
                    self.update_record(index, |record| {
                        record.timing.execution_finished = Some(recorded.time.clone());
                    });
                }
            }
            SessionEvent::ToolResult {
                call_id,
                output,
                status,
                ..
            } => {
                if let Some(index) = self.tool_index(call_id) {
                    self.update_record(index, |record| {
                        record.text = output.clone();
                        record.timing.completed = Some(recorded.time.clone());
                        record.status = match status {
                            ToolResultStatus::Success => TrajectoryStatus::Completed,
                            ToolResultStatus::Error => TrajectoryStatus::Failed,
                            ToolResultStatus::Denied => TrajectoryStatus::Denied,
                            ToolResultStatus::NotFound
                            | ToolResultStatus::AbortedBeforeDispatch => {
                                TrajectoryStatus::NotExecuted
                            }
                            ToolResultStatus::UnknownSideEffects => TrajectoryStatus::Unknown,
                        };
                        record.raw = pretty(recorded);
                    });
                }
            }
            SessionEvent::CompactionStart {
                compaction_id,
                tokens_before,
                ..
            } => {
                let index = self.records.len();
                self.compactions.insert(compaction_id.clone(), index);
                self.push_record(TrajectoryRecord {
                    id: record_id(recorded.seq),
                    source_seq: recorded.seq,
                    kind: TrajectoryKind::Compaction,
                    lane: TrajectoryLane::Model,
                    title: "Compaction".into(),
                    text: format!("{tokens_before} tokens before compaction"),
                    payload: None,
                    raw: pretty(recorded),
                    turn: None,
                    step: None,
                    call_id: None,
                    status: TrajectoryStatus::Running,
                    timing: RecordTiming {
                        started: Some(recorded.time.clone()),
                        ..RecordTiming::default()
                    },
                    usage: None,
                });
            }
            SessionEvent::CompactionEnd {
                compaction_id,
                summary,
                outcome,
                response,
                ..
            } => {
                if let Some(index) = self.compactions.get(compaction_id).copied() {
                    self.update_record(index, |record| {
                        record.text = summary.clone();
                        record.timing.completed = Some(recorded.time.clone());
                        record.status = if *outcome == StepOutcome::Completed {
                            TrajectoryStatus::Completed
                        } else {
                            TrajectoryStatus::Failed
                        };
                        record.usage = response
                            .as_ref()
                            .and_then(|response| response.usage.as_ref())
                            .map(usage_snapshot);
                        record.raw = pretty(recorded);
                    });
                }
            }
            SessionEvent::StepEnd {
                turn,
                step,
                outcome,
                error,
            } => {
                self.completed_turns.insert(*turn);
                self.completed_steps.insert((*turn, *step));
                if *outcome != StepOutcome::Completed {
                    if let Some(index) = self.assistants.get(&(*turn, *step)).copied() {
                        self.update_record(index, |record| {
                            record.status = TrajectoryStatus::Failed;
                            record.timing.completed.get_or_insert(recorded.time.clone());
                        });
                    } else {
                        let (source_seq, started) = self
                            .request_starts
                            .get(&(*turn, *step))
                            .cloned()
                            .or_else(|| self.step_starts.get(&(*turn, *step)).cloned())
                            .unwrap_or((recorded.seq, recorded.time.clone()));
                        self.push_record(TrajectoryRecord {
                            id: record_id(source_seq),
                            source_seq,
                            kind: TrajectoryKind::RequestFailure,
                            lane: TrajectoryLane::Model,
                            title: "Request failed".into(),
                            text: error
                                .clone()
                                .unwrap_or_else(|| "Request interrupted".into()),
                            payload: None,
                            raw: pretty(recorded),
                            turn: Some(*turn),
                            step: Some(*step),
                            call_id: None,
                            status: TrajectoryStatus::Failed,
                            timing: RecordTiming {
                                started: Some(started),
                                completed: Some(recorded.time.clone()),
                                ..RecordTiming::default()
                            },
                            usage: None,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn stats(&self) -> TrajectoryStats {
        TrajectoryStats {
            turns: self.completed_turns.len(),
            steps: self.completed_steps.len(),
            ..self.stats
        }
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    fn ensure_assistant(&mut self, turn: u32, step: u32, recorded: &RecordedEvent) -> usize {
        if let Some(index) = self.assistants.get(&(turn, step)).copied() {
            return index;
        }
        let (source_seq, started) = self
            .request_starts
            .get(&(turn, step))
            .cloned()
            .or_else(|| self.step_starts.get(&(turn, step)).cloned())
            .unwrap_or((recorded.seq, recorded.time.clone()));
        let index = self.records.len();
        self.assistants.insert((turn, step), index);
        self.push_record(TrajectoryRecord {
            id: record_id(source_seq),
            source_seq,
            kind: TrajectoryKind::Assistant,
            lane: TrajectoryLane::Model,
            title: "Assistant".into(),
            text: String::new(),
            payload: None,
            raw: pretty(recorded),
            turn: Some(turn),
            step: Some(step),
            call_id: None,
            status: TrajectoryStatus::Running,
            timing: RecordTiming {
                started: Some(started),
                ..RecordTiming::default()
            },
            usage: None,
        });
        index
    }

    fn tool_index(&self, call_id: &str) -> Option<usize> {
        self.tools.get(call_id).copied()
    }

    fn push_record(&mut self, record: TrajectoryRecord) {
        add_stats(&mut self.stats, record_stats(&record));
        self.records.push(record);
    }

    fn update_record(&mut self, index: usize, update: impl FnOnce(&mut TrajectoryRecord)) {
        let before = record_stats(&self.records[index]);
        update(&mut self.records[index]);
        let after = record_stats(&self.records[index]);
        replace_stats(&mut self.stats, before, after);
    }
}

fn record_stats(record: &TrajectoryRecord) -> TrajectoryStats {
    let mut stats = TrajectoryStats::default();
    if let Some(usage) = record.usage {
        stats.input_tokens = u64::from(usage.input_tokens);
        stats.output_tokens = u64::from(usage.output_tokens);
        stats.cached_tokens = u64::from(usage.cached_tokens);
    }
    match record.kind {
        TrajectoryKind::Assistant => {
            stats.llm_ns = record.timing.duration_ns().unwrap_or_default();
            if let Some(ttft_ns) = record.timing.ttft_ns() {
                stats.ttft_ns = ttft_ns;
                stats.ttft_steps = 1;
            }
            if let Some(decode_ns) = record.timing.generation_ns()
                && let Some(usage) = record.usage
            {
                stats.decode_ns = decode_ns;
                stats.decode_tokens = u64::from(usage.output_tokens);
            }
        }
        TrajectoryKind::Tool => {
            stats.tool_ns = record.timing.duration_ns().unwrap_or_default();
        }
        _ => {}
    }
    stats
}

fn add_stats(target: &mut TrajectoryStats, value: TrajectoryStats) {
    target.llm_ns = target.llm_ns.saturating_add(value.llm_ns);
    target.tool_ns = target.tool_ns.saturating_add(value.tool_ns);
    target.ttft_ns = target.ttft_ns.saturating_add(value.ttft_ns);
    target.ttft_steps = target.ttft_steps.saturating_add(value.ttft_steps);
    target.decode_ns = target.decode_ns.saturating_add(value.decode_ns);
    target.decode_tokens = target.decode_tokens.saturating_add(value.decode_tokens);
    target.input_tokens = target.input_tokens.saturating_add(value.input_tokens);
    target.output_tokens = target.output_tokens.saturating_add(value.output_tokens);
    target.cached_tokens = target.cached_tokens.saturating_add(value.cached_tokens);
}

fn replace_stats(
    target: &mut TrajectoryStats,
    previous: TrajectoryStats,
    current: TrajectoryStats,
) {
    target.llm_ns = target
        .llm_ns
        .saturating_sub(previous.llm_ns)
        .saturating_add(current.llm_ns);
    target.tool_ns = target
        .tool_ns
        .saturating_sub(previous.tool_ns)
        .saturating_add(current.tool_ns);
    target.ttft_ns = target
        .ttft_ns
        .saturating_sub(previous.ttft_ns)
        .saturating_add(current.ttft_ns);
    target.ttft_steps = target
        .ttft_steps
        .saturating_sub(previous.ttft_steps)
        .saturating_add(current.ttft_steps);
    target.decode_ns = target
        .decode_ns
        .saturating_sub(previous.decode_ns)
        .saturating_add(current.decode_ns);
    target.decode_tokens = target
        .decode_tokens
        .saturating_sub(previous.decode_tokens)
        .saturating_add(current.decode_tokens);
    target.input_tokens = target
        .input_tokens
        .saturating_sub(previous.input_tokens)
        .saturating_add(current.input_tokens);
    target.output_tokens = target
        .output_tokens
        .saturating_sub(previous.output_tokens)
        .saturating_add(current.output_tokens);
    target.cached_tokens = target
        .cached_tokens
        .saturating_sub(previous.cached_tokens)
        .saturating_add(current.cached_tokens);
}

fn point_timing(time: &EventTime) -> RecordTiming {
    RecordTiming {
        started: Some(time.clone()),
        completed: Some(time.clone()),
        ..RecordTiming::default()
    }
}

fn usage_snapshot(usage: &kcastle_agent::ResponseUsage) -> TrajectoryUsage {
    TrajectoryUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_tokens: usage.input_tokens_details.cached_tokens,
    }
}

fn record_id(seq: u64) -> MessageId {
    MessageId(seq.saturating_add(1))
}

fn pretty<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

fn items_text<T: serde::Serialize>(items: &T) -> String {
    let Ok(value) = serde_json::to_value(items) else {
        return String::new();
    };
    let mut text = Vec::new();
    collect_text(&value, &mut text);
    if text.is_empty() {
        pretty(items)
    } else {
        text.join("\n")
    }
}

fn collect_text(value: &serde_json::Value, text: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(object) => {
            for (key, value) in object {
                if matches!(key.as_str(), "text" | "arguments" | "output")
                    && let Some(value) = value.as_str().filter(|value| !value.is_empty())
                {
                    text.push(value.to_owned());
                    continue;
                }
                collect_text(value, text);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_text(value, text);
            }
        }
        serde_json::Value::String(value) if !value.is_empty() => text.push(value.clone()),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use kcastle_agent::{
        EasyInputMessage, FunctionCallOutputItemParam, InputItem, Item, SurfaceOp,
        ToolExecutionOutcome, ToolResultStatus, UserMessageMode,
    };

    use super::*;

    fn event(seq: u64, millis: u64, event: SessionEvent) -> RecordedEvent {
        RecordedEvent {
            seq,
            time: EventTime {
                wall_time_ms: 1_000 + millis as i64,
                clock_id: "projection-test".into(),
                monotonic_ns: millis * 1_000_000,
            },
            source_event_seqs: Vec::new(),
            surface_op: None,
            event,
        }
    }

    fn fixture() -> Vec<RecordedEvent> {
        vec![
            event(0, 0, SessionEvent::TurnStart { turn: 1 }),
            event(1, 0, SessionEvent::StepStart { turn: 1, step: 1 }),
            event(
                2,
                1,
                SessionEvent::UserMessage {
                    turn: 1,
                    step: 1,
                    input_id: None,
                    mode: UserMessageMode::Initial,
                    items: vec![InputItem::from(EasyInputMessage::from("run it"))],
                },
            ),
            event(
                3,
                2,
                SessionEvent::CompactionStart {
                    compaction_id: "compaction-1".into(),
                    tokens_before: 1_000,
                    first_kept_id: 1,
                },
            ),
            RecordedEvent {
                source_event_seqs: vec![3],
                ..event(
                    4,
                    102,
                    SessionEvent::CompactionEnd {
                        compaction_id: "compaction-1".into(),
                        summary: "summary".into(),
                        first_kept_id: 1,
                        tokens_before: 1_000,
                        response: None,
                        outcome: StepOutcome::Completed,
                    },
                )
            },
            event(5, 110, SessionEvent::ModelRequestStart { turn: 1, step: 1 }),
            event(
                6,
                120,
                SessionEvent::AssistantChunk {
                    turn: 1,
                    step: 1,
                    chunk: AssistantChunk::ToolCallArgumentsDelta {
                        call_id: "call-1".into(),
                        name: Some("shell".into()),
                        delta: "{\"command\":\"true\"}".into(),
                    },
                },
            ),
            event(
                7,
                140,
                SessionEvent::AssistantMessage {
                    turn: 1,
                    step: 1,
                    items: vec![InputItem::from(EasyInputMessage::from("calling shell"))],
                    response: kcastle_agent::ResponseMetadata {
                        id: "response-1".into(),
                        model: "test".into(),
                        usage: None,
                    },
                },
            ),
            event(
                8,
                141,
                SessionEvent::ToolCall {
                    turn: 1,
                    step: 1,
                    call_id: "call-1".into(),
                    parent_call_id: None,
                    name: "shell".into(),
                    arguments: "{\"command\":\"true\"}".into(),
                },
            ),
            event(
                9,
                150,
                SessionEvent::ToolExecutionStart {
                    call_id: "call-1".into(),
                },
            ),
            event(
                10,
                190,
                SessionEvent::ToolExecutionFinish {
                    call_id: "call-1".into(),
                    outcome: ToolExecutionOutcome::Success,
                },
            ),
            RecordedEvent {
                source_event_seqs: vec![8, 9, 10],
                surface_op: Some(SurfaceOp::Append),
                ..event(
                    11,
                    200,
                    SessionEvent::ToolResult {
                        turn: 1,
                        step: 1,
                        call_id: "call-1".into(),
                        output: "ok".into(),
                        status: ToolResultStatus::Success,
                        item: InputItem::from(Item::from(FunctionCallOutputItemParam {
                            call_id: "call-1".into(),
                            output: "ok".into(),
                            id: None,
                            status: None,
                        })),
                    },
                )
            },
            event(
                12,
                201,
                SessionEvent::StepEnd {
                    turn: 1,
                    step: 1,
                    outcome: StepOutcome::Completed,
                    error: None,
                },
            ),
        ]
    }

    #[test]
    fn derives_assistant_and_full_tool_timing() {
        let projection = TrajectoryProjection::from_events(&fixture());
        let assistant = projection
            .records
            .iter()
            .find(|record| record.kind == TrajectoryKind::Assistant)
            .unwrap();
        assert_eq!(assistant.timing.duration_ns(), Some(30_000_000));
        assert_eq!(assistant.timing.ttft_ns(), Some(10_000_000));
        assert_eq!(assistant.timing.generation_ns(), Some(20_000_000));

        let compaction = projection
            .records
            .iter()
            .find(|record| record.kind == TrajectoryKind::Compaction)
            .unwrap();
        assert_eq!(compaction.timing.duration_ns(), Some(100_000_000));

        let tool = projection
            .records
            .iter()
            .find(|record| record.kind == TrajectoryKind::Tool)
            .unwrap();
        assert_eq!(tool.timing.duration_ns(), Some(59_000_000));
        assert_eq!(tool.timing.pre_execution_ns(), Some(9_000_000));
        assert_eq!(tool.timing.execution_ns(), Some(40_000_000));
        assert_eq!(tool.timing.post_execution_ns(), Some(10_000_000));
    }

    #[test]
    fn live_projection_equals_replayed_projection() {
        let events = fixture();
        let replayed = TrajectoryProjection::from_events(&events);
        let mut live = TrajectoryProjection::default();
        for event in &events {
            live.apply(event);
        }
        assert_eq!(live, replayed);
    }

    #[test]
    fn incremental_stats_match_a_full_record_scan_after_every_event() {
        let mut projection = TrajectoryProjection::default();
        for event in fixture() {
            projection.apply(&event);
            let mut scanned = TrajectoryStats {
                turns: projection.completed_turns.len(),
                steps: projection.completed_steps.len(),
                ..TrajectoryStats::default()
            };
            for record in &projection.records {
                add_stats(&mut scanned, record_stats(record));
            }
            assert_eq!(projection.stats(), scanned);
        }
    }

    #[test]
    fn session_stats_match_dsh_wall_time_semantics() {
        let stats = TrajectoryProjection::from_events(&fixture()).stats();
        assert_eq!(stats.turns, 1);
        assert_eq!(stats.steps, 1);
        assert_eq!(stats.llm_ns, 30_000_000);
        assert_eq!(stats.tool_ns, 59_000_000);
        assert_eq!(stats.ttft_ns, 10_000_000);
        assert_eq!(stats.ttft_steps, 1);
    }
}
