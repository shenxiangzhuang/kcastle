use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use im::{HashMap, Vector};
use kcastle_agent::{RunId, TokenUsage};

use crate::domain::session_document::{
    DisplayOrdinals, ItemStatus, ModelRequestOptions, ProjectionDelta, PromptChangeKind,
    PromptSnapshot, RequestItemView, SessionDocument, SessionStats, TrajectoryItemDetailsView,
    TrajectoryItemView, TrajectoryRequestKey, TrajectoryRequestPurpose,
};
pub(crate) use crate::domain::session_document::{
    TimingMetrics as RecordTiming, TrajectoryItemId, TrajectoryKind, TrajectoryLane,
};

/// Immutable UI record materialized from one canonical document revision.
///
/// The stable domain identity is deliberately retained. Presentation code must
/// never infer identity from the record's position or from a journal sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrajectoryRecord {
    pub(crate) id: TrajectoryItemId,
    pub(crate) source_seq: u64,
    pub(crate) kind: TrajectoryKind,
    pub(crate) title: String,
    pub(crate) text: String,
    pub(crate) payload: Option<String>,
    /// Small pre-normalized metadata index. Large title/body/payload fields are matched directly
    /// so streaming does not retain another full lowercase copy of the response prefix.
    pub(crate) search_text: String,
    pub(crate) turn: Option<u32>,
    pub(crate) step: Option<u32>,
    pub(crate) status: ItemStatus,
    pub(crate) timing: RecordTiming,
    pub(crate) usage: Option<TokenUsage>,
}

/// Immutable request projection paired with the trajectory at one document
/// revision. Request identity and boundary ownership are explicit; neither is
/// inferred from adjacent presentation rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrajectoryRequest {
    pub(crate) key: TrajectoryRequestKey,
    pub(crate) number: u32,
    pub(crate) purpose: TrajectoryRequestPurpose,
    pub(crate) source_seq: u64,
    pub(crate) turn: Option<u32>,
    pub(crate) step: Option<u32>,
    pub(crate) status: ItemStatus,
    pub(crate) error: Option<Arc<str>>,
    pub(crate) anchor: Option<TrajectoryItemId>,
    pub(crate) result: Option<TrajectoryItemId>,
    pub(crate) options: Option<Arc<ModelRequestOptions>>,
    pub(crate) prompt: Option<Arc<PromptSnapshot>>,
    pub(crate) response_id: Option<Arc<str>>,
    pub(crate) response_model: Option<Arc<str>>,
    pub(crate) timing: RecordTiming,
    pub(crate) usage: Option<TokenUsage>,
    pub(crate) cumulative_usage: Option<TokenUsage>,
    pub(crate) tool_call_count: usize,
    pub(crate) subtool_call_count: usize,
    pub(crate) compaction_run_id: Option<RunId>,
    pub(crate) compaction_tokens_before: Option<usize>,
    pub(crate) compaction_first_kept_id: Option<u64>,
}

/// Canonical zero-record presentation for a request which does not yet have a
/// model/tool boundary. The descriptor is retained for every lifecycle state;
/// presentation code must not silently discard failed or terminal requests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UnanchoredRequestPresentation {
    pub(crate) request_index: usize,
    pub(crate) source_seq: u64,
    pub(crate) status: ItemStatus,
}

/// Large prompt payloads stay shared with the canonical request. Details only
/// retain identities and Arc handles, so opening System Diff or Tool Schema is
/// allocation-free after projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryRecordDetails {
    PromptChange {
        kind: PromptChangeKind,
        current: Arc<PromptSnapshot>,
        previous: Option<Arc<PromptSnapshot>>,
    },
    Tool {
        request_key: TrajectoryRequestKey,
        parent_call_id: Option<kcastle_agent::CallId>,
        prompt: Arc<PromptSnapshot>,
        schema_name: Arc<str>,
    },
}

impl TrajectoryRecordDetails {
    pub(crate) fn tool_schema(&self) -> Option<&str> {
        match self {
            Self::Tool {
                prompt,
                schema_name,
                ..
            } => prompt.tool_schema(schema_name),
            Self::PromptChange { .. } => None,
        }
    }

    fn matches_term(&self, term: &str) -> bool {
        match self {
            Self::PromptChange {
                current, previous, ..
            } => {
                contains_normalized(current.instructions(), term)
                    || contains_normalized(current.tools_json(), term)
                    || previous.as_ref().is_some_and(|previous| {
                        contains_normalized(previous.instructions(), term)
                            || contains_normalized(previous.tools_json(), term)
                    })
            }
            Self::Tool {
                prompt,
                schema_name,
                ..
            } => {
                contains_normalized(schema_name, term)
                    || prompt
                        .tool_schema(schema_name)
                        .is_some_and(|schema| contains_normalized(schema, term))
            }
        }
    }
}

impl TrajectoryRecord {
    #[cfg(test)]
    pub(crate) fn matches(&self, query: &str) -> bool {
        query.split_whitespace().all(|term| self.matches_term(term))
    }

    #[cfg(test)]
    pub(crate) fn matches_terms(&self, terms: &[String]) -> bool {
        terms.iter().all(|term| self.matches_term(term))
    }

    fn matches_term(&self, term: &str) -> bool {
        self.search_text.contains(term)
            || contains_normalized(&self.title, term)
            || contains_normalized(&self.text, term)
            || self
                .payload
                .as_deref()
                .is_some_and(|payload| contains_normalized(payload, term))
    }

    pub(crate) fn lane(&self) -> TrajectoryLane {
        match self.kind {
            TrajectoryKind::System
            | TrajectoryKind::User
            | TrajectoryKind::Steering
            | TrajectoryKind::Context => TrajectoryLane::Input,
            TrajectoryKind::Assistant | TrajectoryKind::Compaction => TrajectoryLane::Model,
            TrajectoryKind::Tool => TrajectoryLane::Tools,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryProjection {
    pub(crate) records: Vector<Arc<TrajectoryRecord>>,
    pub(crate) requests: Vector<Arc<TrajectoryRequest>>,
    /// Requests without a record boundary, in canonical request source order.
    pub(crate) unanchored_requests: Vector<UnanchoredRequestPresentation>,
    index_by_id: HashMap<TrajectoryItemId, usize>,
    request_index_by_key: HashMap<TrajectoryRequestKey, usize>,
    request_key_by_record: HashMap<TrajectoryItemId, TrajectoryRequestKey>,
    request_indices_by_boundary: HashMap<TrajectoryItemId, Vector<usize>>,
    details_by_id: HashMap<TrajectoryItemId, Arc<TrajectoryRecordDetails>>,
    fold_eligibility: TrajectoryFoldEligibility,
    stats: SessionStats,
    trajectory_revision: u64,
    request_revision: u64,
    request_boundary_revision: u64,
    geometry_revision: u64,
    projection_lineage: u64,
    /// One cursor for every incremental trajectory consumer. Search and geometry use field flags
    /// from the same bounded journal instead of maintaining independent revision histories.
    change_revision: u64,
    change_history: Vector<TrajectoryChangeDelta>,
    #[cfg(test)]
    materialized_records: usize,
    #[cfg(test)]
    materialized_record_text_bytes: usize,
    #[cfg(test)]
    materialized_requests: usize,
    #[cfg(test)]
    request_index_work: usize,
}

/// Canonical fold eligibility is derived once per structural projection change. Keeping it beside
/// the immutable records makes toolbar renders O(1); append-only updates touch only the new suffix.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TrajectoryFoldEligibility {
    /// A persistent map makes cloning the prior immutable projection O(1). Updating a suffix then
    /// copies only the HAMT path for the affected turn instead of every historical turn count.
    content_per_turn: HashMap<u32, usize>,
    turns: Arc<HashSet<u32>>,
    assistants: Arc<HashSet<TrajectoryItemId>>,
    revision: u64,
}

impl TrajectoryFoldEligibility {
    fn from_records(records: &Vector<Arc<TrajectoryRecord>>, previous: Option<&Self>) -> Self {
        let mut content_per_turn = HashMap::<u32, usize>::new();
        let mut turns = HashSet::new();
        let mut assistants = HashSet::new();
        for (index, record) in records.iter().enumerate() {
            if record.kind != TrajectoryKind::System
                && let Some(turn) = record.turn
            {
                let count = content_per_turn.entry(turn).or_default();
                *count = count.saturating_add(1);
                if *count == 2 {
                    turns.insert(turn);
                }
            }
            if record.kind == TrajectoryKind::Assistant
                && records
                    .get(index + 1)
                    .is_some_and(|next| next.kind == TrajectoryKind::Tool)
            {
                assistants.insert(record.id.clone());
            }
        }
        let revision = previous.map_or(1, |previous| {
            if previous.turns.as_ref() == &turns && previous.assistants.as_ref() == &assistants {
                previous.revision
            } else {
                previous.revision.saturating_add(1)
            }
        });
        Self {
            content_per_turn,
            turns: Arc::new(turns),
            assistants: Arc::new(assistants),
            revision,
        }
    }

    fn after_appended(
        records: &Vector<Arc<TrajectoryRecord>>,
        suffix_start: usize,
        previous: &Self,
    ) -> Self {
        let mut next = previous.clone();
        let mut changed = false;
        for index in suffix_start..records.len() {
            let record = &records[index];
            if record.kind != TrajectoryKind::System
                && let Some(turn) = record.turn
            {
                let count = next.content_per_turn.entry(turn).or_default();
                *count = count.saturating_add(1);
                if *count == 2 {
                    changed |= Arc::make_mut(&mut next.turns).insert(turn);
                }
            }
            if record.kind == TrajectoryKind::Tool
                && index > 0
                && records[index - 1].kind == TrajectoryKind::Assistant
            {
                changed |=
                    Arc::make_mut(&mut next.assistants).insert(records[index - 1].id.clone());
            }
        }
        if changed {
            next.revision = next.revision.saturating_add(1);
        }
        next
    }
}

const CHANGE_HISTORY_LIMIT: usize = 256;
static NEXT_PROJECTION_LINEAGE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TrajectoryChangeFlags {
    search: bool,
    geometry: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TrajectoryChangeDelta {
    start_revision: u64,
    end_revision: u64,
    changed: HashMap<usize, TrajectoryChangeFlags>,
    search_compatible: bool,
    sequence_compatible: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryChanges {
    pub(crate) revision: u64,
    changed: Vector<(usize, TrajectoryChangeFlags)>,
    search_compatible: bool,
    sequence_compatible: bool,
}

impl TrajectoryChanges {
    pub(crate) fn search_indices(&self) -> Option<impl Iterator<Item = usize> + '_> {
        self.search_compatible.then(|| {
            self.changed
                .iter()
                .filter_map(|(index, flags)| flags.search.then_some(*index))
        })
    }

    pub(crate) fn geometry_indices(&self) -> Option<impl Iterator<Item = usize> + '_> {
        self.sequence_compatible.then(|| {
            self.changed
                .iter()
                .filter_map(|(index, flags)| flags.geometry.then_some(*index))
        })
    }
}

impl TrajectoryProjection {
    pub(crate) fn from_document(document: &SessionDocument) -> Self {
        Self::from_document_reusing(document, None)
    }

    pub(crate) fn from_document_reusing(
        document: &SessionDocument,
        previous: Option<&Self>,
    ) -> Self {
        let revisions = document.revisions();
        #[cfg(test)]
        let mut materialized_record_text_bytes_delta = 0_usize;
        let (records, details_by_id) = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            (previous.records.clone(), previous.details_by_id.clone())
        } else {
            let ordinals = document.display_ordinals();
            let previous_records = previous
                .into_iter()
                .flat_map(|projection| projection.records.iter())
                .map(|record| (record.id.clone(), record))
                .collect::<HashMap<_, _>>();
            let mut records = Vector::new();
            let mut details_by_id = HashMap::new();
            for item in document.trajectory() {
                let details = materialize_record_details(&item).map(Arc::new);
                let record = materialize_record(ordinals, item);
                #[cfg(test)]
                {
                    materialized_record_text_bytes_delta = materialized_record_text_bytes_delta
                        .saturating_add(projected_record_owned_text_bytes(&record));
                }
                let record = Arc::new(record);
                let record = previous_records
                    .get(&record.id)
                    .filter(|previous| previous.as_ref() == record.as_ref())
                    .map_or(record, |previous| Arc::clone(previous));
                if let Some(details) = details {
                    let details = previous
                        .and_then(|projection| projection.details_by_id.get(&record.id))
                        .filter(|previous| previous.as_ref() == details.as_ref())
                        .map_or(details, Arc::clone);
                    details_by_id.insert(record.id.clone(), details);
                }
                records.push_back(record);
            }
            (records, details_by_id)
        };
        let stats = document.stats();

        let index_by_id = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            previous.index_by_id.clone()
        } else {
            records
                .iter()
                .enumerate()
                .map(|(index, record)| (record.id.clone(), index))
                .collect()
        };
        let (
            requests,
            unanchored_requests,
            request_index_by_key,
            request_key_by_record,
            request_indices_by_boundary,
            _materialized_requests_delta,
            _request_index_work_delta,
            request_boundaries_changed,
        ) = if let Some(previous) = previous
            && previous.request_revision == revisions.requests
        {
            (
                previous.requests.clone(),
                previous.unanchored_requests.clone(),
                previous.request_index_by_key.clone(),
                previous.request_key_by_record.clone(),
                previous.request_indices_by_boundary.clone(),
                0,
                0,
                false,
            )
        } else {
            materialize_all_requests(document, previous)
        };
        #[cfg(test)]
        let materialized_records = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            previous.materialized_records
        } else {
            records.len()
        };
        #[cfg(test)]
        let materialized_record_text_bytes =
            previous.map_or(materialized_record_text_bytes_delta, |previous| {
                previous
                    .materialized_record_text_bytes
                    .saturating_add(materialized_record_text_bytes_delta)
            });
        #[cfg(test)]
        let materialized_requests = previous.map_or(_materialized_requests_delta, |previous| {
            previous
                .materialized_requests
                .saturating_add(_materialized_requests_delta)
        });
        #[cfg(test)]
        let request_index_work = previous.map_or(_request_index_work_delta, |previous| {
            previous
                .request_index_work
                .saturating_add(_request_index_work_delta)
        });

        let (change_revision, change_history) =
            change_state_after_full_rebuild(previous, revisions.trajectory);
        let fold_eligibility = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            previous.fold_eligibility.clone()
        } else {
            TrajectoryFoldEligibility::from_records(
                &records,
                previous.map(|projection| &projection.fold_eligibility),
            )
        };
        let request_boundary_revision = previous.map_or_else(
            || u64::from(!requests.is_empty()),
            |previous| {
                previous
                    .request_boundary_revision
                    .saturating_add(u64::from(request_boundaries_changed))
            },
        );
        Self {
            records,
            requests,
            unanchored_requests,
            index_by_id,
            request_index_by_key,
            request_key_by_record,
            request_indices_by_boundary,
            details_by_id,
            fold_eligibility,
            stats,
            trajectory_revision: revisions.trajectory,
            request_revision: revisions.requests,
            request_boundary_revision,
            geometry_revision: revisions.geometry,
            projection_lineage: previous
                .filter(|value| value.projection_lineage != 0)
                .map_or_else(next_projection_lineage, |value| value.projection_lineage),
            change_revision,
            change_history,
            #[cfg(test)]
            materialized_records,
            #[cfg(test)]
            materialized_record_text_bytes,
            #[cfg(test)]
            materialized_requests,
            #[cfg(test)]
            request_index_work,
        }
    }

    pub(crate) fn stats(&self) -> SessionStats {
        self.stats
    }

    /// Geometry caches use only the canonical geometry revision. Text-only
    /// streaming updates therefore cannot invalidate an unchanged timeline.
    pub(crate) fn revision(&self) -> u64 {
        self.geometry_revision
    }

    pub(crate) fn change_revision(&self) -> u64 {
        self.change_revision
    }

    pub(crate) fn projection_lineage(&self) -> u64 {
        self.projection_lineage
    }

    /// Returns every trajectory field change after `revision`. Consumers share this cursor and
    /// select only the fields they need. `None` means the bounded journal cannot prove continuity,
    /// so the caller must rebuild once.
    pub(crate) fn changes_since(&self, revision: u64) -> Option<TrajectoryChanges> {
        if revision > self.change_revision {
            return None;
        }
        if revision == self.change_revision {
            return Some(TrajectoryChanges {
                revision,
                search_compatible: true,
                sequence_compatible: true,
                ..TrajectoryChanges::default()
            });
        }
        let next_revision = revision.saturating_add(1);
        let start = self
            .change_history
            .iter()
            .position(|delta| delta.end_revision >= next_revision)?;
        let first = self.change_history.get(start)?;
        if first.start_revision > next_revision
            || self
                .change_history
                .back()
                .is_none_or(|last| last.end_revision != self.change_revision)
        {
            return None;
        }
        let mut merged = std::collections::HashMap::<usize, TrajectoryChangeFlags>::new();
        let mut search_compatible = true;
        let mut sequence_compatible = true;
        let mut expected_revision = first.end_revision.saturating_add(1);
        for (offset, delta) in self.change_history.iter().skip(start).enumerate() {
            if offset > 0 {
                if delta.start_revision != expected_revision {
                    return None;
                }
                expected_revision = delta.end_revision.saturating_add(1);
            }
            search_compatible &= delta.search_compatible;
            sequence_compatible &= delta.sequence_compatible;
            for (index, flags) in &delta.changed {
                let merged_flags = merged.entry(*index).or_default();
                merged_flags.search |= flags.search;
                merged_flags.geometry |= flags.geometry;
            }
        }
        let mut changed = merged.into_iter().collect::<Vector<_>>();
        changed.sort_by(|(left, _), (right, _)| left.cmp(right));
        Some(TrajectoryChanges {
            revision: self.change_revision,
            changed,
            search_compatible,
            sequence_compatible,
        })
    }

    pub(crate) fn record_by_id(&self, id: &TrajectoryItemId) -> Option<&TrajectoryRecord> {
        self.index_by_id
            .get(id)
            .and_then(|index| self.records.get(*index))
            .map(AsRef::as_ref)
    }

    pub(crate) fn record_index(&self, id: &TrajectoryItemId) -> Option<usize> {
        self.index_by_id.get(id).copied()
    }

    pub(crate) fn record_details(&self, id: &TrajectoryItemId) -> Option<&TrajectoryRecordDetails> {
        self.details_by_id.get(id).map(AsRef::as_ref)
    }

    /// Search is defined over the complete user-visible inspector data, while large prompt and
    /// schema payloads remain shared in `details_by_id` rather than being duplicated into every
    /// record's lowercase index.
    pub(crate) fn record_matches_terms(&self, index: usize, terms: &[String]) -> bool {
        let Some(record) = self.records.get(index) else {
            return false;
        };
        let details = self.record_details(&record.id);
        terms.iter().all(|term| {
            record.matches_term(term) || details.is_some_and(|details| details.matches_term(term))
        })
    }

    pub(crate) fn request_by_key(&self, key: &TrajectoryRequestKey) -> Option<&TrajectoryRequest> {
        self.request_index_by_key
            .get(key)
            .and_then(|index| self.requests.get(*index))
            .map(AsRef::as_ref)
    }

    pub(crate) fn request_index(&self, key: &TrajectoryRequestKey) -> Option<usize> {
        self.request_index_by_key.get(key).copied()
    }

    pub(crate) fn request_key_for_record(
        &self,
        record: &TrajectoryItemId,
    ) -> Option<&TrajectoryRequestKey> {
        self.request_key_by_record.get(record)
    }

    pub(crate) fn request_for_record(
        &self,
        record: &TrajectoryItemId,
    ) -> Option<&TrajectoryRequest> {
        self.request_key_for_record(record)
            .and_then(|key| self.request_by_key(key))
    }

    /// Every request whose marker belongs to `boundary`, in canonical request
    /// source order. Unlike `request_for_record`, this preserves coincident
    /// retry markers instead of selecting only the last relation.
    pub(crate) fn requests_for_boundary<'a>(
        &'a self,
        boundary: &TrajectoryItemId,
    ) -> impl DoubleEndedIterator<Item = &'a TrajectoryRequest> + 'a {
        self.request_indices_by_boundary
            .get(boundary)
            .into_iter()
            .flat_map(|indices| indices.iter())
            .filter_map(|index| self.requests.get(*index).map(AsRef::as_ref))
    }

    /// Changes only when marker membership/order or a marker's record boundary
    /// changes. Status, timing, usage, and text updates deliberately do not
    /// invalidate ledger-decoration geometry.
    pub(crate) fn request_boundary_revision(&self) -> u64 {
        self.request_boundary_revision
    }

    pub(crate) fn collapsible_turns(&self) -> Arc<HashSet<u32>> {
        Arc::clone(&self.fold_eligibility.turns)
    }

    pub(crate) fn collapsible_assistants(&self) -> Arc<HashSet<TrajectoryItemId>> {
        Arc::clone(&self.fold_eligibility.assistants)
    }

    pub(crate) fn fold_eligibility_revision(&self) -> u64 {
        self.fold_eligibility.revision
    }

    pub(crate) fn after_delta(
        document: &SessionDocument,
        delta: &ProjectionDelta,
        previous: &Self,
    ) -> Self {
        if delta.trajectory_order.changed() {
            if delta.trajectory_order.is_append()
                && let Some(projection) = Self::after_appended_order(document, delta, previous)
            {
                return projection;
            }
            let mut rebuilt = Self::from_document_reusing(document, Some(previous));
            // Sequence coordinates are record positions. A non-append order change (for example,
            // inserting the initial system item before an already visible input) invalidates every
            // previously stored range even though this is still the same session document.
            rebuilt.projection_lineage = next_projection_lineage();
            return rebuilt;
        }

        let revisions = document.revisions();
        let ordinals = document.display_ordinals();
        let mut records = previous.records.clone();
        let mut details_by_id = previous.details_by_id.clone();
        let mut materialized = 0_usize;
        #[cfg(test)]
        let mut materialized_record_text_bytes = 0_usize;
        let mut search_indices = Vector::new();
        let mut search_reset = false;
        let mut fold_reset = false;
        let mut geometry_indices = Vector::new();
        for id in &delta.changed_trajectory {
            let Some(index) = previous.index_by_id.get(id).copied() else {
                return Self::from_document_reusing(document, Some(previous));
            };
            let Some(item) = document.trajectory_by_id(id) else {
                return Self::from_document_reusing(document, Some(previous));
            };
            update_record_details(&mut details_by_id, id, materialize_record_details(&item));
            let record = materialize_record(ordinals, item);
            #[cfg(test)]
            {
                materialized_record_text_bytes = materialized_record_text_bytes
                    .saturating_add(projected_record_owned_text_bytes(&record));
            }
            let record = Arc::new(record);
            materialized = materialized.saturating_add(1);
            if records[index].as_ref() != record.as_ref() {
                let previous_record = &records[index];
                // Folded ledger summaries cache only presentation aggregates whose inputs are
                // kind/turn/step/title. If any of those canonical fields changes, consumers must
                // replay their projection once; streaming text/timing updates stay incremental.
                if previous_record.turn != record.turn
                    || previous_record.kind != record.kind
                    || previous_record.step != record.step
                    || previous_record.title != record.title
                {
                    search_reset = true;
                    fold_reset |=
                        previous_record.turn != record.turn || previous_record.kind != record.kind;
                } else if search_content_changed(previous_record, &record) {
                    search_indices.push_back(index);
                }
                if revisions.geometry != previous.geometry_revision
                    && !same_geometry(previous_record, &record)
                {
                    geometry_indices.push_back(index);
                }
                records.set(index, record);
            }
        }
        let stats = if delta.stats_changed {
            document.stats()
        } else {
            previous.stats
        };
        let (change_revision, change_history) = advance_change_state(
            previous,
            search_indices,
            geometry_indices,
            !search_reset,
            true,
        );
        let fold_eligibility = if fold_reset {
            TrajectoryFoldEligibility::from_records(&records, Some(&previous.fold_eligibility))
        } else {
            previous.fold_eligibility.clone()
        };
        let (
            requests,
            unanchored_requests,
            request_index_by_key,
            request_key_by_record,
            request_indices_by_boundary,
            _materialized_requests,
            _request_index_work,
            request_boundaries_changed,
        ) = materialize_requests_after_delta(document, delta, previous)
            .unwrap_or_else(|| materialize_all_requests(document, Some(previous)));
        Self {
            records,
            requests,
            unanchored_requests,
            index_by_id: previous.index_by_id.clone(),
            request_index_by_key,
            request_key_by_record,
            request_indices_by_boundary,
            details_by_id,
            fold_eligibility,
            stats,
            trajectory_revision: revisions.trajectory,
            request_revision: revisions.requests,
            request_boundary_revision: previous
                .request_boundary_revision
                .saturating_add(u64::from(request_boundaries_changed)),
            geometry_revision: revisions.geometry,
            projection_lineage: previous.projection_lineage,
            change_revision,
            change_history,
            #[cfg(test)]
            materialized_records: previous.materialized_records.saturating_add(materialized),
            #[cfg(test)]
            materialized_record_text_bytes: previous
                .materialized_record_text_bytes
                .saturating_add(materialized_record_text_bytes),
            #[cfg(test)]
            materialized_requests: previous
                .materialized_requests
                .saturating_add(_materialized_requests),
            #[cfg(test)]
            request_index_work: previous
                .request_index_work
                .saturating_add(_request_index_work),
        }
    }

    fn after_appended_order(
        document: &SessionDocument,
        delta: &ProjectionDelta,
        previous: &Self,
    ) -> Option<Self> {
        let canonical_ids = document.trajectory_ids();
        let previous_len = previous.records.len();
        let suffix = canonical_ids.get(previous_len..)?;
        if suffix.is_empty()
            || suffix
                .iter()
                .any(|id| previous.index_by_id.contains_key(id))
        {
            return None;
        }

        let mut records = previous.records.clone();
        let mut index_by_id = previous.index_by_id.clone();
        let mut details_by_id = previous.details_by_id.clone();
        let ordinals = document.display_ordinals();
        let mut materialized = 0_usize;
        #[cfg(test)]
        let mut materialized_record_text_bytes = 0_usize;
        let mut search_indices = Vector::new();
        let mut search_reset = false;
        let mut fold_reset = false;
        let mut geometry_indices = Vector::new();
        for id in suffix {
            let item = document.trajectory_by_id(id)?;
            update_record_details(&mut details_by_id, id, materialize_record_details(&item));
            let record = materialize_record(ordinals, item);
            #[cfg(test)]
            {
                materialized_record_text_bytes = materialized_record_text_bytes
                    .saturating_add(projected_record_owned_text_bytes(&record));
            }
            let record = Arc::new(record);
            index_by_id.insert(id.clone(), records.len());
            search_indices.push_back(records.len());
            geometry_indices.push_back(records.len());
            records.push_back(record);
            materialized = materialized.saturating_add(1);
        }
        // A transaction may both append and update an existing row (for
        // example, revealing an assistant and requesting its first tool).
        for id in &delta.changed_trajectory {
            let index = *index_by_id.get(id)?;
            if index >= previous_len {
                continue;
            }
            let item = document.trajectory_by_id(id)?;
            update_record_details(&mut details_by_id, id, materialize_record_details(&item));
            let record = materialize_record(ordinals, item);
            #[cfg(test)]
            {
                materialized_record_text_bytes = materialized_record_text_bytes
                    .saturating_add(projected_record_owned_text_bytes(&record));
            }
            let record = Arc::new(record);
            materialized = materialized.saturating_add(1);
            if records[index].as_ref() != record.as_ref() {
                let previous_record = &records[index];
                if previous_record.turn != record.turn
                    || previous_record.kind != record.kind
                    || previous_record.step != record.step
                    || previous_record.title != record.title
                {
                    search_reset = true;
                    fold_reset |=
                        previous_record.turn != record.turn || previous_record.kind != record.kind;
                } else if search_content_changed(previous_record, &record) {
                    search_indices.push_back(index);
                }
                if delta.geometry_changed && !same_geometry(previous_record, &record) {
                    geometry_indices.push_back(index);
                }
                records.set(index, record);
            }
        }
        let revisions = document.revisions();
        let (change_revision, change_history) = advance_change_state(
            previous,
            search_indices,
            geometry_indices,
            !search_reset,
            delta.geometry_changed,
        );
        let fold_eligibility = if fold_reset {
            TrajectoryFoldEligibility::from_records(&records, Some(&previous.fold_eligibility))
        } else {
            TrajectoryFoldEligibility::after_appended(
                &records,
                previous_len,
                &previous.fold_eligibility,
            )
        };
        let (
            requests,
            unanchored_requests,
            request_index_by_key,
            request_key_by_record,
            request_indices_by_boundary,
            _materialized_requests,
            _request_index_work,
            request_boundaries_changed,
        ) = materialize_requests_after_delta(document, delta, previous)
            .unwrap_or_else(|| materialize_all_requests(document, Some(previous)));
        Some(Self {
            records,
            requests,
            unanchored_requests,
            index_by_id,
            request_index_by_key,
            request_key_by_record,
            request_indices_by_boundary,
            details_by_id,
            fold_eligibility,
            stats: if delta.stats_changed {
                document.stats()
            } else {
                previous.stats
            },
            trajectory_revision: revisions.trajectory,
            request_revision: revisions.requests,
            request_boundary_revision: previous
                .request_boundary_revision
                .saturating_add(u64::from(request_boundaries_changed)),
            geometry_revision: revisions.geometry,
            projection_lineage: previous.projection_lineage,
            change_revision,
            change_history,
            #[cfg(test)]
            materialized_records: previous.materialized_records.saturating_add(materialized),
            #[cfg(test)]
            materialized_record_text_bytes: previous
                .materialized_record_text_bytes
                .saturating_add(materialized_record_text_bytes),
            #[cfg(test)]
            materialized_requests: previous
                .materialized_requests
                .saturating_add(_materialized_requests),
            #[cfg(test)]
            request_index_work: previous
                .request_index_work
                .saturating_add(_request_index_work),
        })
    }

    #[cfg(test)]
    pub(crate) fn materialized_records(&self) -> usize {
        self.materialized_records
    }

    #[cfg(test)]
    pub(crate) fn materialized_record_text_bytes(&self) -> usize {
        self.materialized_record_text_bytes
    }

    #[cfg(test)]
    pub(crate) fn materialized_requests(&self) -> usize {
        self.materialized_requests
    }

    #[cfg(test)]
    pub(crate) fn request_index_work(&self) -> usize {
        self.request_index_work
    }
}

type RequestProjectionParts = (
    Vector<Arc<TrajectoryRequest>>,
    Vector<UnanchoredRequestPresentation>,
    HashMap<TrajectoryRequestKey, usize>,
    HashMap<TrajectoryItemId, TrajectoryRequestKey>,
    HashMap<TrajectoryItemId, Vector<usize>>,
    usize,
    usize,
    bool,
);

fn materialize_all_requests(
    document: &SessionDocument,
    previous: Option<&TrajectoryProjection>,
) -> RequestProjectionParts {
    let previous_by_key = previous
        .into_iter()
        .flat_map(|projection| projection.requests.iter())
        .map(|request| (request.key.clone(), request))
        .collect::<HashMap<_, _>>();
    let mut requests = Vector::new();
    let mut cumulative = None;
    let ordinals = document.display_ordinals();
    let mut materialized = 0_usize;
    for (index, key) in document.request_ids().iter().enumerate() {
        let Some(item) = document.request_by_key(key) else {
            continue;
        };
        cumulative = add_request_usage(cumulative, item.usage);
        let request = Arc::new(materialize_request(ordinals, item, index, cumulative));
        materialized = materialized.saturating_add(1);
        requests.push_back(
            previous_by_key
                .get(key)
                .filter(|previous| previous.as_ref() == request.as_ref())
                .map_or(request, |previous| Arc::clone(previous)),
        );
    }
    let RequestProjectionIndexes {
        index_by_key,
        key_by_record,
        indices_by_boundary,
        unanchored_requests,
    } = request_projection_indexes(&requests);
    let request_boundaries_changed = previous.is_none_or(|previous| {
        previous.request_indices_by_boundary != indices_by_boundary
            || !same_unanchored_request_order(&previous.unanchored_requests, &unanchored_requests)
    });
    let index_work = requests.len();
    (
        requests,
        unanchored_requests,
        index_by_key,
        key_by_record,
        indices_by_boundary,
        materialized,
        index_work,
        request_boundaries_changed,
    )
}

fn materialize_requests_after_delta(
    document: &SessionDocument,
    delta: &ProjectionDelta,
    previous: &TrajectoryProjection,
) -> Option<RequestProjectionParts> {
    if delta.request_order.changed() && !delta.request_order.is_append() {
        return None;
    }
    let canonical = document.request_ids();
    if canonical.len() < previous.requests.len()
        || (!delta.request_order.changed() && canonical.len() != previous.requests.len())
    {
        return None;
    }
    let mut first_changed = if canonical.len() > previous.requests.len() {
        previous.requests.len()
    } else {
        canonical.len()
    };
    let mut index_work = 0_usize;
    for key in &delta.changed_requests {
        index_work = index_work.saturating_add(1);
        let index = if let Some(index) = previous.request_index_by_key.get(key).copied() {
            index
        } else {
            let suffix = canonical.get(previous.requests.len()..)?;
            let offset = suffix.iter().position(|candidate| candidate == key);
            index_work = index_work
                .saturating_add(offset.map_or(suffix.len(), |offset| offset.saturating_add(1)));
            previous.requests.len().saturating_add(offset?)
        };
        first_changed = first_changed.min(index);
    }
    if first_changed == canonical.len() {
        return Some((
            previous.requests.clone(),
            previous.unanchored_requests.clone(),
            previous.request_index_by_key.clone(),
            previous.request_key_by_record.clone(),
            previous.request_indices_by_boundary.clone(),
            0,
            index_work,
            false,
        ));
    }

    let mut requests = previous.requests.clone();
    while requests.len() > canonical.len() {
        requests.pop_back();
    }
    let mut cumulative = first_changed
        .checked_sub(1)
        .and_then(|index| requests.get(index))
        .and_then(|request| request.cumulative_usage);
    let ordinals = document.display_ordinals();
    let mut materialized = 0_usize;
    for (index, key) in canonical.iter().enumerate().skip(first_changed) {
        let item = document.request_by_key(key)?;
        cumulative = add_request_usage(cumulative, item.usage);
        let request = Arc::new(materialize_request(ordinals, item, index, cumulative));
        materialized = materialized.saturating_add(1);
        if let Some(previous_request) = requests.get(index)
            && previous_request.as_ref() == request.as_ref()
        {
            continue;
        }
        if index < requests.len() {
            requests.set(index, request);
        } else {
            requests.push_back(request);
        }
    }
    let request_boundaries_changed = canonical.len() != previous.requests.len()
        || requests
            .iter()
            .enumerate()
            .skip(first_changed)
            .any(|(index, request)| {
                previous.requests.get(index).is_none_or(|previous| {
                    previous.anchor != request.anchor || previous.source_seq != request.source_seq
                })
            });
    let mut index_by_key = previous.request_index_by_key.clone();
    let mut key_by_record = previous.request_key_by_record.clone();
    let mut indices_by_boundary = previous.request_indices_by_boundary.clone();
    let mut unanchored_requests = previous.unanchored_requests.clone();
    while unanchored_requests
        .back()
        .is_some_and(|request| request.request_index >= first_changed)
    {
        unanchored_requests.pop_back();
    }
    let affected_boundaries = previous
        .requests
        .iter()
        .skip(first_changed)
        .filter_map(|request| request.anchor.clone())
        .collect::<HashSet<_>>();
    for boundary in affected_boundaries {
        let Some(mut indices) = indices_by_boundary.get(&boundary).cloned() else {
            continue;
        };
        while indices.back().is_some_and(|index| *index >= first_changed) {
            indices.pop_back();
        }
        if indices.is_empty() {
            indices_by_boundary.remove(&boundary);
        } else {
            indices_by_boundary.insert(boundary, indices);
        }
    }
    for request in previous.requests.iter().skip(first_changed) {
        index_work = index_work.saturating_add(1);
        index_by_key.remove(&request.key);
        if let Some(record) = request.anchor.as_ref()
            && key_by_record.get(record) == Some(&request.key)
        {
            key_by_record.remove(record);
        }
        if request.result.as_ref() != request.anchor.as_ref()
            && let Some(result) = request.result.as_ref()
            && key_by_record.get(result) == Some(&request.key)
        {
            key_by_record.remove(result);
        }
    }
    for (index, request) in requests.iter().enumerate().skip(first_changed) {
        index_work = index_work.saturating_add(1);
        index_by_key.insert(request.key.clone(), index);
        if request.anchor.is_none() {
            unanchored_requests.push_back(UnanchoredRequestPresentation {
                request_index: index,
                source_seq: request.source_seq,
                status: request.status,
            });
        }
        if let Some(anchor) = request.anchor.as_ref() {
            key_by_record.insert(anchor.clone(), request.key.clone());
            indices_by_boundary
                .entry(anchor.clone())
                .or_default()
                .push_back(index);
        }
        if request.result.as_ref() != request.anchor.as_ref()
            && let Some(result) = request.result.as_ref()
        {
            key_by_record.insert(result.clone(), request.key.clone());
        }
    }
    Some((
        requests,
        unanchored_requests,
        index_by_key,
        key_by_record,
        indices_by_boundary,
        materialized,
        index_work,
        request_boundaries_changed,
    ))
}

struct RequestProjectionIndexes {
    index_by_key: HashMap<TrajectoryRequestKey, usize>,
    key_by_record: HashMap<TrajectoryItemId, TrajectoryRequestKey>,
    indices_by_boundary: HashMap<TrajectoryItemId, Vector<usize>>,
    unanchored_requests: Vector<UnanchoredRequestPresentation>,
}

fn request_projection_indexes(
    requests: &Vector<Arc<TrajectoryRequest>>,
) -> RequestProjectionIndexes {
    let mut index_by_key = HashMap::new();
    let mut key_by_record = HashMap::new();
    let mut indices_by_boundary = HashMap::<TrajectoryItemId, Vector<usize>>::new();
    let mut unanchored_requests = Vector::new();
    for (index, request) in requests.iter().enumerate() {
        index_by_key.insert(request.key.clone(), index);
        if request.anchor.is_none() {
            unanchored_requests.push_back(UnanchoredRequestPresentation {
                request_index: index,
                source_seq: request.source_seq,
                status: request.status,
            });
        }
        if let Some(anchor) = request.anchor.as_ref() {
            key_by_record.insert(anchor.clone(), request.key.clone());
            indices_by_boundary
                .entry(anchor.clone())
                .or_default()
                .push_back(index);
        }
        if request.result.as_ref() != request.anchor.as_ref()
            && let Some(result) = request.result.as_ref()
        {
            key_by_record.insert(result.clone(), request.key.clone());
        }
    }
    RequestProjectionIndexes {
        index_by_key,
        key_by_record,
        indices_by_boundary,
        unanchored_requests,
    }
}

fn same_unanchored_request_order(
    left: &Vector<UnanchoredRequestPresentation>,
    right: &Vector<UnanchoredRequestPresentation>,
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.request_index == right.request_index && left.source_seq == right.source_seq
        })
}

fn materialize_request(
    ordinals: &DisplayOrdinals,
    item: RequestItemView<'_>,
    index: usize,
    cumulative_usage: Option<TokenUsage>,
) -> TrajectoryRequest {
    let turn = item.turn_id.and_then(|turn_id| ordinals.turn(turn_id));
    let step = item
        .turn_id
        .zip(item.step_id)
        .and_then(|(turn_id, step_id)| ordinals.step(turn_id, step_id));
    TrajectoryRequest {
        key: item.key.clone(),
        number: u32::try_from(index.saturating_add(1)).unwrap_or(u32::MAX),
        purpose: item.purpose,
        source_seq: item.source_seq,
        turn,
        step,
        status: item.status,
        error: item.error.map(Arc::from),
        anchor: item.anchor,
        result: item.result,
        options: item.options.map(Arc::clone),
        prompt: item.prompt.map(Arc::clone),
        response_id: item.response_id.map(Arc::from),
        response_model: item.response_model.map(Arc::from),
        timing: item.timing.clone(),
        usage: item.usage,
        cumulative_usage,
        tool_call_count: item.tool_call_count,
        subtool_call_count: item.subtool_call_count,
        compaction_run_id: item.compaction_run_id.cloned(),
        compaction_tokens_before: item.compaction_tokens_before,
        compaction_first_kept_id: item.compaction_first_kept_id,
    }
}

fn add_request_usage(total: Option<TokenUsage>, usage: Option<TokenUsage>) -> Option<TokenUsage> {
    let Some(usage) = usage else {
        return total;
    };
    let total = total.unwrap_or_default();
    Some(TokenUsage {
        uncached_input_tokens: total
            .uncached_input_tokens
            .saturating_add(usage.uncached_input_tokens),
        cache_read_input_tokens: total
            .cache_read_input_tokens
            .saturating_add(usage.cache_read_input_tokens),
        cache_write_input_tokens: total
            .cache_write_input_tokens
            .saturating_add(usage.cache_write_input_tokens),
        output_tokens: total.output_tokens.saturating_add(usage.output_tokens),
        reasoning_output_tokens: total
            .reasoning_output_tokens
            .saturating_add(usage.reasoning_output_tokens),
    })
}

fn change_state_after_full_rebuild(
    previous: Option<&TrajectoryProjection>,
    trajectory_revision: u64,
) -> (u64, Vector<TrajectoryChangeDelta>) {
    let Some(previous) = previous else {
        return (1, Vector::new());
    };
    if previous.trajectory_revision == trajectory_revision {
        return (previous.change_revision, previous.change_history.clone());
    }
    advance_change_state(previous, Vector::new(), Vector::new(), false, false)
}

fn advance_change_state(
    previous: &TrajectoryProjection,
    search_indices: Vector<usize>,
    geometry_indices: Vector<usize>,
    search_compatible: bool,
    sequence_compatible: bool,
) -> (u64, Vector<TrajectoryChangeDelta>) {
    if search_indices.is_empty()
        && geometry_indices.is_empty()
        && search_compatible
        && sequence_compatible
    {
        return (previous.change_revision, previous.change_history.clone());
    }
    let revision = previous.change_revision.saturating_add(1);
    let mut changed = HashMap::<usize, TrajectoryChangeFlags>::new();
    for index in search_indices {
        changed.entry(index).or_default().search = true;
    }
    for index in geometry_indices {
        changed.entry(index).or_default().geometry = true;
    }
    let mut history = previous.change_history.clone();
    let delta = TrajectoryChangeDelta {
        start_revision: revision,
        end_revision: revision,
        changed,
        search_compatible,
        sequence_compatible,
    };
    if is_search_only_change(&delta) && history.back().is_some_and(is_search_only_change) {
        let mut coalesced = history
            .pop_back()
            .expect("the previous search-only change was just observed");
        coalesced.end_revision = revision;
        for (index, flags) in delta.changed {
            let mut merged = coalesced.changed.get(&index).copied().unwrap_or_default();
            merged.search |= flags.search;
            merged.geometry |= flags.geometry;
            coalesced.changed.insert(index, merged);
        }
        history.push_back(coalesced);
    } else {
        history.push_back(delta);
    }
    while history.len() > CHANGE_HISTORY_LIMIT {
        history.pop_front();
    }
    (revision, history)
}

fn is_search_only_change(delta: &TrajectoryChangeDelta) -> bool {
    delta.search_compatible
        && delta.sequence_compatible
        && delta.changed.values().all(|flags| !flags.geometry)
}

fn same_geometry(previous: &TrajectoryRecord, current: &TrajectoryRecord) -> bool {
    previous.kind == current.kind && previous.timing == current.timing
}

fn search_content_changed(previous: &TrajectoryRecord, current: &TrajectoryRecord) -> bool {
    previous.text != current.text
        || previous.payload != current.payload
        || previous.search_text != current.search_text
}

fn next_projection_lineage() -> u64 {
    NEXT_PROJECTION_LINEAGE.fetch_add(1, Ordering::Relaxed)
}

fn materialize_record_details(item: &TrajectoryItemView<'_>) -> Option<TrajectoryRecordDetails> {
    match &item.details {
        TrajectoryItemDetailsView::None => None,
        TrajectoryItemDetailsView::PromptChange {
            kind,
            current,
            previous,
        } => Some(TrajectoryRecordDetails::PromptChange {
            kind: *kind,
            current: Arc::clone(current),
            previous: previous.map(Arc::clone),
        }),
        TrajectoryItemDetailsView::Tool {
            request_id,
            parent_call_id,
            prompt,
            schema_name,
        } => Some(TrajectoryRecordDetails::Tool {
            request_key: TrajectoryRequestKey::Model((*request_id).clone()),
            parent_call_id: parent_call_id.map(|call_id| (*call_id).clone()),
            prompt: Arc::clone(prompt),
            schema_name: Arc::from(*schema_name),
        }),
    }
}

fn update_record_details(
    details_by_id: &mut HashMap<TrajectoryItemId, Arc<TrajectoryRecordDetails>>,
    id: &TrajectoryItemId,
    details: Option<TrajectoryRecordDetails>,
) {
    let Some(details) = details else {
        details_by_id.remove(id);
        return;
    };
    if details_by_id
        .get(id)
        .is_some_and(|previous| previous.as_ref() == &details)
    {
        return;
    }
    details_by_id.insert(id.clone(), Arc::new(details));
}

fn materialize_record(
    ordinals: &DisplayOrdinals,
    item: TrajectoryItemView<'_>,
) -> TrajectoryRecord {
    let turn = item.turn_id.and_then(|turn_id| ordinals.turn(turn_id));
    let step = item
        .turn_id
        .zip(item.step_id)
        .and_then(|(turn_id, step_id)| ordinals.step(turn_id, step_id));
    let title = item.title.to_owned();
    let text = item.text.to_owned();
    let payload = item.payload.map(ToOwned::to_owned);
    let mut search_text = String::new();
    push_normalized_search_field(&mut search_text, searchable_kind(item.kind));
    if let Some(turn) = turn {
        push_normalized_search_field(&mut search_text, &format!("turn {turn} #{turn}"));
    } else {
        push_normalized_search_field(&mut search_text, "between turns");
    }
    if let Some(step) = step {
        push_normalized_search_field(&mut search_text, &format!("step {step}"));
    }
    if let TrajectoryItemId::Tool(call_id) = item.id {
        push_normalized_search_field(&mut search_text, call_id.as_str());
    }
    TrajectoryRecord {
        id: item.id.clone(),
        source_seq: item.source_seqs.first().copied().unwrap_or_default(),
        kind: item.kind,
        title,
        text,
        payload,
        search_text,
        turn,
        step,
        status: item.status,
        timing: item.timing.clone(),
        usage: item.usage,
    }
}

#[cfg(test)]
fn projected_record_owned_text_bytes(record: &TrajectoryRecord) -> usize {
    record
        .title
        .len()
        .saturating_add(record.text.len())
        .saturating_add(record.payload.as_deref().map_or(0, str::len))
        .saturating_add(record.search_text.len())
}

fn contains_normalized(haystack: &str, normalized_needle: &str) -> bool {
    if normalized_needle.is_empty() {
        return true;
    }
    if haystack.is_ascii() && normalized_needle.is_ascii() {
        return haystack
            .as_bytes()
            .windows(normalized_needle.len())
            .any(|candidate| candidate.eq_ignore_ascii_case(normalized_needle.as_bytes()));
    }
    // The UI normalizes query terms once. Unicode lowercasing can grow a string, so defer this
    // exceptional allocation to active search rather than every streamed projection update.
    haystack.to_lowercase().contains(normalized_needle)
}

fn push_normalized_search_field(search_text: &mut String, value: &str) {
    search_text.push('\n');
    if value.is_ascii() {
        search_text.extend(value.bytes().map(|byte| byte.to_ascii_lowercase() as char));
    } else {
        search_text.push_str(&value.to_lowercase());
    }
}

fn searchable_kind(kind: TrajectoryKind) -> &'static str {
    match kind {
        TrajectoryKind::System => "system input prompt",
        TrajectoryKind::User => "user input message",
        TrajectoryKind::Steering => "steering user input message",
        TrajectoryKind::Context => "context input message",
        TrajectoryKind::Assistant => "message assistant model response",
        TrajectoryKind::Tool => "tool tools call",
        TrajectoryKind::Compaction => "compacted compaction model",
    }
}

// Preserve the current UI-facing timing vocabulary while retaining the exact
// canonical timing object instead of copying individual timestamps.
impl RecordTiming {
    pub(crate) fn generation_ns(&self) -> Option<u64> {
        self.decode_ns()
    }
}

#[cfg(test)]
mod tests {
    use kcastle_agent::SessionEvent;

    use super::*;

    fn materialized_turnless_record(
        id: TrajectoryItemId,
        kind: TrajectoryKind,
        title: &'static str,
    ) -> TrajectoryRecord {
        let timing = RecordTiming::default();
        let source_seqs = [1];
        materialize_record(
            &DisplayOrdinals::default(),
            TrajectoryItemView {
                id: &id,
                kind,
                lane: match kind {
                    TrajectoryKind::System
                    | TrajectoryKind::User
                    | TrajectoryKind::Steering
                    | TrajectoryKind::Context => TrajectoryLane::Input,
                    TrajectoryKind::Assistant | TrajectoryKind::Compaction => TrajectoryLane::Model,
                    TrajectoryKind::Tool => TrajectoryLane::Tools,
                },
                title,
                text: "",
                payload: None,
                status: ItemStatus::Completed,
                timing: &timing,
                usage: None,
                turn_id: None,
                step_id: None,
                source_seqs: &source_seqs,
                details: TrajectoryItemDetailsView::None,
            },
        )
    }

    #[test]
    fn canonical_dsh_kind_tokens_are_searchable() {
        let assistant = materialized_turnless_record(
            TrajectoryItemId::Assistant(kcastle_agent::RequestId::from("request-1")),
            TrajectoryKind::Assistant,
            "Assistant",
        );
        let compaction = materialized_turnless_record(
            TrajectoryItemId::Compaction(kcastle_agent::CompactionId::from("compaction-1")),
            TrajectoryKind::Compaction,
            "Compaction",
        );

        assert!(assistant.matches("message"));
        assert!(compaction.matches("compacted"));
    }

    #[test]
    fn turnless_records_are_searchable_as_between_turns() {
        let record = materialized_turnless_record(
            TrajectoryItemId::Compaction(kcastle_agent::CompactionId::from("compaction-1")),
            TrajectoryKind::Compaction,
            "Compaction",
        );

        assert!(record.matches("between turns"));
    }

    #[test]
    fn fold_eligibility_promotes_only_the_appended_suffix() {
        let mut assistant = materialized_turnless_record(
            TrajectoryItemId::Assistant(kcastle_agent::RequestId::from("request-1")),
            TrajectoryKind::Assistant,
            "Assistant",
        );
        assistant.turn = Some(1);
        let mut records = [Arc::new(assistant)].into_iter().collect::<Vector<_>>();
        let initial = TrajectoryFoldEligibility::from_records(&records, None);
        assert!(initial.turns.is_empty());
        assert!(initial.assistants.is_empty());

        let mut tool = materialized_turnless_record(
            TrajectoryItemId::Tool(kcastle_agent::CallId::from_raw("call-1")),
            TrajectoryKind::Tool,
            "bash",
        );
        tool.turn = Some(1);
        records.push_back(Arc::new(tool));
        let appended = TrajectoryFoldEligibility::after_appended(&records, 1, &initial);

        assert_eq!(appended.turns.as_ref(), &HashSet::from([1]));
        assert_eq!(
            appended.assistants.as_ref(),
            &HashSet::from([records[0].id.clone()])
        );
        assert_eq!(appended.revision, initial.revision + 1);
    }

    #[test]
    fn ten_thousand_fold_appends_keep_persistent_turn_counts() {
        fn require_persistent_map(_: &im::HashMap<u32, usize>) {}

        const RECORDS: usize = 10_000;
        let mut records = Vector::new();
        let mut eligibility = TrajectoryFoldEligibility::from_records(&records, None);
        for index in 0..RECORDS {
            let mut record = materialized_turnless_record(
                TrajectoryItemId::Input(kcastle_agent::InputId::from_raw(format!("input-{index}"))),
                TrajectoryKind::User,
                "User",
            );
            record.turn = Some(u32::try_from(index + 1).unwrap());
            records.push_back(Arc::new(record));
            eligibility = TrajectoryFoldEligibility::after_appended(
                &records,
                records.len() - 1,
                &eligibility,
            );
        }

        // This type assertion is the complexity gate: cloning a persistent map shares its HAMT
        // root, whereas replacing it with std::HashMap would restore O(history) work per append.
        require_persistent_map(&eligibility.content_per_turn);
        assert_eq!(eligibility.content_per_turn.len(), RECORDS);
        assert!(eligibility.turns.is_empty());
        assert!(eligibility.assistants.is_empty());
    }

    #[test]
    fn search_terms_match_independently_in_the_precomputed_record_text() {
        let record = TrajectoryRecord {
            id: TrajectoryItemId::Assistant(kcastle_agent::RequestId::from("request-1")),
            source_seq: 1,
            kind: TrajectoryKind::Assistant,
            title: "Assistant".into(),
            text: "Found the relevant Agent discussion".into(),
            payload: Some("ÄPFEL payload".into()),
            search_text: "message assistant model response".into(),
            turn: Some(1),
            step: Some(1),
            status: ItemStatus::Completed,
            timing: RecordTiming::default(),
            usage: None,
        };

        assert!(record.matches("assistant agent"));
        assert!(record.matches("agent discussion"));
        assert!(record.matches("äpfel payload"));
        assert!(!record.matches("assistant missing"));
    }

    #[test]
    fn projection_lineage_is_unique_per_fresh_document_projection() {
        let document = SessionDocument::default();
        let first = TrajectoryProjection::from_document(&document);
        let same_projection = TrajectoryProjection::from_document_reusing(&document, Some(&first));
        let reloaded = TrajectoryProjection::from_document(&document);

        assert_ne!(first.projection_lineage(), 0);
        assert_eq!(
            same_projection.projection_lineage(),
            first.projection_lineage()
        );
        assert_ne!(reloaded.projection_lineage(), first.projection_lineage());
    }

    #[test]
    fn non_append_order_change_starts_a_new_coordinate_generation() {
        let mut events = crate::domain::session_document::tests::fixture();
        let request = events.remove(5);
        events.truncate(5);
        let mut document = SessionDocument::from_events(events).unwrap();
        let previous = TrajectoryProjection::from_document(&document);

        let delta = document.apply_batch(vec![request]).unwrap();
        assert!(delta.trajectory_order.changed());
        assert!(!delta.trajectory_order.is_append());
        let current = TrajectoryProjection::after_delta(&document, &delta, &previous);

        assert_ne!(current.records[0].id, previous.records[0].id);
        assert_ne!(current.projection_lineage(), previous.projection_lineage());
    }

    #[test]
    fn one_change_history_serves_search_and_geometry_and_rejects_resets() {
        let previous = TrajectoryProjection {
            change_revision: 10,
            projection_lineage: next_projection_lineage(),
            ..TrajectoryProjection::default()
        };
        let (revision, history) = advance_change_state(
            &previous,
            [77_777].into_iter().collect(),
            [88_888].into_iter().collect(),
            true,
            true,
        );
        let current = TrajectoryProjection {
            change_revision: revision,
            projection_lineage: previous.projection_lineage,
            change_history: history,
            ..TrajectoryProjection::default()
        };
        let changes = current.changes_since(10).unwrap();
        assert_eq!(
            changes.search_indices().unwrap().collect::<Vec<_>>(),
            [77_777]
        );
        assert_eq!(
            changes.geometry_indices().unwrap().collect::<Vec<_>>(),
            [88_888]
        );

        let (revision, reset_history) =
            advance_change_state(&current, Vector::new(), Vector::new(), false, false);
        let reset = TrajectoryProjection {
            change_revision: revision,
            projection_lineage: current.projection_lineage,
            change_history: reset_history,
            ..TrajectoryProjection::default()
        };
        let changes = reset.changes_since(10).unwrap();
        assert!(changes.search_indices().is_none());
        assert!(changes.geometry_indices().is_none());
    }

    #[test]
    fn dsh_golden_requests_preserve_global_order_options_usage_and_relationships() {
        let document =
            SessionDocument::from_events(crate::domain::session_document::tests::fixture())
                .unwrap();
        let projection = TrajectoryProjection::from_document(&document);

        assert_eq!(projection.requests.len(), 3);
        assert_eq!(
            projection
                .requests
                .iter()
                .map(|request| (&request.key, request.number, request.purpose))
                .collect::<Vec<_>>(),
            vec![
                (
                    &TrajectoryRequestKey::Model(kcastle_agent::RequestId::from("request-1")),
                    1,
                    TrajectoryRequestPurpose::Assistant,
                ),
                (
                    &TrajectoryRequestKey::Compaction(kcastle_agent::CompactionId::from(
                        "compaction-1",
                    )),
                    2,
                    TrajectoryRequestPurpose::Compaction,
                ),
                (
                    &TrajectoryRequestKey::Model(kcastle_agent::RequestId::from("request-2")),
                    3,
                    TrajectoryRequestPurpose::Assistant,
                ),
            ]
        );

        let first_key = TrajectoryRequestKey::Model(kcastle_agent::RequestId::from("request-1"));
        let first = projection.request_by_key(&first_key).unwrap();
        assert_eq!(projection.request_index(&first_key), Some(0));
        assert_eq!(first.turn, Some(1));
        assert_eq!(first.step, Some(1));
        assert_eq!(first.status, ItemStatus::Completed);
        assert_eq!(first.tool_call_count, 1);
        assert_eq!(first.response_id.as_deref(), Some("response-1"));
        assert_eq!(first.response_model.as_deref(), Some("deepseek-v4"));
        let options = first.options.as_deref().unwrap();
        assert_eq!(options.reason, kcastle_agent::RequestHeaderReason::Initial);
        assert_eq!(options.model.as_ref(), "deepseek-v4");
        assert_eq!(options.reasoning_effort, None);
        assert_eq!(options.max_output_tokens, Some(4_096));
        assert_eq!(
            options.session_config,
            kcastle_agent::SessionConfig::default()
        );
        assert_eq!(first.usage, Some(request_usage_fixture()));
        assert_eq!(first.cumulative_usage, first.usage);
        let first_anchor = TrajectoryItemId::Assistant(kcastle_agent::RequestId::from("request-1"));
        assert_eq!(first.anchor.as_ref(), Some(&first_anchor));
        assert_eq!(first.result.as_ref(), Some(&first_anchor));
        assert_eq!(
            projection.request_key_for_record(&first_anchor),
            Some(&first_key)
        );
        assert_eq!(
            projection
                .request_for_record(&first_anchor)
                .map(|request| &request.key),
            Some(&first_key)
        );

        let compaction = &projection.requests[1];
        assert_eq!(compaction.turn, None);
        assert_eq!(compaction.step, None);
        assert_eq!(compaction.options, None);
        assert_eq!(
            compaction.response_id.as_deref(),
            Some("compaction-response")
        );
        assert_eq!(compaction.response_model.as_deref(), Some("deepseek-v4"));
        assert_eq!(compaction.compaction_tokens_before, Some(123_456));
        assert_eq!(compaction.compaction_first_kept_id, Some(1));
        assert_eq!(
            compaction
                .compaction_run_id
                .as_ref()
                .map(|run| run.as_str()),
            Some("run-1")
        );

        let expected_cumulative = TokenUsage {
            uncached_input_tokens: 1_000_019,
            cache_read_input_tokens: 1_000_079,
            cache_write_input_tokens: 1_000_002,
            output_tokens: 1_000_014,
            reasoning_output_tokens: 1_000_004,
        };
        assert_eq!(compaction.cumulative_usage, Some(expected_cumulative));

        let failed = &projection.requests[2];
        assert_eq!(failed.status, ItemStatus::Failed);
        assert_eq!(failed.error.as_deref(), Some("connection closed"));
        assert_eq!(failed.timing.duration_ns(), Some(2_000_000));
        assert_eq!(failed.usage, None);
        assert_eq!(failed.cumulative_usage, Some(expected_cumulative));
        assert_eq!(
            failed.anchor,
            Some(TrajectoryItemId::Assistant(kcastle_agent::RequestId::from(
                "request-2"
            )))
        );
    }

    #[test]
    fn prompt_updates_and_tool_schema_share_the_canonical_request_payload() {
        let schema = |description: &str| {
            serde_json::json!({
                "type": "function",
                "name": "shell",
                "description": description,
                "parameters": {"type": "object"},
                "strict": false
            })
        };
        let mut events = crate::domain::session_document::tests::fixture();
        let SessionEvent::RequestSnapshot { tools, .. } = &mut events[5].event else {
            panic!("fixture request snapshot moved");
        };
        *tools = vec![serde_json::from_value(schema("first schema")).unwrap()];
        let SessionEvent::RequestSnapshot {
            instructions,
            tools,
            ..
        } = &mut events[24].event
        else {
            panic!("fixture resume snapshot moved");
        };
        *instructions = Some("be concise".to_owned());
        *tools = vec![serde_json::from_value(schema("updated schema")).unwrap()];

        let document = SessionDocument::from_events(events).unwrap();
        let projection = TrajectoryProjection::from_document(&document);
        let first_request = &projection.requests[0];
        let resumed_request = &projection.requests[2];
        let first_prompt = first_request.prompt.as_ref().unwrap();
        let resumed_prompt = resumed_request.prompt.as_ref().unwrap();
        assert_eq!(first_prompt.instructions(), "be precise");
        assert_eq!(first_prompt.tool_count(), 1);
        assert_eq!(first_prompt.tool_schemas().len(), 1);
        assert_eq!(resumed_prompt.instructions(), "be concise");
        assert_ne!(first_prompt.tools_json(), resumed_prompt.tools_json());

        let initial_id = TrajectoryItemId::PromptChange(5);
        let TrajectoryRecordDetails::PromptChange {
            kind,
            current,
            previous,
        } = projection.record_details(&initial_id).unwrap()
        else {
            panic!("initial system row lost its prompt details");
        };
        assert_eq!(*kind, PromptChangeKind::Initial);
        assert!(Arc::ptr_eq(current, first_prompt));
        assert!(previous.is_none());

        let update_id = TrajectoryItemId::PromptChange(24);
        let TrajectoryRecordDetails::PromptChange {
            kind,
            current,
            previous,
        } = projection.record_details(&update_id).unwrap()
        else {
            panic!("prompt update row lost its diff boundary");
        };
        assert_eq!(*kind, PromptChangeKind::SystemAndTools);
        assert!(Arc::ptr_eq(current, resumed_prompt));
        assert!(Arc::ptr_eq(previous.as_ref().unwrap(), first_prompt));

        let tool_id = TrajectoryItemId::Tool(kcastle_agent::CallId::from("call-1"));
        let details = projection.record_details(&tool_id).unwrap();
        let TrajectoryRecordDetails::Tool {
            request_key,
            parent_call_id,
            prompt,
            schema_name,
        } = details
        else {
            panic!("tool row lost its request relation");
        };
        assert_eq!(request_key, &first_request.key);
        assert_eq!(parent_call_id, &None);
        assert_eq!(schema_name.as_ref(), "shell");
        assert!(Arc::ptr_eq(prompt, first_prompt));
        let schema: serde_json::Value =
            serde_json::from_str(details.tool_schema().unwrap()).unwrap();
        assert_eq!(schema["name"], "shell");
        assert_eq!(schema["description"], "first schema");

        let initial_index = projection.record_index(&initial_id).unwrap();
        let update_index = projection.record_index(&update_id).unwrap();
        let tool_index = projection.record_index(&tool_id).unwrap();
        assert!(projection.record_matches_terms(initial_index, &["precise".into()]));
        assert!(projection.record_matches_terms(update_index, &["updated schema".into()]));
        assert!(projection.record_matches_terms(tool_index, &["first schema".into()]));
        assert!(!projection.record_matches_terms(tool_index, &["updated schema".into()]));
    }

    #[test]
    fn unchanged_prompt_is_arc_reused_and_nested_tool_keeps_its_parent() {
        let mut events = crate::domain::session_document::tests::fixture();
        let SessionEvent::RequestSnapshot { tools, .. } = &mut events[5].event else {
            panic!("fixture request snapshot moved");
        };
        *tools = ["parent", "child"]
            .into_iter()
            .map(|name| {
                serde_json::from_value(serde_json::json!({
                    "type": "function",
                    "name": name,
                    "parameters": {"type": "object"},
                    "strict": false
                }))
                .unwrap()
            })
            .collect();
        events.truncate(7);
        let request_id = kcastle_agent::RequestId::from("request-1");
        let parent_id = kcastle_agent::CallId::from("parent-call");
        let child_id = kcastle_agent::CallId::from("child-call");
        for (seq, call_id, name) in [
            (7, parent_id.clone(), "parent"),
            (8, child_id.clone(), "child"),
        ] {
            events.push(crate::domain::session_document::tests::recorded(
                seq,
                SessionEvent::AssistantChunk {
                    request_id: request_id.clone(),
                    chunk: kcastle_agent::AssistantChunk::ToolCallDelta {
                        call_id,
                        name: Some(name.to_owned()),
                        arguments_delta: "{}".to_owned(),
                    },
                },
            ));
        }
        let call_item = |call_id: &kcastle_agent::CallId, name: &str| {
            serde_json::from_value::<kcastle_agent::InputItem>(serde_json::json!({
                "type": "function_call",
                "arguments": "{}",
                "call_id": call_id.as_str(),
                "name": name
            }))
            .unwrap()
        };
        events.push(crate::domain::session_document::tests::recorded(
            9,
            SessionEvent::AssistantCompleted {
                request_id: request_id.clone(),
                items: vec![
                    call_item(&parent_id, "parent"),
                    call_item(&child_id, "child"),
                ],
                response: kcastle_agent::ResponseInfo {
                    id: "response".to_owned(),
                    model: "deepseek-v4".to_owned(),
                    usage: None,
                },
            },
        ));
        events.push(crate::domain::session_document::tests::recorded(
            10,
            SessionEvent::ToolCallRequested {
                request_id: request_id.clone(),
                call_id: parent_id.clone(),
                parent_call_id: None,
            },
        ));
        events.push(crate::domain::session_document::tests::recorded(
            11,
            SessionEvent::ToolCallRequested {
                request_id: request_id.clone(),
                call_id: child_id.clone(),
                parent_call_id: Some(parent_id.clone()),
            },
        ));

        let projection =
            TrajectoryProjection::from_document(&SessionDocument::from_events(events).unwrap());
        let child = TrajectoryItemId::Tool(child_id);
        let details = projection.record_details(&child).unwrap();
        let TrajectoryRecordDetails::Tool {
            request_key,
            parent_call_id,
            schema_name,
            prompt,
        } = details
        else {
            panic!("child tool relation was not materialized");
        };
        assert_eq!(
            request_key,
            &TrajectoryRequestKey::Model(request_id.clone())
        );
        assert_eq!(parent_call_id.as_ref(), Some(&parent_id));
        assert_eq!(schema_name.as_ref(), "child");
        assert!(details.tool_schema().is_some());
        assert!(Arc::ptr_eq(
            prompt,
            projection.requests[0].prompt.as_ref().unwrap()
        ));
        assert_eq!(projection.requests[0].tool_call_count, 1);
        assert_eq!(projection.requests[0].subtool_call_count, 1);

        let unchanged = TrajectoryProjection::from_document(
            &SessionDocument::from_events(crate::domain::session_document::tests::fixture())
                .unwrap(),
        );
        assert!(Arc::ptr_eq(
            unchanged.requests[0].prompt.as_ref().unwrap(),
            unchanged.requests[2].prompt.as_ref().unwrap()
        ));
    }

    #[test]
    fn running_and_pre_output_failure_requests_have_canonical_boundary_states() {
        let mut events = crate::domain::session_document::tests::fixture();
        events.truncate(7);
        let mut document = SessionDocument::from_events(events.clone()).unwrap();
        let running = TrajectoryProjection::from_document(&document);
        let key = TrajectoryRequestKey::Model(kcastle_agent::RequestId::from("request-1"));
        let request = running.request_by_key(&key).unwrap();
        assert_eq!(request.status, ItemStatus::Running);
        assert_eq!(request.turn, Some(1));
        assert_eq!(request.step, Some(1));
        assert_eq!(request.anchor, None);
        assert_eq!(request.result, None);
        assert_eq!(request.timing.duration_ns(), None);
        assert_eq!(
            running
                .unanchored_requests
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![UnanchoredRequestPresentation {
                request_index: 0,
                source_seq: 5,
                status: ItemStatus::Running,
            }]
        );

        let mut terminal_events = events.clone();
        terminal_events.push(crate::domain::session_document::tests::recorded(
            7,
            kcastle_agent::SessionEvent::StepTerminated {
                step_id: kcastle_agent::StepId::from("step-1"),
                outcome: kcastle_agent::StepOutcome::Completed,
                error: None,
            },
        ));
        let terminal = TrajectoryProjection::from_document(
            &SessionDocument::from_events(terminal_events).unwrap(),
        );
        assert_eq!(
            terminal
                .unanchored_requests
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![UnanchoredRequestPresentation {
                request_index: 0,
                source_seq: 5,
                status: ItemStatus::Completed,
            }]
        );

        let failure = crate::domain::session_document::tests::recorded(
            7,
            kcastle_agent::SessionEvent::ModelRequestFailed {
                request_id: kcastle_agent::RequestId::from("request-1"),
                error: "provider unavailable".to_owned(),
            },
        );
        events.push(failure.clone());
        let delta = document.apply_batch(vec![failure]).unwrap();
        let failed = TrajectoryProjection::after_delta(&document, &delta, &running);
        let replayed =
            TrajectoryProjection::from_document(&SessionDocument::from_events(events).unwrap());
        assert_eq!(failed.requests, replayed.requests);

        let request = failed.request_by_key(&key).unwrap();
        assert_eq!(request.status, ItemStatus::Failed);
        assert_eq!(request.error.as_deref(), Some("provider unavailable"));
        assert_eq!(request.timing.duration_ns(), Some(1_000_000));
        assert_eq!(request.anchor, None);
        assert_eq!(request.result, None);
        assert_eq!(failed.records, running.records);
        assert_eq!(
            failed.request_boundary_revision(),
            running.request_boundary_revision(),
            "a lifecycle-only update does not invalidate marker geometry"
        );
        assert_eq!(
            failed
                .unanchored_requests
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![UnanchoredRequestPresentation {
                request_index: 0,
                source_seq: 5,
                status: ItemStatus::Failed,
            }]
        );
    }

    #[test]
    fn retry_attempts_share_one_source_ordered_boundary_index() {
        let run_id = kcastle_agent::RunId::from("run");
        let turn_id = kcastle_agent::TurnId::from("turn");
        let step_id = kcastle_agent::StepId::from("step");
        let first_id = kcastle_agent::RequestId::from("request-1");
        let second_id = kcastle_agent::RequestId::from("request-2");
        let snapshot =
            |request_id: kcastle_agent::RequestId, reason| SessionEvent::RequestSnapshot {
                request_id,
                step_id: step_id.clone(),
                reason,
                model: "model".to_owned(),
                instructions: Some("instructions".to_owned()),
                tools: Vec::new(),
                reasoning_effort: None,
                max_output_tokens: None,
                session_config: kcastle_agent::SessionConfig::default(),
            };
        let events = vec![
            crate::domain::session_document::tests::recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: run_id.clone(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                1,
                SessionEvent::TurnStarted { run_id, turn_id },
            ),
            crate::domain::session_document::tests::recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id: kcastle_agent::TurnId::from("turn"),
                    step_id: step_id.clone(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                3,
                snapshot(
                    first_id.clone(),
                    kcastle_agent::RequestHeaderReason::Initial,
                ),
            ),
            crate::domain::session_document::tests::recorded(
                4,
                SessionEvent::ModelRequestStarted {
                    request_id: first_id.clone(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                5,
                SessionEvent::ModelRequestFailed {
                    request_id: first_id,
                    error: "retry".to_owned(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                6,
                snapshot(
                    second_id.clone(),
                    kcastle_agent::RequestHeaderReason::Resume,
                ),
            ),
            crate::domain::session_document::tests::recorded(
                7,
                SessionEvent::ModelRequestStarted {
                    request_id: second_id.clone(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                8,
                SessionEvent::AssistantChunk {
                    request_id: second_id.clone(),
                    chunk: kcastle_agent::AssistantChunk::OutputTextDelta {
                        delta: "recovered".to_owned(),
                    },
                },
            ),
        ];
        let projection =
            TrajectoryProjection::from_document(&SessionDocument::from_events(events).unwrap());
        let boundary = TrajectoryItemId::Assistant(second_id);
        assert_eq!(
            projection
                .requests_for_boundary(&boundary)
                .map(|request| (request.number, request.status))
                .collect::<Vec<_>>(),
            vec![(1, ItemStatus::Failed), (2, ItemStatus::Running)]
        );
        assert!(projection.unanchored_requests.is_empty());
        assert_eq!(
            projection
                .request_for_record(&boundary)
                .map(|request| request.number),
            Some(2),
            "the legacy single-value selector retains last-request behavior"
        );
    }

    #[test]
    fn turn_owned_compaction_uses_replayed_active_step_without_inventing_options() {
        use kcastle_agent::{
            CompactionId, ResponseInfo, RunId, SessionEvent, StepId, StepOutcome, TokenUsage,
            TurnId,
        };

        let events = vec![
            crate::domain::session_document::tests::recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: RunId::from("run"),
                },
            ),
            crate::domain::session_document::tests::recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id: RunId::from("run"),
                    turn_id: TurnId::from("turn"),
                },
            ),
            crate::domain::session_document::tests::recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id: TurnId::from("turn"),
                    step_id: StepId::from("step"),
                },
            ),
            crate::domain::session_document::tests::recorded(
                3,
                SessionEvent::CompactionStarted {
                    compaction_id: CompactionId::from("compaction"),
                    run_id: RunId::from("run"),
                    tokens_before: 10_000,
                    first_kept_id: 42,
                },
            ),
            crate::domain::session_document::tests::recorded(
                4,
                SessionEvent::CompactionFinished {
                    compaction_id: CompactionId::from("compaction"),
                    outcome: StepOutcome::Completed,
                    summary: Some("summary".into()),
                    response: Some(ResponseInfo {
                        id: "response".into(),
                        model: "model".into(),
                        usage: Some(TokenUsage::default()),
                    }),
                },
            ),
        ];
        let document = SessionDocument::from_events(events).unwrap();
        let projection = TrajectoryProjection::from_document(&document);
        let request = &projection.requests[0];
        assert_eq!(request.purpose, TrajectoryRequestPurpose::Compaction);
        assert_eq!(request.turn, Some(1));
        assert_eq!(request.step, Some(1));
        assert_eq!(request.options, None);
        assert_eq!(request.response_model.as_deref(), Some("model"));
    }

    #[test]
    fn every_request_projection_prefix_matches_replay_and_updates_only_its_suffix() {
        let events = crate::domain::session_document::tests::fixture();
        let mut document = SessionDocument::default();
        let mut projection = TrajectoryProjection::from_document(&document);

        for event in events {
            let first_before = projection.requests.front().cloned();
            let materialized_before = projection.materialized_requests();
            let delta = document.apply_batch(vec![event]).unwrap();
            let next = TrajectoryProjection::after_delta(&document, &delta, &projection);
            let replayed = TrajectoryProjection::from_document(&document);
            assert_eq!(next.requests, replayed.requests);
            for id in document.trajectory_ids() {
                assert_eq!(next.record_details(id), replayed.record_details(id));
            }
            for request in &next.requests {
                assert_eq!(
                    next.request_index(&request.key),
                    replayed.request_index(&request.key)
                );
                if let Some(record) = request.anchor.as_ref().or(request.result.as_ref()) {
                    assert_eq!(
                        next.request_for_record(record).map(|request| &request.key),
                        Some(&request.key)
                    );
                }
            }
            let materialized_delta = next
                .materialized_requests()
                .saturating_sub(materialized_before);
            assert!(
                materialized_delta <= 1,
                "only the changed tail request is materialized"
            );
            if delta.changed_requests.is_empty()
                && let (Some(previous), Some(current)) =
                    (first_before.as_ref(), next.requests.front())
            {
                assert!(Arc::ptr_eq(previous, current));
            }
            projection = next;
        }
    }

    #[test]
    fn ten_thousand_requests_update_the_tail_without_a_full_index_scan() {
        const REQUESTS: usize = 10_000;
        let run_id = kcastle_agent::RunId::from("run");
        let turn_id = kcastle_agent::TurnId::from("turn");
        let step_id = kcastle_agent::StepId::from("step");
        let mut events = vec![
            crate::domain::session_document::tests::recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: run_id.clone(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id,
                    turn_id: turn_id.clone(),
                },
            ),
            crate::domain::session_document::tests::recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id,
                    step_id: step_id.clone(),
                },
            ),
        ];
        for index in 0..REQUESTS {
            events.push(crate::domain::session_document::tests::recorded(
                u64::try_from(index).unwrap().saturating_add(3),
                SessionEvent::RequestSnapshot {
                    request_id: kcastle_agent::RequestId::from_raw(format!("request-{index}")),
                    step_id: step_id.clone(),
                    reason: if index == 0 {
                        kcastle_agent::RequestHeaderReason::Initial
                    } else {
                        kcastle_agent::RequestHeaderReason::Resume
                    },
                    model: "model".to_owned(),
                    instructions: Some("shared instructions".to_owned()),
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: None,
                    session_config: kcastle_agent::SessionConfig::default(),
                },
            ));
        }
        let mut document = SessionDocument::from_events(events).unwrap();
        let projection = TrajectoryProjection::from_document(&document);
        assert_eq!(projection.requests.len(), REQUESTS);
        assert_eq!(projection.unanchored_requests.len(), REQUESTS);
        assert_eq!(projection.materialized_requests(), REQUESTS);
        assert_eq!(projection.request_index_work(), REQUESTS);
        let boundary_revision = projection.request_boundary_revision();
        assert!(
            projection
                .unanchored_requests
                .iter()
                .zip(projection.unanchored_requests.iter().skip(1))
                .all(|(left, right)| left.source_seq < right.source_seq),
            "unanchored request descriptors remain in source order"
        );
        let first = Arc::clone(&projection.requests[0]);
        let materialized_before = projection.materialized_requests();
        let work_before = projection.request_index_work();
        let tail_id = kcastle_agent::RequestId::from_raw(format!("request-{}", REQUESTS - 1));
        let delta = document
            .apply_batch(vec![crate::domain::session_document::tests::recorded(
                u64::try_from(REQUESTS).unwrap().saturating_add(3),
                SessionEvent::ModelRequestStarted {
                    request_id: tail_id.clone(),
                },
            )])
            .unwrap();
        let next = TrajectoryProjection::after_delta(&document, &delta, &projection);

        assert_eq!(
            next.materialized_requests()
                .saturating_sub(materialized_before),
            1
        );
        assert!(
            next.request_index_work().saturating_sub(work_before) <= 3,
            "one hash lookup plus one old/new suffix index update is sufficient"
        );
        assert!(Arc::ptr_eq(&first, &next.requests[0]));
        assert_eq!(
            next.request_index(&TrajectoryRequestKey::Model(tail_id.clone())),
            Some(REQUESTS - 1)
        );
        assert_eq!(next.unanchored_requests.len(), REQUESTS);
        assert_eq!(next.request_boundary_revision(), boundary_revision);
        assert_eq!(
            next.unanchored_requests
                .back()
                .map(|request| request.status),
            Some(ItemStatus::Running)
        );

        let boundary_work_before = next.request_index_work();
        let assistant_id = TrajectoryItemId::Assistant(tail_id.clone());
        let delta = document
            .apply_batch(vec![crate::domain::session_document::tests::recorded(
                u64::try_from(REQUESTS).unwrap().saturating_add(4),
                SessionEvent::AssistantChunk {
                    request_id: tail_id,
                    chunk: kcastle_agent::AssistantChunk::OutputTextDelta {
                        delta: "done".to_owned(),
                    },
                },
            )])
            .unwrap();
        let bounded = TrajectoryProjection::after_delta(&document, &delta, &next);
        assert!(bounded.unanchored_requests.is_empty());
        assert_eq!(
            bounded.request_boundary_revision(),
            boundary_revision.saturating_add(1)
        );
        assert!(
            bounded
                .request_index_work()
                .saturating_sub(boundary_work_before)
                <= REQUESTS.saturating_mul(3),
            "introducing a shared boundary does linear index work"
        );
        assert!(
            bounded
                .requests
                .iter()
                .all(|request| request.anchor.as_ref() == Some(&assistant_id)),
            "all retry attempts in the step share its first model boundary"
        );
        let mut requests = bounded.requests_for_boundary(&assistant_id);
        assert_eq!(requests.next().map(|request| request.number), Some(1));
        assert_eq!(
            requests.next_back().map(|request| request.number),
            Some(10_000)
        );
    }

    fn request_usage_fixture() -> TokenUsage {
        TokenUsage {
            uncached_input_tokens: 20,
            cache_read_input_tokens: 80,
            cache_write_input_tokens: 3,
            output_tokens: 15,
            reasoning_output_tokens: 5,
        }
    }
}
