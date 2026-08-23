use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use im::{HashMap, Vector};
use kcastle_agent::TokenUsage;

use crate::domain::session_document::{
    DisplayOrdinals, ItemStatus, ProjectionDelta, SessionDocument, SessionStats, TrajectoryItemView,
};
pub(crate) use crate::domain::session_document::{
    TimingMetrics as RecordTiming, TrajectoryItemId, TrajectoryKind, TrajectoryLane,
};

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

/// Immutable UI record materialized from one canonical document revision.
///
/// The stable domain identity is deliberately retained. Presentation code must
/// never infer identity from the record's position or from a journal sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrajectoryRecord {
    pub(crate) id: TrajectoryItemId,
    pub(crate) source_seq: u64,
    pub(crate) kind: TrajectoryKind,
    pub(crate) lane: TrajectoryLane,
    pub(crate) title: String,
    pub(crate) text: String,
    pub(crate) payload: Option<String>,
    /// Pre-normalized once when the immutable record is materialized. Rendering and hover/search
    /// selectors can then scan without allocating lowercase copies on every GPUI notification.
    pub(crate) search_text: String,
    pub(crate) turn: Option<u32>,
    pub(crate) step: Option<u32>,
    pub(crate) call_id: Option<String>,
    pub(crate) status: TrajectoryStatus,
    pub(crate) timing: RecordTiming,
    pub(crate) usage: Option<TrajectoryUsage>,
}

impl TrajectoryRecord {
    pub(crate) fn matches(&self, query: &str) -> bool {
        query.is_empty() || self.search_text.contains(query)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryProjection {
    pub(crate) records: Vector<Arc<TrajectoryRecord>>,
    index_by_id: HashMap<TrajectoryItemId, usize>,
    stats: TrajectoryStats,
    document_revision: u64,
    trajectory_revision: u64,
    geometry_revision: u64,
    stats_revision: u64,
    projection_lineage: u64,
    /// Monotonic revision for the small subset of record fields used by trajectory search and
    /// ledger filtering. Timing- and usage-only receipts intentionally leave this unchanged.
    search_revision: u64,
    search_history: Vector<TrajectorySearchDelta>,
    geometry_history: Vector<TrajectoryGeometryDelta>,
    #[cfg(test)]
    materialized_records: usize,
}

const SEARCH_HISTORY_LIMIT: usize = 256;
const GEOMETRY_HISTORY_LIMIT: usize = 256;
static NEXT_PROJECTION_LINEAGE: AtomicU64 = AtomicU64::new(1);

/// One incremental search/filter transition. Indices name their final value in `records`; replaying
/// every delta since a cached revision is therefore idempotent even when one streaming record is
/// mentioned repeatedly. A reset is reserved for the rare case where row identity changes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TrajectorySearchDelta {
    revision: u64,
    changed_indices: Vector<usize>,
    reset: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct TrajectoryGeometryDelta {
    from_revision: u64,
    to_revision: u64,
    changed_indices: Vector<usize>,
    sequence_compatible: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryGeometryChanges {
    pub(crate) changed_indices: Vector<usize>,
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
        let records: Vector<Arc<TrajectoryRecord>> = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            previous.records.clone()
        } else {
            let ordinals = document.display_ordinals();
            let previous_records = previous
                .into_iter()
                .flat_map(|projection| projection.records.iter())
                .map(|record| (record.id.clone(), record))
                .collect::<HashMap<_, _>>();

            document
                .trajectory()
                .into_iter()
                .map(|item| {
                    let record = Arc::new(materialize_record(ordinals, item));
                    previous_records
                        .get(&record.id)
                        .filter(|previous| previous.as_ref() == record.as_ref())
                        .map_or(record, |previous| Arc::clone(previous))
                })
                .collect()
        };
        let stats = if let Some(previous) = previous
            && previous.stats_revision == revisions.stats
        {
            previous.stats
        } else {
            stats_snapshot(document.stats())
        };

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
        #[cfg(test)]
        let materialized_records = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            previous.materialized_records
        } else {
            records.len()
        };

        let (search_revision, search_history) =
            search_state_after_full_rebuild(previous, revisions.trajectory);
        let geometry_history = geometry_state_after_full_rebuild(previous, revisions.geometry);
        Self {
            records,
            index_by_id,
            stats,
            document_revision: revisions.document,
            trajectory_revision: revisions.trajectory,
            geometry_revision: revisions.geometry,
            stats_revision: revisions.stats,
            projection_lineage: previous
                .filter(|value| value.projection_lineage != 0)
                .map_or_else(next_projection_lineage, |value| value.projection_lineage),
            search_revision,
            search_history,
            geometry_history,
            #[cfg(test)]
            materialized_records,
        }
    }

    pub(crate) fn stats(&self) -> TrajectoryStats {
        self.stats
    }

    /// Geometry caches use only the canonical geometry revision. Text-only
    /// streaming updates therefore cannot invalidate an unchanged timeline.
    pub(crate) fn revision(&self) -> u64 {
        self.geometry_revision
    }

    pub(crate) fn source_revision(&self) -> u64 {
        self.document_revision
    }

    pub(crate) fn search_revision(&self) -> u64 {
        self.search_revision
    }

    pub(crate) fn projection_lineage(&self) -> u64 {
        self.projection_lineage
    }

    pub(crate) fn geometry_changes_since(
        &self,
        revision: u64,
    ) -> Option<TrajectoryGeometryChanges> {
        if revision == self.geometry_revision {
            return Some(TrajectoryGeometryChanges::default());
        }
        if revision > self.geometry_revision {
            return None;
        }
        let start = self
            .geometry_history
            .iter()
            .position(|delta| delta.from_revision == revision)?;
        let mut expected = revision;
        let mut changed = std::collections::HashSet::new();
        for delta in self.geometry_history.iter().skip(start) {
            if delta.from_revision != expected || !delta.sequence_compatible {
                return None;
            }
            changed.extend(delta.changed_indices.iter().copied());
            expected = delta.to_revision;
            if expected == self.geometry_revision {
                let mut changed_indices = changed.into_iter().collect::<Vector<_>>();
                changed_indices.sort();
                return Some(TrajectoryGeometryChanges { changed_indices });
            }
        }
        None
    }

    /// Returns only indices whose searchable text changed since `revision`. `None` means the
    /// bounded history cannot prove continuity and the caller must rebuild once. Empty iteration
    /// is the common timing/usage receipt path and never touches historical records.
    pub(crate) fn search_changed_indices_since(
        &self,
        revision: u64,
    ) -> Option<impl Iterator<Item = usize> + '_> {
        if revision > self.search_revision {
            return None;
        }
        let start = if revision == self.search_revision {
            self.search_history.len()
        } else {
            let first_revision = self.search_history.front()?.revision;
            let next_revision = revision.saturating_add(1);
            if next_revision < first_revision {
                return None;
            }
            let start = usize::try_from(next_revision.saturating_sub(first_revision)).ok()?;
            let first = self.search_history.get(start)?;
            if first.revision != revision.saturating_add(1)
                || self
                    .search_history
                    .back()
                    .is_none_or(|last| last.revision != self.search_revision)
                || self
                    .search_history
                    .iter()
                    .skip(start)
                    .any(|delta| delta.reset)
            {
                return None;
            }
            start
        };
        Some(
            self.search_history
                .iter()
                .skip(start)
                .flat_map(search_delta_indices),
        )
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

    pub(crate) fn after_delta(
        document: &SessionDocument,
        delta: &ProjectionDelta,
        previous: &Self,
    ) -> Self {
        if delta.trajectory_order_changed {
            if let Some(projection) = Self::after_appended_order(document, delta, previous) {
                return projection;
            }
            return Self::from_document_reusing(document, Some(previous));
        }

        let revisions = document.revisions();
        let ordinals = document.display_ordinals();
        let mut records = previous.records.clone();
        let mut materialized = 0_usize;
        let mut search_indices = Vector::new();
        let mut search_reset = false;
        let mut geometry_indices = Vector::new();
        for id in &delta.changed_trajectory {
            let Some(index) = previous.index_by_id.get(id).copied() else {
                return Self::from_document_reusing(document, Some(previous));
            };
            let Some(item) = document.trajectory_by_id(id) else {
                return Self::from_document_reusing(document, Some(previous));
            };
            let record = Arc::new(materialize_record(ordinals, item));
            materialized = materialized.saturating_add(1);
            if records[index].as_ref() != record.as_ref() {
                let previous_record = &records[index];
                if previous_record.turn != record.turn || previous_record.kind != record.kind {
                    search_reset = true;
                } else if previous_record.search_text != record.search_text {
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
            stats_snapshot(document.stats())
        } else {
            previous.stats
        };
        let (search_revision, search_history) =
            advance_search_state(previous, search_indices, search_reset);
        let geometry_history =
            advance_geometry_state(previous, revisions.geometry, geometry_indices, true);
        Self {
            records,
            index_by_id: previous.index_by_id.clone(),
            stats,
            document_revision: revisions.document,
            trajectory_revision: revisions.trajectory,
            geometry_revision: revisions.geometry,
            stats_revision: revisions.stats,
            projection_lineage: previous.projection_lineage,
            search_revision,
            search_history,
            geometry_history,
            #[cfg(test)]
            materialized_records: previous.materialized_records.saturating_add(materialized),
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
        if canonical_ids.len() < previous_len
            || suffix.is_empty()
            || suffix
                .iter()
                .any(|id| previous.index_by_id.contains_key(id))
            || canonical_ids.len() != previous_len.saturating_add(suffix.len())
        {
            return None;
        }
        // Every appended canonical ID must be named by this transaction. This
        // rejects completion-time reorder/replacement and delegates that rare
        // case to the full equivalence path.
        if suffix
            .iter()
            .any(|id| !delta.changed_trajectory.contains(id))
        {
            return None;
        }

        let mut records = previous.records.clone();
        let mut index_by_id = previous.index_by_id.clone();
        let ordinals = document.display_ordinals();
        let mut materialized = 0_usize;
        let mut search_indices = Vector::new();
        let mut search_reset = false;
        let mut geometry_indices = Vector::new();
        for id in suffix {
            let item = document.trajectory_by_id(id)?;
            let record = Arc::new(materialize_record(ordinals, item));
            index_by_id.insert(id.clone(), records.len());
            search_indices.push_back(records.len());
            geometry_indices.push_back(records.len());
            records.push_back(record);
            materialized = materialized.saturating_add(1);
        }
        // A transaction may both append and update an existing row (for
        // example, revealing an assistant and requesting its first tool).
        for id in &delta.changed_trajectory {
            if suffix.contains(id) {
                continue;
            }
            let index = *index_by_id.get(id)?;
            let item = document.trajectory_by_id(id)?;
            let record = Arc::new(materialize_record(ordinals, item));
            materialized = materialized.saturating_add(1);
            if records[index].as_ref() != record.as_ref() {
                let previous_record = &records[index];
                if previous_record.turn != record.turn || previous_record.kind != record.kind {
                    search_reset = true;
                } else if previous_record.search_text != record.search_text {
                    search_indices.push_back(index);
                }
                if delta.geometry_changed && !same_geometry(previous_record, &record) {
                    geometry_indices.push_back(index);
                }
                records.set(index, record);
            }
        }
        let revisions = document.revisions();
        let (search_revision, search_history) =
            advance_search_state(previous, search_indices, search_reset);
        let geometry_history = advance_geometry_state(
            previous,
            revisions.geometry,
            geometry_indices,
            delta.geometry_changed,
        );
        Some(Self {
            records,
            index_by_id,
            stats: if delta.stats_changed {
                stats_snapshot(document.stats())
            } else {
                previous.stats
            },
            document_revision: revisions.document,
            trajectory_revision: revisions.trajectory,
            geometry_revision: revisions.geometry,
            stats_revision: revisions.stats,
            projection_lineage: previous.projection_lineage,
            search_revision,
            search_history,
            geometry_history,
            #[cfg(test)]
            materialized_records: previous.materialized_records.saturating_add(materialized),
        })
    }

    #[cfg(test)]
    pub(crate) fn materialized_records(&self) -> usize {
        self.materialized_records
    }
}

fn search_delta_indices(delta: &TrajectorySearchDelta) -> impl Iterator<Item = usize> + '_ {
    delta.changed_indices.iter().copied()
}

fn search_state_after_full_rebuild(
    previous: Option<&TrajectoryProjection>,
    trajectory_revision: u64,
) -> (u64, Vector<TrajectorySearchDelta>) {
    let Some(previous) = previous else {
        return (1, Vector::new());
    };
    if previous.trajectory_revision == trajectory_revision {
        return (previous.search_revision, previous.search_history.clone());
    }
    advance_search_state(previous, Vector::new(), true)
}

fn advance_search_state(
    previous: &TrajectoryProjection,
    changed_indices: Vector<usize>,
    reset: bool,
) -> (u64, Vector<TrajectorySearchDelta>) {
    if changed_indices.is_empty() && !reset {
        return (previous.search_revision, previous.search_history.clone());
    }
    let revision = previous.search_revision.saturating_add(1);
    let mut history = previous.search_history.clone();
    history.push_back(TrajectorySearchDelta {
        revision,
        changed_indices,
        reset,
    });
    while history.len() > SEARCH_HISTORY_LIMIT {
        history.pop_front();
    }
    (revision, history)
}

fn geometry_state_after_full_rebuild(
    previous: Option<&TrajectoryProjection>,
    geometry_revision: u64,
) -> Vector<TrajectoryGeometryDelta> {
    let Some(previous) = previous else {
        return Vector::new();
    };
    if previous.geometry_revision == geometry_revision {
        return previous.geometry_history.clone();
    }
    advance_geometry_state(previous, geometry_revision, Vector::new(), false)
}

fn advance_geometry_state(
    previous: &TrajectoryProjection,
    geometry_revision: u64,
    changed_indices: Vector<usize>,
    sequence_compatible: bool,
) -> Vector<TrajectoryGeometryDelta> {
    if previous.geometry_revision == geometry_revision {
        return previous.geometry_history.clone();
    }
    let mut history = previous.geometry_history.clone();
    history.push_back(TrajectoryGeometryDelta {
        from_revision: previous.geometry_revision,
        to_revision: geometry_revision,
        changed_indices,
        sequence_compatible,
    });
    while history.len() > GEOMETRY_HISTORY_LIMIT {
        history.pop_front();
    }
    history
}

fn same_geometry(previous: &TrajectoryRecord, current: &TrajectoryRecord) -> bool {
    previous.kind == current.kind
        && previous.lane == current.lane
        && previous.timing == current.timing
}

fn next_projection_lineage() -> u64 {
    NEXT_PROJECTION_LINEAGE.fetch_add(1, Ordering::Relaxed)
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
    let call_id = match item.id {
        TrajectoryItemId::Tool(call_id) => Some(call_id.to_string()),
        _ => None,
    };
    let title = item.title.to_owned();
    let text = item.text.to_owned();
    let payload = item.payload.map(ToOwned::to_owned);
    let search_text = normalized_search_text(&title, &text, payload.as_deref());
    TrajectoryRecord {
        id: item.id.clone(),
        source_seq: item.source_seqs.first().copied().unwrap_or_default(),
        kind: item.kind,
        lane: item.lane,
        title,
        text,
        payload,
        search_text,
        turn,
        step,
        call_id,
        status: status_snapshot(item.status),
        timing: item.timing.clone(),
        usage: item.usage.map(usage_snapshot),
    }
}

pub(crate) fn normalized_search_text(title: &str, text: &str, payload: Option<&str>) -> String {
    let mut normalized = String::with_capacity(
        title
            .len()
            .saturating_add(text.len())
            .saturating_add(payload.map_or(0, str::len))
            .saturating_add(2),
    );
    normalized.push_str(title);
    normalized.push('\n');
    normalized.push_str(text);
    if let Some(payload) = payload {
        normalized.push('\n');
        normalized.push_str(payload);
    }
    normalized.make_ascii_lowercase();
    // Unicode case folding can grow the string, so do it only when ASCII normalization was not
    // sufficient. This keeps the common English/tool payload path allocation-stable.
    if normalized.is_ascii() {
        normalized
    } else {
        normalized.to_lowercase()
    }
}

fn status_snapshot(status: ItemStatus) -> TrajectoryStatus {
    match status {
        ItemStatus::Pending | ItemStatus::Running => TrajectoryStatus::Running,
        ItemStatus::Completed => TrajectoryStatus::Completed,
        ItemStatus::Failed | ItemStatus::Aborted => TrajectoryStatus::Failed,
        ItemStatus::Denied => TrajectoryStatus::Denied,
        ItemStatus::NotExecuted => TrajectoryStatus::NotExecuted,
        ItemStatus::Unknown => TrajectoryStatus::Unknown,
    }
}

fn usage_snapshot(usage: TokenUsage) -> TrajectoryUsage {
    TrajectoryUsage {
        input_tokens: saturating_u32(usage.input_tokens()),
        output_tokens: saturating_u32(usage.total_output_tokens()),
        cached_tokens: saturating_u32(usage.cache_read_input_tokens),
    }
}

fn stats_snapshot(stats: SessionStats) -> TrajectoryStats {
    TrajectoryStats {
        turns: stats.turns,
        steps: stats.steps,
        llm_ns: stats.llm_ns,
        tool_ns: stats.tool_ns,
        ttft_ns: stats.ttft_ns,
        ttft_steps: stats.ttft_samples,
        decode_ns: stats.decode_ns,
        decode_tokens: stats.decode_tokens,
        input_tokens: stats.input_tokens(),
        output_tokens: stats.total_output_tokens(),
        cached_tokens: stats.cache_read_input_tokens,
    }
}

fn saturating_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
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
    use super::*;

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
    fn geometry_history_reports_only_changed_indices_and_rejects_resets() {
        let previous = TrajectoryProjection {
            geometry_revision: 10,
            projection_lineage: next_projection_lineage(),
            ..TrajectoryProjection::default()
        };
        let history = advance_geometry_state(&previous, 11, [77_777].into_iter().collect(), true);
        let current = TrajectoryProjection {
            geometry_revision: 11,
            projection_lineage: previous.projection_lineage,
            geometry_history: history,
            ..TrajectoryProjection::default()
        };
        assert_eq!(
            current.geometry_changes_since(10).unwrap().changed_indices,
            [77_777].into_iter().collect::<Vector<_>>()
        );

        let reset_history = advance_geometry_state(&current, 12, Vector::new(), false);
        let reset = TrajectoryProjection {
            geometry_revision: 12,
            projection_lineage: current.projection_lineage,
            geometry_history: reset_history,
            ..TrajectoryProjection::default()
        };
        assert!(reset.geometry_changes_since(10).is_none());
    }
}
