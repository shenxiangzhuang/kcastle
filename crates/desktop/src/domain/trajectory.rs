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
    /// Pre-normalized once when the immutable record is materialized. Rendering and hover/search
    /// selectors can then scan without allocating lowercase copies on every GPUI notification.
    pub(crate) search_text: String,
    pub(crate) turn: Option<u32>,
    pub(crate) step: Option<u32>,
    pub(crate) status: ItemStatus,
    pub(crate) timing: RecordTiming,
    pub(crate) usage: Option<TokenUsage>,
}

impl TrajectoryRecord {
    pub(crate) fn matches(&self, query: &str) -> bool {
        query.is_empty() || self.search_text.contains(query)
    }

    pub(crate) fn lane(&self) -> TrajectoryLane {
        match self.kind {
            TrajectoryKind::System
            | TrajectoryKind::User
            | TrajectoryKind::Steering
            | TrajectoryKind::Context => TrajectoryLane::Input,
            TrajectoryKind::Assistant
            | TrajectoryKind::Compaction
            | TrajectoryKind::RequestFailure => TrajectoryLane::Model,
            TrajectoryKind::Tool => TrajectoryLane::Tools,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryProjection {
    pub(crate) records: Vector<Arc<TrajectoryRecord>>,
    index_by_id: HashMap<TrajectoryItemId, usize>,
    stats: SessionStats,
    trajectory_revision: u64,
    geometry_revision: u64,
    projection_lineage: u64,
    /// One cursor for every incremental trajectory consumer. Search and geometry use field flags
    /// from the same bounded journal instead of maintaining independent revision histories.
    change_revision: u64,
    change_history: Vector<TrajectoryChangeDelta>,
    #[cfg(test)]
    materialized_records: usize,
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
        #[cfg(test)]
        let materialized_records = if let Some(previous) = previous
            && previous.trajectory_revision == revisions.trajectory
        {
            previous.materialized_records
        } else {
            records.len()
        };

        let (change_revision, change_history) =
            change_state_after_full_rebuild(previous, revisions.trajectory);
        Self {
            records,
            index_by_id,
            stats,
            trajectory_revision: revisions.trajectory,
            geometry_revision: revisions.geometry,
            projection_lineage: previous
                .filter(|value| value.projection_lineage != 0)
                .map_or_else(next_projection_lineage, |value| value.projection_lineage),
            change_revision,
            change_history,
            #[cfg(test)]
            materialized_records,
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
        Self {
            records,
            index_by_id: previous.index_by_id.clone(),
            stats,
            trajectory_revision: revisions.trajectory,
            geometry_revision: revisions.geometry,
            projection_lineage: previous.projection_lineage,
            change_revision,
            change_history,
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
        if suffix.is_empty()
            || suffix
                .iter()
                .any(|id| previous.index_by_id.contains_key(id))
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
            let index = *index_by_id.get(id)?;
            if index >= previous_len {
                continue;
            }
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
        let (change_revision, change_history) = advance_change_state(
            previous,
            search_indices,
            geometry_indices,
            !search_reset,
            delta.geometry_changed,
        );
        Some(Self {
            records,
            index_by_id,
            stats: if delta.stats_changed {
                document.stats()
            } else {
                previous.stats
            },
            trajectory_revision: revisions.trajectory,
            geometry_revision: revisions.geometry,
            projection_lineage: previous.projection_lineage,
            change_revision,
            change_history,
            #[cfg(test)]
            materialized_records: previous.materialized_records.saturating_add(materialized),
        })
    }

    #[cfg(test)]
    pub(crate) fn materialized_records(&self) -> usize {
        self.materialized_records
    }
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
    let title = item.title.to_owned();
    let text = item.text.to_owned();
    let payload = item.payload.map(ToOwned::to_owned);
    let search_text = normalized_search_text(&title, &text, payload.as_deref());
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
}
