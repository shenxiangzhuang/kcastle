use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ops::Range;

pub(crate) use super::session_document::TrajectoryLane as TimelineLane;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum TimelineMode {
    #[default]
    Sequence,
    Duration,
    /// Retained as a projection primitive for timing verification; DSH keeps
    /// the corresponding desktop control hidden.
    #[allow(dead_code)]
    Actual,
}

const TIMELINE_LANES: [TimelineLane; 3] = [
    TimelineLane::Input,
    TimelineLane::Model,
    TimelineLane::Tools,
];

const fn lane_index(lane: TimelineLane) -> usize {
    match lane {
        TimelineLane::Input => 0,
        TimelineLane::Model => 1,
        TimelineLane::Tools => 2,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AxisId {
    pub(crate) document_generation: u64,
    pub(crate) geometry_revision: u64,
    pub(crate) mode: TimelineMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DomainRange {
    pub(crate) start: f64,
    pub(crate) end: f64,
}

impl DomainRange {
    pub(crate) fn new(start: f64, end: f64) -> Self {
        let start = finite(start);
        let end = finite(end);
        if start <= end {
            Self { start, end }
        } else {
            Self {
                start: end,
                end: start,
            }
        }
    }

    pub(crate) fn width(self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    pub(crate) fn clamp_to(self, domain: Self) -> Self {
        if domain.width() <= 0.0 {
            return domain;
        }
        let width = self.width().min(domain.width());
        let start = self
            .start
            .clamp(domain.start, (domain.end - width).max(domain.start));
        Self {
            start,
            end: start + width,
        }
    }

    /// Pans this viewport just far enough to reveal `target`. The current zoom
    /// is preserved unless the target itself is wider than the viewport.
    pub(crate) fn pan_to_reveal(self, target: Self, domain: Self) -> Self {
        let viewport = self.clamp_to(domain);
        let target = target.clamp_to(domain);
        if target.start >= viewport.start && target.end <= viewport.end {
            return viewport;
        }
        if target.width() >= viewport.width() {
            return target;
        }
        let width = viewport.width();
        if target.start < viewport.start {
            Self::new(target.start, target.start + width).clamp_to(domain)
        } else {
            Self::new(target.end - width, target.end).clamp_to(domain)
        }
    }

    pub(crate) fn value_at_fraction(self, fraction: f64) -> f64 {
        self.start + self.width() * finite(fraction).clamp(0.0, 1.0)
    }

    pub(crate) fn pan_from(self, domain: Self, anchor: f64, current: f64) -> Self {
        let delta = finite(anchor) - finite(current);
        Self::new(self.start + delta, self.end + delta).clamp_to(domain)
    }

    pub(crate) fn zoom(self, domain: Self, anchor: f64, factor: f64, minimum_width: f64) -> Self {
        let factor = finite(factor).clamp(0.05, 20.0);
        let minimum_width = finite(minimum_width).max(f64::EPSILON).min(domain.width());
        let width = (self.width() * factor).clamp(minimum_width, domain.width());
        let ratio = if self.width() <= 0.0 {
            0.5
        } else {
            ((anchor - self.start) / self.width()).clamp(0.0, 1.0)
        };
        Self::new(anchor - width * ratio, anchor + width * (1.0 - ratio)).clamp_to(domain)
    }

    pub(crate) fn auto_pan(
        self,
        domain: Self,
        pointer_fraction: f64,
        edge_fraction: f64,
        pan_step_fraction: f64,
    ) -> Self {
        if self.width() >= domain.width() {
            return self;
        }
        let pointer_fraction = finite(pointer_fraction).clamp(0.0, 1.0);
        let edge_fraction = finite(edge_fraction).clamp(0.0, 0.5);
        if edge_fraction == 0.0 {
            return self;
        }
        let edge_strength = if pointer_fraction < edge_fraction {
            -((edge_fraction - pointer_fraction) / edge_fraction)
        } else if pointer_fraction > 1.0 - edge_fraction {
            (pointer_fraction - (1.0 - edge_fraction)) / edge_fraction
        } else {
            0.0
        };
        if edge_strength == 0.0 {
            return self;
        }
        let delta = edge_strength.signum()
            * self.width()
            * finite(pan_step_fraction).clamp(0.0, 1.0)
            * edge_strength.abs().clamp(0.2, 1.0);
        Self::new(self.start + delta, self.end + delta).clamp_to(domain)
    }

    pub(crate) fn with_minimum_width(self, domain: Self, minimum_width: f64) -> Self {
        let mut range = self.clamp_to(domain);
        let minimum_width = finite(minimum_width).max(0.0).min(domain.width());
        if range.width() < minimum_width {
            let center = (range.start + range.end) / 2.0;
            range = Self::new(center - minimum_width / 2.0, center + minimum_width / 2.0)
                .clamp_to(domain);
        }
        range
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AxisRange {
    pub(crate) axis: AxisId,
    pub(crate) range: DomainRange,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimelinePoint {
    pub(crate) wall_ms: f64,
    pub(crate) clock_id: String,
    pub(crate) monotonic_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimelineSpan {
    pub(crate) id: usize,
    pub(crate) lane: TimelineLane,
    pub(crate) sequence: u64,
    pub(crate) started: Option<TimelinePoint>,
    pub(crate) completed: Option<TimelinePoint>,
    pub(crate) duration_ms: Option<f64>,
    pub(crate) nested: Option<(TimelinePoint, TimelinePoint)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GeometryCell {
    pub(crate) id: usize,
    pub(crate) lane: TimelineLane,
    pub(crate) range: DomainRange,
    pub(crate) nested: Option<DomainRange>,
}

#[derive(Clone, Debug)]
struct IndexedInterval {
    id: usize,
    start: f64,
    end: f64,
    _order: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RenderWork {
    scanned_intervals: usize,
    aggregate_queries: usize,
    aggregate_nodes: usize,
    stored_identity_words: usize,
}

#[derive(Clone, Copy, Debug)]
struct IntervalAggregate {
    max_end: f64,
    representative_entry: usize,
}

impl Default for IntervalAggregate {
    fn default() -> Self {
        Self {
            max_end: f64::NEG_INFINITY,
            representative_entry: usize::MAX,
        }
    }
}

/// A range-max index used only when interval ends or paint order are not monotonic.
/// Sequence timelines take the O(1) range-summary path and allocate no tree.
#[derive(Debug)]
struct IntervalAggregateTree {
    capacity: usize,
    nodes: Vec<IntervalAggregate>,
}

impl IntervalAggregateTree {
    fn new(entries: &[IndexedInterval]) -> Self {
        let capacity = entries.len().max(1).next_power_of_two();
        let mut tree = Self {
            capacity,
            nodes: vec![IntervalAggregate::default(); capacity * 2],
        };
        for (index, entry) in entries.iter().enumerate() {
            tree.nodes[capacity + index] = IntervalAggregate {
                max_end: entry.end,
                representative_entry: index,
            };
        }
        for index in (1..capacity).rev() {
            tree.nodes[index] =
                combine_aggregates(tree.nodes[index * 2], tree.nodes[index * 2 + 1], entries);
        }
        tree
    }

    fn update(&mut self, index: usize, entries: &[IndexedInterval]) {
        if index >= self.capacity {
            *self = Self::new(entries);
            return;
        }
        let mut node = self.capacity + index;
        self.nodes[node] = IntervalAggregate {
            max_end: entries[index].end,
            representative_entry: index,
        };
        node /= 2;
        while node > 0 {
            self.nodes[node] =
                combine_aggregates(self.nodes[node * 2], self.nodes[node * 2 + 1], entries);
            node /= 2;
        }
    }

    fn query(
        &self,
        range: Range<usize>,
        entries: &[IndexedInterval],
        work: &mut RenderWork,
    ) -> IntervalAggregate {
        let mut left = self.capacity + range.start;
        let mut right = self.capacity + range.end;
        let mut aggregate = IntervalAggregate::default();
        while left < right {
            if left % 2 == 1 {
                aggregate = combine_aggregates(aggregate, self.nodes[left], entries);
                work.aggregate_nodes = work.aggregate_nodes.saturating_add(1);
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                aggregate = combine_aggregates(aggregate, self.nodes[right], entries);
                work.aggregate_nodes = work.aggregate_nodes.saturating_add(1);
            }
            left /= 2;
            right /= 2;
        }
        aggregate
    }
}

fn combine_aggregates(
    left: IntervalAggregate,
    right: IntervalAggregate,
    entries: &[IndexedInterval],
) -> IntervalAggregate {
    let representative_entry = match (left.representative_entry, right.representative_entry) {
        (usize::MAX, right) => right,
        (left, usize::MAX) => left,
        (left, right) => {
            if entries[left]._order >= entries[right]._order {
                left
            } else {
                right
            }
        }
    };
    IntervalAggregate {
        max_end: left.max_end.max(right.max_end),
        representative_entry,
    }
}

#[derive(Debug, Default)]
struct LaneIndex {
    entries: Vec<IndexedInterval>,
    prefix_max_end: Vec<f64>,
    monotonic_summary: bool,
    aggregate_tree: Option<IntervalAggregateTree>,
}

impl LaneIndex {
    fn new(mut entries: Vec<IndexedInterval>) -> Self {
        entries.sort_by(|left, right| total_cmp(left.start, right.start));
        let mut maximum = f64::NEG_INFINITY;
        let prefix_max_end = entries
            .iter()
            .map(|entry| {
                maximum = maximum.max(entry.end);
                maximum
            })
            .collect();
        let monotonic_summary = entries
            .windows(2)
            .all(|pair| pair[0].end <= pair[1].end && pair[0]._order <= pair[1]._order);
        let aggregate_tree = (!monotonic_summary).then(|| IntervalAggregateTree::new(&entries));
        Self {
            entries,
            prefix_max_end,
            monotonic_summary,
            aggregate_tree,
        }
    }

    fn query_bounds(&self, range: DomainRange) -> Range<usize> {
        let right = self
            .entries
            .partition_point(|entry| entry.start <= range.end);
        let left = self.prefix_max_end[..right].partition_point(|end| *end < range.start);
        left..right
    }

    fn query_entries(&self, range: DomainRange) -> impl Iterator<Item = (usize, &IndexedInterval)> {
        let bounds = self.query_bounds(range);
        self.entries[bounds.clone()]
            .iter()
            .enumerate()
            .map(move |(offset, entry)| (bounds.start + offset, entry))
            .filter(move |(_, entry)| entry.end >= range.start)
    }

    fn query(&self, range: DomainRange) -> impl Iterator<Item = &usize> {
        self.query_entries(range).map(|(_, entry)| &entry.id)
    }

    fn push_sequence(&mut self, id: usize, range: DomainRange) {
        debug_assert!(
            self.entries
                .last()
                .is_none_or(|entry| entry.start <= range.start)
        );
        let maximum = self
            .prefix_max_end
            .last()
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
            .max(range.end);
        let order = self.entries.len();
        self.entries.push(IndexedInterval {
            id,
            start: range.start,
            end: range.end,
            _order: order,
        });
        self.prefix_max_end.push(maximum);
        if self.monotonic_summary {
            let previous = self.entries.get(order.wrapping_sub(1));
            self.monotonic_summary = previous
                .is_none_or(|previous| previous.end <= range.end && previous._order <= order);
            if !self.monotonic_summary {
                self.aggregate_tree = Some(IntervalAggregateTree::new(&self.entries));
            }
        } else if let Some(tree) = &mut self.aggregate_tree {
            tree.update(order, &self.entries);
        }
    }

    fn can_replace_last(&self, id: usize, range: DomainRange) -> bool {
        let Some(last) = self.entries.last() else {
            return false;
        };
        last.id == id
            && self
                .entries
                .get(self.entries.len().saturating_sub(2))
                .is_none_or(|previous| previous.start <= range.start)
    }

    fn replace_last(&mut self, id: usize, range: DomainRange) {
        debug_assert!(self.can_replace_last(id, range));
        let index = self.entries.len() - 1;
        let order = self.entries[index]._order;
        self.entries[index] = IndexedInterval {
            id,
            start: range.start,
            end: range.end,
            _order: order,
        };
        let prior_maximum = index
            .checked_sub(1)
            .and_then(|index| self.prefix_max_end.get(index))
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        self.prefix_max_end[index] = prior_maximum.max(range.end);
        if self.monotonic_summary {
            let previous = self.entries.get(index.wrapping_sub(1));
            self.monotonic_summary = previous
                .is_none_or(|previous| previous.end <= range.end && previous._order <= order);
            if !self.monotonic_summary {
                self.aggregate_tree = Some(IntervalAggregateTree::new(&self.entries));
            }
        } else if let Some(tree) = &mut self.aggregate_tree {
            tree.update(index, &self.entries);
        }
    }

    fn can_push(&self, range: DomainRange) -> bool {
        self.entries
            .last()
            .is_none_or(|entry| entry.start <= range.start)
    }

    fn summarize_range(&self, range: Range<usize>, work: &mut RenderWork) -> IntervalAggregate {
        debug_assert!(range.start < range.end);
        work.aggregate_queries = work.aggregate_queries.saturating_add(1);
        if self.monotonic_summary {
            let last = range.end - 1;
            return IntervalAggregate {
                max_end: self.entries[last].end,
                representative_entry: last,
            };
        }
        self.aggregate_tree
            .as_ref()
            .expect("non-monotonic lane has an aggregate tree")
            .query(range, &self.entries, work)
    }

    #[cfg(test)]
    fn hit(&self, value: f64) -> Option<&usize> {
        let right = self.entries.partition_point(|entry| entry.start <= value);
        let left = self.prefix_max_end[..right].partition_point(|end| *end < value);
        self.entries[left..right]
            .iter()
            .filter(|entry| entry.end >= value)
            .max_by_key(|entry| entry._order)
            .map(|entry| &entry.id)
    }
}

#[derive(Debug)]
pub(crate) struct TimelineGeometry {
    pub(crate) axis: AxisId,
    pub(crate) domain: DomainRange,
    pub(crate) cells: Vec<GeometryCell>,
    lanes: [LaneIndex; 3],
    cell_indices: Vec<Option<usize>>,
    source_len: usize,
    source_tail: Option<TimelineSpan>,
    source_tail_cell_index: Option<usize>,
    duration_busy: Option<BusyTimelineSet>,
    duration_fallback_points: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TimedGeometryUpdate {
    pub(crate) inspected_spans: usize,
    pub(crate) updated_cells: usize,
    pub(crate) appended_cells: usize,
    pub(crate) lane_entries_touched: usize,
}

impl TimelineGeometry {
    pub(crate) fn build(axis: AxisId, spans: impl IntoIterator<Item = TimelineSpan>) -> Self {
        let mut spans = spans.into_iter().collect::<Vec<_>>();
        spans.sort_by_key(|span| span.sequence);
        let source_len = spans.len();
        let source_tail = spans.last().cloned();
        let raw = spans
            .iter()
            .map(|span| {
                let started = span
                    .started
                    .clone()
                    .filter(|point| point.wall_ms.is_finite());
                let duration = span
                    .duration_ms
                    .filter(|value| value.is_finite() && *value >= 0.0);
                let completed = span
                    .completed
                    .clone()
                    .filter(|point| point.wall_ms.is_finite());
                (started, completed, duration)
            })
            .collect::<Vec<_>>();

        let busy = (axis.mode == TimelineMode::Duration).then(|| {
            BusyTimelineSet::new(raw.iter().filter_map(|(start, end, duration)| {
                if duration.unwrap_or_default() <= 0.0 {
                    return None;
                }
                let start = start.as_ref()?;
                let end = end.as_ref()?;
                (start.clock_id == end.clock_id)
                    .then(|| (start.clock_id.clone(), start.monotonic_ns, end.monotonic_ns))
            }))
        });

        let mut cells = Vec::with_capacity(spans.len());
        let mut source_tail_cell_index = None;
        // A reopened session can contain completed work from an older monotonic clock while the
        // current process has only emitted instantaneous or unfinished records. Those clock
        // domains are intentionally absent from `BusyTimelineSet` because they have no positive
        // interval to compress. Give every such record a stable point slot after the measured busy
        // domain instead of dropping it until a later completion happens to create an interval.
        let mut duration_fallback_points = 0_usize;
        for (position, span) in spans.into_iter().enumerate() {
            let (raw_start, raw_end, raw_duration) = &raw[position];
            let range = match axis.mode {
                TimelineMode::Sequence => DomainRange::new(position as f64, position as f64 + 1.0),
                TimelineMode::Actual => {
                    let Some(start) = raw_start.as_ref() else {
                        continue;
                    };
                    DomainRange::new(
                        start.wall_ms,
                        raw_end.as_ref().map_or(start.wall_ms, |end| end.wall_ms),
                    )
                }
                TimelineMode::Duration if busy.as_ref().is_none_or(|busy| busy.total_ns == 0) => {
                    // A duration projection has no natural scale when every
                    // item is instantaneous (or still lacks a completion).
                    // Keep those facts visible as stable sequence-positioned
                    // points instead of producing an empty swimlane.
                    let point = position as f64 + 0.5;
                    DomainRange::new(point, point)
                }
                TimelineMode::Duration => {
                    let Some(start) = raw_start.as_ref() else {
                        continue;
                    };
                    if raw_end.as_ref().is_some_and(|end| {
                        end.clock_id != start.clock_id || end.monotonic_ns < start.monotonic_ns
                    }) {
                        continue;
                    }
                    let busy = busy.as_ref().expect("duration mode has a busy timeline");
                    if let Some(projected_start) = busy.compressed(start) {
                        let projected_end = raw_end
                            .as_ref()
                            .and_then(|end| busy.compressed(end))
                            .unwrap_or(projected_start + raw_duration.unwrap_or_default());
                        DomainRange::new(projected_start, projected_end)
                    } else {
                        let point = busy.total_ms() + duration_fallback_points as f64 + 0.5;
                        duration_fallback_points = duration_fallback_points.saturating_add(1);
                        DomainRange::new(point, point)
                    }
                }
            };
            let nested = span.nested.as_ref().and_then(|(nested_start, nested_end)| {
                if !nested_start.wall_ms.is_finite() || !nested_end.wall_ms.is_finite() {
                    return None;
                }
                let nested = match axis.mode {
                    TimelineMode::Sequence => {
                        let raw_start = raw_start.as_ref()?;
                        let raw_end = raw_end.as_ref()?;
                        let (nested_offset, nested_width, raw_width) =
                            relative_timing(raw_start, raw_end, nested_start, nested_end)?;
                        let scale = range.width() / raw_width;
                        DomainRange::new(
                            range.start + nested_offset * scale,
                            range.start + (nested_offset + nested_width) * scale,
                        )
                    }
                    TimelineMode::Actual => {
                        DomainRange::new(nested_start.wall_ms, nested_end.wall_ms)
                    }
                    TimelineMode::Duration => {
                        let busy = busy.as_ref()?;
                        DomainRange::new(
                            busy.compressed(nested_start)?,
                            busy.compressed(nested_end)?,
                        )
                    }
                };
                let clipped_start = nested.start.max(range.start);
                let clipped_end = nested.end.min(range.end);
                if clipped_end <= clipped_start {
                    return None;
                }
                let clipped = DomainRange::new(clipped_start, clipped_end);
                (clipped.width() > 0.0).then_some(clipped)
            });
            cells.push(GeometryCell {
                id: span.id,
                lane: span.lane,
                range,
                nested,
            });
            if position + 1 == source_len {
                source_tail_cell_index = Some(cells.len() - 1);
            }
        }

        let domain = match axis.mode {
            TimelineMode::Sequence => DomainRange::new(0.0, cells.len().max(1) as f64),
            TimelineMode::Duration => DomainRange::new(
                0.0,
                if busy.as_ref().is_none_or(|busy| busy.total_ns == 0) {
                    cells.len().max(1) as f64
                } else {
                    busy.as_ref().map_or(0.0, BusyTimelineSet::total_ms)
                        + duration_fallback_points as f64
                }
                .max(1.0),
            ),
            TimelineMode::Actual => {
                let start = cells
                    .iter()
                    .map(|cell| cell.range.start)
                    .reduce(f64::min)
                    .unwrap_or_default();
                let end = cells
                    .iter()
                    .map(|cell| cell.range.end)
                    .reduce(f64::max)
                    .unwrap_or(start + 1.0);
                DomainRange::new(start, end.max(start + f64::EPSILON))
            }
        };

        let lanes = TIMELINE_LANES.map(|lane| {
            LaneIndex::new(
                cells
                    .iter()
                    .filter(|cell| cell.lane == lane)
                    .enumerate()
                    .map(|(order, cell)| IndexedInterval {
                        id: cell.id,
                        start: cell.range.start,
                        end: cell.range.end,
                        _order: order,
                    })
                    .collect(),
            )
        });
        let mut cell_indices = vec![None; cells.iter().map(|cell| cell.id + 1).max().unwrap_or(0)];
        for (index, cell) in cells.iter().enumerate() {
            cell_indices[cell.id] = Some(index);
        }
        Self {
            axis,
            domain,
            cells,
            lanes,
            cell_indices,
            source_len,
            source_tail,
            source_tail_cell_index,
            duration_busy: busy,
            duration_fallback_points,
        }
    }

    pub(crate) fn query(
        &self,
        lane: TimelineLane,
        range: DomainRange,
    ) -> impl Iterator<Item = &usize> {
        self.lanes[lane_index(lane)].query(range)
    }

    /// Returns the last-painted item at a point in a lane.
    ///
    /// Hit testing and painting intentionally share this geometry index, so a
    /// dense or overlapping lane cannot drift from what the user sees.
    #[cfg(test)]
    pub(crate) fn hit_test(&self, lane: TimelineLane, value: f64) -> Option<&usize> {
        self.lanes[lane_index(lane)].hit(value)
    }

    pub(crate) fn selection(&self, range: AxisRange) -> SelectionResult {
        if range.axis != self.axis {
            return SelectionResult {
                range: AxisRange {
                    axis: self.axis,
                    range: self.domain,
                },
                items: HashSet::new(),
            };
        }
        let range = AxisRange {
            axis: self.axis,
            range: range.range.clamp_to(self.domain),
        };
        let items = TIMELINE_LANES
            .into_iter()
            .flat_map(|lane| self.query(lane, range.range))
            .copied()
            .collect();
        SelectionResult { range, items }
    }

    pub(crate) fn range_for(&self, id: &usize) -> Option<AxisRange> {
        let cell = self
            .cell_indices
            .get(*id)
            .and_then(|index| *index)
            .and_then(|index| self.cells.get(index))?;
        Some(AxisRange {
            axis: self.axis,
            range: cell.range,
        })
    }

    /// Applies stable-ID changes without rebuilding the interval indices used by Sequence mode.
    /// Existing sequence cells keep their outer range; only nested timing can change. New cells
    /// must form an append-only suffix, which lets every lane index grow in O(1) per appended item.
    pub(crate) fn update_sequence(
        &mut self,
        axis: AxisId,
        total_len: usize,
        changes: impl IntoIterator<Item = (usize, TimelineSpan)>,
    ) -> bool {
        if self.axis.mode != TimelineMode::Sequence
            || axis.mode != TimelineMode::Sequence
            || self.axis.document_generation != axis.document_generation
        {
            return false;
        }
        let mut changes = changes.into_iter().collect::<Vec<_>>();
        changes.sort_by_key(|(index, _)| *index);
        changes.dedup_by(|left, right| left.0 == right.0);
        for (index, span) in changes {
            let cell = sequence_cell(index, span);
            if index < self.cells.len() {
                let previous = &self.cells[index];
                if previous.id != cell.id
                    || previous.lane != cell.lane
                    || previous.range != cell.range
                {
                    return false;
                }
                self.cells[index] = cell;
            } else if index == self.cells.len() {
                if self.cell_indices.get(cell.id).is_some_and(Option::is_some) {
                    return false;
                }
                self.lanes[lane_index(cell.lane)].push_sequence(cell.id, cell.range);
                if cell.id >= self.cell_indices.len() {
                    self.cell_indices.resize(cell.id + 1, None);
                }
                self.cell_indices[cell.id] = Some(index);
                self.cells.push(cell);
            } else {
                return false;
            }
        }
        if self.cells.len() != total_len {
            return false;
        }
        self.axis = axis;
        self.domain = DomainRange::new(0.0, self.cells.len().max(1) as f64);
        true
    }

    /// Incrementally applies the common timed-axis delta: the current tail gains timing, and/or
    /// new source records form a contiguous suffix. Any edit that could shift an earlier busy-time
    /// coordinate is rejected, allowing the caller to fall back to `build` without risking drift.
    pub(crate) fn update_timed(
        &mut self,
        axis: AxisId,
        total_len: usize,
        changes: impl IntoIterator<Item = (usize, TimelineSpan)>,
    ) -> Option<TimedGeometryUpdate> {
        if !matches!(
            self.axis.mode,
            TimelineMode::Duration | TimelineMode::Actual
        ) || axis.mode != self.axis.mode
            || axis.document_generation != self.axis.document_generation
            || total_len < self.source_len
        {
            return None;
        }
        let mut changes = changes.into_iter().collect::<Vec<_>>();
        changes.sort_by_key(|(index, _)| *index);
        changes.dedup_by(|left, right| left.0 == right.0);
        let inspected_spans = changes.len();
        let old_len = self.source_len;
        let tail_index = old_len.checked_sub(1);
        if changes
            .iter()
            .any(|(index, _)| *index < old_len && Some(*index) != tail_index)
        {
            return None;
        }
        let appended = changes
            .iter()
            .filter(|(index, _)| *index >= old_len)
            .map(|(index, _)| *index)
            .collect::<Vec<_>>();
        if appended != (old_len..total_len).collect::<Vec<_>>()
            || changes.iter().any(|(index, _)| *index >= total_len)
        {
            return None;
        }

        if let Some((_, changed_tail)) =
            changes.iter().find(|(index, _)| Some(*index) == tail_index)
        {
            let old_tail = self.source_tail.as_ref()?;
            if old_tail.id != changed_tail.id
                || old_tail.lane != changed_tail.lane
                || old_tail.sequence != changed_tail.sequence
                || old_tail.started != changed_tail.started
                || !option_non_decreasing(old_tail.duration_ms, changed_tail.duration_ms)
                || !point_non_decreasing(
                    old_tail.completed.as_ref(),
                    changed_tail.completed.as_ref(),
                )
            {
                return None;
            }
        }
        let mut prior_sequence = self.source_tail.as_ref().map(|span| span.sequence);
        for (_, span) in changes.iter().filter(|(index, _)| *index >= old_len) {
            if prior_sequence.is_some_and(|prior| span.sequence <= prior) {
                return None;
            }
            prior_sequence = Some(span.sequence);
        }

        let busy_checkpoint = (self.axis.mode == TimelineMode::Duration)
            .then(|| {
                self.duration_busy
                    .as_ref()
                    .map(BusyTimelineSet::checkpoint_tail)
            })
            .flatten();
        if self.axis.mode == TimelineMode::Duration {
            if self.duration_fallback_points > 0 {
                return None;
            }
            let busy = self.duration_busy.as_mut()?;
            if busy.total_ns == 0 {
                if changes
                    .iter()
                    .any(|(_, span)| positive_busy_interval(span).is_some())
                {
                    return None;
                }
            } else {
                for (index, span) in &changes {
                    let new_interval = positive_busy_interval(span);
                    let old_interval = (Some(*index) == tail_index)
                        .then(|| {
                            positive_busy_interval(
                                self.source_tail
                                    .as_ref()
                                    .expect("non-empty geometry retained its source tail"),
                            )
                        })
                        .flatten();
                    match (old_interval, new_interval) {
                        (Some((old_clock, old_start, old_end)), Some((clock, start, end))) => {
                            if old_clock != clock || old_start != start || end < old_end {
                                self.rollback_duration_busy(busy_checkpoint);
                                return None;
                            }
                            if end > old_end && !busy.append_tail_interval(clock, start, end) {
                                self.rollback_duration_busy(busy_checkpoint);
                                return None;
                            }
                        }
                        (Some(_), None) => {
                            self.rollback_duration_busy(busy_checkpoint);
                            return None;
                        }
                        (None, Some((clock, start, end))) => {
                            if !busy.append_tail_interval(clock, start, end) {
                                self.rollback_duration_busy(busy_checkpoint);
                                return None;
                            }
                        }
                        (None, None) => {}
                    }
                }
            }
        }

        #[derive(Clone)]
        enum CellPlan {
            Replace(usize, GeometryCell),
            Append(GeometryCell),
            Omitted,
        }

        let mut staged = Vec::<(usize, TimelineSpan, CellPlan)>::with_capacity(changes.len());
        let mut lane_tails = TIMELINE_LANES.map(|lane| {
            self.lanes[lane_index(lane)]
                .entries
                .last()
                .map(|entry| (entry.id, entry.start))
        });
        let mut staged_ids = HashSet::new();
        for (position, span) in changes {
            let cell = match self.axis.mode {
                TimelineMode::Actual => actual_cell(&span),
                TimelineMode::Duration => {
                    let busy = self
                        .duration_busy
                        .as_ref()
                        .expect("duration geometry always owns a busy timeline");
                    if busy.total_ns == 0 {
                        Some(point_duration_cell(position, &span))
                    } else {
                        if span.started.as_ref().is_some_and(|start| {
                            start.wall_ms.is_finite() && busy.compressed(start).is_none()
                        }) {
                            self.rollback_duration_busy(busy_checkpoint);
                            return None;
                        }
                        duration_cell(&span, busy)
                    }
                }
                TimelineMode::Sequence => unreachable!(),
            };
            let old_cell_index = (Some(position) == tail_index)
                .then_some(self.source_tail_cell_index)
                .flatten();
            let plan = match (old_cell_index, cell) {
                (Some(index), Some(cell)) => {
                    let Some(old) = self.cells.get(index) else {
                        self.rollback_duration_busy(busy_checkpoint);
                        return None;
                    };
                    if old.id != cell.id
                        || old.lane != cell.lane
                        || cell.range.start != old.range.start
                        || cell.range.end < old.range.end
                        || !self.lanes[lane_index(cell.lane)].can_replace_last(cell.id, cell.range)
                    {
                        self.rollback_duration_busy(busy_checkpoint);
                        return None;
                    }
                    lane_tails[lane_index(cell.lane)] = Some((cell.id, cell.range.start));
                    CellPlan::Replace(index, cell)
                }
                (Some(_), None) => {
                    self.rollback_duration_busy(busy_checkpoint);
                    return None;
                }
                (None, Some(cell)) => {
                    if self.cell_indices.get(cell.id).is_some_and(Option::is_some)
                        || !staged_ids.insert(cell.id)
                        || lane_tails[lane_index(cell.lane)]
                            .is_some_and(|(_, start)| start > cell.range.start)
                    {
                        self.rollback_duration_busy(busy_checkpoint);
                        return None;
                    }
                    lane_tails[lane_index(cell.lane)] = Some((cell.id, cell.range.start));
                    CellPlan::Append(cell)
                }
                (None, None) => CellPlan::Omitted,
            };
            staged.push((position, span, plan));
        }

        let previous_cell_count = self.cells.len();
        let mut updated_cells = 0_usize;
        let mut final_tail_cell_index = if total_len == old_len {
            self.source_tail_cell_index
        } else {
            None
        };
        for (position, span, plan) in staged {
            let cell_index = match plan {
                CellPlan::Replace(index, cell) => {
                    self.lanes[lane_index(cell.lane)].replace_last(cell.id, cell.range);
                    self.cells[index] = cell;
                    updated_cells = updated_cells.saturating_add(1);
                    Some(index)
                }
                CellPlan::Append(cell) => {
                    debug_assert!(self.lanes[lane_index(cell.lane)].can_push(cell.range));
                    let index = self.cells.len();
                    self.lanes[lane_index(cell.lane)].push_sequence(cell.id, cell.range);
                    if cell.id >= self.cell_indices.len() {
                        self.cell_indices.resize(cell.id + 1, None);
                    }
                    self.cell_indices[cell.id] = Some(index);
                    self.cells.push(cell);
                    Some(index)
                }
                CellPlan::Omitted => None,
            };
            if position + 1 == total_len {
                self.source_tail = Some(span);
                final_tail_cell_index = cell_index;
            }
        }
        self.source_len = total_len;
        self.source_tail_cell_index = final_tail_cell_index;
        self.axis = axis;
        match self.axis.mode {
            TimelineMode::Duration => {
                let busy = self
                    .duration_busy
                    .as_ref()
                    .expect("duration geometry always owns a busy timeline");
                self.domain = DomainRange::new(
                    0.0,
                    if busy.total_ns == 0 {
                        total_len.max(1) as f64
                    } else {
                        busy.total_ms().max(1.0)
                    },
                );
            }
            TimelineMode::Actual => {
                let mut start = if previous_cell_count == 0 {
                    f64::INFINITY
                } else {
                    self.domain.start
                };
                let mut end = if previous_cell_count == 0 {
                    f64::NEG_INFINITY
                } else {
                    self.domain.end
                };
                for cell in &self.cells[previous_cell_count.saturating_sub(updated_cells)..] {
                    start = start.min(cell.range.start);
                    end = end.max(cell.range.end);
                }
                if !start.is_finite() {
                    start = 0.0;
                }
                if !end.is_finite() {
                    end = start + 1.0;
                }
                self.domain = DomainRange::new(start, end.max(start + f64::EPSILON));
            }
            TimelineMode::Sequence => unreachable!(),
        }
        let appended_cells = self.cells.len().saturating_sub(previous_cell_count);
        Some(TimedGeometryUpdate {
            inspected_spans,
            updated_cells,
            appended_cells,
            lane_entries_touched: updated_cells.saturating_add(appended_cells),
        })
    }

    fn rollback_duration_busy(&mut self, checkpoint: Option<BusyTailCheckpoint>) {
        if let (Some(busy), Some(checkpoint)) = (&mut self.duration_busy, checkpoint) {
            busy.rollback_tail(checkpoint);
        }
    }

    pub(crate) fn render_model(
        &self,
        viewport: DomainRange,
        width_px: f64,
        primitive_limit: usize,
    ) -> Vec<RenderCell> {
        self.render_model_with_work(viewport, width_px, primitive_limit)
            .0
    }

    fn render_model_with_work(
        &self,
        viewport: DomainRange,
        width_px: f64,
        primitive_limit: usize,
    ) -> (Vec<RenderCell>, RenderWork) {
        let mut work = RenderWork::default();
        if width_px <= 0.0 || primitive_limit == 0 {
            return (Vec::new(), work);
        }
        let viewport = viewport.clamp_to(self.domain);
        // The exact path inspects at most `primitive_limit + 1` intervals. Once LOD is known to
        // be necessary we discard this bounded prefix and summarize lane-index ranges below.
        let mut visible = Vec::with_capacity(primitive_limit.saturating_add(1));
        'lanes: for lane in TIMELINE_LANES {
            for (_, entry) in self.lanes[lane_index(lane)].query_entries(viewport) {
                work.scanned_intervals = work.scanned_intervals.saturating_add(1);
                if let Some(cell) = self
                    .cell_indices
                    .get(entry.id)
                    .and_then(|index| *index)
                    .and_then(|index| self.cells.get(index))
                {
                    visible.push(cell);
                    if visible.len() > primitive_limit {
                        break 'lanes;
                    }
                }
            }
        }
        visible.sort_by_key(|cell| self.cell_indices.get(cell.id).and_then(|index| *index));
        if visible.len() <= primitive_limit {
            let cells = visible
                .into_iter()
                .map(|cell| RenderCell::from_cell(cell, viewport, width_px))
                .collect::<Vec<_>>();
            work.stored_identity_words = cells.len();
            return (cells, work);
        }

        let bins_per_lane = (primitive_limit / TIMELINE_LANES.len()).max(1);
        let mut rendered = Vec::with_capacity(bins_per_lane * TIMELINE_LANES.len());
        for lane in TIMELINE_LANES {
            let lane_index = &self.lanes[lane_index(lane)];
            let bounds = lane_index.query_bounds(viewport);
            let started_inside = lane_index.entries[..bounds.end]
                .partition_point(|entry| entry.start < viewport.start)
                .max(bounds.start);
            let mut prefix = BinAccumulator::default();
            for (entry_index, entry) in lane_index.entries[bounds.start..started_inside]
                .iter()
                .enumerate()
                .map(|(offset, entry)| (bounds.start + offset, entry))
                .filter(|(_, entry)| entry.end >= viewport.start)
            {
                work.scanned_intervals = work.scanned_intervals.saturating_add(1);
                prefix.add_entry(entry_index, entry, lane_index);
            }

            for bin in 0..bins_per_lane {
                let start = viewport.start + viewport.width() * (bin as f64 / bins_per_lane as f64);
                let end =
                    viewport.start + viewport.width() * ((bin + 1) as f64 / bins_per_lane as f64);
                let left = if bin == 0 {
                    started_inside
                } else {
                    lane_index.entries[..bounds.end]
                        .partition_point(|entry| entry.start < start)
                        .max(started_inside)
                };
                let right = if bin + 1 == bins_per_lane {
                    bounds.end
                } else {
                    lane_index.entries[..bounds.end].partition_point(|entry| entry.start < end)
                }
                .max(left);
                let mut accumulator = if bin == 0 {
                    std::mem::take(&mut prefix)
                } else {
                    BinAccumulator::default()
                };
                if left < right {
                    accumulator.add_range(left..right, lane_index, &mut work);
                }
                if let Some(mut cell) =
                    accumulator.finish(lane, viewport, width_px, bin, bins_per_lane)
                {
                    if cell.ids.len() == 1 {
                        cell.nested = cell
                            .ids
                            .last()
                            .and_then(|id| self.cell_indices.get(*id))
                            .and_then(|index| *index)
                            .and_then(|index| self.cells.get(index))
                            .and_then(|geometry| geometry.nested)
                            .map(|range| {
                                let project = |value: f64| {
                                    ((value - viewport.start) / viewport.width().max(f64::EPSILON)
                                        * width_px)
                                        .clamp(0.0, width_px)
                                };
                                (project(range.start), project(range.end))
                            });
                    }
                    rendered.push(cell);
                }
            }
        }
        work.stored_identity_words = rendered
            .iter()
            .map(|cell| cell.ids.stored_word_count())
            .sum();
        (rendered, work)
    }

    /// Enumerates the exact records represented by a rendered primitive. Cluster identities are
    /// lane-index ranges, so the primitive itself remains O(1) regardless of member count.
    pub(crate) fn render_members<'a>(&'a self, cell: &'a RenderCell) -> RenderMemberIter<'a> {
        let inner = match &cell.ids.members {
            RenderMemberLocator::Explicit(ids) => RenderMemberIterInner::Explicit(ids.iter()),
            RenderMemberLocator::LaneEntries {
                entries, viewport, ..
            } => RenderMemberIterInner::LaneEntries {
                entries: self.lanes[lane_index(cell.lane)].entries[entries.clone()].iter(),
                viewport: *viewport,
            },
        };
        RenderMemberIter { inner }
    }

    /// Resolves a stable record id to its rendered primitive without materializing a per-record
    /// lookup. Exact projections are source-ordered; LOD projections are lane/bin ordered, so both
    /// paths use binary search and retain O(P) projection memory for P primitives.
    pub(crate) fn render_cell_for_record(&self, cells: &[RenderCell], id: usize) -> Option<usize> {
        let geometry_index = self.cell_indices.get(id).and_then(|index| *index)?;
        let geometry_cell = self.cells.get(geometry_index)?;
        let first = cells.first()?;
        if let Some((_, bins_per_lane, viewport)) = first.ids.cluster_key() {
            if geometry_cell.range.start > viewport.end || geometry_cell.range.end < viewport.start
            {
                return None;
            }
            let fraction = ((geometry_cell.range.start - viewport.start)
                / viewport.width().max(f64::EPSILON))
            .clamp(0.0, 1.0);
            let bin = ((fraction * bins_per_lane as f64).floor() as usize)
                .min(bins_per_lane.saturating_sub(1));
            let key = (lane_index(geometry_cell.lane), bin);
            cells
                .binary_search_by_key(&key, |cell| {
                    let (bin, _, _) = cell
                        .ids
                        .cluster_key()
                        .expect("LOD projection contains only clustered identities");
                    (lane_index(cell.lane), bin)
                })
                .ok()
        } else {
            cells
                .binary_search_by_key(&geometry_index, |cell| {
                    cell.ids
                        .last()
                        .and_then(|id| self.cell_indices.get(*id))
                        .and_then(|index| *index)
                        .unwrap_or(usize::MAX)
                })
                .ok()
        }
    }
}

fn sequence_cell(position: usize, span: TimelineSpan) -> GeometryCell {
    let raw_start = span.started.filter(|point| point.wall_ms.is_finite());
    let raw_end = span.completed.filter(|point| point.wall_ms.is_finite());
    let range = DomainRange::new(position as f64, position as f64 + 1.0);
    let nested = span.nested.and_then(|(nested_start, nested_end)| {
        if !nested_start.wall_ms.is_finite() || !nested_end.wall_ms.is_finite() {
            return None;
        }
        let raw_start = raw_start.as_ref()?;
        let raw_end = raw_end.as_ref()?;
        let (nested_offset, nested_width, raw_width) =
            relative_timing(raw_start, raw_end, &nested_start, &nested_end)?;
        let scale = range.width() / raw_width;
        let nested = DomainRange::new(
            range.start + nested_offset * scale,
            range.start + (nested_offset + nested_width) * scale,
        );
        let clipped_start = nested.start.max(range.start);
        let clipped_end = nested.end.min(range.end);
        (clipped_end > clipped_start).then(|| DomainRange::new(clipped_start, clipped_end))
    });
    GeometryCell {
        id: span.id,
        lane: span.lane,
        range,
        nested,
    }
}

fn option_non_decreasing(previous: Option<f64>, current: Option<f64>) -> bool {
    match (previous, current) {
        (Some(previous), Some(current)) => {
            previous.is_finite() && current.is_finite() && current >= previous
        }
        (Some(_), None) => false,
        _ => true,
    }
}

fn point_non_decreasing(previous: Option<&TimelinePoint>, current: Option<&TimelinePoint>) -> bool {
    match (previous, current) {
        (Some(previous), Some(current)) => {
            previous.clock_id == current.clock_id
                && current.monotonic_ns >= previous.monotonic_ns
                && current.wall_ms.is_finite()
                && previous.wall_ms.is_finite()
                && current.wall_ms >= previous.wall_ms
        }
        (Some(_), None) => false,
        _ => true,
    }
}

fn positive_busy_interval(span: &TimelineSpan) -> Option<(&str, u64, u64)> {
    (span
        .duration_ms
        .filter(|duration| *duration > 0.0)
        .is_some())
    .then_some(())?;
    let start = span.started.as_ref()?;
    let end = span.completed.as_ref()?;
    (start.wall_ms.is_finite()
        && end.wall_ms.is_finite()
        && start.clock_id == end.clock_id
        && end.monotonic_ns > start.monotonic_ns)
        .then_some((
            start.clock_id.as_str(),
            start.monotonic_ns,
            end.monotonic_ns,
        ))
}

fn actual_cell(span: &TimelineSpan) -> Option<GeometryCell> {
    let start = span
        .started
        .as_ref()
        .filter(|point| point.wall_ms.is_finite())?;
    let end = span
        .completed
        .as_ref()
        .filter(|point| point.wall_ms.is_finite());
    let range = DomainRange::new(start.wall_ms, end.map_or(start.wall_ms, |end| end.wall_ms));
    let nested = span.nested.as_ref().and_then(|(nested_start, nested_end)| {
        if !nested_start.wall_ms.is_finite() || !nested_end.wall_ms.is_finite() {
            return None;
        }
        clip_nested(
            range,
            DomainRange::new(nested_start.wall_ms, nested_end.wall_ms),
        )
    });
    Some(GeometryCell {
        id: span.id,
        lane: span.lane,
        range,
        nested,
    })
}

fn point_duration_cell(position: usize, span: &TimelineSpan) -> GeometryCell {
    GeometryCell {
        id: span.id,
        lane: span.lane,
        range: DomainRange::new(position as f64 + 0.5, position as f64 + 0.5),
        nested: None,
    }
}

fn duration_cell(span: &TimelineSpan, busy: &BusyTimelineSet) -> Option<GeometryCell> {
    let start = span
        .started
        .as_ref()
        .filter(|point| point.wall_ms.is_finite())?;
    let end = span
        .completed
        .as_ref()
        .filter(|point| point.wall_ms.is_finite());
    if end
        .is_some_and(|end| end.clock_id != start.clock_id || end.monotonic_ns < start.monotonic_ns)
    {
        return None;
    }
    let projected_start = busy.compressed(start)?;
    let projected_end = end
        .and_then(|end| busy.compressed(end))
        .unwrap_or(projected_start + span.duration_ms.unwrap_or_default());
    let range = DomainRange::new(projected_start, projected_end);
    let nested = span.nested.as_ref().and_then(|(nested_start, nested_end)| {
        if !nested_start.wall_ms.is_finite() || !nested_end.wall_ms.is_finite() {
            return None;
        }
        clip_nested(
            range,
            DomainRange::new(busy.compressed(nested_start)?, busy.compressed(nested_end)?),
        )
    });
    Some(GeometryCell {
        id: span.id,
        lane: span.lane,
        range,
        nested,
    })
}

fn clip_nested(parent: DomainRange, nested: DomainRange) -> Option<DomainRange> {
    let clipped = DomainRange::new(nested.start.max(parent.start), nested.end.min(parent.end));
    (clipped.end > clipped.start).then_some(clipped)
}

#[derive(Clone, Debug, PartialEq)]
enum RenderMemberLocator {
    Explicit(Vec<usize>),
    LaneEntries {
        entries: Range<usize>,
        viewport: DomainRange,
        bin: usize,
        bins_per_lane: usize,
    },
}

/// Compact identity for a rendered primitive. `len()` is the semantic member count; exact member
/// mapping is provided by `TimelineGeometry::render_members`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderIdentity {
    representative: usize,
    member_count: usize,
    members: RenderMemberLocator,
}

impl RenderIdentity {
    pub(crate) fn explicit(ids: Vec<usize>) -> Self {
        let representative = ids.last().copied().unwrap_or_default();
        Self {
            representative,
            member_count: ids.len(),
            members: RenderMemberLocator::Explicit(ids),
        }
    }

    fn cluster(
        representative: usize,
        member_count: usize,
        entries: Range<usize>,
        viewport: DomainRange,
        bin: usize,
        bins_per_lane: usize,
    ) -> Self {
        Self {
            representative,
            member_count,
            members: RenderMemberLocator::LaneEntries {
                entries,
                viewport,
                bin,
                bins_per_lane,
            },
        }
    }

    pub(crate) fn last(&self) -> Option<&usize> {
        (self.member_count > 0).then_some(&self.representative)
    }

    pub(crate) fn len(&self) -> usize {
        self.member_count
    }

    fn stored_word_count(&self) -> usize {
        match &self.members {
            RenderMemberLocator::Explicit(ids) => ids.len(),
            RenderMemberLocator::LaneEntries { .. } => 8,
        }
    }

    fn cluster_key(&self) -> Option<(usize, usize, DomainRange)> {
        match self.members {
            RenderMemberLocator::Explicit(_) => None,
            RenderMemberLocator::LaneEntries {
                viewport,
                bin,
                bins_per_lane,
                ..
            } => Some((bin, bins_per_lane, viewport)),
        }
    }
}

impl FromIterator<usize> for RenderIdentity {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self::explicit(iter.into_iter().collect())
    }
}

pub(crate) struct RenderMemberIter<'a> {
    inner: RenderMemberIterInner<'a>,
}

enum RenderMemberIterInner<'a> {
    Explicit(std::slice::Iter<'a, usize>),
    LaneEntries {
        entries: std::slice::Iter<'a, IndexedInterval>,
        viewport: DomainRange,
    },
}

impl<'a> Iterator for RenderMemberIter<'a> {
    type Item = &'a usize;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            RenderMemberIterInner::Explicit(ids) => ids.next(),
            RenderMemberIterInner::LaneEntries { entries, viewport } => entries
                .find(|entry| entry.start <= viewport.end && entry.end >= viewport.start)
                .map(|entry| &entry.id),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderCell {
    pub(crate) ids: RenderIdentity,
    pub(crate) lane: TimelineLane,
    pub(crate) start_px: f64,
    pub(crate) end_px: f64,
    pub(crate) nested: Option<(f64, f64)>,
    pub(crate) clustered: bool,
}

impl RenderCell {
    fn from_cell(cell: &GeometryCell, viewport: DomainRange, width_px: f64) -> Self {
        let project = |value: f64| {
            ((value - viewport.start) / viewport.width().max(f64::EPSILON) * width_px)
                .clamp(0.0, width_px)
        };
        Self {
            ids: RenderIdentity::explicit(vec![cell.id]),
            lane: cell.lane,
            start_px: project(cell.range.start),
            end_px: project(cell.range.end),
            nested: cell
                .nested
                .map(|range| (project(range.start), project(range.end))),
            clustered: false,
        }
    }
}

#[derive(Default)]
struct BinAccumulator {
    first_entry: Option<usize>,
    last_entry: usize,
    representative_entry: usize,
    representative_id: usize,
    member_count: usize,
    minimum_start: f64,
    maximum_end: f64,
}

impl BinAccumulator {
    fn add_entry(&mut self, index: usize, entry: &IndexedInterval, lane: &LaneIndex) {
        if self.first_entry.is_none() {
            self.first_entry = Some(index);
            self.minimum_start = entry.start;
            self.maximum_end = entry.end;
            self.representative_entry = index;
            self.representative_id = entry.id;
        } else {
            self.minimum_start = self.minimum_start.min(entry.start);
            self.maximum_end = self.maximum_end.max(entry.end);
            if lane.entries[self.representative_entry]._order <= entry._order {
                self.representative_entry = index;
                self.representative_id = entry.id;
            }
        }
        self.last_entry = index + 1;
        self.member_count = self.member_count.saturating_add(1);
    }

    fn add_range(&mut self, range: Range<usize>, lane: &LaneIndex, work: &mut RenderWork) {
        let aggregate = lane.summarize_range(range.clone(), work);
        let first = &lane.entries[range.start];
        if self.first_entry.is_none() {
            self.first_entry = Some(range.start);
            self.minimum_start = first.start;
            self.maximum_end = aggregate.max_end;
            self.representative_entry = aggregate.representative_entry;
            self.representative_id = lane.entries[aggregate.representative_entry].id;
        } else {
            self.minimum_start = self.minimum_start.min(first.start);
            self.maximum_end = self.maximum_end.max(aggregate.max_end);
            if lane.entries[self.representative_entry]._order
                <= lane.entries[aggregate.representative_entry]._order
            {
                self.representative_entry = aggregate.representative_entry;
                self.representative_id = lane.entries[aggregate.representative_entry].id;
            }
        }
        self.last_entry = range.end;
        self.member_count = self.member_count.saturating_add(range.len());
    }

    fn finish(
        self,
        lane: TimelineLane,
        viewport: DomainRange,
        width_px: f64,
        bin: usize,
        bins_per_lane: usize,
    ) -> Option<RenderCell> {
        let first_entry = self.first_entry?;
        let project = |value: f64| {
            ((value - viewport.start) / viewport.width().max(f64::EPSILON) * width_px)
                .clamp(0.0, width_px)
        };
        Some(RenderCell {
            ids: RenderIdentity::cluster(
                self.representative_id,
                self.member_count,
                first_entry..self.last_entry,
                viewport,
                bin,
                bins_per_lane,
            ),
            lane,
            start_px: project(self.minimum_start),
            end_px: project(self.maximum_end),
            nested: None,
            clustered: self.member_count > 1,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SelectionResult {
    pub(crate) range: AxisRange,
    pub(crate) items: HashSet<usize>,
}

#[derive(Debug, Default)]
struct BusyTimelineSet {
    clocks: Vec<BusyClock>,
    clock_indices: HashMap<String, usize>,
    total_ns: u64,
}

#[derive(Debug)]
struct BusyClock {
    offset_ns: u64,
    timeline: BusyTimeline,
}

#[derive(Clone, Copy, Debug)]
struct BusyTailCheckpoint {
    clocks_len: usize,
    total_ns: u64,
    last_intervals_len: usize,
    last_prefix_len: usize,
    last_interval: Option<(u64, u64)>,
    last_prefix: Option<u64>,
}

impl BusyTimelineSet {
    fn new(intervals: impl IntoIterator<Item = (String, u64, u64)>) -> Self {
        let mut groups = Vec::<(String, Vec<(u64, u64)>)>::new();
        let mut group_indices = HashMap::<String, usize>::new();
        for (clock_id, start, end) in intervals {
            if end <= start {
                continue;
            }
            let index = *group_indices.entry(clock_id.clone()).or_insert_with(|| {
                let index = groups.len();
                groups.push((clock_id, Vec::new()));
                index
            });
            groups[index].1.push((start, end));
        }

        let mut clocks = Vec::with_capacity(groups.len());
        let mut clock_indices = HashMap::with_capacity(groups.len());
        let mut total_ns = 0_u64;
        for (clock_id, intervals) in groups {
            let timeline = BusyTimeline::new(intervals);
            let clock_total_ns = timeline.total_ns();
            clock_indices.insert(clock_id, clocks.len());
            clocks.push(BusyClock {
                offset_ns: total_ns,
                timeline,
            });
            total_ns = total_ns.saturating_add(clock_total_ns);
        }
        Self {
            clocks,
            clock_indices,
            total_ns,
        }
    }

    fn compressed(&self, point: &TimelinePoint) -> Option<f64> {
        let clock = self.clocks.get(*self.clock_indices.get(&point.clock_id)?)?;
        let compressed_ns = clock
            .offset_ns
            .saturating_add(clock.timeline.compressed_ns(point.monotonic_ns));
        Some(compressed_ns as f64 / 1_000_000.0)
    }

    fn total_ms(&self) -> f64 {
        self.total_ns as f64 / 1_000_000.0
    }

    fn checkpoint_tail(&self) -> BusyTailCheckpoint {
        let last = self.clocks.last();
        BusyTailCheckpoint {
            clocks_len: self.clocks.len(),
            total_ns: self.total_ns,
            last_intervals_len: last.map_or(0, |clock| clock.timeline.intervals.len()),
            last_prefix_len: last.map_or(0, |clock| clock.timeline.prefix_ns.len()),
            last_interval: last.and_then(|clock| clock.timeline.intervals.last().copied()),
            last_prefix: last.and_then(|clock| clock.timeline.prefix_ns.last().copied()),
        }
    }

    fn rollback_tail(&mut self, checkpoint: BusyTailCheckpoint) {
        self.clocks.truncate(checkpoint.clocks_len);
        self.clock_indices
            .retain(|_, index| *index < checkpoint.clocks_len);
        if let Some(last) = self.clocks.last_mut() {
            last.timeline
                .intervals
                .truncate(checkpoint.last_intervals_len);
            last.timeline.prefix_ns.truncate(checkpoint.last_prefix_len);
            if let (Some(target), Some(saved)) =
                (last.timeline.intervals.last_mut(), checkpoint.last_interval)
            {
                *target = saved;
            }
            if let (Some(target), Some(saved)) =
                (last.timeline.prefix_ns.last_mut(), checkpoint.last_prefix)
            {
                *target = saved;
            }
        }
        self.total_ns = checkpoint.total_ns;
    }

    /// Extends only the last monotonic clock. Extending an earlier clock would shift every later
    /// clock's compressed coordinates and therefore deliberately rejects incremental application.
    fn append_tail_interval(&mut self, clock_id: &str, start: u64, end: u64) -> bool {
        if end <= start {
            return true;
        }
        let clock_index = if let Some(index) = self.clock_indices.get(clock_id).copied() {
            if index + 1 != self.clocks.len() {
                return false;
            }
            index
        } else {
            let index = self.clocks.len();
            self.clock_indices.insert(clock_id.to_owned(), index);
            self.clocks.push(BusyClock {
                offset_ns: self.total_ns,
                timeline: BusyTimeline::default(),
            });
            index
        };
        let before = self.clocks[clock_index].timeline.total_ns();
        if !self.clocks[clock_index].timeline.append_tail(start, end) {
            return false;
        }
        let after = self.clocks[clock_index].timeline.total_ns();
        self.total_ns = self.total_ns.saturating_add(after.saturating_sub(before));
        true
    }
}

#[derive(Debug, Default)]
struct BusyTimeline {
    intervals: Vec<(u64, u64)>,
    prefix_ns: Vec<u64>,
}

impl BusyTimeline {
    fn new(mut intervals: Vec<(u64, u64)>) -> Self {
        intervals.retain(|(start, end)| end > start);
        intervals.sort_unstable_by_key(|(start, _)| *start);
        let mut merged = Vec::<(u64, u64)>::new();
        for (start, end) in intervals {
            if let Some((_, previous_end)) = merged.last_mut()
                && start <= *previous_end
            {
                *previous_end = (*previous_end).max(end);
            } else {
                merged.push((start, end));
            }
        }
        let mut total_ns = 0_u64;
        let prefix_ns = merged
            .iter()
            .map(|(start, end)| {
                total_ns = total_ns.saturating_add(end - start);
                total_ns
            })
            .collect();
        Self {
            intervals: merged,
            prefix_ns,
        }
    }

    fn total_ns(&self) -> u64 {
        self.prefix_ns.last().copied().unwrap_or_default()
    }

    fn compressed_ns(&self, value: u64) -> u64 {
        let index = self
            .intervals
            .partition_point(|(_, interval_end)| *interval_end <= value);
        let prior = index
            .checked_sub(1)
            .and_then(|index| self.prefix_ns.get(index))
            .copied()
            .unwrap_or_default();
        let Some((start, end)) = self.intervals.get(index).copied() else {
            return self.total_ns();
        };
        prior.saturating_add(value.saturating_sub(start).min(end - start))
    }

    fn append_tail(&mut self, start: u64, end: u64) -> bool {
        if end <= start {
            return true;
        }
        let Some((last_start, last_end)) = self.intervals.last().copied() else {
            self.intervals.push((start, end));
            self.prefix_ns.push(end - start);
            return true;
        };
        if start < last_start {
            return false;
        }
        if start <= last_end {
            let extended_end = last_end.max(end);
            let extension = extended_end - last_end;
            if let Some(last) = self.intervals.last_mut() {
                last.1 = extended_end;
            }
            if let Some(prefix) = self.prefix_ns.last_mut() {
                *prefix = prefix.saturating_add(extension);
            }
        } else {
            self.intervals.push((start, end));
            self.prefix_ns
                .push(self.total_ns().saturating_add(end - start));
        }
        true
    }
}

fn relative_timing(
    start: &TimelinePoint,
    end: &TimelinePoint,
    nested_start: &TimelinePoint,
    nested_end: &TimelinePoint,
) -> Option<(f64, f64, f64)> {
    if start.clock_id == end.clock_id
        && start.clock_id == nested_start.clock_id
        && start.clock_id == nested_end.clock_id
    {
        let raw_width = end.monotonic_ns.checked_sub(start.monotonic_ns)? as f64;
        let nested_offset = nested_start.monotonic_ns.saturating_sub(start.monotonic_ns) as f64;
        let nested_width = nested_end
            .monotonic_ns
            .checked_sub(nested_start.monotonic_ns)? as f64;
        (raw_width > 0.0).then_some((nested_offset, nested_width, raw_width))
    } else {
        let raw_width = end.wall_ms - start.wall_ms;
        (raw_width > 0.0).then_some((
            (nested_start.wall_ms - start.wall_ms).max(0.0),
            (nested_end.wall_ms - nested_start.wall_ms).max(0.0),
            raw_width,
        ))
    }
}

fn finite(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

fn total_cmp(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use proptest::prelude::*;

    use super::*;

    fn axis(mode: TimelineMode, revision: u64) -> AxisId {
        AxisId {
            document_generation: 1,
            geometry_revision: revision,
            mode,
        }
    }

    fn span(id: usize, lane: TimelineLane, start: f64, end: f64) -> TimelineSpan {
        let point = |milliseconds: f64| TimelinePoint {
            wall_ms: milliseconds,
            clock_id: "test-clock".into(),
            monotonic_ns: (milliseconds * 1_000_000.0) as u64,
        };
        TimelineSpan {
            id,
            lane,
            sequence: id as u64,
            started: Some(point(start)),
            completed: Some(point(end)),
            duration_ms: Some((end - start).max(0.0)),
            nested: None,
        }
    }

    #[test]
    fn duration_axis_removes_only_idle_union() {
        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Duration, 1),
            [
                span(0, TimelineLane::Model, 0.0, 10.0),
                span(1, TimelineLane::Tools, 5.0, 15.0),
                span(2, TimelineLane::Model, 20.0, 25.0),
            ],
        );
        assert_eq!(geometry.domain, DomainRange::new(0.0, 20.0));
        assert_eq!(geometry.cells[2].range, DomainRange::new(15.0, 20.0));
    }

    #[test]
    fn duration_axis_is_not_corrupted_by_wall_clock_jumps() {
        let mut model = span(0, TimelineLane::Model, 0.0, 10.0);
        model.started.as_mut().unwrap().wall_ms = 10_000.0;
        model.completed.as_mut().unwrap().wall_ms = -5_000.0;
        let mut tool = span(1, TimelineLane::Tools, 5.0, 15.0);
        tool.started.as_mut().unwrap().wall_ms = 1_000_000.0;
        tool.completed.as_mut().unwrap().wall_ms = 2_000_000.0;

        let geometry = TimelineGeometry::build(axis(TimelineMode::Duration, 1), [model, tool]);
        assert_eq!(geometry.domain, DomainRange::new(0.0, 15.0));
        assert_eq!(geometry.cells[1].range, DomainRange::new(5.0, 15.0));
    }

    #[test]
    fn duration_axis_appends_independent_monotonic_clock_domains() {
        let first = span(0, TimelineLane::Model, 0.0, 10.0);
        let mut second = span(1, TimelineLane::Model, 0.0, 5.0);
        second.started.as_mut().unwrap().clock_id = "second-clock".into();
        second.completed.as_mut().unwrap().clock_id = "second-clock".into();

        let geometry = TimelineGeometry::build(axis(TimelineMode::Duration, 1), [first, second]);
        assert_eq!(geometry.domain, DomainRange::new(0.0, 15.0));
        assert_eq!(geometry.cells[1].range, DomainRange::new(10.0, 15.0));
    }

    #[test]
    fn duration_axis_keeps_all_zero_or_unfinished_items_visible_as_points() {
        let mut unfinished = span(1, TimelineLane::Model, 0.0, 0.0);
        unfinished.completed = None;
        unfinished.duration_ms = None;
        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Duration, 1),
            [unfinished, span(2, TimelineLane::Tools, 0.0, 0.0)],
        );

        assert_eq!(geometry.domain, DomainRange::new(0.0, 2.0));
        assert_eq!(geometry.cells.len(), 2);
        assert_eq!(geometry.cells[0].range, DomainRange::new(0.5, 0.5));
        assert_eq!(geometry.cells[1].range, DomainRange::new(1.5, 1.5));
    }

    #[test]
    fn duration_axis_keeps_point_only_clock_domains_beside_positive_intervals() {
        let completed = span(0, TimelineLane::Model, 0.0, 10.0);
        let mut point = span(1, TimelineLane::Input, 1.0, 1.0);
        point.started.as_mut().unwrap().clock_id = "point-clock".into();
        point.completed.as_mut().unwrap().clock_id = "point-clock".into();
        let mut unfinished = span(2, TimelineLane::Tools, 2.0, 2.0);
        unfinished.started.as_mut().unwrap().clock_id = "unfinished-clock".into();
        unfinished.completed = None;
        unfinished.duration_ms = None;

        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Duration, 1),
            [completed, point, unfinished],
        );

        assert_eq!(geometry.cells.len(), 3);
        assert_eq!(geometry.cells[0].range, DomainRange::new(0.0, 10.0));
        assert_eq!(geometry.cells[1].range, DomainRange::new(10.5, 10.5));
        assert_eq!(geometry.cells[2].range, DomainRange::new(11.5, 11.5));
        assert_eq!(geometry.domain, DomainRange::new(0.0, 12.0));
    }

    #[test]
    fn nested_interval_outside_its_parent_is_not_reflected_back_inside() {
        let mut parent = span(1, TimelineLane::Tools, 10.0, 20.0);
        parent.nested = Some((
            TimelinePoint {
                wall_ms: 30.0,
                clock_id: "test-clock".into(),
                monotonic_ns: 30_000_000,
            },
            TimelinePoint {
                wall_ms: 40.0,
                clock_id: "test-clock".into(),
                monotonic_ns: 40_000_000,
            },
        ));
        let geometry = TimelineGeometry::build(axis(TimelineMode::Actual, 1), [parent]);
        assert_eq!(geometry.cells[0].nested, None);
    }

    #[test]
    fn interval_query_finds_long_span_that_starts_before_the_range() {
        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Actual, 1),
            [
                span(1, TimelineLane::Tools, 0.0, 100.0),
                span(2, TimelineLane::Tools, 90.0, 91.0),
            ],
        );
        let ids = geometry
            .query(TimelineLane::Tools, DomainRange::new(95.0, 96.0))
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn hit_test_prefers_the_last_painted_overlapping_item() {
        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Actual, 1),
            [
                span(1, TimelineLane::Tools, 0.0, 100.0),
                span(2, TimelineLane::Tools, 10.0, 90.0),
            ],
        );
        assert_eq!(geometry.hit_test(TimelineLane::Tools, 50.0), Some(&2));
    }

    #[test]
    fn pure_range_operations_preserve_timeline_interaction_semantics() {
        let domain = DomainRange::new(0.0, 100.0);
        let mut viewport = DomainRange::new(20.0, 40.0);
        for _ in 0..40 {
            viewport = viewport.auto_pan(domain, 1.0, 0.1, 0.1);
        }
        assert_eq!(viewport, DomainRange::new(80.0, 100.0));
        assert_eq!(viewport.value_at_fraction(1.0), 100.0);
        assert_eq!(
            DomainRange::new(99.0, 99.0).with_minimum_width(domain, 10.0),
            DomainRange::new(90.0, 100.0)
        );
        assert_eq!(
            DomainRange::new(0.0, 100.0).zoom(domain, 50.0, 0.05, 20.0),
            DomainRange::new(40.0, 60.0)
        );
        assert_eq!(
            DomainRange::new(10.0, 30.0).pan_to_reveal(DomainRange::new(70.0, 75.0), domain),
            DomainRange::new(55.0, 75.0)
        );
    }

    #[test]
    fn lod_caps_primitive_count_without_losing_item_identity() {
        let spans = (0..10_000).map(|id| {
            span(
                id,
                TIMELINE_LANES[id % TIMELINE_LANES.len()],
                id as f64,
                id as f64 + 0.5,
            )
        });
        let geometry = TimelineGeometry::build(axis(TimelineMode::Actual, 1), spans);
        let model = geometry.render_model(geometry.domain, 1_500.0, 3_000);
        assert!(model.len() <= 3_000);
        assert_eq!(
            model.iter().map(|cell| cell.ids.len()).sum::<usize>(),
            10_000
        );
        let represented = model
            .iter()
            .flat_map(|cell| geometry.render_members(cell).copied())
            .collect::<HashSet<_>>();
        assert_eq!(represented.len(), 10_000);
        assert!(
            model
                .iter()
                .filter(|cell| cell.clustered)
                .all(|cell| cell.ids.stored_word_count() == 8)
        );
    }

    #[test]
    fn lod_preserves_last_painted_representative_for_non_monotonic_actual_time() {
        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Actual, 1),
            [
                span(0, TimelineLane::Tools, 30.0, 31.0),
                span(1, TimelineLane::Tools, 10.0, 11.0),
                span(2, TimelineLane::Tools, 20.0, 21.0),
                span(3, TimelineLane::Tools, 0.0, 1.0),
            ],
        );
        let model = geometry.render_model(geometry.domain, 100.0, 3);
        assert_eq!(model.len(), 1);
        assert_eq!(model[0].ids.last(), Some(&3));
        assert_eq!(
            geometry
                .render_members(&model[0])
                .copied()
                .collect::<HashSet<_>>(),
            (0..4).collect()
        );
    }

    #[test]
    fn compact_lane_range_filters_expired_prefix_intervals_exactly() {
        let geometry = TimelineGeometry::build(
            axis(TimelineMode::Actual, 1),
            [
                span(0, TimelineLane::Tools, 0.0, 100.0),
                span(1, TimelineLane::Tools, 10.0, 11.0),
                span(2, TimelineLane::Tools, 20.0, 21.0),
                span(3, TimelineLane::Tools, 95.0, 96.0),
            ],
        );
        let model = geometry.render_model(DomainRange::new(90.0, 100.0), 100.0, 1);
        assert_eq!(model.len(), 1);
        assert_eq!(model[0].ids.len(), 2);
        assert_eq!(
            geometry
                .render_members(&model[0])
                .copied()
                .collect::<HashSet<_>>(),
            HashSet::from([0, 3])
        );
        assert_eq!(geometry.render_cell_for_record(&model, 0), Some(0));
        assert_eq!(geometry.render_cell_for_record(&model, 1), None);
        assert_eq!(geometry.render_cell_for_record(&model, 3), Some(0));
    }

    #[test]
    fn hundred_thousand_lod_reprojects_with_bounded_identity_and_visible_work() {
        let spans = (0..100_000).map(|id| {
            span(
                id,
                TIMELINE_LANES[id % TIMELINE_LANES.len()],
                id as f64,
                id as f64 + 0.5,
            )
        });
        let geometry = TimelineGeometry::build(axis(TimelineMode::Sequence, 1), spans);
        let started = Instant::now();
        let mut last_work = RenderWork::default();
        for _ in 0..25 {
            let (model, work) = geometry.render_model_with_work(geometry.domain, 1_500.0, 3_000);
            assert!(model.len() <= 3_000);
            assert_eq!(
                model.iter().map(|cell| cell.ids.len()).sum::<usize>(),
                100_000
            );
            assert!(work.scanned_intervals <= 3_001);
            assert!(work.aggregate_queries <= 3_000);
            assert!(
                work.aggregate_nodes == 0,
                "Sequence lanes use O(1) summaries"
            );
            assert!(work.stored_identity_words <= 24_000);
            for id in [0, 1, 2, 49_999, 99_997, 99_998, 99_999] {
                let cell = geometry
                    .render_cell_for_record(&model, id)
                    .expect("visible record resolves to one compact primitive");
                assert!(
                    geometry
                        .render_members(&model[cell])
                        .any(|member| *member == id)
                );
            }
            last_work = work;
        }
        // A deliberately generous debug-build gate catches accidental O(n) allocation/sorting on
        // every projection without depending on benchmark-only compiler settings.
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(last_work.scanned_intervals, 3_001);
    }

    #[test]
    fn actual_tail_timing_and_append_match_a_full_rebuild() {
        let initial = (0..100_000)
            .map(|id| {
                span(
                    id,
                    TIMELINE_LANES[id % TIMELINE_LANES.len()],
                    id as f64,
                    id as f64 + 0.5,
                )
            })
            .collect::<Vec<_>>();
        let mut incremental =
            TimelineGeometry::build(axis(TimelineMode::Actual, 1), initial.clone());
        let mut changed_tail = initial.last().unwrap().clone();
        changed_tail.completed = Some(TimelinePoint {
            wall_ms: 100_000.0,
            clock_id: "test-clock".into(),
            monotonic_ns: 100_000_000_000,
        });
        changed_tail.duration_ms = Some(1.0);
        let appended = span(100_000, TimelineLane::Model, 100_001.0, 100_002.0);
        let update = incremental
            .update_timed(
                axis(TimelineMode::Actual, 2),
                100_001,
                [(99_999, changed_tail.clone()), (100_000, appended.clone())],
            )
            .expect("tail-only actual delta is incremental");
        assert_eq!(update.inspected_spans, 2);
        assert_eq!(update.updated_cells, 1);
        assert_eq!(update.appended_cells, 1);
        assert_eq!(update.lane_entries_touched, 2);

        let mut expected = initial;
        expected[99_999] = changed_tail;
        expected.push(appended);
        let rebuilt = TimelineGeometry::build(axis(TimelineMode::Actual, 2), expected);
        assert_eq!(incremental.axis, rebuilt.axis);
        assert_eq!(incremental.domain, rebuilt.domain);
        assert_eq!(incremental.cells, rebuilt.cells);
        let (_, work) = incremental.render_model_with_work(incremental.domain, 1_500.0, 3_000);
        assert!(work.scanned_intervals <= 3_001);
        assert!(work.aggregate_queries <= 3_000);
    }

    #[test]
    fn duration_tail_extension_and_append_match_a_full_rebuild() {
        let initial = (0..100_000)
            .map(|id| {
                span(
                    id,
                    TIMELINE_LANES[id % TIMELINE_LANES.len()],
                    id as f64,
                    id as f64 + 0.5,
                )
            })
            .collect::<Vec<_>>();
        let mut incremental =
            TimelineGeometry::build(axis(TimelineMode::Duration, 1), initial.clone());
        let mut changed_tail = initial.last().unwrap().clone();
        changed_tail.completed = Some(TimelinePoint {
            wall_ms: 100_000.0,
            clock_id: "test-clock".into(),
            monotonic_ns: 100_000_000_000,
        });
        changed_tail.duration_ms = Some(1.0);
        let appended = span(100_000, TimelineLane::Model, 100_001.0, 100_002.0);
        let update = incremental
            .update_timed(
                axis(TimelineMode::Duration, 2),
                100_001,
                [(99_999, changed_tail.clone()), (100_000, appended.clone())],
            )
            .expect("tail-only duration delta is incremental");
        assert_eq!(update.inspected_spans, 2);
        assert_eq!(update.updated_cells, 1);
        assert_eq!(update.appended_cells, 1);

        let mut expected = initial;
        expected[99_999] = changed_tail;
        expected.push(appended);
        let rebuilt = TimelineGeometry::build(axis(TimelineMode::Duration, 2), expected);
        assert_eq!(incremental.axis, rebuilt.axis);
        assert_eq!(incremental.domain, rebuilt.domain);
        assert_eq!(incremental.cells, rebuilt.cells);
        let (_, work) = incremental.render_model_with_work(incremental.domain, 1_500.0, 3_000);
        assert!(work.scanned_intervals <= 3_001);
        assert!(work.aggregate_queries <= 3_000);
    }

    #[test]
    fn timed_update_rejects_a_delta_that_would_shift_prior_geometry() {
        let first = span(0, TimelineLane::Model, 0.0, 10.0);
        let mut second = span(1, TimelineLane::Tools, 0.0, 5.0);
        second.started.as_mut().unwrap().clock_id = "second-clock".into();
        second.completed.as_mut().unwrap().clock_id = "second-clock".into();
        let mut geometry = TimelineGeometry::build(
            axis(TimelineMode::Duration, 1),
            [first.clone(), second.clone()],
        );
        let before_cells = geometry.cells.clone();
        let before_domain = geometry.domain;
        let mut changed_first = first;
        changed_first.completed.as_mut().unwrap().monotonic_ns = 20_000_000;
        changed_first.completed.as_mut().unwrap().wall_ms = 20.0;
        changed_first.duration_ms = Some(20.0);

        assert!(
            geometry
                .update_timed(axis(TimelineMode::Duration, 2), 2, [(0, changed_first)])
                .is_none()
        );
        assert_eq!(geometry.cells, before_cells);
        assert_eq!(geometry.domain, before_domain);
        assert_eq!(geometry.axis, axis(TimelineMode::Duration, 1));
    }

    #[test]
    fn rejected_duration_suffix_rolls_back_a_prior_tail_extension() {
        let first = span(0, TimelineLane::Model, 0.0, 10.0);
        let mut second = span(1, TimelineLane::Tools, 0.0, 5.0);
        second.started.as_mut().unwrap().clock_id = "second-clock".into();
        second.completed.as_mut().unwrap().clock_id = "second-clock".into();
        let mut geometry = TimelineGeometry::build(
            axis(TimelineMode::Duration, 1),
            [first.clone(), second.clone()],
        );
        let mut extended_second = second.clone();
        extended_second.completed.as_mut().unwrap().wall_ms = 6.0;
        extended_second.completed.as_mut().unwrap().monotonic_ns = 6_000_000;
        extended_second.duration_ms = Some(6.0);
        let invalid_append = span(2, TimelineLane::Input, 20.0, 21.0);
        assert!(
            geometry
                .update_timed(
                    axis(TimelineMode::Duration, 2),
                    3,
                    [(1, extended_second.clone()), (2, invalid_append)],
                )
                .is_none()
        );

        let mut valid_append = span(2, TimelineLane::Input, 7.0, 8.0);
        valid_append.started.as_mut().unwrap().clock_id = "third-clock".into();
        valid_append.completed.as_mut().unwrap().clock_id = "third-clock".into();
        geometry
            .update_timed(
                axis(TimelineMode::Duration, 2),
                3,
                [(1, extended_second.clone()), (2, valid_append.clone())],
            )
            .expect("rollback leaves the valid tail extension applicable");
        let rebuilt = TimelineGeometry::build(
            axis(TimelineMode::Duration, 2),
            [first, extended_second, valid_append],
        );
        assert_eq!(geometry.domain, rebuilt.domain);
        assert_eq!(geometry.cells, rebuilt.cells);
    }

    #[test]
    fn omitted_actual_tail_remains_the_source_tail_for_the_next_append() {
        let first = span(0, TimelineLane::Model, 0.0, 1.0);
        let mut omitted = span(1, TimelineLane::Input, 2.0, 2.0);
        omitted.started = None;
        omitted.completed = None;
        omitted.duration_ms = None;
        let mut geometry = TimelineGeometry::build(axis(TimelineMode::Actual, 1), [first.clone()]);
        geometry
            .update_timed(axis(TimelineMode::Actual, 2), 2, [(1, omitted.clone())])
            .expect("an omitted suffix still advances canonical source identity");
        assert_eq!(geometry.source_len, 2);
        assert_eq!(geometry.source_tail.as_ref().map(|span| span.id), Some(1));
        assert_eq!(geometry.source_tail_cell_index, None);

        let appended = span(2, TimelineLane::Tools, 3.0, 4.0);
        geometry
            .update_timed(axis(TimelineMode::Actual, 3), 3, [(2, appended.clone())])
            .expect("the next contiguous suffix is recognized after an omitted tail");
        let rebuilt =
            TimelineGeometry::build(axis(TimelineMode::Actual, 3), [first, omitted, appended]);
        assert_eq!(geometry.domain, rebuilt.domain);
        assert_eq!(geometry.cells, rebuilt.cells);
    }

    #[test]
    fn hundred_thousand_sequence_cells_update_from_only_the_changed_suffix() {
        let spans = (0..100_000).map(|id| {
            span(
                id,
                TIMELINE_LANES[id % TIMELINE_LANES.len()],
                id as f64,
                id as f64 + 0.5,
            )
        });
        let mut geometry = TimelineGeometry::build(axis(TimelineMode::Sequence, 1), spans);
        let unchanged = geometry.cells[50_000].clone();

        let mut changed = span(
            77_777,
            TIMELINE_LANES[77_777 % TIMELINE_LANES.len()],
            77_777.0,
            77_777.5,
        );
        changed.nested = Some((
            TimelinePoint {
                wall_ms: 77_777.1,
                clock_id: "test-clock".into(),
                monotonic_ns: 77_777_100_000,
            },
            TimelinePoint {
                wall_ms: 77_777.4,
                clock_id: "test-clock".into(),
                monotonic_ns: 77_777_400_000,
            },
        ));
        let appended = span(100_000, TimelineLane::Model, 100_000.0, 100_000.5);
        let inspected = std::cell::Cell::new(0_usize);
        let changes = [(77_777, changed), (100_000, appended)]
            .into_iter()
            .inspect(|_| inspected.set(inspected.get().saturating_add(1)));

        assert!(geometry.update_sequence(axis(TimelineMode::Sequence, 2), 100_001, changes,));
        assert_eq!(inspected.get(), 2);
        assert_eq!(geometry.cells.len(), 100_001);
        assert_eq!(geometry.cells[50_000], unchanged);
        assert!(geometry.cells[77_777].nested.is_some());
        assert_eq!(
            geometry.range_for(&100_000).map(|range| range.range),
            Some(DomainRange::new(100_000.0, 100_001.0))
        );
        let (model, work) = geometry.render_model_with_work(geometry.domain, 1_500.0, 3_000);
        assert!(model.len() <= 3_000);
        assert!(work.scanned_intervals <= 3_001);
        assert!(work.aggregate_queries <= 3_000);
        assert!(work.stored_identity_words <= 24_000);
    }

    proptest! {
        #[test]
        fn normalized_ranges_always_stay_inside_the_domain(
            start in -10_000.0f64..10_000.0,
            end in -10_000.0f64..10_000.0,
        ) {
            let domain = DomainRange::new(-100.0, 100.0);
            let range = DomainRange::new(start, end).clamp_to(domain);
            prop_assert!(range.start >= domain.start);
            prop_assert!(range.end <= domain.end);
            prop_assert!(range.start <= range.end);
        }
    }
}
