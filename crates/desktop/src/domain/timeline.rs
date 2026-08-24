use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

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

#[derive(Clone, Debug, Default)]
struct LaneIndex {
    entries: Vec<IndexedInterval>,
    prefix_max_end: Vec<f64>,
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
        Self {
            entries,
            prefix_max_end,
        }
    }

    fn query(&self, range: DomainRange) -> impl Iterator<Item = &usize> {
        let right = self
            .entries
            .partition_point(|entry| entry.start <= range.end);
        let left = self.prefix_max_end[..right].partition_point(|end| *end < range.start);
        self.entries[left..right]
            .iter()
            .filter(move |entry| entry.end >= range.start)
            .map(|entry| &entry.id)
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

#[derive(Clone, Debug)]
pub(crate) struct TimelineGeometry {
    pub(crate) axis: AxisId,
    pub(crate) domain: DomainRange,
    pub(crate) cells: Vec<GeometryCell>,
    lanes: [LaneIndex; 3],
    cell_indices: Vec<Option<usize>>,
}

impl TimelineGeometry {
    pub(crate) fn build(axis: AxisId, spans: impl IntoIterator<Item = TimelineSpan>) -> Self {
        let mut spans = spans.into_iter().collect::<Vec<_>>();
        spans.sort_by_key(|span| span.sequence);
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

    pub(crate) fn render_model(
        &self,
        viewport: DomainRange,
        width_px: f64,
        primitive_limit: usize,
    ) -> Vec<RenderCell> {
        if width_px <= 0.0 || primitive_limit == 0 {
            return Vec::new();
        }
        let viewport = viewport.clamp_to(self.domain);
        let mut visible = TIMELINE_LANES
            .into_iter()
            .flat_map(|lane| self.query(lane, viewport))
            .filter_map(|id| {
                self.cell_indices
                    .get(*id)
                    .and_then(|index| *index)
                    .and_then(|index| self.cells.get(index))
            })
            .collect::<Vec<_>>();
        visible.sort_by_key(|cell| self.cell_indices.get(cell.id).and_then(|index| *index));
        if visible.len() <= primitive_limit {
            return visible
                .into_iter()
                .map(|cell| RenderCell::from_cell(cell, viewport, width_px))
                .collect();
        }

        let bins_per_lane = (primitive_limit / TIMELINE_LANES.len()).max(1);
        let bin_width = (width_px / bins_per_lane as f64).max(1.0);
        let mut bins = vec![None::<RenderCell>; bins_per_lane * TIMELINE_LANES.len()];
        for cell in visible {
            let projected = RenderCell::from_cell(cell, viewport, width_px);
            let bin = ((projected.start_px / bin_width).floor() as usize).min(bins_per_lane - 1);
            let index = lane_index(cell.lane) * bins_per_lane + bin;
            if let Some(existing) = &mut bins[index] {
                existing.start_px = existing.start_px.min(projected.start_px);
                existing.end_px = existing.end_px.max(projected.end_px);
                existing.ids.extend(projected.ids);
                existing.clustered = true;
                existing.nested = None;
            } else {
                bins[index] = Some(projected);
            }
        }
        bins.into_iter().flatten().collect()
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RenderCell {
    pub(crate) ids: Vec<usize>,
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
            ids: vec![cell.id],
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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SelectionResult {
    pub(crate) range: AxisRange,
    pub(crate) items: HashSet<usize>,
}

#[derive(Clone, Debug, Default)]
struct BusyTimelineSet {
    clocks: Vec<BusyClock>,
    clock_indices: HashMap<String, usize>,
    total_ns: u64,
}

#[derive(Clone, Debug)]
struct BusyClock {
    offset_ns: u64,
    timeline: BusyTimeline,
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
}

#[derive(Clone, Debug, Default)]
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
