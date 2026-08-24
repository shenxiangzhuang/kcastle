use std::borrow::Borrow;
use std::collections::HashSet;
use std::sync::Arc;

use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement, Pixels, Point, Role, ScrollStrategy, ScrollWheelEvent,
    SharedString, StatefulInteractiveElement, Styled, Window, accesskit, div,
    prelude::FluentBuilder, px, relative,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::Input;
use gpui_component::scroll::ScrollableElement;
use gpui_component::tooltip::Tooltip;
use gpui_component::{ElementExt, Icon, IconName, Sizable};
use im::{HashSet as ImHashSet, Vector};
use time::{OffsetDateTime, UtcOffset, macros::format_description};

use crate::app::{DesktopApp, TimelineDragState, TimelineHoverState};
use crate::assets::DesktopIconName;
use crate::domain::session_document::EventTimeRef;
use crate::domain::timeline::{
    AxisId, AxisRange, DomainRange, RenderCell, TimelineGeometry, TimelineLane, TimelinePoint,
    TimelineSpan,
};
use crate::domain::{
    Action, DetailsSelection, DetailsTab, ItemStatus, LayoutGeneration, ModelRequestOptions,
    PromptChangeKind, PromptSnapshot, TimelineMode, TrajectoryItemId, TrajectoryKind,
    TrajectoryRecord, TrajectoryRecordDetails, TrajectoryRequest, TrajectoryRequestPurpose,
};
use crate::dsh_markdown;
use crate::layout::TrajectoryMode;
use crate::streaming_markdown::StreamingMarkdownState;
use crate::ui_theme::{TrajectoryPalette, metrics, trajectory_palette};

const TIMELINE_INPUT_TOP: f32 = 6.0;
const TIMELINE_MODEL_TOP: f32 = 20.0;
const TIMELINE_TOOLS_TOP: f32 = 34.0;
const TIMELINE_BAR_OFFSET: f32 = 1.0;
const TIMELINE_BAR_HEIGHT: f32 = 8.0;
const TIMELINE_CLICK_SLOP: f32 = 3.0;
const TIMELINE_PRIMITIVE_LIMIT: usize = 3_000;
const TIMELINE_EDGE_PAN_MAX_PX: f64 = 48.0;
const TIMELINE_EDGE_PAN_ZONE_FRACTION: f64 = 0.08;
const TIMELINE_EDGE_PAN_STEP_FRACTION: f64 = 0.035;
const DETAILS_MIN_WIDTH: f32 = 320.0;
const DETAILS_MAX_WIDTH: f32 = 720.0;
const DETAILS_DEFAULT_MAX_WIDTH: f32 = 440.0;
const LEDGER_MIN_WIDTH: f32 = 280.0;
const DETAILS_OVERLAY_MAX_WIDTH: f32 = 420.0;
const DETAILS_OVERLAY_FRACTION: f32 = 0.92;
const DETAILS_KEYBOARD_STEP: f32 = 16.0;
const DETAILS_MEASUREMENT_EPSILON: f32 = 0.5;
const COMPACT_LEDGER_MAX_WIDTH: f32 = 620.0;
const TOOL_SUMMARY_PREVIEW_MAX_BYTES: usize = 160;

fn trajectory_details_default_width(main_width: f32) -> f32 {
    (main_width * 0.38).clamp(DETAILS_MIN_WIDTH, DETAILS_DEFAULT_MAX_WIDTH)
}

fn sanitized_width(width: f32) -> f32 {
    if width.is_finite() {
        width.max(0.0)
    } else {
        0.0
    }
}

fn clamp_trajectory_details_width(width: f32, split_width: f32) -> f32 {
    let max_width = (sanitized_width(split_width) - LEDGER_MIN_WIDTH)
        .clamp(DETAILS_MIN_WIDTH, DETAILS_MAX_WIDTH);
    sanitized_width(width)
        .clamp(DETAILS_MIN_WIDTH, max_width)
        .round()
}

fn resolved_trajectory_details_width(
    mode: TrajectoryMode,
    split_width: f32,
    explicit_width: Option<f32>,
) -> f32 {
    let split_width = sanitized_width(split_width);
    match mode {
        TrajectoryMode::Split => {
            let max_width = (split_width - LEDGER_MIN_WIDTH).max(0.0);
            explicit_width
                .map(|width| sanitized_width(width).clamp(DETAILS_MIN_WIDTH, DETAILS_MAX_WIDTH))
                .unwrap_or_else(|| trajectory_details_default_width(split_width))
                .min(max_width)
        }
        TrajectoryMode::Overlay => explicit_width
            .map(|width| sanitized_width(width).clamp(DETAILS_MIN_WIDTH, DETAILS_MAX_WIDTH))
            .unwrap_or_else(|| {
                (split_width * DETAILS_OVERLAY_FRACTION).min(DETAILS_OVERLAY_MAX_WIDTH)
            })
            .min(split_width * DETAILS_OVERLAY_FRACTION),
        TrajectoryMode::Ledger => 0.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TrajectoryDetailsResizeDrag {
    start_x: f32,
    start_width: f32,
    split_width: f32,
}

#[derive(Debug, Default)]
pub(crate) struct TrajectoryDetailsLayoutState {
    explicit_width: Option<f32>,
    measured_split_width: Option<(LayoutGeneration, f32)>,
    measured_details_width: Option<(LayoutGeneration, f32)>,
    drag: Option<TrajectoryDetailsResizeDrag>,
}

impl TrajectoryDetailsLayoutState {
    fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    fn observe_split_width(&mut self, generation: LayoutGeneration, width: f32) -> bool {
        let changed = Self::observe_width(&mut self.measured_split_width, generation, width);
        if changed {
            self.measured_details_width = None;
        }
        changed
    }

    fn observe_details_width(&mut self, generation: LayoutGeneration, width: f32) -> bool {
        Self::observe_width(&mut self.measured_details_width, generation, width)
    }

    fn observe_width(
        measured: &mut Option<(LayoutGeneration, f32)>,
        generation: LayoutGeneration,
        width: f32,
    ) -> bool {
        if !width.is_finite() || width <= 0.0 {
            return false;
        }
        let unchanged = measured.is_some_and(|(measured_generation, measured_width)| {
            measured_generation == generation
                && (measured_width - width).abs() < DETAILS_MEASUREMENT_EPSILON
        });
        if unchanged {
            false
        } else {
            *measured = Some((generation, width));
            true
        }
    }

    fn split_width(&self, generation: LayoutGeneration, fallback: f32) -> f32 {
        self.measured_split_width
            .filter(|(measured_generation, _)| *measured_generation == generation)
            .map(|(_, width)| width)
            .unwrap_or_else(|| sanitized_width(fallback))
    }

    fn details_width(
        &self,
        mode: TrajectoryMode,
        generation: LayoutGeneration,
        fallback_split_width: f32,
    ) -> f32 {
        resolved_trajectory_details_width(
            mode,
            self.split_width(generation, fallback_split_width),
            self.explicit_width,
        )
    }

    fn measured_details_width(&self, generation: LayoutGeneration) -> Option<f32> {
        self.measured_details_width
            .filter(|(measured_generation, _)| *measured_generation == generation)
            .map(|(_, width)| width)
    }

    fn begin_drag(&mut self, start_x: f32, start_width: f32, split_width: f32) {
        self.drag = Some(TrajectoryDetailsResizeDrag {
            start_x,
            start_width: sanitized_width(start_width),
            split_width: sanitized_width(split_width),
        });
    }

    fn drag_to(&mut self, current_x: f32) -> bool {
        let Some(drag) = self.drag else {
            return false;
        };
        self.set_explicit_width(clamp_trajectory_details_width(
            drag.start_width + drag.start_x - current_x,
            drag.split_width,
        ))
    }

    fn end_drag(&mut self) -> bool {
        self.drag.take().is_some()
    }

    fn step(&mut self, delta: f32, current_width: f32, split_width: f32) -> bool {
        self.set_explicit_width(clamp_trajectory_details_width(
            current_width + delta,
            split_width,
        ))
    }

    fn reset(&mut self) -> bool {
        let changed = self.explicit_width.take().is_some() || self.drag.take().is_some();
        if changed {
            self.measured_details_width = None;
        }
        changed
    }

    fn set_explicit_width(&mut self, width: f32) -> bool {
        if self
            .explicit_width
            .is_some_and(|previous| (previous - width).abs() < f32::EPSILON)
        {
            false
        } else {
            self.explicit_width = Some(width);
            self.measured_details_width = None;
            true
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrajectoryMarkdownSource {
    SystemPrompt,
    Preview,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrajectoryMarkdownCacheKey {
    projection_lineage: u64,
    record_id: TrajectoryItemId,
    source: TrajectoryMarkdownSource,
}

#[derive(Debug, Default)]
pub(crate) struct TrajectoryDetailsMarkdownCache {
    key: Option<TrajectoryMarkdownCacheKey>,
    markdown: StreamingMarkdownState,
    fallback: SharedString,
}

impl TrajectoryDetailsMarkdownCache {
    fn sync(
        &mut self,
        projection_lineage: u64,
        record_id: &TrajectoryItemId,
        source_kind: TrajectoryMarkdownSource,
        source: &str,
    ) {
        let key = TrajectoryMarkdownCacheKey {
            projection_lineage,
            record_id: record_id.clone(),
            source: source_kind,
        };
        if self.key.as_ref() != Some(&key) {
            self.key = Some(key);
            self.markdown = StreamingMarkdownState::default();
            self.fallback = source.to_owned().into();
        } else if self.fallback.as_ref() != source {
            self.fallback = source.to_owned().into();
        }
        self.markdown.update(source);
    }
}

#[cfg(test)]
fn collapsible_trajectory_groups(
    records: &Vector<Arc<TrajectoryRecord>>,
) -> (HashSet<u32>, HashSet<TrajectoryItemId>) {
    let mut content_per_turn = std::collections::HashMap::<u32, usize>::new();
    let mut assistants = HashSet::new();
    for (index, record) in records.iter().enumerate() {
        if record.kind != TrajectoryKind::System
            && let Some(turn) = record.turn
        {
            *content_per_turn.entry(turn).or_default() += 1;
        }
        if record.kind == TrajectoryKind::Assistant
            && records
                .get(index + 1)
                .is_some_and(|next| next.kind == TrajectoryKind::Tool)
        {
            assistants.insert(record.id.clone());
        }
    }
    (
        content_per_turn
            .into_iter()
            .filter_map(|(turn, count)| (count > 1).then_some(turn))
            .collect(),
        assistants,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrajectorySelectionSource {
    Ledger,
    Timeline,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LedgerFoldTarget {
    Turn(u32),
    Assistant(TrajectoryItemId),
}

fn ledger_double_click_target(
    record: &TrajectoryRecord,
    turn_start: bool,
    collapsed_turns: &HashSet<u32>,
    collapsible_turns: &HashSet<u32>,
    collapsible_assistants: &HashSet<TrajectoryItemId>,
) -> Option<LedgerFoldTarget> {
    if let Some(turn) = record.turn
        && collapsed_turns.contains(&turn)
    {
        return Some(LedgerFoldTarget::Turn(turn));
    }
    if record.kind == TrajectoryKind::Assistant && collapsible_assistants.contains(&record.id) {
        return Some(LedgerFoldTarget::Assistant(record.id.clone()));
    }
    let turn = record.turn?;
    (turn_start && collapsible_turns.contains(&turn)).then_some(LedgerFoldTarget::Turn(turn))
}

fn ledger_row_turn(
    row: &TimelineLedgerRow,
    records: &Vector<Arc<TrajectoryRecord>>,
) -> Option<u32> {
    match row {
        TimelineLedgerRow::Record(index) => records.get(*index).and_then(|record| record.turn),
        TimelineLedgerRow::TurnSummary { turn, .. } => Some(*turn),
        TimelineLedgerRow::CallsSummary { assistant, .. } => {
            records.get(*assistant).and_then(|record| record.turn)
        }
        TimelineLedgerRow::RequestBoundary { .. } => None,
    }
}

fn ledger_record_boundaries(
    rows: &TimelineRows,
    row: usize,
    record: &TrajectoryRecord,
    records: &Vector<Arc<TrajectoryRecord>>,
) -> (bool, bool, bool) {
    let Some(turn) = record.turn else {
        return (false, false, false);
    };
    let previous_turn = row
        .checked_sub(1)
        .and_then(|previous| rows.get(previous))
        .and_then(|previous| ledger_row_turn(&previous, records));
    let next_turn = rows
        .get(row + 1)
        .and_then(|next| ledger_row_turn(&next, records));
    (
        previous_turn != Some(turn),
        previous_turn == Some(turn),
        next_turn == Some(turn),
    )
}

#[derive(Debug)]
struct TimelineModel {
    axis: AxisId,
    domain: DomainRange,
    viewport: DomainRange,
    render_width_px: f64,
    cells: Vec<RenderCell>,
}

impl TimelineModel {
    fn hit_test(&self, lane: TimelineLane, fraction: f64) -> Option<usize> {
        let x_px = fraction.clamp(0.0, 1.0) * self.render_width_px;
        self.cells.iter().rev().find_map(|cell| {
            let end_px = cell.end_px.max(cell.start_px + 1.0);
            (cell.lane == lane && x_px >= cell.start_px && x_px <= end_px)
                .then(|| cell.ids.last().copied())
                .flatten()
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TimelineView {
    viewport: Option<AxisRange>,
    selection: Option<AxisRange>,
    render_width_px: f64,
}

#[derive(Clone, Debug)]
enum TimelineMatches {
    All(usize),
    Filtered(ImHashSet<usize>),
}

impl TimelineMatches {
    fn contains(&self, index: &usize) -> bool {
        match self {
            Self::All(len) => *index < *len,
            Self::Filtered(indices) => indices.contains(index),
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        match self {
            Self::All(len) => *len,
            Self::Filtered(indices) => indices.len(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TimelineLedgerRow {
    Record(usize),
    /// Presentation-only anchor for the short live interval before a canonical request has a
    /// record result. It deliberately has no timeline/search/fold identity.
    RequestBoundary {
        request: usize,
        /// Consecutive request-only boundaries share a zero-height ledger position in DSH and
        /// fan their markers horizontally so retries remain individually selectable.
        run_index: u16,
        /// Only the final boundary at the document tail reserves nine pixels below its marker.
        terminal: bool,
    },
    TurnSummary {
        representative: usize,
        turn: u32,
        first_hidden: usize,
        last_hidden: usize,
        step_ids: ImHashSet<u32>,
        call_count: usize,
    },
    CallsSummary {
        assistant: usize,
        first_tool: usize,
        last_tool: usize,
        tool_names: ImHashSet<String>,
        tools: Arc<str>,
        tools_truncated: bool,
    },
}

impl TimelineLedgerRow {
    fn representative_index(&self) -> usize {
        match self {
            Self::Record(index) => *index,
            Self::RequestBoundary { .. } => usize::MAX,
            Self::TurnSummary { representative, .. } => *representative,
            Self::CallsSummary { assistant, .. } => *assistant,
        }
    }

    fn represents(&self, record_index: usize, records: &Vector<Arc<TrajectoryRecord>>) -> bool {
        match self {
            Self::Record(index) => *index == record_index,
            Self::RequestBoundary { .. } => false,
            Self::TurnSummary {
                turn,
                first_hidden,
                last_hidden,
                ..
            } => {
                (*first_hidden..=*last_hidden).contains(&record_index)
                    && records.get(record_index).is_some_and(|record| {
                        record.turn == Some(*turn) && record.kind != TrajectoryKind::System
                    })
            }
            Self::CallsSummary {
                first_tool,
                last_tool,
                ..
            } => (*first_tool..=*last_tool).contains(&record_index),
        }
    }

    fn intersects(
        &self,
        focused: &HashSet<usize>,
        records: &Vector<Arc<TrajectoryRecord>>,
    ) -> bool {
        match self {
            Self::Record(index) => focused.contains(index),
            Self::RequestBoundary { .. } => false,
            Self::TurnSummary {
                first_hidden,
                last_hidden,
                ..
            }
            | Self::CallsSummary {
                first_tool: first_hidden,
                last_tool: last_hidden,
                ..
            } => (*first_hidden..=*last_hidden)
                .any(|index| focused.contains(&index) && self.represents(index, records)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TimelineRows {
    /// Identity mapping: row N is record N. The overwhelmingly common empty-search/default-filter
    /// state therefore owns no N-element allocation.
    All(usize),
    Projected(Vector<TimelineLedgerRow>),
    WithRequestBoundaries {
        base: Box<TimelineRows>,
        boundaries: Arc<[RequestBoundaryPlacement]>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RequestBoundaryPlacement {
    output_row: usize,
    request: usize,
    run_index: u16,
    terminal: bool,
}

/// Tail cursor for append-only folded projections. A collapsed turn can contain explicit System
/// rows after its summary, so searching backward from the ledger tail is quadratic for alternating
/// System/content streams. Stable row positions make every append independent of prior length.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FoldAppendState {
    tail_turn: Option<u32>,
    first_content: Option<usize>,
    first_content_row: Option<usize>,
    turn_summary_row: Option<usize>,
}

impl FoldAppendState {
    fn from_projection(
        records: &Vector<Arc<TrajectoryRecord>>,
        rows: &TimelineRows,
        collapsed_turns: &HashSet<u32>,
    ) -> Self {
        let Some(last_index) = records.len().checked_sub(1) else {
            return Self::default();
        };
        let Some(turn) = records[last_index].turn else {
            return Self::default();
        };
        if !collapsed_turns.contains(&turn) {
            return Self::default();
        }
        let mut group_start = last_index;
        while group_start > 0 && records[group_start - 1].turn == Some(turn) {
            group_start -= 1;
        }
        let first_content =
            (group_start..=last_index).find(|index| records[*index].kind != TrajectoryKind::System);
        let first_content_row = first_content.and_then(|first| {
            rows.position(|row| matches!(row, TimelineLedgerRow::Record(index) if *index == first))
        });
        let turn_summary_row = first_content_row.and_then(|row| {
            rows.get(row + 1).and_then(|candidate| {
                matches!(candidate, TimelineLedgerRow::TurnSummary { turn: row_turn, .. } if row_turn == turn)
                    .then_some(row + 1)
            })
        });
        Self {
            tail_turn: Some(turn),
            first_content,
            first_content_row,
            turn_summary_row,
        }
    }
}

impl TimelineRows {
    fn len(&self) -> usize {
        match self {
            Self::All(len) => *len,
            Self::Projected(rows) => rows.len(),
            Self::WithRequestBoundaries { base, boundaries } => {
                base.len().saturating_add(boundaries.len())
            }
        }
    }

    fn get(&self, row: usize) -> Option<TimelineLedgerRow> {
        match self {
            Self::All(len) => (row < *len).then_some(TimelineLedgerRow::Record(row)),
            Self::Projected(rows) => rows.get(row).cloned(),
            Self::WithRequestBoundaries { base, boundaries } => {
                match boundaries.binary_search_by_key(&row, |boundary| boundary.output_row) {
                    Ok(index) => {
                        boundaries
                            .get(index)
                            .map(|boundary| TimelineLedgerRow::RequestBoundary {
                                request: boundary.request,
                                run_index: boundary.run_index,
                                terminal: boundary.terminal,
                            })
                    }
                    Err(boundaries_before) => base.get(row.saturating_sub(boundaries_before)),
                }
            }
        }
    }

    fn position(&self, mut predicate: impl FnMut(&TimelineLedgerRow) -> bool) -> Option<usize> {
        (0..self.len()).find(|row| self.get(*row).as_ref().is_some_and(&mut predicate))
    }

    fn with_request_boundaries(self, boundaries: Vec<RequestBoundaryPlacement>) -> Self {
        if boundaries.is_empty() {
            return self;
        }
        Self::WithRequestBoundaries {
            base: Box::new(self),
            boundaries: boundaries.into(),
        }
    }

    fn shares_structure(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::All(left), Self::All(right)) => left == right,
            (Self::Projected(left), Self::Projected(right)) => left.ptr_eq(right),
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
struct TimelineFilterSnapshot {
    matched_cells: TimelineCellMatches,
    rows: TimelineRows,
    fold_controls: TimelineFoldControlSnapshot,
}

#[derive(Clone, Debug, Default)]
struct TimelineFoldControlSnapshot {
    all_turns_collapsed: bool,
    all_assistants_collapsed: bool,
}

#[derive(Clone, Debug)]
struct TimelineFoldControlCache {
    projection_lineage: u64,
    eligibility_revision: u64,
    state_revision: u64,
    snapshot: TimelineFoldControlSnapshot,
}

struct TimelinePaintContext<'a> {
    render_width_px: f64,
    focused_cells: Option<&'a HashSet<usize>>,
    matching: &'a TimelineCellMatches,
    hovered_cell: Option<usize>,
    selected_cell: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InspectorTarget {
    Record(usize),
    Request(usize),
}

#[derive(Clone, Debug)]
enum TimelineCellMatches {
    All,
    Filtered(ImHashSet<usize>),
}

impl TimelineCellMatches {
    fn contains(&self, cell: usize) -> bool {
        match self {
            Self::All => true,
            Self::Filtered(cells) => cells.contains(&cell),
        }
    }
}

#[derive(Debug)]
struct TimelineSearchCache {
    change_revision: u64,
    fold_state_revision: u64,
    fold_eligibility_revision: u64,
    query: String,
    terms: Arc<[String]>,
    matching_indices: TimelineMatches,
    rows: TimelineRows,
    fold_append: FoldAppendState,
    collapsed_turns: HashSet<u32>,
    collapsed_assistants: HashSet<TrajectoryItemId>,
    record_count: usize,
    matched_cells: TimelineCellMatches,
    matched_model_revision: u64,
    #[cfg(test)]
    inspected_records: usize,
    #[cfg(test)]
    materialized_row_rebuilds: usize,
    #[cfg(test)]
    matched_cell_rescans: usize,
}

struct TimelineSearchBuild<'a> {
    change_revision: u64,
    query: &'a str,
    fold_state_revision: u64,
    fold_eligibility_revision: u64,
    collapsed_turns: &'a HashSet<u32>,
    collapsed_assistants: &'a HashSet<TrajectoryItemId>,
}

#[derive(Clone, Debug)]
struct TimelineFocusCache {
    selection: AxisRange,
    record_indices: Arc<HashSet<usize>>,
    focused_cells: Arc<HashSet<usize>>,
    model_revision: u64,
    focused_prefix: Arc<[usize]>,
    focused_non_system_prefix: Arc<[usize]>,
}

impl TimelineFocusCache {
    fn new(
        selection: AxisRange,
        records: &Vector<Arc<TrajectoryRecord>>,
        record_indices: HashSet<usize>,
    ) -> Self {
        let mut focused_prefix = Vec::with_capacity(records.len() + 1);
        let mut focused_non_system_prefix = Vec::with_capacity(records.len() + 1);
        focused_prefix.push(0);
        focused_non_system_prefix.push(0);
        for (index, record) in records.iter().enumerate() {
            let focused = usize::from(record_indices.contains(&index));
            focused_prefix.push(focused_prefix.last().copied().unwrap_or_default() + focused);
            focused_non_system_prefix.push(
                focused_non_system_prefix
                    .last()
                    .copied()
                    .unwrap_or_default()
                    + usize::from(focused != 0 && record.kind != TrajectoryKind::System),
            );
        }
        Self {
            selection,
            record_indices: Arc::new(record_indices),
            focused_cells: Arc::new(HashSet::new()),
            model_revision: u64::MAX,
            focused_prefix: focused_prefix.into(),
            focused_non_system_prefix: focused_non_system_prefix.into(),
        }
    }

    fn sync_model(
        &mut self,
        geometry: &TimelineGeometry,
        model: &TimelineModel,
        model_revision: u64,
    ) {
        if self.model_revision == model_revision {
            return;
        }
        let visible_members = model.cells.iter().map(|cell| cell.ids.len()).sum::<usize>();
        let mut focused_cells = HashSet::new();
        if self.record_indices.len().saturating_mul(8) <= visible_members {
            for record in self.record_indices.iter() {
                if let Some(cell) = geometry.render_cell_for_record(&model.cells, *record) {
                    focused_cells.insert(cell);
                }
            }
        } else {
            for (ordinal, cell) in model.cells.iter().enumerate() {
                if geometry
                    .render_members(cell)
                    .any(|record| self.record_indices.contains(record))
                {
                    focused_cells.insert(ordinal);
                }
            }
        }
        self.focused_cells = Arc::new(focused_cells);
        self.model_revision = model_revision;
    }

    fn intersects(&self, first: usize, last: usize) -> bool {
        Self::prefix_intersects(&self.focused_prefix, first, last)
    }

    fn intersects_non_system(&self, first: usize, last: usize) -> bool {
        Self::prefix_intersects(&self.focused_non_system_prefix, first, last)
    }

    fn prefix_intersects(prefix: &[usize], first: usize, last: usize) -> bool {
        if first > last || first >= prefix.len().saturating_sub(1) {
            return false;
        }
        let end = last.saturating_add(1).min(prefix.len().saturating_sub(1));
        prefix[end] > prefix[first]
    }
}

#[derive(Debug)]
struct TimelineCacheIdentity {
    axis: AxisId,
    change_revision: u64,
}

#[derive(Debug)]
struct RequestRowsCache {
    boundary_revision: u64,
    base: TimelineRows,
    rows: TimelineRows,
}

#[derive(Debug)]
pub(crate) struct TimelineModelCache {
    change_revision: u64,
    record_count: usize,
    viewport: Option<AxisRange>,
    render_width_px: f64,
    geometry: Option<TimelineGeometry>,
    search: Option<TimelineSearchCache>,
    request_rows: Option<RequestRowsCache>,
    fold_controls: Option<TimelineFoldControlCache>,
    model: Option<TimelineModel>,
    model_revision: u64,
    focus: Option<TimelineFocusCache>,
    #[cfg(test)]
    timed_incremental_updates: usize,
}

impl TimelineModelCache {
    fn new(
        identity: TimelineCacheIdentity,
        records: &Vector<std::sync::Arc<TrajectoryRecord>>,
        view: TimelineView,
        mut search: Option<TimelineSearchCache>,
    ) -> Self {
        let TimelineCacheIdentity {
            axis,
            change_revision,
        } = identity;
        let geometry = timeline_geometry_from_iter(records.iter(), axis);
        // `model_revision` is local to one cache instance. A retained content/search cache must
        // never mistake a freshly built LOD model for the prior cache's model with the same local
        // counter value.
        if let Some(search) = &mut search {
            search.matched_model_revision = u64::MAX;
        }
        let mut cache = Self {
            change_revision,
            record_count: records.len(),
            viewport: view.viewport,
            render_width_px: view.render_width_px,
            geometry,
            search,
            request_rows: None,
            fold_controls: None,
            model: None,
            model_revision: 0,
            focus: None,
            #[cfg(test)]
            timed_incremental_updates: 0,
        };
        cache.reproject(cache.resolved_viewport(view.viewport));
        cache.sync_focus(records, view.selection, true);
        cache
    }

    fn geometry_matches(&self, axis: AxisId) -> bool {
        self.geometry
            .as_ref()
            .is_some_and(|geometry| geometry.axis == axis)
    }

    fn projection_matches(&self, document_generation: u64, mode: TimelineMode) -> bool {
        self.geometry.as_ref().is_some_and(|geometry| {
            geometry.axis.document_generation == document_generation && geometry.axis.mode == mode
        })
    }

    fn sync_ranges(&mut self, viewport: Option<AxisRange>, render_width_px: f64) {
        let projection_changed =
            self.viewport != viewport || (self.render_width_px - render_width_px).abs() >= 1.0;
        self.viewport = viewport;
        self.render_width_px = render_width_px;
        if projection_changed {
            self.reproject(self.resolved_viewport(viewport));
        }
    }

    fn sync_sequence_geometry(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
        changes: crate::domain::TrajectoryChanges,
    ) -> bool {
        if self
            .geometry
            .as_ref()
            .is_none_or(|geometry| geometry.axis.mode != TimelineMode::Sequence)
        {
            return false;
        }
        let document_generation = self
            .geometry
            .as_ref()
            .map(|geometry| geometry.axis.document_generation)
            .unwrap_or_default();
        let axis = AxisId {
            document_generation,
            geometry_revision: projection.revision(),
            mode: TimelineMode::Sequence,
        };
        let Some(changed_indices) = changes.geometry_indices() else {
            return false;
        };
        let changed_indices = changed_indices.collect::<Vector<_>>();
        let spans = changed_indices.iter().filter_map(|index| {
            projection
                .records
                .get(*index)
                .map(|record| (*index, timeline_span(*index, record.as_ref())))
        });
        let Some(geometry) = &mut self.geometry else {
            return false;
        };
        let previous_domain = geometry.domain;
        let previous_viewport = self
            .model
            .as_ref()
            .map_or(previous_domain, |model| model.viewport);
        let appended = projection.records.len() > self.record_count;
        self.record_count = projection.records.len();
        if !geometry.update_sequence(axis, projection.records.len(), spans) {
            return false;
        }
        self.change_revision = changes.revision;
        let viewport = if appended && previous_viewport == previous_domain {
            geometry.domain
        } else {
            previous_viewport.clamp_to(geometry.domain)
        };
        if appended {
            // Appending changes the Sequence domain and potentially every projected x position.
            // This is the only compatible Sequence transition that still needs a full LOD pass.
            self.reproject(viewport);
        } else {
            self.update_sequence_model_cells(&changed_indices);
        }
        true
    }

    fn sync_timed_geometry(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
        changes: crate::domain::TrajectoryChanges,
    ) -> bool {
        let Some(mode @ (TimelineMode::Duration | TimelineMode::Actual)) =
            self.geometry.as_ref().map(|geometry| geometry.axis.mode)
        else {
            return false;
        };
        let document_generation = self
            .geometry
            .as_ref()
            .map(|geometry| geometry.axis.document_generation)
            .unwrap_or_default();
        let axis = AxisId {
            document_generation,
            geometry_revision: projection.revision(),
            mode,
        };
        let Some(changed_indices) = changes.geometry_indices() else {
            return false;
        };
        let spans = changed_indices.filter_map(|index| {
            projection
                .records
                .get(index)
                .map(|record| (index, timeline_span(index, record.as_ref())))
        });
        let Some(geometry) = &mut self.geometry else {
            return false;
        };
        let previous_domain = geometry.domain;
        let previous_viewport = self
            .model
            .as_ref()
            .map_or(previous_domain, |model| model.viewport);
        if geometry
            .update_timed(axis, projection.records.len(), spans)
            .is_none()
        {
            return false;
        }
        self.record_count = projection.records.len();
        self.change_revision = changes.revision;
        #[cfg(test)]
        {
            self.timed_incremental_updates = self.timed_incremental_updates.saturating_add(1);
        }
        let viewport = if previous_viewport == previous_domain {
            geometry.domain
        } else {
            previous_viewport.clamp_to(geometry.domain)
        };
        self.reproject(viewport);
        true
    }

    /// Consumes the shared bounded change journal even when geometry did not change. This keeps
    /// text-only streaming receipts from aging the geometry cursor out of the journal and returns
    /// whether selection focus depends on any consumed change.
    fn sync_projection(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
    ) -> Option<bool> {
        let changes = projection.changes_since(self.change_revision)?;
        let geometry_changed = self
            .geometry
            .as_ref()
            .is_none_or(|geometry| geometry.axis.geometry_revision != projection.revision());
        let focus_changed = geometry_changed || changes.search_indices().is_none();
        if geometry_changed {
            let updated = match self.geometry.as_ref().map(|geometry| geometry.axis.mode) {
                Some(TimelineMode::Sequence) => self.sync_sequence_geometry(projection, changes),
                Some(TimelineMode::Duration | TimelineMode::Actual) => {
                    self.sync_timed_geometry(projection, changes)
                }
                None => false,
            };
            updated.then_some(focus_changed)
        } else {
            self.change_revision = changes.revision;
            Some(focus_changed)
        }
    }

    fn update_sequence_model_cells(&mut self, changed_indices: &Vector<usize>) {
        let Some(geometry) = &self.geometry else {
            return;
        };
        let Some(model) = &mut self.model else {
            return;
        };
        model.axis = geometry.axis;
        model.domain = geometry.domain;
        model.viewport = model.viewport.clamp_to(geometry.domain);
        for index in changed_indices {
            let Some(cell_index) = geometry.render_cell_for_record(&model.cells, *index) else {
                continue;
            };
            let Some(cell) = model.cells.get_mut(cell_index) else {
                continue;
            };
            if cell.clustered {
                continue;
            }
            cell.nested = geometry
                .cells
                .get(*index)
                .and_then(|geometry_cell| geometry_cell.nested)
                .map(|range| {
                    let (left, width) = normalized_range(range, model.viewport);
                    (
                        left * model.render_width_px,
                        (left + width) * model.render_width_px,
                    )
                });
        }
    }

    fn reproject(&mut self, viewport: DomainRange) {
        self.model = self.geometry.as_ref().map(|geometry| {
            project_timeline(geometry, viewport, self.render_width_px, self.record_count)
        });
        self.model_revision = self.model_revision.saturating_add(1);
    }

    fn display_selection(&self, selection: Option<AxisRange>) -> Option<AxisRange> {
        self.geometry
            .as_ref()
            .and_then(|geometry| resolved_axis_range(selection, geometry))
    }

    fn resolved_viewport(&self, viewport: Option<AxisRange>) -> DomainRange {
        let Some(geometry) = &self.geometry else {
            return DomainRange::new(0.0, 1.0);
        };
        resolved_axis_range(viewport, geometry).map_or(geometry.domain, |range| range.range)
    }

    fn request_rows(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
        base: TimelineRows,
    ) -> TimelineRows {
        let boundary_revision = projection.request_boundary_revision();
        let reuse = self.request_rows.as_ref().is_some_and(|cache| {
            cache.boundary_revision == boundary_revision && cache.base.shares_structure(&base)
        });
        if !reuse {
            let boundaries = request_boundary_placements(&base, projection);
            let rows = base.clone().with_request_boundaries(boundaries);
            self.request_rows = Some(RequestRowsCache {
                boundary_revision,
                base,
                rows,
            });
        }
        self.request_rows
            .as_ref()
            .expect("request rows were initialized")
            .rows
            .clone()
    }

    fn search_snapshot(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
        query: &str,
        fold_state_revision: u64,
        collapsed_turns: &HashSet<u32>,
        collapsed_assistants: &HashSet<TrajectoryItemId>,
    ) -> TimelineFilterSnapshot {
        let query_active = query.split_whitespace().next().is_some();
        let rebuild = self.search.as_ref().is_none_or(|search| {
            search.query != query
                || (!query_active
                    && (search.fold_state_revision != fold_state_revision
                        || search.fold_eligibility_revision
                            != projection.fold_eligibility_revision()))
                || projection
                    .changes_since(search.change_revision)
                    .is_none_or(|changes| changes.search_indices().is_none())
        });
        let changed = if rebuild {
            self.search = Some(TimelineSearchCache::build(
                projection,
                query,
                fold_state_revision,
                collapsed_turns,
                collapsed_assistants,
            ));
            Vector::new()
        } else {
            self.search
                .as_mut()
                .expect("timeline search cache was checked above")
                .sync_incremental(projection)
        };
        if query_active
            && let Some(search) = &mut self.search
            && search.fold_state_revision != fold_state_revision
        {
            // Search deliberately bypasses folds. Remember the revision without rebuilding the
            // N-record match projection; the canonical folds will be applied when search clears.
            search.fold_state_revision = fold_state_revision;
            search.collapsed_turns = collapsed_turns.clone();
            search.collapsed_assistants = collapsed_assistants.clone();
        }
        let (matched_cells, base_rows) = {
            let search = self
                .search
                .as_mut()
                .expect("timeline search cache was initialized");
            if let (Some(geometry), Some(model)) = (&self.geometry, &self.model) {
                search.sync_model_matches(geometry, model, self.model_revision, &changed);
            }
            (search.matched_cells.clone(), search.rows.clone())
        };
        let rows = if query_active {
            // DSH deliberately excludes request-only presentation boundaries from search.
            base_rows
        } else {
            self.request_rows(projection, base_rows)
        };
        let projection_lineage = projection.projection_lineage();
        let eligibility_revision = projection.fold_eligibility_revision();
        let refresh_fold_controls = self.fold_controls.as_ref().is_none_or(|cache| {
            cache.projection_lineage != projection_lineage
                || cache.eligibility_revision != eligibility_revision
                || cache.state_revision != fold_state_revision
        });
        if refresh_fold_controls {
            let turns = projection.collapsible_turns();
            let assistants = projection.collapsible_assistants();
            self.fold_controls = Some(TimelineFoldControlCache {
                projection_lineage,
                eligibility_revision,
                state_revision: fold_state_revision,
                snapshot: TimelineFoldControlSnapshot {
                    all_turns_collapsed: !turns.is_empty()
                        && turns.iter().all(|turn| collapsed_turns.contains(turn)),
                    all_assistants_collapsed: !assistants.is_empty()
                        && assistants
                            .iter()
                            .all(|assistant| collapsed_assistants.contains(assistant)),
                },
            });
        }
        TimelineFilterSnapshot {
            matched_cells,
            rows,
            fold_controls: self
                .fold_controls
                .as_ref()
                .expect("fold controls were initialized")
                .snapshot
                .clone(),
        }
    }

    fn sync_focus(
        &mut self,
        records: &Vector<Arc<TrajectoryRecord>>,
        selection: Option<AxisRange>,
        force: bool,
    ) {
        let Some(selection) = self.display_selection(selection) else {
            self.focus = None;
            return;
        };
        let Some(geometry) = &self.geometry else {
            self.focus = None;
            return;
        };
        let Some(model) = &self.model else {
            self.focus = None;
            return;
        };
        if !force
            && let Some(focus) = self
                .focus
                .as_mut()
                .filter(|focus| focus.selection == selection)
        {
            focus.sync_model(geometry, model, self.model_revision);
            return;
        }
        let items = geometry.selection(selection).items;
        let mut record_indices = HashSet::with_capacity(items.len());
        for index in items {
            record_indices.insert(index);
        }
        let mut focus = TimelineFocusCache::new(selection, records, record_indices);
        focus.sync_model(geometry, model, self.model_revision);
        self.focus = Some(focus);
    }
}

impl TimelineSearchCache {
    fn build(
        projection: &crate::domain::TrajectoryProjection,
        query: &str,
        fold_state_revision: u64,
        collapsed_turns: &HashSet<u32>,
        collapsed_assistants: &HashSet<TrajectoryItemId>,
    ) -> Self {
        let eligible_turns = projection.collapsible_turns();
        let eligible_assistants = projection.collapsible_assistants();
        let effective_turns = collapsed_turns
            .iter()
            .filter(|turn| eligible_turns.contains(turn))
            .copied()
            .collect::<HashSet<_>>();
        let effective_assistants = collapsed_assistants
            .iter()
            .filter(|assistant| eligible_assistants.contains(*assistant))
            .cloned()
            .collect::<HashSet<_>>();
        Self::build_records_with_folds_using(
            &projection.records,
            TimelineSearchBuild {
                change_revision: projection.change_revision(),
                query,
                fold_state_revision,
                fold_eligibility_revision: projection.fold_eligibility_revision(),
                collapsed_turns: &effective_turns,
                collapsed_assistants: &effective_assistants,
            },
            |index, _, terms| projection.record_matches_terms(index, terms),
        )
    }

    #[cfg(test)]
    fn build_records_with_folds(
        records: &Vector<Arc<TrajectoryRecord>>,
        change_revision: u64,
        query: &str,
        fold_state_revision: u64,
        fold_eligibility_revision: u64,
        collapsed_turns: &HashSet<u32>,
        collapsed_assistants: &HashSet<TrajectoryItemId>,
    ) -> Self {
        Self::build_records_with_folds_using(
            records,
            TimelineSearchBuild {
                change_revision,
                query,
                fold_state_revision,
                fold_eligibility_revision,
                collapsed_turns,
                collapsed_assistants,
            },
            |_, record, terms| record.matches_terms(terms),
        )
    }

    fn build_records_with_folds_using(
        records: &Vector<Arc<TrajectoryRecord>>,
        build: TimelineSearchBuild<'_>,
        mut matches: impl FnMut(usize, &TrajectoryRecord, &[String]) -> bool,
    ) -> Self {
        let TimelineSearchBuild {
            change_revision,
            query,
            fold_state_revision,
            fold_eligibility_revision,
            collapsed_turns,
            collapsed_assistants,
        } = build;
        let record_count = records.len();
        let terms: Arc<[String]> = query
            .split_whitespace()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
            .into();
        let query_active = !terms.is_empty();
        if !query_active && collapsed_turns.is_empty() && collapsed_assistants.is_empty() {
            return Self {
                change_revision,
                fold_state_revision,
                fold_eligibility_revision,
                query: String::new(),
                terms,
                matching_indices: TimelineMatches::All(record_count),
                rows: TimelineRows::All(record_count),
                fold_append: FoldAppendState::default(),
                collapsed_turns: HashSet::new(),
                collapsed_assistants: HashSet::new(),
                record_count,
                matched_cells: TimelineCellMatches::All,
                matched_model_revision: 0,
                #[cfg(test)]
                inspected_records: 0,
                #[cfg(test)]
                materialized_row_rebuilds: 0,
                #[cfg(test)]
                matched_cell_rescans: 0,
            };
        }

        let matching_indices = if !query_active {
            TimelineMatches::All(record_count)
        } else {
            let mut matching = ImHashSet::new();
            for (index, record) in records.iter().enumerate() {
                if matches(index, record, &terms) {
                    matching.insert(index);
                }
            }
            TimelineMatches::Filtered(matching)
        };
        let rows = project_ledger_rows(
            records,
            &matching_indices,
            query_active,
            collapsed_turns,
            collapsed_assistants,
        );
        let fold_append = FoldAppendState::from_projection(records, &rows, collapsed_turns);
        Self {
            change_revision,
            fold_state_revision,
            fold_eligibility_revision,
            query: if query_active {
                query.to_owned()
            } else {
                String::new()
            },
            terms,
            matching_indices,
            rows,
            fold_append,
            collapsed_turns: collapsed_turns.clone(),
            collapsed_assistants: collapsed_assistants.clone(),
            record_count,
            matched_cells: TimelineCellMatches::Filtered(ImHashSet::new()),
            matched_model_revision: 0,
            #[cfg(test)]
            inspected_records: if query_active { record_count } else { 0 },
            #[cfg(test)]
            materialized_row_rebuilds: 1,
            #[cfg(test)]
            matched_cell_rescans: 0,
        }
    }

    #[cfg(test)]
    fn build_records(
        records: &Vector<Arc<TrajectoryRecord>>,
        change_revision: u64,
        query: &str,
        collapse_turns: bool,
        collapse_calls: bool,
    ) -> Self {
        let (turns, assistants) = collapsible_trajectory_groups(records);
        let turns = if collapse_turns {
            turns
        } else {
            HashSet::new()
        };
        let assistants = if collapse_calls {
            assistants
        } else {
            HashSet::new()
        };
        Self::build_records_with_folds(records, change_revision, query, 0, 0, &turns, &assistants)
    }

    fn sync_incremental(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
    ) -> Vector<usize> {
        if self.change_revision == projection.change_revision() {
            self.record_count = projection.records.len();
            if matches!(self.matching_indices, TimelineMatches::All(_)) {
                self.matching_indices = TimelineMatches::All(self.record_count);
            }
            if matches!(self.rows, TimelineRows::All(_)) {
                self.rows = TimelineRows::All(self.record_count);
            }
            return Vector::new();
        }
        let Some(changes) = projection.changes_since(self.change_revision) else {
            let query = self.query.clone();
            let turns = self.collapsed_turns.clone();
            let assistants = self.collapsed_assistants.clone();
            *self = Self::build(
                projection,
                &query,
                self.fold_state_revision,
                &turns,
                &assistants,
            );
            return Vector::new();
        };
        let Some(changed) = changes.search_indices() else {
            let query = self.query.clone();
            let turns = self.collapsed_turns.clone();
            let assistants = self.collapsed_assistants.clone();
            *self = Self::build(
                projection,
                &query,
                self.fold_state_revision,
                &turns,
                &assistants,
            );
            return Vector::new();
        };
        self.sync_changed_records_using(
            &projection.records,
            changes.revision,
            changed,
            |index, _, terms| projection.record_matches_terms(index, terms),
        )
    }

    #[cfg(test)]
    fn sync_changed_records(
        &mut self,
        records: &Vector<Arc<TrajectoryRecord>>,
        change_revision: u64,
        changed: impl IntoIterator<Item = usize>,
    ) -> Vector<usize> {
        self.sync_changed_records_using(records, change_revision, changed, |_, record, terms| {
            record.matches_terms(terms)
        })
    }

    fn sync_changed_records_using(
        &mut self,
        records: &Vector<Arc<TrajectoryRecord>>,
        change_revision: u64,
        changed: impl IntoIterator<Item = usize>,
        mut matches: impl FnMut(usize, &TrajectoryRecord, &[String]) -> bool,
    ) -> Vector<usize> {
        if self.query.is_empty()
            && self.collapsed_turns.is_empty()
            && self.collapsed_assistants.is_empty()
        {
            self.record_count = records.len();
            self.matching_indices = TimelineMatches::All(self.record_count);
            self.rows = TimelineRows::All(self.record_count);
            self.change_revision = change_revision;
            self.matched_cells = TimelineCellMatches::All;
            return Vector::new();
        }
        if self.query.is_empty() {
            self.matching_indices = TimelineMatches::All(records.len());
        }
        let mut changed_matches = Vector::new();
        let mut visited = HashSet::new();
        let structure_changed = records.len() != self.record_count;
        for index in changed {
            if !visited.insert(index) {
                continue;
            }
            let appended = index >= self.record_count;
            // Empty search matches every record. Existing streaming text changes cannot affect
            // either matching or row filtering, so only newly appended records are inspected.
            if self.query.is_empty() && !appended {
                continue;
            }
            let Some(record) = records.get(index) else {
                *self = Self::build_records_with_folds_using(
                    records,
                    TimelineSearchBuild {
                        change_revision,
                        query: &self.query,
                        fold_state_revision: self.fold_state_revision,
                        fold_eligibility_revision: self.fold_eligibility_revision,
                        collapsed_turns: &self.collapsed_turns,
                        collapsed_assistants: &self.collapsed_assistants,
                    },
                    &mut matches,
                );
                return Vector::new();
            };
            #[cfg(test)]
            {
                self.inspected_records = self.inspected_records.saturating_add(1);
            }
            let matched_before = self.matching_indices.contains(&index);
            if let TimelineMatches::Filtered(indices) = &mut self.matching_indices {
                if matches(index, record, &self.terms) {
                    indices.insert(index);
                } else {
                    indices.remove(&index);
                }
            }
            let matched_after = self.matching_indices.contains(&index);
            if matched_before != matched_after {
                changed_matches.push_back(index);
                if let TimelineRows::Projected(rows) = &mut self.rows {
                    if matched_after {
                        sorted_insert_record(rows, index);
                    } else {
                        sorted_remove_record(rows, index);
                    }
                }
            }
        }
        if self.query.is_empty() && structure_changed {
            if records.len() < self.record_count {
                self.rows = project_ledger_rows(
                    records,
                    &self.matching_indices,
                    false,
                    &self.collapsed_turns,
                    &self.collapsed_assistants,
                );
                self.fold_append =
                    FoldAppendState::from_projection(records, &self.rows, &self.collapsed_turns);
                #[cfg(test)]
                {
                    self.materialized_row_rebuilds =
                        self.materialized_row_rebuilds.saturating_add(1);
                }
            } else if let TimelineRows::Projected(rows) = &mut self.rows {
                for index in self.record_count..records.len() {
                    append_projected_record(
                        records,
                        index,
                        &self.collapsed_turns,
                        &self.collapsed_assistants,
                        rows,
                        &mut self.fold_append,
                    );
                }
            }
        }
        self.record_count = records.len();
        if matches!(self.matching_indices, TimelineMatches::All(_)) {
            self.matching_indices = TimelineMatches::All(self.record_count);
        }
        if matches!(self.rows, TimelineRows::All(_)) {
            self.rows = TimelineRows::All(self.record_count);
        }
        self.change_revision = change_revision;
        changed_matches
    }

    fn sync_model_matches(
        &mut self,
        geometry: &TimelineGeometry,
        model: &TimelineModel,
        model_revision: u64,
        changed_matches: &Vector<usize>,
    ) {
        if matches!(self.matching_indices, TimelineMatches::All(_)) {
            self.matched_cells = TimelineCellMatches::All;
            self.matched_model_revision = model_revision;
            return;
        }
        if self.matched_model_revision != model_revision {
            let mut cells = ImHashSet::new();
            if let TimelineMatches::Filtered(indices) = &self.matching_indices {
                for index in indices {
                    let Some(cell) = geometry.render_cell_for_record(&model.cells, *index) else {
                        continue;
                    };
                    cells.insert(cell);
                }
            }
            self.matched_cells = TimelineCellMatches::Filtered(cells);
            self.matched_model_revision = model_revision;
            return;
        }
        let TimelineCellMatches::Filtered(cells) = &mut self.matched_cells else {
            return;
        };
        let affected_cells = changed_matches
            .iter()
            .filter_map(|index| geometry.render_cell_for_record(&model.cells, *index))
            .collect::<HashSet<_>>();
        for cell in affected_cells {
            #[cfg(test)]
            {
                self.matched_cell_rescans = self.matched_cell_rescans.saturating_add(1);
            }
            let matched = model.cells.get(cell).is_some_and(|cell| {
                geometry
                    .render_members(cell)
                    .any(|index| self.matching_indices.contains(index))
            });
            if matched {
                cells.insert(cell);
            } else {
                cells.remove(&cell);
            }
        }
    }
}

fn ledger_row_source_seq(
    row: &TimelineLedgerRow,
    records: &Vector<Arc<TrajectoryRecord>>,
) -> Option<u64> {
    let index = match row {
        TimelineLedgerRow::Record(index) => *index,
        TimelineLedgerRow::TurnSummary { representative, .. } => *representative,
        TimelineLedgerRow::CallsSummary { assistant, .. } => *assistant,
        TimelineLedgerRow::RequestBoundary { .. } => return None,
    };
    records.get(index).map(|record| record.source_seq)
}

fn request_boundary_placements(
    base: &TimelineRows,
    projection: &crate::domain::TrajectoryProjection,
) -> Vec<RequestBoundaryPlacement> {
    let mut pending = projection.unanchored_requests.iter().peekable();
    let mut boundaries = Vec::with_capacity(projection.unanchored_requests.len());
    let mut output_row = 0_usize;

    for base_row in 0..base.len() {
        let mut run_index = 0_usize;
        let next_source = base
            .get(base_row)
            .as_ref()
            .and_then(|row| ledger_row_source_seq(row, &projection.records));
        while let Some(request) = pending.peek()
            && next_source.is_some_and(|source| request.source_seq < source)
        {
            boundaries.push(RequestBoundaryPlacement {
                output_row,
                request: request.request_index,
                run_index: u16::try_from(run_index).unwrap_or(u16::MAX),
                terminal: false,
            });
            pending.next();
            output_row = output_row.saturating_add(1);
            run_index = run_index.saturating_add(1);
        }
        output_row = output_row.saturating_add(1);
    }

    let mut run_index = 0_usize;
    while let Some(request) = pending.next() {
        boundaries.push(RequestBoundaryPlacement {
            output_row,
            request: request.request_index,
            run_index: u16::try_from(run_index).unwrap_or(u16::MAX),
            terminal: pending.peek().is_none(),
        });
        output_row = output_row.saturating_add(1);
        run_index = run_index.saturating_add(1);
    }
    boundaries
}

fn project_ledger_rows(
    records: &Vector<Arc<TrajectoryRecord>>,
    matching: &TimelineMatches,
    query_active: bool,
    collapsed_turns: &HashSet<u32>,
    collapsed_assistants: &HashSet<TrajectoryItemId>,
) -> TimelineRows {
    if query_active {
        return TimelineRows::Projected(
            records
                .iter()
                .enumerate()
                .filter_map(|(index, _)| {
                    matching
                        .contains(&index)
                        .then_some(TimelineLedgerRow::Record(index))
                })
                .collect(),
        );
    }
    if collapsed_turns.is_empty() && collapsed_assistants.is_empty() {
        return TimelineRows::All(records.len());
    }

    let mut rows = Vector::new();
    let mut index = 0;
    while index < records.len() {
        let record = &records[index];
        if let Some(turn) = record.turn
            && collapsed_turns.contains(&turn)
        {
            let mut end = index + 1;
            while end < records.len() && records[end].turn == Some(turn) {
                end += 1;
            }
            let content = (index..end)
                .filter(|candidate| records[*candidate].kind != TrajectoryKind::System)
                .collect::<Vec<_>>();
            if content.len() > 1 {
                let first = content[0];
                let hidden = &content[1..];
                let (step_ids, call_count) = turn_summary_aggregate(
                    records,
                    hidden[0],
                    *hidden.last().expect("hidden turn content is nonempty"),
                );
                for candidate in index..end {
                    if candidate == first {
                        rows.push_back(TimelineLedgerRow::Record(candidate));
                        rows.push_back(TimelineLedgerRow::TurnSummary {
                            representative: first,
                            turn,
                            first_hidden: hidden[0],
                            last_hidden: *hidden.last().expect("hidden turn content is nonempty"),
                            step_ids: step_ids.clone(),
                            call_count,
                        });
                    } else if records[candidate].kind == TrajectoryKind::System {
                        rows.push_back(TimelineLedgerRow::Record(candidate));
                    }
                }
                index = end;
                continue;
            }
        }

        rows.push_back(TimelineLedgerRow::Record(index));
        if record.kind == TrajectoryKind::Assistant && collapsed_assistants.contains(&record.id) {
            let first_tool = index + 1;
            let mut end = first_tool;
            while end < records.len() && records[end].kind == TrajectoryKind::Tool {
                end += 1;
            }
            if end > first_tool {
                let (tool_names, tools, tools_truncated) =
                    tool_summary_aggregate(records, first_tool, end - 1);
                rows.push_back(TimelineLedgerRow::CallsSummary {
                    assistant: index,
                    first_tool,
                    last_tool: end - 1,
                    tool_names,
                    tools,
                    tools_truncated,
                });
                index = end;
                continue;
            }
        }
        index += 1;
    }
    TimelineRows::Projected(rows)
}

fn turn_summary_aggregate(
    records: &Vector<Arc<TrajectoryRecord>>,
    first_hidden: usize,
    last_hidden: usize,
) -> (ImHashSet<u32>, usize) {
    let mut step_ids = ImHashSet::new();
    let mut call_count = 0_usize;
    for index in first_hidden..=last_hidden {
        let record = &records[index];
        if record.kind == TrajectoryKind::System {
            continue;
        }
        if let Some(step) = record.step {
            step_ids.insert(step);
        }
        call_count = call_count.saturating_add(usize::from(record.kind == TrajectoryKind::Tool));
    }
    (step_ids, call_count)
}

fn tool_summary_aggregate(
    records: &Vector<Arc<TrajectoryRecord>>,
    first_tool: usize,
    last_tool: usize,
) -> (ImHashSet<String>, Arc<str>, bool) {
    let mut tool_names = ImHashSet::new();
    let mut tools: Arc<str> = Arc::from("");
    let mut truncated = false;
    for index in first_tool..=last_tool {
        let name = records[index].title.trim();
        if !name.is_empty() && !tool_names.contains(name) {
            tool_names.insert(name.to_owned());
            let (next, next_truncated) = append_tool_summary_preview(&tools, truncated, name);
            tools = next;
            truncated = next_truncated;
        }
    }
    (tool_names, tools, truncated)
}

fn append_tool_summary_preview(
    current: &Arc<str>,
    truncated: bool,
    name: &str,
) -> (Arc<str>, bool) {
    if truncated {
        return (Arc::clone(current), true);
    }
    let separator = if current.is_empty() { "" } else { ", " };
    if current
        .len()
        .saturating_add(separator.len())
        .saturating_add(name.len())
        <= TOOL_SUMMARY_PREVIEW_MAX_BYTES
    {
        return (Arc::from(format!("{current}{separator}{name}")), false);
    }

    let marker = if current.is_empty() { "…" } else { ", …" };
    let mut preview = current.to_string();
    let remaining = TOOL_SUMMARY_PREVIEW_MAX_BYTES.saturating_sub(preview.len() + marker.len());
    if current.is_empty() && remaining > 0 {
        let mut end = remaining.min(name.len());
        while end > 0 && !name.is_char_boundary(end) {
            end -= 1;
        }
        preview.push_str(&name[..end]);
    }
    preview.push_str(marker);
    (Arc::from(preview), true)
}

/// Extends the empty-search folded projection without replaying the full session. A turn summary
/// owns exactly the hidden content interval; system rows remain explicit, and call summaries own
/// only the contiguous tool run after an assistant. This keeps append work independent of the
/// already-materialized session length.
fn append_projected_record(
    records: &Vector<Arc<TrajectoryRecord>>,
    index: usize,
    collapsed_turns: &HashSet<u32>,
    collapsed_assistants: &HashSet<TrajectoryItemId>,
    rows: &mut Vector<TimelineLedgerRow>,
    fold_append: &mut FoldAppendState,
) {
    let record = &records[index];
    let continued_turn =
        record.turn.is_some() && index > 0 && records[index - 1].turn == record.turn;
    if !continued_turn {
        *fold_append = record
            .turn
            .filter(|turn| collapsed_turns.contains(turn))
            .map_or_else(FoldAppendState::default, |turn| FoldAppendState {
                tail_turn: Some(turn),
                ..FoldAppendState::default()
            });
    }

    if let Some(turn) = record.turn
        && collapsed_turns.contains(&turn)
        && continued_turn
    {
        debug_assert_eq!(fold_append.tail_turn, Some(turn));
        if record.kind == TrajectoryKind::System {
            rows.push_back(TimelineLedgerRow::Record(index));
            return;
        }

        if let Some(position) = fold_append.turn_summary_row {
            let Some(TimelineLedgerRow::TurnSummary {
                representative,
                first_hidden,
                mut step_ids,
                call_count,
                ..
            }) = rows.get(position).cloned()
            else {
                unreachable!("the located row is a turn summary");
            };
            if let Some(step) = record.step {
                step_ids.insert(step);
            }
            rows.set(
                position,
                TimelineLedgerRow::TurnSummary {
                    representative,
                    turn,
                    first_hidden,
                    last_hidden: index,
                    step_ids,
                    call_count: call_count
                        .saturating_add(usize::from(record.kind == TrajectoryKind::Tool)),
                },
            );
            return;
        }

        if let (Some(first), Some(first_row)) =
            (fold_append.first_content, fold_append.first_content_row)
        {
            let mut step_ids = ImHashSet::new();
            if let Some(step) = record.step {
                step_ids.insert(step);
            }
            let summary_row = first_row + 1;
            rows.insert(
                summary_row,
                TimelineLedgerRow::TurnSummary {
                    representative: first,
                    turn,
                    first_hidden: index,
                    last_hidden: index,
                    step_ids,
                    call_count: usize::from(record.kind == TrajectoryKind::Tool),
                },
            );
            fold_append.turn_summary_row = Some(summary_row);
            return;
        }

        let row = rows.len();
        rows.push_back(TimelineLedgerRow::Record(index));
        fold_append.first_content = Some(index);
        fold_append.first_content_row = Some(row);
        return;
    }

    if record.kind == TrajectoryKind::Tool {
        if let Some(position) = rows.len().checked_sub(1)
            && let Some(TimelineLedgerRow::CallsSummary {
                assistant,
                first_tool,
                last_tool,
                mut tool_names,
                tools,
                tools_truncated,
            }) = rows.get(position).cloned()
            && last_tool + 1 == index
        {
            let name = record.title.trim();
            let (tools, tools_truncated) = if name.is_empty() || tool_names.contains(name) {
                (tools, tools_truncated)
            } else {
                tool_names.insert(name.to_owned());
                append_tool_summary_preview(&tools, tools_truncated, name)
            };
            rows.set(
                position,
                TimelineLedgerRow::CallsSummary {
                    assistant,
                    first_tool,
                    last_tool: index,
                    tool_names,
                    tools,
                    tools_truncated,
                },
            );
            return;
        }
        if index > 0
            && records[index - 1].kind == TrajectoryKind::Assistant
            && collapsed_assistants.contains(&records[index - 1].id)
            && rows.back() == Some(&TimelineLedgerRow::Record(index - 1))
        {
            let (tool_names, tools, tools_truncated) =
                tool_summary_aggregate(records, index, index);
            rows.push_back(TimelineLedgerRow::CallsSummary {
                assistant: index - 1,
                first_tool: index,
                last_tool: index,
                tool_names,
                tools,
                tools_truncated,
            });
            return;
        }
    }
    let row = rows.len();
    rows.push_back(TimelineLedgerRow::Record(index));
    if fold_append.tail_turn == record.turn
        && record.kind != TrajectoryKind::System
        && fold_append.first_content.is_none()
    {
        fold_append.first_content = Some(index);
        fold_append.first_content_row = Some(row);
    }
}

fn sorted_record_position(
    values: &Vector<TimelineLedgerRow>,
    needle: usize,
) -> Result<usize, usize> {
    let mut left = 0;
    let mut right = values.len();
    while left < right {
        let middle = left + (right - left) / 2;
        match values[middle].representative_index().cmp(&needle) {
            std::cmp::Ordering::Less => left = middle + 1,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => return Ok(middle),
        }
    }
    Err(left)
}

fn sorted_insert_record(values: &mut Vector<TimelineLedgerRow>, value: usize) {
    if let Err(index) = sorted_record_position(values, value) {
        values.insert(index, TimelineLedgerRow::Record(value));
    }
}

fn sorted_remove_record(values: &mut Vector<TimelineLedgerRow>, value: usize) {
    if let Ok(index) = sorted_record_position(values, value) {
        values.remove(index);
    }
}

const LEDGER_SUMMARY_ROW_HEIGHT: f32 = 20.0;

fn trajectory_ledger_row_height(row: &TimelineLedgerRow) -> f32 {
    match row {
        TimelineLedgerRow::Record(_) => metrics::LEDGER_ROW_HEIGHT,
        TimelineLedgerRow::RequestBoundary { terminal, .. } => {
            if *terminal {
                9.0
            } else {
                0.0
            }
        }
        TimelineLedgerRow::TurnSummary { .. } | TimelineLedgerRow::CallsSummary { .. } => {
            LEDGER_SUMMARY_ROW_HEIGHT
        }
    }
}

fn turn_summary_text(step_count: usize, call_count: usize) -> String {
    format!(
        "… {step_count} step{} · {call_count} tool call{}",
        if step_count == 1 { "" } else { "s" },
        if call_count == 1 { "" } else { "s" },
    )
}

fn calls_summary_text(call_count: usize, tools: &str) -> String {
    let names = (!tools.is_empty()).then(|| format!(" · {tools}"));
    format!(
        "… {call_count} tool call{}{}",
        if call_count == 1 { "" } else { "s" },
        names.unwrap_or_default(),
    )
}

fn sync_trajectory_list_state(
    state: &gpui::ListState,
    item_count: usize,
    structure_changed: bool,
    restore: Option<gpui::ListOffset>,
    follow_tail: bool,
) {
    let current_count = state.item_count();
    if structure_changed {
        let offset = state.logical_scroll_top();
        state.reset_with_uniform_height(item_count, px(metrics::LEDGER_ROW_HEIGHT));
        state.scroll_to(offset);
    } else if item_count > current_count {
        state.splice(current_count..current_count, item_count - current_count);
    } else if item_count < current_count {
        state.splice(item_count..current_count, 0);
    }
    if let Some(restore) = restore {
        state.scroll_to(restore);
    } else if follow_tail {
        state.scroll_to(gpui::ListOffset {
            item_ix: item_count,
            offset_in_item: px(0.0),
        });
    }
}

fn aligned_trajectory_list_offset(
    rows: &TimelineRows,
    target: usize,
    viewport_height: f32,
    alignment: f32,
) -> gpui::ListOffset {
    let Some(target_row) = rows.get(target) else {
        return gpui::ListOffset {
            item_ix: rows.len(),
            offset_in_item: px(0.0),
        };
    };
    let target_height = trajectory_ledger_row_height(&target_row);
    let desired_before =
        ((viewport_height - target_height).max(0.0) * alignment.clamp(0.0, 1.0)).max(0.0);
    let mut item_ix = target;
    let mut available_before = 0.0;
    while item_ix > 0 && available_before < desired_before {
        item_ix -= 1;
        if let Some(row) = rows.get(item_ix) {
            available_before += trajectory_ledger_row_height(&row);
        }
    }
    gpui::ListOffset {
        item_ix,
        offset_in_item: px((available_before - desired_before).max(0.0)),
    }
}

impl DesktopApp {
    pub(crate) fn trajectory_panel(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let query_active = !query.is_empty();
        let filter = self.timeline_filter_snapshot(&query);
        let selected = self.core.details.selected.as_ref().and_then(|selected| {
            let trajectory = &self.core.session_view.trajectory;
            match selected {
                DetailsSelection::Record(id) => {
                    trajectory.record_index(id).map(InspectorTarget::Record)
                }
                DetailsSelection::Request(key) => {
                    trajectory.request_index(key).map(InspectorTarget::Request)
                }
            }
        });

        div()
            .relative()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .overflow_hidden()
            .bg(trajectory_palette(cx).background)
            .text_color(trajectory_palette(cx).label_primary)
            .child(self.trajectory_toolbar(&filter.fold_controls, cx))
            .child(self.trajectory_overview(&filter.matched_cells, cx))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h(px(0.0))
                    .overflow_hidden()
                    .child(match selected {
                        Some(target) => self.trajectory_selected_panes(
                            target,
                            &filter.rows,
                            query_active,
                            window,
                            cx,
                        ),
                        None => self
                            .trajectory_ledger(&filter.rows, query_active, cx)
                            .into_any_element(),
                    }),
            )
            .children(self.timeline_drag.is_some().then(|| {
                div()
                    .id("trajectory-timeline-drag-capture")
                    .absolute()
                    .top_0()
                    .right_0()
                    .bottom_0()
                    .left_0()
                    .when(
                        self.timeline_drag.as_ref().is_some_and(|drag| drag.pan),
                        |capture| capture.cursor_grabbing(),
                    )
                    .when(
                        self.timeline_drag.as_ref().is_some_and(|drag| !drag.pan),
                        |capture| capture.cursor_crosshair(),
                    )
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, window, cx| {
                        this.timeline_mouse_move(event, window, cx);
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx);
                        }),
                    )
                    .on_mouse_up(
                        MouseButton::Right,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx);
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx);
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Right,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx);
                        }),
                    )
            }))
            .children(self.trajectory_details_layout.is_dragging().then(|| {
                div()
                    .id("trajectory-details-drag-capture")
                    .absolute()
                    .top_0()
                    .right_0()
                    .bottom_0()
                    .left_0()
                    .cursor_col_resize()
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                        if this
                            .trajectory_details_layout
                            .drag_to(f32::from(event.position.x))
                        {
                            cx.notify();
                        }
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, _: &MouseUpEvent, _, cx| {
                            if this.trajectory_details_layout.end_drag() {
                                cx.notify();
                            }
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, _: &MouseUpEvent, _, cx| {
                            if this.trajectory_details_layout.end_drag() {
                                cx.notify();
                            }
                        }),
                    )
            }))
    }

    fn trajectory_selected_panes(
        &self,
        target: InspectorTarget,
        rows: &TimelineRows,
        query_active: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let mode = self.core.layout.trajectory;
        let layout_generation = self.core.layout_generation;
        let fallback_split_width = self.core.layout.main_width;
        let details_width = self.trajectory_details_layout.details_width(
            mode,
            layout_generation,
            fallback_split_width,
        );
        let split_entity = cx.entity().clone();
        let details_entity = split_entity.clone();
        let ledger = div()
            .min_w(px(0.0))
            .min_h(px(0.0))
            .overflow_hidden()
            .when(mode == TrajectoryMode::Split, |element| {
                element.flex_1().min_w(px(LEDGER_MIN_WIDTH))
            })
            .when(mode == TrajectoryMode::Overlay, |element| {
                element.size_full()
            })
            .child(self.trajectory_ledger(rows, query_active, cx));
        let details = div()
            .id("trajectory-v1-details-pane")
            .relative()
            .flex_none()
            .h_full()
            .w(px(details_width))
            .occlude()
            .when(mode == TrajectoryMode::Overlay, |element| {
                element.absolute().top_0().right_0().bottom_0().shadow_xl()
            })
            .on_prepaint(move |bounds, _, cx| {
                details_entity.update(cx, |this, cx| {
                    if this.core.layout_generation == layout_generation
                        && this
                            .trajectory_details_layout
                            .observe_details_width(layout_generation, f32::from(bounds.size.width))
                    {
                        cx.notify();
                    }
                });
            })
            .child(
                div()
                    .size_full()
                    .overflow_hidden()
                    .child(self.trajectory_details(target, window, cx)),
            )
            .child(self.trajectory_details_resize_handle(cx));

        div()
            .id("trajectory-v1-selected-panes")
            .relative()
            .flex()
            .size_full()
            .min_w(px(0.0))
            .min_h(px(0.0))
            .overflow_hidden()
            .child(ledger)
            .child(details)
            .on_prepaint(move |bounds, _, cx| {
                split_entity.update(cx, |this, cx| {
                    if this.core.layout_generation == layout_generation
                        && this
                            .trajectory_details_layout
                            .observe_split_width(layout_generation, f32::from(bounds.size.width))
                    {
                        cx.notify();
                    }
                });
            })
            .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, _, cx| {
                if this
                    .trajectory_details_layout
                    .drag_to(f32::from(event.position.x))
                {
                    cx.notify();
                }
            }))
            .on_mouse_up(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    if this.trajectory_details_layout.end_drag() {
                        cx.notify();
                    }
                }),
            )
            .on_mouse_up_out(
                MouseButton::Left,
                cx.listener(|this, _: &MouseUpEvent, _, cx| {
                    if this.trajectory_details_layout.end_drag() {
                        cx.notify();
                    }
                }),
            )
            .into_any_element()
    }

    fn trajectory_details_resize_handle(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        let generation = self.core.layout_generation;
        let split_width = self
            .trajectory_details_layout
            .split_width(generation, self.core.layout.main_width);
        let current_width = self
            .trajectory_details_layout
            .measured_details_width(generation)
            .unwrap_or_else(|| {
                self.trajectory_details_layout.details_width(
                    self.core.layout.trajectory,
                    generation,
                    self.core.layout.main_width,
                )
            });
        let max_width =
            (split_width - LEDGER_MIN_WIDTH).clamp(DETAILS_MIN_WIDTH, DETAILS_MAX_WIDTH);
        div()
            .id("trajectory-v1-details-resize-handle")
            .absolute()
            .top_0()
            .bottom_0()
            .left(px(-4.0))
            .w(px(8.0))
            .cursor_col_resize()
            .occlude()
            .tab_index(0)
            .role(Role::Splitter)
            .aria_label("Resize event details")
            .aria_orientation(accesskit::Orientation::Vertical)
            .aria_min_numeric_value(DETAILS_MIN_WIDTH as f64)
            .aria_max_numeric_value(max_width as f64)
            .aria_numeric_value(current_width as f64)
            .aria_numeric_value_step(DETAILS_KEYBOARD_STEP as f64)
            .hover(|style| style.bg(colors.primary.opacity(0.08)))
            .focus(|style| style.bg(colors.primary.opacity(0.12)))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, event: &MouseDownEvent, _, cx| {
                    cx.stop_propagation();
                    if event.click_count >= 2 {
                        if this.trajectory_details_layout.reset() {
                            cx.notify();
                        }
                        return;
                    }
                    let generation = this.core.layout_generation;
                    let split_width = this
                        .trajectory_details_layout
                        .split_width(generation, this.core.layout.main_width);
                    let current_width = this
                        .trajectory_details_layout
                        .measured_details_width(generation)
                        .unwrap_or_else(|| {
                            this.trajectory_details_layout.details_width(
                                this.core.layout.trajectory,
                                generation,
                                this.core.layout.main_width,
                            )
                        });
                    this.trajectory_details_layout.begin_drag(
                        f32::from(event.position.x),
                        current_width,
                        split_width,
                    );
                    cx.notify();
                }),
            )
            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                let delta = match event.keystroke.key.as_str() {
                    "left" => DETAILS_KEYBOARD_STEP,
                    "right" => -DETAILS_KEYBOARD_STEP,
                    _ => return,
                };
                let generation = this.core.layout_generation;
                let split_width = this
                    .trajectory_details_layout
                    .split_width(generation, this.core.layout.main_width);
                let current_width = this
                    .trajectory_details_layout
                    .measured_details_width(generation)
                    .unwrap_or_else(|| {
                        this.trajectory_details_layout.details_width(
                            this.core.layout.trajectory,
                            generation,
                            this.core.layout.main_width,
                        )
                    });
                if this
                    .trajectory_details_layout
                    .step(delta, current_width, split_width)
                {
                    cx.notify();
                }
                cx.stop_propagation();
            }))
    }

    fn trajectory_toolbar(
        &self,
        fold_controls: &TimelineFoldControlSnapshot,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        let actual_duration = self.core.trajectory.mode != TimelineMode::Sequence;
        let all_turns_collapsed = fold_controls.all_turns_collapsed;
        let all_assistants_collapsed = fold_controls.all_assistants_collapsed;
        div()
            .flex()
            .items_center()
            .justify_between()
            .h(px(metrics::LEDGER_TOOLBAR_HEIGHT))
            .px(px(6.0))
            .min_w(px(0.0))
            .overflow_hidden()
            .border_b_1()
            .border_color(colors.border_l2)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(2.0))
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .child(
                        Button::new("toggle-trajectory-duration")
                            .icon(DesktopIconName::Clock)
                            .label("Duration")
                            .xsmall()
                            .compact()
                            .ghost()
                            .text_color(if actual_duration {
                                colors.label_primary
                            } else {
                                colors.label_tertiary
                            })
                            .when(actual_duration, |button| button.bg(colors.hover))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.set_trajectory_actual_duration(!actual_duration, window, cx);
                            })),
                    )
                    .child(
                        Button::new("toggle-trajectory-turns")
                            .label(if all_turns_collapsed {
                                "⊞  Turns"
                            } else {
                                "⊟  Turns"
                            })
                            .xsmall()
                            .compact()
                            .ghost()
                            .text_color(colors.label_tertiary)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                let collapsible =
                                    this.core.session_view.trajectory.collapsible_turns();
                                let mut next_turns = this.core.trajectory.collapsed_turns.clone();
                                next_turns.retain(|turn| collapsible.contains(turn));
                                for turn in collapsible.iter() {
                                    if all_turns_collapsed {
                                        next_turns.remove(turn);
                                    } else {
                                        next_turns.insert(*turn);
                                    }
                                }
                                this.dispatch(
                                    Action::SetTrajectoryTurnsCollapsed(next_turns),
                                    window,
                                    cx,
                                );
                            })),
                    )
                    .child(
                        Button::new("toggle-trajectory-calls")
                            .label(if all_assistants_collapsed {
                                "⊞  Calls"
                            } else {
                                "⊟  Calls"
                            })
                            .xsmall()
                            .compact()
                            .ghost()
                            .text_color(colors.label_tertiary)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                let collapsible =
                                    this.core.session_view.trajectory.collapsible_assistants();
                                let mut next_assistants =
                                    this.core.trajectory.collapsed_assistants.clone();
                                next_assistants.retain(|assistant| collapsible.contains(assistant));
                                for assistant in collapsible.iter() {
                                    if all_assistants_collapsed {
                                        next_assistants.remove(assistant);
                                    } else {
                                        next_assistants.insert(assistant.clone());
                                    }
                                }
                                this.dispatch(
                                    Action::SetTrajectoryAssistantsCollapsed(next_assistants),
                                    window,
                                    cx,
                                );
                            })),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap(px(6.0))
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .child(
                        div().w(px(164.0)).min_w(px(84.0)).child(
                            Input::new(&self.trajectory_search)
                                .with_size(px(22.0))
                                .prefix(IconName::Search)
                                .cleanable(true),
                        ),
                    ),
            )
    }

    fn trajectory_overview(
        &self,
        matching: &TimelineCellMatches,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        let model = cache.as_ref().and_then(|cache| cache.model.as_ref());
        let timeline_empty = model.is_none_or(|model| model.cells.is_empty());
        let turn_boundaries = cache
            .as_ref()
            .and_then(|cache| {
                Some(timeline_turn_boundary_fractions(
                    &self.core.session_view.trajectory.records,
                    cache.geometry.as_ref()?,
                    cache.model.as_ref()?.viewport,
                ))
            })
            .unwrap_or_default();
        let focused_cells = cache
            .as_ref()
            .and_then(|cache| cache.focus.as_ref())
            .map(|focus| Arc::clone(&focus.focused_cells));
        let committed_selection = cache
            .as_ref()
            .and_then(|cache| cache.display_selection(self.core.trajectory.selected_range));
        let entity = cx.entity().clone();
        let display_selection = self
            .timeline_drag
            .as_ref()
            .filter(|drag| !drag.pan)
            .and_then(|drag| {
                model
                    .filter(|model| drag.initial_viewport.axis == model.axis)
                    .map(|model| AxisRange {
                        axis: model.axis,
                        range: DomainRange::new(drag.start_value, drag.current_value)
                            .clamp_to(model.domain),
                    })
            })
            .or(committed_selection);
        let selection = display_selection.and_then(|selection| {
            model.map(|model| normalized_range(selection.range, model.viewport))
        });
        let selection_dragging = self.timeline_drag.as_ref().is_some_and(|drag| !drag.pan);
        let timeline_panning = self.timeline_drag.as_ref().is_some_and(|drag| drag.pan);
        div()
            .flex()
            .h(px(50.0))
            .overflow_hidden()
            .bg(colors.code_background)
            .border_b_1()
            .border_color(colors.border_l2)
            .child(
                div()
                    .relative()
                    .flex_none()
                    .w(px(44.0))
                    .h_full()
                    .pl_1()
                    .pr(px(3.0))
                    .items_end()
                    .overflow_hidden()
                    .border_r_1()
                    .border_color(colors.border_l2)
                    .text_size(px(10.0))
                    .line_height(px(10.0))
                    .text_color(colors.label_caption)
                    .child(timeline_lane_label("Input", TIMELINE_INPUT_TOP))
                    .child(timeline_lane_label("Model", TIMELINE_MODEL_TOP))
                    .child(timeline_lane_label("Tools", TIMELINE_TOOLS_TOP)),
            )
            .child(
                div()
                    .id("trajectory-timeline")
                    .relative()
                    .flex_1()
                    .h_full()
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .tab_index(0)
                    .when(timeline_panning, |timeline| timeline.cursor_grabbing())
                    .when(!timeline_panning, |timeline| timeline.cursor_crosshair())
                    .children(selection.map(|(left, _width)| {
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left_0()
                            .w(relative(left.max(0.0) as f32))
                            .bg(colors.background.opacity(0.58))
                    }))
                    .children(selection.map(|(left, width)| {
                        let right = (left + width).clamp(0.0, 1.0);
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left(relative(right as f32))
                            .w(relative((1.0 - right) as f32))
                            .bg(colors.background.opacity(0.58))
                    }))
                    .children(selection.map(|(left, width)| {
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left(relative(left as f32))
                            .w(relative(width.max(0.002) as f32))
                            .bg(colors.primary.opacity(if selection_dragging {
                                0.18
                            } else {
                                0.12
                            }))
                    }))
                    .children(
                        model
                            .zip(cache.as_ref().and_then(|cache| cache.geometry.as_ref()))
                            .map(|(model, geometry)| {
                                self.timeline_lane(
                                    TimelineLane::Input,
                                    geometry,
                                    model,
                                    focused_cells.as_deref(),
                                    matching,
                                    cx,
                                )
                            }),
                    )
                    .children(
                        model
                            .zip(cache.as_ref().and_then(|cache| cache.geometry.as_ref()))
                            .map(|(model, geometry)| {
                                self.timeline_lane(
                                    TimelineLane::Model,
                                    geometry,
                                    model,
                                    focused_cells.as_deref(),
                                    matching,
                                    cx,
                                )
                            }),
                    )
                    .children(
                        model
                            .zip(cache.as_ref().and_then(|cache| cache.geometry.as_ref()))
                            .map(|(model, geometry)| {
                                self.timeline_lane(
                                    TimelineLane::Tools,
                                    geometry,
                                    model,
                                    focused_cells.as_deref(),
                                    matching,
                                    cx,
                                )
                            }),
                    )
                    .children(turn_boundaries.into_iter().map(|fraction| {
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left(relative(fraction as f32))
                            .w(px(1.0))
                            .bg(colors.border_l2)
                    }))
                    .children(selection.map(|(left, width)| {
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left(relative(left as f32))
                            .w(relative(width.max(0.002) as f32))
                            .when(selection_dragging, |edge| edge.border_l_2().border_r_2())
                            .when(!selection_dragging, |edge| edge.border_l_3().border_r_3())
                            .border_color(colors.primary)
                    }))
                    .children(timeline_empty.then(|| {
                        div()
                            .absolute()
                            .top_0()
                            .right_0()
                            .bottom_0()
                            .left_0()
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_size(px(13.0))
                            .text_color(colors.label_caption)
                            .child("No timing data")
                    }))
                    .children(
                        self.timeline_hover
                            .as_ref()
                            .filter(|hover| {
                                hover.record_id.is_none()
                                    && self.timeline_drag.is_none()
                                    && model.is_some_and(|model| hover.axis == model.axis)
                            })
                            .map(|hover| {
                                div()
                                    .absolute()
                                    .top_0()
                                    .bottom_0()
                                    .left(relative(hover.fraction.clamp(0.0, 1.0) as f32))
                                    .w(px(2.0))
                                    .bg(colors.primary)
                            }),
                    )
                    .on_prepaint(move |bounds, _, cx| {
                        entity.update(cx, |this, _| this.timeline_bounds = Some(bounds));
                    })
                    .on_mouse_down(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseDownEvent, window, cx| {
                            this.timeline_mouse_down(event, false, window, cx)
                        }),
                    )
                    .on_mouse_down(
                        MouseButton::Right,
                        cx.listener(|this, event: &MouseDownEvent, window, cx| {
                            this.timeline_mouse_down(event, true, window, cx)
                        }),
                    )
                    .on_mouse_move(cx.listener(|this, event: &MouseMoveEvent, window, cx| {
                        this.timeline_mouse_move(event, window, cx)
                    }))
                    .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                        if !*hovered {
                            let changed = this.timeline_hover.take().is_some();
                            // The mouse-up-out handler owns gesture completion.
                            // Leaving the hitbox must not cancel a valid drag.
                            if changed {
                                cx.notify();
                            }
                        }
                    }))
                    .on_mouse_up(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx)
                        }),
                    )
                    .on_mouse_up(
                        MouseButton::Right,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx)
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Left,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx)
                        }),
                    )
                    .on_mouse_up_out(
                        MouseButton::Right,
                        cx.listener(|this, event: &MouseUpEvent, window, cx| {
                            this.timeline_mouse_up(event, window, cx)
                        }),
                    )
                    .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, window, cx| {
                        this.timeline_wheel(event, window, cx)
                    }))
                    .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                        if event.keystroke.key.as_str() == "escape"
                            && this.core.trajectory.selected_range.is_some()
                        {
                            this.dispatch(Action::SetTimelineSelection(None), window, cx);
                            cx.stop_propagation();
                        }
                    })),
            )
    }

    fn ensure_timeline_model_cache(&self) {
        let projection = &self.core.session_view.trajectory;
        let revision = projection.revision();
        let change_revision = projection.change_revision();
        let document_generation = projection.projection_lineage();
        let mode = self.core.trajectory.mode;
        let axis = AxisId {
            document_generation,
            geometry_revision: revision,
            mode,
        };
        let viewport = self.core.trajectory.visible_range;
        let selection = self.core.trajectory.selected_range;
        let render_width_px = self
            .timeline_bounds
            .map(|bounds| f64::from(f32::from(bounds.size.width)).max(1.0))
            .unwrap_or(1_500.0);
        let mut cache = self.timeline_model_cache.borrow_mut();
        let mut rebuild = cache
            .as_ref()
            .is_none_or(|cache| !cache.projection_matches(document_generation, mode));
        let mut focus_changed = false;
        if !rebuild {
            match cache
                .as_mut()
                .expect("timeline cache was checked above")
                .sync_projection(projection)
            {
                Some(changed) => focus_changed = changed,
                None => rebuild = true,
            }
        }
        if rebuild
            || cache
                .as_ref()
                .is_none_or(|cache| !cache.geometry_matches(axis))
        {
            let retained_search = cache.take().and_then(|previous| {
                previous
                    .geometry
                    .as_ref()
                    .is_some_and(|geometry| {
                        geometry.axis.document_generation == document_generation
                    })
                    .then_some(previous.search)
                    .flatten()
            });
            *cache = Some(TimelineModelCache::new(
                TimelineCacheIdentity {
                    axis,
                    change_revision,
                },
                &projection.records,
                TimelineView {
                    viewport,
                    selection,
                    render_width_px,
                },
                retained_search,
            ));
            focus_changed = false;
        } else if cache.as_ref().is_some_and(|cache| {
            cache.viewport != viewport || (cache.render_width_px - render_width_px).abs() >= 1.0
        }) {
            cache
                .as_mut()
                .expect("timeline cache was checked above")
                .sync_ranges(viewport, render_width_px);
        }
        if let Some(cache) = cache.as_mut() {
            cache.sync_focus(&projection.records, selection, focus_changed);
        }
    }

    fn timeline_filter_snapshot(&self, query: &str) -> TimelineFilterSnapshot {
        self.ensure_timeline_model_cache();
        self.timeline_model_cache
            .borrow_mut()
            .as_mut()
            .map(|cache| {
                cache.search_snapshot(
                    &self.core.session_view.trajectory,
                    query,
                    self.core.trajectory.fold_revision,
                    &self.core.trajectory.collapsed_turns,
                    &self.core.trajectory.collapsed_assistants,
                )
            })
            .unwrap_or(TimelineFilterSnapshot {
                matched_cells: TimelineCellMatches::All,
                rows: TimelineRows::All(0),
                fold_controls: TimelineFoldControlSnapshot::default(),
            })
    }

    fn with_timeline_model<T>(&self, project: impl FnOnce(&TimelineModel) -> T) -> Option<T> {
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        cache
            .as_ref()
            .and_then(|cache| cache.model.as_ref())
            .map(project)
    }

    fn with_timeline_geometry<T>(&self, project: impl FnOnce(&TimelineGeometry) -> T) -> Option<T> {
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        cache
            .as_ref()
            .and_then(|cache| cache.geometry.as_ref())
            .map(project)
    }

    fn timeline_lane(
        &self,
        lane: TimelineLane,
        geometry: &TimelineGeometry,
        model: &TimelineModel,
        focused_cells: Option<&HashSet<usize>>,
        matching: &TimelineCellMatches,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let hovered = self
            .timeline_hover
            .as_ref()
            .filter(|hover| hover.axis == model.axis)
            .and_then(|hover| hover.record_id.as_ref());
        let selected = self
            .core
            .details
            .selected
            .as_ref()
            .and_then(DetailsSelection::record);
        let cell_for_id = |id: &TrajectoryItemId| {
            self.core
                .session_view
                .trajectory
                .record_index(id)
                .and_then(|index| geometry.render_cell_for_record(&model.cells, index))
        };
        let hovered_cell = hovered.and_then(cell_for_id);
        let selected_cell = selected.and_then(cell_for_id);
        let paint = TimelinePaintContext {
            render_width_px: model.render_width_px,
            focused_cells,
            matching,
            hovered_cell,
            selected_cell,
        };
        let emphasized_cells = [hovered_cell, selected_cell];
        let ordinary = model
            .cells
            .iter()
            .enumerate()
            .filter(move |(ordinal, cell)| {
                cell.lane == lane && !emphasized_cells.contains(&Some(*ordinal))
            });
        let emphasized = emphasized_cells
            .into_iter()
            .flatten()
            .enumerate()
            .filter(|(position, cell_index)| {
                emphasized_cells[..*position]
                    .iter()
                    .all(|prior| prior != &Some(*cell_index))
            })
            .filter_map(|(_, cell_index)| {
                model.cells.get(cell_index).map(|cell| (cell_index, cell))
            })
            .filter(|(_, cell)| cell.lane == lane);
        div()
            .absolute()
            .top(px(timeline_lane_top(lane)))
            .left_0()
            .right_0()
            .h(px(10.0))
            .children(
                ordinary
                    .chain(emphasized)
                    .map(|(ordinal, cell)| self.timeline_block(ordinal, cell, &paint, cx)),
            )
    }

    fn timeline_block(
        &self,
        ordinal: usize,
        cell: &RenderCell,
        paint: &TimelinePaintContext<'_>,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let selected = paint.selected_cell == Some(ordinal);
        let hovered = paint.hovered_cell == Some(ordinal);
        let selected_index = selected
            .then_some(
                self.core
                    .details
                    .selected
                    .as_ref()
                    .and_then(DetailsSelection::record),
            )
            .flatten()
            .and_then(|id| self.core.session_view.trajectory.record_index(id));
        let hovered_index = hovered
            .then(|| {
                self.timeline_hover
                    .as_ref()
                    .and_then(|hover| hover.record_id.as_ref())
            })
            .flatten()
            .and_then(|id| self.core.session_view.trajectory.record_index(id));
        let record_index = hovered_index
            .or(selected_index)
            .or_else(|| cell.ids.last().copied())
            .unwrap_or_default();
        let record = &self.core.session_view.trajectory.records[record_index];
        let focused = paint
            .focused_cells
            .is_none_or(|focused| focused.contains(&ordinal));
        let matched = paint.matching.contains(ordinal);
        let color = record_color(record, colors);
        let mut tooltip = record_tooltip(record);
        if cell.clustered {
            tooltip.push_str(&format!("\n{} items in this range", cell.ids.len()));
        }
        let width_px = paint.render_width_px.max(1.0);
        let left = cell.start_px / width_px;
        let width = ((cell.end_px - cell.start_px) / width_px).max(1.0 / width_px);
        let execution = nested_segment_geometry(cell);
        let emphasized = hovered || selected;
        let opacity = timeline_block_opacity(record.kind, focused, matched, emphasized);
        let background_opacity = if record.kind == TrajectoryKind::Assistant && execution.is_some()
        {
            opacity * 0.54
        } else {
            opacity
        };
        // Search dims unmatched bars, but DSH keeps hover/current outlines fully legible so the
        // focused item never disappears while inspecting a filtered trajectory.
        let ring_opacity = if selected { 1.0 } else { 0.8 };
        div()
            .id(("timeline-record-v2", record.source_seq))
            .absolute()
            .left(relative(left as f32))
            .top(px(TIMELINE_BAR_OFFSET))
            .w(relative(width as f32))
            .h(px(TIMELINE_BAR_HEIGHT))
            .rounded(px(1.0))
            .bg(color.opacity(background_opacity))
            .children(execution.map(|(left, width)| {
                div()
                    .absolute()
                    .top_0()
                    .bottom_0()
                    .left(relative(left as f32))
                    .w(relative(width as f32))
                    .rounded(px(1.0))
                    .bg(if record.kind == TrajectoryKind::Tool {
                        colors.label_primary.opacity(opacity * 0.14)
                    } else {
                        color.opacity(opacity)
                    })
            }))
            .children(emphasized.then(|| {
                div()
                    .absolute()
                    .top(px(-1.0))
                    .bottom(px(-1.0))
                    .left(px(-1.0))
                    .right(px(-1.0))
                    .rounded(px(2.0))
                    .border_1()
                    .border_color(colors.code_background.opacity(ring_opacity))
            }))
            .children(emphasized.then(|| {
                div()
                    .absolute()
                    .top(px(-2.0))
                    .bottom(px(-2.0))
                    .left(px(-2.0))
                    .right(px(-2.0))
                    .rounded(px(3.0))
                    .border_1()
                    .border_color(colors.primary.opacity(ring_opacity))
            }))
            .cursor_pointer()
            .tooltip(move |window, cx| {
                Tooltip::new(tooltip.clone())
                    .text_size(px(11.0))
                    .line_height(px(16.0))
                    .build(window, cx)
            })
            .into_any_element()
    }

    fn sync_trajectory_ledger_list(&self, rows: &TimelineRows, query_active: bool) {
        let structure = (
            self.core.session_view.trajectory.projection_lineage(),
            if query_active {
                0
            } else {
                self.core.trajectory.fold_revision
            },
            if query_active {
                0
            } else {
                self.core
                    .session_view
                    .trajectory
                    .request_boundary_revision()
            },
            query_active,
        );
        let structure_changed =
            self.trajectory_list_structure.replace(Some(structure)) != Some(structure);
        sync_trajectory_list_state(
            &self.trajectory_scroll,
            rows.len(),
            structure_changed,
            self.trajectory_scroll_restore.take(),
            self.trajectory_follow_tail.get(),
        );
    }

    fn trajectory_ledger(
        &self,
        rows: &TimelineRows,
        query_active: bool,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        self.sync_trajectory_ledger_list(rows, query_active);
        let rows = rows.clone();
        let compact = self.trajectory_ledger_is_compact();
        let entity = cx.entity().clone();
        let layout_generation = self.core.layout_generation;
        self.ensure_timeline_model_cache();
        let focus = self
            .timeline_model_cache
            .borrow()
            .as_ref()
            .and_then(|cache| cache.focus.as_ref())
            .cloned();
        let list = gpui::list(
            self.trajectory_scroll.clone(),
            cx.processor(move |this, row_index: usize, _, cx| {
                let row = rows
                    .get(row_index)
                    .expect("trajectory list state must match the projected ledger rows");
                match row {
                    TimelineLedgerRow::Record(index) => this.trajectory_row(
                        index,
                        row_index,
                        &rows,
                        focus.as_ref().map(|focus| focus.record_indices.as_ref()),
                        compact,
                        cx,
                    ),
                    TimelineLedgerRow::RequestBoundary {
                        request,
                        run_index,
                        terminal,
                    } => this.trajectory_pending_request_row(
                        request,
                        run_index,
                        terminal,
                        focus.is_some(),
                        compact,
                        cx,
                    ),
                    TimelineLedgerRow::TurnSummary {
                        representative,
                        turn,
                        first_hidden,
                        last_hidden,
                        step_ids,
                        call_count,
                    } => this.trajectory_turn_summary(
                        representative,
                        turn,
                        first_hidden,
                        last_hidden,
                        step_ids.len(),
                        call_count,
                        focus.as_ref(),
                        compact,
                        cx,
                    ),
                    TimelineLedgerRow::CallsSummary {
                        assistant,
                        first_tool,
                        last_tool,
                        tools,
                        ..
                    } => this.trajectory_calls_summary(
                        assistant,
                        first_tool,
                        last_tool,
                        tools,
                        focus.as_ref(),
                        compact,
                        cx,
                    ),
                }
            }),
        )
        .size_full();
        div()
            .id("trajectory-ledger-v1")
            .size_full()
            .child(list)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _: &MouseDownEvent, window, cx| {
                    this.dispatch(Action::SelectDetails(None), window, cx);
                    this.dispatch(Action::SetTimelineSelection(None), window, cx);
                }),
            )
            .on_prepaint(move |bounds, _, cx| {
                entity.update(cx, |this, cx| {
                    let previous = this.trajectory_ledger_is_compact();
                    this.trajectory_ledger_width = Some((layout_generation, bounds.size.width));
                    if this.trajectory_ledger_is_compact() != previous {
                        cx.notify();
                    }
                });
            })
    }

    fn trajectory_ledger_is_compact(&self) -> bool {
        let measured = self
            .trajectory_ledger_width
            .filter(|(generation, _)| *generation == self.core.layout_generation)
            .map(|(_, width)| f32::from(width));
        let width = measured.unwrap_or_else(|| match self.core.layout.trajectory {
            TrajectoryMode::Split => (self.core.layout.main_width
                - trajectory_details_default_width(self.core.layout.main_width))
            .max(0.0),
            TrajectoryMode::Ledger | TrajectoryMode::Overlay => self.core.layout.main_width,
        });
        width <= COMPACT_LEDGER_MAX_WIDTH
    }

    fn trajectory_row(
        &self,
        index: usize,
        ledger_row: usize,
        rows: &TimelineRows,
        focused: Option<&HashSet<usize>>,
        compact: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.session_view.trajectory.records[index];
        let selected = self
            .core
            .details
            .selected
            .as_ref()
            .and_then(DetailsSelection::record)
            == Some(&record.id);
        let active_turn =
            self.core
                .details
                .selected
                .as_ref()
                .and_then(|selection| match selection {
                    DetailsSelection::Record(id) => self
                        .core
                        .session_view
                        .trajectory
                        .record_by_id(id)
                        .and_then(|record| record.turn),
                    DetailsSelection::Request(key) => self
                        .core
                        .session_view
                        .trajectory
                        .request_by_key(key)
                        .and_then(|request| request.turn),
                });
        let active_turn = record.turn.is_some() && record.turn == active_turn;
        let outside = focused.is_some_and(|focused| !focused.contains(&index));
        let opacity = if outside { 0.24 } else { 1.0 };
        let kind_color = record_color(record, colors);
        let marker_hovered = self.request_marker_hover.as_ref().is_some_and(|hovered| {
            self.core
                .session_view
                .trajectory
                .requests_for_boundary(&record.id)
                .any(|request| &request.key == hovered)
        });
        let request_markers = self
            .core
            .session_view
            .trajectory
            .requests_for_boundary(&record.id)
            .enumerate()
            .map(|(run_index, request)| {
                self.trajectory_request_marker(
                    request,
                    outside,
                    opacity,
                    compact,
                    u16::try_from(run_index).unwrap_or(u16::MAX),
                    cx,
                )
            })
            .collect::<Vec<_>>();
        let (turn_start, _, _) = ledger_record_boundaries(
            rows,
            ledger_row,
            record,
            &self.core.session_view.trajectory.records,
        );
        div()
            .id(("trajectory-record-v1", record.source_seq))
            .relative()
            .flex()
            .items_center()
            .w_full()
            .h(px(metrics::LEDGER_ROW_HEIGHT))
            .pr_3()
            .border_b_1()
            .border_color(colors.border_l1.opacity(opacity))
            .when(selected, |row| row.bg(colors.primary.opacity(0.035)))
            .when(!marker_hovered, |row| row.hover(|row| row.bg(colors.hover)))
            .cursor_pointer()
            .tab_index(0)
            .child(trajectory_event_cell(
                record,
                turn_start,
                compact,
                outside,
                opacity,
                kind_color,
                colors,
                active_turn,
                selected,
            ))
            .children(request_markers)
            .child(trajectory_record_preview(
                record,
                colors,
                opacity,
                self.trajectory_ledger_width
                    .filter(|(generation, _)| *generation == self.core.layout_generation)
                    .map(|(_, width)| tool_request_column_width(f32::from(width), compact)),
            ))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(move |this, event: &MouseDownEvent, window, cx| {
                    // Row gestures must not bubble into the ledger's blank-area deselection.
                    cx.stop_propagation();
                    if event.click_count >= 2 {
                        let collapsible_turns =
                            this.core.session_view.trajectory.collapsible_turns();
                        let collapsible_assistants =
                            this.core.session_view.trajectory.collapsible_assistants();
                        let Some(target) = ledger_double_click_target(
                            &this.core.session_view.trajectory.records[index],
                            turn_start,
                            &this.core.trajectory.collapsed_turns,
                            &collapsible_turns,
                            &collapsible_assistants,
                        ) else {
                            return;
                        };
                        let action = match target {
                            LedgerFoldTarget::Turn(turn) => Action::ToggleTrajectoryTurn(turn),
                            LedgerFoldTarget::Assistant(assistant) => {
                                Action::ToggleTrajectoryAssistant(assistant)
                            }
                        };
                        this.dispatch(action, window, cx);
                        return;
                    }
                    if event.click_count == 1 {
                        this.select_trajectory(
                            index,
                            TrajectorySelectionSource::Ledger,
                            window,
                            cx,
                        );
                    }
                }),
            )
            .on_key_down(
                cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.select_trajectory(
                            index,
                            TrajectorySelectionSource::Ledger,
                            window,
                            cx,
                        );
                    }
                }),
            )
            .into_any_element()
    }

    fn trajectory_request_marker(
        &self,
        request: &TrajectoryRequest,
        outside: bool,
        opacity: f32,
        compact: bool,
        run_index: u16,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let selected = self
            .core
            .details
            .selected
            .as_ref()
            .and_then(DetailsSelection::request)
            == Some(&request.key);
        let number = request.number;
        let key = request.key.clone();
        let keyboard_key = key.clone();
        let hover_key = key.clone();
        let error = matches!(
            request.status,
            ItemStatus::Failed | ItemStatus::Aborted | ItemStatus::Unknown
        );
        let default_color = if error {
            colors.error
        } else {
            colors.label_caption
        }
        .opacity(if outside { 0.18 } else { opacity });
        let hover_color = if error { colors.error } else { colors.primary };
        let marker_label = if request.purpose == TrajectoryRequestPurpose::Compaction {
            format!("Request #{number} · Compaction")
        } else {
            format!("Request #{number}")
        };
        let hover_label = marker_label.clone();
        let group = SharedString::from(format!("trajectory-request-marker-{number}"));
        div()
            .id(("trajectory-request-marker", number))
            .absolute()
            .left(px(
                (if compact { 6.0 } else { 12.0 }) + f32::from(run_index) * 8.0
            ))
            .top(px(-8.0))
            .size(px(16.0))
            .flex()
            .items_center()
            .justify_center()
            .cursor_pointer()
            .tab_index(0)
            .role(Role::Button)
            .aria_label(marker_label)
            .group(group.clone())
            .child(
                div()
                    .size(px(if selected { 8.0 } else { 5.0 }))
                    .rounded_full()
                    .when(selected, |dot| {
                        dot.border_1().border_color(colors.primary).bg(if error {
                            colors.error
                        } else {
                            colors.primary.opacity(0.18)
                        })
                    })
                    .when(!selected, |dot| {
                        dot.bg(default_color)
                            .group_hover(group.clone(), move |dot| dot.bg(hover_color))
                    }),
            )
            .child(
                div()
                    .absolute()
                    .top(px(2.0))
                    .left(px(17.0))
                    .invisible()
                    .group_hover(group, |label| label.visible())
                    .px_1()
                    .h(px(14.0))
                    .flex()
                    .items_center()
                    .rounded(px(2.0))
                    .border_1()
                    .border_color(colors.border_l1)
                    .bg(colors.background)
                    .shadow_sm()
                    .text_size(px(9.0))
                    .text_color(colors.label_secondary)
                    .whitespace_nowrap()
                    .child(hover_label),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, _: &MouseDownEvent, _, cx| cx.stop_propagation()),
            )
            .on_hover(cx.listener(move |this, hovered: &bool, _, cx| {
                let changed = if *hovered {
                    if this.request_marker_hover.as_ref() == Some(&hover_key) {
                        false
                    } else {
                        this.request_marker_hover = Some(hover_key.clone());
                        true
                    }
                } else if this.request_marker_hover.as_ref() == Some(&hover_key) {
                    this.request_marker_hover = None;
                    true
                } else {
                    false
                };
                if changed {
                    cx.notify();
                }
            }))
            .on_click(cx.listener(move |this, _, window, cx| {
                this.dispatch(
                    Action::SelectDetails(Some(DetailsSelection::Request(key.clone()))),
                    window,
                    cx,
                );
                this.dispatch(Action::SetDetailsTab(DetailsTab::Summary), window, cx);
                this.details_scroll
                    .set_offset(gpui::point(px(0.0), px(0.0)));
            }))
            .on_key_down(
                cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.dispatch(
                            Action::SelectDetails(Some(DetailsSelection::Request(
                                keyboard_key.clone(),
                            ))),
                            window,
                            cx,
                        );
                        this.dispatch(Action::SetDetailsTab(DetailsTab::Summary), window, cx);
                        this.details_scroll
                            .set_offset(gpui::point(px(0.0), px(0.0)));
                        cx.stop_propagation();
                    }
                }),
            )
            .into_any_element()
    }

    fn trajectory_pending_request_row(
        &self,
        request_index: usize,
        run_index: u16,
        terminal: bool,
        outside_focus: bool,
        compact: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let request = &self.core.session_view.trajectory.requests[request_index];
        let opacity = if outside_focus { 0.24 } else { 1.0 };
        let marker =
            self.trajectory_request_marker(request, outside_focus, opacity, compact, run_index, cx);
        div()
            .id(("trajectory-pending-request", request.number))
            .relative()
            .w_full()
            .h(px(if terminal { 9.0 } else { 0.0 }))
            .child(marker)
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, _: &MouseDownEvent, _, cx| cx.stop_propagation()),
            )
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn trajectory_turn_summary(
        &self,
        representative: usize,
        turn: u32,
        first_hidden: usize,
        last_hidden: usize,
        step_count: usize,
        call_count: usize,
        focus: Option<&TimelineFocusCache>,
        compact: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let outside =
            focus.is_some_and(|focus| !focus.intersects_non_system(first_hidden, last_hidden));
        let opacity = if outside { 0.24 } else { 1.0 };
        div()
            .id(("trajectory-turn-summary-v1", representative))
            .flex()
            .items_center()
            .w_full()
            .h(px(LEDGER_SUMMARY_ROW_HEIGHT))
            .pr_3()
            .border_b_1()
            .border_color(colors.border_l1.opacity(opacity))
            .hover(|row| row.bg(colors.hover))
            .cursor_pointer()
            .tab_index(0)
            .child(
                div()
                    .flex_none()
                    .w(px(if compact { 50.0 } else { 122.0 }))
                    .h_full(),
            )
            .child(
                div()
                    .flex_1()
                    .pl_2()
                    .min_w(px(0.0))
                    .truncate()
                    .text_xs()
                    .text_color(colors.label_tertiary.opacity(opacity))
                    .child(turn_summary_text(step_count, call_count)),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, _: &MouseDownEvent, _, cx| cx.stop_propagation()),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.dispatch(Action::ToggleTrajectoryTurn(turn), window, cx);
            }))
            .on_key_down(
                cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.dispatch(Action::ToggleTrajectoryTurn(turn), window, cx);
                    }
                }),
            )
            .into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn trajectory_calls_summary(
        &self,
        assistant: usize,
        first_tool: usize,
        last_tool: usize,
        tools: Arc<str>,
        focus: Option<&TimelineFocusCache>,
        compact: bool,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let outside = focus.is_some_and(|focus| !focus.intersects(first_tool, last_tool));
        let opacity = if outside { 0.24 } else { 1.0 };
        let count = last_tool - first_tool + 1;
        let assistant_id = self.core.session_view.trajectory.records[assistant]
            .id
            .clone();
        let assistant_key_id = assistant_id.clone();
        div()
            .id(("trajectory-calls-summary-v1", assistant))
            .flex()
            .items_center()
            .w_full()
            .h(px(LEDGER_SUMMARY_ROW_HEIGHT))
            .pr_3()
            .border_b_1()
            .border_color(colors.border_l1.opacity(opacity))
            .hover(|row| row.bg(colors.hover))
            .cursor_pointer()
            .tab_index(0)
            .child(
                div()
                    .flex_none()
                    .w(px(if compact { 50.0 } else { 122.0 }))
                    .h_full(),
            )
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.0))
                    .pl_2()
                    .truncate()
                    .text_xs()
                    .text_color(colors.label_tertiary.opacity(opacity))
                    .child(calls_summary_text(count, &tools)),
            )
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|_, _: &MouseDownEvent, _, cx| cx.stop_propagation()),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.dispatch(
                    Action::ToggleTrajectoryAssistant(assistant_id.clone()),
                    window,
                    cx,
                );
            }))
            .on_key_down(
                cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.dispatch(
                            Action::ToggleTrajectoryAssistant(assistant_key_id.clone()),
                            window,
                            cx,
                        );
                    }
                }),
            )
            .into_any_element()
    }

    fn trajectory_details(
        &self,
        target: InspectorTarget,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let trajectory = &self.core.session_view.trajectory;
        let (tabs, title, location, status, kind_color, request_header) = match target {
            InspectorTarget::Record(index) => {
                let record = &trajectory.records[index];
                (
                    relevant_record_tabs(record, trajectory.record_details(&record.id)),
                    kind_label(record.kind).to_uppercase(),
                    record_location(record),
                    status_label(record.status),
                    record_color(record, colors),
                    false,
                )
            }
            InspectorTarget::Request(index) => {
                let request = &trajectory.requests[index];
                (
                    relevant_request_tabs(request),
                    format!("Request #{}", request.number),
                    if request.purpose == TrajectoryRequestPurpose::Compaction {
                        format!("Compaction · {}", request_location(request))
                    } else {
                        request_location(request)
                    },
                    status_label(request.status),
                    if matches!(
                        request.status,
                        ItemStatus::Failed | ItemStatus::Aborted | ItemStatus::Unknown
                    ) {
                        colors.error
                    } else {
                        colors.label_secondary
                    },
                    true,
                )
            }
        };
        let available = tabs
            .iter()
            .map(|descriptor| descriptor.tab)
            .collect::<Vec<_>>();
        let active = self.core.details.active_tab(&available);
        let identity = if request_header {
            div()
                .flex()
                .flex_1()
                .min_w(px(0.0))
                .items_center()
                .gap_2()
                .child(
                    div()
                        .flex_none()
                        .size(px(5.0))
                        .rounded_full()
                        .bg(kind_color),
                )
                .child(
                    div()
                        .flex_none()
                        .text_size(px(12.0))
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .child(title),
                )
                .child(
                    div()
                        .min_w(px(0.0))
                        .truncate()
                        .text_size(px(11.0))
                        .text_color(colors.label_tertiary)
                        .child(location),
                )
                .into_any_element()
        } else {
            div()
                .flex()
                .flex_1()
                .min_w(px(0.0))
                .items_center()
                .gap_2()
                .child(
                    div()
                        .flex_none()
                        .px_2()
                        .h(px(19.0))
                        .flex()
                        .items_center()
                        .rounded(px(4.0))
                        .text_size(px(10.0))
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(kind_color)
                        .bg(kind_color.opacity(0.1))
                        .child(title),
                )
                .child(
                    div()
                        .min_w(px(0.0))
                        .truncate()
                        .text_xs()
                        .text_color(colors.label_tertiary)
                        .child(format!("{} · {}", location, status)),
                )
                .into_any_element()
        };
        div()
            .flex()
            .flex_col()
            .size_full()
            .bg(colors.background)
            .border_l_1()
            .border_color(colors.border_l1)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(metrics::DETAILS_HEADER_HEIGHT))
                    .pl_3()
                    .pr_2()
                    .border_b_1()
                    .border_color(colors.border_l2)
                    .child(identity)
                    .child(
                        Button::new("close-trajectory-v1-details")
                            .icon(IconName::Close)
                            .with_size(px(28.0))
                            .ghost()
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::SelectDetails(None), window, cx);
                            })),
                    ),
            )
            .child(
                div()
                    .id("trajectory-detail-tabs-v1")
                    .flex()
                    .items_center()
                    .h(px(34.0))
                    .px_2()
                    .gap(px(1.0))
                    .overflow_x_scroll()
                    .border_b_1()
                    .border_color(colors.border_l2)
                    .children(tabs.into_iter().enumerate().map(|(index, descriptor)| {
                        let tab = descriptor.tab;
                        let selected = tab == active;
                        div()
                            .id(("trajectory-detail-tab-v1", index))
                            .relative()
                            .flex()
                            .flex_none()
                            .items_center()
                            .h_full()
                            .px_2()
                            .cursor_pointer()
                            .text_sm()
                            .text_color(if tab == active {
                                colors.primary
                            } else {
                                colors.label_tertiary
                            })
                            .hover(move |element| element.bg(colors.hover))
                            .child(descriptor.label)
                            .children(selected.then(|| {
                                div()
                                    .absolute()
                                    .left(px(9.0))
                                    .right(px(9.0))
                                    .bottom_0()
                                    .h(px(2.0))
                                    .rounded_t(px(1.0))
                                    .bg(colors.primary)
                            }))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.dispatch(Action::SetDetailsTab(tab), window, cx);
                            }))
                    })),
            )
            .child(
                div()
                    .id("trajectory-details-v1-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.0))
                    .p_4()
                    .gap_4()
                    .track_scroll(&self.details_scroll)
                    .overflow_y_scrollbar()
                    .child(self.trajectory_details_body(target, active, window, cx)),
            )
            .into_any_element()
    }

    fn trajectory_details_body(
        &self,
        target: InspectorTarget,
        tab: DetailsTab,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        match target {
            InspectorTarget::Record(index) => {
                self.trajectory_record_details_body(index, tab, window, cx)
            }
            InspectorTarget::Request(index) => self.trajectory_request_details_body(index, tab, cx),
        }
    }

    fn trajectory_record_details_body(
        &self,
        index: usize,
        tab: DetailsTab,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let trajectory = &self.core.session_view.trajectory;
        let record = &trajectory.records[index];
        let details = trajectory.record_details(&record.id);
        match tab {
            DetailsTab::Timing => self.timing_details(record, colors, cx),
            DetailsTab::SystemPrompt => prompt_snapshot(details).map_or_else(
                || empty_detail("No system prompt in this request", colors),
                |prompt| {
                    self.trajectory_markdown(
                        record,
                        TrajectoryMarkdownSource::SystemPrompt,
                        prompt.instructions(),
                        window,
                        cx,
                    )
                },
            ),
            DetailsTab::Preview => self.trajectory_markdown(
                record,
                TrajectoryMarkdownSource::Preview,
                &record.text,
                window,
                cx,
            ),
            DetailsTab::Payload => {
                code_panel(record.payload.as_deref().unwrap_or_default(), colors)
            }
            DetailsTab::Result => code_panel(&record.text, colors),
            DetailsTab::Raw => code_panel(&record.text, colors),
            DetailsTab::Diff => prompt_diff_panel(details, colors),
            DetailsTab::Tools => prompt_snapshot(details).map_or_else(
                || empty_detail("No tools in this request", colors),
                |prompt| prompt_tools_panel(prompt, colors),
            ),
            DetailsTab::Schema => details
                .and_then(TrajectoryRecordDetails::tool_schema)
                .map_or_else(
                    || empty_detail("Tool schema unavailable", colors),
                    |schema| code_panel(schema, colors),
                ),
            DetailsTab::Options => empty_detail("Options belong to the request", colors),
            DetailsTab::Usage => record.usage.map_or_else(
                || empty_detail("Usage unavailable", colors),
                |usage| {
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(detail_pair(
                            "Input tokens",
                            &usage.input_tokens().to_string(),
                            colors,
                        ))
                        .child(detail_pair(
                            "Output tokens",
                            &usage.total_output_tokens().to_string(),
                            colors,
                        ))
                        .child(detail_pair(
                            "Cached tokens",
                            &usage.cache_read_input_tokens.to_string(),
                            colors,
                        ))
                        .into_any_element()
                },
            ),
            DetailsTab::Summary => {
                let request = match details {
                    Some(TrajectoryRecordDetails::Tool { request_key, .. }) => {
                        trajectory.request_by_key(request_key)
                    }
                    _ => trajectory.request_for_record(&record.id),
                };
                let request_link = request.map(|request| {
                    let key = request.key.clone();
                    div()
                        .id(("trajectory-source-request", request.number))
                        .flex()
                        .justify_between()
                        .gap_4()
                        .text_sm()
                        .cursor_pointer()
                        .child(div().text_color(colors.label_tertiary).child("Request"))
                        .child(
                            div()
                                .text_color(colors.primary)
                                .child(format!("Request #{}", request.number)),
                        )
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.dispatch(
                                Action::SelectDetails(Some(DetailsSelection::Request(key.clone()))),
                                window,
                                cx,
                            );
                            this.dispatch(Action::SetDetailsTab(DetailsTab::Summary), window, cx);
                            this.details_scroll
                                .set_offset(gpui::point(px(0.0), px(0.0)));
                        }))
                });
                let assistant_link = matches!(record.kind, TrajectoryKind::Tool)
                    .then(|| request.and_then(|request| request.result.as_ref()))
                    .flatten()
                    .and_then(|id| trajectory.record_index(id))
                    .map(|assistant_index| {
                        let label = trajectory.records[assistant_index].title.clone();
                        div()
                            .id(("trajectory-tool-assistant", record.source_seq))
                            .flex()
                            .justify_between()
                            .gap_4()
                            .text_sm()
                            .cursor_pointer()
                            .child(div().text_color(colors.label_tertiary).child("Assistant"))
                            .child(div().truncate().text_color(colors.primary).child(label))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.reveal_and_select_trajectory(assistant_index, window, cx);
                            }))
                    });
                let parent_link = match details {
                    Some(TrajectoryRecordDetails::Tool {
                        parent_call_id: Some(parent),
                        ..
                    }) => trajectory.record_index(&TrajectoryItemId::Tool(parent.clone())),
                    _ => None,
                }
                .map(|parent_index| {
                    let label = trajectory.records[parent_index].title.clone();
                    div()
                        .id(("trajectory-parent-tool", record.source_seq))
                        .flex()
                        .justify_between()
                        .gap_4()
                        .text_sm()
                        .cursor_pointer()
                        .child(div().text_color(colors.label_tertiary).child("Parent tool"))
                        .child(div().truncate().text_color(colors.primary).child(label))
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.reveal_and_select_trajectory(parent_index, window, cx);
                        }))
                });
                let request_timing = (record.kind == TrajectoryKind::Assistant)
                    .then_some(request)
                    .flatten()
                    .map(|request| {
                        let key = request.key.clone();
                        let value = timing_duration(record);
                        div()
                            .id(("trajectory-assistant-request-timing", request.number))
                            .mt_2()
                            .p_3()
                            .rounded(px(6.0))
                            .border_1()
                            .border_color(colors.border_l2)
                            .cursor_pointer()
                            .hover(|card| card.bg(colors.hover))
                            .child(
                                div()
                                    .flex()
                                    .justify_between()
                                    .gap_4()
                                    .text_sm()
                                    .child(
                                        div()
                                            .text_color(colors.label_secondary)
                                            .child("Request Timing"),
                                    )
                                    .child(value),
                            )
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.dispatch(
                                    Action::SelectDetails(Some(DetailsSelection::Request(
                                        key.clone(),
                                    ))),
                                    window,
                                    cx,
                                );
                                this.dispatch(
                                    Action::SetDetailsTab(DetailsTab::Timing),
                                    window,
                                    cx,
                                );
                                this.details_scroll
                                    .set_offset(gpui::point(px(0.0), px(0.0)));
                            }))
                    });
                let mut body = div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .child(detail_pair("Status", status_label(record.status), colors))
                    .children(request_link)
                    .children(assistant_link)
                    .children(parent_link)
                    .children(match &record.id {
                        TrajectoryItemId::Tool(call_id) => {
                            Some(detail_pair("Call ID", call_id.as_str(), colors))
                        }
                        _ => None,
                    })
                    .children(record.usage.map(|usage| usage_summary(usage, colors)))
                    .children(request_timing);
                if !record.text.trim().is_empty() {
                    body = body.child(code_panel(&record.text, colors));
                }
                body.into_any_element()
            }
        }
    }

    fn trajectory_request_details_body(
        &self,
        index: usize,
        tab: DetailsTab,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let request = &self.core.session_view.trajectory.requests[index];
        match tab {
            DetailsTab::Summary => {
                let result_link = request.result.as_ref().and_then(|id| {
                    let record_index = self.core.session_view.trajectory.record_index(id)?;
                    let label = self.core.session_view.trajectory.records[record_index]
                        .title
                        .clone();
                    Some(
                        div()
                            .id(("trajectory-request-result", request.number))
                            .flex()
                            .justify_between()
                            .gap_4()
                            .text_sm()
                            .cursor_pointer()
                            .child(div().text_color(colors.label_tertiary).child("Result"))
                            .child(div().truncate().text_color(colors.primary).child(label))
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.reveal_and_select_trajectory(record_index, window, cx);
                                this.dispatch(
                                    Action::SetDetailsTab(DetailsTab::Summary),
                                    window,
                                    cx,
                                );
                            })),
                    )
                });
                let model = request.response_model.as_deref().or_else(|| {
                    request
                        .options
                        .as_deref()
                        .map(|options| options.model.as_ref())
                });
                let mut body = div().flex().flex_col().gap_3().child(detail_pair(
                    "Status",
                    status_label(request.status),
                    colors,
                ));
                if request.purpose == TrajectoryRequestPurpose::Compaction {
                    body = body.child(detail_pair(
                        "Purpose",
                        request_purpose_label(request.purpose),
                        colors,
                    ));
                }
                body = body
                    .child(detail_pair(
                        "Tool calls",
                        &request.tool_call_count.to_string(),
                        colors,
                    ))
                    .children((request.subtool_call_count > 0).then(|| {
                        detail_pair(
                            "Subtool calls",
                            &request.subtool_call_count.to_string(),
                            colors,
                        )
                    }))
                    .children(model.map(|model| detail_pair("Model", model, colors)))
                    .children(result_link)
                    .children(request.error.as_deref().map(|error| {
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(section_title("Error", colors))
                            .child(code_panel(error, colors))
                    }));
                if let Some(options) = request.options.as_ref() {
                    let summary = format!(
                        "{}{}",
                        options.model,
                        options
                            .reasoning_effort
                            .as_deref()
                            .map(|effort| format!(" · {effort}"))
                            .unwrap_or_default()
                    );
                    body = body.child(self.request_detail_preview(
                        request.number,
                        DetailsTab::Options,
                        "Options",
                        summary,
                        colors,
                        cx,
                    ));
                }
                let usage = request
                    .usage
                    .map(|usage| {
                        format!(
                            "{} input · {} output",
                            usage.input_tokens(),
                            usage.total_output_tokens()
                        )
                    })
                    .unwrap_or_else(|| "Unavailable".into());
                body.child(self.request_detail_preview(
                    request.number,
                    DetailsTab::Usage,
                    "Usage",
                    usage,
                    colors,
                    cx,
                ))
                .child(self.request_detail_preview(
                    request.number,
                    DetailsTab::Timing,
                    "Timing",
                    format_elapsed_duration(request.timing.duration_ns()),
                    colors,
                    cx,
                ))
                .into_any_element()
            }
            DetailsTab::Options => request.options.as_deref().map_or_else(
                || empty_detail("Options unavailable for this request", colors),
                |options| request_options_details(options, colors),
            ),
            DetailsTab::Usage => request_usage_details(request, colors),
            DetailsTab::Timing => self.request_timing_details(request, colors, cx),
            DetailsTab::SystemPrompt
            | DetailsTab::Diff
            | DetailsTab::Tools
            | DetailsTab::Preview
            | DetailsTab::Raw
            | DetailsTab::Payload
            | DetailsTab::Result
            | DetailsTab::Schema => empty_detail("This tab is not available for requests", colors),
        }
    }

    fn request_detail_preview(
        &self,
        request_number: u32,
        tab: DetailsTab,
        title: &'static str,
        value: String,
        colors: TrajectoryPalette,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let id = SharedString::from(format!("request-{request_number}-{tab:?}-preview"));
        div()
            .id(id)
            .p_3()
            .rounded(px(6.0))
            .border_1()
            .border_color(colors.border_l2)
            .cursor_pointer()
            .hover(|card| card.bg(colors.hover))
            .child(
                div()
                    .flex()
                    .justify_between()
                    .gap_4()
                    .text_sm()
                    .child(div().text_color(colors.label_secondary).child(title))
                    .child(div().truncate().child(value)),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.dispatch(Action::SetDetailsTab(tab), window, cx);
                this.details_scroll
                    .set_offset(gpui::point(px(0.0), px(0.0)));
            }))
            .into_any_element()
    }

    fn request_timing_details(
        &self,
        request: &TrajectoryRequest,
        colors: TrajectoryPalette,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        if let Some(record) = request
            .result
            .as_ref()
            .and_then(|id| self.core.session_view.trajectory.record_by_id(id))
            .filter(|record| record.kind == TrajectoryKind::Assistant)
        {
            return self.timing_details(record, colors, cx);
        }
        let anchor_started = request
            .anchor
            .as_ref()
            .and_then(|id| self.core.session_view.trajectory.record_by_id(id))
            .and_then(|record| record.timing.started.as_ref());
        let started_at = request.timing.started.as_ref().or(anchor_started);
        let started = started_at
            .map(|time| format_wall(time.wall_time_ms(), self.core.details.unix_time))
            .unwrap_or_else(|| "Not available".into());
        let started_value = if started_at.is_some() {
            div()
                .id(("toggle-request-timing-clock-format", request.number))
                .text_right()
                .cursor_pointer()
                .child(started)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::ToggleDetailsUnixTime, window, cx);
                }))
                .into_any_element()
        } else {
            div().text_right().child(started).into_any_element()
        };
        let mut body = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .justify_between()
                    .gap_4()
                    .text_sm()
                    .child(div().text_color(colors.label_tertiary).child("Started"))
                    .child(started_value),
            )
            .child(detail_pair(
                "Duration",
                &format_elapsed_duration(request.timing.duration_ns()),
                colors,
            ));
        if request.timing.started.is_some() {
            body = body.child(detail_pair(
                "Timing source",
                if request.timing.completed.is_some() {
                    "Session timestamps"
                } else {
                    "Session timestamps (running)"
                },
                colors,
            ));
        }
        body.into_any_element()
    }

    fn trajectory_markdown(
        &self,
        record: &Arc<TrajectoryRecord>,
        source_kind: TrajectoryMarkdownSource,
        source: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let generation = self.core.layout_generation;
        let panel_width = self
            .trajectory_details_layout
            .measured_details_width(generation)
            .unwrap_or_else(|| {
                self.trajectory_details_layout.details_width(
                    self.core.layout.trajectory,
                    generation,
                    self.core.layout.main_width,
                )
            });
        let mut cache = self.trajectory_details_markdown.borrow_mut();
        cache.sync(
            self.core.session_view.trajectory.projection_lineage(),
            &record.id,
            source_kind,
            source,
        );
        dsh_markdown::render_markdown(
            record.source_seq,
            &cache.markdown,
            false,
            &cache.fallback,
            (panel_width - 32.0).max(1.0),
            window,
            cx,
        )
    }

    fn timing_details(
        &self,
        record: &TrajectoryRecord,
        colors: TrajectoryPalette,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let started = format_started(record, self.core.details.unix_time);
        let started_value = if record.timing.started.is_some() {
            div()
                .id("toggle-timing-clock-format")
                .text_right()
                .cursor_pointer()
                .child(started)
                .on_click(cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::ToggleDetailsUnixTime, window, cx);
                }))
                .into_any_element()
        } else {
            div().text_right().child(started).into_any_element()
        };
        let mut body = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .flex()
                    .justify_between()
                    .gap_4()
                    .text_sm()
                    .child(div().text_color(colors.label_tertiary).child("Started"))
                    .child(started_value),
            )
            .child(detail_pair(
                if record.kind == TrajectoryKind::Assistant {
                    "Total duration"
                } else {
                    "Duration"
                },
                &timing_duration(record),
                colors,
            ));
        if record.kind != TrajectoryKind::Assistant {
            let timing_source = if record.timing.duration_ns().is_some() {
                "Session timestamps"
            } else if matches!(record.status, ItemStatus::Pending | ItemStatus::Running) {
                "Session timestamps (running)"
            } else {
                "Not available"
            };
            body = body.child(detail_pair("Timing source", timing_source, colors));
        }
        if record.kind == TrajectoryKind::Assistant {
            body = body
                .child(detail_pair("TTFT", &assistant_ttft(record), colors))
                .child(detail_pair(
                    "Generation",
                    &assistant_generation(record),
                    colors,
                ))
                .child(detail_pair(
                    "Throughput",
                    &assistant_throughput(record),
                    colors,
                ));
        }
        if record.kind == TrajectoryKind::Tool {
            body = body
                .child(section_title("Execution breakdown", colors))
                .child(detail_pair(
                    "Requested",
                    &format_timing_point(
                        record.timing.requested.as_ref(),
                        self.core.details.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Authorization resolved",
                    &format_timing_point(
                        record.timing.authorization_resolved.as_ref(),
                        self.core.details.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Dispatch intended",
                    &format_timing_point(
                        record.timing.dispatch_intended.as_ref(),
                        self.core.details.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Execution started",
                    &format_timing_point(
                        record.timing.execution_started.as_ref(),
                        self.core.details.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Request registration",
                    &format_elapsed_duration(record.timing.request_registration_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Authorization wait",
                    &format_elapsed_duration(record.timing.authorization_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Dispatch wait",
                    &format_elapsed_duration(record.timing.dispatch_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Runner start wait",
                    &format_elapsed_duration(record.timing.runner_start_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Execution duration",
                    &record
                        .timing
                        .execution_ns()
                        .map(|ns| format_elapsed_duration(Some(ns)))
                        .unwrap_or_else(|| execution_missing(record)),
                    colors,
                ))
                .child(detail_pair(
                    "Pre-execution",
                    &format_elapsed_duration(record.timing.pre_execution_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Post/commit wait",
                    &format_elapsed_duration(record.timing.post_execution_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Execution source",
                    "Monotonic execution timestamps",
                    colors,
                ));
        }
        body.into_any_element()
    }

    fn select_trajectory(
        &mut self,
        index: usize,
        source: TrajectorySelectionSource,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(record_id) = self
            .core
            .session_view
            .trajectory
            .records
            .get(index)
            .map(|record| record.id.clone())
        else {
            return;
        };
        self.ensure_timeline_model_cache();
        let clear_selection = {
            let cache = self.timeline_model_cache.borrow();
            let focused = cache
                .as_ref()
                .and_then(|cache| cache.focus.as_ref())
                .map(|focus| focus.record_indices.as_ref());
            should_clear_selection_for_record(source, focused, index)
        };
        self.pan_timeline_to_record(&record_id, window, cx);
        if clear_selection {
            self.dispatch(Action::SetTimelineSelection(None), window, cx);
        }
        self.dispatch(
            Action::SelectDetails(Some(DetailsSelection::Record(record_id))),
            window,
            cx,
        );
        self.details_scroll
            .set_offset(gpui::point(px(0.0), px(0.0)));
        self.scroll_trajectory_to_record(index, cx);
    }

    /// DSH treats inspector hierarchy links as navigation, not as a request to keep the target
    /// hidden inside a folded summary. Expand only the groups that own the target before the
    /// canonical selection/scroll path runs.
    fn reveal_and_select_trajectory(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(record) = self.core.session_view.trajectory.records.get(index) else {
            return;
        };
        let turn = record.turn;
        let record_id = record.id.clone();
        let owning_assistant = match self.core.session_view.trajectory.record_details(&record_id) {
            Some(TrajectoryRecordDetails::Tool { request_key, .. }) => self
                .core
                .session_view
                .trajectory
                .request_by_key(request_key)
                .and_then(|request| request.result.clone()),
            _ => None,
        };
        if let Some(turn) = turn
            && self.core.trajectory.collapsed_turns.contains(&turn)
        {
            self.dispatch(Action::ToggleTrajectoryTurn(turn), window, cx);
        }
        if self
            .core
            .trajectory
            .collapsed_assistants
            .contains(&record_id)
        {
            self.dispatch(Action::ToggleTrajectoryAssistant(record_id), window, cx);
        }
        if let Some(assistant) = owning_assistant
            && self
                .core
                .trajectory
                .collapsed_assistants
                .contains(&assistant)
        {
            self.dispatch(Action::ToggleTrajectoryAssistant(assistant), window, cx);
        }
        self.select_trajectory(index, TrajectorySelectionSource::Ledger, window, cx);
    }

    pub(crate) fn scroll_trajectory_to_record(&self, index: usize, cx: &mut Context<Self>) {
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let rows = self.timeline_filter_snapshot(&query).rows;
        self.sync_trajectory_ledger_list(&rows, !query.is_empty());
        if let Some(row) =
            rows.position(|candidate| self.trajectory_row_represents(candidate, index))
        {
            self.scroll_trajectory_list_row(&rows, row, ScrollStrategy::Center);
            cx.notify();
        }
    }

    fn scroll_trajectory_range_into_view(&self, range: AxisRange, cx: &mut Context<Self>) {
        let Some(focused) = self.with_timeline_geometry(|geometry| geometry.selection(range).items)
        else {
            return;
        };
        if focused.is_empty() {
            return;
        }
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let rows = self.timeline_filter_snapshot(&query).rows;
        self.sync_trajectory_ledger_list(&rows, !query.is_empty());
        let positions = (0..rows.len())
            .filter_map(|position| {
                let row = rows.get(position)?;
                row.intersects(&focused, &self.core.session_view.trajectory.records)
                    .then_some(position)
            })
            .collect::<Vec<_>>();
        let viewport_height = f32::from(self.trajectory_scroll.viewport_bounds().size.height);
        let Some((target, strategy)) = focus_scroll_target(&positions, &rows, viewport_height)
        else {
            return;
        };
        self.scroll_trajectory_list_row(&rows, target, strategy);
        cx.notify();
    }

    fn scroll_trajectory_list_row(
        &self,
        rows: &TimelineRows,
        target: usize,
        strategy: ScrollStrategy,
    ) {
        self.trajectory_follow_tail.set(false);
        match strategy {
            ScrollStrategy::Top => self.trajectory_scroll.scroll_to(gpui::ListOffset {
                item_ix: target,
                offset_in_item: px(0.0),
            }),
            ScrollStrategy::Center | ScrollStrategy::Bottom => {
                let viewport_height =
                    f32::from(self.trajectory_scroll.viewport_bounds().size.height);
                let alignment = if matches!(strategy, ScrollStrategy::Center) {
                    0.5
                } else {
                    1.0
                };
                self.trajectory_scroll
                    .scroll_to(aligned_trajectory_list_offset(
                        rows,
                        target,
                        viewport_height,
                        alignment,
                    ));
            }
            ScrollStrategy::Nearest => self.trajectory_scroll.scroll_to_reveal_item(target),
        }
    }

    fn trajectory_row_represents(&self, row: &TimelineLedgerRow, record_index: usize) -> bool {
        row.represents(record_index, &self.core.session_view.trajectory.records)
    }

    fn pan_timeline_to_record(
        &mut self,
        record_id: &TrajectoryItemId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ensure_timeline_model_cache();
        let viewport = {
            let cache = self.timeline_model_cache.borrow();
            let Some(cache) = cache.as_ref() else { return };
            let Some(record_index) = self.core.session_view.trajectory.record_index(record_id)
            else {
                return;
            };
            let Some(target) = cache
                .geometry
                .as_ref()
                .and_then(|geometry| geometry.range_for(&record_index))
            else {
                return;
            };
            let Some(model) = cache.model.as_ref() else {
                return;
            };
            let viewport = model.viewport.pan_to_reveal(target.range, model.domain);
            (viewport != model.viewport).then_some(AxisRange {
                axis: model.axis,
                range: viewport,
            })
        };
        if let Some(viewport) = viewport {
            self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
        }
    }

    fn timeline_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        pan: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .with_timeline_model(|model| model.cells.is_empty())
            .unwrap_or(true)
        {
            return;
        }
        if event.click_count >= 2 {
            self.timeline_drag = None;
            self.dispatch(Action::SetTimelineSelection(None), window, cx);
            cx.stop_propagation();
            return;
        }
        let Some(value) = self.timeline_value(event.position.x) else {
            return;
        };
        let record_id = (!pan)
            .then(|| self.timeline_record_id(event.position))
            .flatten();
        let Some(initial_viewport) = self.with_timeline_model(|model| AxisRange {
            axis: model.axis,
            range: model.viewport,
        }) else {
            return;
        };
        self.timeline_drag = Some(TimelineDragState {
            pan,
            start_value: value,
            current_value: value,
            start_x: f32::from(event.position.x),
            record_id,
            initial_viewport,
        });
        cx.notify();
        cx.stop_propagation();
    }

    fn timeline_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.update_timeline_hover(event.position, cx);
        let Some(drag) = self.timeline_drag.clone() else {
            return;
        };
        if self.with_timeline_model(|model| model.axis) != Some(drag.initial_viewport.axis) {
            self.timeline_drag = None;
            cx.notify();
            return;
        }
        if drag.pan {
            let Some(value) =
                self.timeline_value_in_viewport(event.position.x, Some(drag.initial_viewport))
            else {
                return;
            };
            let viewport = self.with_timeline_model(|model| AxisRange {
                axis: model.axis,
                range: drag
                    .initial_viewport
                    .range
                    .pan_from(model.domain, drag.start_value, value),
            });
            let Some(viewport) = viewport else {
                return;
            };
            self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
            return;
        }

        let Some((pointer_fraction, edge_fraction)) =
            self.timeline_drag_fractions(event.position.x)
        else {
            return;
        };
        let Some((axis, previous, viewport, current)) = self.with_timeline_model(|model| {
            let viewport = model.viewport.auto_pan(
                model.domain,
                pointer_fraction,
                edge_fraction,
                TIMELINE_EDGE_PAN_STEP_FRACTION,
            );
            (
                model.axis,
                model.viewport,
                viewport,
                viewport.value_at_fraction(pointer_fraction),
            )
        }) else {
            return;
        };
        if let Some(drag) = &mut self.timeline_drag {
            drag.current_value = current;
        }
        let viewport = (viewport != previous).then_some(AxisRange {
            axis,
            range: viewport,
        });
        if let Some(viewport) = viewport {
            self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
        } else {
            cx.notify();
        }
    }

    fn timeline_mouse_up(
        &mut self,
        event: &MouseUpEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(drag) = self.timeline_drag.take() else {
            return;
        };
        if self.with_timeline_model(|model| model.axis) != Some(drag.initial_viewport.axis) {
            cx.notify();
            return;
        }
        let moved = (f32::from(event.position.x) - drag.start_x).abs() >= TIMELINE_CLICK_SLOP;
        if drag.pan {
            let end = self
                .timeline_value_in_viewport(event.position.x, Some(drag.initial_viewport))
                .unwrap_or(drag.start_value);
            let viewport = self.with_timeline_model(|model| AxisRange {
                axis: model.axis,
                range: drag
                    .initial_viewport
                    .range
                    .pan_from(model.domain, drag.start_value, end),
            });
            if moved {
                if let Some(viewport) = viewport {
                    self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
                }
            } else {
                self.dispatch(Action::SetTimelineSelection(None), window, cx);
            }
            cx.notify();
            return;
        }
        let Some(end) = self.timeline_value(event.position.x) else {
            self.cancel_timeline_gesture();
            return;
        };
        if !moved
            && let Some(record_id) = drag.record_id.as_ref()
            && let Some(index) = self.timeline_index_for_id(record_id)
        {
            self.cancel_timeline_gesture();
            self.select_trajectory(index, TrajectorySelectionSource::Timeline, window, cx);
            return;
        }
        let Some((axis, domain, viewport)) =
            self.with_timeline_model(|model| (model.axis, model.domain, model.viewport))
        else {
            self.cancel_timeline_gesture();
            return;
        };
        let span_count = self
            .with_timeline_geometry(|geometry| geometry.cells.len())
            .unwrap_or_default();
        let minimum = minimum_timeline_selection_width(domain, viewport, span_count);
        let selection = AxisRange {
            axis,
            range: DomainRange::new(drag.start_value, end).with_minimum_width(domain, minimum),
        };
        self.dispatch(Action::SetTimelineSelection(Some(selection)), window, cx);
        self.scroll_trajectory_range_into_view(selection, cx);
        if !moved
            && drag.record_id.is_none()
            && let Some(index) = self.nearest_timeline_record(event.position)
            && let Some(record) = self.core.session_view.trajectory.records.get(index)
        {
            self.dispatch(
                Action::SelectDetails(Some(DetailsSelection::Record(record.id.clone()))),
                window,
                cx,
            );
            self.details_scroll
                .set_offset(gpui::point(px(0.0), px(0.0)));
            self.scroll_trajectory_to_record(index, cx);
        }
        cx.notify();
    }

    fn nearest_timeline_record(&self, position: Point<Pixels>) -> Option<usize> {
        let bounds = self.timeline_bounds?;
        let x = f64::from(f32::from(position.x - bounds.origin.x));
        let lane = timeline_lane_at(f32::from(position.y - bounds.origin.y));
        self.with_timeline_model(|model| {
            let nearest = |same_lane: bool| {
                model
                    .cells
                    .iter()
                    .filter(|cell| !same_lane || lane == Some(cell.lane))
                    .filter_map(|cell| {
                        let distance = if x < cell.start_px {
                            cell.start_px - x
                        } else if x > cell.end_px {
                            x - cell.end_px
                        } else {
                            0.0
                        };
                        cell.ids.last().copied().map(|index| (distance, index))
                    })
                    .min_by(|left, right| left.0.total_cmp(&right.0))
                    .map(|(_, index)| index)
            };
            nearest(true).or_else(|| nearest(false))
        })
        .flatten()
    }

    fn timeline_wheel(
        &mut self,
        event: &ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .with_timeline_model(|model| model.cells.is_empty())
            .unwrap_or(true)
        {
            return;
        }
        let Some(anchor) = self.timeline_value(event.position.x) else {
            return;
        };
        let delta = event.delta.pixel_delta(window.line_height()).y;
        if delta == px(0.0) {
            return;
        }
        let factor = if delta > px(0.0) { 1.25 } else { 0.8 };
        let minimum_width = self
            .with_timeline_model(|model| match self.core.trajectory.mode {
                TimelineMode::Sequence => 4.0_f64.min(model.domain.width()),
                TimelineMode::Duration | TimelineMode::Actual => 20.0_f64.min(model.domain.width()),
            })
            .unwrap_or(f64::EPSILON);
        let viewport = self.with_timeline_model(|model| AxisRange {
            axis: model.axis,
            range: model
                .viewport
                .zoom(model.domain, anchor, factor, minimum_width),
        });
        let Some(viewport) = viewport else {
            return;
        };
        self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
        cx.stop_propagation();
    }

    fn timeline_value(&self, x: gpui::Pixels) -> Option<f64> {
        self.timeline_value_in_viewport(x, None)
    }

    fn timeline_drag_fractions(&self, x: gpui::Pixels) -> Option<(f64, f64)> {
        let bounds = self.timeline_bounds?;
        let width = f64::from(f32::from(bounds.size.width));
        if !width.is_finite() || width <= 0.0 {
            return None;
        }
        let local_x = f64::from(f32::from(x - bounds.origin.x));
        let pointer_fraction = (local_x / width).clamp(0.0, 1.0);
        let edge_px =
            (width * TIMELINE_EDGE_PAN_ZONE_FRACTION).clamp(1.0, TIMELINE_EDGE_PAN_MAX_PX);
        Some((pointer_fraction, (edge_px / width).clamp(0.0, 0.5)))
    }

    fn timeline_value_in_viewport(
        &self,
        x: gpui::Pixels,
        viewport: Option<AxisRange>,
    ) -> Option<f64> {
        let bounds = self.timeline_bounds?;
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        let cache = cache.as_ref()?;
        let model = cache.model.as_ref()?;
        let viewport = viewport
            .filter(|range| range.axis == model.axis)
            .map_or(model.viewport, |range| range.range);
        let local_x = f64::from(f32::from(x - bounds.origin.x));
        let width = f64::from(f32::from(bounds.size.width));
        Some(viewport.value_at_fraction(local_x / width.max(1.0)))
    }

    fn update_timeline_hover(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        let Some(bounds) = self.timeline_bounds else {
            return;
        };
        let Some((axis, empty)) =
            self.with_timeline_model(|model| (model.axis, model.cells.is_empty()))
        else {
            return;
        };
        if empty {
            if self.timeline_hover.take().is_some() {
                cx.notify();
            }
            return;
        }
        let fraction =
            f64::from(((position.x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0));
        let record_id = self.timeline_record_id(position);
        let hover = Some(TimelineHoverState {
            axis,
            fraction,
            record_id,
        });
        if self.timeline_hover != hover {
            self.timeline_hover = hover;
            cx.notify();
        }
    }

    fn timeline_record_id(&self, position: Point<Pixels>) -> Option<TrajectoryItemId> {
        let bounds = self.timeline_bounds?;
        let local_y = f32::from(position.y - bounds.origin.y);
        let lane = timeline_lane_at(local_y)?;
        let fraction =
            f64::from(((position.x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0));
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        let cache = cache.as_ref()?;
        let index = cache.model.as_ref()?.hit_test(lane, fraction)?;
        self.core
            .session_view
            .trajectory
            .records
            .get(index)
            .map(|record| record.id.clone())
    }

    fn timeline_index_for_id(&self, id: &TrajectoryItemId) -> Option<usize> {
        self.core.session_view.trajectory.record_index(id)
    }

    pub(crate) fn cancel_timeline_gesture(&mut self) {
        self.timeline_drag = None;
    }
}

#[allow(clippy::too_many_arguments)]
fn trajectory_event_cell(
    record: &TrajectoryRecord,
    turn_start: bool,
    compact: bool,
    outside: bool,
    opacity: f32,
    kind_color: gpui::Hsla,
    colors: TrajectoryPalette,
    active_turn: bool,
    selected: bool,
) -> gpui::AnyElement {
    let kind = kind_label(record.kind).to_uppercase();
    let content = if compact {
        let tooltip = kind.clone();
        let icon = match record.kind {
            TrajectoryKind::System => IconName::Settings,
            TrajectoryKind::User | TrajectoryKind::Steering => IconName::User,
            TrajectoryKind::Context => IconName::Info,
            TrajectoryKind::Assistant => IconName::Bot,
            TrajectoryKind::Tool => IconName::SquareTerminal,
            TrajectoryKind::Compaction => IconName::Minimize,
        };
        div()
            .flex()
            .items_center()
            .justify_end()
            .w_full()
            .pl(px(28.0))
            .pr(px(3.0))
            .child(
                div()
                    .id(("trajectory-kind-icon-v1", record.source_seq))
                    .flex()
                    .items_center()
                    .justify_center()
                    .w(px(19.0))
                    .h(px(19.0))
                    .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
                    .child(
                        Icon::new(icon)
                            .size_3()
                            .text_color(kind_color.opacity(opacity)),
                    ),
            )
            .into_any_element()
    } else {
        div()
            .flex()
            .items_center()
            .justify_end()
            .w_full()
            .pl(px(36.0))
            .pr(px(4.0))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_end()
                    .w(px(76.0))
                    .overflow_hidden()
                    .child(
                        div()
                            .px_2()
                            .h(px(19.0))
                            .flex()
                            .items_center()
                            .rounded(px(4.0))
                            .text_size(px(10.0))
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(kind_color.opacity(opacity))
                            .bg(kind_color.opacity(if outside { 0.035 } else { 0.1 }))
                            .max_w_full()
                            .overflow_hidden()
                            .child(kind),
                    ),
            )
            .into_any_element()
    };
    div()
        .relative()
        .flex()
        .items_center()
        .w(px(if compact { 50.0 } else { 122.0 }))
        .h_full()
        .overflow_hidden()
        .text_xs()
        .text_color(colors.label_caption.opacity(opacity))
        .children(active_turn.then(|| {
            div()
                .absolute()
                .left_0()
                .top_0()
                .bottom_0()
                .w(px(2.0))
                .bg(colors.primary.opacity(opacity * 0.22))
        }))
        .children(selected.then(|| {
            div().absolute().left_0().top_0().bottom_0().w(px(3.0)).bg(
                if matches!(record.status, ItemStatus::Failed | ItemStatus::Aborted) {
                    colors.error
                } else {
                    colors.primary
                },
            )
        }))
        .children((turn_start && record.turn.is_some()).then(|| {
            div()
                .absolute()
                .top_0()
                .left_0()
                .h(px(12.0))
                .px(px(5.0))
                .flex()
                .items_center()
                .rounded_br(px(2.0))
                .bg(if active_turn {
                    colors.primary.opacity(0.08)
                } else {
                    colors.code_background
                })
                .text_size(px(8.0))
                .text_color(if active_turn {
                    colors.primary.opacity(opacity)
                } else {
                    colors.label_tertiary.opacity(opacity)
                })
                .child(if compact {
                    format!("#{}", record.turn.unwrap_or_default())
                } else {
                    format!("Turn {}", record.turn.unwrap_or_default())
                })
        }))
        .child(content)
        .into_any_element()
}

fn timeline_turn_boundary_fractions(
    records: &Vector<Arc<TrajectoryRecord>>,
    geometry: &TimelineGeometry,
    viewport: DomainRange,
) -> Vec<f64> {
    let viewport_width = viewport.width().max(f64::EPSILON);
    let mut boundaries = Vec::new();
    for (index, record) in records.iter().enumerate() {
        let Some(turn) = record.turn else { continue };
        if records
            .get(index.wrapping_sub(1))
            .is_some_and(|previous| previous.turn == Some(turn))
        {
            continue;
        }
        let Some(range) = geometry.range_for(&index) else {
            continue;
        };
        let start = range.range.start;
        if start <= geometry.domain.start || start < viewport.start || start > viewport.end {
            continue;
        }
        let fraction = ((start - viewport.start) / viewport_width).clamp(0.0, 1.0);
        if boundaries
            .last()
            .is_none_or(|previous: &f64| (*previous - fraction).abs() > f64::EPSILON)
        {
            boundaries.push(fraction);
        }
    }
    boundaries
}

fn minimum_timeline_selection_width(
    domain: DomainRange,
    viewport: DomainRange,
    span_count: usize,
) -> f64 {
    if span_count == 0 {
        return viewport.width().min(domain.width()).max(f64::EPSILON);
    }
    (domain.width() / span_count as f64)
        .min(viewport.width())
        .max(f64::EPSILON)
}

fn timeline_block_opacity(
    kind: TrajectoryKind,
    focused: bool,
    matched: bool,
    emphasized: bool,
) -> f32 {
    if !matched {
        0.14
    } else if emphasized {
        1.0
    } else if !focused {
        0.2
    } else if matches!(kind, TrajectoryKind::Assistant | TrajectoryKind::Tool) {
        1.0
    } else {
        0.78
    }
}

fn timeline_lane_label(label: &'static str, top: f32) -> gpui::AnyElement {
    div()
        .absolute()
        .top(px(top + 1.0))
        .left_0()
        .right_0()
        .flex()
        .flex_none()
        .items_center()
        .justify_end()
        .h(px(8.0))
        .w_full()
        .overflow_hidden()
        .child(div().max_w_full().truncate().child(label))
        .into_any_element()
}

fn timeline_lane_top(lane: TimelineLane) -> f32 {
    match lane {
        TimelineLane::Input => TIMELINE_INPUT_TOP,
        TimelineLane::Model => TIMELINE_MODEL_TOP,
        TimelineLane::Tools => TIMELINE_TOOLS_TOP,
    }
}

fn timeline_lane_at(local_y: f32) -> Option<TimelineLane> {
    match local_y {
        7.0..=15.0 => Some(TimelineLane::Input),
        21.0..=29.0 => Some(TimelineLane::Model),
        35.0..=43.0 => Some(TimelineLane::Tools),
        _ => None,
    }
}

fn nested_segment_geometry(cell: &RenderCell) -> Option<(f64, f64)> {
    let (start, end) = cell.nested?;
    let cell_width = (cell.end_px - cell.start_px).max(1.0);
    let local_left = ((start - cell.start_px) / cell_width).clamp(0.0, 1.0);
    let available = 1.0 - local_left;
    if available <= 0.0 {
        return None;
    }
    let local_width = ((end - start) / cell_width).max(0.002).min(available);
    (local_width > 0.0).then_some((local_left, local_width))
}

fn focus_scroll_target(
    positions: &[usize],
    rows: &TimelineRows,
    viewport_height: f32,
) -> Option<(usize, ScrollStrategy)> {
    let first = positions.first().copied()?;
    let focused_height = positions
        .iter()
        .filter_map(|position| rows.get(*position))
        .map(|row| trajectory_ledger_row_height(&row))
        .sum::<f32>();
    Some(
        if focused_height > viewport_height.max(metrics::LEDGER_ROW_HEIGHT) {
            (first, ScrollStrategy::Top)
        } else {
            let midpoint = focused_height / 2.0;
            let mut accumulated = 0.0;
            let target = positions
                .iter()
                .copied()
                .find(|position| {
                    accumulated += rows
                        .get(*position)
                        .as_ref()
                        .map_or(0.0, trajectory_ledger_row_height);
                    accumulated >= midpoint
                })
                .unwrap_or(first);
            (target, ScrollStrategy::Center)
        },
    )
}

fn should_clear_selection_for_record(
    source: TrajectorySelectionSource,
    focused: Option<&HashSet<usize>>,
    record_index: usize,
) -> bool {
    match source {
        // A direct bar click replaces the range focus with an item focus. A ledger click within
        // the focused range is only a details navigation and therefore keeps the DSH dimming and
        // range boundaries intact.
        TrajectorySelectionSource::Timeline => true,
        TrajectorySelectionSource::Ledger => {
            focused.is_some_and(|focused| !focused.contains(&record_index))
        }
    }
}

#[cfg(test)]
fn timeline_model<R: Borrow<TrajectoryRecord>>(
    records: &[R],
    mode: TimelineMode,
    viewport: Option<(f64, f64)>,
) -> Option<TimelineModel> {
    let axis = AxisId {
        document_generation: 1,
        geometry_revision: 1,
        mode,
    };
    let geometry = timeline_geometry(records, axis)?;
    Some(project_timeline(
        &geometry,
        viewport.map(domain_range).unwrap_or(geometry.domain),
        1_500.0,
        records.len(),
    ))
}

#[cfg(test)]
fn timeline_geometry<R: Borrow<TrajectoryRecord>>(
    records: &[R],
    axis: AxisId,
) -> Option<TimelineGeometry> {
    timeline_geometry_from_iter(records.iter(), axis)
}

fn timeline_geometry_from_iter<'a, R: Borrow<TrajectoryRecord> + 'a>(
    records: impl Iterator<Item = &'a R>,
    axis: AxisId,
) -> Option<TimelineGeometry> {
    let spans = records
        .enumerate()
        .map(|(index, record)| timeline_span(index, record.borrow()))
        .collect::<Vec<_>>();
    if spans.is_empty() {
        return None;
    }
    Some(TimelineGeometry::build(axis, spans))
}

fn timeline_span(index: usize, record: &TrajectoryRecord) -> TimelineSpan {
    let started = record.timing.started.as_ref().map(timeline_point);
    let completed = record.timing.completed.as_ref().map(timeline_point);
    let nested = match record.kind {
        TrajectoryKind::Tool => record
            .timing
            .execution_started
            .as_ref()
            .zip(record.timing.execution_finished.as_ref()),
        TrajectoryKind::Assistant => record
            .timing
            .first_token
            .as_ref()
            .zip(record.timing.completed.as_ref()),
        _ => None,
    }
    .map(|(start, end)| (timeline_point(start), timeline_point(end)));
    TimelineSpan {
        id: index,
        lane: record.lane(),
        sequence: index as u64,
        started,
        completed,
        duration_ms: record
            .timing
            .duration_ns()
            .map(|duration| duration as f64 / 1_000_000.0),
        nested,
    }
}

fn timeline_point(time: &EventTimeRef) -> TimelinePoint {
    TimelinePoint {
        wall_ms: time.wall_time_ms() as f64,
        clock_id: time.clock_id().to_owned(),
        monotonic_ns: time.monotonic_ns(),
    }
}

fn project_timeline(
    geometry: &TimelineGeometry,
    viewport: DomainRange,
    width_px: f64,
    _record_count: usize,
) -> TimelineModel {
    let viewport = viewport.clamp_to(geometry.domain);
    let width_px = width_px.max(1.0);
    let cells = geometry.render_model(viewport, width_px, TIMELINE_PRIMITIVE_LIMIT);
    TimelineModel {
        axis: geometry.axis,
        domain: geometry.domain,
        viewport,
        render_width_px: width_px,
        cells,
    }
}

#[cfg(test)]
fn domain_range(range: (f64, f64)) -> DomainRange {
    DomainRange::new(range.0, range.1)
}

fn normalized_range(range: DomainRange, viewport: DomainRange) -> (f64, f64) {
    let span = viewport.width().max(f64::EPSILON);
    let left = ((range.start - viewport.start) / span).clamp(0.0, 1.0);
    let right = ((range.end - viewport.start) / span).clamp(0.0, 1.0);
    (left, (right - left).max(0.002))
}

fn resolved_axis_range(range: Option<AxisRange>, geometry: &TimelineGeometry) -> Option<AxisRange> {
    let range = range?;
    // Geometry revisions describe a newer projection of the same semantic axis, not a different
    // interaction space. DSH keeps a numeric range while live assistant/tool timing arrives, so
    // safely rebind it within the same session lineage and mode and clamp it to the new domain.
    // A different lineage or mode has unrelated coordinates and must still be rejected.
    let compatible_axis = range.axis.document_generation == geometry.axis.document_generation
        && range.axis.mode == geometry.axis.mode;
    compatible_axis.then(|| AxisRange {
        axis: geometry.axis,
        range: range.range.clamp_to(geometry.domain),
    })
}

fn record_color(record: &TrajectoryRecord, colors: TrajectoryPalette) -> gpui::Hsla {
    if matches!(
        record.status,
        ItemStatus::Failed | ItemStatus::Aborted | ItemStatus::Unknown
    ) {
        return colors.error;
    }
    match record.kind {
        TrajectoryKind::System => colors.system_foreground,
        TrajectoryKind::User | TrajectoryKind::Steering => colors.user_foreground,
        TrajectoryKind::Context => colors.context_foreground,
        TrajectoryKind::Assistant | TrajectoryKind::Compaction => colors.assistant_foreground,
        TrajectoryKind::Tool => colors.tool_foreground,
    }
}

fn kind_label(kind: TrajectoryKind) -> &'static str {
    match kind {
        TrajectoryKind::System => "System",
        TrajectoryKind::User => "User",
        TrajectoryKind::Context => "Context",
        TrajectoryKind::Steering => "Steering",
        TrajectoryKind::Assistant => "Assistant",
        TrajectoryKind::Tool => "Tool",
        TrajectoryKind::Compaction => "Compaction",
    }
}

fn status_label(status: ItemStatus) -> &'static str {
    match status {
        ItemStatus::Pending | ItemStatus::Running => "Pending",
        ItemStatus::Completed => "Completed",
        ItemStatus::Failed | ItemStatus::Aborted => "Failed",
        ItemStatus::Denied => "Denied",
        ItemStatus::NotExecuted => "Not executed",
        ItemStatus::Unknown => "Unknown side effects",
    }
}

fn record_location(record: &TrajectoryRecord) -> String {
    match (record.turn, record.step) {
        (Some(turn), Some(step)) => format!("T{turn} · S{step}"),
        (Some(turn), None) => format!("T{turn}"),
        _ => "Session".into(),
    }
}

fn request_location(request: &TrajectoryRequest) -> String {
    match (request.turn, request.step) {
        (Some(turn), Some(step)) => format!("T{turn} · S{step}"),
        (Some(turn), None) => format!("T{turn}"),
        _ => "Session".into(),
    }
}

fn request_purpose_label(purpose: TrajectoryRequestPurpose) -> &'static str {
    match purpose {
        TrajectoryRequestPurpose::Assistant => "Assistant generation",
        TrajectoryRequestPurpose::Compaction => "Compaction",
    }
}

fn row_summary(record: &TrajectoryRecord) -> String {
    let first_line = |value: &str| value.lines().next().unwrap_or_default().trim().to_owned();
    match record.kind {
        TrajectoryKind::Tool => {
            let arguments = record
                .payload
                .as_deref()
                .map(first_line)
                .unwrap_or_default();
            let output = first_line(&record.text);
            match (arguments.is_empty(), output.is_empty()) {
                (false, false) => format!("{} {}  →  {}", record.title, arguments, output),
                (false, true) => format!("{} {}", record.title, arguments),
                (true, false) => format!("{}  →  {}", record.title, output),
                (true, true) => record.title.clone(),
            }
        }
        TrajectoryKind::Assistant if record.text.trim().is_empty() => "(tool call only)".into(),
        _ if record.text.trim().is_empty() => record.title.clone(),
        _ => first_line(&record.text),
    }
}

fn tool_request_column_width(ledger_width: f32, compact: bool) -> f32 {
    let event_width = if compact { 50.0 } else { 122.0 };
    let available = (ledger_width - event_width - 44.0).max(0.0);
    (available * 0.58).clamp(180.0, 480.0).min(available)
}

fn trajectory_record_preview(
    record: &TrajectoryRecord,
    colors: TrajectoryPalette,
    opacity: f32,
    tool_request_width: Option<f32>,
) -> gpui::AnyElement {
    if record.kind != TrajectoryKind::Tool {
        return div()
            .flex_1()
            .min_w(px(0.0))
            .pl_2()
            .truncate()
            .text_sm()
            .text_color(colors.label_primary.opacity(opacity))
            .child(row_summary(record))
            .into_any_element();
    }

    let first_line = |value: &str| value.lines().next().unwrap_or_default().trim().to_owned();
    let request = record
        .payload
        .as_deref()
        .map(first_line)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "—".into());
    let result = if record.text.trim().is_empty() {
        match record.status {
            ItemStatus::Pending | ItemStatus::Running => "Pending".into(),
            _ => "—".into(),
        }
    } else {
        first_line(&record.text)
    };
    div()
        .flex()
        .flex_1()
        .min_w(px(0.0))
        .pl_2()
        .items_center()
        .text_sm()
        .text_color(colors.label_primary.opacity(opacity))
        .child(
            div()
                .when(tool_request_width.is_none(), |column| column.flex_1())
                .when_some(tool_request_width, |column, width| {
                    column.flex_none().w(px(width))
                })
                .min_w(px(0.0))
                .truncate()
                .font_family("monospace")
                .text_color(colors.label_secondary.opacity(opacity))
                .child(request),
        )
        .child(
            div()
                .flex_none()
                .px_2()
                .text_xs()
                .text_color(colors.label_caption.opacity(opacity))
                .child("→"),
        )
        .child(
            div()
                .flex_1()
                .min_w(px(0.0))
                .truncate()
                .text_color(colors.label_primary.opacity(opacity))
                .child(result),
        )
        .into_any_element()
}

fn record_tooltip(record: &TrajectoryRecord) -> String {
    let mut parts = vec![kind_label(record.kind).to_uppercase()];
    if let Some(started) = record.timing.started.as_ref() {
        let started_at = format_clock(started.wall_time_ms());
        parts.push(if let Some(duration) = record.timing.duration_ns() {
            let completed_at = started
                .wall_time_ms()
                .saturating_add((duration / 1_000_000) as i64);
            format!("{started_at} → {}", format_clock(completed_at))
        } else {
            format!("Started {started_at}")
        });
    }
    let mut timing = record
        .timing
        .duration_ns()
        .map(|duration| format!("Total {}", format_elapsed_duration(Some(duration))))
        .into_iter()
        .collect::<Vec<_>>();
    if record.kind == TrajectoryKind::Assistant
        && let (Some(ttft), Some(decoding)) =
            (record.timing.ttft_ns(), record.timing.generation_ns())
    {
        timing.push(format!(
            "TTFT {} · Decoding {}",
            format_elapsed_duration(Some(ttft)),
            format_elapsed_duration(Some(decoding))
        ));
    }
    if !timing.is_empty() {
        parts.push(timing.join(" · "));
    }
    parts.join("\n")
}

fn format_clock(wall_time_ms: i64) -> String {
    let Ok(nanoseconds) = i128::from(wall_time_ms).checked_mul(1_000_000).ok_or(()) else {
        return "Not recorded".into();
    };
    let Ok(timestamp) = OffsetDateTime::from_unix_timestamp_nanos(nanoseconds) else {
        return "Not recorded".into();
    };
    let offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
    timestamp
        .to_offset(offset)
        .format(format_description!(
            "[hour]:[minute]:[second].[subsecond digits:3]"
        ))
        .unwrap_or_else(|_| "Not recorded".into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DetailsTabDescriptor {
    tab: DetailsTab,
    label: &'static str,
}

const fn details_tab(tab: DetailsTab, label: &'static str) -> DetailsTabDescriptor {
    DetailsTabDescriptor { tab, label }
}

fn relevant_record_tabs(
    record: &TrajectoryRecord,
    details: Option<&TrajectoryRecordDetails>,
) -> Vec<DetailsTabDescriptor> {
    match record.kind {
        TrajectoryKind::System => {
            let mut tabs = vec![];
            if matches!(
                details,
                Some(TrajectoryRecordDetails::PromptChange { kind, .. })
                    if *kind != PromptChangeKind::Initial
            ) {
                tabs.push(details_tab(DetailsTab::Diff, "Diff"));
            }
            tabs.push(details_tab(DetailsTab::SystemPrompt, "System Prompt"));
            // DSH keeps the tab contract stable even when the request recorded an empty tool
            // catalog. The body then explains that there are no tools instead of moving tabs as
            // the selected record changes.
            tabs.push(details_tab(DetailsTab::Tools, "Tools"));
            tabs
        }
        TrajectoryKind::User | TrajectoryKind::Steering | TrajectoryKind::Context => vec![
            details_tab(DetailsTab::Summary, "Summary"),
            details_tab(DetailsTab::Preview, "Preview"),
            details_tab(DetailsTab::Raw, "Raw"),
        ],
        TrajectoryKind::Assistant => vec![
            details_tab(DetailsTab::Summary, "Summary"),
            details_tab(DetailsTab::Preview, "Preview"),
            details_tab(DetailsTab::Raw, "Raw"),
        ],
        TrajectoryKind::Tool => {
            let mut tabs = vec![details_tab(DetailsTab::Summary, "Summary")];
            if record
                .payload
                .as_deref()
                .is_some_and(|payload| !payload.is_empty())
            {
                tabs.push(details_tab(DetailsTab::Payload, "Payload"));
            }
            if !record.text.is_empty() {
                tabs.push(details_tab(DetailsTab::Result, "Result"));
            }
            // Schema is a stable Tool tab in DSH. Missing historical/schema data is represented
            // by the empty state inside the tab, not by changing the inspector navigation.
            tabs.push(details_tab(DetailsTab::Schema, "Schema"));
            tabs.push(details_tab(DetailsTab::Timing, "Timing"));
            tabs
        }
        TrajectoryKind::Compaction => vec![
            details_tab(DetailsTab::Summary, "Summary"),
            details_tab(DetailsTab::Raw, "Raw Output"),
        ],
    }
}

fn relevant_request_tabs(request: &TrajectoryRequest) -> Vec<DetailsTabDescriptor> {
    let mut tabs = vec![details_tab(DetailsTab::Summary, "Summary")];
    if request.options.is_some() {
        tabs.push(details_tab(DetailsTab::Options, "Options"));
    }
    tabs.push(details_tab(DetailsTab::Usage, "Usage"));
    tabs.push(details_tab(DetailsTab::Timing, "Timing"));
    tabs
}

fn prompt_snapshot(details: Option<&TrajectoryRecordDetails>) -> Option<&PromptSnapshot> {
    match details? {
        TrajectoryRecordDetails::PromptChange { current, .. } => Some(current),
        TrajectoryRecordDetails::Tool { prompt, .. } => Some(prompt),
    }
}

fn prompt_diff_panel(
    details: Option<&TrajectoryRecordDetails>,
    colors: TrajectoryPalette,
) -> gpui::AnyElement {
    let Some(TrajectoryRecordDetails::PromptChange {
        kind,
        current,
        previous: Some(previous),
    }) = details
    else {
        return empty_detail("No prompt diff recorded", colors);
    };
    let mut body = div().flex().flex_col().gap_3();
    if matches!(
        kind,
        PromptChangeKind::System | PromptChangeKind::SystemAndTools
    ) {
        body = body
            .child(section_title("Previous system prompt", colors))
            .child(code_panel(previous.instructions(), colors))
            .child(section_title("Current system prompt", colors))
            .child(code_panel(current.instructions(), colors));
    }
    if matches!(
        kind,
        PromptChangeKind::Tools | PromptChangeKind::SystemAndTools
    ) {
        body = body
            .child(section_title("Previous tools", colors))
            .child(code_panel(previous.tools_json(), colors))
            .child(section_title("Current tools", colors))
            .child(code_panel(current.tools_json(), colors));
    }
    body.into_any_element()
}

fn prompt_tools_panel(prompt: &PromptSnapshot, colors: TrajectoryPalette) -> gpui::AnyElement {
    if prompt.tool_count() == 0 {
        return empty_detail("No tools in this request", colors);
    }
    div()
        .flex()
        .flex_col()
        .gap_3()
        .children(
            prompt
                .tool_schemas()
                .enumerate()
                .flat_map(|(index, (name, schema))| {
                    [
                        section_title(
                            name.unwrap_or(if index == 0 { "Tool" } else { "Unnamed tool" }),
                            colors,
                        ),
                        code_panel(schema, colors),
                    ]
                }),
        )
        .into_any_element()
}

fn usage_summary(usage: kcastle_agent::TokenUsage, colors: TrajectoryPalette) -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .gap_3()
        .child(detail_pair(
            "Input",
            &format!("{} tok", usage.input_tokens()),
            colors,
        ))
        .child(detail_pair(
            "Cached",
            &format!("{} tok", usage.cache_read_input_tokens),
            colors,
        ))
        .child(detail_pair(
            "Cache created",
            &format!("{} tok", usage.cache_write_input_tokens),
            colors,
        ))
        .child(detail_pair(
            "Other",
            &format!("{} tok", usage.uncached_input_tokens),
            colors,
        ))
        .child(detail_pair(
            "Output",
            &format!("{} tok", usage.total_output_tokens()),
            colors,
        ))
        .child(detail_pair(
            "Reasoning",
            &format!("{} tok", usage.reasoning_output_tokens),
            colors,
        ))
        .child(detail_pair(
            "Content",
            &format!("{} tok", usage.output_tokens),
            colors,
        ))
}

fn request_options_details(
    options: &ModelRequestOptions,
    colors: TrajectoryPalette,
) -> gpui::AnyElement {
    let reason = match options.reason {
        kcastle_agent::RequestHeaderReason::Initial => "Initial",
        kcastle_agent::RequestHeaderReason::Resume => "Resume",
        kcastle_agent::RequestHeaderReason::Change => "Change",
    };
    let mut body = div()
        .flex()
        .flex_col()
        .gap_3()
        .child(detail_pair("Reason", reason, colors))
        .child(detail_pair("Model", &options.model, colors));
    if let Some(effort) = options.reasoning_effort.as_deref() {
        body = body.child(detail_pair("Reasoning effort", effort, colors));
    }
    if let Some(maximum) = options.max_output_tokens {
        body = body.child(detail_pair(
            "Max output tokens",
            &maximum.to_string(),
            colors,
        ));
    }
    body = body.child(section_title("Session configuration", colors));
    if let Some(model) = options.session_config.model_id.as_deref() {
        body = body.child(detail_pair("Configured model", model, colors));
    }
    if let Some(effort) = options.session_config.reasoning_effort.as_deref() {
        body = body.child(detail_pair("Configured reasoning effort", effort, colors));
    }
    body.child(detail_pair(
        "Allow all tools",
        if options.session_config.allow_all_tools {
            "Yes"
        } else {
            "No"
        },
        colors,
    ))
    .into_any_element()
}

fn request_usage_details(
    request: &TrajectoryRequest,
    colors: TrajectoryPalette,
) -> gpui::AnyElement {
    let mut body = div().flex().flex_col().gap_3();
    body = body.child(section_title("This request", colors));
    body = match request.usage {
        Some(usage) => body.child(usage_summary(usage, colors)),
        None => body.child(empty_detail("Usage unavailable", colors)),
    };
    body = body.child(section_title("Session cumulative", colors));
    body = match request.cumulative_usage {
        Some(usage) => body.child(usage_summary(usage, colors)),
        None => body.child(empty_detail("Usage unavailable", colors)),
    };
    body.into_any_element()
}

fn detail_pair(label: &str, value: &str, colors: TrajectoryPalette) -> gpui::AnyElement {
    div()
        .flex()
        .justify_between()
        .gap_4()
        .text_sm()
        .child(
            div()
                .text_color(colors.label_tertiary)
                .child(label.to_owned()),
        )
        .child(div().text_right().child(value.to_owned()))
        .into_any_element()
}

fn empty_detail(label: &str, colors: TrajectoryPalette) -> gpui::AnyElement {
    div()
        .text_sm()
        .text_color(colors.label_tertiary)
        .child(label.to_owned())
        .into_any_element()
}

fn section_title(title: &str, colors: TrajectoryPalette) -> gpui::AnyElement {
    div()
        .mt_3()
        .pt_3()
        .border_t_1()
        .border_color(colors.border_l2)
        .text_sm()
        .text_color(colors.label_secondary)
        .child(title.to_owned())
        .into_any_element()
}

fn code_panel(text: &str, colors: TrajectoryPalette) -> gpui::AnyElement {
    div()
        .p_3()
        .rounded(px(6.0))
        .bg(colors.code_background)
        .text_sm()
        .child(text.to_owned())
        .into_any_element()
}

fn format_assistant_duration(nanoseconds: Option<u64>) -> String {
    let Some(nanoseconds) = nanoseconds else {
        return "Not recorded".into();
    };
    let milliseconds = nanoseconds as f64 / 1_000_000.0;
    if milliseconds < 1_000.0 {
        format!("{milliseconds:.0} ms")
    } else if milliseconds < 10_000.0 {
        format!("{:.2} s", milliseconds / 1_000.0)
    } else {
        format!("{:.1} s", milliseconds / 1_000.0)
    }
}

fn format_elapsed_duration(nanoseconds: Option<u64>) -> String {
    let Some(nanoseconds) = nanoseconds else {
        return "—".into();
    };
    let milliseconds = (nanoseconds as f64 / 1_000_000.0).round() as u64;
    let digits = milliseconds.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(digit);
    }
    format!("{formatted} ms")
}

fn format_started(record: &TrajectoryRecord, unix: bool) -> String {
    record
        .timing
        .started
        .as_ref()
        .map(|time| format_wall(time.wall_time_ms(), unix))
        .unwrap_or_else(|| "Not available".into())
}

fn format_wall(wall_time_ms: i64, unix: bool) -> String {
    if unix {
        return format!("{:.3}", wall_time_ms as f64 / 1_000.0);
    }
    let Ok(nanoseconds) = i128::from(wall_time_ms).checked_mul(1_000_000).ok_or(()) else {
        return "Not available".into();
    };
    let Ok(timestamp) = OffsetDateTime::from_unix_timestamp_nanos(nanoseconds) else {
        return "Not recorded".into();
    };
    let offset = UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC);
    timestamp
        .to_offset(offset)
        .format(format_description!(
            "[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]"
        ))
        .unwrap_or_else(|_| "Not available".into())
}

fn format_timing_point(
    time: Option<&EventTimeRef>,
    unix: bool,
    record: &TrajectoryRecord,
) -> String {
    time.map(|time| format_wall(time.wall_time_ms(), unix))
        .unwrap_or_else(|| execution_missing(record))
}

fn timing_duration(record: &TrajectoryRecord) -> String {
    if record.kind != TrajectoryKind::Assistant {
        return format_elapsed_duration(record.timing.duration_ns());
    }
    if !assistant_timing_recorded(record) {
        return "Not recorded".into();
    }
    if record.timing.started.is_none() {
        return "Step start unavailable".into();
    }
    if record.timing.completed.is_none() {
        return "Pending".into();
    }
    format_assistant_duration(record.timing.duration_ns())
}

fn assistant_timing_recorded(record: &TrajectoryRecord) -> bool {
    record.timing.started.is_some()
        || record.timing.first_token.is_some()
        || record.timing.completed.is_some()
}

fn assistant_ttft(record: &TrajectoryRecord) -> String {
    if !assistant_timing_recorded(record) {
        "Not recorded".into()
    } else if record.timing.started.is_none() {
        "Step start unavailable".into()
    } else if record.timing.first_token.is_none() {
        "First token unavailable".into()
    } else {
        format_assistant_duration(record.timing.ttft_ns())
    }
}

fn assistant_generation(record: &TrajectoryRecord) -> String {
    if !assistant_timing_recorded(record) || record.timing.first_token.is_none() {
        "First token unavailable".into()
    } else if record.timing.completed.is_none() {
        "Pending".into()
    } else {
        format_assistant_duration(record.timing.generation_ns())
    }
}

fn assistant_throughput(record: &TrajectoryRecord) -> String {
    let Some(usage) = record.usage else {
        return "Usage unavailable".into();
    };
    let output_tokens = usage.total_output_tokens();
    if !assistant_timing_recorded(record) || record.timing.first_token.is_none() {
        return "First token unavailable".into();
    }
    if record.timing.completed.is_none() {
        return "Pending".into();
    }
    let generation = record.timing.generation_ns().unwrap_or_default();
    if generation == 0 {
        return "Duration too short".into();
    }
    format!(
        "{:.1} tok/s",
        output_tokens as f64 / (generation as f64 / 1_000_000_000.0)
    )
}

fn execution_missing(record: &TrajectoryRecord) -> String {
    match record.status {
        ItemStatus::Denied | ItemStatus::NotExecuted => "Not executed".into(),
        ItemStatus::Unknown => "Unknown".into(),
        _ => "Not recorded".into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use im::{HashSet as ImHashSet, Vector};
    use kcastle_agent::{AssistantChunk, CallId, EventTime, RequestId, SessionEvent, TokenUsage};

    use crate::domain::session_document::SessionDocument;
    use crate::domain::session_document::tests::{fixture, recorded};
    use crate::domain::timeline::{
        AxisId, AxisRange, DomainRange, RenderCell, RenderIdentity, TimelineLane,
    };
    use crate::domain::{
        DetailsTab, ItemStatus, LayoutGeneration, RecordTiming, TimelineMode, TrajectoryItemId,
        TrajectoryKind, TrajectoryProjection, TrajectoryRecord,
    };
    use crate::layout::TrajectoryMode;

    use super::{
        ScrollStrategy, TIMELINE_BAR_HEIGHT, TIMELINE_BAR_OFFSET, TimelineCacheIdentity,
        TimelineCellMatches, TimelineFocusCache, TimelineLedgerRow, TimelineMatches, TimelineModel,
        TimelineModelCache, TimelineRows, TimelineSearchCache, TimelineView,
        TrajectoryDetailsLayoutState, TrajectoryDetailsMarkdownCache, TrajectoryMarkdownSource,
        TrajectorySelectionSource, aligned_trajectory_list_offset, calls_summary_text,
        clamp_trajectory_details_width, focus_scroll_target, minimum_timeline_selection_width,
        nested_segment_geometry, normalized_range, record_tooltip,
        resolved_trajectory_details_width, should_clear_selection_for_record,
        sync_trajectory_list_state, timeline_block_opacity, timeline_geometry, timeline_lane_at,
        timeline_lane_top, timeline_model, timeline_turn_boundary_fractions,
        trajectory_ledger_row_height, turn_summary_text,
    };

    fn time(ms: u64) -> EventTime {
        EventTime {
            wall_time_ms: 1_000 + ms as i64,
            clock_id: "timeline-test".into(),
            monotonic_ns: ms * 1_000_000,
        }
    }

    fn record(id: u64, start: u64, end: u64) -> TrajectoryRecord {
        let timing = RecordTiming {
            started: Some((&time(start)).into()),
            requested: Some((&time(start + 5)).into()),
            authorization_resolved: Some((&time(start + 10)).into()),
            execution_started: Some((&time(start + 20)).into()),
            execution_finished: Some((&time(end - 20)).into()),
            completed: Some((&time(end)).into()),
            ..RecordTiming::default()
        };
        TrajectoryRecord {
            id: TrajectoryItemId::Tool(CallId::from_raw(format!("call-{id}"))),
            source_seq: id,
            kind: TrajectoryKind::Tool,
            title: "tool".into(),
            text: String::new(),
            payload: None,
            turn: Some(1),
            step: Some(1),
            status: ItemStatus::Completed,
            timing,
            usage: None,
            search_text: "tool\n".into(),
        }
    }

    #[test]
    fn details_width_matches_dsh_default_explicit_and_overlay_rules() {
        assert_eq!(clamp_trajectory_details_width(100.0, 900.0), 320.0);
        assert_eq!(clamp_trajectory_details_width(800.0, 900.0), 620.0);
        assert_eq!(clamp_trajectory_details_width(500.4, 1_500.0), 500.0);

        assert_eq!(
            resolved_trajectory_details_width(TrajectoryMode::Split, 900.0, None),
            342.0
        );
        assert_eq!(
            resolved_trajectory_details_width(TrajectoryMode::Split, 1_500.0, None),
            440.0
        );
        assert_eq!(
            resolved_trajectory_details_width(TrajectoryMode::Split, 900.0, Some(700.0)),
            620.0
        );
        assert_eq!(
            resolved_trajectory_details_width(TrajectoryMode::Split, 1_500.0, Some(700.0)),
            700.0
        );
        assert_eq!(
            resolved_trajectory_details_width(TrajectoryMode::Overlay, 600.0, None),
            420.0
        );
        assert_eq!(
            resolved_trajectory_details_width(TrajectoryMode::Overlay, 400.0, Some(500.0)),
            368.0
        );
    }

    #[test]
    fn details_layout_uses_current_generation_and_frozen_drag_geometry() {
        let generation = LayoutGeneration(3);
        let mut state = TrajectoryDetailsLayoutState::default();
        assert!(state.observe_split_width(generation, 900.0));
        assert!(!state.observe_split_width(generation, 900.2));
        assert_eq!(state.split_width(generation, 1_200.0), 900.0);
        assert_eq!(state.split_width(generation.next(), 1_200.0), 1_200.0);

        assert!(state.observe_details_width(generation, 400.0));
        assert_eq!(state.measured_details_width(generation), Some(400.0));
        assert_eq!(state.measured_details_width(generation.next()), None);

        state.begin_drag(100.0, 400.0, 900.0);
        assert!(state.drag_to(80.0));
        assert_eq!(
            state.details_width(TrajectoryMode::Split, generation, 1_200.0),
            420.0
        );
        // A live container measurement must not change the geometry frozen at pointer-down.
        assert!(state.observe_split_width(generation, 1_500.0));
        assert!(state.drag_to(-500.0));
        assert_eq!(
            state.details_width(TrajectoryMode::Split, generation, 1_200.0),
            620.0
        );
        assert!(state.end_drag());
        assert!(!state.end_drag());
    }

    #[test]
    fn details_layout_keyboard_step_and_reset_preserve_dsh_semantics() {
        let generation = LayoutGeneration(1);
        let mut state = TrajectoryDetailsLayoutState::default();
        state.observe_split_width(generation, 1_000.0);
        assert!(state.step(16.0, 400.0, 1_000.0));
        assert_eq!(
            state.details_width(TrajectoryMode::Split, generation, 1_000.0),
            416.0
        );
        assert!(state.step(-16.0, 416.0, 1_000.0));
        assert_eq!(
            state.details_width(TrajectoryMode::Split, generation, 1_000.0),
            400.0
        );
        assert!(state.reset());
        assert_eq!(
            state.details_width(TrajectoryMode::Split, generation, 1_000.0),
            380.0
        );
        assert!(!state.reset());
    }

    #[test]
    fn selected_details_markdown_cache_reparses_only_when_content_identity_changes() {
        let record = record(9, 0, 100);
        let mut cache = TrajectoryDetailsMarkdownCache::default();

        cache.sync(10, &record.id, TrajectoryMarkdownSource::Preview, "first");
        assert_eq!(cache.markdown.revision(), 1);
        cache.sync(10, &record.id, TrajectoryMarkdownSource::Preview, "first");
        assert_eq!(cache.markdown.revision(), 1);

        let mut layout = TrajectoryDetailsLayoutState::default();
        layout.observe_details_width(LayoutGeneration(1), 400.0);
        layout.step(16.0, 400.0, 1_000.0);
        cache.sync(10, &record.id, TrajectoryMarkdownSource::Preview, "first");
        assert_eq!(cache.markdown.revision(), 1);

        cache.sync(
            10,
            &record.id,
            TrajectoryMarkdownSource::Preview,
            "first\n\nsecond",
        );
        assert_eq!(cache.markdown.revision(), 2);

        // A tab or session identity switch owns a fresh parser even if the bytes are equal.
        cache.sync(
            10,
            &record.id,
            TrajectoryMarkdownSource::SystemPrompt,
            "first\n\nsecond",
        );
        assert_eq!(cache.markdown.revision(), 1);
        cache.sync(
            11,
            &record.id,
            TrajectoryMarkdownSource::SystemPrompt,
            "first\n\nsecond",
        );
        assert_eq!(cache.markdown.revision(), 1);
    }

    fn cache_identity(
        document_generation: u64,
        revision: u64,
        mode: TimelineMode,
    ) -> TimelineCacheIdentity {
        TimelineCacheIdentity {
            axis: AxisId {
                document_generation,
                geometry_revision: revision,
                mode,
            },
            change_revision: revision,
        }
    }

    #[test]
    fn timeline_turn_boundaries_follow_turn_starts_in_the_visible_domain() {
        let mut system = record(1, 0, 100);
        system.kind = TrajectoryKind::System;
        system.turn = None;
        let mut first_turn = record(2, 100, 200);
        first_turn.turn = Some(1);
        let mut same_turn = record(3, 200, 300);
        same_turn.turn = Some(1);
        let mut second_turn = record(4, 300, 400);
        second_turn.turn = Some(2);
        let records = [system, first_turn, same_turn, second_turn]
            .into_iter()
            .map(Arc::new)
            .collect::<Vector<_>>();
        let axis = AxisId {
            document_generation: 1,
            geometry_revision: 1,
            mode: TimelineMode::Sequence,
        };
        let materialized = records.iter().cloned().collect::<Vec<_>>();
        let geometry = timeline_geometry(&materialized, axis).expect("timeline geometry");

        assert_eq!(
            timeline_turn_boundary_fractions(&records, &geometry, geometry.domain),
            vec![0.25, 0.75]
        );
        assert_eq!(
            timeline_turn_boundary_fractions(&records, &geometry, DomainRange::new(1.0, 4.0),),
            vec![0.0, 2.0 / 3.0]
        );
    }

    #[test]
    fn timeline_paint_uses_dsh_focus_search_and_role_opacity_precedence() {
        assert_eq!(
            timeline_block_opacity(TrajectoryKind::User, true, true, false),
            0.78
        );
        assert_eq!(
            timeline_block_opacity(TrajectoryKind::Tool, true, true, false),
            1.0
        );
        assert_eq!(
            timeline_block_opacity(TrajectoryKind::User, false, true, false),
            0.2
        );
        assert_eq!(
            timeline_block_opacity(TrajectoryKind::User, false, true, true),
            1.0
        );
        assert_eq!(
            timeline_block_opacity(TrajectoryKind::User, true, false, true),
            0.14
        );
    }

    #[test]
    fn minimum_selection_width_uses_the_full_domain_operation_width() {
        let domain = DomainRange::new(0.0, 100.0);
        assert_eq!(
            minimum_timeline_selection_width(domain, DomainRange::new(40.0, 50.0), 100),
            1.0
        );
        assert_eq!(
            minimum_timeline_selection_width(domain, DomainRange::new(40.0, 40.5), 100),
            0.5
        );
    }

    #[test]
    fn timeline_cache_reuses_geometry_when_only_the_viewport_changes() {
        let projection_lineage = 17;
        let records = [record(1, 0, 100)]
            .into_iter()
            .map(std::sync::Arc::new)
            .collect::<im::Vector<_>>();
        let axis = AxisId {
            document_generation: projection_lineage,
            geometry_revision: 4,
            mode: TimelineMode::Duration,
        };
        let mut cache = TimelineModelCache::new(
            cache_identity(projection_lineage, 4, TimelineMode::Duration),
            &records,
            TimelineView {
                viewport: Some(AxisRange {
                    axis,
                    range: DomainRange::new(0.0, 100.0),
                }),
                selection: None,
                render_width_px: 1_500.0,
            },
            None,
        );
        assert!(cache.geometry_matches(axis));
        assert!(!cache.geometry_matches(AxisId {
            geometry_revision: 5,
            ..axis
        }));

        let geometry_before = cache.geometry.as_ref().unwrap().cells.clone();
        let viewport = AxisRange {
            axis,
            range: DomainRange::new(10.0, 60.0),
        };
        cache.sync_ranges(Some(viewport), 1_500.0);
        assert_eq!(cache.viewport, Some(viewport));
        assert_eq!(cache.geometry.as_ref().unwrap().cells, geometry_before);
        assert_eq!(
            cache.model.as_ref().unwrap().viewport,
            DomainRange::new(10.0, 60.0)
        );
    }

    #[test]
    fn hundred_thousand_row_stream_delta_touches_only_changed_search_record() {
        let mut records = (0..100_000)
            .map(|index| Arc::new(record(index, index, index.saturating_add(100))))
            .collect::<im::Vector<_>>();

        let model_cache = TimelineModelCache::new(
            cache_identity(77, 1, TimelineMode::Sequence),
            &records,
            TimelineView {
                viewport: None,
                selection: None,
                render_width_px: 1_500.0,
            },
            None,
        );
        let model = model_cache.model.as_ref().unwrap();
        let geometry = model_cache.geometry.as_ref().unwrap();
        assert!(model.cells.len() <= super::TIMELINE_PRIMITIVE_LIMIT);
        assert_eq!(
            model.cells.iter().map(|cell| cell.ids.len()).sum::<usize>(),
            100_000
        );
        assert!(model.cells.iter().all(|cell| matches!(
            cell.lane,
            TimelineLane::Input | TimelineLane::Model | TimelineLane::Tools
        )));

        let mut default_cache = TimelineSearchCache::build_records(&records, 1, "", false, false);
        assert!(matches!(default_cache.rows, TimelineRows::All(100_000)));
        assert_eq!(default_cache.inspected_records, 0);
        assert_eq!(default_cache.materialized_row_rebuilds, 0);

        let mut updated = records[77_777].as_ref().clone();
        updated.search_text = "needle".into();
        records.set(77_777, Arc::new(updated));
        default_cache.sync_changed_records(&records, 2, [77_777]);
        assert_eq!(default_cache.inspected_records, 0);
        assert!(matches!(default_cache.rows, TimelineRows::All(100_000)));
        assert_eq!(default_cache.materialized_row_rebuilds, 0);

        let mut search_cache =
            TimelineSearchCache::build_records(&records, 2, "absent", false, false);
        assert_eq!(search_cache.inspected_records, 100_000);
        search_cache.sync_model_matches(
            geometry,
            model,
            model_cache.model_revision,
            &Vector::new(),
        );
        assert!(matches!(
            search_cache.matched_cells,
            TimelineCellMatches::Filtered(ref cells) if cells.is_empty()
        ));
        let inspected_before = search_cache.inspected_records;
        let row_rebuilds_before = search_cache.materialized_row_rebuilds;
        let mut updated = records[77_777].as_ref().clone();
        updated.search_text = "absent now matches".into();
        records.set(77_777, Arc::new(updated));
        let changed = search_cache.sync_changed_records(&records, 3, [77_777]);
        search_cache.sync_model_matches(geometry, model, model_cache.model_revision, &changed);
        assert_eq!(search_cache.inspected_records - inspected_before, 1);
        assert_eq!(search_cache.rows.len(), 1);
        assert_eq!(
            search_cache.rows.get(0),
            Some(TimelineLedgerRow::Record(77_777))
        );
        assert_eq!(search_cache.materialized_row_rebuilds, row_rebuilds_before);
        let matched_cell = geometry
            .render_cell_for_record(&model.cells, 77_777)
            .expect("record is projected");
        assert!(search_cache.matched_cells.contains(matched_cell));

        // A later timing/usage-only receipt advances no search revision and inspects nothing.
        let inspected_before = search_cache.inspected_records;
        search_cache.sync_changed_records(&records, 3, std::iter::empty());
        assert_eq!(search_cache.inspected_records, inspected_before);
    }

    #[test]
    fn zoomed_hundred_thousand_row_timeline_resolves_records_without_a_dense_lookup() {
        let records = (0..100_000_u64)
            .map(|index| Arc::new(record(index + 1, index * 100, index * 100 + 100)))
            .collect::<Vector<_>>();
        let geometry = super::timeline_geometry_from_iter(
            records.iter(),
            AxisId {
                document_generation: 1,
                geometry_revision: 1,
                mode: TimelineMode::Sequence,
            },
        )
        .expect("sequence geometry");
        let model = super::project_timeline(
            &geometry,
            DomainRange::new(50_000.0, 50_100.0),
            1_500.0,
            records.len(),
        );

        assert!(
            geometry
                .render_cell_for_record(&model.cells, 50_050)
                .is_some()
        );
        assert!(geometry.render_cell_for_record(&model.cells, 10).is_none());
    }

    #[test]
    fn empty_query_incremental_append_updates_filtered_rows() {
        let mut records = [Arc::new(record(1, 0, 100))]
            .into_iter()
            .collect::<Vector<_>>();
        let mut collapsed_turns = TimelineSearchCache::build_records(&records, 1, "", true, false);

        let mut next_turn = record(2, 100, 200);
        next_turn.turn = Some(2);
        records.push_back(Arc::new(next_turn));
        collapsed_turns.sync_changed_records(&records, 2, [1]);
        assert_eq!(collapsed_turns.rows.len(), 2);
        assert_eq!(
            collapsed_turns.rows.get(0),
            Some(TimelineLedgerRow::Record(0))
        );
        assert_eq!(
            collapsed_turns.rows.get(1),
            Some(TimelineLedgerRow::Record(1))
        );

        let mut collapsed_calls = TimelineSearchCache::build_records(&records, 2, "", false, true);
        let mut assistant = record(3, 200, 300);
        assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-3"));
        assistant.kind = TrajectoryKind::Assistant;
        records.push_back(Arc::new(assistant));
        collapsed_calls.sync_changed_records(&records, 3, [2]);
        assert_eq!(collapsed_calls.rows.len(), 3);
        assert_eq!(
            collapsed_calls.rows.get(2),
            Some(TimelineLedgerRow::Record(2))
        );
    }

    #[test]
    fn active_search_ignores_turn_and_call_collapsing() {
        let mut tool = record(1, 0, 100);
        tool.search_text = "shell\nneedle".into();
        let records = [Arc::new(tool)].into_iter().collect::<Vector<_>>();

        let cache = TimelineSearchCache::build_records(&records, 1, "needle", true, true);

        assert_eq!(cache.matching_indices.len(), 1);
        assert_eq!(cache.rows.len(), 1);
        assert_eq!(cache.rows.get(0), Some(TimelineLedgerRow::Record(0)));
    }

    #[test]
    fn collapsed_turn_keeps_its_first_record_and_adds_a_summary_row() {
        let mut first = record(1, 0, 100);
        first.id = TrajectoryItemId::Assistant(RequestId::from("request-1"));
        first.kind = TrajectoryKind::Assistant;
        let second = record(2, 100, 200);
        let mut third = record(3, 200, 300);
        third.id = TrajectoryItemId::Assistant(RequestId::from("request-3"));
        third.kind = TrajectoryKind::Assistant;
        third.step = Some(2);
        let records = [Arc::new(first), Arc::new(second), Arc::new(third)]
            .into_iter()
            .collect::<Vector<_>>();

        let cache = TimelineSearchCache::build_records(&records, 1, "", true, false);

        assert_eq!(
            cache.rows,
            TimelineRows::Projected(
                [
                    TimelineLedgerRow::Record(0),
                    TimelineLedgerRow::TurnSummary {
                        representative: 0,
                        turn: 1,
                        first_hidden: 1,
                        last_hidden: 2,
                        step_ids: [1_u32, 2_u32].into_iter().collect(),
                        call_count: 1,
                    },
                ]
                .into_iter()
                .collect()
            )
        );
    }

    #[test]
    fn collapsed_calls_keep_the_assistant_and_add_one_summary_for_its_tool_run() {
        let mut assistant = record(1, 0, 100);
        assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-1"));
        assistant.kind = TrajectoryKind::Assistant;
        let first_tool = record(2, 100, 200);
        let second_tool = record(3, 200, 300);
        let records = [
            Arc::new(assistant),
            Arc::new(first_tool),
            Arc::new(second_tool),
        ]
        .into_iter()
        .collect::<Vector<_>>();

        let cache = TimelineSearchCache::build_records(&records, 1, "", false, true);

        assert_eq!(
            cache.rows,
            TimelineRows::Projected(
                [
                    TimelineLedgerRow::Record(0),
                    TimelineLedgerRow::CallsSummary {
                        assistant: 0,
                        first_tool: 1,
                        last_tool: 2,
                        tool_names: ["tool".to_owned()].into_iter().collect(),
                        tools: Arc::from("tool"),
                        tools_truncated: false,
                    },
                ]
                .into_iter()
                .collect()
            )
        );
    }

    #[test]
    fn collapsed_calls_preserve_standalone_tools_and_deduplicate_names_in_source_order() {
        let mut assistant = record(1, 0, 100);
        assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-1"));
        assistant.kind = TrajectoryKind::Assistant;
        let mut bash_first = record(2, 100, 200);
        bash_first.title = "bash".into();
        let mut bash_second = record(3, 200, 300);
        bash_second.title = "bash".into();
        let mut read = record(4, 300, 400);
        read.title = "read".into();
        let mut context = record(5, 400, 500);
        context.kind = TrajectoryKind::Context;
        let mut standalone = record(6, 500, 600);
        standalone.title = "standalone".into();
        let records = [
            Arc::new(assistant),
            Arc::new(bash_first),
            Arc::new(bash_second),
            Arc::new(read),
            Arc::new(context),
            Arc::new(standalone),
        ]
        .into_iter()
        .collect::<Vector<_>>();

        let cache = TimelineSearchCache::build_records(&records, 1, "", false, true);

        assert_eq!(cache.rows.len(), 4);
        assert_eq!(cache.rows.get(0), Some(TimelineLedgerRow::Record(0)));
        assert_eq!(
            cache.rows.get(1),
            Some(TimelineLedgerRow::CallsSummary {
                assistant: 0,
                first_tool: 1,
                last_tool: 3,
                tool_names: ["bash".to_owned(), "read".to_owned()].into_iter().collect(),
                tools: Arc::from("bash, read"),
                tools_truncated: false,
            })
        );
        assert_eq!(cache.rows.get(2), Some(TimelineLedgerRow::Record(4)));
        assert_eq!(cache.rows.get(3), Some(TimelineLedgerRow::Record(5)));
        assert!(cache.rows.get(1).unwrap().represents(2, &records));
        assert!(!cache.rows.get(1).unwrap().represents(5, &records));
    }

    #[test]
    fn folded_projection_preserves_system_rows_and_incremental_equals_full_replay() {
        let mut system_before = record(1, 0, 100);
        system_before.kind = TrajectoryKind::System;
        let mut assistant = record(2, 100, 200);
        assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-2"));
        assistant.kind = TrajectoryKind::Assistant;
        let mut tool = record(3, 200, 300);
        tool.title = "bash".into();
        let mut system_middle = record(4, 300, 400);
        system_middle.kind = TrajectoryKind::System;
        let mut second_tool = record(5, 400, 500);
        second_tool.title = "read".into();
        second_tool.step = Some(2);
        let source = [system_before, assistant, tool, system_middle, second_tool];
        let collapsed_turns = HashSet::from([1]);
        let collapsed_assistants = HashSet::from([source[1].id.clone()]);

        let mut records = Vector::new();
        let mut cache = TimelineSearchCache::build_records_with_folds(
            &records,
            1,
            "",
            0,
            0,
            &collapsed_turns,
            &collapsed_assistants,
        );
        for (index, record) in source.into_iter().enumerate() {
            records.push_back(Arc::new(record));
            cache.sync_changed_records(&records, index as u64 + 2, [index]);
            assert_eq!(
                cache.rows,
                super::project_ledger_rows(
                    &records,
                    &super::TimelineMatches::All(records.len()),
                    false,
                    &collapsed_turns,
                    &collapsed_assistants,
                ),
                "incremental projection diverged at prefix length {}",
                index + 1,
            );
        }
        assert_eq!(cache.rows.get(0), Some(TimelineLedgerRow::Record(0)));
        assert_eq!(cache.rows.get(1), Some(TimelineLedgerRow::Record(1)));
        assert!(matches!(
            cache.rows.get(2),
            Some(TimelineLedgerRow::TurnSummary {
                first_hidden: 2,
                last_hidden: 4,
                ..
            })
        ));
        assert_eq!(cache.rows.get(3), Some(TimelineLedgerRow::Record(3)));
    }

    #[test]
    fn folded_summary_focus_covers_only_its_hidden_members() {
        let mut assistant = record(1, 0, 100);
        assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-1"));
        assistant.kind = TrajectoryKind::Assistant;
        let records = [Arc::new(assistant), Arc::new(record(2, 100, 200))]
            .into_iter()
            .collect::<Vector<_>>();
        let cache = TimelineSearchCache::build_records(&records, 1, "", true, false);
        let summary = cache.rows.get(1).expect("turn summary");

        assert!(!summary.intersects(&HashSet::from([0]), &records));
        assert!(summary.intersects(&HashSet::from([1]), &records));
    }

    #[test]
    fn ten_thousand_folded_appends_do_not_rebuild_the_existing_projection() {
        let mut records = Vector::new();
        let collapsed_turns = (1..=2_500_u32).collect::<HashSet<_>>();
        let collapsed_assistants = HashSet::new();
        let mut cache = TimelineSearchCache::build_records_with_folds(
            &records,
            1,
            "",
            0,
            0,
            &collapsed_turns,
            &collapsed_assistants,
        );
        for index in 0..10_000_usize {
            let mut next = record(index as u64 + 1, index as u64, index as u64 + 100);
            next.turn = Some((index / 4 + 1) as u32);
            next.step = Some((index / 2 + 1) as u32);
            if index % 4 == 0 {
                next.id = TrajectoryItemId::Assistant(RequestId::from(format!("request-{index}")));
                next.kind = TrajectoryKind::Assistant;
            }
            records.push_back(Arc::new(next));
            cache.sync_changed_records(&records, index as u64 + 2, [index]);
        }

        assert_eq!(cache.inspected_records, 10_000);
        assert_eq!(cache.materialized_row_rebuilds, 1);
        assert_eq!(
            cache.rows,
            super::project_ledger_rows(
                &records,
                &super::TimelineMatches::All(records.len()),
                false,
                &collapsed_turns,
                &collapsed_assistants,
            )
        );
    }

    #[test]
    fn alternating_system_rows_keep_collapsed_turn_appends_linear_and_canonical() {
        let mut records = Vector::new();
        let collapsed_turns = HashSet::from([1]);
        let collapsed_assistants = HashSet::new();
        let mut cache = TimelineSearchCache::build_records_with_folds(
            &records,
            1,
            "",
            0,
            0,
            &collapsed_turns,
            &collapsed_assistants,
        );
        for index in 0..10_000_usize {
            let start = index as u64 * 100;
            let mut next = record(index as u64 + 1, start, start + 100);
            next.turn = Some(1);
            next.step = Some((index / 2 + 1) as u32);
            if index % 2 == 1 {
                next.kind = TrajectoryKind::System;
            }
            records.push_back(Arc::new(next));
            cache.sync_changed_records(&records, index as u64 + 2, [index]);
        }

        assert_eq!(cache.materialized_row_rebuilds, 1);
        assert_eq!(
            cache.rows,
            super::project_ledger_rows(
                &records,
                &super::TimelineMatches::All(records.len()),
                false,
                &collapsed_turns,
                &collapsed_assistants,
            )
        );
    }

    #[test]
    fn tool_summary_preview_has_bounded_append_cost() {
        let mut preview: Arc<str> = Arc::from("");
        let mut truncated = false;
        for index in 0..10_000 {
            (preview, truncated) =
                super::append_tool_summary_preview(&preview, truncated, &format!("tool-{index}"));
        }

        assert!(truncated);
        assert!(preview.len() <= super::TOOL_SUMMARY_PREVIEW_MAX_BYTES + ", …".len());
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn filtered_rows_recompute_turn_start_for_double_click_and_connectors() {
        let records = [Arc::new(record(1, 0, 100)), Arc::new(record(2, 100, 200))]
            .into_iter()
            .collect::<Vector<_>>();
        let matching = super::TimelineMatches::Filtered([1_usize].into_iter().collect());
        let rows =
            super::project_ledger_rows(&records, &matching, true, &HashSet::new(), &HashSet::new());

        assert_eq!(
            super::ledger_record_boundaries(&rows, 0, &records[1], &records),
            (true, false, false)
        );
        assert_eq!(
            super::ledger_double_click_target(
                &records[1],
                true,
                &HashSet::new(),
                &HashSet::from([1]),
                &HashSet::new(),
            ),
            Some(super::LedgerFoldTarget::Turn(1))
        );
    }

    #[test]
    fn multi_term_search_updates_only_the_changed_record() {
        let mut records = (0..100_000)
            .map(|index| Arc::new(record(index, index, index.saturating_add(100))))
            .collect::<Vector<_>>();
        let target = 77_777;
        let mut initial = records[target].as_ref().clone();
        initial.search_text = "alpha".into();
        records.set(target, Arc::new(initial));
        let mut cache = TimelineSearchCache::build_records(&records, 1, "alpha beta", true, true);
        assert_eq!(cache.rows.len(), 0);

        let inspected = cache.inspected_records;
        let rebuilds = cache.materialized_row_rebuilds;
        let mut matching = records[target].as_ref().clone();
        matching.search_text = "alpha unrelated beta".into();
        records.set(target, Arc::new(matching));
        cache.sync_changed_records(&records, 2, [target]);
        assert_eq!(cache.inspected_records - inspected, 1);
        assert_eq!(cache.materialized_row_rebuilds, rebuilds);
        assert_eq!(cache.rows.get(0), Some(TimelineLedgerRow::Record(target)));

        let inspected = cache.inspected_records;
        let mut no_longer_matching = records[target].as_ref().clone();
        no_longer_matching.search_text = "beta only".into();
        records.set(target, Arc::new(no_longer_matching));
        cache.sync_changed_records(&records, 3, [target]);
        assert_eq!(cache.inspected_records - inspected, 1);
        assert_eq!(cache.rows.len(), 0);
    }

    #[test]
    fn text_only_changes_advance_geometry_cursor_without_rebuilding_focus() {
        let events = fixture();
        let mut document = SessionDocument::from_events(events[..7].to_vec()).unwrap();
        let mut projection = TrajectoryProjection::from_document(&document);
        let request_id = RequestId::from("request-1");
        let first = recorded(
            document.cursor().next_seq,
            SessionEvent::AssistantChunk {
                request_id: request_id.clone(),
                chunk: AssistantChunk::OutputTextDelta { delta: "a".into() },
            },
        );
        let delta = document.apply_batch(vec![first]).unwrap();
        projection = TrajectoryProjection::after_delta(&document, &delta, &projection);

        let axis = AxisId {
            document_generation: projection.projection_lineage(),
            geometry_revision: projection.revision(),
            mode: TimelineMode::Sequence,
        };
        let assistant_index = projection
            .record_index(&TrajectoryItemId::Assistant(request_id.clone()))
            .unwrap();
        let selection = AxisRange {
            axis,
            range: DomainRange::new(assistant_index as f64, assistant_index as f64 + 1.0),
        };
        let mut cache = TimelineModelCache::new(
            cache_identity(
                projection.projection_lineage(),
                projection.revision(),
                TimelineMode::Sequence,
            ),
            &projection.records,
            TimelineView {
                viewport: None,
                selection: Some(selection),
                render_width_px: 1_500.0,
            },
            None,
        );
        let mut hidden_cache = TimelineModelCache::new(
            cache_identity(
                projection.projection_lineage(),
                projection.revision(),
                TimelineMode::Sequence,
            ),
            &projection.records,
            TimelineView {
                viewport: None,
                selection: Some(selection),
                render_width_px: 1_500.0,
            },
            None,
        );
        let focused = Arc::clone(&cache.focus.as_ref().unwrap().record_indices);

        for _ in 0..300 {
            let event = recorded(
                document.cursor().next_seq,
                SessionEvent::AssistantChunk {
                    request_id: request_id.clone(),
                    chunk: AssistantChunk::OutputTextDelta { delta: "x".into() },
                },
            );
            let delta = document.apply_batch(vec![event]).unwrap();
            projection = TrajectoryProjection::after_delta(&document, &delta, &projection);
            let focus_changed = cache.sync_projection(&projection).unwrap();
            assert!(!focus_changed);
            cache.sync_focus(&projection.records, Some(selection), focus_changed);
        }
        assert_eq!(cache.change_revision, projection.change_revision());
        assert!(Arc::ptr_eq(
            &focused,
            &cache.focus.as_ref().unwrap().record_indices
        ));

        let completion = events
            .iter()
            .find_map(|event| {
                matches!(event.event, SessionEvent::AssistantCompleted { .. })
                    .then(|| event.event.clone())
            })
            .unwrap();
        let delta = document
            .apply_batch(vec![recorded(document.cursor().next_seq, completion)])
            .unwrap();
        projection = TrajectoryProjection::after_delta(&document, &delta, &projection);
        assert_eq!(cache.sync_projection(&projection), Some(true));
        assert_eq!(hidden_cache.sync_projection(&projection), Some(true));
        assert_eq!(cache.change_revision, projection.change_revision());
        assert!(cache.geometry_matches(AxisId {
            geometry_revision: projection.revision(),
            ..axis
        }));
    }

    #[test]
    fn timed_cache_tail_update_matches_a_fresh_projection_without_rebuilding_geometry() {
        let events = fixture();
        let mut document = SessionDocument::from_events(events[..27].to_vec()).unwrap();
        let mut projection = TrajectoryProjection::from_document(&document);
        let mut caches = [TimelineMode::Duration, TimelineMode::Actual].map(|mode| {
            TimelineModelCache::new(
                TimelineCacheIdentity {
                    axis: AxisId {
                        document_generation: projection.projection_lineage(),
                        geometry_revision: projection.revision(),
                        mode,
                    },
                    change_revision: projection.change_revision(),
                },
                &projection.records,
                TimelineView {
                    viewport: None,
                    selection: None,
                    render_width_px: 1_500.0,
                },
                None,
            )
        });

        let delta = document.apply_batch(vec![events[27].clone()]).unwrap();
        projection = TrajectoryProjection::after_delta(&document, &delta, &projection);

        for (cache, mode) in caches
            .iter_mut()
            .zip([TimelineMode::Duration, TimelineMode::Actual])
        {
            assert_eq!(cache.sync_projection(&projection), Some(true));
            assert_eq!(cache.timed_incremental_updates, 1);
            let rebuilt = TimelineModelCache::new(
                TimelineCacheIdentity {
                    axis: AxisId {
                        document_generation: projection.projection_lineage(),
                        geometry_revision: projection.revision(),
                        mode,
                    },
                    change_revision: projection.change_revision(),
                },
                &projection.records,
                TimelineView {
                    viewport: None,
                    selection: None,
                    render_width_px: 1_500.0,
                },
                None,
            );
            let incremental = cache.model.as_ref().unwrap();
            let fresh = rebuilt.model.as_ref().unwrap();
            assert_eq!(incremental.axis, fresh.axis);
            assert_eq!(incremental.domain, fresh.domain);
            assert_eq!(incremental.viewport, fresh.viewport);
            assert_eq!(incremental.cells, fresh.cells);
        }
    }

    #[test]
    fn timed_ranges_survive_streaming_assistant_and_tool_completion() {
        let events = fixture();
        for (prefix_len, completion_index) in [(9_usize, 9_usize), (15, 15)] {
            let mut document = SessionDocument::from_events(events[..prefix_len].to_vec()).unwrap();
            let mut projection = TrajectoryProjection::from_document(&document);

            let mut cases = [TimelineMode::Duration, TimelineMode::Actual].map(|mode| {
                let axis = AxisId {
                    document_generation: projection.projection_lineage(),
                    geometry_revision: projection.revision(),
                    mode,
                };
                let initial = TimelineModelCache::new(
                    TimelineCacheIdentity {
                        axis,
                        change_revision: projection.change_revision(),
                    },
                    &projection.records,
                    TimelineView {
                        viewport: None,
                        selection: None,
                        render_width_px: 1_500.0,
                    },
                    None,
                );
                let domain = initial.geometry.as_ref().unwrap().domain;
                let viewport = AxisRange {
                    axis,
                    range: DomainRange::new(
                        domain.start + domain.width() * 0.1,
                        domain.start + domain.width() * 0.9,
                    ),
                };
                let selection = AxisRange {
                    axis,
                    range: DomainRange::new(
                        domain.start + domain.width() * 0.25,
                        domain.start + domain.width() * 0.55,
                    ),
                };
                let cache = TimelineModelCache::new(
                    TimelineCacheIdentity {
                        axis,
                        change_revision: projection.change_revision(),
                    },
                    &projection.records,
                    TimelineView {
                        viewport: Some(viewport),
                        selection: Some(selection),
                        render_width_px: 1_500.0,
                    },
                    None,
                );
                (cache, viewport, selection)
            });

            let delta = document
                .apply_batch(vec![events[completion_index].clone()])
                .unwrap();
            projection = TrajectoryProjection::after_delta(&document, &delta, &projection);
            for (cache, viewport, selection) in &mut cases {
                assert_ne!(projection.revision(), selection.axis.geometry_revision);
                if let Some(focus_changed) = cache.sync_projection(&projection) {
                    cache.sync_focus(&projection.records, Some(*selection), focus_changed);
                } else {
                    *cache = TimelineModelCache::new(
                        TimelineCacheIdentity {
                            axis: AxisId {
                                document_generation: projection.projection_lineage(),
                                geometry_revision: projection.revision(),
                                mode: selection.axis.mode,
                            },
                            change_revision: projection.change_revision(),
                        },
                        &projection.records,
                        TimelineView {
                            viewport: Some(*viewport),
                            selection: Some(*selection),
                            render_width_px: 1_500.0,
                        },
                        None,
                    );
                }

                let geometry = cache.geometry.as_ref().unwrap();
                let rebound_selection = cache
                    .display_selection(Some(*selection))
                    .expect("same-session timed selection must survive completion timing");
                assert_eq!(rebound_selection.axis, geometry.axis);
                assert_eq!(
                    rebound_selection.range,
                    selection.range.clamp_to(geometry.domain)
                );
                assert!(cache.focus.is_some());
                let rebound_viewport = viewport.range.clamp_to(geometry.domain);
                assert_eq!(cache.resolved_viewport(Some(*viewport)), rebound_viewport);
                assert_eq!(cache.model.as_ref().unwrap().viewport, rebound_viewport);
            }
        }
    }

    #[test]
    fn selection_focus_is_materialized_once_for_overview_and_ledger() {
        let projection_lineage = 91;
        let revision = 4;
        let records = (0..100)
            .map(|index| Arc::new(record(index, index, index.saturating_add(100))))
            .collect::<im::Vector<_>>();
        let axis = AxisId {
            document_generation: projection_lineage,
            geometry_revision: revision,
            mode: TimelineMode::Sequence,
        };
        let selection = AxisRange {
            axis,
            range: DomainRange::new(10.0, 20.0),
        };
        let mut cache = TimelineModelCache::new(
            cache_identity(projection_lineage, revision, TimelineMode::Sequence),
            &records,
            TimelineView {
                viewport: None,
                selection: Some(selection),
                render_width_px: 1_500.0,
            },
            None,
        );

        cache.sync_focus(&records, Some(selection), false);
        let first = cache.focus.as_ref().unwrap();
        let record_indices = Arc::clone(&first.record_indices);
        cache.sync_focus(&records, Some(selection), false);
        let second = cache.focus.as_ref().unwrap();

        assert!(Arc::ptr_eq(&record_indices, &second.record_indices));
    }

    #[test]
    fn focus_interval_prefixes_exclude_system_records_only_for_turn_summaries() {
        let mut system = record(2, 100, 200);
        system.kind = TrajectoryKind::System;
        let mut trailing_system = record(5, 400, 500);
        trailing_system.kind = TrajectoryKind::System;
        let records = [
            record(1, 0, 100),
            system,
            record(3, 200, 300),
            record(4, 300, 400),
            trailing_system,
        ]
        .into_iter()
        .map(Arc::new)
        .collect::<Vector<_>>();
        let axis = AxisId {
            document_generation: 93,
            geometry_revision: 1,
            mode: TimelineMode::Sequence,
        };
        let focus = TimelineFocusCache::new(
            AxisRange {
                axis,
                range: DomainRange::new(1.0, 4.0),
            },
            &records,
            HashSet::from([1, 3]),
        );

        assert!(focus.intersects(0, 2));
        assert!(!focus.intersects_non_system(0, 2));
        assert!(focus.intersects(2, 4));
        assert!(focus.intersects_non_system(2, 4));
        assert!(!focus.intersects(0, 0));
        assert!(!focus.intersects(records.len(), usize::MAX));
    }

    #[test]
    fn changed_matches_rescan_each_affected_cluster_once() {
        let count = 256;
        let records = (0..count)
            .map(|index| Arc::new(record(index as u64, index as u64, index as u64 + 100)))
            .collect::<Vector<_>>();
        let axis = AxisId {
            document_generation: 94,
            geometry_revision: 1,
            mode: TimelineMode::Sequence,
        };
        let domain = DomainRange::new(0.0, count as f64);
        let model = TimelineModel {
            axis,
            domain,
            viewport: domain,
            render_width_px: 1_500.0,
            cells: vec![RenderCell {
                ids: RenderIdentity::explicit((0..count).collect()),
                lane: TimelineLane::Tools,
                start_px: 0.0,
                end_px: 1_500.0,
                nested: None,
                clustered: true,
            }],
        };
        let geometry = super::timeline_geometry_from_iter(records.iter(), axis).unwrap();
        let mut cache = TimelineSearchCache::build_records(&records, 1, "absent", false, false);
        cache.sync_model_matches(&geometry, &model, 7, &Vector::new());
        cache.matching_indices = TimelineMatches::Filtered([count - 1].into_iter().collect());
        let changed = (0..count).collect::<Vector<_>>();

        cache.sync_model_matches(&geometry, &model, 7, &changed);
        assert_eq!(cache.matched_cell_rescans, 1);
        assert!(matches!(
            cache.matched_cells,
            TimelineCellMatches::Filtered(ref cells) if cells.contains(&0)
        ));

        cache.matching_indices = TimelineMatches::Filtered(ImHashSet::new());
        cache.sync_model_matches(&geometry, &model, 7, &changed);
        assert_eq!(cache.matched_cell_rescans, 2);
        assert!(matches!(
            cache.matched_cells,
            TimelineCellMatches::Filtered(ref cells) if cells.is_empty()
        ));
    }

    #[test]
    fn retained_search_invalidates_its_cell_projection_in_a_new_model_cache() {
        let records = [record(1, 0, 100), record(2, 100, 200)]
            .into_iter()
            .map(Arc::new)
            .collect::<im::Vector<_>>();
        let mut first = TimelineModelCache::new(
            cache_identity(92, 1, TimelineMode::Sequence),
            &records,
            TimelineView {
                viewport: None,
                selection: None,
                render_width_px: 1_500.0,
            },
            Some(TimelineSearchCache::build_records(
                &records, 1, "tool", false, false,
            )),
        );
        let model = first.model.as_ref().unwrap();
        let geometry = first.geometry.as_ref().unwrap();
        first.search.as_mut().unwrap().sync_model_matches(
            geometry,
            model,
            first.model_revision,
            &Vector::new(),
        );
        assert_eq!(
            first.search.as_ref().unwrap().matched_model_revision,
            first.model_revision
        );

        let second = TimelineModelCache::new(
            cache_identity(92, 1, TimelineMode::Duration),
            &records,
            TimelineView {
                viewport: None,
                selection: None,
                render_width_px: 1_500.0,
            },
            first.search.take(),
        );
        assert_eq!(
            second.search.as_ref().unwrap().matched_model_revision,
            u64::MAX
        );
    }

    #[test]
    fn timeline_cache_does_not_rebind_ranges_across_lineage_or_mode() {
        let projection_lineage = 18;
        let records = [record(1, 0, 100)]
            .into_iter()
            .map(Arc::new)
            .collect::<im::Vector<_>>();
        let foreign_lineage_axis = AxisId {
            document_generation: projection_lineage - 1,
            geometry_revision: 3,
            mode: TimelineMode::Duration,
        };
        let cache = TimelineModelCache::new(
            cache_identity(projection_lineage, 4, TimelineMode::Duration),
            &records,
            TimelineView {
                viewport: Some(AxisRange {
                    axis: foreign_lineage_axis,
                    range: DomainRange::new(20.0, 40.0),
                }),
                selection: Some(AxisRange {
                    axis: foreign_lineage_axis,
                    range: DomainRange::new(25.0, 30.0),
                }),
                render_width_px: 1_500.0,
            },
            None,
        );

        let model = cache.model.as_ref().unwrap();
        assert_eq!(model.viewport, model.domain);
        assert_eq!(
            cache.display_selection(Some(AxisRange {
                axis: foreign_lineage_axis,
                range: DomainRange::new(25.0, 30.0),
            })),
            None
        );
        assert_eq!(
            cache.display_selection(Some(AxisRange {
                axis: AxisId {
                    document_generation: projection_lineage,
                    geometry_revision: 3,
                    mode: TimelineMode::Actual,
                },
                range: DomainRange::new(25.0, 30.0),
            })),
            None
        );
    }

    #[test]
    fn viewport_and_selection_are_clamped_and_normalized() {
        let domain = DomainRange::new(0.0, 20.0);
        assert_eq!(
            DomainRange::new(-5.0, 5.0).clamp_to(domain),
            DomainRange::new(0.0, 10.0)
        );
        assert_eq!(
            normalized_range(DomainRange::new(5.0, 10.0), domain),
            (0.25, 0.25)
        );
    }

    #[test]
    fn timeline_modes_keep_distinct_coordinate_semantics() {
        let records = [record(1, 0, 100), record(2, 200, 250)];
        let sequence = timeline_model(&records, TimelineMode::Sequence, None).unwrap();
        assert_eq!(sequence.domain, DomainRange::new(0.0, 2.0));
        let actual = timeline_model(&records, TimelineMode::Actual, None).unwrap();
        assert_eq!(actual.domain, DomainRange::new(1_000.0, 1_250.0));
        let duration = timeline_model(&records, TimelineMode::Duration, None).unwrap();
        assert_eq!(duration.domain, DomainRange::new(0.0, 150.0));
    }

    #[test]
    fn actual_timeline_nests_execution_inside_tool_lifecycle() {
        let model = timeline_model(&[record(1, 0, 100)], TimelineMode::Actual, None).unwrap();
        let cell = &model.cells[0];
        assert!((cell.start_px - 0.0).abs() < 0.000_001);
        assert!((cell.end_px - model.render_width_px).abs() < 0.000_001);
        let (left, width) = nested_segment_geometry(cell).unwrap();
        assert!((left - 0.2).abs() < 0.000_001);
        assert!((width - 0.6).abs() < 0.000_001);
    }

    #[test]
    fn timeline_hover_hits_only_bars_and_prefers_the_topmost_record() {
        let records = [record(1, 0, 100), record(2, 0, 100)];
        let geometry = timeline_geometry(
            &records,
            AxisId {
                document_generation: 1,
                geometry_revision: 1,
                mode: TimelineMode::Actual,
            },
        )
        .unwrap();
        let model = timeline_model(&records, TimelineMode::Actual, None).unwrap();

        for lane in [
            TimelineLane::Input,
            TimelineLane::Model,
            TimelineLane::Tools,
        ] {
            let top = timeline_lane_top(lane) + TIMELINE_BAR_OFFSET;
            assert_eq!(timeline_lane_at(top), Some(lane));
            assert_eq!(timeline_lane_at(top + TIMELINE_BAR_HEIGHT), Some(lane));
        }
        assert_eq!(timeline_lane_at(7.0), Some(TimelineLane::Input));
        assert_eq!(timeline_lane_at(20.0), None);
        assert_eq!(timeline_lane_at(35.0), Some(TimelineLane::Tools));
        assert_eq!(geometry.hit_test(TimelineLane::Tools, 1_050.0), Some(&1));
        assert_eq!(geometry.hit_test(TimelineLane::Model, 1_050.0), None);
        assert_eq!(model.hit_test(TimelineLane::Tools, 0.5), Some(1));
    }

    #[test]
    fn assistant_hover_tooltip_uses_dsh_timing_shape() {
        let mut assistant = record(1, 0, 100);
        assistant.kind = TrajectoryKind::Assistant;
        assistant.timing.first_token = Some((&time(20)).into());

        let tooltip = record_tooltip(&assistant);
        assert!(tooltip.starts_with("ASSISTANT\n"));
        assert!(tooltip.contains(" → "));
        assert!(tooltip.contains("Total 100 ms"));
        assert!(tooltip.contains("TTFT 20 ms · Decoding 80 ms"));
    }

    #[test]
    fn timing_labels_follow_dsh_rounding_and_missing_value_rules() {
        assert_eq!(
            super::format_assistant_duration(Some(999_400_000)),
            "999 ms"
        );
        assert_eq!(
            super::format_assistant_duration(Some(1_234_000_000)),
            "1.23 s"
        );
        assert_eq!(
            super::format_assistant_duration(Some(12_340_000_000)),
            "12.3 s"
        );
        assert_eq!(
            super::format_elapsed_duration(Some(1_234_600_000)),
            "1,235 ms"
        );
        assert_eq!(super::format_elapsed_duration(None), "—");
    }

    #[test]
    fn details_tabs_follow_dsh_role_descriptors_without_inventing_data() {
        let shape = |record: &TrajectoryRecord| {
            super::relevant_record_tabs(record, None)
                .into_iter()
                .map(|descriptor| (descriptor.tab, descriptor.label))
                .collect::<Vec<_>>()
        };

        let mut system = record(1, 0, 100);
        system.kind = TrajectoryKind::System;
        assert_eq!(
            shape(&system),
            vec![
                (DetailsTab::SystemPrompt, "System Prompt"),
                (DetailsTab::Tools, "Tools"),
            ]
        );

        let mut assistant = record(2, 0, 100);
        assistant.kind = TrajectoryKind::Assistant;
        assert_eq!(
            shape(&assistant),
            vec![
                (DetailsTab::Summary, "Summary"),
                (DetailsTab::Preview, "Preview"),
                (DetailsTab::Raw, "Raw"),
            ]
        );

        let mut tool = record(3, 0, 100);
        tool.text.clear();
        tool.payload = None;
        assert_eq!(
            shape(&tool),
            vec![
                (DetailsTab::Summary, "Summary"),
                (DetailsTab::Schema, "Schema"),
                (DetailsTab::Timing, "Timing"),
            ]
        );
        tool.payload = Some(r#"{"path":"README.md"}"#.into());
        tool.text = "done".into();
        assert_eq!(
            shape(&tool),
            vec![
                (DetailsTab::Summary, "Summary"),
                (DetailsTab::Payload, "Payload"),
                (DetailsTab::Result, "Result"),
                (DetailsTab::Schema, "Schema"),
                (DetailsTab::Timing, "Timing"),
            ]
        );

        let mut compaction = record(4, 0, 100);
        compaction.kind = TrajectoryKind::Compaction;
        assert_eq!(
            shape(&compaction),
            vec![
                (DetailsTab::Summary, "Summary"),
                (DetailsTab::Raw, "Raw Output"),
            ]
        );
    }

    #[test]
    fn canonical_prompt_schema_and_request_options_drive_inspector_tabs() {
        let document = SessionDocument::from_events(fixture()).unwrap();
        let projection = TrajectoryProjection::from_document(&document);

        let system = projection
            .records
            .iter()
            .find(|record| record.kind == TrajectoryKind::System)
            .unwrap();
        let system_details = projection.record_details(&system.id);
        let system_tabs = super::relevant_record_tabs(system, system_details)
            .into_iter()
            .map(|descriptor| descriptor.tab)
            .collect::<Vec<_>>();
        assert!(system_tabs.contains(&DetailsTab::SystemPrompt));
        assert!(system_tabs.contains(&DetailsTab::Tools));

        let tool = projection
            .records
            .iter()
            .find(|record| record.kind == TrajectoryKind::Tool)
            .unwrap();
        let tool_details = projection.record_details(&tool.id);
        let tool_tabs = super::relevant_record_tabs(tool, tool_details)
            .into_iter()
            .map(|descriptor| descriptor.tab)
            .collect::<Vec<_>>();
        assert!(tool_tabs.contains(&DetailsTab::Schema));

        let request = projection.requests.front().unwrap();
        let request_tabs = super::relevant_request_tabs(request)
            .into_iter()
            .map(|descriptor| descriptor.tab)
            .collect::<Vec<_>>();
        assert_eq!(
            request_tabs,
            vec![
                DetailsTab::Summary,
                DetailsTab::Options,
                DetailsTab::Usage,
                DetailsTab::Timing,
            ]
        );
    }

    #[test]
    fn assistant_timing_missing_states_use_dsh_precedence() {
        let mut assistant = record(1, 0, 100);
        assistant.kind = TrajectoryKind::Assistant;
        assistant.timing = RecordTiming::default();

        assert_eq!(super::timing_duration(&assistant), "Not recorded");
        assert_eq!(super::assistant_ttft(&assistant), "Not recorded");
        assert_eq!(
            super::assistant_generation(&assistant),
            "First token unavailable"
        );
        assert_eq!(super::assistant_throughput(&assistant), "Usage unavailable");

        assistant.usage = Some(TokenUsage::default());
        assistant.timing.completed = Some((&time(100)).into());
        assert_eq!(super::timing_duration(&assistant), "Step start unavailable");
        assert_eq!(super::assistant_ttft(&assistant), "Step start unavailable");
        assert_eq!(
            super::assistant_throughput(&assistant),
            "First token unavailable"
        );

        assistant.timing = RecordTiming {
            started: Some((&time(0)).into()),
            first_token: Some((&time(20)).into()),
            ..RecordTiming::default()
        };
        assert_eq!(super::timing_duration(&assistant), "Pending");
        assert_eq!(super::assistant_ttft(&assistant), "20 ms");
        assert_eq!(super::assistant_generation(&assistant), "Pending");
        assert_eq!(super::assistant_throughput(&assistant), "Pending");

        assistant.timing.completed = Some((&time(100)).into());
        assert_eq!(super::assistant_generation(&assistant), "80 ms");
        assert_eq!(super::assistant_throughput(&assistant), "0.0 tok/s");
    }

    #[test]
    fn details_default_width_matches_dsh_clamp() {
        assert_eq!(super::trajectory_details_default_width(761.0), 320.0);
        assert_eq!(super::trajectory_details_default_width(900.0), 342.0);
        assert_eq!(super::trajectory_details_default_width(1_500.0), 440.0);
    }

    #[test]
    fn fold_projection_and_double_click_target_only_the_requested_group() {
        let mut first_assistant = record(1, 0, 100);
        first_assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-1"));
        first_assistant.kind = TrajectoryKind::Assistant;
        let mut first_tool = record(2, 100, 200);
        first_tool.turn = Some(1);
        let mut second_assistant = record(3, 200, 300);
        second_assistant.id = TrajectoryItemId::Assistant(RequestId::from("request-2"));
        second_assistant.kind = TrajectoryKind::Assistant;
        second_assistant.turn = Some(2);
        let mut second_tool = record(4, 300, 400);
        second_tool.turn = Some(2);
        let records = [first_assistant, first_tool, second_assistant, second_tool]
            .into_iter()
            .map(Arc::new)
            .collect::<Vector<_>>();

        let collapsed_turns = HashSet::from([1]);
        let collapsed_assistants = HashSet::from([records[2].id.clone()]);
        let rows = super::project_ledger_rows(
            &records,
            &super::TimelineMatches::All(records.len()),
            false,
            &collapsed_turns,
            &collapsed_assistants,
        );
        assert!(matches!(
            rows.get(1),
            Some(TimelineLedgerRow::TurnSummary { turn: 1, .. })
        ));
        assert_eq!(rows.get(2), Some(TimelineLedgerRow::Record(2)));
        assert!(matches!(
            rows.get(3),
            Some(TimelineLedgerRow::CallsSummary { assistant: 2, .. })
        ));
        let collapsible_turns = HashSet::from([1, 2]);
        let collapsible_assistants = HashSet::from([records[0].id.clone(), records[2].id.clone()]);

        assert_eq!(
            super::ledger_double_click_target(
                &records[0],
                true,
                &collapsed_turns,
                &collapsible_turns,
                &collapsible_assistants,
            ),
            Some(super::LedgerFoldTarget::Turn(1))
        );
        assert_eq!(
            super::ledger_double_click_target(
                &records[2],
                true,
                &collapsed_turns,
                &collapsible_turns,
                &collapsible_assistants,
            ),
            Some(super::LedgerFoldTarget::Assistant(records[2].id.clone()))
        );
    }

    #[test]
    fn clipped_nested_segment_does_not_create_an_invalid_width_range() {
        let cell = RenderCell {
            ids: RenderIdentity::explicit(vec![0]),
            lane: TimelineLane::Tools,
            start_px: 0.0,
            end_px: 100.0,
            nested: Some((100.0, 120.0)),
            clustered: false,
        };
        assert_eq!(nested_segment_geometry(&cell), None);
    }

    #[test]
    fn collapsed_ledger_rows_use_dsh_height_and_summary_copy() {
        let turn = TimelineLedgerRow::TurnSummary {
            representative: 0,
            turn: 1,
            first_hidden: 1,
            last_hidden: 2,
            step_ids: ImHashSet::new(),
            call_count: 2,
        };
        let calls = TimelineLedgerRow::CallsSummary {
            assistant: 0,
            first_tool: 1,
            last_tool: 2,
            tool_names: ImHashSet::new(),
            tools: Arc::from("bash · read"),
            tools_truncated: false,
        };

        assert_eq!(
            trajectory_ledger_row_height(&TimelineLedgerRow::Record(0)),
            30.0
        );
        assert_eq!(
            trajectory_ledger_row_height(&TimelineLedgerRow::RequestBoundary {
                request: 4,
                run_index: 0,
                terminal: true,
            }),
            9.0
        );
        assert_eq!(
            trajectory_ledger_row_height(&TimelineLedgerRow::RequestBoundary {
                request: 3,
                run_index: 0,
                terminal: false,
            }),
            0.0
        );
        assert_eq!(trajectory_ledger_row_height(&turn), 20.0);
        assert_eq!(trajectory_ledger_row_height(&calls), 20.0);
        assert_eq!(turn_summary_text(27, 8), "… 27 steps · 8 tool calls");
        assert_eq!(turn_summary_text(1, 1), "… 1 step · 1 tool call");
        assert_eq!(
            calls_summary_text(2, "bash · read"),
            "… 2 tool calls · bash · read"
        );
        assert_eq!(calls_summary_text(1, ""), "… 1 tool call");
    }

    #[test]
    fn pending_request_decorates_rows_without_materializing_identity_rows() {
        let rows = TimelineRows::All(100_000).with_request_boundaries(vec![
            super::RequestBoundaryPlacement {
                output_row: 100_000,
                request: 7,
                run_index: 0,
                terminal: true,
            },
        ]);
        assert_eq!(rows.len(), 100_001);
        assert_eq!(rows.get(99_999), Some(TimelineLedgerRow::Record(99_999)));
        assert_eq!(
            rows.get(100_000),
            Some(TimelineLedgerRow::RequestBoundary {
                request: 7,
                run_index: 0,
                terminal: true,
            })
        );
        assert_eq!(rows.get(100_001), None);
    }

    #[test]
    fn variable_ledger_list_preserves_and_restores_logical_scroll() {
        let state = gpui::ListState::new(4, gpui::ListAlignment::Top, gpui::px(100.0));
        state.scroll_to(gpui::ListOffset {
            item_ix: 2,
            offset_in_item: gpui::px(7.0),
        });

        sync_trajectory_list_state(&state, 7, false, None, false);
        assert_eq!(state.item_count(), 7);
        assert_eq!(state.logical_scroll_top().item_ix, 2);
        assert_eq!(state.logical_scroll_top().offset_in_item, gpui::px(7.0));

        sync_trajectory_list_state(
            &state,
            3,
            true,
            Some(gpui::ListOffset {
                item_ix: 1,
                offset_in_item: gpui::px(4.0),
            }),
            false,
        );
        assert_eq!(state.item_count(), 3);
        assert_eq!(state.logical_scroll_top().item_ix, 1);
        assert_eq!(state.logical_scroll_top().offset_in_item, gpui::px(4.0));
    }

    #[test]
    fn variable_ledger_list_follows_the_tail_only_when_enabled() {
        let state = gpui::ListState::new(0, gpui::ListAlignment::Bottom, gpui::px(100.0));
        sync_trajectory_list_state(&state, 4, true, None, true);
        assert_eq!(state.logical_scroll_top().item_ix, 4);

        sync_trajectory_list_state(&state, 7, false, None, true);
        assert_eq!(state.logical_scroll_top().item_ix, 7);

        state.scroll_to(gpui::ListOffset {
            item_ix: 2,
            offset_in_item: gpui::px(3.0),
        });
        sync_trajectory_list_state(&state, 8, false, None, false);
        assert_eq!(state.logical_scroll_top().item_ix, 2);
        assert_eq!(state.logical_scroll_top().offset_in_item, gpui::px(3.0));
    }

    #[test]
    fn variable_ledger_scroll_alignment_accounts_for_summary_height() {
        let rows = TimelineRows::Projected(
            [
                TimelineLedgerRow::Record(0),
                TimelineLedgerRow::TurnSummary {
                    representative: 0,
                    turn: 1,
                    first_hidden: 1,
                    last_hidden: 1,
                    step_ids: ImHashSet::new(),
                    call_count: 0,
                },
                TimelineLedgerRow::Record(2),
                TimelineLedgerRow::Record(3),
            ]
            .into_iter()
            .collect(),
        );

        let centered = aligned_trajectory_list_offset(&rows, 3, 80.0, 0.5);
        assert_eq!(centered.item_ix, 2);
        assert_eq!(centered.offset_in_item, gpui::px(5.0));
        let bottom = aligned_trajectory_list_offset(&rows, 3, 80.0, 1.0);
        assert_eq!(bottom.item_ix, 1);
        assert_eq!(bottom.offset_in_item, gpui::px(0.0));
    }

    #[test]
    fn focused_rows_center_small_ranges_and_anchor_large_ranges() {
        let rows = TimelineRows::All(30);
        assert_eq!(
            focus_scroll_target(&[4, 5, 6], &rows, 120.0),
            Some((5, ScrollStrategy::Center))
        );
        assert_eq!(
            focus_scroll_target(&(10..24).collect::<Vec<_>>(), &rows, 300.0),
            Some((10, ScrollStrategy::Top))
        );
    }

    #[test]
    fn ledger_item_inside_a_focused_range_keeps_the_selection() {
        let focused = HashSet::from([4, 5, 6]);
        assert!(!should_clear_selection_for_record(
            TrajectorySelectionSource::Ledger,
            Some(&focused),
            5,
        ));
        assert!(should_clear_selection_for_record(
            TrajectorySelectionSource::Ledger,
            Some(&focused),
            7,
        ));
        assert!(should_clear_selection_for_record(
            TrajectorySelectionSource::Timeline,
            Some(&focused),
            5,
        ));
    }
}
