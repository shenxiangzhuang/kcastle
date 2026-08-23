use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement, Pixels, Point, ScrollStrategy, ScrollWheelEvent,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder, px, relative,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::Input;
use gpui_component::resizable::{h_resizable, resizable_panel};
use gpui_component::scroll::ScrollableElement;
use gpui_component::tooltip::Tooltip;
use gpui_component::{ElementExt, IconName, Sizable};
use im::{HashMap as ImHashMap, HashSet as ImHashSet, Vector};
use time::{OffsetDateTime, UtcOffset, macros::format_description};

use crate::app::{DesktopApp, TimelineDragState, TimelineHoverState};
use crate::domain::session_document::EventTimeRef;
use crate::domain::timeline::{
    AxisId, AxisRange, DomainRange, TimelineAction, TimelineEffect, TimelineGeometry,
    TimelineInteraction, TimelineLane, TimelinePoint, TimelineSpan,
};
use crate::domain::{
    Action, DetailsTab, TimelineMode, TrajectoryItemId, TrajectoryKind, TrajectoryLane,
    TrajectoryRecord, TrajectoryStatus,
};
use crate::layout::TrajectoryMode;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrajectorySelectionSource {
    Ledger,
    Timeline,
}

#[derive(Debug)]
struct TimelineCell {
    ordinal: usize,
    primary_index: usize,
    hit_id: TrajectoryItemId,
    item_count: usize,
    lane: TimelineLane,
    left: f64,
    width: f64,
    execution_left: Option<f64>,
    execution_width: Option<f64>,
    clustered: bool,
}

#[derive(Debug)]
struct TimelineModel {
    axis: AxisId,
    domain: DomainRange,
    viewport: DomainRange,
    cells: Vec<TimelineCell>,
    lane_cell_indices: [Vec<usize>; 3],
    cell_by_id: HashMap<TrajectoryItemId, usize>,
    cell_by_record_index: Vec<Option<usize>>,
}

impl TimelineModel {
    fn hit_test(&self, lane: TimelineLane, fraction: f64) -> Option<&TrajectoryItemId> {
        self.cells.iter().rev().find_map(|cell| {
            let right = (cell.left + cell.width).min(1.0);
            (cell.lane == lane && fraction >= cell.left && fraction <= right)
                .then_some(&cell.hit_id)
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

    fn len(&self) -> usize {
        match self {
            Self::All(len) => *len,
            Self::Filtered(indices) => indices.len(),
        }
    }
}

#[derive(Clone, Debug)]
enum TimelineRows {
    /// Identity mapping: row N is record N. The overwhelmingly common empty-search/default-filter
    /// state therefore owns no N-element allocation.
    All(usize),
    Filtered(Vector<usize>),
}

impl TimelineRows {
    fn len(&self) -> usize {
        match self {
            Self::All(len) => *len,
            Self::Filtered(indices) => indices.len(),
        }
    }

    fn get(&self, row: usize) -> Option<usize> {
        match self {
            Self::All(len) => (row < *len).then_some(row),
            Self::Filtered(indices) => indices.get(row).copied(),
        }
    }

    fn position(&self, mut predicate: impl FnMut(usize) -> bool) -> Option<usize> {
        (0..self.len()).find(|row| self.get(*row).is_some_and(&mut predicate))
    }
}

#[derive(Clone, Debug)]
struct TimelineFilterSnapshot {
    match_count: usize,
    matched_cells: TimelineCellMatches,
    rows: TimelineRows,
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
    search_revision: u64,
    query: String,
    matching_indices: TimelineMatches,
    rows: TimelineRows,
    collapsed_turns: bool,
    collapsed_calls: bool,
    record_count: usize,
    turn_stats: HashMap<u32, TurnStats>,
    eligible_by_turn: ImHashMap<u32, Vector<usize>>,
    eligible_turn_by_index: ImHashMap<usize, u32>,
    matched_cells: TimelineCellMatches,
    matched_cell_counts: ImHashMap<usize, usize>,
    matched_model_revision: u64,
    changed_matches: Vector<(usize, bool, bool)>,
    #[cfg(test)]
    inspected_records: usize,
    #[cfg(test)]
    materialized_row_rebuilds: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TurnStats {
    item_count: usize,
    call_count: usize,
}

#[derive(Clone, Debug)]
struct TimelineFocusCache {
    selection: AxisRange,
    model_revision: u64,
    record_indices: Arc<HashSet<usize>>,
    turn_indices: Arc<HashSet<u32>>,
    cell_indices: Arc<HashSet<usize>>,
}

#[derive(Debug)]
struct TimelineCacheIdentity {
    workspace: PathBuf,
    session: PathBuf,
    revision: u64,
    projection_lineage: u64,
    mode: TimelineMode,
}

#[derive(Debug)]
pub(crate) struct TimelineModelCache {
    workspace: PathBuf,
    session: PathBuf,
    revision: u64,
    mode: TimelineMode,
    viewport: Option<AxisRange>,
    selection: Option<AxisRange>,
    render_width_px: f64,
    geometry: Option<TimelineGeometry<TrajectoryItemId>>,
    interaction: Option<TimelineInteraction>,
    index_by_id: HashMap<TrajectoryItemId, usize>,
    search: Option<TimelineSearchCache>,
    model: Option<TimelineModel>,
    model_revision: u64,
    focus: Option<TimelineFocusCache>,
    projection_lineage: u64,
}

impl TimelineModelCache {
    fn new(
        identity: TimelineCacheIdentity,
        records: &Vector<std::sync::Arc<TrajectoryRecord>>,
        view: TimelineView,
        mut search: Option<TimelineSearchCache>,
    ) -> Self {
        let TimelineCacheIdentity {
            workspace,
            session,
            revision,
            projection_lineage,
            mode,
        } = identity;
        let axis = AxisId {
            document_generation: document_generation(&workspace, &session, projection_lineage),
            geometry_revision: revision,
            mode,
        };
        let index_by_id = records
            .iter()
            .enumerate()
            .map(|(index, record)| (record.id.clone(), index))
            .collect();
        let geometry = timeline_geometry_from_iter(records.iter(), axis);
        let mut interaction = geometry
            .as_ref()
            .map(|geometry| TimelineInteraction::new(geometry.axis, geometry.domain));
        if let Some(interaction) = &mut interaction {
            interaction.sync_external_ranges(view.viewport, view.selection);
        }
        // `model_revision` is local to one cache instance. A retained content/search cache must
        // never mistake a freshly built LOD model for the prior cache's model with the same local
        // counter value.
        if let Some(search) = &mut search {
            search.matched_model_revision = u64::MAX;
            search.changed_matches.clear();
        }
        let mut cache = Self {
            workspace,
            session,
            revision,
            mode,
            viewport: view.viewport,
            selection: view.selection,
            render_width_px: view.render_width_px,
            geometry,
            interaction,
            index_by_id,
            search,
            model: None,
            model_revision: 0,
            focus: None,
            projection_lineage,
        };
        cache.reproject();
        cache
    }

    fn geometry_matches(
        &self,
        workspace: &Path,
        session: &Path,
        revision: u64,
        projection_lineage: u64,
        mode: TimelineMode,
    ) -> bool {
        self.workspace == workspace
            && self.session == session
            && self.revision == revision
            && self.projection_lineage == projection_lineage
            && self.mode == mode
    }

    fn projection_matches(
        &self,
        workspace: &Path,
        session: &Path,
        projection_lineage: u64,
        mode: TimelineMode,
    ) -> bool {
        self.workspace == workspace
            && self.session == session
            && self.projection_lineage == projection_lineage
            && self.mode == mode
    }

    fn sync_ranges(
        &mut self,
        viewport: Option<AxisRange>,
        selection: Option<AxisRange>,
        render_width_px: f64,
    ) {
        if self.selection != selection {
            self.focus = None;
        }
        let projection_changed =
            self.viewport != viewport || (self.render_width_px - render_width_px).abs() >= 1.0;
        self.viewport = viewport;
        self.selection = selection;
        self.render_width_px = render_width_px;
        if let (Some(interaction), Some(geometry)) = (&mut self.interaction, &self.geometry) {
            interaction.reduce(TimelineAction::ProjectionChanged {
                axis: geometry.axis,
                domain: geometry.domain,
            });
            interaction.sync_external_ranges(viewport, selection);
        }
        if projection_changed {
            self.reproject();
        }
    }

    fn sync_sequence_geometry(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
        changes: crate::domain::TrajectoryGeometryChanges,
    ) -> bool {
        if self.mode != TimelineMode::Sequence {
            return false;
        }
        let axis = AxisId {
            document_generation: document_generation(
                &self.workspace,
                &self.session,
                self.projection_lineage,
            ),
            geometry_revision: projection.revision(),
            mode: self.mode,
        };
        let spans = changes.changed_indices.iter().filter_map(|index| {
            projection
                .records
                .get(*index)
                .map(|record| (*index, timeline_span(*index, record.as_ref())))
        });
        let Some(geometry) = &mut self.geometry else {
            return false;
        };
        let previous_domain = geometry.domain;
        let appended = projection.records.len() > geometry.cells.len();
        if !geometry.update_sequence(axis, projection.records.len(), spans) {
            return false;
        }
        for index in &changes.changed_indices {
            let Some(record) = projection.records.get(*index) else {
                return false;
            };
            self.index_by_id.insert(record.id.clone(), *index);
        }
        self.revision = projection.revision();
        if let Some(interaction) = &mut self.interaction {
            // Sequence coordinates are stable across an explicitly verified incremental
            // transition. Rebind the already validated ranges to the new revision instead of
            // treating them as arbitrary stale external input. A full rebuild still rejects
            // ranges from another geometry revision.
            let previous_viewport = interaction.viewport;
            let previous_selection = interaction.selection;
            interaction.reduce(TimelineAction::PointerCancel);
            interaction.axis = axis;
            interaction.domain = geometry.domain;
            interaction.viewport = if appended && previous_viewport == previous_domain {
                geometry.domain
            } else {
                previous_viewport.clamp_to(geometry.domain)
            };
            interaction.selection = previous_selection.map(|selection| AxisRange {
                axis,
                range: selection.range.clamp_to(geometry.domain),
            });
        }
        if appended {
            // Appending changes the Sequence domain and potentially every projected x position.
            // This is the only compatible Sequence transition that still needs a full LOD pass.
            self.reproject();
        } else {
            self.update_sequence_model_cells(&changes.changed_indices);
            if let (Some(focus), Some(selection)) =
                (&mut self.focus, self.interaction.and_then(|i| i.selection))
            {
                focus.selection = selection;
            }
        }
        true
    }

    fn update_sequence_model_cells(&mut self, changed_indices: &Vector<usize>) {
        let Some(geometry) = &self.geometry else {
            return;
        };
        let Some(interaction) = self.interaction else {
            return;
        };
        let Some(model) = &mut self.model else {
            return;
        };
        model.axis = geometry.axis;
        model.domain = geometry.domain;
        model.viewport = interaction.viewport;
        for index in changed_indices {
            let Some(cell_index) = model
                .cell_by_record_index
                .get(*index)
                .and_then(|cell| *cell)
            else {
                continue;
            };
            let Some(cell) = model.cells.get_mut(cell_index) else {
                continue;
            };
            if cell.clustered {
                continue;
            }
            let nested = geometry
                .cells
                .get(*index)
                .and_then(|geometry_cell| geometry_cell.nested)
                .map(|range| normalized_range(range, interaction.viewport));
            cell.execution_left = nested.map(|value| value.0);
            cell.execution_width = nested.map(|value| value.1);
        }
    }

    fn reduce(&mut self, action: TimelineAction) -> Option<TimelineEffect> {
        let previous_viewport = self
            .interaction
            .as_ref()
            .map(|interaction| interaction.viewport);
        let effect = self
            .interaction
            .as_mut()
            .and_then(|interaction| interaction.reduce(action));
        if self
            .interaction
            .as_ref()
            .is_some_and(|interaction| Some(interaction.viewport) != previous_viewport)
        {
            self.reproject();
        }
        effect
    }

    fn reproject(&mut self) {
        self.model =
            self.geometry
                .as_ref()
                .zip(self.interaction.as_ref())
                .map(|(geometry, interaction)| {
                    project_timeline(
                        geometry,
                        interaction.viewport,
                        self.render_width_px,
                        &self.index_by_id,
                    )
                });
        self.model_revision = self.model_revision.saturating_add(1);
        self.focus = None;
    }

    fn display_selection(&self) -> Option<AxisRange> {
        self.interaction
            .as_ref()
            .and_then(TimelineInteraction::display_selection)
    }

    fn search_snapshot(
        &mut self,
        projection: &crate::domain::TrajectoryProjection,
        query: &str,
        collapsed_turns: bool,
        collapsed_calls: bool,
    ) -> TimelineFilterSnapshot {
        let rebuild = self.search.as_ref().is_none_or(|search| {
            search.query != query
                || search.collapsed_turns != collapsed_turns
                || search.collapsed_calls != collapsed_calls
                || projection
                    .search_changed_indices_since(search.search_revision)
                    .is_none()
        });
        if rebuild {
            self.search = Some(TimelineSearchCache::build(
                projection,
                query,
                collapsed_turns,
                collapsed_calls,
            ));
        } else {
            self.search
                .as_mut()
                .expect("timeline search cache was checked above")
                .sync_incremental(projection);
        }
        let search = self
            .search
            .as_mut()
            .expect("timeline search cache was initialized");
        if let Some(model) = &self.model {
            search.sync_model_matches(model, self.model_revision);
        }
        TimelineFilterSnapshot {
            match_count: search.matching_indices.len(),
            matched_cells: search.matched_cells.clone(),
            rows: search.rows.clone(),
        }
    }

    fn sync_focus(&mut self, records: &Vector<Arc<TrajectoryRecord>>) {
        let Some(selection) = self
            .interaction
            .and_then(|interaction| interaction.selection)
        else {
            self.focus = None;
            return;
        };
        if self.focus.as_ref().is_some_and(|focus| {
            focus.selection == selection && focus.model_revision == self.model_revision
        }) {
            return;
        }
        let Some(geometry) = &self.geometry else {
            self.focus = None;
            return;
        };
        let items = geometry.selection(selection).items;
        let mut record_indices = HashSet::with_capacity(items.len());
        let mut turn_indices = HashSet::new();
        let mut cell_indices = HashSet::new();
        for id in items {
            if let Some(index) = self.index_by_id.get(&id).copied() {
                record_indices.insert(index);
                if let Some(turn) = records.get(index).and_then(|record| record.turn) {
                    turn_indices.insert(turn);
                }
            }
            if let Some(cell) = self
                .model
                .as_ref()
                .and_then(|model| model.cell_by_id.get(&id))
            {
                cell_indices.insert(*cell);
            }
        }
        self.focus = Some(TimelineFocusCache {
            selection,
            model_revision: self.model_revision,
            record_indices: Arc::new(record_indices),
            turn_indices: Arc::new(turn_indices),
            cell_indices: Arc::new(cell_indices),
        });
    }
}

impl TimelineSearchCache {
    fn build(
        projection: &crate::domain::TrajectoryProjection,
        query: &str,
        collapsed_turns: bool,
        collapsed_calls: bool,
    ) -> Self {
        Self::build_records(
            &projection.records,
            projection.search_revision(),
            query,
            collapsed_turns,
            collapsed_calls,
        )
    }

    fn build_records(
        records: &Vector<Arc<TrajectoryRecord>>,
        search_revision: u64,
        query: &str,
        collapsed_turns: bool,
        collapsed_calls: bool,
    ) -> Self {
        let record_count = records.len();
        if query.is_empty() && !collapsed_turns && !collapsed_calls {
            return Self {
                search_revision,
                query: String::new(),
                matching_indices: TimelineMatches::All(record_count),
                rows: TimelineRows::All(record_count),
                collapsed_turns,
                collapsed_calls,
                record_count,
                turn_stats: HashMap::new(),
                eligible_by_turn: ImHashMap::new(),
                eligible_turn_by_index: ImHashMap::new(),
                matched_cells: TimelineCellMatches::All,
                matched_cell_counts: ImHashMap::new(),
                matched_model_revision: 0,
                changed_matches: Vector::new(),
                #[cfg(test)]
                inspected_records: 0,
                #[cfg(test)]
                materialized_row_rebuilds: 0,
            };
        }

        let mut cache = Self {
            search_revision,
            query: query.to_owned(),
            matching_indices: if query.is_empty() {
                TimelineMatches::All(record_count)
            } else {
                TimelineMatches::Filtered(ImHashSet::new())
            },
            rows: TimelineRows::Filtered(Vector::new()),
            collapsed_turns,
            collapsed_calls,
            record_count,
            turn_stats: HashMap::new(),
            eligible_by_turn: ImHashMap::new(),
            eligible_turn_by_index: ImHashMap::new(),
            matched_cells: TimelineCellMatches::Filtered(ImHashSet::new()),
            matched_cell_counts: ImHashMap::new(),
            matched_model_revision: 0,
            changed_matches: Vector::new(),
            #[cfg(test)]
            inspected_records: 0,
            #[cfg(test)]
            materialized_row_rebuilds: 1,
        };
        for (index, record) in records.iter().enumerate() {
            #[cfg(test)]
            {
                cache.inspected_records = cache.inspected_records.saturating_add(1);
            }
            if !query.is_empty()
                && record.matches(query)
                && let TimelineMatches::Filtered(indices) = &mut cache.matching_indices
            {
                indices.insert(index);
            }
            if collapsed_turns && let Some(turn) = record.turn {
                let stats = cache.turn_stats.entry(turn).or_default();
                stats.item_count = stats.item_count.saturating_add(1);
                stats.call_count = stats
                    .call_count
                    .saturating_add((record.kind == TrajectoryKind::Tool) as usize);
            }
            if cache.matching_indices.contains(&index) {
                cache.add_row(index, record);
            }
        }
        cache
    }

    fn sync_incremental(&mut self, projection: &crate::domain::TrajectoryProjection) {
        self.changed_matches.clear();
        if self.search_revision == projection.search_revision() {
            self.record_count = projection.records.len();
            if matches!(self.matching_indices, TimelineMatches::All(_)) {
                self.matching_indices = TimelineMatches::All(self.record_count);
            }
            if matches!(self.rows, TimelineRows::All(_)) {
                self.rows = TimelineRows::All(self.record_count);
            }
            return;
        }
        let Some(changed) = projection.search_changed_indices_since(self.search_revision) else {
            *self = Self::build(
                projection,
                &self.query,
                self.collapsed_turns,
                self.collapsed_calls,
            );
            return;
        };
        self.sync_changed_records(&projection.records, projection.search_revision(), changed);
    }

    fn sync_changed_records(
        &mut self,
        records: &Vector<Arc<TrajectoryRecord>>,
        search_revision: u64,
        changed: impl IntoIterator<Item = usize>,
    ) {
        if self.query.is_empty() && !self.collapsed_turns && !self.collapsed_calls {
            self.record_count = records.len();
            self.matching_indices = TimelineMatches::All(self.record_count);
            self.rows = TimelineRows::All(self.record_count);
            self.search_revision = search_revision;
            self.matched_cells = TimelineCellMatches::All;
            self.matched_cell_counts.clear();
            self.changed_matches.clear();
            return;
        }
        self.changed_matches.clear();
        let mut visited = HashSet::new();
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
                *self = Self::build_records(
                    records,
                    search_revision,
                    &self.query,
                    self.collapsed_turns,
                    self.collapsed_calls,
                );
                return;
            };
            #[cfg(test)]
            {
                self.inspected_records = self.inspected_records.saturating_add(1);
            }
            self.remove_row(index);
            let matched_before = self.matching_indices.contains(&index);
            if let TimelineMatches::Filtered(indices) = &mut self.matching_indices {
                if record.matches(&self.query) {
                    indices.insert(index);
                } else {
                    indices.remove(&index);
                }
            }
            let matched_after = self.matching_indices.contains(&index);
            if matched_before != matched_after {
                self.changed_matches
                    .push_back((index, matched_before, matched_after));
            }
            if appended
                && self.collapsed_turns
                && let Some(turn) = record.turn
            {
                let stats = self.turn_stats.entry(turn).or_default();
                stats.item_count = stats.item_count.saturating_add(1);
                stats.call_count = stats
                    .call_count
                    .saturating_add((record.kind == TrajectoryKind::Tool) as usize);
            }
            if self.matching_indices.contains(&index) {
                self.add_row(index, record);
            }
        }
        self.record_count = records.len();
        if matches!(self.matching_indices, TimelineMatches::All(_)) {
            self.matching_indices = TimelineMatches::All(self.record_count);
        }
        if matches!(self.rows, TimelineRows::All(_)) {
            self.rows = TimelineRows::All(self.record_count);
        }
        self.search_revision = search_revision;
    }

    fn sync_model_matches(&mut self, model: &TimelineModel, model_revision: u64) {
        if matches!(self.matching_indices, TimelineMatches::All(_)) {
            self.matched_cells = TimelineCellMatches::All;
            self.matched_cell_counts.clear();
            self.matched_model_revision = model_revision;
            self.changed_matches.clear();
            return;
        }
        if self.matched_model_revision != model_revision {
            let mut counts = ImHashMap::<usize, usize>::new();
            let mut cells = ImHashSet::new();
            if let TimelineMatches::Filtered(indices) = &self.matching_indices {
                for index in indices {
                    let Some(cell) = model
                        .cell_by_record_index
                        .get(*index)
                        .and_then(|cell| *cell)
                    else {
                        continue;
                    };
                    let count = counts.get(&cell).copied().unwrap_or_default() + 1;
                    counts.insert(cell, count);
                    cells.insert(cell);
                }
            }
            self.matched_cell_counts = counts;
            self.matched_cells = TimelineCellMatches::Filtered(cells);
            self.matched_model_revision = model_revision;
            self.changed_matches.clear();
            return;
        }
        let TimelineCellMatches::Filtered(cells) = &mut self.matched_cells else {
            return;
        };
        for (index, matched_before, matched_after) in std::mem::take(&mut self.changed_matches) {
            let Some(cell) = model.cell_by_record_index.get(index).and_then(|cell| *cell) else {
                continue;
            };
            let before = self
                .matched_cell_counts
                .get(&cell)
                .copied()
                .unwrap_or_default();
            let after = if matched_before && !matched_after {
                before.saturating_sub(1)
            } else if !matched_before && matched_after {
                before.saturating_add(1)
            } else {
                before
            };
            if after == 0 {
                self.matched_cell_counts.remove(&cell);
                cells.remove(&cell);
            } else {
                self.matched_cell_counts.insert(cell, after);
                cells.insert(cell);
            }
        }
    }

    fn row_eligible(&self, record: &TrajectoryRecord) -> bool {
        !self.collapsed_calls || record.kind != TrajectoryKind::Tool
    }

    fn add_row(&mut self, index: usize, record: &TrajectoryRecord) {
        if !self.row_eligible(record) {
            return;
        }
        if !self.collapsed_turns {
            if let TimelineRows::Filtered(rows) = &mut self.rows {
                sorted_insert(rows, index);
            }
            return;
        }
        let Some(turn) = record.turn else {
            if let TimelineRows::Filtered(rows) = &mut self.rows {
                sorted_insert(rows, index);
            }
            return;
        };
        let candidates = self.eligible_by_turn.entry(turn).or_default();
        let previous = candidates.front().copied();
        sorted_insert(candidates, index);
        self.eligible_turn_by_index.insert(index, turn);
        let current = candidates.front().copied();
        if previous != current
            && let TimelineRows::Filtered(rows) = &mut self.rows
        {
            if let Some(previous) = previous {
                sorted_remove(rows, previous);
            }
            if let Some(current) = current {
                sorted_insert(rows, current);
            }
        }
    }

    fn remove_row(&mut self, index: usize) {
        if !self.collapsed_turns {
            if let TimelineRows::Filtered(rows) = &mut self.rows {
                sorted_remove(rows, index);
            }
            return;
        }
        let Some(turn) = self.eligible_turn_by_index.remove(&index) else {
            if let TimelineRows::Filtered(rows) = &mut self.rows {
                sorted_remove(rows, index);
            }
            return;
        };
        let Some(candidates) = self.eligible_by_turn.get_mut(&turn) else {
            return;
        };
        let previous = candidates.front().copied();
        sorted_remove(candidates, index);
        let current = candidates.front().copied();
        if candidates.is_empty() {
            self.eligible_by_turn.remove(&turn);
        }
        if previous != current
            && let TimelineRows::Filtered(rows) = &mut self.rows
        {
            if let Some(previous) = previous {
                sorted_remove(rows, previous);
            }
            if let Some(current) = current {
                sorted_insert(rows, current);
            }
        }
    }
}

fn sorted_position(values: &Vector<usize>, needle: usize) -> Result<usize, usize> {
    let mut left = 0;
    let mut right = values.len();
    while left < right {
        let middle = left + (right - left) / 2;
        match values[middle].cmp(&needle) {
            std::cmp::Ordering::Less => left = middle + 1,
            std::cmp::Ordering::Greater => right = middle,
            std::cmp::Ordering::Equal => return Ok(middle),
        }
    }
    Err(left)
}

fn sorted_insert(values: &mut Vector<usize>, value: usize) {
    if let Err(index) = sorted_position(values, value) {
        values.insert(index, value);
    }
}

fn sorted_remove(values: &mut Vector<usize>, value: usize) {
    if let Ok(index) = sorted_position(values, value) {
        values.remove(index);
    }
}

impl DesktopApp {
    pub(crate) fn trajectory_panel(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let filter = self.timeline_filter_snapshot(&query);
        let selected = self
            .core
            .details
            .selected
            .as_ref()
            .and_then(|selected| self.core.session_view.trajectory.record_index(selected));
        let narrow_details = self.core.layout.trajectory == TrajectoryMode::Overlay;

        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .overflow_hidden()
            .bg(trajectory_palette(cx).background)
            .text_color(trajectory_palette(cx).label_primary)
            .child(self.trajectory_toolbar(!query.is_empty(), filter.match_count, cx))
            .child(self.trajectory_overview(&filter.matched_cells, cx))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h(px(0.0))
                    .overflow_hidden()
                    .child(match selected {
                        Some(index) if narrow_details => div()
                            .relative()
                            .size_full()
                            .child(self.trajectory_ledger(&filter.rows, cx))
                            .child(
                                div()
                                    .absolute()
                                    .top_0()
                                    .right_0()
                                    .bottom_0()
                                    .w_full()
                                    .max_w(px(720.0))
                                    .shadow_xl()
                                    .child(self.trajectory_details(index, cx)),
                            )
                            .into_any_element(),
                        Some(index) => h_resizable("trajectory-v1-panes")
                            .child(
                                resizable_panel()
                                    .size_range(px(320.0)..px(2_000.0))
                                    .child(self.trajectory_ledger(&filter.rows, cx)),
                            )
                            .child(
                                resizable_panel()
                                    .size(px(410.0))
                                    .size_range(px(320.0)..px(720.0))
                                    .child(self.trajectory_details(index, cx)),
                            )
                            .into_any_element(),
                        None => self.trajectory_ledger(&filter.rows, cx).into_any_element(),
                    }),
            )
    }

    fn trajectory_toolbar(
        &self,
        query_active: bool,
        match_count: usize,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        let actual_duration = self.core.trajectory.mode != TimelineMode::Sequence;
        let matches = query_active.then_some(match_count);
        div()
            .flex()
            .items_center()
            .justify_between()
            .h(px(metrics::LEDGER_TOOLBAR_HEIGHT))
            .px_2()
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
                            .label("◷  Duration")
                            .compact()
                            .ghost()
                            .text_color(if actual_duration {
                                colors.label_primary
                            } else {
                                colors.label_tertiary
                            })
                            .when(actual_duration, |button| button.bg(colors.hover))
                            .on_click(cx.listener(|this, _, window, cx| {
                                let mode = match this.core.trajectory.mode {
                                    TimelineMode::Sequence => TimelineMode::Duration,
                                    TimelineMode::Duration | TimelineMode::Actual => {
                                        TimelineMode::Sequence
                                    }
                                };
                                this.dispatch(Action::SetTimelineMode(mode), window, cx);
                            })),
                    )
                    .child(
                        Button::new("toggle-trajectory-turns")
                            .label(if self.core.trajectory.collapsed_turns {
                                "⊞  Turns"
                            } else {
                                "⊟  Turns"
                            })
                            .compact()
                            .ghost()
                            .text_color(colors.label_tertiary)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::ToggleTrajectoryTurns, window, cx);
                            })),
                    )
                    .child(
                        Button::new("toggle-trajectory-calls")
                            .label(if self.core.trajectory.collapsed_calls {
                                "⊞  Calls"
                            } else {
                                "⊟  Calls"
                            })
                            .compact()
                            .ghost()
                            .text_color(colors.label_tertiary)
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::ToggleTrajectoryCalls, window, cx);
                            })),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_none()
                    .items_center()
                    .gap_2()
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .children(matches.map(|count| {
                        div()
                            .text_xs()
                            .text_color(colors.label_tertiary)
                            .child(format!("{count} matches"))
                    }))
                    .child(
                        div()
                            .w(px(164.0))
                            .child(Input::new(&self.trajectory_search).small().cleanable(true)),
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
        let focused_cells = cache
            .as_ref()
            .and_then(|cache| cache.focus.as_ref())
            .map(|focus| Arc::clone(&focus.cell_indices));
        let display_selection = cache
            .as_ref()
            .and_then(TimelineModelCache::display_selection);
        let entity = cx.entity().clone();
        let selection = display_selection.and_then(|selection| {
            model.map(|model| normalized_range(selection.range, model.viewport))
        });
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
                    .w(px(56.0))
                    .h_full()
                    .pl_1()
                    .pr_2()
                    .items_end()
                    .overflow_hidden()
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
                    .cursor_crosshair()
                    .children(model.map(|model| {
                        self.timeline_lane(
                            TimelineLane::Input,
                            model,
                            focused_cells.as_deref(),
                            matching,
                            cx,
                        )
                    }))
                    .children(model.map(|model| {
                        self.timeline_lane(
                            TimelineLane::Model,
                            model,
                            focused_cells.as_deref(),
                            matching,
                            cx,
                        )
                    }))
                    .children(model.map(|model| {
                        self.timeline_lane(
                            TimelineLane::Tools,
                            model,
                            focused_cells.as_deref(),
                            matching,
                            cx,
                        )
                    }))
                    .children(selection.map(|(left, _width)| {
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left_0()
                            .w(relative(left.max(0.0) as f32))
                            .bg(colors.background.opacity(0.62))
                    }))
                    .children(selection.map(|(left, width)| {
                        let right = (left + width).clamp(0.0, 1.0);
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left(relative(right as f32))
                            .w(relative((1.0 - right) as f32))
                            .bg(colors.background.opacity(0.62))
                    }))
                    .children(selection.map(|(left, width)| {
                        div()
                            .absolute()
                            .top_0()
                            .bottom_0()
                            .left(relative(left as f32))
                            .w(relative(width.max(0.002) as f32))
                            .border_l_2()
                            .border_r_2()
                            .border_color(colors.primary)
                    }))
                    .children(
                        self.timeline_hover
                            .as_ref()
                            .filter(|hover| {
                                hover.record_id.is_none() && self.timeline_drag.is_none()
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
                    })),
            )
    }

    fn ensure_timeline_model_cache(&self) {
        let workspace = &self.core.workspace.cwd;
        let session = &self.core.session.current;
        let projection = &self.core.session_view.trajectory;
        let revision = projection.revision();
        let projection_lineage = projection.projection_lineage();
        let mode = self.core.trajectory.mode;
        let viewport = self.core.trajectory.visible_range;
        let selection = self.core.trajectory.selected_range;
        let render_width_px = self
            .timeline_bounds
            .map(|bounds| f64::from(f32::from(bounds.size.width)).max(1.0))
            .unwrap_or(1_500.0);
        let mut cache = self.timeline_model_cache.borrow_mut();
        let incrementally_updated = cache.as_mut().is_some_and(|cache| {
            cache.projection_matches(workspace, session, projection_lineage, mode)
                && cache.revision != revision
                && projection
                    .geometry_changes_since(cache.revision)
                    .is_some_and(|changes| cache.sync_sequence_geometry(projection, changes))
        });
        if !incrementally_updated
            && cache.as_ref().is_none_or(|cache| {
                !cache.geometry_matches(workspace, session, revision, projection_lineage, mode)
            })
        {
            let retained_search = cache.take().and_then(|previous| {
                (previous.workspace == *workspace
                    && previous.session == *session
                    && previous.projection_lineage == projection_lineage)
                    .then_some(previous.search)
                    .flatten()
            });
            *cache = Some(TimelineModelCache::new(
                TimelineCacheIdentity {
                    workspace: workspace.clone(),
                    session: session.clone(),
                    revision,
                    projection_lineage,
                    mode,
                },
                &projection.records,
                TimelineView {
                    viewport,
                    selection,
                    render_width_px,
                },
                retained_search,
            ));
        } else if cache.as_ref().is_some_and(|cache| {
            cache.viewport != viewport
                || cache.selection != selection
                || (cache.render_width_px - render_width_px).abs() >= 1.0
        }) {
            cache
                .as_mut()
                .expect("timeline cache was checked above")
                .sync_ranges(viewport, selection, render_width_px);
        }
        if let Some(cache) = cache.as_mut() {
            cache.sync_focus(&projection.records);
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
                    self.core.trajectory.collapsed_turns,
                    self.core.trajectory.collapsed_calls,
                )
            })
            .unwrap_or(TimelineFilterSnapshot {
                match_count: 0,
                matched_cells: TimelineCellMatches::All,
                rows: TimelineRows::All(0),
            })
    }

    fn timeline_turn_stats(&self, turn: u32) -> TurnStats {
        self.timeline_model_cache
            .borrow()
            .as_ref()
            .and_then(|cache| cache.search.as_ref())
            .and_then(|search| search.turn_stats.get(&turn))
            .copied()
            .unwrap_or_default()
    }

    fn with_timeline_model<T>(&self, project: impl FnOnce(&TimelineModel) -> T) -> Option<T> {
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        cache
            .as_ref()
            .and_then(|cache| cache.model.as_ref())
            .map(project)
    }

    fn with_timeline_geometry<T>(
        &self,
        project: impl FnOnce(&TimelineGeometry<TrajectoryItemId>) -> T,
    ) -> Option<T> {
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
        model: &TimelineModel,
        focused: Option<&HashSet<usize>>,
        matching: &TimelineCellMatches,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let hovered = self
            .timeline_hover
            .as_ref()
            .and_then(|hover| hover.record_id.as_ref());
        let selected = self.core.details.selected.as_ref();
        let hovered_cell = hovered.and_then(|id| model.cell_by_id.get(id).copied());
        let selected_cell = selected.and_then(|id| model.cell_by_id.get(id).copied());
        let emphasized_cells = [hovered_cell, selected_cell];
        let lane_cells = &model.lane_cell_indices[timeline_lane_index(lane)];
        let ordinary = lane_cells.iter().filter_map(|cell_index| {
            let cell = model.cells.get(*cell_index)?;
            (!emphasized_cells.contains(&Some(cell.ordinal))).then_some(cell)
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
            .filter_map(|(_, cell_index)| model.cells.get(cell_index))
            .filter(|cell| cell.lane == lane);
        div()
            .absolute()
            .top(px(timeline_lane_top(lane)))
            .left_0()
            .right_0()
            .h(px(10.0))
            .children(ordinary.chain(emphasized).map(|cell| {
                self.timeline_block(cell, focused, matching, hovered_cell, selected_cell, cx)
            }))
    }

    fn timeline_block(
        &self,
        cell: &TimelineCell,
        focused_cells: Option<&HashSet<usize>>,
        matching: &TimelineCellMatches,
        hovered_cell: Option<usize>,
        selected_cell: Option<usize>,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let selected = selected_cell == Some(cell.ordinal);
        let hovered = hovered_cell == Some(cell.ordinal);
        let selected_index = selected
            .then_some(self.core.details.selected.as_ref())
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
            .unwrap_or(cell.primary_index);
        let record = &self.core.session_view.trajectory.records[record_index];
        let focused = focused_cells.is_none_or(|focused| focused.contains(&cell.ordinal));
        let matched = matching.contains(cell.ordinal);
        let color = record_color(record, colors);
        let mut tooltip = record_tooltip(record);
        if cell.clustered {
            tooltip.push_str(&format!("\n{} items in this range", cell.item_count));
        }
        let execution = nested_segment_geometry(cell);
        let palette_visible = focused || hovered || selected;
        div()
            .id(("timeline-record-v2", record.source_seq))
            .absolute()
            .left(relative(cell.left as f32))
            .top(px(TIMELINE_BAR_OFFSET))
            .w(relative(cell.width as f32))
            .h(px(TIMELINE_BAR_HEIGHT))
            .rounded(px(2.0))
            .bg(if matched && palette_visible {
                color.opacity(0.28)
            } else if palette_visible {
                color.opacity(0.1)
            } else {
                colors.label_caption.opacity(0.07)
            })
            .border_1()
            .border_color(if palette_visible {
                color
            } else {
                colors.label_caption.opacity(0.18)
            })
            .children((selected || hovered).then(|| {
                div()
                    .absolute()
                    .top(px(-2.0))
                    .bottom(px(-2.0))
                    .left(px(-2.0))
                    .right(px(-2.0))
                    .rounded(px(3.0))
                    .border_1()
                    .border_color(colors.primary)
            }))
            .children(execution.map(|(left, width)| {
                div()
                    .absolute()
                    .top_0()
                    .bottom_0()
                    .left(relative(left as f32))
                    .w(relative(width as f32))
                    .rounded(px(1.0))
                    .bg(if palette_visible {
                        color
                    } else {
                        colors.label_caption.opacity(0.2)
                    })
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

    fn trajectory_ledger(&self, rows: &TimelineRows, cx: &mut Context<Self>) -> impl IntoElement {
        let rows = rows.clone();
        self.ensure_timeline_model_cache();
        let (focused, focused_turns) = self
            .timeline_model_cache
            .borrow()
            .as_ref()
            .and_then(|cache| cache.focus.as_ref())
            .map_or((None, None), |focus| {
                (
                    Some(Arc::clone(&focus.record_indices)),
                    Some(Arc::clone(&focus.turn_indices)),
                )
            });
        gpui::uniform_list(
            "trajectory-ledger-v1",
            rows.len(),
            cx.processor(move |this, range: std::ops::Range<usize>, _, cx| {
                range
                    .filter_map(|row| rows.get(row))
                    .map(|index| {
                        this.trajectory_row(index, focused.as_deref(), focused_turns.as_deref(), cx)
                    })
                    .collect::<Vec<_>>()
            }),
        )
        .size_full()
        .track_scroll(&self.trajectory_scroll)
    }

    fn trajectory_row(
        &self,
        index: usize,
        focused: Option<&HashSet<usize>>,
        focused_turns: Option<&HashSet<u32>>,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.session_view.trajectory.records[index];
        if self.core.trajectory.collapsed_turns && record.turn.is_some() {
            return self.trajectory_turn_summary(index, focused_turns, cx);
        }
        let selected = self.core.details.selected.as_ref() == Some(&record.id);
        let outside = focused.is_some_and(|focused| !focused.contains(&index));
        let opacity = if outside { 0.24 } else { 1.0 };
        let kind_color = record_color(record, colors);
        let turn_start = record.turn.is_some()
            && self
                .core
                .session_view
                .trajectory
                .records
                .get(index.wrapping_sub(1))
                .and_then(|previous| previous.turn)
                != record.turn;
        let continues_from_previous = record.turn.is_some()
            && self
                .core
                .session_view
                .trajectory
                .records
                .get(index.wrapping_sub(1))
                .is_some_and(|previous| previous.turn == record.turn);
        let continues_to_next = record.turn.is_some()
            && self
                .core
                .session_view
                .trajectory
                .records
                .get(index + 1)
                .is_some_and(|next| next.turn == record.turn);
        let duration = format_duration(record.timing.duration_ns());
        div()
            .id(("trajectory-record-v1", record.source_seq))
            .relative()
            .flex()
            .items_center()
            .w_full()
            .h(px(metrics::LEDGER_ROW_HEIGHT))
            .pl_2()
            .pr_3()
            .gap_2()
            .border_b_1()
            .border_color(colors.border_l1.opacity(opacity))
            .when(selected, |row| row.bg(colors.hover))
            .hover(|row| row.bg(colors.hover))
            .cursor_pointer()
            .child(
                div()
                    .relative()
                    .flex()
                    .items_center()
                    .w(px(82.0))
                    .h_full()
                    .pl_5()
                    .text_xs()
                    .text_color(colors.label_caption.opacity(opacity))
                    .children((continues_from_previous || continues_to_next).then(|| {
                        div()
                            .absolute()
                            .left(px(6.5))
                            .top(if continues_from_previous {
                                px(0.0)
                            } else {
                                px(metrics::LEDGER_ROW_HEIGHT / 2.0)
                            })
                            .bottom(if continues_to_next {
                                px(0.0)
                            } else {
                                px(metrics::LEDGER_ROW_HEIGHT / 2.0)
                            })
                            .w(px(1.0))
                            .bg(colors.border_l2.opacity(opacity))
                    }))
                    .children(record.turn.is_some().then(|| {
                        div()
                            .absolute()
                            .left(px(7.0))
                            .top(px(metrics::LEDGER_ROW_HEIGHT / 2.0))
                            .w(px(9.0))
                            .h(px(1.0))
                            .bg(colors.border_l2.opacity(opacity))
                    }))
                    .child(
                        div()
                            .absolute()
                            .left(px(4.0))
                            .size(px(6.0))
                            .rounded_full()
                            .bg(colors
                                .label_caption
                                .opacity(if outside { 0.12 } else { 0.7 })),
                    )
                    .child(if turn_start {
                        format!("Turn {}", record.turn.unwrap_or_default())
                    } else {
                        String::new()
                    }),
            )
            .child(
                div()
                    .flex()
                    .w(px(104.0))
                    .overflow_hidden()
                    .when(record.kind == TrajectoryKind::Tool, |kind| kind.pl_4())
                    .child(
                        div()
                            .px_2()
                            .py_1()
                            .rounded(px(6.0))
                            .text_xs()
                            .font_weight(gpui::FontWeight::SEMIBOLD)
                            .text_color(kind_color.opacity(opacity))
                            .bg(kind_color.opacity(if outside { 0.035 } else { 0.1 }))
                            .max_w_full()
                            .overflow_hidden()
                            .child(kind_label(record.kind).to_uppercase()),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_w(px(0.0))
                    .truncate()
                    .text_sm()
                    .text_color(colors.label_primary.opacity(opacity))
                    .child(row_summary(record)),
            )
            .child(
                div()
                    .w(px(86.0))
                    .text_right()
                    .text_xs()
                    .text_color(colors.label_tertiary.opacity(opacity))
                    .child(duration),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.select_trajectory(index, TrajectorySelectionSource::Ledger, window, cx)
            }))
            .into_any_element()
    }

    fn trajectory_turn_summary(
        &self,
        index: usize,
        focused_turns: Option<&HashSet<u32>>,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let turn = self.core.session_view.trajectory.records[index]
            .turn
            .expect("collapsed turn summary requires a turn");
        let stats = self.timeline_turn_stats(turn);
        let outside = focused_turns.is_some_and(|focused| !focused.contains(&turn));
        let opacity = if outside { 0.24 } else { 1.0 };
        div()
            .id(("trajectory-turn-summary-v1", turn))
            .flex()
            .items_center()
            .w_full()
            .h(px(metrics::LEDGER_ROW_HEIGHT))
            .px_3()
            .gap_3()
            .border_b_1()
            .border_color(colors.border_l1.opacity(opacity))
            .hover(|row| row.bg(colors.hover))
            .cursor_pointer()
            .child(
                div()
                    .w(px(82.0))
                    .truncate()
                    .text_xs()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(colors.label_caption.opacity(opacity))
                    .child(format!("Turn {turn}")),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(gpui::FontWeight::MEDIUM)
                            .text_color(colors.label_primary.opacity(opacity))
                            .child(format!("{} items", stats.item_count)),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(colors.label_tertiary.opacity(opacity))
                            .child(format!("· {} calls · click to expand", stats.call_count)),
                    ),
            )
            .on_click(cx.listener(|this, _, window, cx| {
                this.dispatch(Action::ToggleTrajectoryTurns, window, cx);
            }))
            .into_any_element()
    }

    fn trajectory_details(&self, index: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.session_view.trajectory.records[index];
        let tabs = relevant_tabs(record);
        let active = if tabs.contains(&self.core.details.tab) {
            self.core.details.tab
        } else {
            DetailsTab::Summary
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
                    .px_4()
                    .child(
                        div().flex().flex_col().child(record.title.clone()).child(
                            div()
                                .text_xs()
                                .text_color(colors.label_tertiary)
                                .child(format!(
                                    "{} · {}",
                                    record_location(record),
                                    status_label(record.status)
                                )),
                        ),
                    )
                    .child(
                        Button::new("close-trajectory-v1-details")
                            .icon(IconName::Close)
                            .ghost()
                            .compact()
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::SelectDetails(None), window, cx);
                            })),
                    ),
            )
            .child(
                div()
                    .flex()
                    .px_3()
                    .border_b_1()
                    .border_color(colors.border_l2)
                    .children(tabs.into_iter().enumerate().map(|(index, tab)| {
                        Button::new(("trajectory-detail-tab-v1", index))
                            .label(tab_label(tab))
                            .compact()
                            .ghost()
                            .text_color(if tab == active {
                                colors.primary
                            } else {
                                colors.label_secondary
                            })
                            .when(tab == active, |button| {
                                button.bg(colors.primary.opacity(0.08))
                            })
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
                    .child(self.trajectory_details_body(index, active, cx)),
            )
            .into_any_element()
    }

    fn trajectory_details_body(
        &self,
        index: usize,
        tab: DetailsTab,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.session_view.trajectory.records[index];
        match tab {
            DetailsTab::Timing => self.timing_details(record, colors, cx),
            DetailsTab::Payload => code_panel(
                record
                    .payload
                    .as_deref()
                    .unwrap_or("This record has no payload."),
                colors,
            ),
            DetailsTab::Raw => code_panel(self.selected_details_raw.as_ref(), colors),
            DetailsTab::Result | DetailsTab::Preview => code_panel(&record.text, colors),
            DetailsTab::Summary => div()
                .flex()
                .flex_col()
                .gap_4()
                .child(detail_pair("Kind", kind_label(record.kind), colors))
                .child(detail_pair("Status", status_label(record.status), colors))
                .child(detail_pair(
                    "Source event",
                    &record.source_seq.to_string(),
                    colors,
                ))
                .children(
                    record
                        .call_id
                        .as_deref()
                        .map(|call_id| detail_pair("Call ID", call_id, colors)),
                )
                .children(record.usage.map(|usage| {
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(detail_pair(
                            "Input tokens",
                            &usage.input_tokens.to_string(),
                            colors,
                        ))
                        .child(detail_pair(
                            "Output tokens",
                            &usage.output_tokens.to_string(),
                            colors,
                        ))
                        .child(detail_pair(
                            "Cached tokens",
                            &usage.cached_tokens.to_string(),
                            colors,
                        ))
                }))
                .child(code_panel(&record.text, colors))
                .into_any_element(),
        }
    }

    fn timing_details(
        &self,
        record: &TrajectoryRecord,
        colors: TrajectoryPalette,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let started = format_started(record, self.core.trajectory.unix_time);
        let mut body = div()
            .flex()
            .flex_col()
            .gap_3()
            .child(
                div()
                    .id("toggle-timing-clock-format")
                    .cursor_pointer()
                    .child(detail_pair("Started", &started, colors))
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.dispatch(Action::ToggleTimelineUnixTime, window, cx);
                    })),
            )
            .child(detail_pair("Duration", &timing_duration(record), colors));
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
                .child(detail_pair(
                    "Timing source",
                    "Committed session events",
                    colors,
                ))
                .child(section_title("Execution breakdown", colors))
                .child(detail_pair(
                    "Requested",
                    &format_timing_point(
                        record.timing.requested.as_ref(),
                        self.core.trajectory.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Authorization resolved",
                    &format_timing_point(
                        record.timing.authorization_resolved.as_ref(),
                        self.core.trajectory.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Dispatch intended",
                    &format_timing_point(
                        record.timing.dispatch_intended.as_ref(),
                        self.core.trajectory.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Execution started",
                    &format_timing_point(
                        record.timing.execution_started.as_ref(),
                        self.core.trajectory.unix_time,
                        record,
                    ),
                    colors,
                ))
                .child(detail_pair(
                    "Request registration",
                    &format_duration(record.timing.request_registration_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Authorization wait",
                    &format_duration(record.timing.authorization_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Dispatch wait",
                    &format_duration(record.timing.dispatch_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Runner start wait",
                    &format_duration(record.timing.runner_start_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Execution duration",
                    &record
                        .timing
                        .execution_ns()
                        .map(|ns| format_duration(Some(ns)))
                        .unwrap_or_else(|| execution_missing(record)),
                    colors,
                ))
                .child(detail_pair(
                    "Pre-execution",
                    &format_duration(record.timing.pre_execution_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Post/commit wait",
                    &format_duration(record.timing.post_execution_ns()),
                    colors,
                ))
                .child(detail_pair(
                    "Execution source",
                    "Monotonic execution timestamps",
                    colors,
                ));
        }
        if record.kind == TrajectoryKind::Compaction {
            body = body.child(detail_pair(
                "Timing source",
                if record.status == TrajectoryStatus::Running {
                    "Session timestamps (running)"
                } else {
                    "Session timestamps"
                },
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
        self.dispatch(Action::SelectDetails(Some(record_id)), window, cx);
        self.refresh_selected_details_raw(cx);
        self.details_scroll
            .set_offset(gpui::point(px(0.0), px(0.0)));
        self.scroll_trajectory_to_record(index, cx);
    }

    pub(crate) fn scroll_trajectory_to_record(&self, index: usize, cx: &mut Context<Self>) {
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let rows = self.timeline_filter_snapshot(&query).rows;
        if let Some(row) =
            rows.position(|candidate| self.trajectory_row_represents(candidate, index))
        {
            self.trajectory_scroll
                .scroll_to_item(row, ScrollStrategy::Center);
            cx.notify();
        }
    }

    fn scroll_trajectory_range_into_view(&self, range: AxisRange, cx: &mut Context<Self>) {
        let Some(focused_ids) =
            self.with_timeline_geometry(|geometry| geometry.selection(range).items)
        else {
            return;
        };
        let focused = self.timeline_indices_for_ids(&focused_ids);
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
        let focused_turns = if self.core.trajectory.collapsed_turns {
            focused
                .iter()
                .filter_map(|index| {
                    self.core
                        .session_view
                        .trajectory
                        .records
                        .get(*index)
                        .and_then(|record| record.turn)
                })
                .collect::<HashSet<_>>()
        } else {
            HashSet::new()
        };
        let positions = (0..rows.len())
            .filter_map(|position| {
                let index = rows.get(position)?;
                let record = self.core.session_view.trajectory.records.get(index)?;
                (focused.contains(&index)
                    || record
                        .turn
                        .is_some_and(|turn| focused_turns.contains(&turn)))
                .then_some(position)
            })
            .collect::<Vec<_>>();
        let Some((target, strategy)) = focus_scroll_target(&positions) else {
            return;
        };
        self.trajectory_scroll.scroll_to_item(target, strategy);
        cx.notify();
    }

    fn trajectory_row_represents(&self, row_index: usize, record_index: usize) -> bool {
        if row_index == record_index {
            return true;
        }
        if !self.core.trajectory.collapsed_turns {
            return false;
        }
        self.core
            .session_view
            .trajectory
            .records
            .get(row_index)
            .and_then(|record| record.turn)
            .zip(
                self.core
                    .session_view
                    .trajectory
                    .records
                    .get(record_index)
                    .and_then(|record| record.turn),
            )
            .is_some_and(|(row_turn, record_turn)| row_turn == record_turn)
    }

    fn pan_timeline_to_record(
        &mut self,
        record_id: &TrajectoryItemId,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.ensure_timeline_model_cache();
        let viewport = {
            let mut cache = self.timeline_model_cache.borrow_mut();
            let Some(cache) = cache.as_mut() else { return };
            let Some(target) = cache
                .geometry
                .as_ref()
                .and_then(|geometry| geometry.range_for(record_id))
            else {
                return;
            };
            let previous = cache.interaction.map(|interaction| interaction.viewport);
            cache.reduce(TimelineAction::Reveal { target });
            cache.interaction.and_then(|interaction| {
                (Some(interaction.viewport) != previous).then_some(AxisRange {
                    axis: interaction.axis,
                    range: interaction.viewport,
                })
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
        if event.click_count >= 2 {
            self.ensure_timeline_model_cache();
            if let Some(cache) = self.timeline_model_cache.borrow_mut().as_mut() {
                cache.reduce(TimelineAction::Reset);
            }
            self.timeline_drag = None;
            self.dispatch(Action::SetTimelineSelection(None), window, cx);
            self.dispatch(Action::SetTimelineViewport(None), window, cx);
            return;
        }
        let Some(value) = self.timeline_value(event.position.x) else {
            return;
        };
        let record_id = (!pan)
            .then(|| self.timeline_record_id(event.position))
            .flatten();
        let initial_viewport = self.with_timeline_model(|model| AxisRange {
            axis: model.axis,
            range: model.viewport,
        });
        self.timeline_drag = Some(TimelineDragState {
            pan,
            start_value: value,
            start_x: f32::from(event.position.x),
            record_id,
            initial_viewport,
        });
        self.ensure_timeline_model_cache();
        if let Some(cache) = self.timeline_model_cache.borrow_mut().as_mut() {
            cache.reduce(TimelineAction::PointerDown { value, pan });
        }
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
        if drag.pan {
            let Some(value) =
                self.timeline_value_in_viewport(event.position.x, drag.initial_viewport)
            else {
                return;
            };
            self.ensure_timeline_model_cache();
            let viewport = {
                let mut cache = self.timeline_model_cache.borrow_mut();
                let Some(cache) = cache.as_mut() else {
                    return;
                };
                cache.reduce(TimelineAction::PointerMove { value });
                cache.interaction.map(|interaction| AxisRange {
                    axis: interaction.axis,
                    range: interaction.viewport,
                })
            };
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
        self.ensure_timeline_model_cache();
        let viewport = {
            let mut cache = self.timeline_model_cache.borrow_mut();
            let Some(cache) = cache.as_mut() else {
                return;
            };
            let previous = cache.interaction.map(|interaction| interaction.viewport);
            cache.reduce(TimelineAction::SelectionDrag {
                pointer_fraction,
                edge_fraction,
                pan_step_fraction: TIMELINE_EDGE_PAN_STEP_FRACTION,
            });
            cache.interaction.and_then(|interaction| {
                (Some(interaction.viewport) != previous).then_some(AxisRange {
                    axis: interaction.axis,
                    range: interaction.viewport,
                })
            })
        };
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
        let moved = (f32::from(event.position.x) - drag.start_x).abs() >= TIMELINE_CLICK_SLOP;
        if drag.pan {
            let end = self
                .timeline_value_in_viewport(event.position.x, drag.initial_viewport)
                .unwrap_or(drag.start_value);
            self.ensure_timeline_model_cache();
            let viewport = if let Some(cache) = self.timeline_model_cache.borrow_mut().as_mut() {
                cache.reduce(TimelineAction::PointerMove { value: end });
                cache.reduce(TimelineAction::PointerUp {
                    value: end,
                    minimum_width: 0.0,
                });
                cache.interaction.map(|interaction| AxisRange {
                    axis: interaction.axis,
                    range: interaction.viewport,
                })
            } else {
                None
            };
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
        let Some((viewport, record_count)) = self.with_timeline_model(|model| {
            (
                model.viewport,
                self.core.session_view.trajectory.records.len(),
            )
        }) else {
            self.cancel_timeline_gesture();
            return;
        };
        let minimum = (viewport.width() / record_count.max(1) as f64).max(f64::EPSILON);
        self.ensure_timeline_model_cache();
        let effect = self
            .timeline_model_cache
            .borrow_mut()
            .as_mut()
            .and_then(|cache| {
                cache.reduce(TimelineAction::PointerUp {
                    value: end,
                    minimum_width: minimum,
                })
            });
        let Some(TimelineEffect::SelectionCommitted(selection)) = effect else {
            return;
        };
        self.dispatch(Action::SetTimelineSelection(Some(selection)), window, cx);
        self.scroll_trajectory_range_into_view(selection, cx);
        if event.modifiers.shift {
            self.zoom_to_timeline_selection(window, cx);
        }
        cx.notify();
    }

    fn timeline_wheel(
        &mut self,
        event: &ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
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
        self.ensure_timeline_model_cache();
        let viewport = {
            let mut cache = self.timeline_model_cache.borrow_mut();
            let Some(cache) = cache.as_mut() else {
                return;
            };
            cache.reduce(TimelineAction::WheelZoom {
                anchor,
                factor,
                minimum_width,
            });
            cache.interaction.map(|interaction| AxisRange {
                axis: interaction.axis,
                range: interaction.viewport,
            })
        };
        let Some(viewport) = viewport else {
            return;
        };
        self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
        cx.stop_propagation();
    }

    fn zoom_to_timeline_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.ensure_timeline_model_cache();
        let viewport = {
            let mut cache = self.timeline_model_cache.borrow_mut();
            let Some(cache) = cache.as_mut() else {
                return;
            };
            cache.reduce(TimelineAction::ZoomToSelection);
            cache.interaction.map(|interaction| AxisRange {
                axis: interaction.axis,
                range: interaction.viewport,
            })
        };
        let Some(viewport) = viewport else {
            return;
        };
        self.dispatch(Action::SetTimelineViewport(Some(viewport)), window, cx);
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
        let geometry = cache.geometry.as_ref()?;
        let interaction = cache.interaction?;
        let viewport = viewport
            .filter(|range| range.axis == interaction.axis)
            .map_or(interaction.viewport, |range| range.range);
        let local_x = f64::from(f32::from(x - bounds.origin.x));
        let width = f64::from(f32::from(bounds.size.width));
        Some(geometry.domain_at_pixel(viewport, width, local_x))
    }

    fn update_timeline_hover(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        let Some(bounds) = self.timeline_bounds else {
            return;
        };
        let fraction =
            f64::from(((position.x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0));
        let record_id = self.timeline_record_id(position);
        let hover = Some(TimelineHoverState {
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
        cache.model.as_ref()?.hit_test(lane, fraction).cloned()
    }

    fn timeline_index_for_id(&self, id: &TrajectoryItemId) -> Option<usize> {
        self.ensure_timeline_model_cache();
        self.timeline_model_cache
            .borrow()
            .as_ref()
            .and_then(|cache| cache.index_by_id.get(id).copied())
    }

    fn timeline_indices_for_ids(&self, ids: &HashSet<TrajectoryItemId>) -> HashSet<usize> {
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        let Some(cache) = cache.as_ref() else {
            return HashSet::new();
        };
        ids.iter()
            .filter_map(|id| cache.index_by_id.get(id).copied())
            .collect()
    }

    pub(crate) fn cancel_timeline_gesture(&mut self) {
        self.timeline_drag = None;
        self.ensure_timeline_model_cache();
        if let Some(cache) = self.timeline_model_cache.borrow_mut().as_mut() {
            cache.reduce(TimelineAction::PointerCancel);
        }
    }
}

fn timeline_lane_label(label: &'static str, top: f32) -> gpui::AnyElement {
    div()
        .absolute()
        .top(px(top))
        .left_0()
        .right_0()
        .flex()
        .flex_none()
        .items_center()
        .justify_end()
        .h(px(10.0))
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

fn nested_segment_geometry(cell: &TimelineCell) -> Option<(f64, f64)> {
    let (left, width) = cell.execution_left.zip(cell.execution_width)?;
    let cell_width = cell.width.max(0.000_001);
    let local_left = ((left - cell.left) / cell_width).clamp(0.0, 1.0);
    let available = 1.0 - local_left;
    if available <= 0.0 {
        return None;
    }
    let local_width = (width / cell_width).max(0.002).min(available);
    (local_width > 0.0).then_some((local_left, local_width))
}

fn focus_scroll_target(positions: &[usize]) -> Option<(usize, ScrollStrategy)> {
    let first = positions.first().copied()?;
    Some(if positions.len() > 12 {
        (first, ScrollStrategy::Top)
    } else {
        (positions[positions.len() / 2], ScrollStrategy::Center)
    })
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
    let index_by_id = records
        .iter()
        .enumerate()
        .map(|(index, record)| (record.borrow().id.clone(), index))
        .collect();
    Some(project_timeline(
        &geometry,
        viewport.map(domain_range).unwrap_or(geometry.domain),
        1_500.0,
        &index_by_id,
    ))
}

#[cfg(test)]
fn timeline_geometry<R: Borrow<TrajectoryRecord>>(
    records: &[R],
    axis: AxisId,
) -> Option<TimelineGeometry<TrajectoryItemId>> {
    timeline_geometry_from_iter(records.iter(), axis)
}

fn timeline_geometry_from_iter<'a, R: Borrow<TrajectoryRecord> + 'a>(
    records: impl Iterator<Item = &'a R>,
    axis: AxisId,
) -> Option<TimelineGeometry<TrajectoryItemId>> {
    let spans = records
        .enumerate()
        .map(|(index, record)| timeline_span(index, record.borrow()))
        .collect::<Vec<_>>();
    if spans.is_empty() {
        return None;
    }
    Some(TimelineGeometry::build(axis, spans))
}

fn timeline_span(index: usize, record: &TrajectoryRecord) -> TimelineSpan<TrajectoryItemId> {
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
        id: record.id.clone(),
        lane: timeline_lane(record.lane),
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
    geometry: &TimelineGeometry<TrajectoryItemId>,
    viewport: DomainRange,
    width_px: f64,
    index_by_id: &HashMap<TrajectoryItemId, usize>,
) -> TimelineModel {
    let viewport = viewport.clamp_to(geometry.domain);
    let width_px = width_px.max(1.0);
    let minimum_fraction = 1.0 / width_px;
    let mut cell_by_id = HashMap::new();
    let mut cell_by_record_index = vec![None; index_by_id.len()];
    let mut cells = Vec::new();
    let mut lane_cell_indices: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
    for cell in geometry.render_model(viewport, width_px, TIMELINE_PRIMITIVE_LIMIT) {
        let Some(primary_id) = cell.ids.first().cloned() else {
            continue;
        };
        let Some(hit_id) = cell.ids.last().cloned() else {
            continue;
        };
        let Some(primary_index) = index_by_id.get(&primary_id).copied() else {
            continue;
        };
        let ordinal = cells.len();
        let left = cell.start_px / width_px;
        let right = cell.end_px / width_px;
        let execution = cell
            .nested
            .map(|(start, end)| (start / width_px, (end - start) / width_px));
        let item_count = cell.ids.len();
        for id in cell.ids {
            cell_by_id.insert(id.clone(), ordinal);
            if let Some(index) = index_by_id.get(&id).copied()
                && let Some(target) = cell_by_record_index.get_mut(index)
            {
                *target = Some(ordinal);
            }
        }
        lane_cell_indices[timeline_lane_index(cell.lane)].push(ordinal);
        cells.push(TimelineCell {
            ordinal,
            primary_index,
            hit_id,
            item_count,
            lane: cell.lane,
            left,
            width: (right - left).max(minimum_fraction),
            execution_left: execution.map(|value| value.0),
            execution_width: execution.map(|value| value.1),
            clustered: cell.clustered,
        });
    }
    TimelineModel {
        axis: geometry.axis,
        domain: geometry.domain,
        viewport,
        cells,
        lane_cell_indices,
        cell_by_id,
        cell_by_record_index,
    }
}

fn timeline_lane_index(lane: TimelineLane) -> usize {
    match lane {
        TimelineLane::Input => 0,
        TimelineLane::Model => 1,
        TimelineLane::Tools => 2,
    }
}

fn timeline_lane(lane: TrajectoryLane) -> TimelineLane {
    match lane {
        TrajectoryLane::Input => TimelineLane::Input,
        TrajectoryLane::Model => TimelineLane::Model,
        TrajectoryLane::Tools => TimelineLane::Tools,
    }
}

fn document_generation(workspace: &Path, session: &Path, projection_lineage: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    workspace.hash(&mut hasher);
    session.hash(&mut hasher);
    projection_lineage.hash(&mut hasher);
    hasher.finish()
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

fn record_color(record: &TrajectoryRecord, colors: TrajectoryPalette) -> gpui::Hsla {
    if matches!(
        record.status,
        TrajectoryStatus::Failed | TrajectoryStatus::Unknown
    ) {
        return colors.error;
    }
    match record.kind {
        TrajectoryKind::System => colors.system_foreground,
        TrajectoryKind::User | TrajectoryKind::Steering => colors.user_foreground,
        TrajectoryKind::Context => colors.context_foreground,
        TrajectoryKind::Assistant | TrajectoryKind::Compaction => colors.assistant_foreground,
        TrajectoryKind::Tool => colors.tool_foreground,
        TrajectoryKind::RequestFailure => colors.error,
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
        TrajectoryKind::RequestFailure => "Failure",
    }
}

fn status_label(status: TrajectoryStatus) -> &'static str {
    match status {
        TrajectoryStatus::Running => "Running",
        TrajectoryStatus::Completed => "Completed",
        TrajectoryStatus::Failed => "Error",
        TrajectoryStatus::Denied => "Denied",
        TrajectoryStatus::NotExecuted => "Not executed",
        TrajectoryStatus::Unknown => "Unknown side effects",
    }
}

fn record_location(record: &TrajectoryRecord) -> String {
    match (record.turn, record.step) {
        (Some(turn), Some(step)) => format!("T{turn} · S{step}"),
        (Some(turn), None) => format!("T{turn}"),
        _ => "Session".into(),
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
        .map(|duration| format!("Total {}", format_duration(Some(duration))))
        .into_iter()
        .collect::<Vec<_>>();
    if record.kind == TrajectoryKind::Assistant
        && let (Some(ttft), Some(decoding)) =
            (record.timing.ttft_ns(), record.timing.generation_ns())
    {
        timing.push(format!(
            "TTFT {} · Decoding {}",
            format_duration(Some(ttft)),
            format_duration(Some(decoding))
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

fn relevant_tabs(record: &TrajectoryRecord) -> Vec<DetailsTab> {
    match record.kind {
        TrajectoryKind::Tool => vec![
            DetailsTab::Summary,
            DetailsTab::Payload,
            DetailsTab::Result,
            DetailsTab::Raw,
            DetailsTab::Timing,
        ],
        _ => vec![
            DetailsTab::Summary,
            DetailsTab::Preview,
            DetailsTab::Raw,
            DetailsTab::Timing,
        ],
    }
}

fn tab_label(tab: DetailsTab) -> &'static str {
    match tab {
        DetailsTab::Summary => "Summary",
        DetailsTab::Preview => "Preview",
        DetailsTab::Raw => "Raw",
        DetailsTab::Payload => "Payload",
        DetailsTab::Result => "Result",
        DetailsTab::Timing => "Timing",
    }
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

fn format_duration(nanoseconds: Option<u64>) -> String {
    let Some(nanoseconds) = nanoseconds else {
        return "Not recorded".into();
    };
    let milliseconds = nanoseconds as f64 / 1_000_000.0;
    if milliseconds < 1.0 {
        format!("{:.0} µs", nanoseconds as f64 / 1_000.0)
    } else if milliseconds < 1_000.0 {
        format!("{milliseconds:.1} ms")
    } else {
        format!("{:.2} s", milliseconds / 1_000.0)
    }
}

fn format_started(record: &TrajectoryRecord, unix: bool) -> String {
    record
        .timing
        .started
        .as_ref()
        .map(|time| format_wall(time.wall_time_ms(), unix))
        .unwrap_or_else(|| "Not recorded".into())
}

fn format_wall(wall_time_ms: i64, unix: bool) -> String {
    if unix {
        return format!("{:.3}", wall_time_ms as f64 / 1_000.0);
    }
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
            "[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]"
        ))
        .unwrap_or_else(|_| "Not recorded".into())
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
    if record.timing.started.is_none() {
        return "Not recorded".into();
    }
    if record.timing.completed.is_none() {
        return if record.status == TrajectoryStatus::Running {
            "Pending".into()
        } else {
            "Not recorded".into()
        };
    }
    format_duration(record.timing.duration_ns())
}

fn assistant_ttft(record: &TrajectoryRecord) -> String {
    if record.timing.completed.is_none() && record.status != TrajectoryStatus::Running {
        "Not recorded".into()
    } else if record.timing.started.is_none() {
        "Step start unavailable".into()
    } else if record.timing.first_token.is_none() {
        "First token unavailable".into()
    } else {
        format_duration(record.timing.ttft_ns())
    }
}

fn assistant_generation(record: &TrajectoryRecord) -> String {
    if record.timing.completed.is_none() {
        if record.status == TrajectoryStatus::Running {
            "Pending".into()
        } else {
            "Not recorded".into()
        }
    } else if record.timing.first_token.is_none() {
        "First token unavailable".into()
    } else {
        format_duration(record.timing.generation_ns())
    }
}

fn assistant_throughput(record: &TrajectoryRecord) -> String {
    if record.timing.completed.is_none() && record.status != TrajectoryStatus::Running {
        return "Not recorded".into();
    }
    let Some(usage) = record.usage else {
        return "Usage unavailable".into();
    };
    if usage.output_tokens == 0 {
        return "Output tokens unavailable".into();
    }
    let Some(generation) = record.timing.generation_ns() else {
        return "First token unavailable".into();
    };
    if generation == 0 {
        return "Duration too short".into();
    }
    format!(
        "{:.1} tok/s",
        usage.output_tokens as f64 / (generation as f64 / 1_000_000_000.0)
    )
}

fn execution_missing(record: &TrajectoryRecord) -> String {
    match record.status {
        TrajectoryStatus::Denied | TrajectoryStatus::NotExecuted => "Not executed".into(),
        TrajectoryStatus::Unknown => "Unknown".into(),
        _ => "Not recorded".into(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::Arc;

    use kcastle_agent::{CallId, EventTime};

    use crate::domain::timeline::{AxisId, AxisRange, DomainRange, TimelineLane};
    use crate::domain::{
        RecordTiming, TimelineMode, TrajectoryItemId, TrajectoryKind, TrajectoryLane,
        TrajectoryRecord, TrajectoryStatus,
    };

    use super::{
        ScrollStrategy, TIMELINE_BAR_HEIGHT, TIMELINE_BAR_OFFSET, TimelineCacheIdentity,
        TimelineCell, TimelineCellMatches, TimelineModelCache, TimelineRows, TimelineSearchCache,
        TimelineView, TrajectorySelectionSource, focus_scroll_target, nested_segment_geometry,
        normalized_range, record_tooltip, should_clear_selection_for_record, timeline_geometry,
        timeline_lane_at, timeline_lane_top, timeline_model,
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
            lane: TrajectoryLane::Tools,
            title: "tool".into(),
            text: String::new(),
            payload: None,
            turn: Some(1),
            step: Some(1),
            call_id: Some(format!("call-{id}")),
            status: TrajectoryStatus::Completed,
            timing,
            usage: None,
            search_text: "tool\n".into(),
        }
    }

    #[test]
    fn timeline_cache_reuses_geometry_when_only_the_viewport_changes() {
        let workspace = PathBuf::from("workspace");
        let session = PathBuf::from("session.jsonl");
        let projection_lineage = 17;
        let records = [record(1, 0, 100)]
            .into_iter()
            .map(std::sync::Arc::new)
            .collect::<im::Vector<_>>();
        let axis = AxisId {
            document_generation: super::document_generation(
                &workspace,
                &session,
                projection_lineage,
            ),
            geometry_revision: 4,
            mode: TimelineMode::Duration,
        };
        let mut cache = TimelineModelCache::new(
            TimelineCacheIdentity {
                workspace: workspace.clone(),
                session: session.clone(),
                revision: 4,
                projection_lineage,
                mode: TimelineMode::Duration,
            },
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
        assert!(cache.geometry_matches(
            &workspace,
            &session,
            4,
            projection_lineage,
            TimelineMode::Duration
        ));
        assert!(!cache.geometry_matches(
            &workspace,
            &session,
            5,
            projection_lineage,
            TimelineMode::Duration
        ));

        let geometry_before = cache.geometry.as_ref().unwrap().cells.clone();
        let viewport = AxisRange {
            axis,
            range: DomainRange::new(10.0, 60.0),
        };
        cache.sync_ranges(Some(viewport), None, 1_500.0);
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
            TimelineCacheIdentity {
                workspace: PathBuf::from("large-workspace"),
                session: PathBuf::from("large-session.db"),
                revision: 1,
                projection_lineage: 77,
                mode: TimelineMode::Sequence,
            },
            &records,
            TimelineView {
                viewport: None,
                selection: None,
                render_width_px: 1_500.0,
            },
            None,
        );
        let model = model_cache.model.as_ref().unwrap();
        assert!(model.cells.len() <= super::TIMELINE_PRIMITIVE_LIMIT);
        assert_eq!(
            model
                .cells
                .iter()
                .map(|cell| cell.item_count)
                .sum::<usize>(),
            100_000
        );
        assert_eq!(
            model.lane_cell_indices.iter().map(Vec::len).sum::<usize>(),
            model.cells.len()
        );

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
        search_cache.sync_model_matches(model, model_cache.model_revision);
        assert!(matches!(
            search_cache.matched_cells,
            TimelineCellMatches::Filtered(ref cells) if cells.is_empty()
        ));
        let inspected_before = search_cache.inspected_records;
        let row_rebuilds_before = search_cache.materialized_row_rebuilds;
        let mut updated = records[77_777].as_ref().clone();
        updated.search_text = "absent now matches".into();
        records.set(77_777, Arc::new(updated));
        search_cache.sync_changed_records(&records, 3, [77_777]);
        search_cache.sync_model_matches(model, model_cache.model_revision);
        assert_eq!(search_cache.inspected_records - inspected_before, 1);
        assert_eq!(search_cache.rows.len(), 1);
        assert_eq!(search_cache.rows.get(0), Some(77_777));
        assert_eq!(search_cache.materialized_row_rebuilds, row_rebuilds_before);
        let matched_cell = model.cell_by_record_index[77_777].unwrap();
        assert!(search_cache.matched_cells.contains(matched_cell));

        // A later timing/usage-only receipt advances no search revision and inspects nothing.
        let inspected_before = search_cache.inspected_records;
        search_cache.sync_changed_records(&records, 3, std::iter::empty());
        assert_eq!(search_cache.inspected_records, inspected_before);
    }

    #[test]
    fn selection_focus_is_materialized_once_for_overview_and_ledger() {
        let workspace = PathBuf::from("workspace");
        let session = PathBuf::from("session.db");
        let projection_lineage = 91;
        let revision = 4;
        let records = (0..100)
            .map(|index| Arc::new(record(index, index, index.saturating_add(100))))
            .collect::<im::Vector<_>>();
        let axis = AxisId {
            document_generation: super::document_generation(
                &workspace,
                &session,
                projection_lineage,
            ),
            geometry_revision: revision,
            mode: TimelineMode::Sequence,
        };
        let selection = AxisRange {
            axis,
            range: DomainRange::new(10.0, 20.0),
        };
        let mut cache = TimelineModelCache::new(
            TimelineCacheIdentity {
                workspace,
                session,
                revision,
                projection_lineage,
                mode: TimelineMode::Sequence,
            },
            &records,
            TimelineView {
                viewport: None,
                selection: Some(selection),
                render_width_px: 1_500.0,
            },
            None,
        );

        cache.sync_focus(&records);
        let first = cache.focus.as_ref().unwrap();
        let record_indices = Arc::clone(&first.record_indices);
        let turn_indices = Arc::clone(&first.turn_indices);
        let cell_indices = Arc::clone(&first.cell_indices);
        cache.sync_focus(&records);
        let second = cache.focus.as_ref().unwrap();

        assert!(Arc::ptr_eq(&record_indices, &second.record_indices));
        assert!(Arc::ptr_eq(&turn_indices, &second.turn_indices));
        assert!(Arc::ptr_eq(&cell_indices, &second.cell_indices));
    }

    #[test]
    fn same_session_path_with_another_projection_lineage_has_another_axis_identity() {
        let workspace = PathBuf::from("workspace");
        let session = PathBuf::from("session.db");
        assert_ne!(
            super::document_generation(&workspace, &session, 100),
            super::document_generation(&workspace, &session, 101)
        );
    }

    #[test]
    fn retained_search_invalidates_its_cell_projection_in_a_new_model_cache() {
        let records = [record(1, 0, 100), record(2, 100, 200)]
            .into_iter()
            .map(Arc::new)
            .collect::<im::Vector<_>>();
        let mut first = TimelineModelCache::new(
            TimelineCacheIdentity {
                workspace: PathBuf::from("workspace"),
                session: PathBuf::from("session.db"),
                revision: 1,
                projection_lineage: 92,
                mode: TimelineMode::Sequence,
            },
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
        first
            .search
            .as_mut()
            .unwrap()
            .sync_model_matches(model, first.model_revision);
        assert_eq!(
            first.search.as_ref().unwrap().matched_model_revision,
            first.model_revision
        );

        let second = TimelineModelCache::new(
            TimelineCacheIdentity {
                workspace: PathBuf::from("workspace"),
                session: PathBuf::from("session.db"),
                revision: 1,
                projection_lineage: 92,
                mode: TimelineMode::Duration,
            },
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
    fn timeline_cache_does_not_rebind_ranges_from_an_old_geometry_revision() {
        let workspace = PathBuf::from("workspace");
        let session = PathBuf::from("session.db");
        let projection_lineage = 18;
        let records = [record(1, 0, 100)]
            .into_iter()
            .map(Arc::new)
            .collect::<im::Vector<_>>();
        let stale_axis = AxisId {
            document_generation: super::document_generation(
                &workspace,
                &session,
                projection_lineage,
            ),
            geometry_revision: 3,
            mode: TimelineMode::Duration,
        };
        let cache = TimelineModelCache::new(
            TimelineCacheIdentity {
                workspace,
                session,
                revision: 4,
                projection_lineage,
                mode: TimelineMode::Duration,
            },
            &records,
            TimelineView {
                viewport: Some(AxisRange {
                    axis: stale_axis,
                    range: DomainRange::new(20.0, 40.0),
                }),
                selection: Some(AxisRange {
                    axis: stale_axis,
                    range: DomainRange::new(25.0, 30.0),
                }),
                render_width_px: 1_500.0,
            },
            None,
        );

        let interaction = cache.interaction.unwrap();
        assert_eq!(interaction.viewport, interaction.domain);
        assert_eq!(interaction.selection, None);
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
        assert!((cell.left - 0.0).abs() < 0.000_001);
        assert!((cell.width - 1.0).abs() < 0.000_001);
        assert!((cell.execution_left.unwrap() - 0.2).abs() < 0.000_001);
        assert!((cell.execution_width.unwrap() - 0.6).abs() < 0.000_001);
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
        assert_eq!(
            geometry.hit_test(TimelineLane::Tools, 1_050.0),
            Some(&records[1].id)
        );
        assert_eq!(geometry.hit_test(TimelineLane::Model, 1_050.0), None);
        assert_eq!(
            model.hit_test(TimelineLane::Tools, 0.5),
            Some(&records[1].id)
        );
    }

    #[test]
    fn assistant_hover_tooltip_uses_dsh_timing_shape() {
        let mut assistant = record(1, 0, 100);
        assistant.kind = TrajectoryKind::Assistant;
        assistant.lane = TrajectoryLane::Model;
        assistant.timing.first_token = Some((&time(20)).into());

        let tooltip = record_tooltip(&assistant);
        assert!(tooltip.starts_with("ASSISTANT\n"));
        assert!(tooltip.contains(" → "));
        assert!(tooltip.contains("Total 100.0 ms"));
        assert!(tooltip.contains("TTFT 20.0 ms · Decoding 80.0 ms"));
    }

    #[test]
    fn clipped_nested_segment_does_not_create_an_invalid_width_range() {
        let id = record(1, 0, 100).id;
        let cell = TimelineCell {
            ordinal: 0,
            primary_index: 0,
            hit_id: id,
            item_count: 1,
            lane: TimelineLane::Tools,
            left: 0.0,
            width: 1.0,
            execution_left: Some(1.0),
            execution_width: Some(0.2),
            clustered: false,
        };
        assert_eq!(nested_segment_geometry(&cell), None);
    }

    #[test]
    fn focused_rows_center_small_ranges_and_anchor_large_ranges() {
        assert_eq!(
            focus_scroll_target(&[4, 5, 6]),
            Some((5, ScrollStrategy::Center))
        );
        assert_eq!(
            focus_scroll_target(&(10..24).collect::<Vec<_>>()),
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
