use std::collections::HashSet;
use std::path::{Path, PathBuf};

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
use time::{OffsetDateTime, UtcOffset, macros::format_description};

use crate::app::{DesktopApp, TimelineDragState, TimelineHoverState};
use crate::domain::{
    Action, DetailsTab, TimelineMode, TrajectoryKind, TrajectoryLane, TrajectoryRecord,
    TrajectoryStatus,
};
use crate::layout::TrajectoryMode;
use crate::ui_theme::{TrajectoryPalette, metrics, trajectory_palette};

#[derive(Clone, Copy, Debug)]
struct TimelineCell {
    index: usize,
    start: f64,
    end: f64,
    left: f64,
    width: f64,
    execution_left: Option<f64>,
    execution_width: Option<f64>,
}

#[derive(Clone, Debug)]
struct TimelineModel {
    domain: (f64, f64),
    viewport: (f64, f64),
    cells: Vec<TimelineCell>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TimelineGeometryCell {
    index: usize,
    start: f64,
    end: f64,
    execution: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
struct TimelineGeometry {
    domain: (f64, f64),
    cells: Vec<TimelineGeometryCell>,
}

#[derive(Clone, Debug)]
pub(crate) struct TimelineModelCache {
    workspace: PathBuf,
    session: PathBuf,
    revision: u64,
    mode: TimelineMode,
    viewport: Option<(f64, f64)>,
    geometry: Option<TimelineGeometry>,
    model: Option<TimelineModel>,
}

impl TimelineModelCache {
    fn new(
        workspace: PathBuf,
        session: PathBuf,
        revision: u64,
        records: &[TrajectoryRecord],
        mode: TimelineMode,
        viewport: Option<(f64, f64)>,
    ) -> Self {
        let geometry = timeline_geometry(records, mode);
        let model = geometry
            .as_ref()
            .map(|geometry| project_timeline(geometry, viewport));
        Self {
            workspace,
            session,
            revision,
            mode,
            viewport,
            geometry,
            model,
        }
    }

    fn geometry_matches(
        &self,
        workspace: &Path,
        session: &Path,
        revision: u64,
        mode: TimelineMode,
    ) -> bool {
        self.workspace == workspace
            && self.session == session
            && self.revision == revision
            && self.mode == mode
    }

    fn set_viewport(&mut self, viewport: Option<(f64, f64)>) {
        self.viewport = viewport;
        self.model = self
            .geometry
            .as_ref()
            .map(|geometry| project_timeline(geometry, viewport));
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
        let rows = self.trajectory_rows(&query);
        let selected = self.core.details.selected.and_then(|selected| {
            self.core
                .trajectory_data
                .records
                .iter()
                .position(|record| record.id == selected)
        });
        let narrow_details = self.core.layout.trajectory == TrajectoryMode::Overlay;

        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .overflow_hidden()
            .bg(trajectory_palette(cx).background)
            .text_color(trajectory_palette(cx).label_primary)
            .child(self.trajectory_toolbar(cx))
            .child(self.trajectory_overview(&query, cx))
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
                            .child(self.trajectory_ledger(&rows, cx))
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
                                    .child(self.trajectory_ledger(&rows, cx)),
                            )
                            .child(
                                resizable_panel()
                                    .size(px(410.0))
                                    .size_range(px(320.0)..px(720.0))
                                    .child(self.trajectory_details(index, cx)),
                            )
                            .into_any_element(),
                        None => self.trajectory_ledger(&rows, cx).into_any_element(),
                    }),
            )
    }

    fn trajectory_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        let actual_duration = self.core.trajectory.mode != TimelineMode::Sequence;
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let matches = (!query.is_empty()).then(|| {
            self.core
                .trajectory_data
                .records
                .iter()
                .filter(|record| record.matches(&query))
                .count()
        });
        div()
            .flex()
            .items_center()
            .justify_between()
            .h(px(metrics::LEDGER_TOOLBAR_HEIGHT))
            .px_2()
            .border_b_1()
            .border_color(colors.border_l2)
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap(px(2.0))
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
                                let mode = if this.core.trajectory.mode == TimelineMode::Sequence {
                                    TimelineMode::Duration
                                } else {
                                    TimelineMode::Sequence
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
                    .items_center()
                    .gap_2()
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

    fn trajectory_overview(&self, query: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        self.ensure_timeline_model_cache();
        let cache = self.timeline_model_cache.borrow();
        let model = cache.as_ref().and_then(|cache| cache.model.as_ref());
        let entity = cx.entity().clone();
        let selection = self
            .core
            .trajectory
            .selected_range
            .and_then(|range| model.map(|model| normalized_range(range, model.viewport)));
        div()
            .flex()
            .h(px(50.0))
            .overflow_hidden()
            .bg(colors.code_background)
            .border_b_1()
            .border_color(colors.border_l2)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .justify_center()
                    .w(px(44.0))
                    .h_full()
                    .pr_1()
                    .items_end()
                    .gap(px(4.0))
                    .overflow_hidden()
                    .text_size(px(10.0))
                    .line_height(px(10.0))
                    .text_color(colors.label_caption)
                    .child(timeline_lane_label("Input"))
                    .child(timeline_lane_label("Model"))
                    .child(timeline_lane_label("Tools")),
            )
            .child(
                div()
                    .id("trajectory-timeline")
                    .relative()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .h_full()
                    .justify_center()
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .gap(px(6.0))
                    .cursor_crosshair()
                    .children(
                        model.map(|model| {
                            self.timeline_lane(TrajectoryLane::Input, model, query, cx)
                        }),
                    )
                    .children(
                        model.map(|model| {
                            self.timeline_lane(TrajectoryLane::Model, model, query, cx)
                        }),
                    )
                    .children(
                        model.map(|model| {
                            self.timeline_lane(TrajectoryLane::Tools, model, query, cx)
                        }),
                    )
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
                            .bg(colors.primary.opacity(0.12))
                            .border_l_2()
                            .border_r_2()
                            .border_color(colors.primary)
                    }))
                    .children(
                        self.timeline_hover
                            .filter(|hover| {
                                hover.record_index.is_none() && self.timeline_drag.is_none()
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
                        if !*hovered && this.timeline_hover.take().is_some() {
                            cx.notify();
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
                    .on_scroll_wheel(cx.listener(|this, event: &ScrollWheelEvent, window, cx| {
                        this.timeline_wheel(event, window, cx)
                    })),
            )
    }

    fn ensure_timeline_model_cache(&self) {
        let workspace = &self.core.workspace.cwd;
        let session = &self.core.session.current;
        let revision = self.core.trajectory_data.revision();
        let mode = self.core.trajectory.mode;
        let viewport = self.core.trajectory.visible_range;
        let mut cache = self.timeline_model_cache.borrow_mut();
        if cache
            .as_ref()
            .is_none_or(|cache| !cache.geometry_matches(workspace, session, revision, mode))
        {
            *cache = Some(TimelineModelCache::new(
                workspace.clone(),
                session.clone(),
                revision,
                &self.core.trajectory_data.records,
                mode,
                viewport,
            ));
        } else if cache
            .as_ref()
            .is_some_and(|cache| cache.viewport != viewport)
        {
            cache
                .as_mut()
                .expect("timeline cache was checked above")
                .set_viewport(viewport);
        }
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
        lane: TrajectoryLane,
        model: &TimelineModel,
        query: &str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let hovered = self.timeline_hover.and_then(|hover| hover.record_index);
        let selected = self.core.details.selected;
        let mut cells = model
            .cells
            .iter()
            .filter(|cell| self.core.trajectory_data.records[cell.index].lane == lane)
            .copied()
            .collect::<Vec<_>>();
        cells.sort_by_key(|cell| {
            let record = &self.core.trajectory_data.records[cell.index];
            hovered == Some(cell.index) || selected == Some(record.id)
        });
        div().relative().h(px(10.0)).children(
            cells
                .into_iter()
                .map(|cell| self.timeline_block(cell, model.viewport, query, cx)),
        )
    }

    fn timeline_block(
        &self,
        cell: TimelineCell,
        _viewport: (f64, f64),
        query: &str,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.trajectory_data.records[cell.index];
        let selected = self.core.details.selected == Some(record.id);
        let hovered = self
            .timeline_hover
            .is_some_and(|hover| hover.record_index == Some(cell.index));
        let focused = self
            .core
            .trajectory
            .selected_range
            .is_none_or(|range| cell_intersects_range(cell, range));
        let matched = record.matches(query);
        let color = record_color(record, colors);
        let tooltip = record_tooltip(record);
        let execution = nested_segment_geometry(cell);
        div()
            .id(("timeline-record-v1", record.source_seq))
            .absolute()
            .left(relative(cell.left as f32))
            .top(px(1.0))
            .w(relative(cell.width.max(0.002) as f32))
            .h(px(8.0))
            .rounded(px(2.0))
            .bg(if hovered {
                color
            } else if matched && focused {
                color.opacity(0.28)
            } else if focused {
                color.opacity(0.1)
            } else {
                color.opacity(0.035)
            })
            .border_1()
            .border_color(if selected || hovered {
                colors.code_background
            } else if focused {
                color
            } else {
                color.opacity(0.2)
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
                    .border_color(if selected {
                        colors.primary
                    } else {
                        colors.primary.opacity(0.8)
                    })
            }))
            .children(execution.map(|(left, width)| {
                div()
                    .absolute()
                    .top_0()
                    .bottom_0()
                    .left(relative(left as f32))
                    .w(relative(width as f32))
                    .rounded(px(1.0))
                    .bg(if focused || hovered {
                        color
                    } else {
                        color.opacity(0.2)
                    })
            }))
            .cursor_pointer()
            .tooltip(move |window, cx| {
                Tooltip::new(tooltip.clone())
                    .text_size(px(11.0))
                    .line_height(px(16.0))
                    .build(window, cx)
            })
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .on_click(cx.listener(move |this, _, window, cx| {
                this.select_trajectory(cell.index, window, cx)
            }))
            .into_any_element()
    }

    fn trajectory_rows(&self, query: &str) -> Vec<usize> {
        let mut seen_turns = HashSet::new();
        self.core
            .trajectory_data
            .records
            .iter()
            .enumerate()
            .filter(|(_, record)| record.matches(query))
            .filter(|(_, record)| {
                !self.core.trajectory.collapsed_calls || record.kind != TrajectoryKind::Tool
            })
            .filter(|(_, record)| {
                if !self.core.trajectory.collapsed_turns {
                    return true;
                }
                record.turn.is_none_or(|turn| seen_turns.insert(turn))
            })
            .map(|(index, _)| index)
            .collect()
    }

    fn trajectory_ledger(&self, rows: &[usize], cx: &mut Context<Self>) -> impl IntoElement {
        let rows = rows.to_vec();
        let focused = self.core.trajectory.selected_range.and_then(|range| {
            self.with_timeline_geometry(|geometry| {
                geometry
                    .cells
                    .iter()
                    .filter(|cell| geometry_cell_intersects_range(**cell, range))
                    .map(|cell| cell.index)
                    .collect::<HashSet<_>>()
            })
        });
        gpui::uniform_list(
            "trajectory-ledger-v1",
            rows.len(),
            cx.processor(move |this, range: std::ops::Range<usize>, _, cx| {
                range
                    .filter_map(|row| rows.get(row).copied())
                    .map(|index| this.trajectory_row(index, focused.as_ref(), cx))
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
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.trajectory_data.records[index];
        let selected = self.core.details.selected == Some(record.id);
        let outside = focused.is_some_and(|focused| !focused.contains(&index));
        let opacity = if outside { 0.24 } else { 1.0 };
        let kind_color = record_color(record, colors);
        let turn_start = record.turn.is_some()
            && self
                .core
                .trajectory_data
                .records
                .get(index.wrapping_sub(1))
                .and_then(|previous| previous.turn)
                != record.turn;
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
                div().flex().w(px(104.0)).child(
                    div()
                        .px_2()
                        .py_1()
                        .rounded(px(6.0))
                        .text_xs()
                        .font_weight(gpui::FontWeight::SEMIBOLD)
                        .text_color(kind_color.opacity(opacity))
                        .bg(kind_color.opacity(if outside { 0.035 } else { 0.1 }))
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
            .on_click(
                cx.listener(move |this, _, window, cx| this.select_trajectory(index, window, cx)),
            )
            .into_any_element()
    }

    fn trajectory_details(&self, index: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let record = &self.core.trajectory_data.records[index];
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
                            .when(tab == active, |button| button.bg(colors.hover))
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
        let record = &self.core.trajectory_data.records[index];
        match tab {
            DetailsTab::Timing => self.timing_details(record, colors, cx),
            DetailsTab::Payload => code_panel(
                record
                    .payload
                    .as_deref()
                    .unwrap_or("This record has no payload."),
                colors,
            ),
            DetailsTab::Raw => code_panel(&record.raw, colors),
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
                .child(detail_pair("Timing source", "Session timestamps", colors))
                .child(section_title("Execution breakdown", colors))
                .child(detail_pair(
                    "Execution started",
                    &record
                        .timing
                        .execution_started
                        .as_ref()
                        .map(|time| format_wall(time.wall_time_ms, self.core.trajectory.unix_time))
                        .unwrap_or_else(|| execution_missing(record)),
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

    fn select_trajectory(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(record_id) = self
            .core
            .trajectory_data
            .records
            .get(index)
            .map(|record| record.id)
        else {
            return;
        };
        self.dispatch(Action::SetTimelineSelection(None), window, cx);
        self.dispatch(Action::SelectDetails(Some(record_id)), window, cx);
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
        let rows = self.trajectory_rows(&query);
        if let Some(row) = rows.iter().position(|candidate| *candidate == index) {
            self.trajectory_scroll
                .scroll_to_item(row, ScrollStrategy::Center);
            cx.notify();
        }
    }

    fn scroll_trajectory_range_into_view(&self, range: (f64, f64), cx: &mut Context<Self>) {
        let Some(focused) = self.with_timeline_geometry(|geometry| {
            geometry
                .cells
                .iter()
                .filter(|cell| geometry_cell_intersects_range(**cell, range))
                .map(|cell| cell.index)
                .collect::<HashSet<_>>()
        }) else {
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
        let rows = self.trajectory_rows(&query);
        let positions = rows
            .iter()
            .enumerate()
            .filter_map(|(position, index)| focused.contains(index).then_some(position))
            .collect::<Vec<_>>();
        let Some((target, strategy)) = focus_scroll_target(&positions) else {
            return;
        };
        self.trajectory_scroll.scroll_to_item(target, strategy);
        cx.notify();
    }

    fn timeline_mouse_down(
        &mut self,
        event: &MouseDownEvent,
        pan: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if event.click_count >= 2 {
            self.dispatch(Action::SetTimelineSelection(None), window, cx);
            return;
        }
        let Some(value) = self.timeline_value(event.position.x) else {
            return;
        };
        self.timeline_drag = Some(TimelineDragState {
            pan,
            start_value: value,
            initial_viewport: self.core.trajectory.visible_range,
        });
        if pan {
            self.dispatch(Action::SetTimelineSelection(None), window, cx);
        } else {
            self.dispatch(
                Action::SetTimelineSelection(Some((value, value))),
                window,
                cx,
            );
        }
        cx.stop_propagation();
    }

    fn timeline_mouse_move(
        &mut self,
        event: &MouseMoveEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.update_timeline_hover(event.position, cx);
        let Some(drag) = self.timeline_drag else {
            return;
        };
        let Some(value) = self.timeline_value(event.position.x) else {
            return;
        };
        if drag.pan {
            let Some(domain) = self.with_timeline_model(|model| model.domain) else {
                return;
            };
            let initial = drag.initial_viewport.unwrap_or(domain);
            let delta = drag.start_value - value;
            self.dispatch(
                Action::SetTimelineViewport(Some(clamp_range(
                    (initial.0 + delta, initial.1 + delta),
                    domain,
                ))),
                window,
                cx,
            );
        } else {
            self.dispatch(
                Action::SetTimelineSelection(Some(ordered((drag.start_value, value)))),
                window,
                cx,
            );
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
        if drag.pan {
            cx.notify();
            return;
        }
        let Some(end) = self.timeline_value(event.position.x) else {
            return;
        };
        let Some((viewport, domain)) =
            self.with_timeline_model(|model| (model.viewport, model.domain))
        else {
            return;
        };
        let minimum = ((viewport.1 - viewport.0)
            / self.core.trajectory_data.records.len().max(1) as f64)
            .max(f64::EPSILON);
        let mut range = ordered((drag.start_value, end));
        if range.1 - range.0 < minimum {
            let center = (range.0 + range.1) / 2.0;
            range = clamp_range((center - minimum / 2.0, center + minimum / 2.0), domain);
        }
        self.dispatch(Action::SetTimelineSelection(Some(range)), window, cx);
        self.scroll_trajectory_range_into_view(range, cx);
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
        let Some((viewport, domain)) =
            self.with_timeline_model(|model| (model.viewport, model.domain))
        else {
            return;
        };
        let Some(anchor) = self.timeline_value(event.position.x) else {
            return;
        };
        let delta = event.delta.pixel_delta(window.line_height()).y;
        if delta == px(0.0) {
            return;
        }
        let factor = if delta > px(0.0) { 1.25 } else { 0.8 };
        let span = viewport.1 - viewport.0;
        let minimum = if self.core.trajectory.mode == TimelineMode::Sequence {
            4.0_f64.min(domain.1 - domain.0)
        } else {
            20.0_f64.min(domain.1 - domain.0)
        };
        let new_span = (span * factor).clamp(minimum.max(f64::EPSILON), domain.1 - domain.0);
        let ratio = ((anchor - viewport.0) / span.max(f64::EPSILON)).clamp(0.0, 1.0);
        let range = (anchor - new_span * ratio, anchor + new_span * (1.0 - ratio));
        self.dispatch(
            Action::SetTimelineViewport(Some(clamp_range(range, domain))),
            window,
            cx,
        );
        cx.stop_propagation();
    }

    fn zoom_to_timeline_selection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(range) = self.core.trajectory.selected_range.map(ordered) else {
            return;
        };
        if range.1 - range.0 <= f64::EPSILON {
            return;
        }
        self.dispatch(Action::SetTimelineViewport(Some(range)), window, cx);
    }

    fn timeline_value(&self, x: gpui::Pixels) -> Option<f64> {
        let bounds = self.timeline_bounds?;
        let fraction = ((x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0);
        self.with_timeline_model(|model| {
            model.viewport.0 + f64::from(fraction) * (model.viewport.1 - model.viewport.0)
        })
    }

    fn update_timeline_hover(&mut self, position: Point<Pixels>, cx: &mut Context<Self>) {
        let Some(bounds) = self.timeline_bounds else {
            return;
        };
        let fraction =
            f64::from(((position.x - bounds.origin.x) / bounds.size.width).clamp(0.0, 1.0));
        let local_y = f32::from(position.y - bounds.origin.y);
        let record_index = self
            .with_timeline_model(|model| {
                timeline_record_at(model, &self.core.trajectory_data.records, fraction, local_y)
            })
            .flatten();
        let hover = Some(TimelineHoverState {
            fraction,
            record_index,
        });
        if self.timeline_hover != hover {
            self.timeline_hover = hover;
            cx.notify();
        }
    }
}

fn timeline_lane_label(label: &'static str) -> gpui::AnyElement {
    div()
        .flex()
        .flex_none()
        .items_center()
        .justify_end()
        .h(px(10.0))
        .w_full()
        .overflow_hidden()
        .child(label)
        .into_any_element()
}

fn timeline_lane_at(local_y: f32) -> Option<TrajectoryLane> {
    match local_y {
        7.0..=15.0 => Some(TrajectoryLane::Input),
        21.0..=29.0 => Some(TrajectoryLane::Model),
        35.0..=43.0 => Some(TrajectoryLane::Tools),
        _ => None,
    }
}

fn timeline_record_at(
    model: &TimelineModel,
    records: &[TrajectoryRecord],
    fraction: f64,
    local_y: f32,
) -> Option<usize> {
    let lane = timeline_lane_at(local_y)?;
    model.cells.iter().rev().find_map(|cell| {
        let right = (cell.left + cell.width.max(0.002)).min(1.0);
        (records[cell.index].lane == lane && fraction >= cell.left && fraction <= right)
            .then_some(cell.index)
    })
}

fn nested_segment_geometry(cell: TimelineCell) -> Option<(f64, f64)> {
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

fn cell_intersects_range(cell: TimelineCell, range: (f64, f64)) -> bool {
    let range = ordered(range);
    cell.start <= range.1 && cell.end >= range.0
}

fn geometry_cell_intersects_range(cell: TimelineGeometryCell, range: (f64, f64)) -> bool {
    let range = ordered(range);
    cell.start <= range.1 && cell.end >= range.0
}

fn focus_scroll_target(positions: &[usize]) -> Option<(usize, ScrollStrategy)> {
    let first = positions.first().copied()?;
    Some(if positions.len() > 12 {
        (first, ScrollStrategy::Top)
    } else {
        (positions[positions.len() / 2], ScrollStrategy::Center)
    })
}

#[cfg(test)]
fn timeline_model(
    records: &[TrajectoryRecord],
    mode: TimelineMode,
    viewport: Option<(f64, f64)>,
) -> Option<TimelineModel> {
    timeline_geometry(records, mode).map(|geometry| project_timeline(&geometry, viewport))
}

fn timeline_geometry(records: &[TrajectoryRecord], mode: TimelineMode) -> Option<TimelineGeometry> {
    if records.is_empty() {
        return None;
    }
    let raw = records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let start = record
                .timing
                .started
                .as_ref()
                .map(|time| time.wall_time_ms as f64);
            let duration = record
                .timing
                .duration_ns()
                .map(|ns| ns as f64 / 1_000_000.0);
            let end = start
                .zip(duration)
                .map(|(start, duration)| start + duration)
                .or(start);
            (index, start, end, duration)
        })
        .collect::<Vec<_>>();
    let busy_timeline = (mode == TimelineMode::Duration).then(|| {
        BusyTimeline::new(
            raw.iter()
                .filter_map(|(_, start, end, duration)| {
                    (duration.unwrap_or_default() > 0.0).then_some(((*start)?, (*end)?))
                })
                .collect(),
        )
    });

    let (domain, coordinates) = match mode {
        TimelineMode::Sequence => {
            let coordinates = raw
                .iter()
                .enumerate()
                .map(|(sequence, (index, _, _, _))| {
                    (*index, sequence as f64, sequence as f64 + 1.0)
                })
                .collect::<Vec<_>>();
            ((0.0, records.len() as f64), coordinates)
        }
        TimelineMode::Actual => {
            let start = raw
                .iter()
                .filter_map(|(_, start, _, _)| *start)
                .reduce(f64::min)?;
            let mut end = raw
                .iter()
                .filter_map(|(_, _, end, _)| *end)
                .reduce(f64::max)?;
            if end <= start {
                end = start + 1.0;
            }
            let coordinates = raw
                .iter()
                .filter_map(|(index, start, end, _)| Some((*index, (*start)?, (*end)?)))
                .collect::<Vec<_>>();
            ((start, end), coordinates)
        }
        TimelineMode::Duration => {
            let busy = busy_timeline
                .as_ref()
                .expect("duration mode builds a busy timeline");
            let domain = (0.0, busy.total().max(1.0));
            let coordinates = raw
                .iter()
                .filter_map(|(index, start, end, duration)| {
                    let start = busy.compressed_time((*start)?);
                    let width = duration.unwrap_or_default();
                    let end = if width > 0.0 {
                        start + width
                    } else {
                        busy.compressed_time((*end)?)
                    };
                    Some((*index, start, end))
                })
                .collect::<Vec<_>>();
            (domain, coordinates)
        }
    };
    let mut cells = Vec::new();
    for (index, start, end) in coordinates {
        let record = &records[index];
        let inner_segment = match record.kind {
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
        };
        let execution = inner_segment.and_then(|(start, finish)| {
            let start = start.wall_time_ms as f64;
            let raw_finish = start
                + finish.duration_since(match record.kind {
                    TrajectoryKind::Tool => record.timing.execution_started.as_ref()?,
                    TrajectoryKind::Assistant => record.timing.first_token.as_ref()?,
                    _ => return None,
                })? as f64
                    / 1_000_000.0;
            let (start, finish) = if mode == TimelineMode::Duration {
                let busy = busy_timeline
                    .as_ref()
                    .expect("duration mode builds a busy timeline");
                (
                    busy.compressed_time(start),
                    busy.compressed_time(raw_finish),
                )
            } else if mode == TimelineMode::Sequence {
                return None;
            } else {
                (start, raw_finish)
            };
            Some((start, finish))
        });
        cells.push(TimelineGeometryCell {
            index,
            start,
            end,
            execution,
        });
    }
    Some(TimelineGeometry { domain, cells })
}

fn project_timeline(geometry: &TimelineGeometry, viewport: Option<(f64, f64)>) -> TimelineModel {
    let viewport = clamp_range(viewport.unwrap_or(geometry.domain), geometry.domain);
    let span = (viewport.1 - viewport.0).max(f64::EPSILON);
    let cells = geometry
        .cells
        .iter()
        .filter(|cell| cell.end >= viewport.0 && cell.start <= viewport.1)
        .map(|cell| {
            let left = ((cell.start - viewport.0) / span).clamp(0.0, 1.0);
            let right = ((cell.end - viewport.0) / span).clamp(0.0, 1.0);
            let execution = cell.execution.map(|(start, finish)| {
                (
                    ((start - viewport.0) / span).clamp(0.0, 1.0),
                    ((finish - start) / span).max(0.002),
                )
            });
            TimelineCell {
                index: cell.index,
                start: cell.start,
                end: cell.end,
                left,
                width: (right - left).max(0.002),
                execution_left: execution.map(|value| value.0),
                execution_width: execution.map(|value| value.1),
            }
        })
        .collect();
    TimelineModel {
        domain: geometry.domain,
        viewport,
        cells,
    }
}

fn merge_intervals(mut intervals: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    intervals.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut merged: Vec<(f64, f64)> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

#[derive(Clone, Debug)]
struct BusyTimeline {
    intervals: Vec<(f64, f64)>,
    elapsed_before: Vec<f64>,
    total: f64,
}

impl BusyTimeline {
    fn new(intervals: Vec<(f64, f64)>) -> Self {
        let intervals = merge_intervals(intervals);
        let mut elapsed_before = Vec::with_capacity(intervals.len());
        let mut total = 0.0;
        for (start, end) in &intervals {
            elapsed_before.push(total);
            total += end - start;
        }
        Self {
            intervals,
            elapsed_before,
            total,
        }
    }

    fn total(&self) -> f64 {
        self.total
    }

    fn compressed_time(&self, time: f64) -> f64 {
        let index = self.intervals.partition_point(|(_, end)| *end <= time);
        let Some((start, _)) = self.intervals.get(index) else {
            return self.total;
        };
        let elapsed = self.elapsed_before[index];
        if time <= *start {
            elapsed
        } else {
            elapsed + time - start
        }
    }
}

fn ordered(range: (f64, f64)) -> (f64, f64) {
    (range.0.min(range.1), range.0.max(range.1))
}

fn clamp_range(range: (f64, f64), domain: (f64, f64)) -> (f64, f64) {
    let range = ordered(range);
    let domain = ordered(domain);
    let span = (range.1 - range.0).min(domain.1 - domain.0);
    let start = range.0.clamp(domain.0, domain.1 - span);
    (start, start + span)
}

fn normalized_range(range: (f64, f64), viewport: (f64, f64)) -> (f64, f64) {
    let range = ordered(range);
    let span = (viewport.1 - viewport.0).max(f64::EPSILON);
    let left = ((range.0 - viewport.0) / span).clamp(0.0, 1.0);
    let right = ((range.1 - viewport.0) / span).clamp(0.0, 1.0);
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
        let started_at = format_clock(started.wall_time_ms);
        parts.push(if let Some(duration) = record.timing.duration_ns() {
            let completed_at = started
                .wall_time_ms
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
        .map(|time| format_wall(time.wall_time_ms, unix))
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

fn timing_duration(record: &TrajectoryRecord) -> String {
    if record.timing.started.is_none() {
        return "Not recorded".into();
    }
    if record.timing.completed.is_none() {
        return "Pending".into();
    }
    format_duration(record.timing.duration_ns())
}

fn assistant_ttft(record: &TrajectoryRecord) -> String {
    if record.timing.started.is_none() {
        "Step start unavailable".into()
    } else if record.timing.first_token.is_none() {
        "First token unavailable".into()
    } else {
        format_duration(record.timing.ttft_ns())
    }
}

fn assistant_generation(record: &TrajectoryRecord) -> String {
    if record.timing.completed.is_none() {
        "Pending".into()
    } else if record.timing.first_token.is_none() {
        "First token unavailable".into()
    } else {
        format_duration(record.timing.generation_ns())
    }
}

fn assistant_throughput(record: &TrajectoryRecord) -> String {
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
    use std::path::PathBuf;

    use kcastle_agent::EventTime;
    use proptest::prelude::*;

    use crate::domain::{
        MessageId, RecordTiming, TimelineMode, TrajectoryKind, TrajectoryLane, TrajectoryRecord,
        TrajectoryStatus,
    };

    use super::{
        BusyTimeline, ScrollStrategy, TimelineCell, TimelineModelCache, cell_intersects_range,
        clamp_range, focus_scroll_target, nested_segment_geometry, normalized_range,
        record_tooltip, timeline_lane_at, timeline_model, timeline_record_at,
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
            started: Some(time(start)),
            execution_started: Some(time(start + 20)),
            execution_finished: Some(time(end - 20)),
            completed: Some(time(end)),
            ..RecordTiming::default()
        };
        TrajectoryRecord {
            id: MessageId(id),
            source_seq: id,
            kind: TrajectoryKind::Tool,
            lane: TrajectoryLane::Tools,
            title: "tool".into(),
            text: String::new(),
            payload: None,
            raw: String::new(),
            turn: Some(1),
            step: Some(1),
            call_id: Some(format!("call-{id}")),
            status: TrajectoryStatus::Completed,
            timing,
            usage: None,
        }
    }

    #[test]
    fn duration_axis_removes_only_idle_gaps() {
        let timeline = BusyTimeline::new(vec![(0.0, 10.0), (5.0, 20.0), (30.0, 35.0)]);
        assert_eq!(timeline.intervals, [(0.0, 20.0), (30.0, 35.0)]);
        assert_eq!(timeline.compressed_time(15.0), 15.0);
        assert_eq!(timeline.compressed_time(32.0), 22.0);
    }

    proptest! {
        #[test]
        fn busy_timeline_binary_lookup_matches_the_linear_definition(
            input in prop::collection::vec((0_u32..10_000, 1_u32..1_000), 0..100),
            query in 0_u32..12_000,
        ) {
            let intervals = input
                .into_iter()
                .map(|(start, duration)| (f64::from(start), f64::from(start + duration)))
                .collect::<Vec<_>>();
            let timeline = BusyTimeline::new(intervals);
            let query = f64::from(query);
            let mut expected = 0.0;
            for (start, end) in &timeline.intervals {
                if query <= *start {
                    break;
                }
                if query < *end {
                    expected += query - start;
                    break;
                }
                expected += end - start;
            }
            prop_assert!((timeline.compressed_time(query) - expected).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn timeline_cache_reuses_geometry_when_only_the_viewport_changes() {
        let workspace = PathBuf::from("workspace");
        let session = PathBuf::from("session.jsonl");
        let records = [record(1, 0, 100)];
        let mut cache = TimelineModelCache::new(
            workspace.clone(),
            session.clone(),
            4,
            &records,
            TimelineMode::Duration,
            Some((0.0, 100.0)),
        );
        assert!(cache.geometry_matches(&workspace, &session, 4, TimelineMode::Duration));
        assert!(!cache.geometry_matches(&workspace, &session, 5, TimelineMode::Duration));

        let geometry_before = cache.geometry.as_ref().unwrap().cells.clone();
        cache.set_viewport(Some((10.0, 60.0)));
        assert_eq!(cache.viewport, Some((10.0, 60.0)));
        assert_eq!(cache.geometry.as_ref().unwrap().cells, geometry_before);
        assert_eq!(cache.model.as_ref().unwrap().viewport, (10.0, 60.0));
    }

    #[test]
    fn viewport_and_selection_are_clamped_and_normalized() {
        assert_eq!(clamp_range((-5.0, 5.0), (0.0, 20.0)), (0.0, 10.0));
        assert_eq!(normalized_range((5.0, 10.0), (0.0, 20.0)), (0.25, 0.25));
    }

    #[test]
    fn timeline_modes_keep_distinct_coordinate_semantics() {
        let records = [record(1, 0, 100), record(2, 200, 250)];
        let sequence = timeline_model(&records, TimelineMode::Sequence, None).unwrap();
        assert_eq!(sequence.domain, (0.0, 2.0));
        let actual = timeline_model(&records, TimelineMode::Actual, None).unwrap();
        assert_eq!(actual.domain, (1_000.0, 1_250.0));
        let duration = timeline_model(&records, TimelineMode::Duration, None).unwrap();
        assert_eq!(duration.domain, (0.0, 150.0));
    }

    #[test]
    fn actual_timeline_nests_execution_inside_tool_lifecycle() {
        let model = timeline_model(&[record(1, 0, 100)], TimelineMode::Actual, None).unwrap();
        let cell = model.cells[0];
        assert!((cell.left - 0.0).abs() < 0.000_001);
        assert!((cell.width - 1.0).abs() < 0.000_001);
        assert!((cell.execution_left.unwrap() - 0.2).abs() < 0.000_001);
        assert!((cell.execution_width.unwrap() - 0.6).abs() < 0.000_001);
    }

    #[test]
    fn timeline_hover_hits_only_bars_and_prefers_the_topmost_record() {
        let records = [record(1, 0, 100), record(2, 0, 100)];
        let model = timeline_model(&records, TimelineMode::Actual, None).unwrap();

        assert_eq!(timeline_lane_at(7.0), Some(TrajectoryLane::Input));
        assert_eq!(timeline_lane_at(20.0), None);
        assert_eq!(timeline_lane_at(35.0), Some(TrajectoryLane::Tools));
        assert_eq!(timeline_record_at(&model, &records, 0.5, 36.0), Some(1));
        assert_eq!(timeline_record_at(&model, &records, 0.5, 22.0), None);
    }

    #[test]
    fn assistant_hover_tooltip_uses_dsh_timing_shape() {
        let mut assistant = record(1, 0, 100);
        assistant.kind = TrajectoryKind::Assistant;
        assistant.lane = TrajectoryLane::Model;
        assistant.timing.first_token = Some(time(20));

        let tooltip = record_tooltip(&assistant);
        assert!(tooltip.starts_with("ASSISTANT\n"));
        assert!(tooltip.contains(" → "));
        assert!(tooltip.contains("Total 100.0 ms"));
        assert!(tooltip.contains("TTFT 20.0 ms · Decoding 80.0 ms"));
    }

    #[test]
    fn clipped_nested_segment_does_not_create_an_invalid_width_range() {
        let cell = TimelineCell {
            index: 0,
            start: 10.0,
            end: 20.0,
            left: 0.0,
            width: 1.0,
            execution_left: Some(1.0),
            execution_width: Some(0.2),
        };
        assert_eq!(nested_segment_geometry(cell), None);
        assert!(cell_intersects_range(cell, (15.0, 30.0)));
        assert!(!cell_intersects_range(cell, (21.0, 30.0)));
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
}
