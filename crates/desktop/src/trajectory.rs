use gpui::{
    Context, HighlightStyle, InteractiveElement, IntoElement, ParentElement, ScrollStrategy,
    SharedString, StatefulInteractiveElement, Styled, StyledText, Window, div,
    prelude::FluentBuilder, px, relative, rgba, uniform_list,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::Input;
use gpui_component::resizable::{h_resizable, resizable_panel};
use gpui_component::scroll::ScrollableElement;
use gpui_component::text::TextView;
use gpui_component::tooltip::Tooltip;
use gpui_component::{Disableable, IconName, Sizable};
use time::{OffsetDateTime, UtcOffset, macros::format_description};

use crate::app::DesktopApp;
use crate::domain::{Action, DetailsTab, Message, Role};
use crate::layout::TrajectoryMode;
use crate::ui_theme::{TrajectoryPalette, metrics, trajectory_palette};

#[derive(Clone, Debug)]
enum LedgerRow {
    Message {
        index: usize,
        turn: usize,
        step: usize,
    },
    Summary {
        key: usize,
        turn: usize,
        text: String,
    },
}

impl DesktopApp {
    pub(crate) fn trajectory_panel(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let rows = self.ledger_rows(&query);
        let selected = self.core.details.selected.and_then(|selected| {
            self.core
                .conversation
                .messages
                .iter()
                .position(|message| message.key == selected)
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
                    .child(
                        selected
                            .map(|index| {
                                if narrow_details {
                                    div()
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
                                                .child(self.details_panel(index, cx)),
                                        )
                                        .into_any_element()
                                } else {
                                    h_resizable("trajectory-panes")
                                        .child(
                                            resizable_panel()
                                                .size_range(px(320.0)..px(2_000.0))
                                                .child(self.trajectory_ledger(&rows, cx)),
                                        )
                                        .child(
                                            resizable_panel()
                                                .size(px(410.0))
                                                .size_range(px(320.0)..px(720.0))
                                                .child(self.details_panel(index, cx)),
                                        )
                                        .into_any_element()
                                }
                            })
                            .unwrap_or_else(|| {
                                self.trajectory_ledger(&rows, cx).into_any_element()
                            }),
                    ),
            )
    }

    fn trajectory_ledger(&self, rows: &[LedgerRow], cx: &mut Context<Self>) -> impl IntoElement {
        let rows = rows.to_vec();
        uniform_list(
            "trajectory-ledger",
            rows.len(),
            cx.processor(move |this, range: std::ops::Range<usize>, _, cx| {
                range
                    .filter_map(|index| rows.get(index))
                    .map(|row| this.ledger_row(row, cx))
                    .collect::<Vec<_>>()
            }),
        )
        .size_full()
        .track_scroll(&self.trajectory_scroll)
    }

    fn trajectory_toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = trajectory_palette(cx);
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        let match_count = (!query.is_empty()).then(|| {
            self.core
                .conversation
                .messages
                .iter()
                .filter(|message| message_matches(message, &query))
                .count()
        });
        let has_timing = self
            .core
            .conversation
            .messages
            .iter()
            .any(|message| message.duration_ms.is_some());
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
                    .gap_1()
                    .child(
                        Button::new("trajectory-duration")
                            .icon(IconName::Calendar)
                            .label("Duration")
                            .ghost()
                            .compact()
                            .text_color(colors.label_tertiary)
                            .disabled(!has_timing)
                            .tooltip(if has_timing {
                                "Switch equal-width and recorded-duration blocks"
                            } else {
                                "Recorded timing is unavailable in this JSONL session"
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::ToggleTrajectoryDuration, window, cx);
                            })),
                    )
                    .child(
                        Button::new("trajectory-turns")
                            .icon(if self.core.trajectory.collapsed_turns {
                                IconName::Plus
                            } else {
                                IconName::Minus
                            })
                            .label("Turns")
                            .ghost()
                            .compact()
                            .text_color(colors.label_tertiary)
                            .tooltip(if self.core.trajectory.collapsed_turns {
                                "Expand turns"
                            } else {
                                "Collapse turns"
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::ToggleTrajectoryTurns, window, cx);
                            })),
                    )
                    .child(
                        Button::new("trajectory-calls")
                            .icon(if self.core.trajectory.collapsed_calls {
                                IconName::Plus
                            } else {
                                IconName::Minus
                            })
                            .label("Calls")
                            .ghost()
                            .compact()
                            .text_color(colors.label_tertiary)
                            .tooltip(if self.core.trajectory.collapsed_calls {
                                "Expand calls"
                            } else {
                                "Collapse calls"
                            })
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
                    .children(match_count.map(|count| {
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
        div()
            .flex()
            .h(px(50.0))
            .border_b_1()
            .border_color(colors.border_l2)
            .child(
                div()
                    .flex()
                    .flex_col()
                    .justify_center()
                    .w(px(44.0))
                    .pr_1()
                    .items_end()
                    .gap(px(3.0))
                    .text_xs()
                    .text_color(colors.label_caption)
                    .child("Input")
                    .child("Model")
                    .child("Tools"),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .justify_center()
                    .min_w(px(0.0))
                    .overflow_hidden()
                    .gap(px(6.0))
                    .child(self.overview_lane(Role::User, query, cx))
                    .child(self.overview_model_lane(query, cx))
                    .child(self.overview_lane(Role::Tool, query, cx)),
            )
    }

    fn overview_model_lane(&self, query: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let geometry = timeline_geometry(
            &self.core.conversation.messages,
            self.core.trajectory.show_duration,
        );
        div().relative().h(px(8.0)).children(
            self.core
                .conversation
                .messages
                .iter()
                .enumerate()
                .filter(|(_, message)| matches!(message.role, Role::Reasoning | Role::Assistant))
                .map(|(index, message)| {
                    self.overview_block(index, message, geometry[index], query, cx)
                }),
        )
    }

    fn overview_lane(&self, role: Role, query: &str, cx: &mut Context<Self>) -> impl IntoElement {
        let geometry = timeline_geometry(
            &self.core.conversation.messages,
            self.core.trajectory.show_duration,
        );
        div().relative().h(px(8.0)).children(
            self.core
                .conversation
                .messages
                .iter()
                .enumerate()
                .filter(move |(_, message)| message.role == role)
                .map(|(index, message)| {
                    self.overview_block(index, message, geometry[index], query, cx)
                }),
        )
    }

    fn overview_block(
        &self,
        index: usize,
        message: &Message,
        geometry: (f32, f32),
        query: &str,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let selected = self.core.details.selected == Some(message.key);
        let matched = query.is_empty() || message_matches(message, query);
        let (position, width) = geometry;
        let tooltip = format!("{}\n{}", role_label(message.role), timing_text(message));
        let record_color = if message.failed {
            colors.error
        } else {
            role_foreground(message.role, colors)
        };
        div()
            .id(("timeline-record", index))
            .absolute()
            .left(relative(position))
            .top_0()
            .w(relative(width))
            .h(px(8.0))
            .rounded(px(1.5))
            .bg(if matched {
                record_color
            } else {
                record_color.opacity(0.18)
            })
            .when(selected, |block| {
                block.border_1().border_color(colors.primary)
            })
            .cursor_pointer()
            .tooltip(move |window, cx| Tooltip::new(tooltip.clone()).build(window, cx))
            .on_click(
                cx.listener(move |this, _, window, cx| this.select_trajectory(index, window, cx)),
            )
            .into_any_element()
    }

    fn ledger_rows(&self, query: &str) -> Vec<LedgerRow> {
        if self.core.trajectory.collapsed_turns {
            return self.collapsed_turn_rows(query);
        }
        if self.core.trajectory.collapsed_calls {
            return self.collapsed_call_rows(query);
        }
        self.core
            .conversation
            .messages
            .iter()
            .enumerate()
            .filter_map(|(index, message)| {
                (query.is_empty() || message_matches(message, query)).then_some(
                    LedgerRow::Message {
                        index,
                        turn: message.turn,
                        step: message.step,
                    },
                )
            })
            .collect()
    }

    fn collapsed_turn_rows(&self, query: &str) -> Vec<LedgerRow> {
        let mut rows = Vec::new();
        let mut turn = 0;
        let mut index = 0;
        while index < self.core.conversation.messages.len() {
            let message = &self.core.conversation.messages[index];
            if message.role != Role::User {
                if turn == 0 && (query.is_empty() || message_matches(message, query)) {
                    rows.push(LedgerRow::Message {
                        index,
                        turn: 0,
                        step: 0,
                    });
                }
                index += 1;
                continue;
            }
            turn += 1;
            let start = index;
            index += 1;
            while index < self.core.conversation.messages.len()
                && self.core.conversation.messages[index].role != Role::User
            {
                index += 1;
            }
            let body = &self.core.conversation.messages[start + 1..index];
            let steps = body
                .iter()
                .map(|message| message.step)
                .filter(|step| *step > 0)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let calls = body
                .iter()
                .filter(|message| message.role == Role::Tool)
                .count();
            let matches = query.is_empty()
                || self.core.conversation.messages[start..index]
                    .iter()
                    .any(|message| message_matches(message, query));
            if matches {
                rows.push(LedgerRow::Message {
                    index: start,
                    turn,
                    step: 0,
                });
                rows.push(LedgerRow::Summary {
                    key: start,
                    turn,
                    text: format!(
                        "… {steps} {} · {calls} {}",
                        plural(steps, "step", "steps"),
                        plural(calls, "tool call", "tool calls")
                    ),
                });
            }
        }
        rows
    }

    fn collapsed_call_rows(&self, query: &str) -> Vec<LedgerRow> {
        let mut rows = Vec::new();
        let mut calls = 0;
        let mut summary_key = 0;
        for (index, message) in self.core.conversation.messages.iter().enumerate() {
            if calls > 0 && matches!(message.role, Role::User | Role::Reasoning | Role::Assistant) {
                rows.push(call_summary(
                    summary_key,
                    self.core.conversation.messages[summary_key].turn,
                    calls,
                ));
                calls = 0;
            }
            if message.role == Role::Tool {
                if calls == 0 {
                    summary_key = index;
                }
                calls += 1;
            } else if query.is_empty() || message_matches(message, query) {
                rows.push(LedgerRow::Message {
                    index,
                    turn: message.turn,
                    step: message.step,
                });
            }
        }
        if calls > 0 {
            let turn = self
                .core
                .conversation
                .messages
                .last()
                .map(|message| message.turn)
                .unwrap_or_default();
            rows.push(call_summary(summary_key, turn, calls));
        }
        if query.is_empty() {
            rows
        } else {
            rows.into_iter()
                .filter(|row| match row {
                    LedgerRow::Message { .. } => true,
                    LedgerRow::Summary { text, .. } => text.to_lowercase().contains(query),
                })
                .collect()
        }
    }

    fn ledger_row(&self, row: &LedgerRow, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        match row {
            LedgerRow::Message { index, turn, step } => {
                let index = *index;
                let message = &self.core.conversation.messages[index];
                let selected = self.core.details.selected == Some(message.key);
                div()
                    .id(("trajectory-row", index))
                    .flex()
                    .w_full()
                    .items_center()
                    .h(px(metrics::LEDGER_ROW_HEIGHT))
                    .px_3()
                    .border_b_1()
                    .border_color(colors.border_l1)
                    .when(selected, |item| item.bg(colors.active))
                    .hover(move |item| item.bg(colors.hover))
                    .cursor_pointer()
                    .tab_index(0)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.select_trajectory(index, window, cx)
                    }))
                    .on_key_down(cx.listener(
                        move |this, event: &gpui::KeyDownEvent, window, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                this.select_trajectory(index, window, cx);
                            }
                        },
                    ))
                    .child(turn_marker(*turn, *step, message.role, colors))
                    .child(role_chip(message.role, colors))
                    .children(message.title.clone().map(|title| {
                        div()
                            .w(px(72.0))
                            .flex_none()
                            .truncate()
                            .font_family("SF Mono")
                            .text_sm()
                            .text_color(colors.label_primary)
                            .child(title)
                    }))
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .truncate()
                            .text_sm()
                            .text_color(if message.failed {
                                colors.error
                            } else {
                                colors.label_primary
                            })
                            .child(message_summary(message)),
                    )
                    .into_any_element()
            }
            LedgerRow::Summary { key, turn, text } => div()
                .id(("trajectory-summary", *key))
                .flex()
                .w_full()
                .items_center()
                .h(px(metrics::LEDGER_ROW_HEIGHT))
                .px_3()
                .border_b_1()
                .border_color(colors.border_l1)
                .cursor_pointer()
                .tab_index(0)
                .hover(move |item| item.bg(colors.hover))
                .on_click(cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::ExpandTrajectoryGroups, window, cx);
                }))
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.dispatch(Action::ExpandTrajectoryGroups, window, cx);
                    }
                }))
                .child(turn_marker(*turn, 0, Role::Notice, colors))
                .child(
                    div()
                        .pl(px(8.0))
                        .text_sm()
                        .text_color(colors.label_secondary)
                        .child(text.clone()),
                )
                .into_any_element(),
        }
    }

    fn details_panel(&self, index: usize, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        let message = &self.core.conversation.messages[index];
        let tabs = detail_tabs(message).iter().copied().fold(
            div()
                .flex()
                .h(px(metrics::TAB_HEIGHT))
                .overflow_hidden()
                .border_b_1()
                .border_color(colors.border_l2),
            |tabs, tab| tabs.child(details_tab(tab, self.core.details.tab == tab, colors, cx)),
        );
        div()
            .flex()
            .flex_col()
            .w_full()
            .h_full()
            .border_l_1()
            .border_color(colors.border_l2)
            .bg(colors.background)
            .text_color(colors.label_primary)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(metrics::DETAILS_HEADER_HEIGHT))
                    .px_4()
                    .border_b_1()
                    .border_color(colors.border_l2)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(role_chip(message.role, colors))
                            .child(
                                div()
                                    .text_sm()
                                    .text_color(colors.label_tertiary)
                                    .child(detail_location(message)),
                            ),
                    )
                    .child(
                        Button::new("close-trajectory-details")
                            .icon(IconName::Close)
                            .ghost()
                            .compact()
                            .text_color(colors.label_secondary)
                            .tooltip("Close details")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.dispatch(Action::SelectDetails(None), window, cx);
                            })),
                    ),
            )
            .child(tabs)
            .child(
                div()
                    .id("trajectory-details-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.0))
                    .p_4()
                    .track_scroll(&self.details_scroll)
                    .overflow_y_scrollbar()
                    .child(self.details_body(index, message, cx)),
            )
            .into_any_element()
    }

    fn details_body(
        &self,
        index: usize,
        message: &Message,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = trajectory_palette(cx);
        match self.core.details.tab {
            DetailsTab::Summary => match message.role {
                Role::Tool => div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .child(summary_pair(
                        "Hierarchy",
                        hierarchy_label(message.role),
                        colors,
                    ))
                    .child(summary_pair("Status", status_label(message), colors))
                    .children(message.payload.as_deref().map(|value| {
                        details_linked_section(
                            ("detail-summary-payload-link", index),
                            "Payload",
                            details_markdown(
                                detail_content_id("detail-summary-payload", message),
                                pretty_json(value),
                                Some("json"),
                                cx,
                            ),
                            colors,
                            cx.listener(|this, _, _, cx| {
                                this.open_details_tab(DetailsTab::Payload, cx);
                            }),
                        )
                    }))
                    .child(details_linked_section(
                        ("detail-summary-result-link", index),
                        "Result",
                        details_markdown(
                            detail_content_id("detail-summary-result", message),
                            result_text(message),
                            result_language(message),
                            cx,
                        ),
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.open_details_tab(DetailsTab::Result, cx);
                        }),
                    ))
                    .child(details_linked_section(
                        ("detail-summary-schema-link", index),
                        "Schema",
                        details_schema(("detail-summary-schema", index), message, colors),
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.open_details_tab(DetailsTab::Schema, cx);
                        }),
                    ))
                    .child(details_linked_section(
                        ("detail-summary-timing-link", index),
                        "Timing",
                        details_timing(message, colors),
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.open_details_tab(DetailsTab::Timing, cx);
                        }),
                    ))
                    .into_any_element(),
                _ => div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .child(summary_pair("Status", status_label(message), colors))
                    .when(matches!(message.role, Role::User | Role::Notice), |body| {
                        body.child(summary_pair(
                            "Duration",
                            &format_duration(message.duration_ms),
                            colors,
                        ))
                    })
                    .child(details_linked_section(
                        ("detail-summary-preview-link", index),
                        "Preview",
                        details_markdown(
                            detail_content_id("detail-summary-preview", message),
                            message.text.clone(),
                            None,
                            cx,
                        ),
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.open_details_tab(DetailsTab::Preview, cx);
                        }),
                    ))
                    .into_any_element(),
            },
            DetailsTab::Preview => details_markdown(
                detail_content_id("detail-preview", message),
                message.text.clone(),
                None,
                cx,
            ),
            DetailsTab::Raw => {
                details_plain_code(("detail-raw", index).into(), message.text.clone(), colors)
            }
            DetailsTab::Payload => details_markdown(
                detail_content_id("detail-payload", message),
                message
                    .payload
                    .as_deref()
                    .map(pretty_json)
                    .unwrap_or_else(|| "This record has no tool payload.".into()),
                message.payload.as_ref().map(|_| "json"),
                cx,
            ),
            DetailsTab::Result => details_markdown(
                detail_content_id("detail-result", message),
                result_text(message),
                result_language(message),
                cx,
            ),
            DetailsTab::Schema => details_schema(("detail-schema", index), message, colors),
            DetailsTab::Timing => details_timing(message, colors),
        }
    }

    fn select_trajectory(&mut self, index: usize, window: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(
            Action::SelectDetails(Some(self.core.conversation.messages[index].key)),
            window,
            cx,
        );
        self.dispatch_local(Action::SetDetailsTab(DetailsTab::Summary), cx);
        self.dispatch_local(Action::ExpandTrajectoryGroups, cx);
        self.details_scroll
            .set_offset(gpui::point(px(0.0), px(0.0)));
        self.scroll_trajectory_to_record(index, cx);
        cx.notify();
    }

    fn open_details_tab(&mut self, tab: DetailsTab, cx: &mut Context<Self>) {
        self.dispatch_local(Action::SetDetailsTab(tab), cx);
        self.details_scroll
            .set_offset(gpui::point(px(0.0), px(0.0)));
        cx.notify();
    }

    pub(crate) fn scroll_trajectory_to_record(&self, record_index: usize, cx: &mut Context<Self>) {
        let query = self
            .trajectory_search
            .read(cx)
            .value()
            .trim()
            .to_lowercase();
        if let Some(row_index) = self.ledger_rows(&query).iter().position(
            |row| matches!(row, LedgerRow::Message { index, .. } if *index == record_index),
        ) {
            self.trajectory_scroll
                .scroll_to_item(row_index, ScrollStrategy::Center);
        }
    }
}

const MARKDOWN_DETAIL_TABS: [DetailsTab; 3] =
    [DetailsTab::Summary, DetailsTab::Preview, DetailsTab::Raw];

const TOOL_DETAIL_TABS: [DetailsTab; 5] = [
    DetailsTab::Summary,
    DetailsTab::Payload,
    DetailsTab::Result,
    DetailsTab::Schema,
    DetailsTab::Timing,
];

fn detail_tabs(message: &Message) -> &'static [DetailsTab] {
    match message.role {
        Role::Tool => &TOOL_DETAIL_TABS,
        Role::User | Role::Reasoning | Role::Assistant | Role::Notice => &MARKDOWN_DETAIL_TABS,
    }
}

fn details_tab(
    tab: DetailsTab,
    selected: bool,
    colors: TrajectoryPalette,
    cx: &mut Context<DesktopApp>,
) -> impl IntoElement {
    let (id, label, width) = match tab {
        DetailsTab::Summary => ("details-summary", "Summary", 76.0),
        DetailsTab::Preview => ("details-preview", "Preview", 72.0),
        DetailsTab::Raw => ("details-raw", "Raw", 48.0),
        DetailsTab::Payload => ("details-payload", "Payload", 72.0),
        DetailsTab::Result => ("details-result", "Result", 62.0),
        DetailsTab::Schema => ("details-schema", "Schema", 70.0),
        DetailsTab::Timing => ("details-timing", "Timing", 65.0),
    };
    div()
        .id(id)
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .w(px(width))
        .h_full()
        .border_b_2()
        .border_color(if selected {
            colors.primary
        } else {
            rgba(0x00000000).into()
        })
        .text_sm()
        .text_color(if selected {
            colors.primary
        } else {
            colors.label_tertiary
        })
        .cursor_pointer()
        .hover(move |item| {
            item.text_color(if selected {
                colors.primary
            } else {
                colors.label_secondary
            })
        })
        .tab_index(0)
        .on_click(cx.listener(move |this, _, _, cx| {
            this.open_details_tab(tab, cx);
        }))
        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, _, cx| {
            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                this.open_details_tab(tab, cx);
            }
        }))
        .child(label)
}

fn call_summary(key: usize, turn: usize, calls: usize) -> LedgerRow {
    LedgerRow::Summary {
        key,
        turn,
        text: format!("… {calls} {}", plural(calls, "tool call", "tool calls")),
    }
}

fn plural<'a>(count: usize, singular: &'a str, plural: &'a str) -> &'a str {
    if count == 1 { singular } else { plural }
}

fn message_matches(message: &Message, query: &str) -> bool {
    role_search_label(message.role).contains(query) || message.search_text.contains(query)
}

fn role_search_label(role: Role) -> &'static str {
    match role {
        Role::User => "user input",
        Role::Reasoning | Role::Assistant => "assistant model reasoning",
        Role::Tool => "tool call tools",
        Role::Notice => "context notice",
    }
}

fn message_summary(message: &Message) -> String {
    match message.role {
        Role::Tool => {
            let payload = message.payload.as_deref().unwrap_or("{}");
            let result = if message.pending {
                "Running…"
            } else {
                message
                    .text
                    .lines()
                    .next()
                    .filter(|line| !line.is_empty())
                    .unwrap_or("(no output)")
            };
            format!("{payload}  →  {result}")
        }
        _ => message
            .text
            .lines()
            .next()
            .filter(|line| !line.is_empty())
            .unwrap_or("(empty)")
            .to_owned(),
    }
}

fn turn_marker(
    turn: usize,
    _step: usize,
    role: Role,
    colors: TrajectoryPalette,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .w(px(54.0))
        .flex_none()
        .text_xs()
        .text_color(if role == Role::User {
            colors.primary
        } else {
            colors.label_caption
        })
        .child(if turn == 0 {
            String::new()
        } else if role == Role::User {
            format!("Turn {turn}")
        } else {
            "•".into()
        })
}

fn role_chip(role: Role, colors: TrajectoryPalette) -> impl IntoElement {
    let (foreground, background) = role_colors(role, colors);
    div().w(px(88.0)).mr(px(12.0)).flex_none().child(
        div()
            .flex()
            .px_2()
            .py(px(2.0))
            .rounded_md()
            .bg(background)
            .font_weight(gpui::FontWeight::SEMIBOLD)
            .text_xs()
            .text_color(foreground)
            .child(role_label(role)),
    )
}

fn role_label(role: Role) -> &'static str {
    match role {
        Role::User => "USER",
        Role::Reasoning | Role::Assistant => "ASSISTANT",
        Role::Tool => "TOOL",
        Role::Notice => "CONTEXT",
    }
}

fn hierarchy_label(role: Role) -> &'static str {
    match role {
        Role::User => "User Message",
        Role::Reasoning => "Assistant Message › Reasoning",
        Role::Assistant => "Assistant Message",
        Role::Tool => "Assistant Message › Tool Call",
        Role::Notice => "Session Context",
    }
}

fn detail_location(message: &Message) -> String {
    let section = if message.turn == 0 {
        "Session".to_owned()
    } else {
        format!("Turn {}", message.turn)
    };
    let group = match message.role {
        Role::User | Role::Notice => "Message".to_owned(),
        Role::Reasoning | Role::Assistant | Role::Tool => {
            format!("Step {}", message.step.max(1))
        }
    };
    format!("{section} · {group}")
}

fn role_foreground(role: Role, colors: TrajectoryPalette) -> gpui::Hsla {
    role_colors(role, colors).0
}

fn role_colors(role: Role, colors: TrajectoryPalette) -> (gpui::Hsla, gpui::Hsla) {
    match role {
        Role::User => (colors.user_foreground, colors.user_background),
        Role::Reasoning | Role::Assistant => {
            (colors.assistant_foreground, colors.assistant_background)
        }
        Role::Tool => (colors.tool_foreground, colors.tool_background),
        Role::Notice => (colors.context_foreground, colors.context_background),
    }
}

fn pretty_json(value: &str) -> String {
    serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| value.to_owned())
}

fn status_label(message: &Message) -> &'static str {
    if message.pending {
        "Running"
    } else if message.failed {
        "Failed"
    } else {
        "Completed"
    }
}

fn format_duration(duration_ms: Option<u128>) -> String {
    duration_ms
        .map(|duration| format!("{duration} ms"))
        .unwrap_or_else(|| "—".into())
}

fn format_started_at(started_at_ms: Option<u128>) -> String {
    let Some(started_at_ms) = started_at_ms else {
        return "Not available".into();
    };
    let Ok(nanoseconds) = i128::try_from(started_at_ms.saturating_mul(1_000_000)) else {
        return "Not available".into();
    };
    let Ok(timestamp) = OffsetDateTime::from_unix_timestamp_nanos(nanoseconds) else {
        return "Not available".into();
    };
    let local = UtcOffset::current_local_offset()
        .map(|offset| timestamp.to_offset(offset))
        .unwrap_or(timestamp);
    local
        .format(format_description!(
            "[year]-[month]-[day] [hour]:[minute]:[second].[subsecond digits:3]"
        ))
        .unwrap_or_else(|_| "Not available".into())
}

fn details_timing(message: &Message, colors: TrajectoryPalette) -> gpui::AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(summary_pair(
            "Started",
            &format_started_at(message.started_at_ms),
            colors,
        ))
        .child(summary_pair(
            "Duration",
            &format_duration(message.duration_ms),
            colors,
        ))
        .child(summary_pair(
            "Timing source",
            if message.started_at_ms.is_some() {
                "Session timestamps"
            } else {
                "Not available"
            },
            colors,
        ))
        .into_any_element()
}

fn result_text(message: &Message) -> String {
    if message.pending {
        "Tool call is still running.".into()
    } else if message.text.is_empty() {
        "(no output)".into()
    } else {
        message.text.clone()
    }
}

fn timing_text(message: &Message) -> String {
    match (message.started_at_ms, message.duration_ms) {
        (Some(_), Some(duration)) => format!("{duration} ms · Live desktop events"),
        (Some(_), None) => "Running · Live desktop events".into(),
        _ => "Timing unavailable · Not recorded in JSONL".into(),
    }
}

fn timeline_geometry(messages: &[Message], by_duration: bool) -> Vec<(f32, f32)> {
    if by_duration {
        let first = messages
            .iter()
            .filter_map(|message| message.started_at_ms)
            .min();
        let last = messages
            .iter()
            .filter_map(|message| {
                message
                    .started_at_ms
                    .map(|started| started + message.duration_ms.unwrap_or_default())
            })
            .max();
        if let (Some(first), Some(last)) = (first, last)
            && last > first
        {
            let span = (last - first) as f32;
            return messages
                .iter()
                .enumerate()
                .map(|(index, message)| {
                    let start = message
                        .started_at_ms
                        .map(|started| ((started - first) as f32 / span * 0.95 + 0.01).min(0.96))
                        .unwrap_or_else(|| sequence_position(index, messages.len()));
                    let width = message
                        .duration_ms
                        .map(|duration| (duration as f32 / span * 0.95).clamp(0.006, 0.24))
                        .unwrap_or(0.006)
                        .min(0.99 - start);
                    (start, width)
                })
                .collect();
        }
    }
    (0..messages.len())
        .map(|index| (sequence_position(index, messages.len()), 0.006))
        .collect()
}

fn sequence_position(index: usize, count: usize) -> f32 {
    0.01 + index as f32 / count.saturating_sub(1).max(1) as f32 * 0.95
}

fn summary_pair(label: &'static str, value: &str, colors: TrajectoryPalette) -> impl IntoElement {
    div()
        .flex()
        .items_start()
        .child(
            div()
                .w(px(112.0))
                .flex_none()
                .text_color(colors.label_tertiary)
                .child(label),
        )
        .child(
            div()
                .flex_1()
                .min_w(px(0.0))
                .text_color(colors.label_primary)
                .child(value.to_owned()),
        )
}

fn details_linked_section(
    id: impl Into<gpui::ElementId>,
    label: &'static str,
    content: gpui::AnyElement,
    colors: TrajectoryPalette,
    on_click: impl Fn(&gpui::ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .child(
            div()
                .id(id)
                .flex()
                .items_center()
                .gap_1()
                .font_weight(gpui::FontWeight::MEDIUM)
                .text_color(colors.label_secondary)
                .cursor_pointer()
                .hover(move |heading| heading.text_color(colors.label_primary))
                .on_click(on_click)
                .child(label)
                .child("›"),
        )
        .child(content)
}

fn detail_content_id(prefix: &str, message: &Message) -> SharedString {
    SharedString::from(format!(
        "{prefix}-{}-{}",
        message.key,
        if message.pending { message.revision } else { 0 }
    ))
}

fn details_markdown(
    id: impl Into<gpui::ElementId>,
    value: String,
    language: Option<&str>,
    cx: &mut Context<DesktopApp>,
) -> gpui::AnyElement {
    let colors = trajectory_palette(cx);
    let id = id.into();
    match language {
        Some("json") => details_json_code(id, value, colors),
        Some(_) => details_plain_code(id, value, colors),
        None => TextView::markdown(id, value)
            .text_color(colors.label_primary)
            .into_any_element(),
    }
}

fn details_plain_code(
    id: gpui::ElementId,
    value: String,
    colors: TrajectoryPalette,
) -> gpui::AnyElement {
    div()
        .id(id)
        .w_full()
        .p_2()
        .rounded_md()
        .bg(colors.code_background)
        .font_family("SF Mono")
        .text_sm()
        .text_color(colors.label_primary)
        .child(value)
        .into_any_element()
}

fn details_json_code(
    id: gpui::ElementId,
    value: String,
    colors: TrajectoryPalette,
) -> gpui::AnyElement {
    let highlights = json_token_ranges(&value).into_iter().map(|(range, kind)| {
        let color = match kind {
            JsonTokenKind::Property => colors.json_property,
            JsonTokenKind::String => colors.json_string,
            JsonTokenKind::Keyword => colors.json_keyword,
        };
        (
            range,
            HighlightStyle {
                color: Some(color),
                ..Default::default()
            },
        )
    });
    div()
        .id(id)
        .w_full()
        .p_2()
        .rounded_md()
        .bg(colors.code_background)
        .font_family("SF Mono")
        .text_sm()
        .text_color(colors.json_punctuation)
        .child(StyledText::new(value).with_highlights(highlights))
        .into_any_element()
}

struct ParsedToolSchema {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

fn parse_tool_schema(value: &str) -> Option<ParsedToolSchema> {
    let value = serde_json::from_str::<serde_json::Value>(value).ok()?;
    let object = value.as_object()?;
    Some(ParsedToolSchema {
        name: object.get("name")?.as_str()?.to_owned(),
        description: object
            .get("description")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        parameters: object.get("parameters")?.clone(),
    })
}

fn details_schema(
    id: impl Into<gpui::ElementId>,
    message: &Message,
    colors: TrajectoryPalette,
) -> gpui::AnyElement {
    let Some(schema) = message.schema.as_deref() else {
        return div()
            .text_sm()
            .text_color(colors.label_secondary)
            .child("Schema metadata is unavailable for this record.")
            .into_any_element();
    };
    let Some(schema) = parse_tool_schema(schema) else {
        return details_json_code(id.into(), pretty_json(schema), colors);
    };
    let parameters = serde_json::to_string_pretty(&schema.parameters)
        .unwrap_or_else(|_| schema.parameters.to_string());
    div()
        .flex()
        .flex_col()
        .gap_3()
        .child(
            div()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .text_color(colors.label_primary)
                .child(schema.name),
        )
        .when(!schema.description.is_empty(), |body| {
            body.child(
                div()
                    .text_sm()
                    .text_color(colors.label_secondary)
                    .child(schema.description),
            )
        })
        .child(
            div()
                .font_weight(gpui::FontWeight::MEDIUM)
                .text_color(colors.label_secondary)
                .child("Parameters"),
        )
        .child(details_json_code(id.into(), parameters, colors))
        .into_any_element()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum JsonTokenKind {
    Property,
    String,
    Keyword,
}

fn json_token_ranges(value: &str) -> Vec<(std::ops::Range<usize>, JsonTokenKind)> {
    let bytes = value.as_bytes();
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let start = index;
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index = (index + 2).min(bytes.len()),
                        b'"' => {
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
                let mut cursor = index;
                while cursor < bytes.len() && bytes[cursor].is_ascii_whitespace() {
                    cursor += 1;
                }
                ranges.push((
                    start..index,
                    if bytes.get(cursor) == Some(&b':') {
                        JsonTokenKind::Property
                    } else {
                        JsonTokenKind::String
                    },
                ));
            }
            b'-' | b'0'..=b'9' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && matches!(bytes[index], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    index += 1;
                }
                ranges.push((start..index, JsonTokenKind::Keyword));
            }
            b't' if bytes[index..].starts_with(b"true") => {
                ranges.push((index..index + 4, JsonTokenKind::Keyword));
                index += 4;
            }
            b'n' if bytes[index..].starts_with(b"null") => {
                ranges.push((index..index + 4, JsonTokenKind::Keyword));
                index += 4;
            }
            b'f' if bytes[index..].starts_with(b"false") => {
                ranges.push((index..index + 5, JsonTokenKind::Keyword));
                index += 5;
            }
            _ => index += 1,
        }
    }
    ranges
}

fn result_language(message: &Message) -> Option<&'static str> {
    match message.role {
        Role::Assistant | Role::Reasoning | Role::User | Role::Notice => None,
        Role::Tool => {
            let title = message
                .title
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            if title.contains("shell") || title.contains("bash") || title.contains("terminal") {
                Some("bash")
            } else if serde_json::from_str::<serde_json::Value>(&message.text).is_ok() {
                Some("json")
            } else {
                Some("text")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(role: Role, text: &str) -> Message {
        Message {
            key: crate::domain::next_message_id(),
            revision: 0,
            role,
            tool_call_id: None,
            title: None,
            text: text.into(),
            payload: None,
            schema: None,
            pending: false,
            failed: false,
            expanded: false,
            rating: None,
            started_at_ms: None,
            duration_ms: None,
            turn: 0,
            step: 0,
            request_id: None,
            search_text: text.to_lowercase(),
        }
    }

    #[test]
    fn tool_payload_result_and_turn_position_stay_independent() {
        let user = record(Role::User, "inspect project");
        let mut tool = record(Role::Tool, "exit_code=0\ndone");
        tool.title = Some("shell".into());
        tool.payload = Some(r#"{"command":"cargo test"}"#.into());
        let mut messages = vec![user, record(Role::Assistant, "running checks"), tool];
        crate::domain::reindex_messages(&mut messages);

        assert!(message_matches(&messages[2], "cargo test"));
        assert!(message_summary(&messages[2]).contains("exit_code=0"));
        assert_eq!(detail_location(&messages[2]), "Turn 1 · Step 1");
        assert_eq!(result_text(&messages[2]), "exit_code=0\ndone");
    }

    #[test]
    fn timeline_spreads_records_across_the_available_width() {
        let messages = vec![
            record(Role::User, "first"),
            record(Role::Tool, "middle"),
            record(Role::Assistant, "last"),
        ];
        let geometry = timeline_geometry(&messages, false);
        assert!((geometry[0].0 - 0.01).abs() < f32::EPSILON);
        assert!((geometry[1].0 - 0.485).abs() < 0.000_001);
        assert!((geometry[2].0 - 0.96).abs() < f32::EPSILON);
        assert!(geometry.iter().all(|(_, width)| *width >= 0.006));
    }

    #[test]
    fn json_tokens_use_dsh_property_string_and_keyword_groups() {
        let value = r#"{"command":"pwd","ok":true,"count":2}"#;
        let tokens = json_token_ranges(value)
            .into_iter()
            .map(|(range, kind)| (&value[range], kind))
            .collect::<Vec<_>>();
        assert_eq!(
            tokens,
            vec![
                (r#""command""#, JsonTokenKind::Property),
                (r#""pwd""#, JsonTokenKind::String),
                (r#""ok""#, JsonTokenKind::Property),
                ("true", JsonTokenKind::Keyword),
                (r#""count""#, JsonTokenKind::Property),
                ("2", JsonTokenKind::Keyword),
            ]
        );
    }

    #[test]
    fn detail_tabs_follow_the_selected_cell_kind() {
        assert_eq!(
            detail_tabs(&record(Role::User, "hello")),
            &[DetailsTab::Summary, DetailsTab::Preview, DetailsTab::Raw]
        );
        assert_eq!(
            detail_tabs(&record(Role::Assistant, "answer")),
            &[DetailsTab::Summary, DetailsTab::Preview, DetailsTab::Raw]
        );
        assert_eq!(
            detail_tabs(&record(Role::Reasoning, "thinking")),
            &[DetailsTab::Summary, DetailsTab::Preview, DetailsTab::Raw]
        );
        assert_eq!(
            detail_tabs(&record(Role::Notice, "context")),
            &[DetailsTab::Summary, DetailsTab::Preview, DetailsTab::Raw]
        );

        let mut tool = record(Role::Tool, "exit_code=0");
        tool.payload = Some(r#"{"command":"pwd"}"#.into());
        assert_eq!(
            detail_tabs(&tool),
            &[
                DetailsTab::Summary,
                DetailsTab::Payload,
                DetailsTab::Result,
                DetailsTab::Schema,
                DetailsTab::Timing,
            ]
        );
    }

    #[test]
    fn tool_schema_uses_dsh_name_description_and_parameters_shape() {
        let schema = parse_tool_schema(
            r#"{"name":"shell","description":"Run a command","parameters":{"type":"object"}}"#,
        )
        .expect("valid function schema");
        assert_eq!(schema.name, "shell");
        assert_eq!(schema.description, "Run a command");
        assert_eq!(schema.parameters["type"], "object");
    }

    #[test]
    fn detail_location_uses_message_and_step_groups() {
        let mut user = record(Role::User, "hello");
        user.turn = 2;
        assert_eq!(detail_location(&user), "Turn 2 · Message");

        let mut tool = record(Role::Tool, "done");
        tool.turn = 2;
        tool.step = 3;
        assert_eq!(detail_location(&tool), "Turn 2 · Step 3");
    }

    #[test]
    fn duration_geometry_uses_one_time_domain_and_stays_in_bounds() {
        let mut first = record(Role::Assistant, "first");
        first.started_at_ms = Some(1_000);
        first.duration_ms = Some(100);
        let mut second = record(Role::Tool, "second");
        second.started_at_ms = Some(1_500);
        second.duration_ms = Some(500);
        let geometry = timeline_geometry(&[first, second], true);

        assert!(geometry[1].1 > geometry[0].1);
        assert!(
            geometry
                .iter()
                .all(|(position, width)| *position >= 0.0 && position + width <= 0.99)
        );
    }
}
