use gpui::{
    Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, WindowControlArea, div, prelude::FluentBuilder, px,
    rgba,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::clipboard::Clipboard;
use gpui_component::spinner::Spinner;
use gpui_component::text::TextView;
use gpui_component::{Icon, IconName, Sizable};

use crate::app::DesktopApp;
use crate::application::conversation_view_model;
use crate::domain::{Message, Role, Surface};
use crate::dsh_markdown;
use crate::layout::SidebarMode;
use crate::platform::gpui::MessagePresentation;
use crate::platform::gpui::measured_container;
use crate::ui_theme::{UiPalette, metrics, palette, trajectory_palette};

impl DesktopApp {
    pub(crate) fn conversation_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        let trajectory_colors = trajectory_palette(cx);
        let title_left_padding = if self.core.layout.sidebar == SidebarMode::Rail {
            px(metrics::COLLAPSED_CONTENT_LEADING)
        } else {
            px(20.0)
        };
        div()
            .flex()
            .flex_col()
            .flex_none()
            .border_b_1()
            .border_color(colors.border)
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(40.0))
                    .pl(title_left_padding)
                    .pr_5()
                    .child(
                        div()
                            .flex()
                            .flex_1()
                            .items_center()
                            .min_w(px(0.0))
                            .gap_3()
                            .window_control_area(WindowControlArea::Drag)
                            .child(
                                div()
                                    .max_w(px(460.0))
                                    .truncate()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child(conversation_view_model(&self.core).title.to_owned()),
                            )
                            .children(self.control.is_some().then(|| {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .text_xs()
                                    .text_color(colors.primary)
                                    .child(div().size(px(6.0)).rounded_full().bg(colors.primary))
                                    .child("Running")
                            })),
                    )
                    .child(
                        Button::new("session-log")
                            .icon(IconName::ArrowDown)
                            .label("Session log")
                            .outline()
                            .compact()
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.export_session_log(window, cx)
                            })),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_end()
                    .h(px(34.0))
                    .px_5()
                    .gap_7()
                    .child(tab(
                        "chat-tab",
                        "Chat",
                        self.core.surface == Surface::Chat,
                        trajectory_colors.primary,
                        trajectory_colors.label_tertiary,
                        cx.listener(|this, _, window, cx| this.set_trajectory(false, window, cx)),
                    ))
                    .child(tab(
                        "trajectory-tab",
                        "Trajectory",
                        self.core.surface == Surface::Trajectory,
                        trajectory_colors.primary,
                        trajectory_colors.label_tertiary,
                        cx.listener(|this, _, window, cx| this.set_trajectory(true, window, cx)),
                    )),
            )
    }

    pub(crate) fn conversation_body(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        if self.core.surface == Surface::Trajectory {
            self.trajectory_panel(window, cx).into_any_element()
        } else {
            self.chat_timeline(window, cx).into_any_element()
        }
    }

    fn chat_timeline(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        // TODO(responsive-chat): GPUI can retain a stale scroll extent after window/fullscreen
        // reflow, leaving the final Markdown blocks unreachable behind the composer. Narrow
        // layouts can also produce inconsistent table columns because each row flexes
        // independently. Revisit this with a layout-aware tail anchor and shared table tracks.
        let transcript_owner = cx.entity().downgrade();
        div()
            .relative()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .child(
                div()
                    .id("transcript")
                    .relative()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.0))
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .on_scroll_wheel(cx.listener(
                        |this, event: &gpui::ScrollWheelEvent, window, cx| {
                            this.handle_chat_scroll(event, window, cx)
                        },
                    ))
                    .px(px(self.core.layout.chat_side_padding))
                    .pt_4()
                    .pb(px(self.core.layout.tail_inset))
                    .child(measured_container(
                        transcript_owner,
                        |bounds, this: &mut DesktopApp, cx| {
                            this.observe_transcript_bounds(bounds, cx)
                        },
                        |this: &mut DesktopApp, cx| this.apply_pending_chat_anchor(cx),
                    ))
                    .child(
                        transcript_content_column(self.core.layout.content_max_width)
                            .gap_4()
                            .children(self.core.conversation.messages.iter().enumerate().map(
                                |(index, message)| self.message_view(index, message, window, cx),
                            )),
                    ),
            )
            .children((!self.chat_at_bottom()).then(|| {
                div()
                    .absolute()
                    .left_0()
                    .right_0()
                    .bottom(px(12.0))
                    .flex()
                    .justify_center()
                    .child(
                        Button::new("back-to-bottom")
                            .icon(IconName::ArrowDown)
                            .when(self.core.unread_stream_updates > 0, |button| {
                                button.label(format!("{} new", self.core.unread_stream_updates))
                            })
                            .outline()
                            .compact()
                            .rounded(px(999.0))
                            .bg(colors.surface)
                            .shadow_lg()
                            .tooltip("Back to bottom")
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.scroll_chat_to_bottom(window, cx)
                            })),
                    )
            }))
    }

    fn message_view(
        &self,
        index: usize,
        message: &Message,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = palette(cx);
        let presentation = self.message_presentations.get(message.key);
        let content = match message.role {
            Role::User => div()
                .id(("user-message-row", index))
                .group(SharedString::from(format!("user-message-{index}")))
                .flex()
                .flex_col()
                .items_end()
                .w_full()
                .gap(px(6.0))
                .child(
                    div()
                        .max_w(px(525.0))
                        .px_4()
                        .py(px(10.0))
                        .rounded(px(22.0))
                        .bg(colors.user_bubble)
                        .line_height(px(metrics::BODY_LINE_HEIGHT))
                        .child(presentation.render_text.clone()),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .h(px(28.0))
                        .gap(px(10.0))
                        .child(
                            div()
                                .invisible()
                                .group_hover(
                                    SharedString::from(format!("user-message-{index}")),
                                    |time| time.visible(),
                                )
                                .text_xs()
                                .text_color(colors.muted_text)
                                .child(message_time_label(message)),
                        )
                        .child(
                            Clipboard::new(("copy-user", index))
                                .value(presentation.render_text.clone()),
                        ),
                )
                .into_any_element(),
            Role::Assistant => div()
                .id(("assistant-message-row", index))
                .group(SharedString::from(format!("assistant-message-{index}")))
                .flex()
                .flex_col()
                .w_full()
                .gap_4()
                .text_color(colors.text)
                .line_height(px(metrics::MESSAGE_LINE_HEIGHT))
                .child(assistant_body(
                    message,
                    presentation,
                    self.core.layout.content_max_width,
                    window,
                    cx,
                ))
                .children((!message.pending).then(|| {
                    div()
                        .flex()
                        .items_center()
                        .h(px(28.0))
                        .gap(px(10.0))
                        .child(
                            Clipboard::new(("copy-assistant", index))
                                .value(presentation.render_text.clone()),
                        )
                        .child(
                            Button::new(("good-response", index))
                                .icon(IconName::ThumbsUp)
                                .ghost()
                                .compact()
                                .when(message.rating == Some(true), |button| button.primary())
                                .tooltip("Good response")
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.rate_message(index, true, cx)
                                })),
                        )
                        .child(
                            Button::new(("bad-response", index))
                                .icon(IconName::ThumbsDown)
                                .ghost()
                                .compact()
                                .when(message.rating == Some(false), |button| button.danger())
                                .tooltip("Bad response")
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.rate_message(index, false, cx)
                                })),
                        )
                        .child(
                            div()
                                .invisible()
                                .group_hover(
                                    SharedString::from(format!("assistant-message-{index}")),
                                    |time| time.visible(),
                                )
                                .text_xs()
                                .text_color(colors.muted_text)
                                .child(message_time_label(message)),
                        )
                }))
                .into_any_element(),
            Role::Reasoning => {
                let preview = message
                    .text
                    .lines()
                    .next()
                    .filter(|line| !line.is_empty())
                    .unwrap_or("Thinking…")
                    .to_owned();
                div()
                    .flex()
                    .flex_col()
                    .w_full()
                    .gap(px(6.0))
                    .child(
                        div()
                            .id(("reasoning-row", index))
                            .flex()
                            .items_center()
                            .gap(px(6.0))
                            .h(px(24.0))
                            .line_height(px(24.0))
                            .rounded_md()
                            .cursor_pointer()
                            .tab_index(0)
                            .hover(move |row| row.bg(colors.hover))
                            .on_click(
                                cx.listener(move |this, _, _, cx| this.toggle_reasoning(index, cx)),
                            )
                            .on_key_down(cx.listener(
                                move |this, event: &gpui::KeyDownEvent, _, cx| {
                                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                        this.toggle_reasoning(index, cx);
                                    }
                                },
                            ))
                            .child(
                                Icon::new(if message.expanded {
                                    IconName::ChevronDown
                                } else {
                                    IconName::ChevronRight
                                })
                                .size_4()
                                .text_color(colors.assistant),
                            )
                            .child(div().text_sm().text_color(colors.text).child("Think"))
                            .child(if message.pending && !self.settings.reduce_motion() {
                                Spinner::new()
                                    .small()
                                    .color(colors.primary)
                                    .into_any_element()
                            } else {
                                div()
                                    .size(px(if message.pending { 6.0 } else { 3.0 }))
                                    .rounded_full()
                                    .bg(if message.pending {
                                        colors.primary
                                    } else {
                                        colors.muted_text
                                    })
                                    .into_any_element()
                            })
                            .child(
                                div()
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .truncate()
                                    .text_sm()
                                    .text_color(colors.muted_text)
                                    .child(preview),
                            ),
                    )
                    .when(message.expanded, |row| {
                        row.child(
                            div()
                                .ml(px(22.0))
                                .pl_3()
                                .border_l_1()
                                .border_color(colors.border)
                                .text_sm()
                                .line_height(px(24.0))
                                .text_color(colors.muted_text)
                                .child(presentation.render_text.clone()),
                        )
                    })
                    .into_any_element()
            }
            Role::Tool => self.tool_row(index, message, window, cx),
            Role::Notice => div()
                .flex()
                .items_center()
                .gap_2()
                .text_sm()
                .text_color(colors.muted_text)
                .child(Icon::new(IconName::Info).size_4())
                .child(presentation.render_text.clone())
                .into_any_element(),
        };
        let owner = cx.entity().downgrade();
        let message_id = message.key;
        div()
            .relative()
            .w_full()
            .child(measured_container(
                owner,
                move |bounds, this: &mut DesktopApp, cx| {
                    this.observe_message_bounds(message_id, bounds, cx)
                },
                |this: &mut DesktopApp, cx| this.apply_pending_chat_anchor(cx),
            ))
            .child(content)
            .into_any_element()
    }

    fn tool_row(
        &self,
        index: usize,
        message: &Message,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let colors = palette(cx);
        let title = message.title.as_deref().unwrap_or("Tool");
        let summary = message
            .payload
            .as_deref()
            .and_then(tool_description)
            .or_else(|| message.text.lines().next().map(str::to_owned))
            .unwrap_or_default();
        div()
            .id(("tool-row", index))
            .flex()
            .flex_col()
            .cursor_pointer()
            .tab_index(0)
            .on_click(cx.listener(move |this, _, window, cx| this.toggle_tool(index, window, cx)))
            .on_key_down(
                cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                    if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        this.toggle_tool(index, window, cx);
                    }
                }),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .h(px(24.0))
                    .gap(px(6.0))
                    .line_height(px(24.0))
                    .rounded_md()
                    .hover(move |row| row.bg(colors.hover))
                    .text_sm()
                    .child(
                        Icon::new(if message.expanded {
                            IconName::ChevronDown
                        } else {
                            tool_icon(title)
                        })
                        .size_4()
                        .text_color(if message.failed {
                            colors.danger
                        } else if message.pending {
                            colors.warning
                        } else {
                            colors.muted_text
                        }),
                    )
                    .child(div().text_color(colors.text).child(title.to_owned()))
                    .child(div().size(px(4.0)).rounded_full().bg(if message.failed {
                        colors.danger
                    } else if message.pending {
                        colors.warning
                    } else {
                        colors.muted_text
                    }))
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .truncate()
                            .text_color(if message.failed {
                                colors.danger
                            } else {
                                colors.muted_text
                            })
                            .child(if message.pending {
                                "Running…".to_owned()
                            } else if summary.is_empty() {
                                "Completed".to_owned()
                            } else {
                                summary.to_owned()
                            }),
                    ),
            )
            .when(message.expanded, |element| {
                element.child(
                    div()
                        .ml(px(22.0))
                        .mt_1()
                        .flex()
                        .flex_col()
                        .rounded_xl()
                        .border_1()
                        .border_color(colors.border)
                        .bg(colors.subtle)
                        .overflow_hidden()
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .h(px(34.0))
                                .px_3()
                                .border_b_1()
                                .border_color(colors.border)
                                .text_xs()
                                .text_color(colors.muted_text)
                                .child(if message.pending { "Running" } else { "Done" })
                                .child(
                                    Button::new(("inspect-tool", index))
                                        .icon(IconName::Inspector)
                                        .label("Inspect")
                                        .ghost()
                                        .compact()
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            cx.stop_propagation();
                                            this.inspect_tool(index, window, cx)
                                        })),
                                ),
                        )
                        .children(message.payload.clone().map(|payload| {
                            detail_code_block(
                                SharedString::from(format!("tool-payload-{}", message.key)),
                                "Payload",
                                pretty_json(&payload),
                                "json",
                                colors,
                                window,
                                cx,
                            )
                        }))
                        .child(detail_code_block(
                            SharedString::from(format!(
                                "tool-result-{}-{}",
                                message.key,
                                if message.pending {
                                    "pending"
                                } else {
                                    "settled"
                                }
                            )),
                            "Result",
                            if message.pending {
                                "Waiting for tool result…".into()
                            } else if message.text.is_empty() {
                                "(no output)".into()
                            } else {
                                message.text.clone()
                            },
                            tool_language(title),
                            colors,
                            window,
                            cx,
                        )),
                )
            })
            .into_any_element()
    }
}

fn transcript_content_column(content_max_width: f32) -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .flex_none()
        .w(px(content_max_width))
        .mx_auto()
}

fn assistant_body(
    message: &Message,
    presentation: &MessagePresentation,
    available_width: f32,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> gpui::AnyElement {
    dsh_markdown::render_markdown(
        message.key.0,
        &presentation.markdown,
        message.pending,
        &presentation.render_text,
        available_width,
        window,
        cx,
    )
}

fn tab(
    id: &'static str,
    label: &'static str,
    active: bool,
    active_color: gpui::Hsla,
    inactive_color: gpui::Hsla,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .h_full()
        .px_1()
        .border_b_2()
        .border_color(if active {
            active_color
        } else {
            rgba(0x00000000).into()
        })
        .child(
            Button::new(id)
                .label(label)
                .ghost()
                .compact()
                .text_color(if active { active_color } else { inactive_color })
                .on_click(on_click),
        )
}

fn tool_description(payload: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    value.get("description")?.as_str().map(str::to_owned)
}

fn detail_code_block(
    id: impl Into<gpui::ElementId>,
    label: &'static str,
    value: String,
    language: &'static str,
    colors: UiPalette,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> gpui::AnyElement {
    let fence = if value.contains("```") { "````" } else { "```" };
    let markdown = format!("{fence}{language}\n{value}\n{fence}");
    div()
        .flex()
        .flex_col()
        .gap_1()
        .p_3()
        .border_b_1()
        .border_color(colors.border)
        .child(div().text_xs().text_color(colors.muted_text).child(label))
        .child(TextView::markdown(id, markdown, window, cx))
        .into_any_element()
}

fn pretty_json(value: &str) -> String {
    serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .and_then(|value| serde_json::to_string_pretty(&value).ok())
        .unwrap_or_else(|| value.to_owned())
}

fn tool_language(title: &str) -> &'static str {
    let title = title.to_ascii_lowercase();
    if title.contains("shell") || title.contains("bash") || title.contains("terminal") {
        "bash"
    } else if title.contains("json") {
        "json"
    } else {
        "text"
    }
}

fn message_time_label(message: &Message) -> String {
    if message.pending {
        "Streaming".into()
    } else if let Some(duration) = message.duration_ms {
        format!("{duration} ms")
    } else if message.started_at_ms.is_some() {
        "Just now".into()
    } else {
        "Restored".into()
    }
}

fn tool_icon(title: &str) -> IconName {
    let title = title.to_ascii_lowercase();
    if title.contains("read") || title.contains("edit") || title.contains("file") {
        IconName::File
    } else if title.contains("search") || title.contains("grep") || title.contains("glob") {
        IconName::Search
    } else {
        IconName::SquareTerminal
    }
}

#[cfg(test)]
mod tests {
    use gpui::{
        AppContext, Context, InteractiveElement, IntoElement, ParentElement, Render, ScrollHandle,
        StatefulInteractiveElement, Styled, TestAppContext, Window, div, px, size,
    };

    use super::transcript_content_column;
    use crate::layout::{LayoutInput, resolve_layout};
    use crate::platform::gpui::measured_container;
    use crate::ui_theme::metrics;

    #[gpui::test]
    fn transcript_reserves_scrollable_space_above_the_composer(cx: &mut TestAppContext) {
        let scroll = ScrollHandle::new();

        struct TranscriptHarness(ScrollHandle);

        impl Render for TranscriptHarness {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                let layout = resolve_layout(LayoutInput {
                    viewport_width: 320.0,
                    viewport_height: 400.0,
                    composer_height: 100.0,
                    ..LayoutInput::default()
                });
                div()
                    .flex()
                    .flex_col()
                    .size_full()
                    .child(div().flex_none().h(px(74.0)))
                    .child(
                        div()
                            .relative()
                            .flex()
                            .flex_col()
                            .flex_1()
                            .min_h(px(0.0))
                            .child(
                                div()
                                    .id("transcript-regression")
                                    .flex()
                                    .flex_col()
                                    .flex_1()
                                    .min_h(px(0.0))
                                    .overflow_y_scroll()
                                    .track_scroll(&self.0)
                                    .pb(px(layout.tail_inset))
                                    .child(
                                        transcript_content_column(layout.content_max_width)
                                            .child(div().w_full().h(px(400.0))),
                                    ),
                            ),
                    )
                    .child(div().flex_none().h(px(100.0)))
            }
        }

        let (_, cx) = cx.add_window_view(|_, _| TranscriptHarness(scroll.clone()));
        cx.simulate_resize(size(px(320.0), px(400.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let scrollport_height = 400.0 - 74.0 - 100.0;
        let layout = resolve_layout(LayoutInput {
            viewport_width: 320.0,
            viewport_height: 400.0,
            composer_height: 100.0,
            ..LayoutInput::default()
        });
        let expected = 400.0 + layout.tail_inset - scrollport_height;
        assert!(scroll.max_offset().height >= px(expected - 1.0));
    }

    #[gpui::test]
    fn resolved_reading_width_is_definite_before_markdown_height_measurement(
        cx: &mut TestAppContext,
    ) {
        struct ReadingColumnHarness {
            measured_width: f32,
        }

        impl Render for ReadingColumnHarness {
            fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
                let owner = cx.entity().downgrade();
                div()
                    .size_full()
                    .child(
                        transcript_content_column(748.0)
                            .relative()
                            .child(measured_container(
                                owner,
                                |bounds, harness: &mut ReadingColumnHarness, _| {
                                    let changed =
                                        (harness.measured_width - bounds.width).abs() >= 0.5;
                                    harness.measured_width = bounds.width;
                                    changed
                                },
                                |_: &mut ReadingColumnHarness, _| {},
                            )),
                    )
            }
        }

        let (view, cx) = cx.add_window_view(|_, _| ReadingColumnHarness {
            measured_width: 0.0,
        });
        // The pure layout resolver owns responsive width. A temporarily stale, narrower
        // platform parent must not turn the reading column back into an indefinite percentage,
        // because Markdown would then measure height at the wrong width.
        cx.simulate_resize(size(px(600.0), px(400.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let measured_width = cx.read_entity(&view, |harness, _| harness.measured_width);
        assert!((measured_width - 748.0).abs() < 1.0);
    }

    #[test]
    fn assistant_typography_uses_the_dsh_reading_rhythm() {
        assert_eq!(metrics::MESSAGE_LINE_HEIGHT, 28.0);
    }

    #[test]
    fn assistant_text_column_matches_the_composer_inset() {
        let layout = resolve_layout(LayoutInput::default());
        assert_eq!(
            layout.content_max_width + layout.chat_side_padding * 2.0,
            layout.composer_max_width
        );
    }
}
