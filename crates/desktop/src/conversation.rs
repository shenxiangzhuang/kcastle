use gpui_kit::component::button::{Button, ButtonCustomVariant, ButtonVariants};
use gpui_kit::component::spinner::Spinner;
use gpui_kit::component::{Icon, IconName, Selectable, Sizable};
use gpui_kit::{
    Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, WindowControlArea, accesskit::Role as AxRole, div,
    prelude::FluentBuilder, px, rgba,
};

use crate::app::DesktopApp;
use crate::application::conversation_view_model;
use crate::domain::{Message, Role, Surface};
use crate::dsh_markdown;
use crate::layout::SidebarMode;
use crate::ui_automation::ids;
use crate::ui_theme::{TrajectoryPalette, metrics, palette, trajectory_palette};

#[cfg(test)]
mod performance;

impl DesktopApp {
    pub(crate) fn conversation_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        let trajectory_colors = trajectory_palette(cx);
        let title_left_padding = if self.core.layout.sidebar == SidebarMode::Rail {
            px(metrics::COLLAPSED_CONTENT_LEADING)
        } else {
            px(20.0)
        };
        let show_chat = cx.listener(|this, _, window, cx| this.set_trajectory(false, window, cx));
        let show_trajectory =
            cx.listener(|this, _, window, cx| this.set_trajectory(true, window, cx));
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
                                    .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                                    .child(conversation_view_model(&self.core).title.to_owned()),
                            )
                            .children(self.session_running().then(|| {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .text_xs()
                                    .text_color(colors.primary)
                                    .child(div().size(px(6.0)).rounded_full().bg(colors.primary))
                                    .child("Running")
                            })),
                    ),
            )
            .child(
                div()
                    .id("conversation-tabs")
                    .role(AxRole::TabList)
                    .accessibility_id(ids::CONVERSATION_TABS)
                    .aria_label("Conversation views")
                    .flex()
                    .items_end()
                    .h(px(32.0))
                    .px_5()
                    .gap_3()
                    .child(tab(
                        "chat-tab",
                        ids::CHAT_TAB,
                        "Chat",
                        self.core.surface == Surface::Chat,
                        trajectory_colors,
                        cx,
                        show_chat,
                    ))
                    .child(tab(
                        "trajectory-tab",
                        ids::TRAJECTORY_TAB,
                        "Trajectory",
                        self.core.surface == Surface::Trajectory,
                        trajectory_colors,
                        cx,
                        show_trajectory,
                    )),
            )
    }

    pub(crate) fn conversation_body(
        &self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui_kit::AnyElement {
        if self.core.surface == Surface::Trajectory {
            self.trajectory_panel(window, cx).into_any_element()
        } else {
            self.chat_timeline(window, cx).into_any_element()
        }
    }

    fn chat_timeline(&self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        use gpui_kit::component::{ActiveTheme, scroll::ScrollableElement};
        let colors = palette(cx);
        let state = {
            let mut chat = self.chat.borrow_mut();
            chat.sync(
                &self.core.session_view.conversation.messages,
                &self.core.transient_messages,
                &self.message_presentations.borrow(),
                self.core.session_view.trajectory.projection_lineage(),
                cx.theme().is_dark(),
            );
            chat.begin_frame();
            chat.list.clone()
        };
        div()
            .id("chat-panel")
            .role(AxRole::TabPanel)
            .accessibility_id(ids::CHAT_PANEL)
            .aria_label("Chat")
            .relative()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .overflow_hidden()
            .child(
                div()
                    .id("transcript")
                    .role(AxRole::Log)
                    .accessibility_id(ids::TRANSCRIPT)
                    .aria_label("Conversation transcript")
                    .flex_1()
                    .min_h(px(0.0))
                    .vertical_scrollbar(&state)
                    .child(
                        gpui_kit::list(
                            state,
                            cx.processor(|this, index, window, cx| {
                                this.render_chat_row(index, window, cx)
                            }),
                        )
                        .w_full()
                        .h_full()
                        .pt(px(self.core.layout.transcript_top_inset))
                        .pb(px(self.core.layout.tail_inset)),
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
                        div()
                            .relative()
                            .when(cfg!(test), |element| {
                                element.debug_selector(|| "back-to-bottom".to_owned())
                            })
                            .flex()
                            .child(
                                Button::new("back-to-bottom")
                                    .accessibility_id(ids::BACK_TO_BOTTOM)
                                    .icon(IconName::ArrowDown)
                                    .when(self.core.unread_stream_updates > 0, |button| {
                                        button.label(format!(
                                            "{} new",
                                            self.core.unread_stream_updates
                                        ))
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
                            .child(self.html_previews.button_occlusion()),
                    )
            }))
    }

    fn render_chat_row(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui_kit::AnyElement {
        let row = self.chat.borrow_mut().row(index, window, cx);
        if !self.chat.borrow().demand_scheduled {
            self.chat.borrow_mut().demand_scheduled = true;
            let owner = cx.entity().downgrade();
            cx.defer(move |cx| {
                let _ = owner.update(cx, |this, cx| this.finish_chat_frame(cx));
            });
        }
        let Some((row, selection, prepared)) = row else {
            return div().into_any_element();
        };
        let colors = palette(cx);
        let body = if let Some(selection) = selection {
            // Retained browsers bridge cache eviction, not rejection of oversized source.
            let html = if row.message.role == Role::Assistant
                && row.preparation_range().is_some_and(|range| {
                    range.len() <= crate::platform::gpui::MAX_CODE_SOURCE_BYTES
                }) {
                prepared
                    .as_ref()
                    .and_then(|prepared| prepared.html_document())
                    .map(std::borrow::Cow::Borrowed)
                    .or_else(|| {
                        row.code_visible().and_then(|_| {
                            self.html_previews
                                .retained_source(row.key)
                                .map(std::borrow::Cow::Owned)
                        })
                    })
            } else {
                None
            };
            #[cfg(test)]
            let plain_selector = (prepared.is_none() && html.is_none())
                .then(|| format!("chat-plain:{}", row.message.key.0));
            let content = if let Some(html) = html {
                self.html_previews
                    .render(row.key, &html, row.plain(), &selection, cx)
            } else if let Some(prepared) = prepared {
                dsh_markdown::render_prepared_markdown(
                    row.message.key.0,
                    &prepared,
                    row.code_visible(),
                    self.core.layout.content_max_width,
                    &selection,
                    window,
                    cx,
                )
            } else {
                dsh_markdown::plain_text(row.plain().to_owned().into(), Some(&selection))
                    .into_any_element()
            };
            let content = selection.wrap(content);
            div()
                .map(|body| {
                    #[cfg(test)]
                    let body = body.when_some(plain_selector, |body, selector| {
                        body.debug_selector(move || selector)
                    });
                    body
                })
                .w_full()
                .min_h(px(24.0))
                .text_color(colors.text)
                .line_height(px(metrics::MESSAGE_LINE_HEIGHT))
                .when(row.message.role == Role::Assistant, |body| {
                    body.text_size(px(16.0))
                })
                .pt(px(row
                    .chunk
                    .as_ref()
                    .and_then(|chunk| chunk.gap_before)
                    .unwrap_or(0) as f32))
                .when(row.message.role != Role::Assistant, |body| body.pb(px(4.0)))
                .when(row.message.role == Role::User, |body| {
                    body.flex().justify_end()
                })
                .child(
                    div()
                        .when(row.message.role != Role::User, |body| body.w_full())
                        .when(row.message.role == Role::User, |body| {
                            body.max_w(px(525.0))
                                .px_4()
                                .py(px(10.0))
                                .rounded(px(22.0))
                                .line_height(px(metrics::BODY_LINE_HEIGHT))
                                .bg(colors.user_bubble)
                        })
                        .when(row.message.role == Role::Reasoning, |body| {
                            body.ml(px(22.0))
                                .pl_3()
                                .border_l_1()
                                .border_color(colors.border)
                                .text_sm()
                                .text_color(colors.muted_text)
                        })
                        .child(content),
                )
                .into_any_element()
        } else {
            self.message_view(row.message_index, &row.message, window, cx)
        };
        div()
            .id(gpui_kit::SharedString::from(format!(
                "chat-row-{}-{}-{}",
                row.key.message, row.key.field, row.key.start
            )))
            .w_full()
            .child(transcript_content_column(self.core.layout.content_max_width).child(body))
            .into_any_element()
    }

    #[allow(
        clippy::unreachable,
        reason = "notice messages return before presentation rendering"
    )]
    fn message_view(
        &self,
        index: usize,
        message: &Message,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> gpui_kit::AnyElement {
        let colors = palette(cx);
        let content = if message.role == Role::Notice {
            div()
                .flex()
                .items_center()
                .gap_2()
                .text_sm()
                .text_color(colors.muted_text)
                .child(Icon::new(IconName::Info).size_4())
                .child(message.text.clone())
                .into_any_element()
        } else {
            let mut presentations = self.message_presentations.borrow_mut();
            let presentation = presentations.sync_message(message.key);
            match message.role {
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
                            .child(copy_message_button("copy-user", index, message, cx)),
                    )
                    .into_any_element(),
                Role::Assistant => div()
                    .id(("assistant-message-row", index))
                    .group(SharedString::from(format!("assistant-message-{index}")))
                    .flex()
                    .flex_col()
                    .w_full()
                    .gap(px(metrics::ASSISTANT_ACTIONS_TOP_GAP))
                    .text_color(colors.text)
                    .line_height(px(metrics::MESSAGE_LINE_HEIGHT))
                    .children((!message.pending).then(|| {
                        div()
                            .flex()
                            .items_center()
                            .h(px(28.0))
                            .gap(px(10.0))
                            .child(copy_message_button("copy-assistant", index, message, cx))
                            .child(
                                Button::new(("good-response", index))
                                    .icon(IconName::ThumbsUp)
                                    .ghost()
                                    .compact()
                                    .when(presentation.rating() == Some(true), |button| {
                                        button.primary()
                                    })
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
                                    .when(presentation.rating() == Some(false), |button| {
                                        button.danger()
                                    })
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
                    let expanded = presentation.expanded();
                    let preview = reasoning_preview(&message.text, message.pending);
                    let follow_summary_end = message.pending && !expanded;
                    presentation.align_reasoning_summary(
                        follow_summary_end,
                        message.revision,
                        window,
                    );
                    let reasoning_summary_scroll = presentation.reasoning_summary_scroll();
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
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    this.toggle_reasoning(index, cx)
                                }))
                                .on_key_down(cx.listener(
                                    move |this, event: &gpui_kit::KeyDownEvent, _, cx| {
                                        if matches!(event.keystroke.key.as_str(), "enter" | "space")
                                        {
                                            this.toggle_reasoning(index, cx);
                                        }
                                    },
                                ))
                                .child(
                                    Icon::new(if expanded {
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
                                .child(if expanded {
                                    div().flex_1().into_any_element()
                                } else if message.pending {
                                    div()
                                        .id(("reasoning-summary", index))
                                        .flex()
                                        .flex_1()
                                        .min_w(px(0.0))
                                        .overflow_x_scroll()
                                        .track_scroll(&reasoning_summary_scroll)
                                        .text_sm()
                                        .text_color(colors.muted_text)
                                        .child(div().flex_none().whitespace_nowrap().child(preview))
                                        .into_any_element()
                                } else {
                                    div()
                                        .flex_1()
                                        .min_w(px(0.0))
                                        .truncate()
                                        .text_sm()
                                        .text_color(colors.muted_text)
                                        .child(preview)
                                        .into_any_element()
                                }),
                        )
                        .into_any_element()
                }
                Role::Tool => self.tool_row(index, message, presentation, cx),
                Role::Notice => unreachable!("notice rendering is handled without presentation"),
            }
        };
        div()
            .w_full()
            .pb(px(12.0))
            .child(content)
            .into_any_element()
    }

    fn tool_row(
        &self,
        index: usize,
        message: &Message,
        presentation: &crate::platform::gpui::MessagePresentation,
        cx: &mut Context<Self>,
    ) -> gpui_kit::AnyElement {
        let colors = palette(cx);
        let title = message.title.as_deref().unwrap_or("Tool");
        let summary = message
            .payload
            .as_deref()
            .and_then(tool_description)
            .or_else(|| Some(reasoning_preview(&message.text, false)))
            .unwrap_or_default();
        div()
            .id(("tool-row", index))
            .flex()
            .flex_col()
            .cursor_pointer()
            .tab_index(0)
            .on_click(cx.listener(move |this, _, window, cx| this.toggle_tool(index, window, cx)))
            .on_key_down(
                cx.listener(move |this, event: &gpui_kit::KeyDownEvent, window, cx| {
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
                        Icon::new(if presentation.expanded() {
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
            .when(presentation.expanded(), |element| {
                element.child(
                    Button::new(("inspect-tool", index))
                        .icon(IconName::Inspector)
                        .label("Inspect")
                        .ghost()
                        .compact()
                        .on_click(cx.listener(move |this, _, window, cx| {
                            cx.stop_propagation();
                            this.inspect_tool(index, window, cx);
                        })),
                )
            })
            .into_any_element()
    }
}

fn transcript_content_column(content_max_width: f32) -> gpui_kit::Div {
    div()
        .flex()
        .flex_col()
        .flex_none()
        .w(px(content_max_width))
        .mx_auto()
}

fn copy_message_button(
    id: &'static str,
    index: usize,
    message: &Message,
    cx: &Context<DesktopApp>,
) -> Button {
    let key = message.key;
    Button::new((id, index))
        .icon(IconName::Copy)
        .ghost()
        .compact()
        .tooltip("Copy message")
        .on_click(cx.listener(move |this, _, _, cx| {
            if let Some(message) = this
                .core
                .session_view
                .conversation
                .messages
                .iter()
                .chain(this.core.transient_messages.iter())
                .find(|message| message.key == key)
            {
                cx.write_to_clipboard(gpui_kit::ClipboardItem::new_string(message.text.clone()));
            }
        }))
}

fn tab(
    id: &'static str,
    automation_id: &'static str,
    label: &'static str,
    active: bool,
    colors: TrajectoryPalette,
    cx: &gpui_kit::App,
    on_click: impl Fn(&gpui_kit::ClickEvent, &mut Window, &mut gpui_kit::App) + 'static,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .h_full()
        .px_1()
        .border_b_2()
        .border_color(if active {
            colors.primary
        } else {
            rgba(0x00000000).into()
        })
        .child(conversation_tab_button(
            id,
            automation_id,
            label,
            active,
            colors,
            cx,
            on_click,
        ))
}

fn conversation_tab_button(
    id: &'static str,
    automation_id: &'static str,
    label: &'static str,
    active: bool,
    colors: TrajectoryPalette,
    cx: &gpui_kit::App,
    on_click: impl Fn(&gpui_kit::ClickEvent, &mut Window, &mut gpui_kit::App) + 'static,
) -> Button {
    Button::new(id)
        .role(AxRole::Tab)
        .accessibility_id(automation_id)
        .selected(active)
        .label(label)
        .custom(
            ButtonCustomVariant::new(cx)
                .foreground(if active {
                    colors.label_primary
                } else {
                    colors.label_secondary
                })
                .hover(colors.hover),
        )
        .compact()
        .on_click(on_click)
}

fn tool_description(payload: &str) -> Option<String> {
    // Header work is bounded; the complete payload remains available in the expanded rows.
    if payload.len() > 16 * 1024 {
        return None;
    }
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    value.get("description")?.as_str().map(str::to_owned)
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

fn reasoning_preview(text: &str, running: bool) -> String {
    let line = if running {
        let visible = text.trim_end();
        visible
            .rsplit_once('\n')
            .map_or(visible, |(_, latest)| latest)
    } else {
        text.split_once('\n').map_or(text, |(first, _)| first)
    };
    if line.is_empty() {
        "Thinking…".into()
    } else {
        line[..line.floor_char_boundary(line.len().min(512))].to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use gpui_kit::{
        AppContext, Context, InteractiveElement, IntoElement, ParentElement, Render, ScrollHandle,
        StatefulInteractiveElement, Styled, TestAppContext, Window, div, px, size,
    };

    use super::{reasoning_preview, transcript_content_column};
    use crate::app::DesktopApp;
    use crate::domain::{Message, MessageId, Role};
    use crate::layout::{LayoutInput, resolve_layout};
    use crate::platform::gpui::measured_container;
    use crate::ui_theme::metrics;

    #[gpui_kit::test]
    fn transcript_scroll_range_covers_its_content(cx: &mut TestAppContext) {
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
        assert!(scroll.max_offset().y >= px(expected - 1.0));
    }

    #[gpui_kit::test]
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
                                |_: &mut ReadingColumnHarness, _, _| {},
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
        assert_eq!(metrics::MESSAGE_LINE_HEIGHT, 26.0);
    }

    #[gpui_kit::test]
    fn chat_only_prepares_the_viewport(cx: &mut TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-chat-viewport-{}",
            kcastle_agent::SessionId::new()
        ));
        let (startup, _) = crate::desktop_startup(root.clone()).unwrap();
        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let mut app = DesktopApp::new(startup, window, cx);
            for id in 0..1000 {
                app.core.transient_messages.push_back(Arc::new(Message {
                    key: MessageId(10000 + id),
                    revision: 0,
                    role: Role::Assistant,
                    text: format!("Message {id}\n\n{}", "A readable paragraph. ".repeat(40)),
                    tool_call_id: None,
                    title: None,
                    payload: None,
                    schema: None,
                    pending: false,
                    failed: false,
                    started_at_ms: None,
                    duration_ms: None,
                    turn: 0,
                    step: 0,
                    request_id: None,
                }));
            }
            window.blur(cx);
            app
        });
        cx.simulate_resize(size(px(1180.0), px(720.0)));
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert!(
                app.chat.borrow().prepared_chunks() > 0,
                "visible content must progress beyond plain text"
            );
            assert!(
                app.message_presentations.borrow().retained_messages() < 30
                    && app.chat.borrow().retained_chunks() < 30,
                "offscreen messages must not allocate presentations or parse Markdown"
            );
        });
        let viewport = view.read_with(cx, |app, _| app.chat.borrow().list.viewport_bounds());
        for (delta, follows) in [(200.0, false), (-100000.0, true)] {
            cx.simulate_event(gpui_kit::ScrollWheelEvent {
                position: viewport.center(),
                delta: gpui_kit::ScrollDelta::Pixels(gpui_kit::point(px(0.0), px(delta))),
                ..Default::default()
            });
            cx.run_until_parked();
            view.read_with(cx, |app, _| assert_eq!(app.core.follow_chat_tail, follows));
        }
        view.update(cx, |app, cx| {
            let chat = app.chat.borrow();
            chat.list.set_follow_mode(gpui_kit::FollowMode::Normal);
            chat.list.scroll_to(gpui_kit::ListOffset {
                item_ix: 800,
                offset_in_item: px(10.0),
            });
            cx.notify();
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert!(app.chat.borrow().retained_chunks() < 30);
            assert!(
                !app.chat.borrow().selection_initialized(MessageId(10999)),
                "old viewport must be evicted"
            );
            assert!(app.chat.borrow().prepared_chunks() > 0);
        });
        let anchor = view.read_with(cx, |app, _| app.chat.borrow().anchor());
        cx.simulate_resize(size(px(900.0), px(720.0)));
        cx.run_until_parked();
        view.read_with(cx, |app, _| match (anchor, app.chat.borrow().anchor()) {
            (
                crate::layout::ScrollAnchor::Block {
                    id: a,
                    field: af,
                    source_offset: ao,
                    ..
                },
                crate::layout::ScrollAnchor::Block {
                    id: b,
                    field: bf,
                    source_offset: bo,
                    ..
                },
            ) => assert_eq!((a, af, ao), (b, bf, bo)),
            other => panic!("resize lost source anchor: {other:?}"),
        });
        let starts_before_huge = view.read_with(cx, |app, _| {
            app.chat
                .borrow()
                .worker_starts
                .load(std::sync::atomic::Ordering::Relaxed)
        });
        view.update(cx, |app, cx| {
            let mut huge = (**app.core.transient_messages.front().unwrap()).clone();
            huge.text = "A **long** message paragraph.\n\n".repeat(20000);
            huge.revision += 1;
            app.core.transient_messages = im::Vector::unit(Arc::new(huge));
            app.chat
                .borrow()
                .list
                .set_follow_mode(gpui_kit::FollowMode::Tail);
            cx.notify();
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert!(app.chat.borrow().rows.len() > 1000);
            assert!(
                app.chat.borrow().retained_chunks() < 60,
                "one huge message must also be virtualized"
            );
            assert!(
                app.chat
                    .borrow()
                    .worker_starts
                    .load(std::sync::atomic::Ordering::Relaxed)
                    - starts_before_huge
                    < 60,
                "one huge message must not prepare every chunk and then evict the results"
            );
        });
        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn assistant_text_column_matches_the_composer_inset() {
        let layout = resolve_layout(LayoutInput::default());
        assert_eq!(
            layout.content_max_width + layout.chat_side_padding * 2.0,
            layout.composer_max_width
        );
    }

    #[test]
    fn running_reasoning_previews_the_latest_non_blank_line() {
        assert_eq!(
            reasoning_preview("Inspect the session\nNewest reasoning tokens\n", true),
            "Newest reasoning tokens"
        );
        assert_eq!(
            reasoning_preview("Inspect the session\nNewest reasoning tokens", false),
            "Inspect the session"
        );
    }

    #[gpui_kit::test]
    fn non_selectable_messages_do_not_allocate_selection_state(cx: &mut TestAppContext) {
        fn message(id: u64, role: Role) -> Arc<Message> {
            Arc::new(Message {
                key: MessageId(id),
                revision: 0,
                role,
                tool_call_id: None,
                title: None,
                text: "content".into(),
                payload: None,
                schema: None,
                pending: false,
                failed: false,
                started_at_ms: None,
                duration_ms: None,
                turn: 0,
                step: 0,
                request_id: None,
            })
        }

        let root = std::env::temp_dir().join(format!(
            "kcastle-selection-allocation-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (startup, _) = crate::desktop_startup(root.clone()).unwrap();
        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let mut app = DesktopApp::new(startup, window, cx);
            app.core
                .transient_messages
                .push_back(message(901, Role::User));
            app.core
                .transient_messages
                .push_back(message(902, Role::Reasoning));
            app.core
                .transient_messages
                .push_back(message(903, Role::Tool));
            window.blur(cx);
            app
        });
        cx.simulate_resize(size(px(900.0), px(700.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        view.read_with(cx, |app, _| {
            let chat = app.chat.borrow();
            assert!(chat.selection_initialized(MessageId(901)));
            assert!(!chat.selection_initialized(MessageId(902)));
            assert!(!chat.selection_initialized(MessageId(903)));
        });

        view.update(cx, |app, cx| {
            assert_eq!(
                app.message_presentations
                    .get_mut()
                    .toggle_expanded(MessageId(902)),
                Some(true)
            );
            cx.notify();
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert!(app.chat.borrow().selection_initialized(MessageId(902)));
        });

        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        std::fs::remove_dir_all(root).unwrap();
    }
}
