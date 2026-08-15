use std::time::Duration;

use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement,
    StatefulInteractiveElement, Styled, div, prelude::FluentBuilder, px, relative,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::Input;
use gpui_component::scroll::ScrollableElement;
use gpui_component::{Disableable, Icon, IconName, Sizable};

use crate::app::{ComposerMenu, DesktopApp, step_count};
use crate::ui_theme::{UiPalette, metrics, palette};

impl DesktopApp {
    pub(crate) fn empty_conversation(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        let workspace = self
            .project_store
            .project(self.active_project)
            .map(|project| project.name.clone())
            .unwrap_or_else(|| "Choose workspace".into());
        div()
            .relative()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .items_center()
            .justify_center()
            .px_4()
            .pb(px(34.0))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .w_full()
                    .max_w(px(metrics::COMPOSER_WIDTH))
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_center()
                            .gap_2()
                            .child(Icon::new(IconName::Bot).size_6())
                            .child(
                                div()
                                    .text_xl()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Into the Unknown"),
                            )
                            .child(
                                div()
                                    .px_2()
                                    .py(px(2.0))
                                    .rounded_full()
                                    .border_1()
                                    .border_color(colors.border)
                                    .bg(colors.subtle)
                                    .text_xs()
                                    .text_color(colors.muted_text)
                                    .child("Preview"),
                            ),
                    )
                    .child(
                        div().flex().items_center().gap_4().child(
                            Button::new("hero-workspace")
                                .icon(IconName::FolderOpen)
                                .label(workspace)
                                .ghost()
                                .compact()
                                .tooltip("Choose workspace")
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.open_composer_menu(ComposerMenu::Workspace, window, cx)
                                })),
                        ),
                    )
                    .child(self.composer_card(true, cx)),
            )
    }

    pub(crate) fn docked_composer(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        div()
            .flex()
            .flex_col()
            .flex_none()
            .items_center()
            .px_4()
            .pb_2()
            .gap_2()
            .child(self.composer_card(false, cx))
            .child(
                div()
                    .w_full()
                    .max_w(px(metrics::COMPOSER_WIDTH))
                    .text_center()
                    .text_xs()
                    .truncate()
                    .text_color(colors.muted_text)
                    .child(format!(
                        "{} turns · {} steps   |   input {} · cached {} · output {} tokens",
                        self.turns,
                        step_count(&self.messages),
                        compact_number(self.input_tokens),
                        compact_number(self.cached_tokens),
                        compact_number(self.output_tokens)
                    )),
            )
    }

    fn composer_card(&self, hero: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        let running = self.control.is_some();
        let preparing = self.preparing_session;
        let empty = self.input.read(cx).value().trim().is_empty();
        let elapsed = self
            .started_at
            .map(|started_at| started_at.elapsed())
            .unwrap_or_default();
        let model = self.models[self.selected_model]
            .model
            .reasoning_effort()
            .map(|effort| format!("{}  {}", self.model, effort_label(effort)))
            .unwrap_or_else(|| self.model.clone());
        div()
            .flex()
            .flex_col()
            .w_full()
            .max_w(px(metrics::COMPOSER_WIDTH))
            .rounded(px(metrics::COMPOSER_RADIUS))
            .border_1()
            .border_color(colors.border)
            .bg(colors.surface)
            .shadow_lg()
            .child(
                div()
                    .capture_key_down(cx.listener(|this, event, window, cx| {
                        this.handle_root_key(event, window, cx)
                    }))
                    .child(
                        Input::new(&self.input)
                            .appearance(false)
                            .bordered(false)
                            .large(),
                    ),
            )
            .children(self.composer_menu_view(cx))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .px_2()
                    .pb_2()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .child(
                                Button::new(if hero { "hero-commands" } else { "commands" })
                                    .icon(IconName::Plus)
                                    .ghost()
                                    .compact()
                                    .tooltip("Commands")
                                    .on_key_down(cx.listener(|this, event, window, cx| {
                                        this.handle_root_key(event, window, cx)
                                    }))
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.open_composer_menu(ComposerMenu::Commands, window, cx)
                                    })),
                            )
                            .child(
                                Button::new(if hero {
                                    "hero-access-settings"
                                } else {
                                    "access-settings"
                                })
                                .icon(if self.settings.allow_all_tools() {
                                    IconName::CircleCheck
                                } else {
                                    IconName::TriangleAlert
                                })
                                .label(if self.settings.allow_all_tools() {
                                    "Allow all tools"
                                } else {
                                    "Ask before tools"
                                })
                                .ghost()
                                .compact()
                                .tooltip("Select tool approval behavior")
                                .on_key_down(cx.listener(|this, event, window, cx| {
                                    this.handle_root_key(event, window, cx)
                                }))
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.open_composer_menu(
                                            ComposerMenu::Permission,
                                            window,
                                            cx,
                                        )
                                    },
                                )),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .min_w(px(0.0))
                            .gap_2()
                            .child(
                                Button::new(if hero {
                                    "hero-model-settings"
                                } else {
                                    "model-settings"
                                })
                                .label(model)
                                .ghost()
                                .compact()
                                .tooltip("Select model and reasoning effort")
                                .disabled(running)
                                .on_key_down(cx.listener(|this, event, window, cx| {
                                    this.handle_root_key(event, window, cx)
                                }))
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        this.open_composer_menu(ComposerMenu::Model, window, cx)
                                    },
                                )),
                            )
                            .children(running.then(|| {
                                div()
                                    .text_xs()
                                    .text_color(colors.muted_text)
                                    .child(format_duration(elapsed))
                            }))
                            .child(if running {
                                Button::new("stop")
                                    .icon(IconName::Close)
                                    .rounded(px(999.0))
                                    .tooltip("Stop")
                                    .on_click(cx.listener(|this, _, _, cx| this.abort(cx)))
                                    .into_any_element()
                            } else {
                                Button::new("send")
                                    .icon(IconName::ArrowUp)
                                    .primary()
                                    .loading(preparing)
                                    .disabled(empty || preparing)
                                    .rounded(px(999.0))
                                    .tooltip("Send message")
                                    .on_click(
                                        cx.listener(|this, _, window, cx| this.submit(window, cx)),
                                    )
                                    .into_any_element()
                            }),
                    ),
            )
    }

    pub(crate) fn composer_menu_view(&self, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        let colors = palette(cx);
        let menu = self.composer_menu?;
        let body = match menu {
            ComposerMenu::Commands => div()
                .flex()
                .flex_col()
                .child(menu_title("Commands", cx))
                .child(menu_item(
                    "command-export",
                    IconName::ArrowDown,
                    "Export session",
                    "Save the current JSONL session log",
                    self.composer_menu_highlight == 0,
                    colors,
                    cx.listener(|this, _, window, cx| {
                        this.composer_menu = None;
                        this.export_session_log(window, cx)
                    }),
                ))
                .child(menu_item(
                    "command-permission",
                    IconName::CircleCheck,
                    "Permission",
                    "Choose how tool calls are approved",
                    self.composer_menu_highlight == 1,
                    colors,
                    cx.listener(|this, _, _, cx| {
                        this.set_composer_menu(Some(ComposerMenu::Permission), cx)
                    }),
                ))
                .child(menu_item(
                    "command-model",
                    IconName::Bot,
                    "Model",
                    "Choose model reasoning effort",
                    self.composer_menu_highlight == 2,
                    colors,
                    cx.listener(|this, _, _, cx| {
                        this.set_composer_menu(Some(ComposerMenu::Model), cx)
                    }),
                ))
                .into_any_element(),
            ComposerMenu::Permission => div()
                .flex()
                .flex_col()
                .child(menu_title("Tool permission", cx))
                .child(menu_choice(
                    "permission-ask",
                    "Ask before tools",
                    "Show an approval card before every shell call",
                    !self.settings.allow_all_tools(),
                    self.composer_menu_highlight == 0,
                    colors,
                    cx.listener(|this, _, _, cx| this.set_allow_all_tools(false, cx)),
                ))
                .child(menu_choice(
                    "permission-allow",
                    "Allow all tools",
                    "Automatically approve tool calls in this app",
                    self.settings.allow_all_tools(),
                    self.composer_menu_highlight == 1,
                    colors,
                    cx.listener(|this, _, _, cx| this.set_allow_all_tools(true, cx)),
                ))
                .into_any_element(),
            ComposerMenu::Model => {
                let selected_model = self.models[self.selected_model].label();
                let effort = self.models[self.selected_model]
                    .model
                    .reasoning_effort()
                    .map(effort_label)
                    .unwrap_or("Default");
                div()
                    .flex()
                    .flex_col()
                    .child(menu_title("Model and effort", cx))
                    .child(menu_choice(
                        "model-root-model",
                        &selected_model,
                        "Model",
                        false,
                        self.composer_menu_highlight == 0,
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.set_composer_menu(Some(ComposerMenu::Models), cx)
                        }),
                    ))
                    .child(menu_choice(
                        "model-root-effort",
                        effort,
                        "Reasoning effort",
                        false,
                        self.composer_menu_highlight == 1,
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.set_composer_menu(Some(ComposerMenu::Effort), cx)
                        }),
                    ))
                    .into_any_element()
            }
            ComposerMenu::Models => {
                let models = self
                    .models
                    .iter()
                    .enumerate()
                    .map(|(index, model)| (index, model.label(), index == self.selected_model))
                    .collect::<Vec<_>>();
                div()
                    .flex()
                    .flex_col()
                    .child(menu_title("Select model", cx))
                    .children(models.into_iter().map(|(index, label, selected)| {
                        menu_choice(
                            ("composer-model", index),
                            &label,
                            "Use for future responses",
                            selected,
                            self.composer_menu_highlight == index,
                            colors,
                            cx.listener(move |this, _, _, cx| this.select_model(index, cx)),
                        )
                    }))
                    .into_any_element()
            }
            ComposerMenu::Effort => {
                let efforts = self.models[self.selected_model]
                    .model
                    .reasoning_efforts()
                    .to_vec();
                let selected = self.models[self.selected_model]
                    .model
                    .reasoning_effort()
                    .cloned();
                div()
                    .flex()
                    .flex_col()
                    .child(menu_title("Reasoning effort", cx))
                    .children(efforts.into_iter().enumerate().map(|(index, effort)| {
                        let is_selected = selected.as_ref() == Some(&effort);
                        let label = effort_label(&effort).to_owned();
                        menu_choice(
                            ("composer-effort", index),
                            &label,
                            "Use for future responses",
                            is_selected,
                            self.composer_menu_highlight == index,
                            colors,
                            cx.listener(move |this, _, _, cx| {
                                this.set_reasoning_effort(effort.clone(), cx);
                                this.composer_menu = None;
                            }),
                        )
                    }))
                    .into_any_element()
            }
            ComposerMenu::Workspace => {
                let projects = self
                    .project_store
                    .projects()
                    .iter()
                    .enumerate()
                    .map(|(index, project)| {
                        (index, project.name.clone(), index == self.active_project)
                    })
                    .collect::<Vec<_>>();
                let mut project_items = Vec::new();
                for (index, name, selected) in projects {
                    project_items.push(menu_choice(
                        ("composer-workspace", index),
                        &name,
                        "Switch the active working directory",
                        selected,
                        self.composer_menu_highlight == index,
                        colors,
                        cx.listener(move |this, _, window, cx| {
                            this.composer_menu = None;
                            this.switch_project(index, window, cx)
                        }),
                    ));
                }
                div()
                    .flex()
                    .flex_col()
                    .child(menu_title("Workspace", cx))
                    .children(project_items)
                    .child(menu_item(
                        "workspace-add",
                        IconName::Plus,
                        "Add workspace…",
                        "Choose another folder",
                        self.composer_menu_highlight == self.project_store.projects().len(),
                        colors,
                        cx.listener(|this, _, window, cx| {
                            this.composer_menu = None;
                            this.add_project(window, cx)
                        }),
                    ))
                    .into_any_element()
            }
        };
        Some(
            div()
                .absolute()
                .track_focus(&self.composer_menu_focus)
                .capture_key_down(
                    cx.listener(|this, event, window, cx| this.handle_root_key(event, window, cx)),
                )
                .left_0()
                .right_0()
                .bottom(relative(1.0))
                .mb_2()
                .flex()
                .occlude()
                .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
                .child(
                    div()
                        .flex()
                        .w_full()
                        .when(
                            matches!(
                                menu,
                                ComposerMenu::Model | ComposerMenu::Models | ComposerMenu::Effort
                            ),
                            |row| row.justify_end(),
                        )
                        .child(
                            div()
                                .w(px(240.0))
                                .max_h(px(360.0))
                                .rounded_xl()
                                .border_1()
                                .border_color(colors.border)
                                .bg(colors.surface)
                                .shadow_xl()
                                .overflow_y_scrollbar()
                                .child(body),
                        ),
                )
                .into_any_element(),
        )
    }

    pub(crate) fn approval_card(&self, cx: &mut Context<Self>) -> Option<gpui::AnyElement> {
        let colors = palette(cx);
        self.approval.as_ref().map(|approval| {
            let allow_id = approval.call_id.clone();
            let deny_id = approval.call_id.clone();
            div()
                .flex()
                .justify_center()
                .px_4()
                .pb_2()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_4()
                        .w_full()
                        .max_w(px(metrics::COMPOSER_WIDTH))
                        .p_4()
                        .rounded_xl()
                        .border_1()
                        .border_color(colors.border)
                        .bg(colors.surface)
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .min_w(px(0.0))
                                .gap_1()
                                .child(
                                    div()
                                        .font_weight(gpui::FontWeight::SEMIBOLD)
                                        .child(format!("Allow {}?", approval.name)),
                                )
                                .child(
                                    div()
                                        .truncate()
                                        .font_family("SF Mono")
                                        .text_xs()
                                        .text_color(colors.muted_text)
                                        .child(approval.arguments.clone()),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_none()
                                .gap_2()
                                .child(Button::new("deny-tool").label("Deny").on_click(
                                    cx.listener(move |this, _, _, cx| {
                                        this.decide(deny_id.clone(), false, cx)
                                    }),
                                ))
                                .child(
                                    Button::new("allow-tool").label("Allow").primary().on_click(
                                        cx.listener(move |this, _, _, cx| {
                                            this.decide(allow_id.clone(), true, cx)
                                        }),
                                    ),
                                ),
                        ),
                )
                .into_any_element()
        })
    }
}

fn menu_title(title: &'static str, cx: &mut Context<DesktopApp>) -> impl IntoElement {
    let colors = palette(cx);
    div()
        .flex()
        .items_center()
        .justify_between()
        .h(px(metrics::TAB_HEIGHT))
        .px_3()
        .border_b_1()
        .border_color(colors.border)
        .font_weight(gpui::FontWeight::SEMIBOLD)
        .child(title)
        .child(
            Button::new("close-composer-menu")
                .icon(IconName::Close)
                .ghost()
                .compact()
                .on_click(cx.listener(|this, _, _, cx| this.set_composer_menu(None, cx))),
        )
}

fn menu_item(
    id: impl Into<gpui::ElementId>,
    icon: IconName,
    title: &'static str,
    description: &'static str,
    highlighted: bool,
    colors: UiPalette,
    on_click: impl Fn(&gpui::ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .flex()
        .items_center()
        .gap_3()
        .min_h(px(metrics::DETAILS_HEADER_HEIGHT))
        .px_3()
        .cursor_pointer()
        .when(highlighted, |item| item.bg(colors.selected))
        .hover(move |item| item.bg(colors.hover))
        .on_click(on_click)
        .child(Icon::new(icon).size_4().text_color(colors.muted_text))
        .child(
            div()
                .flex()
                .flex_col()
                .gap(px(2.0))
                .child(div().text_sm().child(title))
                .child(
                    div()
                        .text_xs()
                        .text_color(colors.muted_text)
                        .child(description),
                ),
        )
}

fn menu_choice(
    id: impl Into<gpui::ElementId>,
    title: &str,
    description: &'static str,
    selected: bool,
    highlighted: bool,
    colors: UiPalette,
    on_click: impl Fn(&gpui::ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> gpui::AnyElement {
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_between()
        .gap_3()
        .min_h(px(58.0))
        .px_3()
        .cursor_pointer()
        .when(highlighted, |item| item.bg(colors.selected))
        .hover(move |item| item.bg(colors.hover))
        .on_click(on_click)
        .child(
            div()
                .flex()
                .flex_col()
                .min_w(px(0.0))
                .gap(px(2.0))
                .child(div().truncate().text_sm().child(title.to_owned()))
                .child(
                    div()
                        .text_xs()
                        .text_color(colors.muted_text)
                        .child(description),
                ),
        )
        .children(selected.then(|| Icon::new(IconName::Check).size_4()))
        .into_any_element()
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 60 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

fn compact_number(value: u32) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn effort_label(effort: &kcastle_agent::ReasoningEffort) -> &'static str {
    match effort {
        kcastle_agent::ReasoningEffort::None => "Off",
        kcastle_agent::ReasoningEffort::Low => "Low",
        kcastle_agent::ReasoningEffort::Medium => "Medium",
        kcastle_agent::ReasoningEffort::High => "High",
        kcastle_agent::ReasoningEffort::Xhigh => "XHigh",
        _ => "Other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_are_compact() {
        assert_eq!(compact_number(984), "984");
        assert_eq!(compact_number(12_400), "12.4K");
        assert_eq!(compact_number(2_300_000), "2.3M");
    }
}
