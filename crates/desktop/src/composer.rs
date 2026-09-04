use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::component::input::Textarea;
use gpui_kit::component::scroll::ScrollableElement;
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::{Disableable, Icon, IconName};
use gpui_kit::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, accesskit::Role, div, prelude::FluentBuilder, px,
    relative,
};

use crate::app::{DesktopApp, composer_model_indices};
use crate::application::{composer_status, empty_conversation_view_model};
use crate::domain::{Action, ComposerMenu, RunState};
use crate::platform::gpui::measured_container;
use crate::ui_automation::ids;
use crate::ui_theme::{UiPalette, metrics, palette};

impl DesktopApp {
    pub(crate) fn empty_conversation(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        let view = empty_conversation_view_model(&self.core);
        let workspace = self
            .project_store
            .project(self.core.workspace.active_project)
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
                    .max_w(px(self.core.layout.composer_max_width))
                    .gap_3()
                    .when(view.show_intro, |hero| {
                        hero.child(
                            div()
                                .flex()
                                .items_center()
                                .justify_center()
                                .gap_2()
                                .child(Icon::new(IconName::Bot).size_6())
                                .child(
                                    div()
                                        .text_xl()
                                        .font_weight(gpui_kit::FontWeight::SEMIBOLD)
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
                    })
                    .when(view.show_workspace, |hero| {
                        hero.child(
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
                    })
                    .when(view.show_composer, |hero| {
                        hero.child(self.composer_card(true, cx))
                    }),
            )
    }

    pub(crate) fn docked_composer(
        &self,
        window: &Window,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let colors = palette(cx);
        let status = composer_status(&self.core);
        let full_status = status.clone();
        let shaped_status: SharedString = status.clone().into();
        let style = window.text_style();
        let status_width = window
            .text_system()
            .shape_line(
                shaped_status.clone(),
                window.rem_size() * 0.75,
                &[style.to_run(shaped_status.len())],
                None,
            )
            .width;
        let status_is_truncated = status_width > px(self.core.layout.composer_max_width.max(0.0));
        div()
            .flex()
            .flex_col()
            .flex_none()
            .items_center()
            .px_4()
            .pb_2()
            .gap_2()
            .child(self.composer_card(false, cx))
            .when(!status.is_empty(), |composer| {
                composer.child(
                    div()
                        .id("composer-session-stats")
                        .w_full()
                        .max_w(px(self.core.layout.composer_max_width))
                        .text_center()
                        .text_xs()
                        .truncate()
                        .text_color(colors.muted_text)
                        .child(status)
                        .when(status_is_truncated, |status| {
                            status.tooltip(move |window, cx| {
                                Tooltip::new(full_status.clone()).build(window, cx)
                            })
                        }),
                )
            })
    }

    fn composer_card(&self, hero: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        let running = self.session_running();
        let selection_pending = self.selection_pending();
        let preparing = selection_pending || matches!(self.core.run, RunState::Preparing);
        let empty = self.input.read(cx).value().trim().is_empty();
        let model_configured = self.models[self.selected_model].model.has_api_key();
        let elapsed = self
            .selected_started_at
            .map(|started_at| started_at.elapsed())
            .unwrap_or_default();
        let model = if model_configured {
            self.selected_reasoning_effort
                .as_ref()
                .map(|effort| format!("{}  {}", self.model, effort_label(effort)))
                .unwrap_or_else(|| self.model.clone())
        } else {
            "Configure model".into()
        };
        let measurement_owner = cx.entity().downgrade();
        div()
            .id(if hero {
                "hero-composer"
            } else {
                "docked-composer"
            })
            .role(Role::Form)
            .accessibility_id(ids::COMPOSER)
            .aria_label("Message composer")
            .relative()
            .flex()
            .flex_col()
            .w_full()
            .max_w(px(self.core.layout.composer_max_width))
            .rounded(px(metrics::COMPOSER_RADIUS))
            .border_1()
            .border_color(colors.border)
            .bg(colors.surface)
            .shadow_lg()
            .child(measured_container(
                measurement_owner,
                |bounds, this: &mut DesktopApp, cx| {
                    this.update_composer_measurement(bounds.height, cx)
                },
                |this: &mut DesktopApp, window, cx| this.restore_chat_tail_after_layout(window, cx),
            ))
            .child(
                div()
                    .id(if hero {
                        "hero-composer-input"
                    } else {
                        "docked-composer-input"
                    })
                    .role(Role::Group)
                    .accessibility_id(ids::COMPOSER_INPUT)
                    .aria_label("Message the agent")
                    .capture_key_down(cx.listener(|this, event, window, cx| {
                        this.handle_root_key(event, window, cx)
                    }))
                    .child(
                        Textarea::new(&self.input)
                            .aria_label("Message the agent")
                            .appearance(false)
                            .bordered(false)
                            .text_base(),
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
                    .pb(px(metrics::COMPOSER_CONTROLS_BOTTOM_INSET))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .child(
                                Button::new(if hero { "hero-commands" } else { "commands" })
                                    .accessibility_id(ids::COMPOSER_COMMANDS)
                                    .icon(IconName::Plus)
                                    .ghost()
                                    .compact()
                                    .disabled(selection_pending)
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
                                .accessibility_id(ids::COMPOSER_PERMISSION)
                                .icon(
                                    if self.selected_runtime.read(cx).snapshot().allow_all_tools {
                                        IconName::CircleCheck
                                    } else {
                                        IconName::TriangleAlert
                                    },
                                )
                                .label(
                                    if self.selected_runtime.read(cx).snapshot().allow_all_tools {
                                        "Allow all tools"
                                    } else {
                                        "Ask before tools"
                                    },
                                )
                                .ghost()
                                .compact()
                                .disabled(selection_pending)
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
                                .accessibility_id(ids::COMPOSER_MODEL)
                                .label(model)
                                .ghost()
                                .compact()
                                .tooltip(if model_configured {
                                    "Select model and reasoning effort"
                                } else {
                                    "Configure an OpenAI or DeepSeek provider"
                                })
                                .disabled(running || selection_pending)
                                .on_key_down(cx.listener(|this, event, window, cx| {
                                    this.handle_root_key(event, window, cx)
                                }))
                                .on_click(cx.listener(
                                    |this, _, window, cx| {
                                        if this.models[this.selected_model].model.has_api_key() {
                                            this.open_composer_menu(
                                                ComposerMenu::Model,
                                                window,
                                                cx,
                                            );
                                        } else {
                                            this.open_model_settings_dialog(window, cx);
                                        }
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
                                    .accessibility_id(ids::COMPOSER_STOP)
                                    .icon(IconName::Close)
                                    .rounded(px(999.0))
                                    .tooltip("Stop")
                                    .on_click(cx.listener(|this, _, _, cx| this.abort(cx)))
                                    .into_any_element()
                            } else {
                                Button::new("send")
                                    .accessibility_id(ids::COMPOSER_SEND)
                                    .role(Role::DefaultButton)
                                    .icon(IconName::ArrowUp)
                                    .primary()
                                    .loading(preparing)
                                    .disabled(empty || preparing || !model_configured)
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

    pub(crate) fn composer_menu_view(
        &self,
        cx: &mut Context<Self>,
    ) -> Option<gpui_kit::AnyElement> {
        let colors = palette(cx);
        let menu = self.core.composer.menu?;
        let body = match menu {
            ComposerMenu::Commands => div()
                .flex()
                .flex_col()
                .child(menu_title("Commands", cx))
                .child(menu_item(
                    "command-export",
                    IconName::ArrowDown,
                    "Export session",
                    "Export the current session as JSONL",
                    self.core.composer.highlighted_item == 0,
                    colors,
                    cx.listener(|this, _, window, cx| {
                        this.dispatch(Action::SetComposerMenu(None), window, cx);
                        this.export_session_log(window, cx)
                    }),
                ))
                .child(menu_item(
                    "command-permission",
                    IconName::CircleCheck,
                    "Permission",
                    "Choose how tool calls are approved",
                    self.core.composer.highlighted_item == 1,
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
                    self.core.composer.highlighted_item == 2,
                    colors,
                    cx.listener(|this, _, window, cx| {
                        if composer_model_indices(&this.models).next().is_some() {
                            this.set_composer_menu(Some(ComposerMenu::Model), cx);
                        } else {
                            this.open_model_settings_dialog(window, cx);
                        }
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
                    !self.selected_runtime.read(cx).snapshot().allow_all_tools,
                    self.core.composer.highlighted_item == 0,
                    colors,
                    cx.listener(|this, _, _, cx| this.set_allow_all_tools(false, cx)),
                ))
                .child(menu_choice(
                    "permission-allow",
                    "Allow all tools",
                    "Automatically approve tool calls in this app",
                    self.selected_runtime.read(cx).snapshot().allow_all_tools,
                    self.core.composer.highlighted_item == 1,
                    colors,
                    cx.listener(|this, _, _, cx| this.set_allow_all_tools(true, cx)),
                ))
                .into_any_element(),
            ComposerMenu::Model => {
                let selected_model = self.models[self.selected_model].label();
                let effort = self
                    .selected_reasoning_effort
                    .as_ref()
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
                        self.core.composer.highlighted_item == 0,
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
                        self.core.composer.highlighted_item == 1,
                        colors,
                        cx.listener(|this, _, _, cx| {
                            this.set_composer_menu(Some(ComposerMenu::Effort), cx)
                        }),
                    ))
                    .into_any_element()
            }
            ComposerMenu::Models => {
                let models = &self.models;
                let models = composer_model_indices(models)
                    .enumerate()
                    .map(|(position, index)| {
                        (
                            position,
                            index,
                            models[index].label(),
                            index == self.selected_model,
                        )
                    })
                    .collect::<Vec<_>>();
                div()
                    .flex()
                    .flex_col()
                    .child(menu_title("Select model", cx))
                    .children(
                        models
                            .into_iter()
                            .map(|(position, index, label, selected)| {
                                menu_choice(
                                    ("composer-model", index),
                                    &label,
                                    "Use for future responses",
                                    selected,
                                    self.core.composer.highlighted_item == position,
                                    colors,
                                    cx.listener(move |this, _, _, cx| this.select_model(index, cx)),
                                )
                            }),
                    )
                    .into_any_element()
            }
            ComposerMenu::Effort => {
                let efforts = self.models[self.selected_model]
                    .model
                    .reasoning_efforts()
                    .to_vec();
                let selected = self.selected_reasoning_effort;
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
                            self.core.composer.highlighted_item == index,
                            colors,
                            cx.listener(move |this, _, _, cx| {
                                this.set_reasoning_effort(effort, cx);
                                this.dispatch_local(Action::SetComposerMenu(None), cx);
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
                        (
                            index,
                            project.name.clone(),
                            index == self.core.workspace.active_project,
                        )
                    })
                    .collect::<Vec<_>>();
                let mut project_items = Vec::new();
                for (index, name, selected) in projects {
                    project_items.push(menu_choice(
                        ("composer-workspace", index),
                        &name,
                        "Switch the active working directory",
                        selected,
                        self.core.composer.highlighted_item == index,
                        colors,
                        cx.listener(move |this, _, window, cx| {
                            this.dispatch(Action::SetComposerMenu(None), window, cx);
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
                        self.core.composer.highlighted_item == self.project_store.projects().len(),
                        colors,
                        cx.listener(|this, _, window, cx| {
                            this.dispatch(Action::SetComposerMenu(None), window, cx);
                            this.add_project(window, cx)
                        }),
                    ))
                    .into_any_element()
            }
        };
        Some(
            div()
                .id("composer-menu")
                .role(Role::Menu)
                .accessibility_id(ids::COMPOSER_MENU)
                .aria_label("Composer menu")
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

    pub(crate) fn approval_card(&self, cx: &mut Context<Self>) -> Option<gpui_kit::AnyElement> {
        let colors = palette(cx);
        self.core.approval.as_ref().map(|approval| {
            let allow_id = approval.call_id.clone();
            let deny_id = approval.call_id.clone();
            div()
                .flex()
                .justify_center()
                .px_4()
                .pb_2()
                .child(
                    div()
                        .id("approval-card")
                        .role(Role::AlertDialog)
                        .accessibility_id(ids::APPROVAL)
                        .aria_label(format!("Allow {}?", approval.name))
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_4()
                        .w_full()
                        .max_w(px(self.core.layout.composer_max_width))
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
                                        .font_weight(gpui_kit::FontWeight::SEMIBOLD)
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
                                .child(
                                    Button::new("deny-tool")
                                        .accessibility_id(ids::APPROVAL_DENY)
                                        .label("Deny")
                                        .on_click(cx.listener(move |this, _, _, cx| {
                                            this.decide(deny_id.clone(), false, cx)
                                        })),
                                )
                                .child(
                                    Button::new("allow-tool")
                                        .accessibility_id(ids::APPROVAL_ALLOW)
                                        .label("Allow")
                                        .primary()
                                        .on_click(cx.listener(move |this, _, _, cx| {
                                            this.decide(allow_id.clone(), true, cx)
                                        })),
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
        .h(px(metrics::TAB_HEIGHT))
        .px_3()
        .border_b_1()
        .border_color(colors.border)
        .font_weight(gpui_kit::FontWeight::SEMIBOLD)
        .child(title)
}

fn menu_item(
    id: impl Into<gpui_kit::ElementId>,
    icon: IconName,
    title: &'static str,
    description: &'static str,
    highlighted: bool,
    colors: UiPalette,
    on_click: impl Fn(&gpui_kit::ClickEvent, &mut gpui_kit::Window, &mut gpui_kit::App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .role(Role::MenuItem)
        .aria_label(title)
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
    id: impl Into<gpui_kit::ElementId>,
    title: &str,
    description: &'static str,
    selected: bool,
    highlighted: bool,
    colors: UiPalette,
    on_click: impl Fn(&gpui_kit::ClickEvent, &mut gpui_kit::Window, &mut gpui_kit::App) + 'static,
) -> gpui_kit::AnyElement {
    div()
        .id(id)
        .role(Role::MenuItem)
        .aria_label(title.to_owned())
        .aria_selected(selected)
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

fn effort_label(effort: &kcastle_agent::ReasoningEffort) -> &'static str {
    match effort {
        kcastle_agent::ReasoningEffort::None => "Off",
        kcastle_agent::ReasoningEffort::Minimal => "Minimal",
        kcastle_agent::ReasoningEffort::Low => "Low",
        kcastle_agent::ReasoningEffort::Medium => "Medium",
        kcastle_agent::ReasoningEffort::High => "High",
        kcastle_agent::ReasoningEffort::Xhigh => "XHigh",
    }
}
