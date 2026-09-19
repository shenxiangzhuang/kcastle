use std::time::Duration;

use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::component::input::Textarea;
use gpui_kit::component::menu::{PopupMenu, PopupMenuItem};
use gpui_kit::component::popover::Popover;
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::{Disableable, Icon, IconName};
use gpui_kit::{
    Context, Focusable, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, accesskit::Role, div, prelude::FluentBuilder, px,
};

use crate::app::{DesktopApp, composer_model_indices};
use crate::application::{composer_status, empty_conversation_view_model};
use crate::domain::{Action, ComposerMenu, RunState};
use crate::platform::gpui::measured_container;
use crate::ui_automation::ids;
use crate::ui_theme::{metrics, palette};

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
                                    .map(|button| {
                                        self.composer_menu_trigger(
                                            ComposerMenu::Workspace,
                                            button,
                                            cx,
                                        )
                                    }),
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
                                    .map(|button| {
                                        self.composer_menu_trigger(
                                            ComposerMenu::Commands,
                                            button,
                                            cx,
                                        )
                                    }),
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
                                .map(|button| {
                                    self.composer_menu_trigger(ComposerMenu::Permission, button, cx)
                                }),
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
                                .map(|button| {
                                    self.composer_menu_trigger(ComposerMenu::Model, button, cx)
                                }),
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

    fn composer_menu_trigger(
        &self,
        kind: ComposerMenu,
        button: Button,
        cx: &mut Context<Self>,
    ) -> gpui_kit::AnyElement {
        let owner = cx.entity().downgrade();
        Popover::new(("composer-popup", kind as usize))
            .anchor(if kind == ComposerMenu::Model {
                gpui_kit::Anchor::BottomRight
            } else {
                gpui_kit::Anchor::BottomLeft
            })
            .appearance(false)
            .trigger(button)
            .open(self.core.composer.menu == Some(kind))
            .when_some(self.composer_popup.as_ref(), |popover, popup| {
                popover.track_focus(&popup.focus_handle(cx))
            })
            .on_open_change(cx.listener(move |this, open, window, cx| {
                if *open {
                    this.open_composer_menu(kind, window, cx);
                } else if this.core.composer.menu == Some(kind) {
                    this.dispatch(Action::SetComposerMenu(None), window, cx);
                    this.composer_popup = None;
                }
            }))
            .content(move |_, _, cx| {
                div()
                    .id("composer-menu")
                    .accessibility_id(ids::COMPOSER_MENU)
                    .children(
                        owner
                            .upgrade()
                            .and_then(|owner| owner.read(cx).composer_popup.clone()),
                    )
            })
            .into_any_element()
    }

    pub(crate) fn open_composer_menu(
        &mut self,
        kind: ComposerMenu,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if kind == ComposerMenu::Model && !self.models[self.selected_model].model.has_api_key() {
            self.open_model_settings_dialog(window, cx);
            return;
        }
        self.dispatch(Action::SetComposerMenu(Some(kind)), window, cx);
        let owner = cx.entity().downgrade();
        // Menu builders read the app; run after the opening update releases its borrow.
        window.defer(cx, move |window, cx| {
            let Some(app) = owner.upgrade() else {
                return;
            };
            if app.read(cx).core.composer.menu != Some(kind) {
                return;
            }
            let popup = PopupMenu::build(window, cx, move |menu, window, cx| {
                composer_popup_menu(owner, kind, menu, window, cx)
            });
            app.update(cx, |this, cx| {
                cx.subscribe_in(
                    &popup,
                    window,
                    |this, _, _: &gpui_kit::DismissEvent, window, cx| {
                        this.dispatch(Action::SetComposerMenu(None), window, cx);
                        // Do not steal focus from an action that just opened a dialog.
                        if this.modal.is_none() {
                            this.input.update(cx, |input, cx| input.focus(window, cx));
                        }
                    },
                )
                .detach();
                popup.focus_handle(cx).focus(window, cx);
                this.composer_popup = Some(popup);
                cx.notify();
            });
        });
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

fn composer_popup_menu(
    owner: gpui_kit::WeakEntity<DesktopApp>,
    kind: ComposerMenu,
    mut menu: PopupMenu,
    window: &mut Window,
    cx: &mut Context<PopupMenu>,
) -> PopupMenu {
    menu = menu.min_w(px(240.0)).max_h(px(360.0)).scrollable(true);
    let Some(app) = owner.upgrade() else {
        return menu;
    };
    match kind {
        ComposerMenu::Commands => {
            let export = owner.clone();
            let permission = owner.clone();
            menu = menu
                .item(
                    PopupMenuItem::new("Export session")
                        .icon(IconName::ArrowDown)
                        .on_click(move |_, window, cx| {
                            let _ =
                                export.update(cx, |this, cx| this.export_session_log(window, cx));
                        }),
                )
                .submenu("Permission", window, cx, move |menu, window, cx| {
                    composer_popup_menu(
                        permission.clone(),
                        ComposerMenu::Permission,
                        menu,
                        window,
                        cx,
                    )
                });
            if composer_model_indices(&app.read(cx).models)
                .next()
                .is_some()
            {
                menu.submenu("Model", window, cx, move |menu, window, cx| {
                    composer_popup_menu(owner.clone(), ComposerMenu::Model, menu, window, cx)
                })
            } else {
                menu.item(
                    PopupMenuItem::new("Configure model").on_click(move |_, window, cx| {
                        let _ = owner
                            .update(cx, |this, cx| this.open_model_settings_dialog(window, cx));
                    }),
                )
            }
        }
        ComposerMenu::Permission => {
            let allow = app
                .read(cx)
                .selected_runtime
                .read(cx)
                .snapshot()
                .allow_all_tools;
            for (value, label) in [(false, "Ask before tools"), (true, "Allow all tools")] {
                let owner = owner.clone();
                menu = menu.item(PopupMenuItem::new(label).checked(allow == value).on_click(
                    move |_, _, cx| {
                        let _ = owner.update(cx, |this, cx| this.set_allow_all_tools(value, cx));
                    },
                ));
            }
            menu
        }
        ComposerMenu::Model => {
            let models = owner.clone();
            menu.submenu("Model", window, cx, move |menu, window, cx| {
                composer_popup_menu(models.clone(), ComposerMenu::Models, menu, window, cx)
            })
            .submenu("Reasoning effort", window, cx, move |menu, window, cx| {
                composer_popup_menu(owner.clone(), ComposerMenu::Effort, menu, window, cx)
            })
        }
        ComposerMenu::Models => {
            let app = app.read(cx);
            for index in composer_model_indices(&app.models) {
                let owner = owner.clone();
                menu = menu.item(
                    PopupMenuItem::new(app.models[index].label())
                        .checked(index == app.selected_model)
                        .on_click(move |_, _, cx| {
                            let _ = owner.update(cx, |this, cx| this.select_model(index, cx));
                        }),
                );
            }
            menu
        }
        ComposerMenu::Effort => {
            let app = app.read(cx);
            for effort in app.models[app.selected_model].model.reasoning_efforts() {
                let owner = owner.clone();
                let effort = *effort;
                menu = menu.item(
                    PopupMenuItem::new(effort_label(&effort))
                        .checked(app.selected_reasoning_effort == Some(effort))
                        .on_click(move |_, _, cx| {
                            let _ =
                                owner.update(cx, |this, cx| this.set_reasoning_effort(effort, cx));
                        }),
                );
            }
            menu
        }
        ComposerMenu::Workspace => {
            let app = app.read(cx);
            for (index, project) in app.project_store.projects().iter().enumerate() {
                let owner = owner.clone();
                menu = menu.item(
                    PopupMenuItem::new(project.name.clone())
                        .checked(index == app.core.workspace.active_project)
                        .on_click(move |_, window, cx| {
                            let _ =
                                owner.update(cx, |this, cx| this.switch_project(index, window, cx));
                        }),
                );
            }
            menu.separator().item(
                PopupMenuItem::new("Add workspace")
                    .icon(IconName::Plus)
                    .on_click(move |_, window, cx| {
                        let _ = owner.update(cx, |this, cx| this.add_project(window, cx));
                    }),
            )
        }
    }
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
