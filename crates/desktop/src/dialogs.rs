use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder, px,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::{Input, InputState};
use gpui_component::scroll::ScrollableElement;
use gpui_component::{Disableable, Icon, IconName, Selectable, Sizable};
use kcastle_agent::ReasoningEffort;

use crate::app::DesktopApp;
use crate::settings::{Appearance, EnterBehavior};
use crate::ui_theme::{UiPalette, palette};

pub(crate) enum Modal {
    RenameSession(Entity<InputState>),
    DeleteSession,
    RemoveProject(usize),
    Settings(SettingsTab),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingsTab {
    General,
    Models,
}

impl DesktopApp {
    pub(crate) fn open_rename_session_dialog(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.current_session.as_os_str().is_empty() || self.control.is_some() {
            return;
        }
        let input = cx.new(|cx| InputState::new(window, cx).default_value(self.title.clone()));
        self.modal = Some(Modal::RenameSession(input.clone()));
        input.update(cx, |input, cx| input.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn open_delete_session_dialog(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.current_session.as_os_str().is_empty() || self.control.is_some() {
            return;
        }
        self.modal = Some(Modal::DeleteSession);
        self.modal_focus.focus(window);
        cx.notify();
    }

    pub(crate) fn open_remove_project_dialog(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.project_store.project(index).is_some() {
            self.modal = Some(Modal::RemoveProject(index));
            self.modal_focus.focus(window);
            cx.notify();
        }
    }

    pub(crate) fn open_settings_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.show_sidebar_options = false;
        self.composer_menu = None;
        self.modal = Some(Modal::Settings(SettingsTab::General));
        self.modal_focus.focus(window);
        cx.notify();
    }

    pub(crate) fn set_settings_tab(&mut self, tab: SettingsTab, cx: &mut Context<Self>) {
        self.show_sidebar_options = false;
        self.composer_menu = None;
        self.modal = Some(Modal::Settings(tab));
        cx.notify();
    }

    pub(crate) fn close_modal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.modal = None;
        self.input.update(cx, |input, cx| input.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn confirm_rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(Modal::RenameSession(input)) = &self.modal else {
            return;
        };
        let title = input.read(cx).value().trim().to_owned();
        if title.is_empty() {
            return;
        }
        self.modal = None;
        self.rename_current_session(title, window, cx);
    }

    pub(crate) fn modal_view(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        let colors = palette(cx);
        let content = match &self.modal {
            Some(Modal::RenameSession(input)) => modal_card("Rename session", colors)
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.muted_text)
                        .child("Use a short title that will be easy to find later."),
                )
                .child(Input::new(input).large())
                .child(
                    modal_actions()
                        .child(Button::new("cancel-rename").label("Cancel").on_click(
                            cx.listener(|this, _, window, cx| this.close_modal(window, cx)),
                        ))
                        .child(
                            Button::new("confirm-rename")
                                .label("Rename")
                                .primary()
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.confirm_rename(window, cx)
                                })),
                        ),
                )
                .into_any_element(),
            Some(Modal::DeleteSession) => modal_card("Delete session?", colors)
                .child(format!("“{}” will be permanently deleted.", self.title))
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.muted_text)
                        .child("This cannot be undone."),
                )
                .child(
                    modal_actions()
                        .child(
                            Button::new("cancel-delete-session")
                                .label("Cancel")
                                .on_click(
                                    cx.listener(|this, _, window, cx| this.close_modal(window, cx)),
                                ),
                        )
                        .child(
                            Button::new("confirm-delete-session")
                                .label("Delete")
                                .danger()
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.modal = None;
                                    this.delete_current_session(window, cx)
                                })),
                        ),
                )
                .into_any_element(),
            Some(Modal::RemoveProject(index)) => {
                let index = *index;
                let name = self
                    .project_store
                    .project(index)
                    .map(|project| project.name.clone())
                    .unwrap_or_default();
                modal_card("Remove project?", colors)
                    .child(format!("Remove “{name}” from K Castle?"))
                    .child(
                        div()
                            .text_sm()
                            .text_color(colors.muted_text)
                            .child("The project folder and its session history stay on disk."),
                    )
                    .child(
                        modal_actions()
                            .child(
                                Button::new("cancel-remove-project")
                                    .label("Cancel")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.close_modal(window, cx)
                                    })),
                            )
                            .child(
                                Button::new("confirm-remove-project")
                                    .label("Remove")
                                    .danger()
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.modal = None;
                                        this.remove_project(index, window, cx)
                                    })),
                            ),
                    )
                    .into_any_element()
            }
            Some(Modal::Settings(tab)) => {
                let (efforts, selected) = self
                    .models
                    .get(self.selected_model)
                    .map(|configured| {
                        (
                            configured.model.reasoning_efforts(),
                            configured.model.reasoning_effort().cloned(),
                        )
                    })
                    .unwrap_or_default();
                let buttons = efforts
                    .iter()
                    .enumerate()
                    .map(|(index, effort)| {
                        let effort = effort.clone();
                        Button::new(("reasoning-effort", index))
                            .label(reasoning_label(&effort))
                            .when(selected.as_ref() == Some(&effort), |button| {
                                button.primary()
                            })
                            .disabled(self.control.is_some())
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.set_reasoning_effort(effort.clone(), cx)
                            }))
                    })
                    .collect::<Vec<_>>();
                let model_options = self
                    .models
                    .iter()
                    .enumerate()
                    .map(|(index, configured)| {
                        (index, configured.label(), index == self.selected_model)
                    })
                    .collect::<Vec<_>>();
                let model_rows = model_options
                    .into_iter()
                    .map(|(index, label, selected)| {
                        settings_model_row(index, label, selected, self.control.is_some(), cx)
                    })
                    .collect::<Vec<_>>();
                let selected_tab = *tab;
                let body = match selected_tab {
                    SettingsTab::General => div()
                        .flex()
                        .flex_col()
                        .child(permission_settings_row(self.settings.allow_all_tools(), cx))
                        .child(settings_row(
                            "Project",
                            "The working directory for new sessions.",
                            display_path(&self.cwd),
                            colors,
                        ))
                        .child(appearance_settings_row(self.settings.appearance(), cx))
                        .child(motion_settings_row(self.settings.reduce_motion(), cx))
                        .child(enter_behavior_settings_row(
                            self.settings.enter_behavior(),
                            cx,
                        ))
                        .into_any_element(),
                    SettingsTab::Models => div()
                        .flex()
                        .flex_col()
                        .children(model_rows)
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .gap_6()
                                .min_h(px(74.0))
                                .border_b_1()
                                .border_color(colors.border)
                                .child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .gap_1()
                                        .child(
                                            div()
                                                .font_weight(gpui::FontWeight::MEDIUM)
                                                .child("Reasoning effort"),
                                        )
                                        .child(
                                            div()
                                                .text_sm()
                                                .text_color(colors.muted_text)
                                                .child("Used for future model responses."),
                                        ),
                                )
                                .child(div().flex().items_center().gap_2().children(buttons)),
                        )
                        .into_any_element(),
                };
                div()
                    .flex()
                    .w(px(800.0))
                    .h(px(570.0))
                    .rounded(px(24.0))
                    .bg(colors.surface)
                    .shadow_xl()
                    .overflow_hidden()
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .w(px(188.0))
                            .flex_none()
                            .p_3()
                            .bg(colors.sidebar)
                            .border_r_1()
                            .border_color(colors.border)
                            .child(
                                div()
                                    .px_2()
                                    .pt_2()
                                    .pb_4()
                                    .font_weight(gpui::FontWeight::SEMIBOLD)
                                    .child("Settings"),
                            )
                            .child(
                                settings_nav(
                                    "settings-general",
                                    "General",
                                    IconName::Settings,
                                    selected_tab == SettingsTab::General,
                                    colors,
                                )
                                .on_click(cx.listener(
                                    |this, _, _, cx| {
                                        this.set_settings_tab(SettingsTab::General, cx)
                                    },
                                )),
                            )
                            .child(
                                settings_nav(
                                    "settings-models",
                                    "Models",
                                    IconName::Bot,
                                    selected_tab == SettingsTab::Models,
                                    colors,
                                )
                                .on_click(cx.listener(
                                    |this, _, _, cx| this.set_settings_tab(SettingsTab::Models, cx),
                                )),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .flex_1()
                            .min_w(px(0.0))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .h(px(58.0))
                                    .px_6()
                                    .child(
                                        div()
                                            .text_lg()
                                            .font_weight(gpui::FontWeight::SEMIBOLD)
                                            .child(match selected_tab {
                                                SettingsTab::General => "General",
                                                SettingsTab::Models => "Models",
                                            }),
                                    )
                                    .child(
                                        Button::new("close-settings")
                                            .icon(IconName::Close)
                                            .ghost()
                                            .compact()
                                            .tooltip("Close")
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.close_modal(window, cx)
                                            })),
                                    ),
                            )
                            .child(
                                div()
                                    .flex_1()
                                    .min_h(px(0.0))
                                    .overflow_y_scrollbar()
                                    .px_6()
                                    .pb_6()
                                    .child(body),
                            ),
                    )
                    .into_any_element()
            }
            None => return None,
        };

        Some(
            div()
                .id("modal-overlay")
                .absolute()
                .top_0()
                .right_0()
                .bottom_0()
                .left_0()
                .flex()
                .occlude()
                .items_center()
                .justify_center()
                .bg(colors.overlay)
                .track_focus(&self.modal_focus)
                .tab_index(0)
                .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                    if event.keystroke.key == "enter"
                        && matches!(this.modal, Some(Modal::RenameSession(_)))
                    {
                        this.confirm_rename(window, cx);
                    }
                }))
                .on_click(cx.listener(|this, _, window, cx| this.close_modal(window, cx)))
                .child(
                    div()
                        .id("modal-content")
                        .on_click(|_, _, cx| cx.stop_propagation())
                        .child(content),
                )
                .into_any_element(),
        )
    }
}

fn modal_card(title: &'static str, colors: UiPalette) -> gpui::Div {
    div()
        .flex()
        .flex_col()
        .gap_4()
        .w(px(480.0))
        .p_6()
        .rounded_xl()
        .border_1()
        .border_color(colors.border)
        .bg(colors.surface)
        .shadow_xl()
        .child(
            div()
                .text_lg()
                .font_weight(gpui::FontWeight::SEMIBOLD)
                .child(title),
        )
}

fn modal_actions() -> gpui::Div {
    div().flex().items_center().justify_end().gap_2().pt_2()
}

fn settings_row(
    label: &'static str,
    description: &'static str,
    value: String,
    colors: UiPalette,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap_6()
        .min_h(px(74.0))
        .border_b_1()
        .border_color(colors.border)
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .min_w(px(0.0))
                .gap_1()
                .child(div().font_weight(gpui::FontWeight::MEDIUM).child(label))
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.muted_text)
                        .child(description),
                ),
        )
        .child(
            div()
                .w(px(240.0))
                .flex_none()
                .truncate()
                .text_right()
                .text_sm()
                .text_color(colors.muted_text)
                .child(value),
        )
}

fn permission_settings_row(allow_all: bool, cx: &mut Context<DesktopApp>) -> impl IntoElement {
    let colors = palette(cx);
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap_6()
        .min_h(px(82.0))
        .border_b_1()
        .border_color(colors.border)
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .gap_1()
                .child(
                    div()
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .child("Permission"),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.muted_text)
                        .child("Choose whether shell calls require approval."),
                ),
        )
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    Button::new("settings-permission-ask")
                        .label("Ask")
                        .when(!allow_all, |button| button.primary())
                        .on_click(
                            cx.listener(|this, _, _, cx| this.set_allow_all_tools(false, cx)),
                        ),
                )
                .child(
                    Button::new("settings-permission-allow")
                        .label("Allow all")
                        .when(allow_all, |button| button.primary())
                        .on_click(cx.listener(|this, _, _, cx| this.set_allow_all_tools(true, cx))),
                ),
        )
}

fn appearance_settings_row(
    appearance: Appearance,
    cx: &mut Context<DesktopApp>,
) -> impl IntoElement {
    let colors = palette(cx);
    settings_control_row(
        "Appearance",
        "Use the system appearance or choose a fixed theme.",
        div()
            .flex()
            .items_center()
            .gap_1()
            .child(
                Button::new("appearance-system")
                    .label("System")
                    .compact()
                    .when(appearance == Appearance::System, |button| button.primary())
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_appearance(Appearance::System, window, cx)
                    })),
            )
            .child(
                Button::new("appearance-light")
                    .label("Light")
                    .compact()
                    .when(appearance == Appearance::Light, |button| button.primary())
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_appearance(Appearance::Light, window, cx)
                    })),
            )
            .child(
                Button::new("appearance-dark")
                    .label("Dark")
                    .compact()
                    .when(appearance == Appearance::Dark, |button| button.primary())
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.set_appearance(Appearance::Dark, window, cx)
                    })),
            )
            .into_any_element(),
        colors,
    )
}

fn enter_behavior_settings_row(
    behavior: EnterBehavior,
    cx: &mut Context<DesktopApp>,
) -> impl IntoElement {
    let colors = palette(cx);
    settings_control_row(
        "Enter while busy",
        "Steer the active turn or queue a follow-up after it settles.",
        div()
            .flex()
            .items_center()
            .gap_1()
            .child(
                Button::new("enter-steer")
                    .label("Steer")
                    .compact()
                    .when(behavior == EnterBehavior::Steer, |button| button.primary())
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.set_enter_behavior(EnterBehavior::Steer, cx)
                    })),
            )
            .child(
                Button::new("enter-queue")
                    .label("Queue")
                    .compact()
                    .when(behavior == EnterBehavior::Queue, |button| button.primary())
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.set_enter_behavior(EnterBehavior::Queue, cx)
                    })),
            )
            .into_any_element(),
        colors,
    )
}

fn motion_settings_row(reduce_motion: bool, cx: &mut Context<DesktopApp>) -> impl IntoElement {
    let colors = palette(cx);
    settings_control_row(
        "Motion",
        "Reduce non-essential interface animation.",
        div()
            .flex()
            .items_center()
            .gap_1()
            .child(
                Button::new("motion-standard")
                    .label("Standard")
                    .compact()
                    .when(!reduce_motion, |button| button.primary())
                    .on_click(cx.listener(|this, _, _, cx| this.set_reduce_motion(false, cx))),
            )
            .child(
                Button::new("motion-reduced")
                    .label("Reduced")
                    .compact()
                    .when(reduce_motion, |button| button.primary())
                    .on_click(cx.listener(|this, _, _, cx| this.set_reduce_motion(true, cx))),
            )
            .into_any_element(),
        colors,
    )
}

fn settings_control_row(
    label: &'static str,
    description: &'static str,
    control: gpui::AnyElement,
    colors: UiPalette,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap_6()
        .min_h(px(82.0))
        .border_b_1()
        .border_color(colors.border)
        .child(
            div()
                .flex()
                .flex_col()
                .flex_1()
                .min_w(px(0.0))
                .gap_1()
                .child(div().font_weight(gpui::FontWeight::MEDIUM).child(label))
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.muted_text)
                        .child(description),
                ),
        )
        .child(control)
}

fn settings_model_row(
    index: usize,
    label: String,
    selected: bool,
    busy: bool,
    cx: &mut Context<DesktopApp>,
) -> gpui::AnyElement {
    let colors = palette(cx);
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap_4()
        .min_h(px(68.0))
        .border_b_1()
        .border_color(colors.border)
        .child(
            div()
                .flex()
                .items_center()
                .min_w(px(0.0))
                .gap_2()
                .child(
                    Icon::new(IconName::Bot)
                        .size_4()
                        .text_color(colors.muted_text),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .flex_1()
                        .min_w(px(0.0))
                        .gap_1()
                        .child(
                            div()
                                .truncate()
                                .font_weight(gpui::FontWeight::MEDIUM)
                                .child(label),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(colors.muted_text)
                                .child(if selected {
                                    "Current model"
                                } else {
                                    "Available"
                                }),
                        ),
                ),
        )
        .child(
            Button::new(("select-settings-model", index))
                .label(if selected { "Selected" } else { "Use" })
                .compact()
                .when(selected, |button| button.primary())
                .disabled(selected || busy)
                .on_click(cx.listener(move |this, _, _, cx| this.select_model(index, cx))),
        )
        .into_any_element()
}

fn settings_nav(
    id: &'static str,
    label: &'static str,
    icon: IconName,
    selected: bool,
    _colors: UiPalette,
) -> Button {
    Button::new(id)
        .icon(icon)
        .label(label)
        .ghost()
        .w_full()
        .selected(selected)
}

fn reasoning_label(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "Off",
        ReasoningEffort::Low => "Low",
        ReasoningEffort::Medium => "Medium",
        ReasoningEffort::High => "High",
        ReasoningEffort::Xhigh => "XHigh",
        _ => "Other",
    }
}

fn display_path(path: &std::path::Path) -> String {
    std::env::var_os("HOME")
        .and_then(|home| {
            path.strip_prefix(home)
                .ok()
                .map(|relative| format!("~/{}", relative.display()))
        })
        .unwrap_or_else(|| path.display().to_string())
}
