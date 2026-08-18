use std::path::PathBuf;

use gpui::{
    AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder, px, rgb,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::{Input, InputState};
use gpui_component::scroll::ScrollableElement;
use gpui_component::{Disableable, Icon, IconName, Selectable, Sizable};
use kcastle_agent::{DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID};

use crate::app::{DesktopApp, active_model_index};
use crate::domain::Action;
use crate::settings::{Appearance, EnterBehavior, ProviderModel, ProviderProfile};
use crate::ui_theme::{UiPalette, palette};

pub(crate) enum Modal {
    RenameSession {
        project_index: usize,
        path: PathBuf,
        input: Entity<InputState>,
    },
    DeleteSession {
        project_index: usize,
        path: PathBuf,
        title: String,
    },
    RemoveProject(usize),
    Settings(Box<SettingsDialog>),
}

fn settings_dialog_view(
    app: &DesktopApp,
    dialog: &SettingsDialog,
    cx: &mut Context<DesktopApp>,
    colors: UiPalette,
) -> gpui::AnyElement {
    let selected_tab = dialog.tab;
    let body = match selected_tab {
        SettingsTab::General => div()
            .flex()
            .flex_col()
            .child(permission_settings_row(app.settings.allow_all_tools(), cx))
            .child(settings_row(
                "Project",
                "The working directory for new sessions.",
                display_path(&app.core.workspace.cwd),
                colors,
            ))
            .child(appearance_settings_row(app.settings.appearance(), cx))
            .child(motion_settings_row(app.settings.reduce_motion(), cx))
            .child(enter_behavior_settings_row(
                app.settings.enter_behavior(),
                cx,
            ))
            .child(
                div()
                    .pt_6()
                    .text_xs()
                    .text_color(colors.muted_text)
                    .child(concat!("v", env!("CARGO_PKG_VERSION"))),
            )
            .into_any_element(),
        SettingsTab::Models => models_settings_view(app, dialog, cx, colors),
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
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            settings_nav(
                                "settings-general",
                                "General",
                                IconName::Settings,
                                selected_tab == SettingsTab::General,
                                colors,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.set_settings_tab(SettingsTab::General, cx)
                            })),
                        )
                        .child(
                            settings_nav(
                                "settings-models",
                                "Models",
                                IconName::Bot,
                                selected_tab == SettingsTab::Models,
                                colors,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.set_settings_tab(SettingsTab::Models, cx)
                            })),
                        ),
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
                                .on_click(
                                    cx.listener(|this, _, window, cx| this.close_modal(window, cx)),
                                ),
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

fn models_settings_view(
    app: &DesktopApp,
    dialog: &SettingsDialog,
    cx: &mut Context<DesktopApp>,
    colors: UiPalette,
) -> gpui::AnyElement {
    let busy = app.task_active();
    let editor_provider = dialog
        .model_editor
        .as_ref()
        .map(|editor| editor.provider_id);
    let rows = [DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID]
        .into_iter()
        .filter(|provider_id| provider_configured(app, provider_id))
        .enumerate()
        .map(|(provider_index, provider_id)| {
            let is_open = editor_provider == Some(provider_id);
            let profile = provider_profile(app, provider_id);
            let model_count = app
                .models
                .iter()
                .filter(|model| model.provider_id == provider_id)
                .count();
            let current = app
                .models
                .get(app.selected_model)
                .filter(|model| model.provider_id == provider_id)
                .map(|model| model.profile.model_id.clone());
            let mut row =
                div()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .p_3()
                    .rounded(px(12.0))
                    .border_1()
                    .border_color(colors.border)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .font_weight(gpui::FontWeight::MEDIUM)
                                    .child(profile.display_name.clone()),
                            )
                            .child(div().w(px(8.0)).h(px(8.0)).rounded_full().bg(rgb(0x36a763)))
                            .child(div().text_xs().text_color(colors.muted_text).child(
                                match current {
                                    Some(model) => {
                                        format!("{model_count} models · {model} current")
                                    }
                                    None => format!("{model_count} models"),
                                },
                            ))
                            .child(div().flex_1())
                            .child(
                                Button::new(("edit-provider", provider_index))
                                    .label(if is_open { "Close" } else { "Edit" })
                                    .compact()
                                    .disabled(busy)
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        if is_open {
                                            this.cancel_model_editor(cx);
                                        } else {
                                            this.edit_provider(provider_id, window, cx);
                                        }
                                    })),
                            ),
                    );
            if is_open && let Some(editor) = &dialog.model_editor {
                row = row.child(model_editor_view(editor, cx, colors));
            }
            row.into_any_element()
        })
        .collect::<Vec<_>>();
    let missing_known = [OPENAI_PROVIDER_ID, DEEPSEEK_PROVIDER_ID]
        .into_iter()
        .filter(|provider_id| !provider_configured(app, provider_id))
        .collect::<Vec<_>>();
    let standalone_editor = dialog
        .model_editor
        .as_ref()
        .filter(|editor| !provider_configured(app, editor.provider_id));

    div()
        .flex()
        .flex_col()
        .gap_3()
        .max_w(px(720.0))
        .child(
            div()
                .text_sm()
                .text_color(colors.muted_text)
                .child("Choose a provider, then enter its API key."),
        )
        .children(dialog.saved_provider.as_ref().map(|provider| {
            div()
                .text_xs()
                .text_color(rgb(0x36a763))
                .child(format!("Saved {provider}."))
        }))
        .child(div().flex().flex_col().gap_2().mt_3().children(rows))
        .children(standalone_editor.map(|editor| model_editor_view(editor, cx, colors)))
        .child(
            div()
                .flex()
                .gap_2()
                .children(
                    missing_known
                        .into_iter()
                        .enumerate()
                        .map(|(index, provider_id)| {
                            let label = crate::default_provider_profile(provider_id).display_name;
                            Button::new(("add-known-provider", index))
                                .icon(IconName::Plus)
                                .label(label)
                                .w_full()
                                .disabled(busy)
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    this.edit_provider(provider_id, window, cx);
                                }))
                        }),
                ),
        )
        .into_any_element()
}

fn model_editor_view(
    editor: &ModelEditor,
    cx: &mut Context<DesktopApp>,
    colors: UiPalette,
) -> gpui::AnyElement {
    let profile = crate::default_provider_profile(editor.provider_id);
    let model_rows = editor
        .models
        .iter()
        .enumerate()
        .map(|(index, row)| model_editor_row_view(index, row, editor.models.len(), cx, colors))
        .collect::<Vec<_>>();
    div()
        .flex()
        .flex_col()
        .gap_3()
        .p_4()
        .rounded(px(12.0))
        .bg(colors.subtle)
        .child(
            div()
                .flex()
                .items_baseline()
                .gap_2()
                .child(
                    div()
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .child(profile.display_name),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(colors.muted_text)
                        .child(editor.provider_id),
                ),
        )
        .child(model_editor_field("API key", &editor.api_key, true))
        .child(
            div().border_t_1().border_color(colors.border).pt_2().child(
                Button::new("toggle-model-custom-settings")
                    .icon(if editor.advanced_open {
                        IconName::ChevronDown
                    } else {
                        IconName::ChevronRight
                    })
                    .label("Customized settings")
                    .ghost()
                    .compact()
                    .on_click(cx.listener(|this, _, _, cx| this.toggle_model_advanced(cx))),
            ),
        )
        .when(editor.advanced_open, |card| {
            card.child(model_editor_field("API address", &editor.api_base, false))
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .pt_2()
                        .border_t_1()
                        .border_color(colors.border)
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap_1()
                                .child(
                                    div()
                                        .text_sm()
                                        .font_weight(gpui::FontWeight::MEDIUM)
                                        .child("Model catalog"),
                                )
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(colors.muted_text)
                                        .child("Models available from this provider."),
                                ),
                        )
                        .children(model_rows)
                        .child(
                            Button::new("add-provider-model")
                                .icon(IconName::Plus)
                                .label("Add model")
                                .compact()
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.add_provider_model(window, cx)
                                })),
                        ),
                )
        })
        .children(editor.validation_error.as_ref().map(|error| {
            div()
                .text_xs()
                .text_color(colors.danger)
                .child(error.clone())
        }))
        .child(
            div()
                .flex()
                .justify_end()
                .gap_2()
                .child(
                    Button::new("cancel-model-editor")
                        .label("Cancel")
                        .on_click(cx.listener(|this, _, _, cx| this.cancel_model_editor(cx))),
                )
                .child(
                    Button::new("save-model-editor")
                        .label("Save")
                        .primary()
                        .on_click(cx.listener(|this, _, _, cx| this.save_model_editor(cx))),
                ),
        )
        .into_any_element()
}

fn model_editor_row_view(
    index: usize,
    row: &ModelEditorRow,
    model_count: usize,
    cx: &mut Context<DesktopApp>,
    colors: UiPalette,
) -> gpui::AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_2()
        .p_2()
        .rounded(px(8.0))
        .border_1()
        .border_color(colors.border)
        .child(
            div()
                .flex()
                .gap_2()
                .child(
                    div()
                        .flex_1()
                        .child(model_editor_field("Model ID", &row.model_id, false)),
                )
                .child(div().flex_1().child(model_editor_field(
                    "Display name",
                    &row.display_name,
                    false,
                )))
                .child(
                    Button::new(("remove-provider-model", index))
                        .icon(IconName::Delete)
                        .ghost()
                        .compact()
                        .disabled(model_count <= 1)
                        .tooltip("Delete model")
                        .on_click(
                            cx.listener(move |this, _, _, cx| {
                                this.remove_provider_model(index, cx)
                            }),
                        ),
                ),
        )
        .child(
            div()
                .flex()
                .gap_2()
                .child(div().flex_1().child(model_editor_field(
                    "Context window",
                    &row.context_window,
                    false,
                )))
                .child(div().flex_1().child(model_editor_field(
                    "Max output tokens",
                    &row.max_output_tokens,
                    false,
                )))
                .child(div().w(px(28.0)).flex_none()),
        )
        .into_any_element()
}

fn model_editor_field(
    label: &'static str,
    state: &Entity<InputState>,
    password: bool,
) -> gpui::AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .text_xs()
                .font_weight(gpui::FontWeight::MEDIUM)
                .child(label),
        )
        .child(
            Input::new(state)
                .small()
                .when(password, |input| input.mask_toggle()),
        )
        .into_any_element()
}

fn model_editor(
    provider_id: &'static str,
    profile: ProviderProfile,
    configured_key: bool,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> ModelEditor {
    let mut input =
        |value: String, placeholder: &'static str, masked: bool, cx: &mut Context<DesktopApp>| {
            cx.new(|cx| {
                InputState::new(window, cx)
                    .default_value(value)
                    .placeholder(placeholder)
                    .masked(masked)
            })
        };
    ModelEditor {
        provider_id,
        api_key: input(
            String::new(),
            if configured_key {
                "Configured — enter a new value to replace"
            } else {
                "Enter your API key"
            },
            true,
            cx,
        ),
        api_base: input(profile.api_base, "https://api.example.com/v1", false, cx),
        models: profile
            .models
            .into_iter()
            .map(|model| model_editor_row(model, window, cx))
            .collect(),
        advanced_open: true,
        validation_error: None,
    }
}

fn model_editor_row(
    profile: ProviderModel,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> ModelEditorRow {
    ModelEditorRow {
        model_id: cx.new(|cx| {
            InputState::new(window, cx)
                .default_value(profile.model_id)
                .placeholder("Model ID")
        }),
        display_name: cx.new(|cx| {
            InputState::new(window, cx)
                .default_value(profile.display_name)
                .placeholder("Display name")
        }),
        context_window: cx.new(|cx| {
            InputState::new(window, cx)
                .default_value(profile.context_window.to_string())
                .placeholder("128K")
        }),
        max_output_tokens: cx.new(|cx| {
            InputState::new(window, cx)
                .default_value(
                    profile
                        .max_output_tokens
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                )
                .placeholder("Provider default")
        }),
    }
}

fn parse_capacity(value: &str) -> Option<usize> {
    let normalized = value.trim().to_ascii_uppercase();
    let (digits, multiplier) = normalized
        .strip_suffix('K')
        .map(|digits| (digits, 1_000usize))
        .or_else(|| {
            normalized
                .strip_suffix('M')
                .map(|digits| (digits, 1_000_000))
        })
        .unwrap_or((normalized.as_str(), 1));
    digits
        .parse::<usize>()
        .ok()
        .and_then(|count| count.checked_mul(multiplier))
        .filter(|count| *count > 0)
}

fn parse_optional_output_tokens(value: &str) -> Option<Option<u32>> {
    if value.trim().is_empty() {
        return Some(None);
    }
    parse_capacity(value)
        .and_then(|value| u32::try_from(value).ok())
        .map(Some)
}

fn provider_configured(app: &DesktopApp, provider_id: &str) -> bool {
    app.models
        .iter()
        .any(|model| model.provider_id == provider_id && model.model.has_api_key())
}

fn provider_profile(app: &DesktopApp, provider_id: &'static str) -> ProviderProfile {
    app.settings
        .provider_profiles()
        .iter()
        .find(|profile| profile.provider_id == provider_id)
        .cloned()
        .unwrap_or_else(|| crate::default_provider_profile(provider_id))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SettingsTab {
    General,
    Models,
}

pub(crate) struct SettingsDialog {
    tab: SettingsTab,
    model_editor: Option<ModelEditor>,
    saved_provider: Option<String>,
}

struct ModelEditor {
    provider_id: &'static str,
    api_key: Entity<InputState>,
    api_base: Entity<InputState>,
    models: Vec<ModelEditorRow>,
    advanced_open: bool,
    validation_error: Option<String>,
}

struct ModelEditorRow {
    model_id: Entity<InputState>,
    display_name: Entity<InputState>,
    context_window: Entity<InputState>,
    max_output_tokens: Entity<InputState>,
}

impl DesktopApp {
    pub(crate) fn open_target_rename_session_dialog(
        &mut self,
        project_index: usize,
        path: PathBuf,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if path.as_os_str().is_empty() || self.session_is_active(project_index, &path, cx) {
            return;
        }
        let input = cx.new(|cx| InputState::new(window, cx).default_value(title));
        self.modal = Some(Modal::RenameSession {
            project_index,
            path,
            input: input.clone(),
        });
        input.update(cx, |input, cx| input.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn open_target_delete_session_dialog(
        &mut self,
        project_index: usize,
        path: PathBuf,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if path.as_os_str().is_empty() || self.session_is_active(project_index, &path, cx) {
            return;
        }
        self.modal = Some(Modal::DeleteSession {
            project_index,
            path,
            title,
        });
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
        self.dispatch(Action::CloseTransientOverlays, window, cx);
        self.modal = Some(Modal::Settings(Box::new(SettingsDialog {
            tab: SettingsTab::General,
            model_editor: None,
            saved_provider: None,
        })));
        self.modal_focus.focus(window);
        cx.notify();
    }

    pub(crate) fn open_model_settings_dialog(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.open_settings_dialog(window, cx);
        self.set_settings_tab(SettingsTab::Models, cx);
    }

    pub(crate) fn set_settings_tab(&mut self, tab: SettingsTab, cx: &mut Context<Self>) {
        self.dispatch_local(Action::CloseTransientOverlays, cx);
        if let Some(Modal::Settings(dialog)) = &mut self.modal {
            dialog.tab = tab;
            dialog.model_editor = None;
        }
        cx.notify();
    }

    pub(crate) fn edit_provider(
        &mut self,
        provider_id: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let profile = provider_profile(self, provider_id);
        let configured_key = self
            .models
            .iter()
            .find(|model| model.provider_id == provider_id)
            .is_some_and(|model| model.model.has_api_key());
        let editor = model_editor(provider_id, profile, configured_key, window, cx);
        self.set_model_editor(editor, window, cx);
    }

    fn set_model_editor(
        &mut self,
        editor: ModelEditor,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let focus = editor.api_key.clone();
        if let Some(Modal::Settings(dialog)) = &mut self.modal {
            dialog.tab = SettingsTab::Models;
            dialog.saved_provider = None;
            dialog.model_editor = Some(editor);
        }
        focus.update(cx, |input, cx| input.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn cancel_model_editor(&mut self, cx: &mut Context<Self>) {
        if let Some(Modal::Settings(dialog)) = &mut self.modal {
            dialog.model_editor = None;
        }
        cx.notify();
    }

    pub(crate) fn toggle_model_advanced(&mut self, cx: &mut Context<Self>) {
        if let Some(Modal::Settings(dialog)) = &mut self.modal
            && let Some(editor) = &mut dialog.model_editor
        {
            editor.advanced_open = !editor.advanced_open;
        }
        cx.notify();
    }

    pub(crate) fn add_provider_model(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let row = model_editor_row(ProviderModel::new("", "", 128_000, None), window, cx);
        let focus = row.model_id.clone();
        if let Some(Modal::Settings(dialog)) = &mut self.modal
            && let Some(editor) = &mut dialog.model_editor
        {
            editor.models.push(row);
            editor.validation_error = None;
        }
        focus.update(cx, |input, cx| input.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn remove_provider_model(&mut self, index: usize, cx: &mut Context<Self>) {
        if let Some(Modal::Settings(dialog)) = &mut self.modal
            && let Some(editor) = &mut dialog.model_editor
            && editor.models.len() > 1
            && index < editor.models.len()
        {
            editor.models.remove(index);
            editor.validation_error = None;
        }
        cx.notify();
    }

    pub(crate) fn save_model_editor(&mut self, cx: &mut Context<Self>) {
        let Some(Modal::Settings(dialog)) = &self.modal else {
            return;
        };
        let Some(editor) = &dialog.model_editor else {
            return;
        };
        if self.task_active() {
            return;
        }
        let provider_id = editor.provider_id;
        let api_key = editor.api_key.read(cx).value().trim().to_owned();
        let api_base = editor.api_base.read(cx).value().trim().to_owned();
        let configured = provider_configured(self, provider_id);
        let mut model_profiles = Vec::with_capacity(editor.models.len());
        let mut model_ids = std::collections::HashSet::new();
        let mut error = if editor.models.is_empty() {
            Some("At least one model is required.".into())
        } else if api_base.is_empty() {
            Some("API address is required.".into())
        } else if !configured && api_key.is_empty() {
            Some("API key is required for a new provider.".into())
        } else {
            None
        };
        if error.is_none() {
            for (index, row) in editor.models.iter().enumerate() {
                let model_id = row.model_id.read(cx).value().trim().to_owned();
                let display_name = row.display_name.read(cx).value().trim().to_owned();
                let context = row.context_window.read(cx).value().trim().to_owned();
                let max_output = row.max_output_tokens.read(cx).value().trim().to_owned();
                let row_number = index + 1;
                let context_window = parse_capacity(&context);
                let max_output_tokens = parse_optional_output_tokens(&max_output);
                error = if model_id.is_empty() {
                    Some(format!("Model {row_number}: model ID is required."))
                } else if !model_ids.insert(model_id.clone()) {
                    Some(format!("Model {row_number}: model ID must be unique."))
                } else {
                    match (context_window, max_output_tokens) {
                        (None, _) => Some(format!(
                            "Model {row_number}: context window must be a positive number such as 256K or 1M."
                        )),
                        (_, None) => Some(format!(
                            "Model {row_number}: max output tokens must be blank or a positive number."
                        )),
                        (Some(context_window), Some(max_output_tokens)) => {
                            model_profiles.push(ProviderModel::new(
                                model_id,
                                display_name,
                                context_window,
                                max_output_tokens,
                            ));
                            None
                        }
                    }
                };
                if error.is_some() {
                    break;
                }
            }
        }
        if let Some(error) = error {
            if let Some(Modal::Settings(dialog)) = &mut self.modal
                && let Some(editor) = &mut dialog.model_editor
            {
                editor.validation_error = Some(error);
            }
            cx.notify();
            return;
        }

        let defaults = crate::default_provider_profile(provider_id);
        let profile = ProviderProfile::new(
            provider_id,
            defaults.display_name.clone(),
            api_base.clone(),
            model_profiles,
        );
        let key_override = (!api_key.is_empty()).then_some(api_key.clone());
        if let Err(error) = self
            .settings
            .save_provider_profile(profile.clone(), key_override.clone())
        {
            if let Some(Modal::Settings(dialog)) = &mut self.modal
                && let Some(editor) = &mut dialog.model_editor
            {
                editor.validation_error = Some(format!("Could not save provider: {error}"));
            }
            cx.notify();
            return;
        }

        let previous_models = self.models.clone();
        let previous_selected_id = previous_models
            .get(self.selected_model)
            .map(|model| model.id.clone());
        let template = previous_models
            .iter()
            .find(|model| model.provider_id == provider_id)
            .map(|model| model.model.clone());
        let replacement = profile
            .models
            .iter()
            .cloned()
            .map(|model_profile| {
                let model = previous_models
                    .iter()
                    .find(|model| {
                        model.provider_id == provider_id
                            && model.profile.model_id == model_profile.model_id
                    })
                    .map(|configured| {
                        configured.model.reconfigured(
                            profile.display_name.clone(),
                            key_override.clone(),
                            profile.api_base.clone(),
                            model_profile.model_id.clone(),
                            model_profile.context_window,
                        )
                    })
                    .or_else(|| {
                        template.as_ref().map(|template| {
                            template.reconfigured(
                                profile.display_name.clone(),
                                key_override.clone(),
                                profile.api_base.clone(),
                                model_profile.model_id.clone(),
                                model_profile.context_window,
                            )
                        })
                    })
                    .unwrap_or_else(|| {
                        crate::build_model(&profile, &model_profile, api_key.clone())
                    })
                    .with_max_output_tokens(model_profile.max_output_tokens);
                let mut configured =
                    crate::app::ConfiguredModel::new(provider_id, model_profile, model);
                self.settings.apply(&configured.id, &mut configured.model);
                configured
            })
            .collect::<Vec<_>>();
        self.models = [DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID]
            .into_iter()
            .flat_map(|id| {
                if id == provider_id {
                    replacement.clone()
                } else {
                    previous_models
                        .iter()
                        .filter(|model| model.provider_id == id)
                        .cloned()
                        .collect()
                }
            })
            .collect();
        self.selected_model =
            active_model_index(&self.models, previous_selected_id.as_deref()).unwrap_or(0);
        let selected = self.models[self.selected_model].clone();
        self.model = selected.label();
        self.refresh_idle_runtime_models(cx);
        if let Err(error) = self.settings.set_selected_model(&selected.id) {
            self.notice(format!("Could not save model selection: {error}"));
        }
        if let Some(Modal::Settings(dialog)) = &mut self.modal {
            dialog.model_editor = None;
            dialog.saved_provider = Some(profile.display_name);
        }
        cx.notify();
    }

    pub(crate) fn close_modal(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.modal = None;
        self.input.update(cx, |input, cx| input.focus(window, cx));
        cx.notify();
    }

    pub(crate) fn confirm_rename(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(Modal::RenameSession {
            project_index,
            path,
            input,
        }) = &self.modal
        else {
            return;
        };
        let project_index = *project_index;
        let path = path.clone();
        let title = input.read(cx).value().trim().to_owned();
        if title.is_empty() {
            return;
        }
        self.modal = None;
        self.rename_target_session(project_index, path, title, window, cx);
    }

    pub(crate) fn modal_view(
        &self,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<gpui::AnyElement> {
        let colors = palette(cx);
        let content = match &self.modal {
            Some(Modal::RenameSession { input, .. }) => modal_card("Rename session", colors)
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
            Some(Modal::DeleteSession {
                project_index,
                path,
                title,
            }) => {
                let project_index = *project_index;
                let path = path.clone();
                modal_card("Delete session?", colors)
                    .child(format!("“{title}” will be permanently deleted."))
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
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.close_modal(window, cx)
                                    })),
                            )
                            .child(
                                Button::new("confirm-delete-session")
                                    .label("Delete")
                                    .danger()
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.modal = None;
                                        this.delete_target_session(
                                            project_index,
                                            path.clone(),
                                            window,
                                            cx,
                                        )
                                    })),
                            ),
                    )
                    .into_any_element()
            }
            Some(Modal::RemoveProject(index)) => {
                let index = *index;
                let name = self
                    .project_store
                    .project(index)
                    .map(|project| project.name.clone())
                    .unwrap_or_default();
                modal_card("Remove project?", colors)
                    .child(format!("Remove “{name}” from {}?", crate::APP_NAME))
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
            Some(Modal::Settings(dialog)) => settings_dialog_view(self, dialog, cx, colors),
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
                        && matches!(this.modal, Some(Modal::RenameSession { .. }))
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
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.set_default_allow_all_tools(false, cx)
                        })),
                )
                .child(
                    Button::new("settings-permission-allow")
                        .label("Allow all")
                        .when(allow_all, |button| button.primary())
                        .on_click(
                            cx.listener(|this, _, _, cx| {
                                this.set_default_allow_all_tools(true, cx)
                            }),
                        ),
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

fn settings_nav(
    id: &'static str,
    label: &'static str,
    icon: IconName,
    selected: bool,
    _colors: UiPalette,
) -> Button {
    Button::new(id)
        .child(
            div()
                .flex()
                .items_center()
                .gap_3()
                .child(Icon::new(icon).size_4())
                .child(label),
        )
        .ghost()
        .w_full()
        .justify_start()
        .selected(selected)
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

#[cfg(test)]
mod tests {
    use super::{parse_capacity, parse_optional_output_tokens};

    #[test]
    fn model_capacity_accepts_plain_k_and_m_counts() {
        assert_eq!(parse_capacity("131072"), Some(131_072));
        assert_eq!(parse_capacity("256K"), Some(256_000));
        assert_eq!(parse_capacity("1m"), Some(1_000_000));
        assert_eq!(parse_capacity("0"), None);
        assert_eq!(parse_capacity("large"), None);
    }

    #[test]
    fn max_output_tokens_accepts_blank_or_a_capacity() {
        assert_eq!(parse_optional_output_tokens(""), Some(None));
        assert_eq!(parse_optional_output_tokens("256K"), Some(Some(256_000)));
        assert_eq!(parse_optional_output_tokens("0"), None);
        assert_eq!(parse_optional_output_tokens("large"), None);
    }
}
