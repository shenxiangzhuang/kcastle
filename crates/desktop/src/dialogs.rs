use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use gpui::{
    App, AppContext, Context, Entity, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder, px, rgb,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::{Input, InputState};
use gpui_component::select::{Select, SelectEvent, SelectState};
use gpui_component::setting::{
    SelectIndex, SettingField, SettingGroup, SettingItem, SettingPage, Settings,
};
use gpui_component::{Disableable, Icon, IconName, IndexPath, Sizable};
use kcastle_agent::SessionInfo;

use crate::agent_config::{DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID};
use crate::app::{ConfiguredModel, DesktopApp, active_model_index};
use crate::assets::DesktopIconName;
use crate::domain::Action;
use crate::settings::{Appearance, EnterBehavior, ProviderModel, ProviderProfile};
use crate::ui_theme::{UiPalette, palette};

pub(crate) enum Modal {
    RenameSession {
        project_index: usize,
        path: PathBuf,
        input: Entity<InputState>,
    },
    DeleteArchivedSession {
        project_index: usize,
        session: SessionInfo,
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
    let view = cx.entity();
    let project = display_path(&app.core.workspace.cwd);
    let general_group = SettingGroup::new()
        .item(
            SettingItem::new(
                "Allow all tools",
                SettingField::switch(
                    {
                        let view = view.clone();
                        move |cx: &App| view.read(cx).settings.allow_all_tools()
                    },
                    {
                        let view = view.clone();
                        move |allow, cx: &mut App| {
                            view.update(cx, |app, cx| app.set_default_allow_all_tools(allow, cx));
                        }
                    },
                ),
            )
            .description("Skip approval prompts for shell and tool calls."),
        )
        .item(
            SettingItem::new(
                "Project",
                SettingField::render(move |_, _, cx| {
                    div()
                        .max_w(px(280.0))
                        .truncate()
                        .text_sm()
                        .text_color(palette(cx).muted_text)
                        .child(project.clone())
                }),
            )
            .description("The working directory for new sessions."),
        )
        .item(
            SettingItem::new(
                "Appearance",
                SettingField::render({
                    let view = view.clone();
                    move |options, _, cx| appearance_setting_field(&view, options, cx)
                }),
            )
            .description("Use the system appearance or choose a fixed theme."),
        )
        .item(
            SettingItem::new(
                "Reduce motion",
                SettingField::switch(
                    {
                        let view = view.clone();
                        move |cx: &App| view.read(cx).settings.reduce_motion()
                    },
                    {
                        let view = view.clone();
                        move |reduce, cx: &mut App| {
                            view.update(cx, |app, cx| app.set_reduce_motion(reduce, cx));
                        }
                    },
                ),
            )
            .description("Reduce non-essential interface animation."),
        )
        .item(
            SettingItem::new(
                "Enter while busy",
                SettingField::dropdown(
                    vec![
                        ("steer".into(), "Steer".into()),
                        ("queue".into(), "Queue".into()),
                    ],
                    {
                        let view = view.clone();
                        move |cx: &App| match view.read(cx).settings.enter_behavior() {
                            EnterBehavior::Steer => SharedString::from("steer"),
                            EnterBehavior::Queue => SharedString::from("queue"),
                        }
                    },
                    {
                        let view = view.clone();
                        move |value, cx: &mut App| {
                            let behavior = if value == "queue" {
                                EnterBehavior::Queue
                            } else {
                                EnterBehavior::Steer
                            };
                            view.update(cx, |app, cx| app.set_enter_behavior(behavior, cx));
                        }
                    },
                ),
            )
            .description("Steer the active turn or queue a follow-up after it settles."),
        );
    let models_group = SettingGroup::new().item(
        SettingItem::render({
            let view = view.clone();
            move |_, _, cx| {
                let app = view.read(cx);
                let Some(Modal::Settings(dialog)) = &app.modal else {
                    return div().into_any_element();
                };
                models_settings_view(app, dialog, &view, palette(cx))
            }
        })
        .keywords(["API", "provider", "model", "DeepSeek", "OpenAI"]),
    );
    let general_page =
        settings_page("General", IconName::Settings2, view.clone()).group(general_group);
    let models_page = settings_page("Models", IconName::Bot, view.clone()).group(models_group);
    let archived_page = SettingPage::new("Archived Sessions")
        .icon(Icon::new(DesktopIconName::Archive))
        .resettable(false)
        .title_suffix({
            let view = view.clone();
            move |_, _| settings_close_button(view.clone())
        })
        .group(
            SettingGroup::new().item(
                SettingItem::render({
                    let view = view.clone();
                    move |_, _, cx| archived_sessions_view(view.read(cx), &view, palette(cx))
                })
                .keywords(["archive", "session", "restore", "delete"]),
            ),
        );
    let about_page = settings_page("About", IconName::Info, view.clone()).group(
        SettingGroup::new()
            .item(SettingItem::new(
                "Version",
                SettingField::render(|_, _, cx| {
                    div()
                        .text_sm()
                        .text_color(palette(cx).muted_text)
                        .child(concat!("v", env!("CARGO_PKG_VERSION")))
                }),
            ))
            .item(
                SettingItem::new(
                    "GitHub repository",
                    SettingField::render(|options, _, _| {
                        Button::new("open-repository")
                            .icon(IconName::Github)
                            .label("Open")
                            .outline()
                            .with_size(options.size())
                            .on_click(|_, _, cx| cx.open_url(env!("CARGO_PKG_REPOSITORY")))
                    }),
                )
                .description("Source code, issues, and project documentation."),
            )
            .item(
                SettingItem::new(
                    "Releases",
                    SettingField::render(|options, _, _| {
                        Button::new("open-releases")
                            .label("Open")
                            .outline()
                            .with_size(options.size())
                            .on_click(|_, _, cx| {
                                cx.open_url(concat!(env!("CARGO_PKG_REPOSITORY"), "/releases"))
                            })
                    }),
                )
                .description("Download installers and view release notes."),
            ),
    );
    div()
        .flex()
        .w(px(800.0))
        .h(px(570.0))
        .rounded(px(24.0))
        .bg(colors.surface)
        .shadow_xl()
        .overflow_hidden()
        .child(
            Settings::new(("settings", dialog.id))
                .sidebar_width(px(188.0))
                .default_selected_index(dialog.initial_page.select_index())
                .pages([general_page, models_page, archived_page, about_page]),
        )
        .into_any_element()
}

fn settings_page(title: &'static str, icon: IconName, view: Entity<DesktopApp>) -> SettingPage {
    SettingPage::new(title)
        .icon(Icon::new(icon))
        .resettable(false)
        .title_suffix(move |_, _| settings_close_button(view.clone()))
}

fn settings_close_button(view: Entity<DesktopApp>) -> Button {
    Button::new("close-settings")
        .icon(IconName::Close)
        .ghost()
        .compact()
        .tooltip("Close")
        .on_click(move |_, window, cx| {
            view.update(cx, |app, cx| app.close_modal(window, cx));
        })
}

fn archived_sessions_view(
    app: &DesktopApp,
    view: &Entity<DesktopApp>,
    colors: UiPalette,
) -> gpui::AnyElement {
    let mut projects = Vec::new();
    for (project_index, project) in app.project_store.projects().iter().enumerate() {
        let sessions = app
            .project_archived_sessions
            .get(&project.sessions_dir)
            .cloned()
            .unwrap_or_default();
        if sessions.is_empty() {
            continue;
        }
        let count = sessions.len();
        let rows = sessions
            .into_iter()
            .enumerate()
            .map(|(session_index, session)| {
                let restore_view = view.clone();
                let restore_session = session.clone();
                let delete_view = view.clone();
                let delete_session = session.clone();
                div()
                    .flex()
                    .items_center()
                    .h(px(40.0))
                    .px_2()
                    .gap_2()
                    .border_b_1()
                    .border_color(colors.border)
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .truncate()
                            .text_sm()
                            .child(session.title),
                    )
                    .child(
                        Button::new(SharedString::from(format!(
                            "restore-archived-{project_index}-{session_index}"
                        )))
                        .icon(IconName::Undo2)
                        .ghost()
                        .small()
                        .tooltip("Restore session")
                        .on_click(move |_, _window, cx| {
                            restore_view.update(cx, |app, cx| {
                                app.restore_archived_session(
                                    project_index,
                                    restore_session.clone(),
                                    cx,
                                )
                            });
                        }),
                    )
                    .child(
                        Button::new(SharedString::from(format!(
                            "delete-archived-{project_index}-{session_index}"
                        )))
                        .icon(IconName::Delete)
                        .ghost()
                        .small()
                        .text_color(colors.danger)
                        .tooltip("Delete permanently")
                        .on_click(move |_, window, cx| {
                            delete_view.update(cx, |app, cx| {
                                app.open_delete_archived_session_dialog(
                                    project_index,
                                    delete_session.clone(),
                                    window,
                                    cx,
                                )
                            });
                        }),
                    )
            });
        projects.push(
            div()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .flex()
                        .items_center()
                        .gap_2()
                        .text_sm()
                        .font_weight(gpui::FontWeight::MEDIUM)
                        .child(Icon::new(IconName::Folder).size_4())
                        .child(project.name.clone())
                        .child(
                            div()
                                .text_xs()
                                .text_color(colors.muted_text)
                                .child(count.to_string()),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .overflow_hidden()
                        .rounded(px(8.0))
                        .border_1()
                        .border_color(colors.border)
                        .children(rows),
                ),
        );
    }

    if projects.is_empty() {
        return div()
            .flex()
            .flex_col()
            .items_center()
            .justify_center()
            .gap_2()
            .py_8()
            .text_color(colors.muted_text)
            .child(Icon::new(DesktopIconName::Archive).size_6())
            .child("No archived sessions")
            .into_any_element();
    }

    div()
        .flex()
        .flex_col()
        .w_full()
        .gap_4()
        .children(projects)
        .into_any_element()
}

fn appearance_setting_field(
    view: &Entity<DesktopApp>,
    options: &gpui_component::setting::RenderOptions,
    cx: &App,
) -> gpui::Div {
    let appearance = view.read(cx).settings.appearance();
    div().flex().items_center().gap_1().children(
        [
            ("appearance-system", "System", Appearance::System),
            ("appearance-light", "Light", Appearance::Light),
            ("appearance-dark", "Dark", Appearance::Dark),
        ]
        .map(|(id, label, value)| {
            let view = view.clone();
            Button::new(id)
                .label(label)
                .compact()
                .with_size(options.size())
                .when(appearance == value, |button| button.primary())
                .on_click(move |_, window, cx| {
                    view.update(cx, |app, cx| app.set_appearance(value, window, cx));
                })
        }),
    )
}

fn models_settings_view(
    app: &DesktopApp,
    dialog: &SettingsDialog,
    view: &Entity<DesktopApp>,
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
                                    .on_click({
                                        let view = view.clone();
                                        move |_, window, cx| {
                                            view.update(cx, |app, cx| {
                                                if is_open {
                                                    app.cancel_model_editor(cx);
                                                } else {
                                                    app.edit_provider(provider_id, window, cx);
                                                }
                                            });
                                        }
                                    }),
                            ),
                    );
            if is_open && let Some(editor) = &dialog.model_editor {
                row = row.child(model_editor_view(editor, view, colors));
            }
            row.into_any_element()
        })
        .collect::<Vec<_>>();
    let addable = addable_provider_ids(&app.models).collect::<Vec<_>>();
    let standalone_editor = dialog
        .model_editor
        .as_ref()
        .filter(|editor| !provider_configured(app, editor.provider_id));
    let adding_provider = standalone_editor.is_some();
    let add_provider_view = view.clone();

    div()
        .flex()
        .flex_col()
        .gap_3()
        .max_w(px(720.0))
        .child(
            div()
                .text_sm()
                .text_color(colors.muted_text)
                .child("Enter your API keys to use models from the following providers."),
        )
        .children(dialog.saved_provider.as_ref().map(|provider| {
            div()
                .text_xs()
                .text_color(rgb(0x36a763))
                .child(format!("Saved {provider}."))
        }))
        .child(div().flex().flex_col().gap_2().mt_3().children(rows))
        .children(standalone_editor.map(|editor| model_editor_view(editor, view, colors)))
        .when(!adding_provider, move |container| {
            let app_view = add_provider_view.clone();
            container.child(
                Button::new("add-provider")
                    .icon(IconName::Plus)
                    .label("Add provider")
                    .w_full()
                    .disabled(busy || addable.is_empty())
                    .on_click(move |_, window, cx| {
                        app_view.update(cx, |app, cx| app.add_provider(window, cx));
                    }),
            )
        })
        .into_any_element()
}

fn model_editor_view(
    editor: &ModelEditor,
    view: &Entity<DesktopApp>,
    colors: UiPalette,
) -> gpui::AnyElement {
    let profile = crate::default_provider_profile(editor.provider_id);
    let provider_field = match &editor.provider_select {
        Some(provider_select) => div()
            .flex()
            .flex_col()
            .gap_1()
            .child(
                div()
                    .text_xs()
                    .font_weight(gpui::FontWeight::MEDIUM)
                    .child("Provider"),
            )
            .child(Select::new(provider_select).small().w_full())
            .into_any_element(),
        None => div()
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
            )
            .into_any_element(),
    };
    let model_rows = editor
        .models
        .iter()
        .enumerate()
        .map(|(index, row)| model_editor_row_view(index, row, editor.models.len(), view, colors))
        .collect::<Vec<_>>();
    div()
        .flex()
        .flex_col()
        .gap_3()
        .p_4()
        .rounded(px(12.0))
        .bg(colors.subtle)
        .child(provider_field)
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
                    .on_click({
                        let view = view.clone();
                        move |_, _, cx| {
                            view.update(cx, |app, cx| app.toggle_model_advanced(cx));
                        }
                    }),
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
                                .on_click({
                                    let view = view.clone();
                                    move |_, window, cx| {
                                        view.update(cx, |app, cx| {
                                            app.add_provider_model(window, cx)
                                        });
                                    }
                                }),
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
                        .on_click({
                            let view = view.clone();
                            move |_, _, cx| {
                                view.update(cx, |app, cx| app.cancel_model_editor(cx));
                            }
                        }),
                )
                .child(
                    Button::new("save-model-editor")
                        .label("Save")
                        .primary()
                        .on_click({
                            let view = view.clone();
                            move |_, _, cx| {
                                view.update(cx, |app, cx| app.save_model_editor(cx));
                            }
                        }),
                ),
        )
        .into_any_element()
}

fn model_editor_row_view(
    index: usize,
    row: &ModelEditorRow,
    model_count: usize,
    view: &Entity<DesktopApp>,
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
                        .on_click({
                            let view = view.clone();
                            move |_, _, cx| {
                                view.update(cx, |app, cx| app.remove_provider_model(index, cx));
                            }
                        }),
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
        provider_select: None,
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

fn addable_provider_ids(models: &[ConfiguredModel]) -> impl Iterator<Item = &'static str> + '_ {
    [DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID]
        .into_iter()
        .filter(|provider_id| {
            !models
                .iter()
                .any(|model| model.provider_id == *provider_id && model.model.has_api_key())
        })
}

fn provider_id_for_display_name(display_name: &str) -> Option<&'static str> {
    [DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID]
        .into_iter()
        .find(|provider_id| {
            crate::default_provider_profile(provider_id).display_name == display_name
        })
}

fn provider_profile(app: &DesktopApp, provider_id: &'static str) -> ProviderProfile {
    app.settings
        .provider_profiles()
        .iter()
        .find(|profile| profile.provider_id == provider_id)
        .cloned()
        .unwrap_or_else(|| crate::default_provider_profile(provider_id))
}

static NEXT_SETTINGS_DIALOG_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Copy)]
enum SettingsPage {
    General,
    Models,
    Archives,
}

pub(crate) struct SettingsDialog {
    id: usize,
    initial_page: SettingsPage,
    model_editor: Option<ModelEditor>,
    saved_provider: Option<String>,
}

impl SettingsPage {
    fn select_index(self) -> SelectIndex {
        SelectIndex {
            page_ix: match self {
                Self::General => 0,
                Self::Models => 1,
                Self::Archives => 2,
            },
            group_ix: None,
        }
    }
}

impl SettingsDialog {
    fn new(initial_page: SettingsPage) -> Self {
        Self {
            id: NEXT_SETTINGS_DIALOG_ID.fetch_add(1, Ordering::Relaxed),
            initial_page,
            model_editor: None,
            saved_provider: None,
        }
    }
}

struct ModelEditor {
    provider_id: &'static str,
    provider_select: Option<Entity<SelectState<Vec<String>>>>,
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

    pub(crate) fn open_delete_archived_session_dialog(
        &mut self,
        project_index: usize,
        session: SessionInfo,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.modal = Some(Modal::DeleteArchivedSession {
            project_index,
            session,
        });
        self.modal_focus.focus(window, cx);
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
            self.modal_focus.focus(window, cx);
            cx.notify();
        }
    }

    pub(crate) fn open_settings_dialog(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(Action::CloseTransientOverlays, window, cx);
        self.modal = Some(Modal::Settings(Box::new(SettingsDialog::new(
            SettingsPage::General,
        ))));
        self.modal_focus.focus(window, cx);
        cx.notify();
    }

    pub(crate) fn open_model_settings_dialog(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.dispatch(Action::CloseTransientOverlays, window, cx);
        self.modal = Some(Modal::Settings(Box::new(SettingsDialog::new(
            SettingsPage::Models,
        ))));
        self.modal_focus.focus(window, cx);
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

    pub(crate) fn add_provider(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let provider_ids = addable_provider_ids(&self.models).collect::<Vec<_>>();
        let Some(&provider_id) = provider_ids.first() else {
            return;
        };
        let provider_names = provider_ids
            .into_iter()
            .map(|provider_id| crate::default_provider_profile(provider_id).display_name)
            .collect::<Vec<_>>();
        let provider_select =
            cx.new(|cx| SelectState::new(provider_names, Some(IndexPath::default()), window, cx));
        cx.subscribe_in(
            &provider_select,
            window,
            |this, _, event: &SelectEvent<Vec<String>>, window, cx| {
                let SelectEvent::Confirm(Some(display_name)) = event else {
                    return;
                };
                if let Some(provider_id) = provider_id_for_display_name(display_name) {
                    this.select_add_provider(provider_id, window, cx);
                }
            },
        )
        .detach();

        let mut editor = model_editor(
            provider_id,
            provider_profile(self, provider_id),
            false,
            window,
            cx,
        );
        editor.provider_select = Some(provider_select);
        self.set_model_editor(editor, window, cx);
    }

    fn select_add_provider(
        &mut self,
        provider_id: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if provider_configured(self, provider_id) {
            return;
        }
        if matches!(
            &self.modal,
            Some(Modal::Settings(dialog))
                if dialog.model_editor.as_ref().is_some_and(|editor| editor.provider_id == provider_id)
        ) {
            return;
        }
        let mut replacement = model_editor(
            provider_id,
            provider_profile(self, provider_id),
            false,
            window,
            cx,
        );
        let Some(Modal::Settings(dialog)) = &mut self.modal else {
            return;
        };
        let Some(editor) = dialog.model_editor.take() else {
            return;
        };
        replacement.provider_select = editor.provider_select;
        dialog.model_editor = Some(replacement);
        cx.notify();
    }

    fn set_model_editor(
        &mut self,
        editor: ModelEditor,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let focus = editor.api_key.clone();
        if let Some(Modal::Settings(dialog)) = &mut self.modal {
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
            Some(Modal::DeleteArchivedSession {
                project_index,
                session,
            }) => {
                let project_index = *project_index;
                let session = session.clone();
                let title = session.title.clone();
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
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.modal = Some(Modal::Settings(Box::new(
                                            SettingsDialog::new(SettingsPage::Archives),
                                        )));
                                        cx.notify();
                                    })),
                            )
                            .child(
                                Button::new("confirm-delete-session")
                                    .label("Delete")
                                    .danger()
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        this.modal = Some(Modal::Settings(Box::new(
                                            SettingsDialog::new(SettingsPage::Archives),
                                        )));
                                        this.delete_archived_session(
                                            project_index,
                                            session.clone(),
                                            cx,
                                        );
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
    use kcastle_agent::Model;

    use crate::agent_config::{DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID};

    use super::{
        SettingsDialog, SettingsPage, addable_provider_ids, parse_capacity,
        parse_optional_output_tokens,
    };
    use crate::app::ConfiguredModel;
    use crate::settings::ProviderModel;

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

    #[test]
    fn add_provider_only_offers_unconfigured_providers() {
        let models = [ConfiguredModel::new(
            DEEPSEEK_PROVIDER_ID,
            ProviderModel::new("deepseek-test", "DeepSeek Test", 10_000, None),
            Model::new(
                "DeepSeek",
                "secret",
                "http://localhost",
                "deepseek-test",
                10_000,
            ),
        )];

        assert_eq!(
            addable_provider_ids(&models).collect::<Vec<_>>(),
            vec![OPENAI_PROVIDER_ID]
        );
    }

    #[test]
    fn settings_openings_have_isolated_component_state() {
        let general = SettingsDialog::new(SettingsPage::General);
        let models = SettingsDialog::new(SettingsPage::Models);
        let archives = SettingsDialog::new(SettingsPage::Archives);

        assert_ne!(general.id, models.id);
        assert_ne!(models.id, archives.id);
        assert!(matches!(general.initial_page, SettingsPage::General));
        assert!(matches!(models.initial_page, SettingsPage::Models));
        assert_eq!(general.initial_page.select_index().page_ix, 0);
        assert_eq!(general.initial_page.select_index().group_ix, None);
        assert_eq!(models.initial_page.select_index().page_ix, 1);
        assert_eq!(models.initial_page.select_index().group_ix, None);
        assert!(matches!(archives.initial_page, SettingsPage::Archives));
        assert_eq!(archives.initial_page.select_index().page_ix, 2);
        assert_eq!(archives.initial_page.select_index().group_ix, None);
    }
}
