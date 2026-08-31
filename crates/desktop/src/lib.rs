#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::unwrap_used
    )
)]

use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::path::PathBuf;

use gpui::{
    App, AppContext, Bounds, Context, InteractiveElement, IntoElement, ParentElement, Render,
    StatefulInteractiveElement, Styled, TitlebarOptions, WindowBackgroundAppearance, WindowBounds,
    WindowOptions, accesskit::Role, div, px, size,
};
use gpui_component::{Root, Theme, ThemeMode};
use kcastle_agent::{Agent, Session};

mod agent_config;
mod app;
mod app_store;
mod application;
#[cfg(test)]
mod architecture_tests;
mod assets;
mod composer;
mod conversation;
mod crash;
mod dialogs;
mod domain;
mod dsh_markdown;
mod layout;
mod platform;
mod project;
mod settings;
mod sidebar;
mod streaming_markdown;
mod trajectory;
mod ui;
mod ui_automation;
mod ui_theme;
mod updater;

use agent_config::{
    DEEPSEEK_PROVIDER_ID, INSTRUCTIONS, OPENAI_PROVIDER_ID, build_model, default_provider_profile,
};
use app::{ConfiguredModel, DesktopApp, DesktopStartup, active_model_index};
use app_store::AppStore;
use assets::DesktopAssets;
use project::ProjectStore;
use settings::{Appearance, ProviderProfile, SettingsStore};

pub(crate) const APP_NAME: &str = "Kcastle";
const DATA_ROOT_ENV: &str = "KCASTLE_DATA_DIR";

fn init_ui(cx: &mut App) {
    gpui_component::init(cx);
    register_syntax_languages();
}

fn register_syntax_languages() {
    let config = gpui_component::highlighter::LanguageConfig::new(
        "haskell",
        tree_sitter_haskell::LANGUAGE.into(),
        vec![],
        tree_sitter_haskell::HIGHLIGHTS_QUERY,
        tree_sitter_haskell::INJECTIONS_QUERY,
        tree_sitter_haskell::LOCALS_QUERY,
    );
    let registry = gpui_component::highlighter::LanguageRegistry::singleton();
    registry.register("haskell", &config);
    registry.register("hs", &config);
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let _runtime = runtime.enter();
    let root = data_root().map_err(|error| error.to_string());
    if let Ok(root) = &root {
        crash::install(root);
    }
    let startup = root
        .as_ref()
        .map_err(Clone::clone)
        .and_then(|root| desktop_startup(root.clone()).map_err(|error| error.to_string()));
    let application = gpui_platform::application().with_assets(DesktopAssets);
    application.on_reopen(move |cx| {
        if cx.windows().is_empty() {
            let startup = root
                .as_ref()
                .map_err(Clone::clone)
                .and_then(|root| desktop_startup(root.clone()).map_err(|error| error.to_string()));
            let result = match startup {
                Ok((startup, appearance)) => open_desktop_window(startup, appearance, cx),
                Err(error) => open_startup_error_window(error, cx),
            };
            if let Err(error) = result {
                eprintln!("failed to reopen desktop window: {error}");
            }
        }
        cx.activate(true);
    });
    application.run(move |cx| {
        init_ui(cx);
        let result = match startup {
            Ok((startup, appearance)) => open_desktop_window(startup, appearance, cx),
            Err(error) => open_startup_error_window(error, cx),
        };
        if let Err(error) = result {
            eprintln!("failed to open desktop window: {error}");
        }
        cx.activate(true);
    });
    Ok(())
}

fn desktop_startup(root: PathBuf) -> Result<(DesktopStartup, Appearance), Box<dyn Error>> {
    let app_store = AppStore::open(root)?;
    let mut settings = SettingsStore::load(app_store.clone())?;
    let mut models = models_from_profiles(settings.provider_profiles());
    for configured in &mut models {
        settings.apply(&configured.id, &mut configured.model);
    }
    let preferred_model = settings.selected_model().and_then(|selected| {
        models
            .iter()
            .find(|model| {
                model.id == selected
                    || format!("{}/{}", model.model.name(), model.model.model()) == selected
            })
            .map(|model| model.id.clone())
    });
    let selected_model = active_model_index(&models, preferred_model.as_deref()).unwrap_or(0);
    if models[selected_model].model.has_api_key()
        && settings.selected_model() != Some(models[selected_model].id.as_str())
    {
        settings.set_selected_model(&models[selected_model].id)?;
    }
    let model = models[selected_model].model.clone();
    let appearance = settings.appearance();
    let (projects, active_project) = ProjectStore::load(app_store, None)?;
    let project = projects
        .project(active_project)
        .ok_or("project store has no active project")?;
    let agent = Agent::new(model, INSTRUCTIONS, Session::memory(), project.path.clone());
    Ok((
        DesktopStartup {
            agent,
            models,
            selected_model,
            project_store: projects,
            active_project,
            settings,
        },
        appearance,
    ))
}

fn open_desktop_window(
    startup: DesktopStartup,
    appearance: Appearance,
    cx: &mut App,
) -> Result<(), Box<dyn Error>> {
    match appearance {
        Appearance::System => Theme::sync_system_appearance(None, cx),
        Appearance::Light => Theme::change(ThemeMode::Light, None, cx),
        Appearance::Dark => Theme::change(ThemeMode::Dark, None, cx),
    }
    cx.open_window(desktop_window_options(cx), move |window, cx| {
        let view = cx.new(|cx| DesktopApp::new(startup, window, cx));
        cx.new(|cx| Root::new(view, window, cx))
    })?;
    Ok(())
}

fn open_startup_error_window(message: String, cx: &mut App) -> Result<(), Box<dyn Error>> {
    Theme::sync_system_appearance(None, cx);
    cx.open_window(desktop_window_options(cx), move |window, cx| {
        let view = cx.new(|_| StartupErrorView { message });
        cx.new(|cx| Root::new(view, window, cx))
    })?;
    Ok(())
}

fn desktop_window_options(cx: &App) -> WindowOptions {
    let bounds = Bounds::centered(None, size(px(1180.0), px(720.0)), cx);
    WindowOptions {
        window_bounds: Some(WindowBounds::Windowed(bounds)),
        window_min_size: Some(size(px(720.0), px(620.0))),
        titlebar: Some(TitlebarOptions {
            title: Some(APP_NAME.into()),
            appears_transparent: true,
            traffic_light_position: None,
        }),
        window_background: WindowBackgroundAppearance::Blurred,
        app_id: Some("dev.kcastle.desktop".into()),
        ..Default::default()
    }
}

struct StartupErrorView {
    message: String,
}

impl Render for StartupErrorView {
    fn render(&mut self, _window: &mut gpui::Window, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = ui_theme::palette(cx);
        div()
            .id("startup-error")
            .role(Role::Alert)
            .aria_label("Kcastle startup error")
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .bg(colors.canvas)
            .text_color(colors.text)
            .child(
                div()
                    .max_w(px(560.0))
                    .p_6()
                    .rounded_lg()
                    .border_1()
                    .border_color(colors.border)
                    .bg(colors.surface)
                    .child(div().text_xl().child("Kcastle couldn't start"))
                    .child(
                        div()
                            .mt_3()
                            .text_sm()
                            .text_color(colors.danger)
                            .child(self.message.clone()),
                    )
                    .child(
                        div()
                            .mt_3()
                            .text_sm()
                            .text_color(colors.muted_text)
                            .child("Close Kcastle and reopen it after fixing the problem."),
                    ),
            )
    }
}

fn models_from_profiles(profiles: &[ProviderProfile]) -> Vec<ConfiguredModel> {
    let mut models = Vec::new();
    for provider_id in [DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID] {
        let profile = profiles
            .iter()
            .find(|profile| profile.provider_id == provider_id)
            .cloned()
            .unwrap_or_else(|| default_provider_profile(provider_id));
        let key = profile
            .api_key()
            .map(str::to_owned)
            .filter(|key| !key.trim().is_empty())
            .unwrap_or_default();
        models.extend(profile.models.iter().cloned().map(|model_profile| {
            let model = build_model(&profile, &model_profile, key.clone());
            ConfiguredModel::new(provider_id, model_profile, model)
        }));
    }
    models
}

fn data_root() -> Result<PathBuf, Box<dyn Error>> {
    resolve_data_root(
        env::var_os(DATA_ROOT_ENV),
        env::var_os("HOME"),
        env::var_os("USERPROFILE"),
    )
    .map_err(Into::into)
}

fn resolve_data_root(
    override_root: Option<OsString>,
    home: Option<OsString>,
    user_profile: Option<OsString>,
) -> Result<PathBuf, &'static str> {
    if let Some(root) = override_root {
        if root.is_empty() {
            return Err("KCASTLE_DATA_DIR cannot be empty");
        }
        return Ok(PathBuf::from(root));
    }
    home.or(user_profile)
        .map(PathBuf::from)
        .map(|home| home.join(".kcastle"))
        .ok_or("cannot locate the home directory")
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::agent_config::{DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID};

    use super::{
        desktop_startup, models_from_profiles, register_syntax_languages, resolve_data_root,
    };

    #[test]
    fn haskell_code_blocks_produce_syntax_styles() {
        use gpui_component::highlighter::{HighlightTheme, SyntaxHighlighter};
        use gpui_component::input::Rope;

        register_syntax_languages();
        let source = "quicksort (p:xs) = quicksort smaller\n  where smaller = filter (< p) xs";
        for language in ["haskell", "hs"] {
            let mut highlighter = SyntaxHighlighter::new(language);
            assert!(highlighter.update(None, &Rope::from(source), None));
            assert!(highlighter.tree().is_some());
            assert!(
                highlighter
                    .styles(&(0..source.len()), HighlightTheme::default_light().as_ref())
                    .iter()
                    .any(|(_, style)| style.color.is_some())
            );
        }
    }

    #[test]
    fn data_root_defaults_to_the_home_directory() {
        assert_eq!(
            resolve_data_root(None, Some(OsString::from("/users/test")), None).unwrap(),
            PathBuf::from("/users/test/.kcastle")
        );
        assert_eq!(
            resolve_data_root(None, None, Some(OsString::from("C:/Users/test"))).unwrap(),
            PathBuf::from("C:/Users/test/.kcastle")
        );
    }

    #[test]
    fn data_root_override_is_verbatim_and_must_not_be_empty() {
        assert_eq!(
            resolve_data_root(
                Some(OsString::from("/tmp/kcastle-acceptance")),
                Some(OsString::from("/users/test")),
                None,
            )
            .unwrap(),
            PathBuf::from("/tmp/kcastle-acceptance")
        );
        assert_eq!(
            resolve_data_root(
                Some(OsString::from("relative/acceptance")),
                Some(OsString::from("/users/test")),
                None,
            )
            .unwrap(),
            PathBuf::from("relative/acceptance")
        );
        assert_eq!(
            resolve_data_root(Some(OsString::new()), None, None),
            Err("KCASTLE_DATA_DIR cannot be empty")
        );
    }

    #[test]
    fn desktop_startup_can_be_rebuilt_after_last_window_closes() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("kcastle-reopen-test-{suffix}"));
        let (first, first_appearance) = desktop_startup(root.clone()).unwrap();
        let model_count = first.models.len();
        drop(first);

        let (second, second_appearance) = desktop_startup(root.clone()).unwrap();

        assert_eq!(second.models.len(), model_count);
        assert_eq!(second_appearance, first_appearance);
        drop(second);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn models_are_available_only_from_desktop_configuration() {
        let models = models_from_profiles(&[]);

        assert!(
            models.iter().any(
                |model| model.provider_id == DEEPSEEK_PROVIDER_ID && !model.model.has_api_key()
            )
        );
        assert!(
            models
                .iter()
                .any(|model| model.provider_id == OPENAI_PROVIDER_ID && !model.model.has_api_key())
        );
    }
}
