use std::env;
use std::error::Error;
use std::path::PathBuf;

use gpui::{
    App, AppContext, Application, Bounds, KeyBinding, TitlebarOptions, WindowBackgroundAppearance,
    WindowBounds, WindowOptions, px, size,
};
use gpui_component::input::Enter;
use gpui_component::{Root, Theme, ThemeMode};
use kcastle_agent::{
    Agent, DEEPSEEK_MODEL_PRESETS, DEEPSEEK_PROVIDER_ID, Model, OPENAI_MODEL_PRESETS,
    OPENAI_PROVIDER_ID, Session,
};

mod app;
mod application;
#[cfg(test)]
mod architecture_tests;
mod assets;
mod composer;
mod conversation;
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
mod ui_theme;
mod updater;

use app::{ConfiguredModel, DesktopApp, DesktopStartup};
use assets::DesktopAssets;
use project::ProjectStore;
use settings::{Appearance, ProviderModel, ProviderProfile, SettingsStore};

pub(crate) const APP_NAME: &str = "Kcastle";
pub(crate) const INSTRUCTIONS: &str = "You are Kcastle, a concise coding agent. Use the shell tool when it helps. Inspect before editing, report tool errors honestly, and stop when the task is complete.";

fn init_ui(cx: &mut App) {
    gpui_component::init(cx);
    cx.bind_keys([KeyBinding::new(
        "shift-enter",
        Enter { secondary: true },
        Some("Input"),
    )]);
}

pub fn run() -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let _runtime = runtime.enter();
    let root = home_dir()?.join(".kcastle");
    let (startup, appearance) = desktop_startup(root.clone())?;
    let application = Application::new().with_assets(DesktopAssets);
    application.on_reopen(move |cx| {
        if cx.windows().is_empty()
            && let Err(error) = desktop_startup(root.clone())
                .and_then(|(startup, appearance)| open_desktop_window(startup, appearance, cx))
        {
            eprintln!("failed to reopen desktop window: {error}");
        }
        cx.activate(true);
    });
    application.run(move |cx| {
        init_ui(cx);
        open_desktop_window(startup, appearance, cx).expect("failed to open desktop window");
        cx.activate(true);
    });
    Ok(())
}

fn desktop_startup(root: PathBuf) -> Result<(DesktopStartup, Appearance), Box<dyn Error>> {
    let mut settings = SettingsStore::load(root.clone())?;
    let mut models = models_from_profiles(settings.provider_profiles());
    for configured in &mut models {
        settings.apply(&configured.id, &mut configured.model);
    }
    let selected_model = settings
        .selected_model()
        .and_then(|selected| {
            models.iter().position(|model| {
                model.id == selected
                    || format!("{}/{}", model.model.name(), model.model.model()) == selected
            })
        })
        .unwrap_or(0);
    if settings.selected_model() != Some(models[selected_model].id.as_str()) {
        settings.set_selected_model(&models[selected_model].id)?;
    }
    let model = models[selected_model].model.clone();
    let appearance = settings.appearance();
    let (projects, active_project) = ProjectStore::load(root, None)?;
    let project = projects
        .project(active_project)
        .expect("active project should exist");
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
    let bounds = Bounds::centered(None, size(px(1180.0), px(720.0)), cx);
    cx.open_window(
        WindowOptions {
            window_bounds: Some(WindowBounds::Windowed(bounds)),
            window_min_size: Some(size(px(900.0), px(620.0))),
            titlebar: Some(TitlebarOptions {
                title: Some(APP_NAME.into()),
                appears_transparent: true,
                traffic_light_position: None,
            }),
            window_background: WindowBackgroundAppearance::Blurred,
            app_id: Some("dev.kcastle.desktop".into()),
            ..Default::default()
        },
        move |window, cx| {
            let view = cx.new(|cx| DesktopApp::new(startup, window, cx));
            cx.new(|cx| Root::new(view, window, cx))
        },
    )?;
    Ok(())
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

pub(crate) fn default_provider_profile(provider_id: &str) -> ProviderProfile {
    match provider_id {
        DEEPSEEK_PROVIDER_ID => ProviderProfile::new(
            DEEPSEEK_PROVIDER_ID,
            "DeepSeek",
            "https://api.deepseek.com",
            DEEPSEEK_MODEL_PRESETS
                .iter()
                .map(|preset| {
                    ProviderModel::new(preset.id, preset.display_name, preset.context_window, None)
                })
                .collect(),
        ),
        OPENAI_PROVIDER_ID => ProviderProfile::new(
            OPENAI_PROVIDER_ID,
            "OpenAI",
            "https://api.openai.com/v1",
            OPENAI_MODEL_PRESETS
                .iter()
                .map(|preset| {
                    ProviderModel::new(preset.id, preset.display_name, preset.context_window, None)
                })
                .collect(),
        ),
        _ => unreachable!("unsupported provider: {provider_id}"),
    }
}

pub(crate) fn build_model(
    provider: &ProviderProfile,
    profile: &ProviderModel,
    api_key: String,
) -> Model {
    Model::new(
        provider.display_name.clone(),
        api_key,
        provider.api_base.clone(),
        profile.model_id.clone(),
        profile.context_window,
    )
    .with_max_output_tokens(profile.max_output_tokens)
    .with_provider_reasoning(&provider.provider_id)
}

fn home_dir() -> Result<PathBuf, Box<dyn Error>> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| "cannot locate the home directory".into())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;
    use std::rc::Rc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use gpui::{
        AppContext, Context, Entity, IntoElement, Render, Subscription, TestAppContext, Window,
    };
    use gpui_component::Root;
    use gpui_component::input::{Input, InputEvent, InputState};
    use kcastle_agent::{DEEPSEEK_PROVIDER_ID, OPENAI_PROVIDER_ID};

    use super::{desktop_startup, init_ui, models_from_profiles};

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

    struct InputHarness {
        input: Entity<InputState>,
        enter_events: Vec<bool>,
        _subscription: Subscription,
    }

    impl Render for InputHarness {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            Input::new(&self.input)
        }
    }

    #[gpui::test]
    fn shift_enter_inserts_a_newline_without_submitting(cx: &mut TestAppContext) {
        cx.update(init_ui);
        let harness = Rc::new(RefCell::new(None));
        let test_harness = harness.clone();
        let (_, cx) = cx.add_window_view(|window, cx| {
            let harness = cx.new(|cx| {
                let input = cx.new(|cx| InputState::new(window, cx).auto_grow(1, 14));
                input.update(cx, |input, cx| input.focus(window, cx));
                let subscription = cx.subscribe(
                    &input,
                    |this: &mut InputHarness, _, event: &InputEvent, _| {
                        if let InputEvent::PressEnter { secondary } = event {
                            this.enter_events.push(*secondary);
                        }
                    },
                );
                InputHarness {
                    input,
                    enter_events: Vec::new(),
                    _subscription: subscription,
                }
            });
            test_harness.replace(Some(harness.clone()));
            Root::new(harness, window, cx)
        });
        cx.refresh().unwrap();

        cx.simulate_input("first line");
        cx.simulate_keystrokes("shift-enter");
        cx.simulate_input("second line");

        let harness = harness.borrow().clone().unwrap();
        let input = cx.read_entity(&harness, |view, _| view.input.clone());
        let value = cx.read_entity(&input, |input: &InputState, _| input.value());
        let enter_events = cx.read_entity(&harness, |view, _| view.enter_events.clone());
        assert_eq!(value.as_ref(), "first line\nsecond line");
        assert_eq!(enter_events, [true]);
    }
}
