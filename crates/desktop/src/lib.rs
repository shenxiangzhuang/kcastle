use std::env;
use std::error::Error;
use std::path::PathBuf;

use gpui::{
    AppContext, Application, Bounds, TitlebarOptions, WindowBackgroundAppearance, WindowBounds,
    WindowOptions, px, size,
};
use gpui_component::{Root, Theme, ThemeMode};
use kcastle_agent::{Agent, Model, ReasoningEffort, Session};

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

use app::{ConfiguredModel, DesktopApp, DesktopStartup};
use assets::DesktopAssets;
use project::ProjectStore;
use settings::{Appearance, SettingsStore};

const INSTRUCTIONS: &str = "You are K, a concise coding agent. Use the shell tool when it helps. Inspect before editing, report tool errors honestly, and stop when the task is complete.";
const DEEPSEEK_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::None,
    ReasoningEffort::Low,
    ReasoningEffort::High,
];
const OPENAI_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::None,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::Xhigh,
];

pub fn run() -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let _runtime = runtime.enter();
    let cwd = env::current_dir()?;
    let root = home_dir()?.join(".kcastle");
    let mut settings = SettingsStore::load(root.clone())?;
    let mut models = models_from_env()?;
    for configured in &mut models {
        settings.apply(&configured.id, &mut configured.model);
    }
    let selected_model = settings
        .selected_model()
        .and_then(|selected| models.iter().position(|model| model.id == selected))
        .unwrap_or(0);
    if settings.selected_model() != Some(models[selected_model].id.as_str()) {
        settings.set_selected_model(&models[selected_model].id)?;
    }
    let model = models[selected_model].model.clone();
    let appearance = settings.appearance();
    let (projects, active_project) = ProjectStore::load(root, cwd)?;
    let project = projects
        .project(active_project)
        .expect("active project should exist");
    let agent = Agent::new(model, INSTRUCTIONS, Session::memory(), project.path.clone());

    Application::new()
        .with_assets(DesktopAssets)
        .run(move |cx| {
            gpui_component::init(cx);
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
                        title: Some("K Castle".into()),
                        appears_transparent: true,
                        traffic_light_position: None,
                    }),
                    window_background: WindowBackgroundAppearance::Blurred,
                    app_id: Some("dev.kcastle.desktop".into()),
                    ..Default::default()
                },
                |window, cx| {
                    let view = cx.new(|cx| {
                        DesktopApp::new(
                            DesktopStartup {
                                agent,
                                models,
                                selected_model,
                                project_store: projects,
                                active_project,
                                settings,
                            },
                            window,
                            cx,
                        )
                    });
                    cx.new(|cx| Root::new(view, window, cx))
                },
            )
            .expect("failed to open desktop window");
            cx.activate(true);
        });
    Ok(())
}

fn models_from_env() -> Result<Vec<ConfiguredModel>, Box<dyn Error>> {
    let mut models = Vec::new();
    if let Ok(key) = env::var("DEEPSEEK_API_KEY")
        && !key.trim().is_empty()
    {
        models.push(ConfiguredModel::new(
            Model::new(
                "DeepSeek",
                key,
                "https://api.deepseek.com",
                "deepseek-v4-flash",
                1_000_000,
            )
            .with_reasoning(DEEPSEEK_REASONING_EFFORTS, ReasoningEffort::High),
        ));
    }
    if let Ok(key) = env::var("OPENAI_API_KEY")
        && !key.trim().is_empty()
    {
        models.push(ConfiguredModel::new(
            Model::new(
                "OpenAI",
                key,
                "https://api.openai.com/v1",
                "gpt-5.6-sol",
                1_050_000,
            )
            .with_reasoning(OPENAI_REASONING_EFFORTS, ReasoningEffort::Medium),
        ));
    }
    if models.is_empty() {
        Err("set DEEPSEEK_API_KEY or OPENAI_API_KEY".into())
    } else {
        Ok(models)
    }
}

fn home_dir() -> Result<PathBuf, Box<dyn Error>> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| "cannot locate the home directory".into())
}
