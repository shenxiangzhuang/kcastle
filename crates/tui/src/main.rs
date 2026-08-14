mod app;

use std::env;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use app::{App, UiAction};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event,
    EventStream, KeyEventKind,
};
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use futures_util::StreamExt;
use kcastle_agent::{ActiveAgent, Agent, AgentEvent, Model, ReasoningEffort, Session, SessionInfo};
use ratatui::text::Text;
use ratatui::widgets::{Paragraph, Widget, Wrap};
use ratatui::{TerminalOptions, Viewport};

const INSTRUCTIONS: &str = "You are K, a concise coding agent. Use the shell tool when it helps. Inspect before editing, report tool errors honestly, and stop when the task is complete.";
const INLINE_VIEWPORT_HEIGHT: u16 = 12;
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

const HELP: &str = "K in Castle — native agent harness\n\nUSAGE:\n    kcastle [--prompt TEXT] [--allow-tools]\n\nOPTIONS:\n    -h, --help       Show help\n    -V, --version    Show version\n    -p, --prompt     Run one non-interactive prompt\n        --allow-tools  Allow tools in non-interactive mode\n\nTUI COMMANDS:\n    /session         Manage saved sessions\n    /model           Select model and reasoning level\n    /compact [focus] Compact active context\n    /permissions     Toggle ask / allow all\n    /tool            Browse tool calls\n    /queue MESSAGE   Run after the active task settles\n    /help            Show commands\n    /exit            Exit\n";

enum Command {
    Tui,
    Prompt { prompt: String, allow_tools: bool },
    Help,
    Version,
}

struct Runtime {
    agent: Option<Agent>,
    active: Option<ActiveAgent>,
    models: Vec<Model>,
    selected_model: usize,
    sessions_dir: PathBuf,
    session_path: PathBuf,
    compacting: bool,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    match parse_args()? {
        Command::Help => {
            print!("{HELP}");
            return Ok(());
        }
        Command::Version => {
            println!("kcastle {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        command => {
            let models = models_from_env()?;
            let sessions_dir = home_dir()?.join(".kcastle/sessions");
            let cwd = env::current_dir()?;
            match command {
                Command::Prompt {
                    prompt,
                    allow_tools,
                } => {
                    let session = Session::create(&sessions_dir).await?;
                    let agent = Agent::new(models[0].clone(), INSTRUCTIONS, session, cwd.clone());
                    run_prompt(agent, prompt, allow_tools).await?;
                }
                Command::Tui => {
                    let agent = Agent::new(
                        models[0].clone(),
                        INSTRUCTIONS,
                        Session::memory(),
                        cwd.clone(),
                    );
                    run_tui(agent, models, sessions_dir, cwd).await?;
                }
                Command::Help | Command::Version => unreachable!(),
            }
        }
    }
    Ok(())
}

fn parse_args() -> Result<Command, Box<dyn Error>> {
    parse_args_from(env::args().skip(1))
}

fn parse_args_from(mut args: impl Iterator<Item = String>) -> Result<Command, Box<dyn Error>> {
    let Some(first) = args.next() else {
        return Ok(Command::Tui);
    };
    let command = match first.as_str() {
        "-h" | "--help" => Command::Help,
        "-V" | "--version" => Command::Version,
        "-p" | "--prompt" => {
            let prompt = args
                .next()
                .ok_or("--prompt requires a non-empty argument")?;
            let allow_tools = match args.next() {
                None => false,
                Some(value) if value == "--allow-tools" => true,
                Some(value) => return Err(format!("unexpected argument: {value}").into()),
            };
            return Ok(Command::Prompt {
                prompt,
                allow_tools,
            });
        }
        _ => return Err(format!("unknown argument: {first}").into()),
    };
    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument: {extra}").into());
    }
    Ok(command)
}

fn models_from_env() -> Result<Vec<Model>, Box<dyn Error>> {
    let mut models = Vec::new();
    if let Ok(key) = env::var("DEEPSEEK_API_KEY")
        && !key.trim().is_empty()
    {
        models.extend([
            Model::new(
                "DeepSeek",
                key.clone(),
                "https://api.deepseek.com",
                "deepseek-v4-flash",
                1_000_000,
            )
            .with_reasoning(DEEPSEEK_REASONING_EFFORTS, ReasoningEffort::High),
            Model::new(
                "DeepSeek",
                key,
                "https://api.deepseek.com",
                "deepseek-v4-pro",
                1_000_000,
            )
            .with_reasoning(DEEPSEEK_REASONING_EFFORTS, ReasoningEffort::High),
        ]);
    }
    if let Ok(key) = env::var("OPENAI_API_KEY")
        && !key.trim().is_empty()
    {
        models.extend([
            Model::new(
                "OpenAI",
                key.clone(),
                "https://api.openai.com/v1",
                "gpt-5.6-sol",
                1_050_000,
            )
            .with_reasoning(OPENAI_REASONING_EFFORTS, ReasoningEffort::Medium),
            Model::new(
                "OpenAI",
                key.clone(),
                "https://api.openai.com/v1",
                "gpt-5.6-terra",
                1_050_000,
            )
            .with_reasoning(OPENAI_REASONING_EFFORTS, ReasoningEffort::Medium),
            Model::new(
                "OpenAI",
                key,
                "https://api.openai.com/v1",
                "gpt-5.6-luna",
                1_050_000,
            )
            .with_reasoning(OPENAI_REASONING_EFFORTS, ReasoningEffort::Medium),
        ]);
    }
    if models.is_empty() {
        return Err("set DEEPSEEK_API_KEY or OPENAI_API_KEY".into());
    }
    Ok(models)
}

fn home_dir() -> Result<PathBuf, Box<dyn Error>> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| "cannot locate the home directory".into())
}

async fn run_prompt(agent: Agent, prompt: String, allow_tools: bool) -> Result<(), Box<dyn Error>> {
    let mut active = agent.start(prompt);
    let control = active.control();
    let mut failure = None;
    while let Some(event) = active.next_event().await {
        match event {
            AgentEvent::TextDelta(delta) => {
                print!("{delta}");
                io::stdout().flush()?;
            }
            AgentEvent::ApprovalRequired(call) => {
                control.approve(call.call_id, allow_tools)?;
            }
            AgentEvent::RunFinished(_) => println!(),
            AgentEvent::RunFailed(error) => failure = Some(error),
            AgentEvent::RunAborted => failure = Some("aborted".into()),
            _ => {}
        }
    }
    let _agent = active.finish().await?;
    if let Some(error) = failure {
        return Err(error.into());
    }
    Ok(())
}

async fn run_tui(
    agent: Agent,
    models: Vec<Model>,
    sessions_dir: PathBuf,
    cwd: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let usage = agent.latest_usage().map(|usage| {
        (
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.input_tokens_details.cached_tokens,
        )
    });
    let allow_all = read_permission(agent.session_info());
    let mut app = App::new(agent.model(), agent.transcript(), &cwd, usage, allow_all);
    let session_path = agent.session_info().path.clone();
    let mut runtime = Runtime {
        agent: Some(agent),
        active: None,
        models,
        selected_model: 0,
        sessions_dir,
        session_path,
        compacting: false,
    };
    let mut terminal = ratatui::try_init_with_options(TerminalOptions {
        viewport: Viewport::Inline(INLINE_VIEWPORT_HEIGHT),
    })?;
    if let Err(error) = crossterm::execute!(io::stdout(), EnableBracketedPaste) {
        ratatui::restore();
        return Err(error.into());
    }
    let result = tui_loop(&mut terminal, &mut app, &mut runtime).await;
    let input_result =
        crossterm::execute!(io::stdout(), DisableBracketedPaste, DisableMouseCapture);
    ratatui::restore();
    input_result?;
    result
}

async fn tui_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    runtime: &mut Runtime,
) -> Result<(), Box<dyn Error>> {
    let mut terminal_events = EventStream::new();
    let mut render_tick = tokio::time::interval(Duration::from_millis(250));
    render_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut fullscreen_overlay = None;
    let mut dirty = true;
    while !app.should_exit() {
        if dirty && runtime.active.is_none() {
            draw_ui(terminal, &mut fullscreen_overlay, app, false)?;
            dirty = false;
        }
        enum Next {
            Terminal(Option<Result<Event, io::Error>>),
            Agent(Option<AgentEvent>),
            Render,
        }
        let next = if let Some(active) = runtime.active.as_mut() {
            tokio::select! {
                event = terminal_events.next() => Next::Terminal(event),
                event = active.next_event() => Next::Agent(event),
                _ = render_tick.tick() => Next::Render,
            }
        } else {
            Next::Terminal(terminal_events.next().await)
        };

        match next {
            Next::Terminal(Some(Ok(event))) => {
                match event {
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        let action = app.handle_key(key, runtime.active.is_some());
                        handle_action(action, app, runtime).await?;
                    }
                    Event::Mouse(mouse) => app.handle_mouse(mouse),
                    Event::Paste(value) => app.paste(&value),
                    _ => {}
                }
                if !app.should_exit() {
                    draw_ui(
                        terminal,
                        &mut fullscreen_overlay,
                        app,
                        runtime.active.is_some(),
                    )?;
                }
                dirty = false;
                continue;
            }
            Next::Terminal(Some(Err(error))) => return Err(error.into()),
            Next::Terminal(None) => app.request_exit(),
            Next::Agent(Some(event)) => {
                if let Some((call_id, allow)) = app.apply_event(event)
                    && let Some(active) = &runtime.active
                {
                    active.control().approve(call_id, allow)?;
                }
            }
            Next::Agent(None) => finish_active(app, runtime).await?,
            Next::Render => {
                if dirty || runtime.active.is_some() {
                    draw_ui(terminal, &mut fullscreen_overlay, app, true)?;
                    dirty = false;
                }
                continue;
            }
        }
        dirty = true;
    }

    close_fullscreen_overlay(terminal, &mut fullscreen_overlay)?;

    if let Some(active) = &runtime.active {
        active.control().abort();
    }
    if runtime.active.is_some() {
        finish_active(app, runtime).await?;
    }
    Ok(())
}

fn draw_ui(
    terminal: &mut ratatui::DefaultTerminal,
    fullscreen_overlay: &mut Option<ratatui::DefaultTerminal>,
    app: &mut App,
    running: bool,
) -> Result<(), Box<dyn Error>> {
    if app.fullscreen_modal_open() && fullscreen_overlay.is_none() {
        let overlay =
            ratatui::Terminal::new(ratatui::backend::CrosstermBackend::new(io::stdout()))?;
        crossterm::execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        *fullscreen_overlay = Some(overlay);
    } else if !app.fullscreen_modal_open() {
        close_fullscreen_overlay(terminal, fullscreen_overlay)?;
    }

    if let Some(overlay) = fullscreen_overlay {
        overlay.draw(|frame| app.render(frame, running))?;
    } else {
        flush_history(terminal, app)?;
        terminal.draw(|frame| app.render(frame, running))?;
    }
    Ok(())
}

fn close_fullscreen_overlay(
    terminal: &mut ratatui::DefaultTerminal,
    fullscreen_overlay: &mut Option<ratatui::DefaultTerminal>,
) -> io::Result<()> {
    if fullscreen_overlay.take().is_some() {
        crossterm::execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen)?;
        terminal.clear()?;
    }
    Ok(())
}

fn flush_history(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
) -> Result<(), Box<dyn Error>> {
    let width = terminal.size()?.width.max(1);
    let lines = app.take_committed_lines(width as usize);
    for chunk in lines.chunks(256) {
        let paragraph = Paragraph::new(Text::from(chunk.to_vec())).wrap(Wrap { trim: false });
        let height = paragraph.line_count(width).max(1).min(u16::MAX as usize) as u16;
        terminal.insert_before(height, move |buffer| paragraph.render(buffer.area, buffer))?;
    }
    Ok(())
}

async fn handle_action(
    action: UiAction,
    app: &mut App,
    runtime: &mut Runtime,
) -> Result<(), Box<dyn Error>> {
    match action {
        UiAction::None => {}
        UiAction::Abort => {
            if let Some(active) = &runtime.active {
                active.control().abort();
            }
        }
        UiAction::Exit => app.request_exit(),
        UiAction::Approve { call_id, allow } => {
            if let Some(active) = &runtime.active {
                active.control().approve(call_id, allow)?;
            }
        }
        UiAction::SetPermissions {
            allow_all,
            pending_call_id,
        } => match write_runtime_permission(runtime, allow_all) {
            Ok(()) => {
                app.set_permission_mode(allow_all);
                if let Some(call_id) = pending_call_id
                    && let Some(active) = &runtime.active
                {
                    active.control().approve(call_id, true)?;
                }
            }
            Err(error) => {
                app.notice(format!("Permission save failed: {error}"));
                if let Some(call_id) = pending_call_id
                    && let Some(active) = &runtime.active
                {
                    active.control().approve(call_id, false)?;
                }
            }
        },
        UiAction::Prefill(value) => app.prefill(&value),
        UiAction::Resume(path) => {
            if runtime.active.is_some() {
                app.notice("Cannot resume while the agent is running");
            } else {
                match Session::open(path).await {
                    Ok(session) => {
                        let allow_all = read_permission(session.info());
                        runtime.session_path = session.info().path.clone();
                        let agent = runtime.agent.as_mut().expect("idle agent");
                        agent.set_session(session);
                        app.load_transcript(agent.transcript());
                        app.set_identity(agent.model());
                        app.set_usage_values(agent.latest_usage().map(|usage| {
                            (
                                usage.input_tokens,
                                usage.output_tokens,
                                usage.total_tokens,
                                usage.input_tokens_details.cached_tokens,
                            )
                        }));
                        app.set_permission_mode(allow_all);
                    }
                    Err(error) => app.notice(format!("Resume failed: {error}")),
                }
            }
        }
        UiAction::DeleteSession { session, selected } => {
            let notice = if session.path == runtime.session_path {
                "Current session cannot be deleted".into()
            } else {
                match Session::delete(&session) {
                    Ok(()) => {
                        let permissions = permission_path(&session.path);
                        if let Err(error) = fs::remove_file(permissions)
                            && error.kind() != io::ErrorKind::NotFound
                        {
                            format!(
                                "Deleted “{}”; permission cleanup failed: {error}",
                                session.title
                            )
                        } else {
                            format!("Deleted “{}” permanently", session.title)
                        }
                    }
                    Err(error) => format!("Could not delete “{}”: {error}", session.title),
                }
            };
            open_session_manager(app, runtime, Some(selected), Some(notice));
        }
        UiAction::SelectModel(index) => {
            if runtime.active.is_some() {
                app.notice("Cannot switch models while the agent is running");
            } else if let Some(model) = runtime.models.get(index) {
                if model.reasoning_efforts().is_empty() {
                    runtime.selected_model = index;
                    let agent = runtime.agent.as_mut().expect("idle agent");
                    agent.set_model(model.clone());
                    app.set_identity(agent.model());
                    app.set_usage_values(None);
                } else {
                    app.show_reasoning(model, index);
                }
            }
        }
        UiAction::SelectReasoning { model, effort } => {
            if runtime.active.is_some() {
                app.notice("Cannot change reasoning while the agent is running");
            } else if let Some(reasoning_effort) = runtime
                .models
                .get(model)
                .and_then(|model| model.reasoning_efforts().get(effort))
                .cloned()
            {
                runtime.models[model].set_reasoning_effort(reasoning_effort.clone());
                let agent = runtime.agent.as_mut().expect("idle agent");
                if model == runtime.selected_model {
                    agent.set_reasoning_effort(reasoning_effort);
                } else {
                    runtime.selected_model = model;
                    agent.set_model(runtime.models[model].clone());
                    app.set_usage_values(None);
                }
                app.set_identity(agent.model());
            }
        }
        UiAction::Submit(value) => handle_submit(value, app, runtime).await?,
    }
    Ok(())
}

async fn handle_submit(
    value: String,
    app: &mut App,
    runtime: &mut Runtime,
) -> Result<(), Box<dyn Error>> {
    let trimmed = value.trim();
    if let Some(command) = trimmed.strip_prefix('/') {
        let (name, argument) = command.split_once(' ').unwrap_or((command, ""));
        match name {
            "session" | "resume" if runtime.active.is_none() => {
                open_session_manager(app, runtime, None, None);
            }
            "session" | "resume" => app.notice("Cannot manage sessions while the agent is running"),
            "model" => app.show_models(&runtime.models, runtime.selected_model),
            "permissions" => {
                if let Some(allow_all) = app.request_permission_toggle() {
                    match write_runtime_permission(runtime, allow_all) {
                        Ok(()) => app.set_permission_mode(allow_all),
                        Err(error) => app.notice(format!("Permission save failed: {error}")),
                    }
                }
            }
            "tool" => app.show_tools(),
            "compact" if runtime.session_path.as_os_str().is_empty() => {
                app.notice("Nothing to compact");
            }
            "compact" if runtime.active.is_none() => {
                let agent = runtime.agent.take().expect("idle agent");
                let active = agent.start_compaction(
                    (!argument.trim().is_empty()).then(|| argument.trim().to_owned()),
                );
                runtime.active = Some(active);
                runtime.compacting = true;
            }
            "compact" => app.notice("Cannot compact while the agent is running"),
            "queue" if runtime.compacting => {
                app.notice("Cannot queue while compacting");
            }
            "queue" if runtime.active.is_some() && !argument.trim().is_empty() => {
                runtime
                    .active
                    .as_ref()
                    .expect("active agent")
                    .control()
                    .queue(argument.trim().to_owned())?;
                app.push_user(argument.trim().to_owned());
            }
            "queue" => app.notice("Usage: /queue MESSAGE while the agent is running"),
            "help" => app.notice(HELP),
            "exit" => app.request_exit(),
            _ => app.notice(format!("Unknown command: /{name}")),
        }
        return Ok(());
    }

    if runtime.compacting {
        app.notice("Cannot steer while compacting");
        return Ok(());
    }
    ensure_persistent_session(app, runtime).await?;
    app.push_user(value.clone());
    if let Some(active) = &runtime.active {
        active.control().steer(value)?;
    } else {
        let agent = runtime.agent.take().expect("idle agent");
        let active = agent.start(value);
        runtime.active = Some(active);
    }
    Ok(())
}

async fn ensure_persistent_session(
    app: &mut App,
    runtime: &mut Runtime,
) -> Result<(), Box<dyn Error>> {
    if !runtime.session_path.as_os_str().is_empty() {
        return Ok(());
    }
    let session = Session::create(&runtime.sessions_dir).await?;
    runtime.session_path = session.info().path.clone();
    let agent = runtime.agent.as_mut().expect("idle agent");
    agent.set_session(session);
    app.set_identity(agent.model());
    if app.permission_mode()
        && let Err(error) = write_permission(&runtime.session_path, true)
    {
        app.set_permission_mode(false);
        app.notice(format!("Permission save failed; using ask mode: {error}"));
    }
    Ok(())
}

fn open_session_manager(
    app: &mut App,
    runtime: &Runtime,
    selected: Option<usize>,
    notice: Option<String>,
) {
    match Session::list(&runtime.sessions_dir) {
        Ok(sessions) => {
            app.show_sessions(sessions, &runtime.session_path, selected, notice);
        }
        Err(error) => app.notice(format!("Session list failed: {error}")),
    }
}

fn permission_path(session_path: &Path) -> PathBuf {
    session_path.with_extension("permissions")
}

fn read_permission(session: &SessionInfo) -> bool {
    if session.path.as_os_str().is_empty() {
        return false;
    }
    fs::read_to_string(permission_path(&session.path))
        .is_ok_and(|value| value.trim() == "allow all")
}

fn write_runtime_permission(runtime: &Runtime, allow_all: bool) -> io::Result<()> {
    if runtime.session_path.as_os_str().is_empty() {
        Ok(())
    } else {
        write_permission(&runtime.session_path, allow_all)
    }
}

fn write_permission(session_path: &Path, allow_all: bool) -> io::Result<()> {
    let path = permission_path(session_path);
    fs::write(path, if allow_all { "allow all\n" } else { "ask\n" })
}

async fn finish_active(app: &mut App, runtime: &mut Runtime) -> Result<(), Box<dyn Error>> {
    let active = runtime.active.take().expect("active agent");
    let agent = active.finish().await?;
    app.set_identity(agent.model());
    runtime.agent = Some(agent);
    runtime.compacting = false;
    Ok(())
}

#[cfg(test)]
mod tests {
    use kcastle_agent::{Agent, Model, Session};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::{
        App, Command, Runtime, SessionInfo, ensure_persistent_session, handle_submit,
        parse_args_from, read_permission, write_permission,
    };

    #[test]
    fn prompt_tools_require_explicit_flag() {
        let command = parse_args_from(
            ["--prompt", "hello", "--allow-tools"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert!(matches!(
            command,
            Command::Prompt {
                prompt,
                allow_tools: true
            } if prompt == "hello"
        ));
        assert!(
            parse_args_from(
                ["--prompt", "hello", "--unknown"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err()
        );
    }

    #[test]
    fn permission_mode_round_trips_per_session() {
        let path = std::env::temp_dir().join(format!(
            "kcastle-permission-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let info = SessionInfo {
            path: path.clone(),
            title: "test".into(),
            created_at: 0,
        };
        assert!(!read_permission(&info));
        write_permission(&path, true).unwrap();
        assert!(read_permission(&info));
        write_permission(&path, false).unwrap();
        assert!(!read_permission(&info));
        std::fs::remove_file(path.with_extension("permissions")).unwrap();
    }

    #[tokio::test]
    async fn tui_session_stays_ephemeral_until_persistence_is_needed() {
        let directory = std::env::temp_dir().join(format!(
            "kcastle-lazy-session-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model = Model::new("test", "key", "http://localhost", "model", 10_000);
        let session = Session::memory();
        let cwd = std::path::PathBuf::from(".");
        let mut app = App::new(&model, Vec::new(), &cwd, None, false);
        let agent = Agent::new(model.clone(), "test", session, &cwd);
        let mut runtime = Runtime {
            agent: Some(agent),
            active: None,
            models: vec![model],
            selected_model: 0,
            sessions_dir: directory.clone(),
            session_path: std::path::PathBuf::new(),
            compacting: false,
        };

        handle_submit("/help".into(), &mut app, &mut runtime)
            .await
            .unwrap();
        assert!(Session::list(&directory).unwrap().is_empty());

        app.set_permission_mode(true);
        ensure_persistent_session(&mut app, &mut runtime)
            .await
            .unwrap();
        let sessions = Session::list(&directory).unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(runtime.session_path, sessions[0].path);
        assert!(read_permission(&sessions[0]));

        ensure_persistent_session(&mut app, &mut runtime)
            .await
            .unwrap();
        assert_eq!(Session::list(&directory).unwrap().len(), 1);
        drop(runtime);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn input_is_rejected_while_compacting() {
        let model = Model::new("test", "key", "http://localhost", "model", 10_000);
        let session = Session::memory();
        let cwd = std::path::PathBuf::from(".");
        let mut app = App::new(&model, Vec::new(), &cwd, None, false);
        let agent = Agent::new(model.clone(), "test", session, &cwd);
        let mut runtime = Runtime {
            agent: Some(agent),
            active: None,
            models: vec![model],
            selected_model: 0,
            sessions_dir: std::path::PathBuf::new(),
            session_path: std::path::PathBuf::new(),
            compacting: true,
        };

        handle_submit("do not lose this".into(), &mut app, &mut runtime)
            .await
            .unwrap();
        assert!(runtime.active.is_none());
        assert!(runtime.agent.is_some());
        let mut terminal = Terminal::new(TestBackend::new(50, 12)).unwrap();
        terminal.draw(|frame| app.render(frame, true)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("Cannot steer while compacting"));
    }
}
