mod app;

use std::env;
use std::error::Error;
use std::io::{self, Write};
use std::path::PathBuf;

use app::{App, UiAction};
use crossterm::event::{Event, EventStream, KeyEventKind};
use futures_util::StreamExt;
use kcastle_agent::{ActiveAgent, Agent, AgentEvent, Model, RunControl, Session};

const INSTRUCTIONS: &str = "You are K, a concise coding agent. Use the shell tool when it helps. Inspect before editing, report tool errors honestly, and stop when the task is complete.";

const HELP: &str = "K in Castle — native agent harness\n\nUSAGE:\n    kcastle [--prompt TEXT]\n\nOPTIONS:\n    -h, --help       Show help\n    -V, --version    Show version\n    -p, --prompt     Run one non-interactive prompt\n\nTUI COMMANDS:\n    /resume          Open a saved session\n    /model           Switch model backend\n    /compact [focus] Compact active context\n    /permissions     Toggle ask / allow all\n    /queue MESSAGE   Run after the active task settles\n    /help            Show commands\n    /exit            Exit\n";

enum Command {
    Tui,
    Prompt(String),
    Help,
    Version,
}

struct Runtime {
    agent: Option<Agent>,
    active: Option<ActiveAgent>,
    control: Option<RunControl>,
    models: Vec<Model>,
    selected_model: usize,
    sessions_dir: PathBuf,
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
            let session = Session::create(&sessions_dir).await?;
            let agent = Agent::new(
                models[0].clone(),
                INSTRUCTIONS,
                session,
                env::current_dir()?,
            );
            match command {
                Command::Prompt(prompt) => run_prompt(agent, prompt).await?,
                Command::Tui => run_tui(agent, models, sessions_dir).await?,
                Command::Help | Command::Version => unreachable!(),
            }
        }
    }
    Ok(())
}

fn parse_args() -> Result<Command, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let Some(first) = args.next() else {
        return Ok(Command::Tui);
    };
    let command = match first.as_str() {
        "-h" | "--help" => Command::Help,
        "-V" | "--version" => Command::Version,
        "-p" | "--prompt" => Command::Prompt(
            args.next()
                .ok_or("--prompt requires a non-empty argument")?,
        ),
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
        models.push(Model::new(
            "DeepSeek",
            key,
            "https://api.deepseek.com",
            "deepseek-v4-flash",
            128_000,
        ));
    }
    if let Ok(key) = env::var("OPENAI_API_KEY")
        && !key.trim().is_empty()
    {
        models.push(Model::new(
            "OpenAI",
            key,
            "https://api.openai.com/v1",
            "gpt-5.5",
            1_050_000,
        ));
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

async fn run_prompt(agent: Agent, prompt: String) -> Result<(), Box<dyn Error>> {
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
                control.approve(call.call_id, false)?;
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
) -> Result<(), Box<dyn Error>> {
    let mut app = App::new(agent.model(), agent.session_info(), agent.transcript());
    let mut runtime = Runtime {
        agent: Some(agent),
        active: None,
        control: None,
        models,
        selected_model: 0,
        sessions_dir,
    };
    let mut terminal = ratatui::init();
    let result = tui_loop(&mut terminal, &mut app, &mut runtime).await;
    ratatui::restore();
    result
}

async fn tui_loop(
    terminal: &mut ratatui::DefaultTerminal,
    app: &mut App,
    runtime: &mut Runtime,
) -> Result<(), Box<dyn Error>> {
    let mut terminal_events = EventStream::new();
    while !app.should_exit() {
        terminal.draw(|frame| app.render(frame, runtime.active.is_some()))?;
        enum Next {
            Terminal(Option<Result<Event, io::Error>>),
            Agent(Option<AgentEvent>),
        }
        let next = if let Some(active) = runtime.active.as_mut() {
            tokio::select! {
                event = terminal_events.next() => Next::Terminal(event),
                event = active.next_event() => Next::Agent(event),
            }
        } else {
            Next::Terminal(terminal_events.next().await)
        };

        match next {
            Next::Terminal(Some(Ok(event))) => match event {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    let action = app.handle_key(key, runtime.active.is_some());
                    handle_action(action, app, runtime).await?;
                }
                Event::Mouse(mouse) => app.handle_mouse(mouse),
                Event::Paste(value) => app.paste(&value),
                _ => {}
            },
            Next::Terminal(Some(Err(error))) => return Err(error.into()),
            Next::Terminal(None) => app.request_exit(),
            Next::Agent(Some(event)) => {
                if let Some((call_id, allow)) = app.apply_event(event)
                    && let Some(control) = &runtime.control
                {
                    control.approve(call_id, allow)?;
                }
            }
            Next::Agent(None) => finish_active(app, runtime).await?,
        }
    }

    if let Some(control) = &runtime.control {
        control.abort();
    }
    if runtime.active.is_some() {
        finish_active(app, runtime).await?;
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
            if let Some(control) = &runtime.control {
                control.abort();
            }
        }
        UiAction::Exit => app.request_exit(),
        UiAction::Approve { call_id, allow } => {
            if let Some(control) = &runtime.control {
                control.approve(call_id, allow)?;
            }
        }
        UiAction::Resume(path) => {
            if runtime.active.is_some() {
                app.notice("Cannot resume while the agent is running");
            } else {
                match Session::open(path).await {
                    Ok(session) => {
                        let agent = runtime.agent.as_mut().expect("idle agent");
                        agent.set_session(session);
                        app.load_transcript(agent.transcript());
                        app.set_identity(agent.model(), agent.session_info());
                    }
                    Err(error) => app.notice(format!("Resume failed: {error}")),
                }
            }
        }
        UiAction::SelectModel(index) => {
            if runtime.active.is_some() {
                app.notice("Cannot switch models while the agent is running");
            } else if let Some(model) = runtime.models.get(index).cloned() {
                runtime.selected_model = index;
                let agent = runtime.agent.as_mut().expect("idle agent");
                agent.set_model(model);
                app.set_identity(agent.model(), agent.session_info());
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
            "resume" => app.show_sessions(Session::list(&runtime.sessions_dir)?),
            "model" => app.show_models(&runtime.models, runtime.selected_model),
            "permissions" => app.toggle_permissions(),
            "compact" if runtime.active.is_none() => {
                let agent = runtime.agent.take().expect("idle agent");
                let active = agent.start_compaction(
                    (!argument.trim().is_empty()).then(|| argument.trim().to_owned()),
                );
                runtime.control = Some(active.control());
                runtime.active = Some(active);
            }
            "compact" => app.notice("Cannot compact while the agent is running"),
            "queue" if runtime.active.is_some() && !argument.trim().is_empty() => {
                runtime
                    .control
                    .as_ref()
                    .expect("active control")
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

    app.push_user(value.clone());
    if let Some(control) = &runtime.control {
        control.steer(value)?;
    } else {
        let agent = runtime.agent.take().expect("idle agent");
        let active = agent.start(value);
        runtime.control = Some(active.control());
        runtime.active = Some(active);
    }
    Ok(())
}

async fn finish_active(app: &mut App, runtime: &mut Runtime) -> Result<(), Box<dyn Error>> {
    let active = runtime.active.take().expect("active agent");
    let agent = active.finish().await?;
    app.set_identity(agent.model(), agent.session_info());
    runtime.agent = Some(agent);
    runtime.control = None;
    Ok(())
}
