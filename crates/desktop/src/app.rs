use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpui::{
    AppContext, Context, Entity, FocusHandle, PathPromptOptions, Pixels, Point, ScrollHandle,
    ScrollWheelEvent, SharedString, Subscription, UniformListScrollHandle, Window, point, px,
};
use gpui_component::input::{InputEvent, InputState};
use gpui_component::{Theme, ThemeMode};
use kcastle_agent::{
    Agent, AgentEvent, Model, RunControl, Session, SessionInfo, ToolResult, TranscriptItem,
};

use crate::dialogs::Modal;
use crate::project::ProjectStore;
use crate::settings::{Appearance, EnterBehavior, SettingsStore};
use crate::streaming_markdown::StreamingMarkdownState;

#[derive(Clone)]
pub(crate) struct ConfiguredModel {
    pub(crate) id: String,
    pub(crate) model: Model,
}

impl ConfiguredModel {
    pub(crate) fn new(model: Model) -> Self {
        Self {
            id: format!("{}/{}", model.name(), model.model()),
            model,
        }
    }

    pub(crate) fn label(&self) -> String {
        format!("{} · {}", self.model.name(), self.model.model())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    User,
    Reasoning,
    Assistant,
    Tool,
    Notice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DetailsTab {
    Summary,
    Preview,
    Raw,
    Payload,
    Result,
    Schema,
    Timing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComposerMenu {
    Commands,
    Permission,
    Model,
    Models,
    Effort,
    Workspace,
}

#[derive(Debug)]
pub(crate) struct Message {
    pub(crate) render_key: u64,
    pub(crate) role: Role,
    pub(crate) id: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) text: String,
    pub(crate) render_text: SharedString,
    pub(crate) payload: Option<String>,
    pub(crate) schema: Option<String>,
    pub(crate) pending: bool,
    pub(crate) failed: bool,
    pub(crate) expanded: bool,
    pub(crate) rating: Option<bool>,
    pub(crate) started_at_ms: Option<u128>,
    pub(crate) duration_ms: Option<u128>,
    pub(crate) turn: usize,
    pub(crate) step: usize,
    pub(crate) request_id: Option<String>,
    pub(crate) search_text: String,
    pub(crate) streaming_markdown: StreamingMarkdownState,
}

static NEXT_MESSAGE_RENDER_KEY: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_message_render_key() -> u64 {
    NEXT_MESSAGE_RENDER_KEY.fetch_add(1, Ordering::Relaxed)
}

fn is_frame_stream_event(event: &AgentEvent) -> bool {
    matches!(
        event,
        AgentEvent::ReasoningDelta(_) | AgentEvent::TextDelta(_)
    )
}

fn arm_stream_frame(window: &mut Window, ready: tokio::sync::oneshot::Sender<()>) {
    window.on_next_frame(move |_, _| {
        let _ = ready.send(());
    });
    window.refresh();
}

#[derive(Clone, Copy, Debug)]
struct SessionViewState {
    chat_offset: Point<Pixels>,
    chat_follow: bool,
    trajectory_offset: Point<Pixels>,
    details_offset: Point<Pixels>,
    selected_trajectory: Option<usize>,
    details_tab: DetailsTab,
}

impl Default for SessionViewState {
    fn default() -> Self {
        Self {
            chat_offset: point(px(0.0), px(0.0)),
            chat_follow: true,
            trajectory_offset: point(px(0.0), px(0.0)),
            details_offset: point(px(0.0), px(0.0)),
            selected_trajectory: None,
            details_tab: DetailsTab::Summary,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SessionSearchDocument {
    pub(crate) searchable: String,
    pub(crate) summary: String,
    pub(crate) snippets: Vec<String>,
}

pub(crate) struct Approval {
    pub(crate) call_id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

pub(crate) struct DesktopStartup {
    pub(crate) agent: Agent,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) project_store: ProjectStore,
    pub(crate) active_project: usize,
    pub(crate) settings: SettingsStore,
}

pub(crate) struct DesktopApp {
    pub(crate) agent: Option<Agent>,
    pub(crate) control: Option<RunControl>,
    pub(crate) input: Entity<InputState>,
    pub(crate) session_search: Entity<InputState>,
    pub(crate) trajectory_search: Entity<InputState>,
    pub(crate) messages: Vec<Message>,
    pub(crate) approval: Option<Approval>,
    pub(crate) modal: Option<Modal>,
    pub(crate) modal_focus: FocusHandle,
    pub(crate) composer_menu: Option<ComposerMenu>,
    pub(crate) composer_menu_focus: FocusHandle,
    pub(crate) composer_menu_highlight: usize,
    pub(crate) scroll: ScrollHandle,
    pub(crate) follow_chat_tail: bool,
    pub(crate) unread_stream_updates: usize,
    pub(crate) trajectory_scroll: UniformListScrollHandle,
    pub(crate) details_scroll: ScrollHandle,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) model: String,
    pub(crate) cwd: PathBuf,
    pub(crate) project_store: ProjectStore,
    pub(crate) settings: SettingsStore,
    pub(crate) active_project: usize,
    pub(crate) expanded_projects: HashSet<PathBuf>,
    pub(crate) sessions_dir: PathBuf,
    pub(crate) sessions: Vec<SessionInfo>,
    pub(crate) current_session: PathBuf,
    pub(crate) title: String,
    pub(crate) preparing_session: bool,
    pub(crate) show_sidebar: bool,
    pub(crate) show_trajectory: bool,
    pub(crate) trajectory_collapsed_turns: bool,
    pub(crate) trajectory_collapsed_calls: bool,
    pub(crate) trajectory_duration: bool,
    pub(crate) selected_trajectory: Option<usize>,
    pub(crate) details_tab: DetailsTab,
    pub(crate) search_sessions: bool,
    pub(crate) show_sidebar_options: bool,
    pub(crate) session_action_target: Option<PathBuf>,
    pub(crate) group_sessions_by_workspace: bool,
    pub(crate) sort_sessions_by_recent: bool,
    pub(crate) started_at: Option<Instant>,
    pub(crate) turns: usize,
    pub(crate) tool_calls: usize,
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) cached_tokens: u32,
    pub(crate) tool_schemas: HashMap<String, String>,
    pub(crate) session_search_documents: HashMap<PathBuf, SessionSearchDocument>,
    view_states: HashMap<String, SessionViewState>,
    _subscriptions: Vec<Subscription>,
}

impl DesktopApp {
    pub(crate) fn new(
        startup: DesktopStartup,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let DesktopStartup {
            agent,
            models,
            selected_model,
            project_store,
            active_project,
            settings,
        } = startup;
        let project = project_store
            .project(active_project)
            .expect("active project should exist")
            .clone();
        let model = format!("{} · {}", agent.model().name(), agent.model().model());
        let tool_schemas = tool_schema_map(&agent);
        let current_session = agent.session_info().path.clone();
        let sessions = Session::list(&project.sessions_dir).unwrap_or_default();
        let session_search_documents = build_session_search_documents(&project_store);
        let input = cx.new(|cx| {
            InputState::new(window, cx)
                .auto_grow(1, 14)
                .placeholder("Describe what you want to build")
        });
        let session_search =
            cx.new(|cx| InputState::new(window, cx).placeholder("Search sessions…"));
        let trajectory_search =
            cx.new(|cx| InputState::new(window, cx).placeholder("Search trajectory"));
        let subscription = cx.subscribe_in(
            &input,
            window,
            |this, _, event: &InputEvent, window, cx| match event {
                InputEvent::PressEnter { secondary: false } => {
                    if this.composer_menu.is_none() {
                        this.submit(window, cx);
                    }
                }
                InputEvent::Change => {
                    if this.follow_chat_tail {
                        this.scroll.scroll_to_bottom();
                    }
                    cx.notify();
                }
                _ => {}
            },
        );
        let search_subscription = cx.subscribe_in(
            &session_search,
            window,
            |_this, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        );
        let trajectory_subscription = cx.subscribe_in(
            &trajectory_search,
            window,
            |_this, _, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    cx.notify();
                }
            },
        );
        input.update(cx, |input, cx| input.focus(window, cx));
        let appearance_subscription = cx.observe_window_appearance(window, |this, window, cx| {
            if this.settings.appearance() == Appearance::System {
                Theme::sync_system_appearance(Some(window), cx);
                cx.notify();
            }
        });
        Self {
            agent: Some(agent),
            control: None,
            input,
            session_search,
            trajectory_search,
            messages: Vec::new(),
            approval: None,
            modal: None,
            modal_focus: cx.focus_handle(),
            composer_menu: None,
            composer_menu_focus: cx.focus_handle(),
            composer_menu_highlight: 0,
            scroll: ScrollHandle::new(),
            follow_chat_tail: true,
            unread_stream_updates: 0,
            trajectory_scroll: UniformListScrollHandle::new(),
            details_scroll: ScrollHandle::new(),
            models,
            selected_model,
            model,
            cwd: project.path.clone(),
            project_store,
            settings,
            active_project,
            expanded_projects: HashSet::from([project.path.clone()]),
            sessions_dir: project.sessions_dir,
            sessions,
            current_session,
            title: "New chat".into(),
            preparing_session: false,
            show_sidebar: true,
            show_trajectory: false,
            trajectory_collapsed_turns: false,
            trajectory_collapsed_calls: false,
            trajectory_duration: false,
            selected_trajectory: None,
            details_tab: DetailsTab::Summary,
            search_sessions: false,
            show_sidebar_options: false,
            session_action_target: None,
            group_sessions_by_workspace: true,
            sort_sessions_by_recent: true,
            started_at: None,
            turns: 0,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            tool_schemas,
            session_search_documents,
            view_states: HashMap::new(),
            _subscriptions: vec![
                subscription,
                search_subscription,
                trajectory_subscription,
                appearance_subscription,
            ],
        }
    }

    pub(crate) fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.preparing_session {
            return;
        }
        let value = self.input.read(cx).value().trim().to_owned();
        if value.is_empty() {
            return;
        }
        self.input
            .update(cx, |input, cx| input.set_value("", window, cx));
        self.input.update(cx, |input, cx| {
            input.set_placeholder("Message the agent", window, cx)
        });
        self.messages.push(message(Role::User, value.clone()));
        reindex_messages(&mut self.messages);
        self.turns = self
            .messages
            .iter()
            .filter(|message| message.role == Role::User)
            .count();
        self.follow_chat_tail = true;
        self.unread_stream_updates = 0;
        self.scroll.scroll_to_bottom();
        if self.title == "New chat" {
            self.title = short_title(&value);
        }

        if let Some(control) = &self.control {
            let result = match self.settings.enter_behavior() {
                EnterBehavior::Steer => control.steer(value),
                EnterBehavior::Queue => control.queue(value),
            };
            if let Err(error) = result {
                self.notice(error.to_string());
            }
            cx.notify();
            return;
        }

        if self.current_session.as_os_str().is_empty() {
            self.preparing_session = true;
            let sessions_dir = self.sessions_dir.clone();
            let retry_value = value.clone();
            cx.spawn_in(window, async move |this, cx| {
                let session = Session::create(sessions_dir).await;
                let _ = cx.update(|window, app| {
                    if let Some(this) = this.upgrade() {
                        this.update(app, |this, cx| match session {
                            Ok(session) => {
                                this.preparing_session = false;
                                this.activate_session_for_run(session);
                                this.start_run(retry_value, window, cx);
                            }
                            Err(error) => {
                                this.preparing_session = false;
                                this.messages.pop();
                                this.title = "New chat".into();
                                this.input.update(cx, |input, cx| {
                                    input.set_value(&retry_value, window, cx);
                                    input.set_placeholder(
                                        "Describe what you want to build",
                                        window,
                                        cx,
                                    );
                                });
                                this.notice(format!("Could not create session: {error}"));
                                cx.notify();
                            }
                        });
                    }
                });
            })
            .detach();
            return;
        }

        self.start_run(value, window, cx);
    }

    fn start_run(&mut self, value: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(agent) = self.agent.take() else {
            return;
        };
        let mut active = agent.start(value);
        self.control = Some(active.control());
        self.started_at = Some(Instant::now());
        cx.notify();

        cx.spawn_in(window, async move |this, cx| {
            let mut stream_ended = false;
            while !stream_ended {
                let Some(first) = active.next_event().await else {
                    break;
                };
                let collect_frame = is_frame_stream_event(&first);
                let mut events = vec![first];
                if collect_frame {
                    let (frame_tx, mut frame_rx) = tokio::sync::oneshot::channel();
                    cx.update(|window, _| {
                        arm_stream_frame(window, frame_tx);
                    })
                    .ok();
                    let mut reached_frame = false;
                    let mut reached_structure = false;
                    while events.len() < 128 {
                        tokio::select! {
                            biased;
                            _ = &mut frame_rx => {
                                reached_frame = true;
                                break;
                            }
                            event = active.next_event() => match event {
                            Some(event) => {
                                let structural = !is_frame_stream_event(&event);
                                events.push(event);
                                if structural {
                                    reached_structure = true;
                                    break;
                                }
                            }
                            None => {
                                stream_ended = true;
                                break;
                            }
                            },
                        }
                    }
                    if !reached_frame && !reached_structure && !stream_ended {
                        let _ = frame_rx.await;
                    }
                }
                let _ = cx.update(|_, app| {
                    if let Some(entity) = this.upgrade() {
                        entity.update(app, |this, cx| {
                            let follow_tail = this.follow_chat_tail;
                            let previous_len = this.messages.len();
                            let has_delta = events.iter().any(|event| {
                                matches!(
                                    event,
                                    AgentEvent::ReasoningDelta(_) | AgentEvent::TextDelta(_)
                                )
                            });
                            if !follow_tail {
                                this.unread_stream_updates += events
                                    .iter()
                                    .filter(|event| {
                                        matches!(
                                            event,
                                            AgentEvent::ReasoningDelta(_)
                                                | AgentEvent::TextDelta(_)
                                        )
                                    })
                                    .count();
                            }
                            for event in events {
                                this.apply_event(event);
                            }
                            if has_delta {
                                refresh_live_render_state(&mut this.messages);
                            }
                            if this.messages.len() != previous_len {
                                reindex_messages(&mut this.messages);
                            }
                            if follow_tail {
                                this.scroll.scroll_to_bottom();
                            }
                            cx.notify();
                        });
                    }
                });
            }
            let agent = active.finish().await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        this.control = None;
                        this.approval = None;
                        this.started_at = None;
                        match agent {
                            Ok(agent) => {
                                this.current_session = agent.session_info().path.clone();
                                this.sessions =
                                    Session::list(&this.sessions_dir).unwrap_or_default();
                                this.refresh_session_search_documents();
                                this.agent = Some(agent);
                            }
                            Err(error) => this.notice(error.to_string()),
                        }
                        this.input.update(cx, |input, cx| input.focus(window, cx));
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    fn apply_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::ReasoningDelta(delta) => push_reasoning_delta(&mut self.messages, &delta),
            AgentEvent::TextDelta(delta) => push_delta(&mut self.messages, &delta),
            AgentEvent::ApprovalRequired(call) => {
                if self.settings.allow_all_tools() {
                    if let Some(control) = &self.control
                        && let Err(error) = control.approve(call.call_id, true)
                    {
                        self.notice(error.to_string());
                    }
                } else {
                    self.approval = Some(Approval {
                        call_id: call.call_id,
                        name: call.name,
                        arguments: call.arguments,
                    });
                }
            }
            AgentEvent::ToolStarted(call) => {
                self.tool_calls += 1;
                settle_active_response_message(&mut self.messages);
                let schema = self.tool_schemas.get(&call.name).cloned();
                self.messages.push(Message {
                    render_key: next_message_render_key(),
                    role: Role::Tool,
                    id: Some(call.call_id),
                    title: Some(call.name),
                    text: String::new(),
                    render_text: SharedString::default(),
                    payload: Some(call.arguments),
                    schema,
                    pending: true,
                    failed: false,
                    expanded: false,
                    rating: None,
                    started_at_ms: Some(now_ms()),
                    duration_ms: None,
                    turn: 0,
                    step: 0,
                    request_id: None,
                    search_text: String::new(),
                    streaming_markdown: StreamingMarkdownState::default(),
                });
            }
            AgentEvent::ToolFinished { call, result } => self.tool_result(&call.call_id, result),
            AgentEvent::RunFinished(summary) => {
                for message in self.messages.iter_mut().rev() {
                    if message.role == Role::User {
                        break;
                    }
                    if message.request_id.is_none()
                        || message
                            .request_id
                            .as_deref()
                            .is_some_and(|id| id.starts_with("turn-"))
                    {
                        message.request_id = Some(summary.response_id.clone());
                    }
                    if matches!(message.role, Role::Reasoning | Role::Assistant) {
                        message.pending = false;
                    }
                }
                if let Some(usage) = summary.usage {
                    self.input_tokens = usage.input_tokens;
                    self.output_tokens = usage.output_tokens;
                    self.cached_tokens = usage.input_tokens_details.cached_tokens;
                }
            }
            AgentEvent::RunFailed(error) => {
                self.finish_reasoning();
                self.notice(error);
            }
            AgentEvent::RunAborted => {
                self.finish_reasoning();
                self.notice("Stopped");
            }
            AgentEvent::CompactionStarted { .. } => self.notice("Compacting context…"),
            AgentEvent::CompactionFinished { .. } => self.notice("Context compacted"),
            _ => {}
        }
    }

    fn finish_reasoning(&mut self) {
        for message in self.messages.iter_mut().rev() {
            if message.role == Role::User {
                break;
            }
            if matches!(message.role, Role::Reasoning | Role::Assistant) {
                message.pending = false;
            }
        }
    }

    pub(crate) fn decide(&mut self, call_id: String, allow: bool, cx: &mut Context<Self>) {
        if let Some(control) = &self.control
            && let Err(error) = control.approve(call_id, allow)
        {
            self.notice(error.to_string());
        }
        self.approval = None;
        self.notice(if allow { "Tool allowed" } else { "Tool denied" });
        cx.notify();
    }

    pub(crate) fn abort(&mut self, cx: &mut Context<Self>) {
        if let Some(control) = &self.control {
            control.abort();
            self.notice("Stopping…");
            cx.notify();
        }
    }

    fn tool_result(&mut self, call_id: &str, result: ToolResult) {
        if let Some(message) = self
            .messages
            .iter_mut()
            .rev()
            .find(|message| message.id.as_deref() == Some(call_id))
        {
            message.text = result.output;
            message.render_text = message.text.clone().into();
            message.pending = false;
            message.failed = result.is_error;
            message.duration_ms = message
                .started_at_ms
                .map(|started| now_ms().saturating_sub(started));
            refresh_message_search_text(message);
        }
    }

    fn notice(&mut self, text: impl Into<String>) {
        self.messages.push(message(Role::Notice, text.into()));
        reindex_messages(&mut self.messages);
    }

    pub(crate) fn toggle_sidebar(&mut self, cx: &mut Context<Self>) {
        self.show_sidebar = !self.show_sidebar;
        self.show_sidebar_options = false;
        cx.notify();
    }

    pub(crate) fn set_trajectory(&mut self, trajectory: bool, cx: &mut Context<Self>) {
        self.save_current_view_state();
        self.show_trajectory = trajectory;
        self.composer_menu = None;
        self.restore_current_view_state();
        cx.notify();
    }

    fn view_state_key(&self) -> String {
        let session = if self.current_session.as_os_str().is_empty() {
            "<new>".into()
        } else {
            self.current_session.display().to_string()
        };
        format!("{}\n{session}", self.cwd.display())
    }

    fn save_current_view_state(&mut self) {
        let key = self.view_state_key();
        let state = self.view_states.entry(key).or_default();
        if self.show_trajectory {
            state.trajectory_offset = self.trajectory_scroll.0.borrow().base_handle.offset();
            state.selected_trajectory = self.selected_trajectory;
            state.details_tab = self.details_tab;
            state.details_offset = self.details_scroll.offset();
        } else {
            state.chat_offset = self.scroll.offset();
            state.chat_follow = self.follow_chat_tail;
        }
    }

    fn restore_current_view_state(&mut self) {
        let state = self
            .view_states
            .get(&self.view_state_key())
            .copied()
            .unwrap_or_default();
        if self.show_trajectory {
            self.trajectory_scroll
                .0
                .borrow()
                .base_handle
                .set_offset(state.trajectory_offset);
            self.selected_trajectory = state
                .selected_trajectory
                .filter(|index| *index < self.messages.len());
            self.details_tab = state.details_tab;
            self.details_scroll.set_offset(state.details_offset);
        } else if state.chat_follow {
            self.follow_chat_tail = true;
            self.unread_stream_updates = 0;
            self.scroll.scroll_to_bottom();
        } else {
            self.follow_chat_tail = false;
            self.scroll.set_offset(state.chat_offset);
        }
    }

    pub(crate) fn toggle_session_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.search_sessions = !self.search_sessions;
        if self.search_sessions {
            self.show_sidebar_options = false;
            self.session_search
                .update(cx, |input, cx| input.focus(window, cx));
        } else {
            self.session_search
                .update(cx, |input, cx| input.set_value("", window, cx));
            self.input.update(cx, |input, cx| input.focus(window, cx));
        }
        cx.notify();
    }

    pub(crate) fn toggle_sidebar_options(&mut self, cx: &mut Context<Self>) {
        self.show_sidebar_options = !self.show_sidebar_options;
        cx.notify();
    }

    pub(crate) fn toggle_project(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(path) = self
            .project_store
            .project(index)
            .map(|project| project.path.clone())
        else {
            return;
        };
        if index == self.active_project {
            if !self.expanded_projects.remove(&path) {
                self.expanded_projects.insert(path);
            }
            cx.notify();
        } else {
            self.expanded_projects.insert(path);
            self.switch_project(index, window, cx);
        }
    }

    pub(crate) fn export_session_log(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.current_session.as_os_str().is_empty() {
            self.notice("Start the session before exporting its log");
            cx.notify();
            return;
        }
        let source = self.current_session.clone();
        let suggested = format!("{}.jsonl", safe_file_name(&self.title));
        let receiver = cx.prompt_for_new_path(&self.cwd, Some(&suggested));
        cx.spawn_in(window, async move |this, cx| {
            let selection = receiver.await;
            let copied = match selection {
                Ok(Ok(Some(destination))) => tokio::fs::copy(source, destination)
                    .await
                    .map(|_| Some("Session log exported".to_owned()))
                    .map_err(|error| error.to_string()),
                Ok(Ok(None)) => Ok(None),
                Ok(Err(error)) => Err(error.to_string()),
                Err(error) => Err(error.to_string()),
            };
            let _ = cx.update(|_, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match copied {
                            Ok(Some(message)) => this.notice(message),
                            Ok(None) => {}
                            Err(error) => this.notice(format!("Could not export log: {error}")),
                        }
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    pub(crate) fn toggle_tool(
        &mut self,
        index: usize,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let follow_tail = self.follow_chat_tail;
        let mut expanded = false;
        if let Some(message) = self.messages.get_mut(index)
            && message.role == Role::Tool
        {
            message.expanded = !message.expanded;
            expanded = message.expanded;
        }
        if expanded && follow_tail {
            self.scroll.scroll_to_bottom();
        }
        cx.notify();
    }

    pub(crate) fn toggle_reasoning(&mut self, index: usize, cx: &mut Context<Self>) {
        let follow_tail = self.follow_chat_tail;
        let mut expanded = false;
        if let Some(message) = self.messages.get_mut(index)
            && message.role == Role::Reasoning
        {
            message.expanded = !message.expanded;
            expanded = message.expanded;
        }
        if expanded && follow_tail {
            self.scroll.scroll_to_bottom();
        }
        cx.notify();
    }

    pub(crate) fn inspect_tool(&mut self, index: usize, cx: &mut Context<Self>) {
        if self
            .messages
            .get(index)
            .is_some_and(|message| message.role == Role::Tool)
        {
            self.save_current_view_state();
            self.show_trajectory = true;
            self.selected_trajectory = Some(index);
            self.details_tab = DetailsTab::Summary;
            self.details_scroll.set_offset(point(px(0.0), px(0.0)));
            self.trajectory_collapsed_turns = false;
            self.trajectory_collapsed_calls = false;
            self.scroll_trajectory_to_record(index, cx);
            cx.notify();
        }
    }

    pub(crate) fn rate_message(&mut self, index: usize, positive: bool, cx: &mut Context<Self>) {
        if let Some(message) = self.messages.get_mut(index)
            && message.role == Role::Assistant
        {
            message.rating = (message.rating != Some(positive)).then_some(positive);
            cx.notify();
        }
    }

    pub(crate) fn set_composer_menu(&mut self, menu: Option<ComposerMenu>, cx: &mut Context<Self>) {
        self.show_sidebar_options = false;
        self.composer_menu = menu;
        self.composer_menu_highlight = 0;
        cx.notify();
    }

    pub(crate) fn open_composer_menu(
        &mut self,
        menu: ComposerMenu,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_composer_menu(Some(menu), cx);
        self.composer_menu_focus.focus(window);
    }

    pub(crate) fn move_composer_menu(&mut self, direction: isize, cx: &mut Context<Self>) {
        let count = self.composer_menu_item_count();
        if count == 0 {
            return;
        }
        self.composer_menu_highlight =
            (self.composer_menu_highlight as isize + direction).rem_euclid(count as isize) as usize;
        cx.notify();
    }

    fn composer_menu_item_count(&self) -> usize {
        match self.composer_menu {
            Some(ComposerMenu::Commands) => 3,
            Some(ComposerMenu::Permission | ComposerMenu::Model) => 2,
            Some(ComposerMenu::Models) => self.models.len(),
            Some(ComposerMenu::Effort) => self.models[self.selected_model]
                .model
                .reasoning_efforts()
                .len(),
            Some(ComposerMenu::Workspace) => self.project_store.projects().len() + 1,
            None => 0,
        }
    }

    pub(crate) fn activate_composer_menu(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let index = self.composer_menu_highlight;
        match self.composer_menu {
            Some(ComposerMenu::Commands) => match index {
                0 => {
                    self.composer_menu = None;
                    self.export_session_log(window, cx);
                }
                1 => self.set_composer_menu(Some(ComposerMenu::Permission), cx),
                _ => self.set_composer_menu(Some(ComposerMenu::Model), cx),
            },
            Some(ComposerMenu::Permission) => self.set_allow_all_tools(index == 1, cx),
            Some(ComposerMenu::Model) => self.set_composer_menu(
                Some(if index == 0 {
                    ComposerMenu::Models
                } else {
                    ComposerMenu::Effort
                }),
                cx,
            ),
            Some(ComposerMenu::Models) => self.select_model(index, cx),
            Some(ComposerMenu::Effort) => {
                if let Some(effort) = self.models[self.selected_model]
                    .model
                    .reasoning_efforts()
                    .get(index)
                    .cloned()
                {
                    self.set_reasoning_effort(effort, cx);
                    self.composer_menu = None;
                }
            }
            Some(ComposerMenu::Workspace) => {
                if index < self.project_store.projects().len() {
                    self.composer_menu = None;
                    self.switch_project(index, window, cx);
                } else {
                    self.composer_menu = None;
                    self.add_project(window, cx);
                }
            }
            None => {}
        }
    }

    pub(crate) fn handle_root_key(
        &mut self,
        event: &gpui::KeyDownEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.composer_menu.is_some() {
            match event.keystroke.key.as_str() {
                "escape" => self.dismiss_transient(window, cx),
                "up" | "arrowup" => self.move_composer_menu(-1, cx),
                "down" | "arrowdown" => self.move_composer_menu(1, cx),
                "enter" | "return" => self.activate_composer_menu(window, cx),
                _ => return,
            }
            cx.stop_propagation();
            return;
        }
        if event.keystroke.key == "escape" {
            self.dismiss_transient(window, cx);
        }
    }

    pub(crate) fn dismiss_transient(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.modal.is_some() {
            self.close_modal(window, cx);
            return;
        }
        if matches!(
            self.composer_menu,
            Some(ComposerMenu::Models | ComposerMenu::Effort)
        ) {
            self.composer_menu = Some(ComposerMenu::Model);
        } else {
            self.composer_menu = None;
        }
        self.show_sidebar_options = false;
        self.session_action_target = None;
        if self.search_sessions {
            self.search_sessions = false;
            self.session_search
                .update(cx, |input, cx| input.set_value("", window, cx));
            self.input.update(cx, |input, cx| input.focus(window, cx));
        }
        cx.notify();
    }

    pub(crate) fn set_allow_all_tools(&mut self, allow: bool, cx: &mut Context<Self>) {
        if let Err(error) = self.settings.set_allow_all_tools(allow) {
            self.notice(format!("Could not save permission setting: {error}"));
        }
        self.composer_menu = None;
        cx.notify();
    }

    pub(crate) fn new_chat(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.control.is_some() || self.preparing_session {
            self.notice("Stop the active task before starting a new chat");
            cx.notify();
            return;
        }
        self.save_current_view_state();
        self.activate_session(Session::memory());
        self.restore_current_view_state();
        self.input.update(cx, |input, cx| {
            input.set_placeholder("Describe what you want to build", window, cx);
            input.focus(window, cx);
        });
        cx.notify();
    }

    pub(crate) fn new_chat_in_project(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if index == self.active_project {
            self.new_chat(window, cx);
        } else {
            self.switch_project(index, window, cx);
        }
    }

    pub(crate) fn add_project(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.control.is_some() || self.preparing_session {
            self.notice("Stop the active task before opening another project");
            cx.notify();
            return;
        }
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Open Project".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let selection = receiver.await;
            let _ =
                cx.update(|window, app| {
                    if let Some(this) = this.upgrade() {
                        this.update(app, |this, cx| {
                            match selection {
                                Ok(Ok(Some(paths))) => {
                                    if let Some(path) = paths.into_iter().next() {
                                        match this.project_store.add(path) {
                                            Ok(index) => {
                                                this.refresh_session_search_documents();
                                                this.switch_project(index, window, cx)
                                            }
                                            Err(error) => this
                                                .notice(format!("Could not add project: {error}")),
                                        }
                                    }
                                }
                                Ok(Err(error)) => {
                                    this.notice(format!("Could not open project picker: {error}"))
                                }
                                Err(error) => this
                                    .notice(format!("Project picker closed unexpectedly: {error}")),
                                Ok(Ok(None)) => {}
                            }
                            cx.notify();
                        });
                    }
                });
        })
        .detach();
    }

    pub(crate) fn switch_project(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if index == self.active_project {
            return;
        }
        if self.control.is_some() || self.preparing_session {
            self.notice("Stop the active task before switching projects");
            cx.notify();
            return;
        }
        let Some(project) = self.project_store.project(index).cloned() else {
            return;
        };
        self.save_current_view_state();
        if let Some(agent) = &mut self.agent {
            agent.set_cwd(project.path.clone());
            agent.set_session(Session::memory());
        }
        self.active_project = index;
        self.expanded_projects.insert(project.path.clone());
        self.cwd = project.path;
        self.sessions_dir = project.sessions_dir;
        self.sessions = Session::list(&self.sessions_dir).unwrap_or_default();
        self.refresh_session_search_documents();
        self.reset_conversation();
        self.restore_current_view_state();
        self.input.update(cx, |input, cx| {
            input.set_placeholder("Describe what you want to build", window, cx);
            input.focus(window, cx);
        });
        cx.notify();
    }

    pub(crate) fn remove_project(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.control.is_some() || self.preparing_session {
            self.notice("Stop the active task before removing a project");
            cx.notify();
            return;
        }
        if self.project_store.projects().len() == 1 {
            self.notice("At least one project must remain open");
            cx.notify();
            return;
        }
        if let Err(error) = self.project_store.remove(index) {
            self.notice(format!("Could not remove project: {error}"));
            cx.notify();
            return;
        }
        let next = if index < self.active_project {
            self.active_project - 1
        } else if index == self.active_project {
            index.min(self.project_store.projects().len() - 1)
        } else {
            self.active_project
        };
        self.active_project = usize::MAX;
        self.switch_project(next, window, cx);
    }

    pub(crate) fn open_session(
        &mut self,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if path == self.current_session {
            return;
        }
        if self.control.is_some() || self.preparing_session {
            self.notice("Stop the active task before opening another session");
            cx.notify();
            return;
        }
        self.save_current_view_state();
        cx.spawn_in(window, async move |this, cx| {
            let session = Session::open(path).await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match session {
                            Ok(session) => {
                                this.activate_session(session);
                                this.restore_current_view_state();
                                this.input.update(cx, |input, cx| {
                                    input.set_placeholder("Message the agent", window, cx)
                                });
                            }
                            Err(error) => this.notice(format!("Could not open session: {error}")),
                        }
                        this.input.update(cx, |input, cx| input.focus(window, cx));
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    pub(crate) fn open_project_session(
        &mut self,
        project_index: usize,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if project_index != self.active_project {
            self.switch_project(project_index, window, cx);
        }
        if project_index == self.active_project {
            self.open_session(path, window, cx);
        }
    }

    pub(crate) fn rename_current_session(
        &mut self,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.control.is_some()
            || self.preparing_session
            || self.current_session.as_os_str().is_empty()
        {
            return;
        }
        let Some(mut agent) = self.agent.take() else {
            return;
        };
        cx.spawn_in(window, async move |this, cx| {
            let result = agent.rename_session(&title).await;
            let _ = cx.update(|_, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match result {
                            Ok(()) => {
                                this.title = agent.session_info().title.clone();
                                this.sessions =
                                    Session::list(&this.sessions_dir).unwrap_or_default();
                                this.refresh_session_search_documents();
                            }
                            Err(error) => this.notice(format!("Could not rename session: {error}")),
                        }
                        this.agent = Some(agent);
                        cx.notify();
                    });
                }
            });
        })
        .detach();
    }

    pub(crate) fn delete_current_session(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.control.is_some()
            || self.preparing_session
            || self.current_session.as_os_str().is_empty()
        {
            return;
        }
        let Some(session) = self
            .sessions
            .iter()
            .find(|session| same_path(&session.path, &self.current_session))
            .cloned()
        else {
            return;
        };
        if let Some(agent) = &mut self.agent {
            agent.set_session(Session::memory());
        }
        match Session::delete(&session) {
            Ok(()) => {
                self.sessions = Session::list(&self.sessions_dir).unwrap_or_default();
                self.refresh_session_search_documents();
                self.reset_conversation();
                self.input.update(cx, |input, cx| {
                    input.set_placeholder("Describe what you want to build", window, cx);
                    input.focus(window, cx);
                });
            }
            Err(error) => self.notice(format!("Could not delete session: {error}")),
        }
        cx.notify();
    }

    pub(crate) fn set_reasoning_effort(
        &mut self,
        effort: kcastle_agent::ReasoningEffort,
        cx: &mut Context<Self>,
    ) {
        if self.control.is_some() {
            return;
        }
        let model_id = self.models[self.selected_model].id.clone();
        self.models[self.selected_model]
            .model
            .set_reasoning_effort(effort.clone());
        if let Some(agent) = &mut self.agent {
            agent.set_reasoning_effort(effort.clone());
        }
        if let Err(error) = self.settings.set_effort(&model_id, &effort) {
            self.notice(format!("Could not save settings: {error}"));
        }
        cx.notify();
    }

    pub(crate) fn select_model(&mut self, index: usize, cx: &mut Context<Self>) {
        if self.control.is_some() || index >= self.models.len() || index == self.selected_model {
            return;
        }
        let configured = self.models[index].clone();
        let label = configured.label();
        if let Some(agent) = &mut self.agent {
            agent.set_model(configured.model);
        }
        self.selected_model = index;
        self.model = label;
        if let Err(error) = self.settings.set_selected_model(&configured.id) {
            self.notice(format!("Could not save model selection: {error}"));
        }
        self.composer_menu = None;
        cx.notify();
    }

    pub(crate) fn set_appearance(
        &mut self,
        appearance: Appearance,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Err(error) = self.settings.set_appearance(appearance) {
            self.notice(format!("Could not save appearance: {error}"));
        }
        match appearance {
            Appearance::System => Theme::sync_system_appearance(Some(window), cx),
            Appearance::Light => Theme::change(ThemeMode::Light, Some(window), cx),
            Appearance::Dark => Theme::change(ThemeMode::Dark, Some(window), cx),
        }
        cx.notify();
    }

    pub(crate) fn set_enter_behavior(&mut self, behavior: EnterBehavior, cx: &mut Context<Self>) {
        if let Err(error) = self.settings.set_enter_behavior(behavior) {
            self.notice(format!("Could not save Enter behavior: {error}"));
        }
        cx.notify();
    }

    pub(crate) fn set_reduce_motion(&mut self, reduce: bool, cx: &mut Context<Self>) {
        if let Err(error) = self.settings.set_reduce_motion(reduce) {
            self.notice(format!("Could not save motion preference: {error}"));
        }
        cx.notify();
    }

    pub(crate) fn session_document_matches(&self, path: &Path, query: &str) -> bool {
        self.session_search_documents
            .get(path)
            .is_some_and(|document| document.searchable.contains(query))
    }

    pub(crate) fn session_document_summary(&self, path: &Path, query: &str) -> Option<String> {
        let document = self.session_search_documents.get(path)?;
        matching_search_snippet(&document.snippets, query)
            .or_else(|| (!document.summary.is_empty()).then(|| document.summary.clone()))
    }

    fn refresh_session_search_documents(&mut self) {
        self.session_search_documents = build_session_search_documents(&self.project_store);
    }

    pub(crate) fn chat_at_bottom(&self) -> bool {
        within_bottom_threshold(self.scroll.max_offset().height, self.scroll.offset().y)
    }

    pub(crate) fn handle_chat_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let delta_y = event.delta.pixel_delta(window.line_height()).y;
        if update_chat_follow_on_scroll(
            delta_y,
            self.chat_at_bottom(),
            &mut self.follow_chat_tail,
            &mut self.unread_stream_updates,
        ) {
            cx.notify();
        }
    }

    pub(crate) fn scroll_chat_to_bottom(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.follow_chat_tail = true;
        self.unread_stream_updates = 0;
        self.scroll.scroll_to_bottom();
        cx.notify();
    }

    fn activate_session(&mut self, session: Session) {
        let title = session.info().title.clone();
        let path = session.info().path.clone();
        let transcript = session.state().transcript();
        let usage = session
            .state()
            .latest_response()
            .and_then(|response| response.usage.clone());
        if let Some(agent) = &mut self.agent {
            agent.set_session(session);
        }
        self.messages = messages_from_transcript(transcript);
        for message in &mut self.messages {
            if message.role == Role::Tool
                && let Some(name) = message.title.as_deref()
            {
                message.schema = self.tool_schemas.get(name).cloned();
            }
        }
        reindex_messages(&mut self.messages);
        self.title = if title == "Untitled session" {
            "New chat".into()
        } else {
            title
        };
        self.current_session = path;
        self.sessions = Session::list(&self.sessions_dir).unwrap_or_default();
        self.show_trajectory = false;
        self.selected_trajectory = None;
        self.turns = self
            .messages
            .iter()
            .filter(|message| message.role == Role::User)
            .count();
        self.tool_calls = self
            .messages
            .iter()
            .filter(|message| message.role == Role::Tool)
            .count();
        if let Some(usage) = usage {
            self.input_tokens = usage.input_tokens;
            self.output_tokens = usage.output_tokens;
            self.cached_tokens = usage.input_tokens_details.cached_tokens;
        } else {
            self.input_tokens = 0;
            self.output_tokens = 0;
            self.cached_tokens = 0;
        }
        self.scroll.scroll_to_bottom();
        self.follow_chat_tail = true;
        self.unread_stream_updates = 0;
    }

    fn activate_session_for_run(&mut self, session: Session) {
        let previous_key = self.view_state_key();
        self.current_session = session.info().path.clone();
        let current_key = self.view_state_key();
        if let Some(state) = self.view_states.remove(&previous_key) {
            self.view_states.insert(current_key, state);
        }
        if let Some(agent) = &mut self.agent {
            agent.set_session(session);
        }
        self.sessions = Session::list(&self.sessions_dir).unwrap_or_default();
    }

    fn reset_conversation(&mut self) {
        self.messages.clear();
        self.approval = None;
        self.modal = None;
        self.composer_menu = None;
        self.current_session.clear();
        self.title = "New chat".into();
        self.show_trajectory = false;
        self.selected_trajectory = None;
        self.turns = 0;
        self.tool_calls = 0;
        self.input_tokens = 0;
        self.output_tokens = 0;
        self.cached_tokens = 0;
        self.scroll.scroll_to_bottom();
        self.follow_chat_tail = true;
        self.unread_stream_updates = 0;
    }
}

impl Drop for DesktopApp {
    fn drop(&mut self) {
        if let Some(control) = &self.control {
            control.abort();
        }
    }
}

fn message(role: Role, text: String) -> Message {
    let render_text = SharedString::from(text.clone());
    Message {
        render_key: next_message_render_key(),
        role,
        id: None,
        title: None,
        text,
        render_text,
        payload: None,
        schema: None,
        pending: false,
        failed: false,
        expanded: false,
        rating: None,
        started_at_ms: Some(now_ms()),
        duration_ms: None,
        turn: 0,
        step: 0,
        request_id: None,
        search_text: String::new(),
        streaming_markdown: StreamingMarkdownState::default(),
    }
}

fn messages_from_transcript(transcript: Vec<TranscriptItem>) -> Vec<Message> {
    let mut messages = Vec::new();
    for item in transcript {
        match item {
            TranscriptItem::User(text) => messages.push(restored_message(Role::User, text)),
            TranscriptItem::Reasoning(text) => {
                let mut message = restored_message(Role::Reasoning, text);
                message.title = Some("Think".into());
                messages.push(message);
            }
            TranscriptItem::Assistant(text) => {
                messages.push(restored_message(Role::Assistant, text))
            }
            TranscriptItem::ToolCall {
                call_id,
                name,
                arguments,
            } => messages.push(Message {
                render_key: next_message_render_key(),
                role: Role::Tool,
                id: Some(call_id),
                title: Some(name),
                text: String::new(),
                render_text: SharedString::default(),
                payload: Some(arguments),
                schema: None,
                pending: true,
                failed: false,
                expanded: false,
                rating: None,
                started_at_ms: None,
                duration_ms: None,
                turn: 0,
                step: 0,
                request_id: None,
                search_text: String::new(),
                streaming_markdown: StreamingMarkdownState::default(),
            }),
            TranscriptItem::ToolOutput { call_id, output } => {
                if let Some(message) = messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.id.as_deref() == Some(&call_id))
                {
                    message.text = output;
                    message.render_text = message.text.clone().into();
                    message.pending = false;
                    message.failed = restored_tool_failed(&message.text);
                }
            }
            TranscriptItem::Summary(text) => messages.push(restored_message(Role::Notice, text)),
        }
    }
    reindex_messages(&mut messages);
    messages
}

fn restored_message(role: Role, text: String) -> Message {
    let mut message = message(role, text);
    message.started_at_ms = None;
    if message.role == Role::Assistant {
        message.streaming_markdown.update(&message.text);
    }
    message
}

fn restored_tool_failed(output: &str) -> bool {
    let first_line = output.lines().next().unwrap_or_default();
    output.starts_with("Tool call denied by user")
        || output.starts_with("Tool execution was interrupted")
        || first_line
            .strip_prefix("exit_code=")
            .is_some_and(|code| code.trim() != "0")
}

fn push_delta(messages: &mut Vec<Message>, delta: &str) {
    if let Some(reasoning) = messages
        .last_mut()
        .filter(|message| message.role == Role::Reasoning)
    {
        reasoning.pending = false;
    }
    if let Some(message) = messages
        .last_mut()
        .filter(|message| message.role == Role::Assistant)
    {
        message.text.push_str(delta);
    } else {
        let mut assistant = message(Role::Assistant, delta.to_owned());
        assistant.pending = true;
        messages.push(assistant);
    }
}

fn push_reasoning_delta(messages: &mut Vec<Message>, delta: &str) {
    if let Some(assistant) = messages
        .last_mut()
        .filter(|message| message.role == Role::Assistant)
    {
        assistant.pending = false;
    }
    if let Some(message) = messages
        .last_mut()
        .filter(|message| message.role == Role::Reasoning)
    {
        message.text.push_str(delta);
    } else {
        let mut reasoning = message(Role::Reasoning, delta.to_owned());
        reasoning.title = Some("Think".into());
        reasoning.pending = true;
        messages.push(reasoning);
    }
}

fn settle_active_response_message(messages: &mut [Message]) {
    if let Some(message) = messages
        .last_mut()
        .filter(|message| matches!(message.role, Role::Reasoning | Role::Assistant))
    {
        message.pending = false;
    }
}

fn refresh_live_render_state(messages: &mut [Message]) {
    for message in messages.iter_mut().rev() {
        if message.role == Role::User {
            break;
        }
        if !matches!(message.role, Role::Reasoning | Role::Assistant) {
            continue;
        }
        if message.render_text.as_ref() != message.text {
            message.render_text = message.text.clone().into();
            refresh_message_search_text(message);
        }
        if message.role == Role::Assistant {
            message.streaming_markdown.update(&message.text);
        }
    }
}

fn refresh_message_search_text(message: &mut Message) {
    message.search_text = [
        message.title.as_deref().unwrap_or_default(),
        message.payload.as_deref().unwrap_or_default(),
        message.schema.as_deref().unwrap_or_default(),
        message.text.as_str(),
    ]
    .join("\n")
    .to_lowercase();
}

pub(crate) fn reindex_messages(messages: &mut [Message]) {
    let mut turn = 0;
    let mut step = 0;
    let mut assistant_phase = false;
    for message in messages {
        match message.role {
            Role::User => {
                turn += 1;
                step = 0;
                assistant_phase = false;
            }
            Role::Reasoning | Role::Assistant => {
                if !assistant_phase {
                    step += 1;
                    assistant_phase = true;
                }
            }
            Role::Tool | Role::Notice => assistant_phase = false,
        }
        message.turn = turn;
        message.step = step;
        if message.request_id.is_none() && turn > 0 {
            message.request_id = Some(if step > 0 {
                format!("turn-{turn}-step-{step}")
            } else {
                format!("turn-{turn}")
            });
        }
        refresh_message_search_text(message);
    }
}

pub(crate) fn step_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|message| (message.turn, message.step))
        .filter(|(_, step)| *step > 0)
        .collect::<HashSet<_>>()
        .len()
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn tool_schema_map(agent: &Agent) -> HashMap<String, String> {
    agent
        .tool_schemas()
        .into_iter()
        .filter_map(|schema| {
            let value = serde_json::to_value(schema).ok()?;
            let function = value.get("function").unwrap_or(&value);
            let name = function.get("name")?.as_str()?.to_owned();
            let display = serde_json::to_string_pretty(function).ok()?;
            Some((name, display))
        })
        .collect()
}

fn short_title(value: &str) -> String {
    const LIMIT: usize = 42;
    let mut chars = value.chars();
    let title: String = chars.by_ref().take(LIMIT).collect();
    if chars.next().is_some() {
        format!("{}…", title.trim_end())
    } else {
        title
    }
}

fn safe_file_name(value: &str) -> String {
    let name = value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() || matches!(character, '-' | '_' | ' ') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let name = name.trim().trim_matches('-');
    if name.is_empty() {
        "session".into()
    } else {
        name.into()
    }
}

fn within_bottom_threshold(max_offset: gpui::Pixels, offset_y: gpui::Pixels) -> bool {
    max_offset + offset_y <= px(24.0)
}

fn update_chat_follow_on_scroll(
    delta_y: gpui::Pixels,
    at_bottom: bool,
    follow_chat_tail: &mut bool,
    unread_stream_updates: &mut usize,
) -> bool {
    if delta_y > px(0.0) && *follow_chat_tail {
        *follow_chat_tail = false;
        true
    } else if delta_y < px(0.0) && at_bottom {
        let changed = !*follow_chat_tail || *unread_stream_updates > 0;
        *follow_chat_tail = true;
        *unread_stream_updates = 0;
        changed
    } else {
        false
    }
}

pub(crate) fn session_age(created_at: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let elapsed = now.saturating_sub(created_at);
    match elapsed {
        0..60 => "now".into(),
        60..3600 => format!("{}m", elapsed / 60),
        3600..86400 => format!("{}h", elapsed / 3600),
        _ => format!("{}d", elapsed / 86400),
    }
}

pub(crate) fn same_path(left: &Path, right: &Path) -> bool {
    left == right
}

fn build_session_search_documents(
    project_store: &ProjectStore,
) -> HashMap<PathBuf, SessionSearchDocument> {
    let mut documents = HashMap::new();
    for project in project_store.projects() {
        for session in Session::list(&project.sessions_dir).unwrap_or_default() {
            let Ok(contents) = std::fs::read_to_string(&session.path) else {
                continue;
            };
            let mut values = Vec::new();
            for line in contents.lines() {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                    collect_search_strings(&value, &mut values);
                }
            }
            let summary = values
                .iter()
                .find(|value| value.chars().count() >= 4 && !value.starts_with("resp_"))
                .map(|value| truncate_chars(value, 88))
                .unwrap_or_default();
            documents.insert(
                session.path,
                SessionSearchDocument {
                    searchable: values.join("\n").to_lowercase(),
                    summary,
                    snippets: values,
                },
            );
        }
    }
    documents
}

fn collect_search_strings(value: &serde_json::Value, output: &mut Vec<String>) {
    match value {
        serde_json::Value::String(value) => {
            if !value.trim().is_empty() {
                output.push(value.clone());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_search_strings(value, output);
            }
        }
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if !matches!(key.as_str(), "id" | "call_id" | "created_at" | "model") {
                    collect_search_strings(value, output);
                }
            }
        }
        _ => {}
    }
}

fn truncate_chars(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let text = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{}…", text.trim_end())
    } else {
        text
    }
}

fn matching_search_snippet(values: &[String], query: &str) -> Option<String> {
    values
        .iter()
        .find(|value| value.to_lowercase().contains(query))
        .map(|value| truncate_chars(value, 88))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_deltas_share_one_message() {
        let mut messages = Vec::new();
        push_delta(&mut messages, "hello");
        push_delta(&mut messages, " world");
        refresh_live_render_state(&mut messages);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "hello world");
        assert_eq!(messages[0].render_text.as_ref(), "hello world");
        assert_eq!(messages[0].streaming_markdown.revision(), 1);
    }

    #[test]
    fn markdown_state_is_ready_when_a_delta_and_settle_share_one_frame() {
        let mut messages = Vec::new();
        push_delta(&mut messages, "## Result\n\n- one\n- two");
        settle_active_response_message(&mut messages);
        refresh_live_render_state(&mut messages);

        assert!(!messages[0].pending);
        assert!(!messages[0].streaming_markdown.tail_blocks().is_empty());
    }

    #[test]
    fn restored_assistant_messages_use_the_semantic_markdown_renderer() {
        let message = restored_message(Role::Assistant, "## Result\n\nbody".into());
        assert!(!message.streaming_markdown.tail_blocks().is_empty());
    }

    #[test]
    fn only_visible_stream_deltas_wait_for_the_next_frame() {
        assert!(is_frame_stream_event(&AgentEvent::TextDelta("x".into())));
        assert!(is_frame_stream_event(&AgentEvent::ReasoningDelta(
            "x".into()
        )));
        assert!(!is_frame_stream_event(&AgentEvent::RunAborted));
    }

    #[gpui::test]
    fn stream_frame_gate_can_be_armed_outside_the_draw_phase(cx: &mut gpui::TestAppContext) {
        let window = cx.add_empty_window();
        let (ready_tx, _ready_rx) = tokio::sync::oneshot::channel();

        window.update(|window, _| arm_stream_frame(window, ready_tx));
    }

    #[test]
    fn message_render_keys_do_not_alias_across_session_reloads() {
        let first = message(Role::Assistant, "first".into());
        let second = message(Role::Assistant, "second".into());
        assert_ne!(first.render_key, second.render_key);
    }

    #[test]
    fn reasoning_and_answer_streams_stay_separate() {
        let mut messages = Vec::new();
        push_reasoning_delta(&mut messages, "inspect ");
        push_reasoning_delta(&mut messages, "workspace");
        push_delta(&mut messages, "done");

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::Reasoning);
        assert_eq!(messages[0].text, "inspect workspace");
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(messages[1].text, "done");
    }

    #[test]
    fn only_the_current_response_segment_remains_streaming() {
        let mut messages = Vec::new();
        push_delta(&mut messages, "before tool");
        settle_active_response_message(&mut messages);
        push_reasoning_delta(&mut messages, "after tool");

        assert!(!messages[0].pending);
        assert!(messages[1].pending);
    }

    #[test]
    fn titles_are_unicode_safe() {
        let input =
            "这是一个很长的中文会话标题，用于验证按字符截断不会破坏 UTF-8，而且确实会显示省略号";
        let title = short_title(input);
        assert!(title.ends_with('…'));
        assert_eq!(title.chars().count(), 43);
    }

    #[test]
    fn exported_session_names_are_safe() {
        assert_eq!(safe_file_name("Fix: auth/login?"), "Fix- auth-login");
        assert_eq!(safe_file_name("\u{4f1a}\u{8bdd}"), "\u{4f1a}\u{8bdd}");
    }

    #[test]
    fn restored_tool_failures_keep_their_status() {
        assert!(restored_tool_failed("Tool call denied by user"));
        assert!(restored_tool_failed("exit_code=2\ncommand failed"));
        assert!(!restored_tool_failed("exit_code=0\nall good"));
    }

    #[test]
    fn streaming_only_follows_when_the_view_is_near_the_tail() {
        assert!(within_bottom_threshold(px(500.0), px(-480.0)));
        assert!(within_bottom_threshold(px(500.0), px(-500.0)));
        assert!(!within_bottom_threshold(px(500.0), px(-450.0)));
    }

    #[test]
    fn repeated_scroll_events_do_not_request_redundant_chat_renders() {
        let mut follow = true;
        let mut unread = 0;

        assert!(update_chat_follow_on_scroll(
            px(1.0),
            false,
            &mut follow,
            &mut unread,
        ));
        assert!(!follow);

        assert!(!update_chat_follow_on_scroll(
            px(1.0),
            false,
            &mut follow,
            &mut unread,
        ));
        assert!(!follow);
        assert_eq!(unread, 0);

        unread = 3;
        assert!(update_chat_follow_on_scroll(
            px(-1.0),
            true,
            &mut follow,
            &mut unread,
        ));
        assert!(follow);
        assert_eq!(unread, 0);

        assert!(!update_chat_follow_on_scroll(
            px(-1.0),
            true,
            &mut follow,
            &mut unread,
        ));
    }

    #[test]
    fn reasoning_and_answer_share_one_semantic_step() {
        let mut messages = vec![
            message(Role::User, "inspect".into()),
            message(Role::Reasoning, "thinking".into()),
            message(Role::Assistant, "calling a tool".into()),
            message(Role::Tool, "done".into()),
            message(Role::Reasoning, "checking result".into()),
            message(Role::Assistant, "finished".into()),
        ];
        reindex_messages(&mut messages);

        assert_eq!(messages[1].step, 1);
        assert_eq!(messages[2].step, 1);
        assert_eq!(messages[4].step, 2);
        assert_eq!(messages[5].step, 2);
        assert_eq!(step_count(&messages), 2);
    }

    #[test]
    fn message_search_cache_covers_payload_result_and_schema() {
        let mut tool = message(Role::Tool, "compiled successfully".into());
        tool.title = Some("shell".into());
        tool.payload = Some(r#"{"command":"cargo test"}"#.into());
        tool.schema = Some(r#"{"description":"Run a command"}"#.into());
        reindex_messages(std::slice::from_mut(&mut tool));

        assert!(tool.search_text.contains("cargo test"));
        assert!(tool.search_text.contains("compiled successfully"));
        assert!(tool.search_text.contains("run a command"));
    }

    #[test]
    fn search_snippets_are_compact_and_unicode_safe() {
        assert_eq!(truncate_chars("  hello   world  ", 20), "hello world");
        assert_eq!(truncate_chars("中文会话内容", 4), "中文会话…");
        assert_eq!(
            matching_search_snippet(
                &["session".into(), "The requested ambiguous phrase".into()],
                "ambiguous"
            ),
            Some("The requested ambiguous phrase".into())
        );
    }
}
