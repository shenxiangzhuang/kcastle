use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpui::{
    AppContext, Context, Entity, FocusHandle, PathPromptOptions, Pixels, Point, ScrollHandle,
    ScrollWheelEvent, Subscription, UniformListScrollHandle, Window, point, px,
};
use gpui_component::input::{InputEvent, InputState};
use gpui_component::{Theme, ThemeMode};
use kcastle_agent::{
    Agent, AgentEvent, Model, RunControl, Session, SessionInfo, ToolResult, TranscriptItem,
};

use crate::application::{
    MAX_EVENTS_PER_FRAME, StreamBatch, StreamTelemetry, is_frame_stream_event,
};
use crate::dialogs::Modal;
use crate::domain::{
    Action, AppState, ApprovalState, ComposerMenu, ConversationAction, ConversationState,
    DetailsTab, Effect, Message, MessageId, Role, RunId, RunState, ScrollIntent,
    SessionOperationKind, Surface, UsageSnapshot, reduce, reindex_messages,
};
use crate::layout::{LayoutInput, ScrollAnchor, ScrollRestore, resolve_scroll_restore};
use crate::platform::NativeTitlebarController;
use crate::platform::gpui::{
    GpuiLayoutRuntime, MeasuredBounds, MessagePresentationStore, arm_next_frame, run_effects,
};
use crate::project::ProjectStore;
use crate::settings::{Appearance, EnterBehavior, ProviderModel, SettingsStore};

#[derive(Clone)]
pub(crate) struct ConfiguredModel {
    pub(crate) id: String,
    pub(crate) model: Model,
    pub(crate) provider_id: String,
    pub(crate) profile: ProviderModel,
}

impl ConfiguredModel {
    pub(crate) fn new(
        provider_id: impl Into<String>,
        profile: ProviderModel,
        model: Model,
    ) -> Self {
        let provider_id = provider_id.into();
        Self {
            id: format!("{provider_id}/{}", profile.model_id),
            model,
            provider_id,
            profile,
        }
    }

    pub(crate) fn label(&self) -> String {
        let model = if self.profile.display_name.trim().is_empty() {
            &self.profile.model_id
        } else {
            &self.profile.display_name
        };
        format!("{} · {model}", self.model.name())
    }
}

static NEXT_MESSAGE_RENDER_KEY: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_message_render_key() -> MessageId {
    MessageId(NEXT_MESSAGE_RENDER_KEY.fetch_add(1, Ordering::Relaxed))
}

#[derive(Clone, Copy, Debug)]
struct SessionViewState {
    chat_anchor: ScrollAnchor,
    trajectory_offset: Point<Pixels>,
    details_offset: Point<Pixels>,
    selected_trajectory: Option<MessageId>,
    details_tab: DetailsTab,
}

impl Default for SessionViewState {
    fn default() -> Self {
        Self {
            chat_anchor: ScrollAnchor::Tail,
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

pub(crate) struct DesktopStartup {
    pub(crate) agent: Agent,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) project_store: ProjectStore,
    pub(crate) active_project: usize,
    pub(crate) settings: SettingsStore,
}

pub(crate) struct DesktopApp {
    pub(crate) core: AppState,
    pub(crate) layout_runtime: GpuiLayoutRuntime,
    pub(crate) message_presentations: MessagePresentationStore,
    pub(crate) agent: Option<Agent>,
    pub(crate) control: Option<RunControl>,
    pub(crate) input: Entity<InputState>,
    pub(crate) session_search: Entity<InputState>,
    pub(crate) trajectory_search: Entity<InputState>,
    pub(crate) modal: Option<Modal>,
    pub(crate) modal_focus: FocusHandle,
    pub(crate) composer_menu_focus: FocusHandle,
    pub(crate) scroll: ScrollHandle,
    pub(crate) trajectory_scroll: UniformListScrollHandle,
    pub(crate) details_scroll: ScrollHandle,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) model: String,
    pub(crate) project_store: ProjectStore,
    pub(crate) settings: SettingsStore,
    pub(crate) started_at: Option<Instant>,
    pub(crate) stream_telemetry: StreamTelemetry,
    pub(crate) tool_schemas: HashMap<String, String>,
    pub(crate) project_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
    pub(crate) session_activity: HashMap<PathBuf, u64>,
    pub(crate) session_search_documents: HashMap<PathBuf, SessionSearchDocument>,
    native_titlebar: NativeTitlebarController,
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
        let model = models[selected_model].label();
        let tool_schemas = tool_schema_map(&agent);
        let current_session = agent.session_info().path.clone();
        let project_sessions = load_project_sessions(&project_store);
        let session_activity = load_session_activity(&project_sessions);
        let sessions = project_sessions
            .get(&project.sessions_dir)
            .cloned()
            .unwrap_or_default();
        let viewport = window.viewport_size();
        let mut core = AppState::new(LayoutInput {
            viewport_width: f32::from(viewport.width),
            viewport_height: f32::from(viewport.height),
            rem_size: f32::from(window.rem_size()),
            ..LayoutInput::default()
        });
        core.workspace.cwd = project.path.clone();
        core.workspace.active_project = active_project;
        core.workspace.expanded_projects = HashSet::from([project.path.clone()]);
        core.workspace.sessions_dir = project.sessions_dir.clone();
        core.session.current = current_session.clone();
        core.session.sessions = sessions;
        let session_search_documents =
            build_session_search_documents(&project_store, &project_sessions);
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
                    if this.core.composer.menu.is_none() {
                        this.submit(window, cx);
                    }
                }
                InputEvent::Change => {
                    if this.core.follow_chat_tail {
                        this.apply_chat_tail();
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
        let native_titlebar = NativeTitlebarController::install(window);
        let appearance_subscription = cx.observe_window_appearance(window, |this, window, cx| {
            if this.settings.appearance() == Appearance::System {
                Theme::sync_system_appearance(Some(window), cx);
                cx.notify();
            }
        });
        let bounds_subscription = cx.observe_window_bounds(window, |this, window, cx| {
            this.native_titlebar.sync(window);
            this.sync_window_layout(window, cx);
        });
        Self {
            core,
            layout_runtime: GpuiLayoutRuntime::default(),
            message_presentations: MessagePresentationStore::default(),
            agent: Some(agent),
            control: None,
            input,
            session_search,
            trajectory_search,
            modal: None,
            modal_focus: cx.focus_handle(),
            composer_menu_focus: cx.focus_handle(),
            scroll: ScrollHandle::new(),
            trajectory_scroll: UniformListScrollHandle::new(),
            details_scroll: ScrollHandle::new(),
            models,
            selected_model,
            model,
            project_store,
            settings,
            started_at: None,
            stream_telemetry: StreamTelemetry::default(),
            tool_schemas,
            project_sessions,
            session_activity,
            session_search_documents,
            native_titlebar,
            view_states: HashMap::new(),
            _subscriptions: vec![
                subscription,
                search_subscription,
                trajectory_subscription,
                appearance_subscription,
                bounds_subscription,
            ],
        }
    }

    fn sync_window_layout(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let viewport = window.viewport_size();
        let mut input = self.core.layout_input;
        input.viewport_width = f32::from(viewport.width);
        input.viewport_height = f32::from(viewport.height);
        input.rem_size = f32::from(window.rem_size());
        input.sidebar_requested = self.core.sidebar_requested;
        input.trajectory_visible = self.core.surface == Surface::Trajectory;
        input.details_visible = self.core.details.selected.is_some();
        self.dispatch(Action::LayoutInputChanged(input), window, cx);
    }

    pub(crate) fn dispatch(&mut self, action: Action, window: &mut Window, cx: &mut Context<Self>) {
        let effects = self.transition(action);
        run_effects(self, effects, window, cx);
        cx.notify();
    }

    fn transition(&mut self, action: Action) -> Vec<Effect> {
        let previous_generation = self.core.layout_generation;
        let anchor = self
            .layout_runtime
            .capture_chat_anchor(previous_generation, self.core.follow_chat_tail);
        let mut effects = reduce(&mut self.core, action);
        if self.core.layout_generation != previous_generation {
            self.layout_runtime.pending_chat_anchor = Some((self.core.layout_generation, anchor));
            self.layout_runtime.restore_scheduled = false;
            effects.retain(|effect| !matches!(effect, Effect::ApplyChatTail));
        }
        effects
    }

    pub(crate) fn dispatch_local(&mut self, action: Action, cx: &mut Context<Self>) {
        for effect in self.transition(action) {
            match effect {
                Effect::ApplyChatTail => self.apply_chat_tail(),
                effect => panic!("async effect {effect:?} requires a Window dispatch"),
            }
        }
        cx.notify();
    }

    pub(crate) fn task_active(&self) -> bool {
        matches!(
            self.core.run,
            RunState::CreatingSession { .. } | RunState::Running { .. }
        ) || self.core.pending_session_operation.is_some()
    }

    pub(crate) fn update_composer_measurement(
        &mut self,
        height: f32,
        cx: &mut Context<Self>,
    ) -> bool {
        if !height.is_finite()
            || height <= 0.0
            || (self.core.layout_input.composer_height - height).abs() < 0.5
        {
            return false;
        }
        let mut input = self.core.layout_input;
        input.composer_height = height;
        let effects = self.transition(Action::LayoutInputChanged(input));
        let restore_tail = effects
            .iter()
            .any(|effect| matches!(effect, Effect::ApplyChatTail));
        debug_assert!(
            effects
                .iter()
                .all(|effect| matches!(effect, Effect::ApplyChatTail)),
            "layout measurement produced a non-layout effect"
        );
        cx.notify();
        restore_tail
    }

    pub(crate) fn update_main_measurement(&mut self, width: f32, cx: &mut Context<Self>) -> bool {
        if !width.is_finite()
            || width <= 0.0
            || (self.core.layout_input.measured_main_width - width).abs() < 0.5
        {
            return false;
        }
        let mut input = self.core.layout_input;
        input.measured_main_width = width;
        let changed_before = self.core.layout_generation;
        let _ = self.transition(Action::LayoutInputChanged(input));
        let changed = self.core.layout_generation != changed_before;
        if changed {
            cx.notify();
        }
        changed
    }

    pub(crate) fn restore_chat_tail_after_layout(&mut self, cx: &mut Context<Self>) {
        if self.core.follow_chat_tail {
            self.apply_chat_tail();
            cx.notify();
        }
    }

    pub(crate) fn observe_transcript_bounds(
        &mut self,
        bounds: MeasuredBounds,
        _cx: &mut Context<Self>,
    ) -> bool {
        self.layout_runtime
            .observe_transcript(self.core.layout_generation, bounds);
        self.can_restore_pending_chat_anchor()
    }

    pub(crate) fn observe_message_bounds(
        &mut self,
        id: crate::domain::MessageId,
        bounds: MeasuredBounds,
        _cx: &mut Context<Self>,
    ) -> bool {
        self.layout_runtime
            .observe_message(self.core.layout_generation, id, bounds);
        self.can_restore_pending_chat_anchor()
    }

    fn can_restore_pending_chat_anchor(&mut self) -> bool {
        if self.layout_runtime.restore_scheduled {
            return false;
        }
        let Some((generation, anchor)) = self.layout_runtime.pending_chat_anchor else {
            return false;
        };
        let can_restore =
            match resolve_scroll_restore(generation, self.core.layout_generation, anchor) {
                ScrollRestore::Tail => self
                    .layout_runtime
                    .has_current_transcript(self.core.layout_generation),
                ScrollRestore::Message { .. } => self
                    .layout_runtime
                    .restored_offset_y(generation, anchor, f32::from(self.scroll.offset().y))
                    .is_some(),
                ScrollRestore::IgnoreStale => true,
            };
        if can_restore {
            self.layout_runtime.restore_scheduled = true;
        }
        can_restore
    }

    pub(crate) fn apply_pending_chat_anchor(&mut self, cx: &mut Context<Self>) {
        self.layout_runtime.restore_scheduled = false;
        let Some((generation, anchor)) = self.layout_runtime.pending_chat_anchor else {
            return;
        };
        match resolve_scroll_restore(generation, self.core.layout_generation, anchor) {
            ScrollRestore::Tail => {
                self.apply_chat_tail();
                self.layout_runtime.pending_chat_anchor = None;
            }
            ScrollRestore::Message { .. } => {
                if let Some(offset_y) = self.layout_runtime.restored_offset_y(
                    generation,
                    anchor,
                    f32::from(self.scroll.offset().y),
                ) {
                    let offset = self.scroll.offset();
                    self.scroll.set_offset(point(offset.x, px(offset_y)));
                    self.layout_runtime.pending_chat_anchor = None;
                }
            }
            ScrollRestore::IgnoreStale => self.layout_runtime.pending_chat_anchor = None,
        }
        cx.notify();
    }

    pub(crate) fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if matches!(self.core.run, RunState::CreatingSession { .. })
            || self.core.pending_session_operation.is_some()
        {
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
        self.dispatch(
            Action::Conversation(ConversationAction::SubmitUser(message(
                Role::User,
                value.clone(),
            ))),
            window,
            cx,
        );
        self.sync_message_presentations();
        self.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);

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

        let action = if self.core.session.current.as_os_str().is_empty() {
            Action::BeginSessionCreation(value)
        } else {
            Action::BeginRun(value)
        };
        self.dispatch(action, window, cx);
    }

    pub(crate) fn create_session_for_run(
        &mut self,
        operation: crate::domain::OperationId,
        retry_value: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let sessions_dir = self.core.workspace.sessions_dir.clone();
        cx.spawn_in(window, async move |this, cx| {
            let session = Session::create(&sessions_dir).await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| match session {
                        Ok(session) => {
                            let previous_key = this.view_state_key();
                            let current_session = session.info().path.clone();
                            let sessions = this.reload_sessions(&sessions_dir);
                            let effects = this.transition(Action::SessionCreated {
                                operation,
                                current_session,
                                sessions,
                            });
                            if effects
                                .iter()
                                .any(|effect| matches!(effect, Effect::StartRun { .. }))
                            {
                                this.activate_session_for_run(session, previous_key);
                                run_effects(this, effects, window, cx);
                            }
                            cx.notify();
                        }
                        Err(error) => {
                            let message = error.to_string();
                            let before = this.core.run.clone();
                            let effects = this.transition(Action::SessionCreationFailed {
                                operation,
                                message: message.clone(),
                            });
                            if before != this.core.run {
                                let _ = this.transition(Action::Conversation(
                                    ConversationAction::RollbackSubmittedUser,
                                ));
                                this.input.update(cx, |input, cx| {
                                    input.set_value(&retry_value, window, cx);
                                    input.set_placeholder(
                                        "Describe what you want to build",
                                        window,
                                        cx,
                                    );
                                });
                                this.notice(format!("Could not create session: {message}"));
                            }
                            run_effects(this, effects, window, cx);
                            cx.notify();
                        }
                    });
                }
            });
        })
        .detach();
    }

    pub(crate) fn start_run(
        &mut self,
        run: RunId,
        value: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(agent) = self.agent.take() else {
            self.dispatch(
                Action::RunStartFailed {
                    run,
                    message: "Agent is unavailable".into(),
                },
                window,
                cx,
            );
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
                let mut batch = StreamBatch::new(first);
                if collect_frame {
                    let (frame_tx, mut frame_rx) = tokio::sync::oneshot::channel();
                    cx.update(|window, _| {
                        arm_next_frame(window, frame_tx);
                    })
                    .ok();
                    let mut reached_frame = false;
                    let mut reached_structure = false;
                    while batch.len() < MAX_EVENTS_PER_FRAME {
                        tokio::select! {
                            biased;
                            _ = &mut frame_rx => {
                                reached_frame = true;
                                break;
                            }
                            event = active.next_event() => match event {
                            Some(event) => {
                                let structural = !is_frame_stream_event(&event);
                                batch.push(event);
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
                let _ = cx.update(|window, app| {
                    if let Some(entity) = this.upgrade() {
                        entity.update(app, |this, cx| {
                            if !matches!(this.core.run, RunState::Running { run: active } if active == run)
                            {
                                return;
                            }
                            let previous_len = this.core.conversation.messages.len();
                            let delta_count = batch.raw_delta_count();
                            this.stream_telemetry.record(&batch);
                            for event in batch.into_events() {
                                this.apply_event(event);
                            }
                            if delta_count > 0 {
                                let _ = this.transition(Action::Conversation(
                                    ConversationAction::RefreshLiveSearch,
                                ));
                            }
                            if this.core.conversation.messages.len() != previous_len {
                                reindex_messages(&mut this.core.conversation.messages);
                            }
                            this.sync_message_presentations();
                            let effects =
                                this.transition(Action::StreamDeltasReceived(delta_count));
                            run_effects(this, effects, window, cx);
                            cx.notify();
                        });
                    }
                });
            }
            let agent = active.finish().await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        if !matches!(this.core.run, RunState::Running { run: active } if active == run)
                        {
                            return;
                        }
                        let effects = this.transition(Action::RunFinished(run));
                        this.control = None;
                        this.started_at = None;
                        match agent {
                            Ok(agent) => {
                                let sessions = this.reload_active_sessions();
                                this.dispatch(
                                    Action::SetCurrentSession(agent.session_info().path.clone()),
                                    window,
                                    cx,
                                );
                                this.dispatch(
                                    Action::RefreshSessions(sessions),
                                    window,
                                    cx,
                                );
                                this.refresh_session_search_documents();
                                this.agent = Some(agent);
                            }
                            Err(error) => this.notice(error.to_string()),
                        }
                        run_effects(this, effects, window, cx);
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
            AgentEvent::ReasoningDelta(delta) => {
                let _ = self.transition(Action::Conversation(ConversationAction::ReasoningDelta {
                    new_message: message(Role::Reasoning, delta.clone()),
                    delta,
                }));
            }
            AgentEvent::TextDelta(delta) => {
                let _ = self.transition(Action::Conversation(ConversationAction::TextDelta {
                    new_message: message(Role::Assistant, delta.clone()),
                    delta,
                }));
            }
            AgentEvent::ApprovalRequired(call) => {
                if self.settings.allow_all_tools() {
                    if let Some(control) = &self.control
                        && let Err(error) = control.approve(call.call_id, true)
                    {
                        self.notice(error.to_string());
                    }
                } else {
                    let _ = self.transition(Action::SetApproval(Some(ApprovalState {
                        call_id: call.call_id,
                        name: call.name,
                        arguments: call.arguments,
                    })));
                }
            }
            AgentEvent::ToolStarted(call) => {
                let schema = self.tool_schemas.get(&call.name).cloned();
                let _ = self.transition(Action::Conversation(ConversationAction::ToolStarted(
                    Message {
                        key: next_message_render_key(),
                        revision: 0,
                        role: Role::Tool,
                        tool_call_id: Some(call.call_id),
                        title: Some(call.name),
                        text: String::new(),
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
                    },
                )));
            }
            AgentEvent::ToolFinished { call, result } => self.tool_result(&call.call_id, result),
            AgentEvent::RunFinished(summary) => {
                let usage = summary.usage.map(|usage| UsageSnapshot {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cached_tokens: usage.input_tokens_details.cached_tokens,
                });
                let _ = self.transition(Action::Conversation(ConversationAction::RunFinished {
                    response_id: summary.response_id,
                    usage,
                }));
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
        let _ = self.transition(Action::Conversation(ConversationAction::FinishReasoning));
    }

    pub(crate) fn decide(&mut self, call_id: String, allow: bool, cx: &mut Context<Self>) {
        if let Some(control) = &self.control
            && let Err(error) = control.approve(call_id, allow)
        {
            self.notice(error.to_string());
        }
        self.dispatch_local(Action::SetApproval(None), cx);
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
        let duration_ms = self
            .core
            .conversation
            .messages
            .iter()
            .rev()
            .find(|message| message.tool_call_id.as_deref() == Some(call_id))
            .and_then(|message| {
                message
                    .started_at_ms
                    .map(|started| now_ms().saturating_sub(started))
            });
        let _ = self.transition(Action::Conversation(ConversationAction::ToolFinished {
            call_id: call_id.to_owned(),
            output: result.output,
            is_error: result.is_error,
            duration_ms,
        }));
    }

    pub(crate) fn notice(&mut self, text: impl Into<String>) {
        let _ = self.transition(Action::Conversation(ConversationAction::AppendNotice(
            message(Role::Notice, text.into()),
        )));
        self.sync_message_presentations();
    }

    fn sync_message_presentations(&mut self) {
        let messages = &self.core.conversation.messages;
        self.message_presentations
            .sync(messages.iter().map(|message| {
                (
                    message.key,
                    message.text.as_str(),
                    message.role == Role::Assistant,
                )
            }));
    }

    pub(crate) fn toggle_sidebar(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(Action::CloseTransientOverlays, window, cx);
        self.dispatch(Action::ToggleSidebar, window, cx);
    }

    pub(crate) fn set_trajectory(
        &mut self,
        trajectory: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.save_current_view_state();
        self.dispatch(Action::SetComposerMenu(None), window, cx);
        self.dispatch(
            if trajectory {
                Action::ShowTrajectory
            } else {
                Action::ShowChat
            },
            window,
            cx,
        );
        self.restore_current_view_state(cx);
        cx.notify();
    }

    fn view_state_key(&self) -> String {
        let session = if self.core.session.current.as_os_str().is_empty() {
            "<new>".into()
        } else {
            self.core.session.current.display().to_string()
        };
        format!("{}\n{session}", self.core.workspace.cwd.display())
    }

    fn save_current_view_state(&mut self) {
        let key = self.view_state_key();
        let state = self.view_states.entry(key).or_default();
        if self.core.surface == Surface::Trajectory {
            state.trajectory_offset = self.trajectory_scroll.0.borrow().base_handle.offset();
            state.selected_trajectory = self.core.details.selected;
            state.details_tab = self.core.details.tab;
            state.details_offset = self.details_scroll.offset();
        } else {
            state.chat_anchor = self
                .layout_runtime
                .capture_chat_anchor(self.core.layout_generation, self.core.follow_chat_tail);
        }
    }

    fn restore_current_view_state(&mut self, cx: &mut Context<Self>) {
        let state = self
            .view_states
            .get(&self.view_state_key())
            .copied()
            .unwrap_or_default();
        if self.core.surface == Surface::Trajectory {
            self.trajectory_scroll
                .0
                .borrow()
                .base_handle
                .set_offset(state.trajectory_offset);
            let selected = state.selected_trajectory.filter(|selected| {
                self.core
                    .conversation
                    .messages
                    .iter()
                    .any(|message| message.key == *selected)
            });
            let _ = self.transition(Action::RestoreSessionView {
                selected,
                details_tab: state.details_tab,
                follow_chat_tail: matches!(state.chat_anchor, ScrollAnchor::Tail),
            });
            if let Some(selected) = selected
                && let Some(index) = self
                    .core
                    .conversation
                    .messages
                    .iter()
                    .position(|message| message.key == selected)
            {
                self.scroll_trajectory_to_record(index, cx);
            }
            self.details_scroll.set_offset(state.details_offset);
        } else {
            let _ = self.transition(Action::RestoreSessionView {
                selected: None,
                details_tab: state.details_tab,
                follow_chat_tail: matches!(state.chat_anchor, ScrollAnchor::Tail),
            });
            self.layout_runtime.pending_chat_anchor =
                Some((self.core.layout_generation, state.chat_anchor));
            self.layout_runtime.restore_scheduled = false;
        }
    }

    pub(crate) fn toggle_session_search(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let opening = !self.core.sidebar.search_sessions;
        self.dispatch(Action::ToggleSessionSearch, window, cx);
        if opening {
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
        self.dispatch_local(Action::ToggleSidebarOptions, cx);
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
        if index == self.core.workspace.active_project {
            self.dispatch(Action::ToggleProjectExpanded(path), window, cx);
        } else {
            self.dispatch(Action::ExpandProject(path), window, cx);
            self.switch_project(index, window, cx);
        }
    }

    pub(crate) fn export_session_log(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.core.session.current.as_os_str().is_empty() {
            self.notice("Start the session before exporting its log");
            cx.notify();
            return;
        }
        let source = self.core.session.current.clone();
        let suggested = format!("{}.jsonl", safe_file_name(&self.core.conversation.title));
        let receiver = cx.prompt_for_new_path(&self.core.workspace.cwd, Some(&suggested));
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
        self.dispatch_local(
            Action::Conversation(ConversationAction::ToggleExpanded {
                index,
                role: Role::Tool,
            }),
            cx,
        );
    }

    pub(crate) fn toggle_reasoning(&mut self, index: usize, cx: &mut Context<Self>) {
        self.dispatch_local(
            Action::Conversation(ConversationAction::ToggleExpanded {
                index,
                role: Role::Reasoning,
            }),
            cx,
        );
    }

    pub(crate) fn inspect_tool(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .core
            .conversation
            .messages
            .get(index)
            .is_some_and(|message| message.role == Role::Tool)
        {
            self.save_current_view_state();
            let message_id = self.core.conversation.messages[index].key;
            let mut effects = self.transition(Action::ShowTrajectory);
            effects.extend(self.transition(Action::SelectDetails(Some(message_id))));
            run_effects(self, effects, window, cx);
            self.dispatch_local(Action::SetDetailsTab(DetailsTab::Summary), cx);
            self.details_scroll.set_offset(point(px(0.0), px(0.0)));
            self.dispatch_local(Action::ExpandTrajectoryGroups, cx);
            self.scroll_trajectory_to_record(index, cx);
            cx.notify();
        }
    }

    pub(crate) fn rate_message(&mut self, index: usize, positive: bool, cx: &mut Context<Self>) {
        self.dispatch_local(
            Action::Conversation(ConversationAction::RateAssistant { index, positive }),
            cx,
        );
    }

    pub(crate) fn set_composer_menu(&mut self, menu: Option<ComposerMenu>, cx: &mut Context<Self>) {
        self.dispatch_local(Action::SetComposerMenu(menu), cx);
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
        self.dispatch_local(
            Action::MoveComposerHighlight {
                delta: direction,
                item_count: count,
            },
            cx,
        );
    }

    fn composer_menu_item_count(&self) -> usize {
        match self.core.composer.menu {
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
        let index = self.core.composer.highlighted_item;
        match self.core.composer.menu {
            Some(ComposerMenu::Commands) => match index {
                0 => {
                    self.dispatch(Action::SetComposerMenu(None), window, cx);
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
                    self.dispatch_local(Action::SetComposerMenu(None), cx);
                }
            }
            Some(ComposerMenu::Workspace) => {
                if index < self.project_store.projects().len() {
                    self.dispatch(Action::SetComposerMenu(None), window, cx);
                    self.switch_project(index, window, cx);
                } else {
                    self.dispatch(Action::SetComposerMenu(None), window, cx);
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
        let modifiers = event.keystroke.modifiers;
        if event.keystroke.key.eq_ignore_ascii_case("b")
            && modifiers.secondary()
            && !modifiers.alt
            && !modifiers.shift
            && !modifiers.function
        {
            self.toggle_sidebar(window, cx);
            cx.stop_propagation();
            return;
        }
        if self.core.composer.menu.is_some() {
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
        let was_searching = self.core.sidebar.search_sessions;
        self.dispatch(Action::DismissTransient, window, cx);
        if was_searching {
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
        self.dispatch_local(Action::SetComposerMenu(None), cx);
        cx.notify();
    }

    pub(crate) fn new_chat(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.task_active() {
            self.notice("Stop the active task before starting a new chat");
            cx.notify();
            return;
        }
        self.save_current_view_state();
        self.activate_session(Session::memory());
        self.restore_current_view_state(cx);
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
        if index == self.core.workspace.active_project {
            self.new_chat(window, cx);
        } else {
            self.switch_project(index, window, cx);
        }
    }

    pub(crate) fn add_project(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.task_active() {
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
                                                this.refresh_project_session_cache();
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
        if index == self.core.workspace.active_project {
            return;
        }
        if self.task_active() {
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
        let sessions = self
            .project_sessions
            .get(&project.sessions_dir)
            .cloned()
            .unwrap_or_default();
        self.dispatch(
            Action::ActivateWorkspace {
                index,
                cwd: project.path,
                sessions_dir: project.sessions_dir,
                sessions,
            },
            window,
            cx,
        );
        self.refresh_session_search_documents();
        self.reset_conversation();
        self.restore_current_view_state(cx);
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
        if self.task_active() {
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
        self.refresh_project_session_cache();
        let next = if index < self.core.workspace.active_project {
            self.core.workspace.active_project - 1
        } else if index == self.core.workspace.active_project {
            index.min(self.project_store.projects().len() - 1)
        } else {
            self.core.workspace.active_project
        };
        self.dispatch(Action::SetActiveProject(usize::MAX), window, cx);
        self.switch_project(next, window, cx);
    }

    pub(crate) fn open_session(
        &mut self,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if path == self.core.session.current {
            return;
        }
        if self.task_active() {
            self.notice("Stop the active task before opening another session");
            cx.notify();
            return;
        }
        self.save_current_view_state();
        self.dispatch(Action::BeginOpenSession(path), window, cx);
    }

    pub(crate) fn open_session_effect(
        &mut self,
        operation: crate::domain::OperationId,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        cx.spawn_in(window, async move |this, cx| {
            let session = Session::open(path).await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match session {
                            Ok(session) => {
                                let current_session = session.info().path.clone();
                                let was_pending = matches!(
                                    this.core.pending_session_operation.as_ref(),
                                    Some(pending)
                                        if pending.operation == operation
                                            && matches!(
                                                &pending.kind,
                                                SessionOperationKind::Open { path }
                                                    if *path == current_session
                                            )
                                );
                                let conversation =
                                    conversation_from_session(&session, &this.tool_schemas);
                                let sessions = this.reload_active_sessions();
                                let effects = this.transition(Action::SessionOpened {
                                    operation,
                                    conversation,
                                    current_session: current_session.clone(),
                                    sessions,
                                });
                                let accepted = was_pending
                                    && this.core.session.current == current_session
                                    && this.core.pending_session_operation.is_none();
                                if accepted {
                                    if let Some(agent) = &mut this.agent {
                                        agent.set_session(session);
                                    }
                                    this.sync_message_presentations();
                                    this.restore_current_view_state(cx);
                                    this.input.update(cx, |input, cx| {
                                        input.set_placeholder("Message the agent", window, cx)
                                    });
                                }
                                run_effects(this, effects, window, cx);
                            }
                            Err(error) => {
                                let message = format!("Could not open session: {error}");
                                let pending_before = this.core.pending_session_operation.clone();
                                this.dispatch(
                                    Action::SessionOperationFailed {
                                        operation,
                                        message: message.clone(),
                                    },
                                    window,
                                    cx,
                                );
                                if pending_before != this.core.pending_session_operation {
                                    this.notice(message);
                                }
                            }
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
        if project_index != self.core.workspace.active_project {
            self.switch_project(project_index, window, cx);
        }
        if project_index == self.core.workspace.active_project {
            self.open_session(path, window, cx);
        }
    }

    pub(crate) fn rename_current_session(
        &mut self,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.task_active() || self.core.session.current.as_os_str().is_empty() {
            return;
        }
        self.dispatch(Action::BeginRenameSession(title), window, cx);
    }

    pub(crate) fn rename_session_effect(
        &mut self,
        operation: crate::domain::OperationId,
        title: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(mut agent) = self.agent.take() else {
            self.dispatch(
                Action::SessionOperationFailed {
                    operation,
                    message: "Agent is unavailable".into(),
                },
                window,
                cx,
            );
            return;
        };
        cx.spawn_in(window, async move |this, cx| {
            let result = agent.rename_session(&title).await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match result {
                            Ok(()) => {
                                let was_pending = this
                                    .core
                                    .pending_session_operation
                                    .as_ref()
                                    .is_some_and(|pending| pending.operation == operation);
                                let sessions = this.reload_active_sessions();
                                this.dispatch(
                                    Action::SessionRenamed {
                                        operation,
                                        title: agent.session_info().title.clone(),
                                        sessions,
                                    },
                                    window,
                                    cx,
                                );
                                if was_pending {
                                    this.refresh_session_search_documents();
                                }
                            }
                            Err(error) => {
                                let message = format!("Could not rename session: {error}");
                                let pending_before = this.core.pending_session_operation.clone();
                                this.dispatch(
                                    Action::SessionOperationFailed {
                                        operation,
                                        message: message.clone(),
                                    },
                                    window,
                                    cx,
                                );
                                if pending_before != this.core.pending_session_operation {
                                    this.notice(message);
                                }
                            }
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
        if self.task_active() || self.core.session.current.as_os_str().is_empty() {
            return;
        }
        let Some(session) = self
            .core
            .session
            .sessions
            .iter()
            .find(|session| same_path(&session.path, &self.core.session.current))
            .cloned()
        else {
            return;
        };
        if let Some(agent) = &mut self.agent {
            agent.set_session(Session::memory());
        }
        match Session::delete(&session) {
            Ok(()) => {
                let sessions = self.reload_active_sessions();
                self.dispatch(Action::RefreshSessions(sessions), window, cx);
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
        if self.control.is_some() || self.core.pending_session_operation.is_some() {
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
        if self.control.is_some()
            || self.core.pending_session_operation.is_some()
            || index >= self.models.len()
            || index == self.selected_model
        {
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
        self.dispatch_local(Action::SetComposerMenu(None), cx);
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

    pub(crate) fn session_modified_at(&self, session: &SessionInfo) -> u64 {
        self.session_activity
            .get(&session.path)
            .copied()
            .unwrap_or(session.created_at)
    }

    fn refresh_session_search_documents(&mut self) {
        self.session_search_documents =
            build_session_search_documents(&self.project_store, &self.project_sessions);
    }

    fn reload_sessions(&mut self, sessions_dir: &Path) -> Vec<SessionInfo> {
        let sessions = Session::list(sessions_dir).unwrap_or_default();
        if let Some(previous) = self.project_sessions.get(sessions_dir) {
            for session in previous {
                self.session_activity.remove(&session.path);
            }
        }
        for session in &sessions {
            self.session_activity
                .insert(session.path.clone(), session_modified_at_from_disk(session));
        }
        self.project_sessions
            .insert(sessions_dir.to_owned(), sessions.clone());
        sessions
    }

    fn reload_active_sessions(&mut self) -> Vec<SessionInfo> {
        let sessions_dir = self.core.workspace.sessions_dir.clone();
        self.reload_sessions(&sessions_dir)
    }

    fn refresh_project_session_cache(&mut self) {
        let sessions = load_project_sessions(&self.project_store);
        self.session_activity = load_session_activity(&sessions);
        self.project_sessions = sessions;
    }

    pub(crate) fn chat_at_bottom(&self) -> bool {
        within_bottom_threshold(self.scroll.max_offset().height, self.scroll.offset().y)
    }

    pub(crate) fn apply_chat_tail(&mut self) {
        let max_offset = self.scroll.max_offset().height;
        let leading_inset = px(self.core.layout.transcript_top_inset);
        if max_offset <= leading_inset + px(0.5) {
            let offset = self.scroll.offset();
            self.scroll.set_offset(point(offset.x, px(0.0)));
            // The transcript content column follows the measurement sentinel.
            // Keeping it active preserves the inset through the next reflow, while
            // still aligning its bottom once the conversation outgrows the viewport.
            self.scroll.scroll_to_item(1);
        } else {
            self.scroll.scroll_to_bottom();
        }
    }

    pub(crate) fn handle_chat_scroll(
        &mut self,
        event: &ScrollWheelEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let delta_y = event.delta.pixel_delta(window.line_height()).y;
        let action = if delta_y > px(0.0) && self.core.follow_chat_tail {
            Some(Action::Scroll(ScrollIntent::Away))
        } else if delta_y < px(0.0) {
            Some(Action::Scroll(ScrollIntent::Toward {
                at_tail: self.chat_at_bottom(),
            }))
        } else {
            None
        };
        if let Some(action) = action {
            self.dispatch(action, window, cx);
        }
    }

    pub(crate) fn scroll_chat_to_bottom(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);
    }

    fn activate_session(&mut self, session: Session) {
        let path = session.info().path.clone();
        let conversation = conversation_from_session(&session, &self.tool_schemas);
        if let Some(agent) = &mut self.agent {
            agent.set_session(session);
        }
        let sessions = self.reload_active_sessions();
        let _ = self.transition(Action::ReplaceConversation {
            conversation,
            current_session: path,
            sessions,
        });
        self.sync_message_presentations();
        self.apply_chat_tail();
    }

    fn activate_session_for_run(&mut self, session: Session, previous_key: String) {
        let current_key = self.view_state_key();
        if let Some(state) = self.view_states.remove(&previous_key) {
            self.view_states.insert(current_key, state);
        }
        if let Some(agent) = &mut self.agent {
            agent.set_session(session);
        }
    }

    fn reset_conversation(&mut self) {
        let _ = self.transition(Action::ResetConversation);
        self.message_presentations.clear();
        self.modal = None;
        self.apply_chat_tail();
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
    Message {
        key: next_message_render_key(),
        revision: 0,
        role,
        tool_call_id: None,
        title: None,
        text,
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
                key: next_message_render_key(),
                revision: 0,
                role: Role::Tool,
                tool_call_id: Some(call_id),
                title: Some(name),
                text: String::new(),
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
            }),
            TranscriptItem::ToolOutput { call_id, output } => {
                if let Some(message) = messages
                    .iter_mut()
                    .rev()
                    .find(|message| message.tool_call_id.as_deref() == Some(&call_id))
                {
                    message.text = output;
                    message.revision = message.revision.saturating_add(1);
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

fn conversation_from_session(
    session: &Session,
    tool_schemas: &HashMap<String, String>,
) -> ConversationState {
    let mut messages = messages_from_transcript(session.state().transcript());
    for message in &mut messages {
        if message.role == Role::Tool
            && let Some(name) = message.title.as_deref()
        {
            message.schema = tool_schemas.get(name).cloned();
        }
    }
    reindex_messages(&mut messages);
    let title = if session.info().title == "Untitled session" {
        "New chat".into()
    } else {
        session.info().title.clone()
    };
    let turns = messages
        .iter()
        .filter(|message| message.role == Role::User)
        .count();
    let tool_calls = messages
        .iter()
        .filter(|message| message.role == Role::Tool)
        .count();
    let (input_tokens, output_tokens, cached_tokens) = session
        .state()
        .latest_response()
        .and_then(|response| response.usage.as_ref())
        .map(|usage| {
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.input_tokens_details.cached_tokens,
            )
        })
        .unwrap_or_default();
    ConversationState {
        messages,
        title,
        turns,
        tool_calls,
        input_tokens,
        output_tokens,
        cached_tokens,
    }
}

fn restored_message(role: Role, text: String) -> Message {
    let mut message = message(role, text);
    message.started_at_ms = None;
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

#[cfg(test)]
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
    project_sessions: &HashMap<PathBuf, Vec<SessionInfo>>,
) -> HashMap<PathBuf, SessionSearchDocument> {
    let mut documents = HashMap::new();
    for project in project_store.projects() {
        let Some(sessions) = project_sessions.get(&project.sessions_dir) else {
            continue;
        };
        for session in sessions {
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
                session.path.clone(),
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

fn load_project_sessions(project_store: &ProjectStore) -> HashMap<PathBuf, Vec<SessionInfo>> {
    project_store
        .projects()
        .iter()
        .map(|project| {
            (
                project.sessions_dir.clone(),
                Session::list(&project.sessions_dir).unwrap_or_default(),
            )
        })
        .collect()
}

fn load_session_activity(
    project_sessions: &HashMap<PathBuf, Vec<SessionInfo>>,
) -> HashMap<PathBuf, u64> {
    project_sessions
        .values()
        .flatten()
        .map(|session| (session.path.clone(), session_modified_at_from_disk(session)))
        .collect()
}

fn session_modified_at_from_disk(session: &SessionInfo) -> u64 {
    std::fs::metadata(&session.path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs())
        .unwrap_or(session.created_at)
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

    fn update_messages(messages: &mut Vec<Message>, action: impl FnOnce(&mut ConversationState)) {
        let mut state = ConversationState {
            messages: std::mem::take(messages),
            ..ConversationState::default()
        };
        action(&mut state);
        *messages = state.messages;
    }

    fn push_delta(messages: &mut Vec<Message>, delta: &str) {
        update_messages(messages, |state| {
            crate::domain::reduce_conversation(
                state,
                ConversationAction::TextDelta {
                    delta: delta.into(),
                    new_message: message(Role::Assistant, delta.into()),
                },
            );
        });
    }

    fn push_reasoning_delta(messages: &mut Vec<Message>, delta: &str) {
        update_messages(messages, |state| {
            crate::domain::reduce_conversation(
                state,
                ConversationAction::ReasoningDelta {
                    delta: delta.into(),
                    new_message: message(Role::Reasoning, delta.into()),
                },
            );
        });
    }

    fn settle_active_response_message(messages: &mut Vec<Message>) {
        update_messages(messages, |state| {
            crate::domain::reduce_conversation(state, ConversationAction::FinishReasoning);
        });
    }

    fn refresh_live_render_state(messages: &mut Vec<Message>) {
        update_messages(messages, |state| {
            crate::domain::reduce_conversation(state, ConversationAction::RefreshLiveSearch);
        });
    }

    #[test]
    fn streaming_deltas_share_one_message() {
        let mut messages = Vec::new();
        push_delta(&mut messages, "hello");
        push_delta(&mut messages, " world");
        refresh_live_render_state(&mut messages);
        let mut presentations = MessagePresentationStore::default();
        presentations.sync(messages.iter().map(|message| {
            (
                message.key,
                message.text.as_str(),
                message.role == Role::Assistant,
            )
        }));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text, "hello world");
        let presentation = presentations.get(messages[0].key);
        assert_eq!(presentation.render_text.as_ref(), "hello world");
        assert_eq!(presentation.markdown.revision(), 1);
    }

    #[test]
    fn markdown_state_is_ready_when_a_delta_and_settle_share_one_frame() {
        let mut messages = Vec::new();
        push_delta(&mut messages, "## Result\n\n- one\n- two");
        settle_active_response_message(&mut messages);
        refresh_live_render_state(&mut messages);
        let mut presentations = MessagePresentationStore::default();
        presentations.sync(messages.iter().map(|message| {
            (
                message.key,
                message.text.as_str(),
                message.role == Role::Assistant,
            )
        }));

        assert!(!messages[0].pending);
        assert!(
            !presentations
                .get(messages[0].key)
                .markdown
                .tail_blocks()
                .is_empty()
        );
    }

    #[test]
    fn restored_assistant_messages_use_the_semantic_markdown_renderer() {
        let message = restored_message(Role::Assistant, "## Result\n\nbody".into());
        let mut presentations = MessagePresentationStore::default();
        presentations.sync([(message.key, message.text.as_str(), true)]);
        assert!(
            !presentations
                .get(message.key)
                .markdown
                .tail_blocks()
                .is_empty()
        );
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

        window.update(|window, _| arm_next_frame(window, ready_tx));
    }

    #[gpui::test]
    fn first_submitted_user_keeps_leading_gap_after_empty_state_reflow(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-first-turn-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), workspace.clone()).unwrap();
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), workspace);

        cx.update(crate::init_ui);
        let view = std::rc::Rc::new(std::cell::RefCell::new(None));
        let test_view = view.clone();
        let (_, cx) = cx.add_window_view(|window, cx| {
            let view = cx.new(|cx| {
                DesktopApp::new(
                    DesktopStartup {
                        agent,
                        models: vec![configured],
                        selected_model: 0,
                        project_store,
                        active_project,
                        settings,
                    },
                    window,
                    cx,
                )
            });
            test_view.replace(Some(view.clone()));
            gpui_component::Root::new(view, window, cx)
        });
        let view = view.borrow().clone().unwrap();
        cx.simulate_resize(gpui::size(px(1180.0), px(720.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                this.dispatch(
                    Action::Conversation(ConversationAction::SubmitUser(message(
                        Role::User,
                        "hello".into(),
                    ))),
                    window,
                    cx,
                );
                this.sync_message_presentations();
                this.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);
            });
        });
        cx.refresh().unwrap();
        cx.run_until_parked();

        let (offset, max_offset, viewport, content) = cx.read_entity(&view, |app, _| {
            (
                app.scroll.offset(),
                app.scroll.max_offset(),
                app.scroll.bounds(),
                app.scroll.bounds_for_item(1).unwrap(),
            )
        });
        assert!(
            content.top() + offset.y >= viewport.top() + px(16.0)
                && content.bottom() + offset.y <= viewport.bottom(),
            "first user did not keep its leading gap: offset={offset:?}, max={max_offset:?}, viewport={viewport:?}, content={content:?}"
        );

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                this.dispatch(
                    Action::Conversation(ConversationAction::SubmitUser(message(
                        Role::User,
                        "overflow ".repeat(800),
                    ))),
                    window,
                    cx,
                );
                this.sync_message_presentations();
                this.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);
            });
        });
        cx.refresh().unwrap();
        cx.run_until_parked();

        let (offset, max_offset) = cx.read_entity(&view, |app, _| {
            (app.scroll.offset(), app.scroll.max_offset())
        });
        assert!(
            f32::from(max_offset.height + offset.y).abs() <= 1.0,
            "overflowing conversation did not follow the tail: offset={offset:?}, max={max_offset:?}"
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn message_render_keys_do_not_alias_across_session_reloads() {
        let first = message(Role::Assistant, "first".into());
        let second = message(Role::Assistant, "second".into());
        assert_ne!(first.key, second.key);
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
        let mut conversation = ConversationState::default();
        crate::domain::reduce_conversation(
            &mut conversation,
            ConversationAction::SubmitUser(message(Role::User, input.into())),
        );
        let title = conversation.title;
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
        assert_eq!(crate::application::step_count(&messages), 2);
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
