use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpui::{
    AppContext, Context, Entity, FocusHandle, PathPromptOptions, Pixels, Point, ScrollHandle,
    ScrollWheelEvent, Subscription, UniformListScrollHandle, Window, point, px,
};
use gpui_component::input::{InputEvent, InputState, TextareaState};
use gpui_component::{Theme, ThemeMode};
use kcastle_agent::{
    Agent, Model, Session, SessionConfig, SessionEvent, SessionId, SessionInfo, SessionIssue,
    TranscriptItem,
};

#[cfg(test)]
use kcastle_agent::AgentEvent;

#[cfg(test)]
use crate::application::is_frame_stream_event;
use crate::dialogs::Modal;
use crate::domain::{
    Action, AppState, ComposerMenu, ConversationAction, ConversationState, DetailsTab, Effect,
    Message, MessageId, Role, RunState, ScrollIntent, Surface, next_message_id, reduce,
    reindex_messages,
};
use crate::layout::{LayoutInput, ScrollAnchor, ScrollRestore, resolve_scroll_restore};
use crate::platform::NativeTitlebarController;
#[cfg(test)]
use crate::platform::gpui::arm_next_frame;
use crate::platform::gpui::{
    DeferredScrollAlignment, GpuiLayoutRuntime, MeasuredBounds, MessagePresentationStore,
    SessionRuntime, SessionRuntimeSnapshot, SessionRuntimeStatus, run_effects,
};
use crate::project::{ProjectId, ProjectStore};
use crate::settings::{Appearance, EnterBehavior, ProviderModel, SettingsStore};
use crate::updater::AvailableUpdate;

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

pub(crate) fn composer_model_indices(
    models: &[ConfiguredModel],
) -> impl Iterator<Item = usize> + '_ {
    models
        .iter()
        .enumerate()
        .filter(|(_, model)| model.model.has_api_key())
        .map(|(index, _)| index)
}

pub(crate) fn active_model_index(
    models: &[ConfiguredModel],
    preferred_id: Option<&str>,
) -> Option<usize> {
    preferred_id
        .and_then(|preferred| {
            models
                .iter()
                .position(|model| model.id == preferred && model.model.has_api_key())
        })
        .or_else(|| composer_model_indices(models).next())
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

struct ProjectSessionRuntimes {
    selected: SessionId,
    sessions: HashMap<SessionId, Entity<SessionRuntime>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RuntimeObservation {
    completed_runs: u64,
    transcript_updates: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SidebarSessionStatus {
    Preparing,
    Running,
    ApprovalNeeded,
    Failed,
    Unread,
}

pub(crate) struct DesktopApp {
    pub(crate) core: AppState,
    pub(crate) layout_runtime: GpuiLayoutRuntime,
    pub(crate) message_presentations: MessagePresentationStore,
    pub(crate) selected_runtime: Entity<SessionRuntime>,
    project_runtimes: HashMap<ProjectId, ProjectSessionRuntimes>,
    pub(crate) input: Entity<TextareaState>,
    pub(crate) session_search: Entity<InputState>,
    pub(crate) trajectory_search: Entity<InputState>,
    pub(crate) modal: Option<Modal>,
    pub(crate) modal_focus: FocusHandle,
    pub(crate) composer_menu_focus: FocusHandle,
    pub(crate) scroll: ScrollHandle,
    chat_tail_alignment: DeferredScrollAlignment,
    pub(crate) trajectory_scroll: UniformListScrollHandle,
    pub(crate) details_scroll: ScrollHandle,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) selected_reasoning_effort: Option<kcastle_agent::ReasoningEffort>,
    pub(crate) model: String,
    pub(crate) project_store: ProjectStore,
    pub(crate) settings: SettingsStore,
    pub(crate) selected_started_at: Option<Instant>,
    pub(crate) tool_schemas: HashMap<String, String>,
    pub(crate) project_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
    pub(crate) project_session_issues: HashMap<PathBuf, Vec<SessionIssue>>,
    pub(crate) session_activity: HashMap<PathBuf, u64>,
    pub(crate) session_search_documents: HashMap<PathBuf, SessionSearchDocument>,
    pub(crate) available_update: Option<AvailableUpdate>,
    unread_sessions: HashSet<(ProjectId, SessionId)>,
    runtime_observations: HashMap<(ProjectId, SessionId), RuntimeObservation>,
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
        let current_session_id = agent.session_info().id.clone();
        let runtime_config = config_for_model(&models[selected_model], settings.allow_all_tools());
        let selected_reasoning_effort = runtime_config
            .reasoning_effort
            .as_deref()
            .and_then(parse_reasoning_effort);
        let runtime = cx.new(|_| {
            SessionRuntime::new(
                agent,
                project.id.as_str().to_owned(),
                project.sessions_dir.clone(),
                ConversationState::default(),
                runtime_config,
            )
        });
        let runtime_subscription = cx.observe(&runtime, |this, runtime, cx| {
            this.sync_runtime_snapshot(&runtime, cx);
        });
        let mut project_runtimes = HashMap::new();
        project_runtimes.insert(
            project.id.clone(),
            ProjectSessionRuntimes {
                selected: current_session_id.clone(),
                sessions: HashMap::from([(current_session_id, runtime.clone())]),
            },
        );
        let (project_sessions, project_session_issues) = load_project_sessions(&project_store);
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
            TextareaState::new(window, cx)
                .auto_grow(1, 14)
                .submit_on_enter(true)
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
                event if is_composer_submit_event(event) => {
                    if this.core.composer.menu.is_none() {
                        this.submit(window, cx);
                    }
                }
                InputEvent::Change => {
                    if this.core.follow_chat_tail {
                        this.schedule_chat_tail(window);
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
        let app = Self {
            core,
            layout_runtime: GpuiLayoutRuntime::default(),
            message_presentations: MessagePresentationStore::default(),
            selected_runtime: runtime,
            project_runtimes,
            input,
            session_search,
            trajectory_search,
            modal: None,
            modal_focus: cx.focus_handle(),
            composer_menu_focus: cx.focus_handle(),
            scroll: ScrollHandle::new(),
            chat_tail_alignment: DeferredScrollAlignment::default(),
            trajectory_scroll: UniformListScrollHandle::new(),
            details_scroll: ScrollHandle::new(),
            models,
            selected_model,
            selected_reasoning_effort,
            model,
            project_store,
            settings,
            selected_started_at: None,
            tool_schemas,
            project_sessions,
            project_session_issues,
            session_activity,
            session_search_documents,
            available_update: None,
            unread_sessions: HashSet::new(),
            runtime_observations: HashMap::new(),
            native_titlebar,
            view_states: HashMap::new(),
            _subscriptions: vec![
                subscription,
                search_subscription,
                trajectory_subscription,
                appearance_subscription,
                bounds_subscription,
                runtime_subscription,
            ],
        };
        #[cfg(not(test))]
        app.check_for_updates(window, cx);
        app
    }

    fn sync_runtime_snapshot(&mut self, runtime: &Entity<SessionRuntime>, cx: &mut Context<Self>) {
        let snapshot = runtime.read(cx).snapshot();
        let location = self.runtime_location(runtime);
        let selected = runtime.entity_id() == self.selected_runtime.entity_id();
        let mut selected_transcript_updates = 0;
        if let Some((project_id, session_id)) = &location {
            let key = (project_id.clone(), session_id.clone());
            let previous = self.runtime_observations.entry(key.clone()).or_default();
            let unread_completion = has_new_unread_completion(
                previous.completed_runs,
                snapshot.completed_runs,
                selected,
            );
            selected_transcript_updates = visible_transcript_update_count(
                previous.transcript_updates,
                snapshot.transcript_updates,
                selected,
            );
            *previous = RuntimeObservation {
                completed_runs: snapshot.completed_runs,
                transcript_updates: snapshot.transcript_updates,
            };
            if unread_completion {
                self.unread_sessions.insert(key.clone());
            }
            if selected {
                self.unread_sessions.remove(&key);
            }
        }
        if let Some((project_id, _)) = &location
            && !snapshot.session.path.as_os_str().is_empty()
            && !self
                .project_store
                .projects()
                .iter()
                .find(|project| &project.id == project_id)
                .and_then(|project| self.project_sessions.get(&project.sessions_dir))
                .is_some_and(|sessions| sessions.iter().any(|info| info.id == snapshot.session.id))
        {
            self.refresh_project_session_cache();
            self.refresh_session_search_documents();
        }
        if !selected {
            cx.notify();
            return;
        }
        self.apply_selected_runtime_snapshot(snapshot);
        self.sync_message_presentations();
        if selected_transcript_updates > 0 {
            self.dispatch_local(
                Action::StreamDeltasReceived(selected_transcript_updates),
                cx,
            );
        }
        cx.notify();
    }

    fn runtime_location(&self, runtime: &Entity<SessionRuntime>) -> Option<(ProjectId, SessionId)> {
        self.project_runtimes
            .iter()
            .find_map(|(project_id, runtimes)| {
                runtimes
                    .sessions
                    .iter()
                    .find(|(_, candidate)| candidate.entity_id() == runtime.entity_id())
                    .map(|(session_id, _)| (project_id.clone(), session_id.clone()))
            })
    }

    fn apply_selected_runtime_snapshot(&mut self, snapshot: SessionRuntimeSnapshot) {
        if let Some(model_id) = snapshot.config.model_id.as_deref()
            && let Some(index) = self.models.iter().position(|model| model.id == model_id)
            && self.models[index].model.has_api_key()
        {
            self.selected_model = index;
            self.model = self.models[index].label();
        }
        self.selected_reasoning_effort = snapshot
            .config
            .reasoning_effort
            .as_deref()
            .and_then(parse_reasoning_effort)
            .or_else(|| {
                self.models[self.selected_model]
                    .model
                    .reasoning_effort()
                    .cloned()
            });
        let previous_path = self.core.session.current.clone();
        self.core.conversation = snapshot.conversation;
        self.core.approval = snapshot.approval;
        self.core.session.current = snapshot.session.path.clone();
        self.selected_started_at = snapshot.started_at;
        self.core.run = match snapshot.status {
            SessionRuntimeStatus::Idle => RunState::Idle,
            SessionRuntimeStatus::Creating | SessionRuntimeStatus::Configuring => {
                RunState::Preparing
            }
            SessionRuntimeStatus::Running => RunState::Running {
                run: snapshot.active_run.unwrap_or_default(),
            },
            SessionRuntimeStatus::Failed(message) => RunState::Failed { message },
        };
        if previous_path != snapshot.session.path {
            if let Some(project) = self
                .project_store
                .project(self.core.workspace.active_project)
                && let Some(runtimes) = self.project_runtimes.get_mut(&project.id)
            {
                runtimes.selected = snapshot.session.id.clone();
            }
            self.core.session.sessions = self.reload_active_sessions();
            self.refresh_project_session_cache();
            self.refresh_session_search_documents();
        }
    }

    fn register_runtime(
        &mut self,
        project_id: ProjectId,
        runtime: Entity<SessionRuntime>,
        cx: &mut Context<Self>,
    ) {
        let session_id = runtime.read(cx).snapshot().session.id;
        let subscription = cx.observe(&runtime, |this, runtime, cx| {
            this.sync_runtime_snapshot(&runtime, cx);
        });
        self._subscriptions.push(subscription);
        let project =
            self.project_runtimes
                .entry(project_id)
                .or_insert_with(|| ProjectSessionRuntimes {
                    selected: session_id.clone(),
                    sessions: HashMap::new(),
                });
        project.sessions.insert(session_id, runtime);
    }

    fn create_runtime(
        &mut self,
        project_index: usize,
        session: Session,
        cx: &mut Context<Self>,
    ) -> Option<Entity<SessionRuntime>> {
        let project = self.project_store.project(project_index)?.clone();
        let mut conversation = conversation_from_session(&session, &self.tool_schemas);
        if session.recovery_needed() {
            conversation.messages.push(restored_message(
                Role::Notice,
                "The session log has an incomplete tail. It will be backed up and repaired when this session next writes."
                    .into(),
            ));
            reindex_messages(&mut conversation.messages);
        }
        let mut config = session.config().clone();
        let model_index = active_model_index(&self.models, config.model_id.as_deref())
            .unwrap_or(self.selected_model);
        let configured = &self.models[model_index];
        if config.model_id.as_deref() != Some(&configured.id) {
            config = config_for_model(configured, self.settings.allow_all_tools());
        }
        let mut model = configured.model.clone();
        if let Some(effort) = config.reasoning_effort.as_deref()
            && let Some(effort) = model
                .reasoning_efforts()
                .iter()
                .find(|candidate| reasoning_key(candidate) == effort)
        {
            model.set_reasoning_effort(effort.clone());
        }
        let agent = Agent::new(model, crate::INSTRUCTIONS, session, project.path.clone());
        let runtime = cx.new(|_| {
            SessionRuntime::new(
                agent,
                project.id.as_str().to_owned(),
                project.sessions_dir,
                conversation,
                config,
            )
        });
        self.register_runtime(project.id, runtime.clone(), cx);
        Some(runtime)
    }

    fn active_project_runtime(&self, path: &Path) -> Option<Entity<SessionRuntime>> {
        self.project_runtime(self.core.workspace.active_project, path)
    }

    fn project_runtime(&self, project_index: usize, path: &Path) -> Option<Entity<SessionRuntime>> {
        let project = self.project_store.project(project_index)?;
        let session_id = self
            .project_sessions
            .get(&project.sessions_dir)?
            .iter()
            .find(|session| same_path(&session.path, path))?
            .id
            .clone();
        self.project_runtimes
            .get(&project.id)?
            .sessions
            .get(&session_id)
            .cloned()
    }

    pub(crate) fn session_is_active(
        &self,
        project_index: usize,
        path: &Path,
        cx: &Context<Self>,
    ) -> bool {
        self.project_runtime(project_index, path)
            .is_some_and(|runtime| runtime.read(cx).is_active())
    }

    pub(crate) fn session_status_indicator(
        &self,
        project_index: usize,
        path: &Path,
        cx: &Context<Self>,
    ) -> Option<SidebarSessionStatus> {
        let project_id = self.project_store.project(project_index)?.id.clone();
        let snapshot = self
            .project_runtime(project_index, path)?
            .read(cx)
            .snapshot();
        let unread = self
            .unread_sessions
            .contains(&(project_id, snapshot.session.id));
        resolve_sidebar_session_status(&snapshot.status, snapshot.approval.is_some(), unread)
    }

    pub(crate) fn project_has_active_sessions(
        &self,
        project_index: usize,
        cx: &Context<Self>,
    ) -> bool {
        let Some(project) = self.project_store.project(project_index) else {
            return false;
        };
        self.project_runtimes
            .get(&project.id)
            .is_some_and(|runtimes| {
                runtimes
                    .sessions
                    .values()
                    .any(|runtime| runtime.read(cx).is_active())
            })
    }

    #[cfg(not(test))]
    pub(crate) fn has_active_sessions(&self, cx: &Context<Self>) -> bool {
        self.project_runtimes.values().any(|runtimes| {
            runtimes
                .sessions
                .values()
                .any(|runtime| runtime.read(cx).is_active())
        })
    }

    fn target_session_info(&self, project_index: usize, path: &Path) -> Option<SessionInfo> {
        let project = self.project_store.project(project_index)?;
        self.project_sessions
            .get(&project.sessions_dir)?
            .iter()
            .find(|session| same_path(&session.path, path))
            .cloned()
    }

    fn select_runtime(&mut self, runtime: Entity<SessionRuntime>, cx: &mut Context<Self>) {
        if let Some(key) = self.runtime_location(&runtime) {
            self.unread_sessions.remove(&key);
        }
        self.selected_runtime = runtime.clone();
        let snapshot = runtime.read(cx).snapshot();
        if let Some(project) = self
            .project_store
            .project(self.core.workspace.active_project)
            && let Some(runtimes) = self.project_runtimes.get_mut(&project.id)
        {
            runtimes.selected = snapshot.session.id.clone();
        }
        self.apply_selected_runtime_snapshot(snapshot);
        self.sync_message_presentations();
        self.restore_current_view_state(cx);
        cx.notify();
    }

    fn select_or_create_project_draft(
        &mut self,
        project_index: usize,
        cx: &mut Context<Self>,
    ) -> Option<Entity<SessionRuntime>> {
        let project = self.project_store.project(project_index)?.clone();
        if let Some(runtime) = self
            .project_runtimes
            .get(&project.id)
            .and_then(|runtimes| {
                runtimes.sessions.values().find(|runtime| {
                    runtime
                        .read(cx)
                        .snapshot()
                        .session
                        .path
                        .as_os_str()
                        .is_empty()
                })
            })
            .cloned()
        {
            return Some(runtime);
        }
        self.create_runtime(project_index, Session::memory(), cx)
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
            let Effect::ApplyChatTail = effect;
            self.layout_runtime.request_tail_realign();
        }
        cx.notify();
    }

    pub(crate) fn task_active(&self) -> bool {
        matches!(
            self.core.run,
            RunState::Preparing | RunState::Running { .. }
        )
    }

    pub(crate) fn session_running(&self) -> bool {
        matches!(self.core.run, RunState::Running { .. })
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

    pub(crate) fn restore_chat_tail_after_layout(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.core.follow_chat_tail {
            self.schedule_chat_tail(window);
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
        let restore_anchor = self.can_restore_pending_chat_anchor();
        let realign_tail =
            self.core.follow_chat_tail && self.layout_runtime.schedule_tail_realign();
        restore_anchor || realign_tail
    }

    pub(crate) fn observe_message_bounds(
        &mut self,
        id: crate::domain::MessageId,
        bounds: MeasuredBounds,
        _cx: &mut Context<Self>,
    ) -> bool {
        let layout_changed =
            self.layout_runtime
                .observe_message(self.core.layout_generation, id, bounds);
        if self.core.follow_chat_tail && layout_changed {
            self.layout_runtime.request_tail_realign();
        }
        let restore_anchor = self.can_restore_pending_chat_anchor();
        let realign_tail =
            self.core.follow_chat_tail && self.layout_runtime.schedule_tail_realign();
        restore_anchor || realign_tail
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

    pub(crate) fn apply_pending_chat_anchor(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.layout_runtime.restore_scheduled = false;
        if let Some((generation, anchor)) = self.layout_runtime.pending_chat_anchor {
            match resolve_scroll_restore(generation, self.core.layout_generation, anchor) {
                ScrollRestore::Tail => {
                    self.schedule_chat_tail(window);
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
        }
        if self.layout_runtime.take_tail_realign() && self.core.follow_chat_tail {
            self.schedule_chat_tail(window);
        }
        cx.notify();
    }

    pub(crate) fn submit(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.models[self.selected_model].model.has_api_key() {
            self.open_model_settings_dialog(window, cx);
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
        let behavior = self.settings.enter_behavior();
        self.selected_runtime.update(cx, |runtime, cx| {
            runtime.submit(value, behavior, window, cx)
        });
        self.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);
    }

    pub(crate) fn decide(&mut self, call_id: String, allow: bool, cx: &mut Context<Self>) {
        self.selected_runtime
            .update(cx, |runtime, cx| runtime.decide(call_id, allow, cx));
    }

    pub(crate) fn abort(&mut self, cx: &mut Context<Self>) {
        self.selected_runtime
            .update(cx, |runtime, cx| runtime.abort(cx));
    }

    pub(crate) fn notice(&mut self, text: impl Into<String>) {
        let _ = self.transition(Action::Conversation(Box::new(
            ConversationAction::AppendNotice(message(Role::Notice, text.into())),
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
            Action::Conversation(Box::new(ConversationAction::ToggleExpanded {
                index,
                role: Role::Tool,
            })),
            cx,
        );
        self.sync_runtime_message_expansion(index, Role::Tool, cx);
    }

    pub(crate) fn toggle_reasoning(&mut self, index: usize, cx: &mut Context<Self>) {
        self.dispatch_local(
            Action::Conversation(Box::new(ConversationAction::ToggleExpanded {
                index,
                role: Role::Reasoning,
            })),
            cx,
        );
        self.sync_runtime_message_expansion(index, Role::Reasoning, cx);
    }

    fn sync_runtime_message_expansion(&mut self, index: usize, role: Role, cx: &mut Context<Self>) {
        let Some(expanded) = self
            .core
            .conversation
            .messages
            .get(index)
            .filter(|message| message.role == role)
            .map(|message| message.expanded)
        else {
            return;
        };
        self.selected_runtime.update(cx, |runtime, cx| {
            runtime.set_message_expanded(index, role, expanded, cx)
        });
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
            Action::Conversation(Box::new(ConversationAction::RateAssistant {
                index,
                positive,
            })),
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
        self.composer_menu_focus.focus(window, cx);
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
            Some(ComposerMenu::Models) => composer_model_indices(&self.models).count(),
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
                _ if composer_model_indices(&self.models).next().is_some() => {
                    self.set_composer_menu(Some(ComposerMenu::Model), cx)
                }
                _ => self.open_model_settings_dialog(window, cx),
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
            Some(ComposerMenu::Models) => {
                let selected = composer_model_indices(&self.models).nth(index);
                if let Some(index) = selected {
                    self.select_model(index, cx);
                }
            }
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
        if !self
            .selected_runtime
            .update(cx, |runtime, cx| runtime.set_allow_all_tools(allow, cx))
        {
            self.notice("Stop this session before changing its tool permission");
        }
        self.dispatch_local(Action::SetComposerMenu(None), cx);
        cx.notify();
    }

    pub(crate) fn set_default_allow_all_tools(&mut self, allow: bool, cx: &mut Context<Self>) {
        if let Err(error) = self.settings.set_allow_all_tools(allow) {
            self.notice(format!("Could not save default permission: {error}"));
        }
        cx.notify();
    }

    pub(crate) fn new_chat(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.save_current_view_state();
        let Some(runtime) =
            self.select_or_create_project_draft(self.core.workspace.active_project, cx)
        else {
            self.notice("Could not create a session runtime for this project");
            return;
        };
        self.select_runtime(runtime, cx);
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

    pub(crate) fn relocate_project(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let receiver = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Relocate Project".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            let selection = receiver.await;
            let _ = cx.update(|_, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match selection {
                            Ok(Ok(Some(paths))) => {
                                if let Some(path) = paths.into_iter().next()
                                    && let Err(error) = this.project_store.relocate(index, path)
                                {
                                    this.notice(format!("Could not relocate project: {error}"));
                                }
                            }
                            Ok(Err(error)) => {
                                this.notice(format!("Could not open project picker: {error}"));
                            }
                            Err(error) => {
                                this.notice(format!("Project picker closed unexpectedly: {error}"))
                            }
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
        let Some(project) = self.project_store.project(index).cloned() else {
            return;
        };
        self.save_current_view_state();
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
        let runtime = self
            .project_runtimes
            .get(&project.id)
            .and_then(|runtimes| runtimes.sessions.get(&runtimes.selected))
            .cloned()
            .or_else(|| self.select_or_create_project_draft(index, cx));
        if let Some(runtime) = runtime {
            self.select_runtime(runtime, cx);
        }
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
        let Some(project) = self.project_store.project(index).cloned() else {
            return;
        };
        if project.is_default() {
            self.notice("The default project cannot be removed");
            cx.notify();
            return;
        }
        if self
            .project_runtimes
            .get(&project.id)
            .is_some_and(|runtimes| {
                runtimes
                    .sessions
                    .values()
                    .any(|runtime| runtime.read(cx).is_active())
            })
        {
            self.notice("Stop this project's active sessions before removing it");
            cx.notify();
            return;
        }
        if let Err(error) = self.project_store.remove(index) {
            self.notice(format!("Could not remove project: {error}"));
            cx.notify();
            return;
        }
        self.project_runtimes.remove(&project.id);
        self.unread_sessions
            .retain(|(project_id, _)| project_id != &project.id);
        self.runtime_observations
            .retain(|(project_id, _), _| project_id != &project.id);
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
        self.save_current_view_state();
        if let Some(runtime) = self.active_project_runtime(&path) {
            self.select_runtime(runtime, cx);
            self.input.update(cx, |input, cx| {
                input.set_placeholder("Message the agent", window, cx);
                input.focus(window, cx);
            });
            return;
        }
        self.open_session_async(path, window, cx);
    }

    fn open_session_async(&mut self, path: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        let project_id = self
            .project_store
            .project(self.core.workspace.active_project)
            .map(|project| project.id.as_str().to_owned())
            .unwrap_or_default();
        cx.spawn_in(window, async move |this, cx| {
            let session = Session::open_readonly_in_project(path, &project_id);
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        match session {
                            Ok(session) => {
                                if let Some(runtime) = this.create_runtime(
                                    this.core.workspace.active_project,
                                    session,
                                    cx,
                                ) {
                                    this.select_runtime(runtime, cx);
                                    this.input.update(cx, |input, cx| {
                                        input.set_placeholder("Message the agent", window, cx);
                                    });
                                }
                            }
                            Err(error) => {
                                let message = format!("Could not open session: {error}");
                                this.notice(message);
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

    pub(crate) fn rename_target_session(
        &mut self,
        project_index: usize,
        path: PathBuf,
        title: String,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if path.as_os_str().is_empty() || self.session_is_active(project_index, &path, cx) {
            self.notice("This session cannot be renamed while it is active");
            return;
        }
        let runtime = self.project_runtime(project_index, &path).or_else(|| {
            let project_id = self.project_store.project(project_index)?.id.as_str();
            let session = Session::open_readonly_in_project(&path, project_id).ok()?;
            self.create_runtime(project_index, session, cx)
        });
        let Some(runtime) = runtime else {
            self.notice("Could not open the target session for renaming");
            return;
        };
        if !runtime.update(cx, |runtime, cx| runtime.rename(title, cx)) {
            self.notice("This session cannot be renamed while it is active");
        }
    }

    pub(crate) fn delete_target_session(
        &mut self,
        project_index: usize,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if path.as_os_str().is_empty() || self.session_is_active(project_index, &path, cx) {
            self.notice("This session cannot be deleted while it is active");
            return;
        }
        let Some(session) = self.target_session_info(project_index, &path) else {
            self.notice("Could not find the target session");
            return;
        };
        match Session::delete(&session) {
            Ok(()) => {
                if let Some(project) = self.project_store.project(project_index) {
                    let key = (project.id.clone(), session.id.clone());
                    if let Some(runtimes) = self.project_runtimes.get_mut(&project.id) {
                        runtimes.sessions.remove(&session.id);
                    }
                    self.unread_sessions.remove(&key);
                    self.runtime_observations.remove(&key);
                }
                let sessions_dir = self
                    .project_store
                    .project(project_index)
                    .map(|project| project.sessions_dir.clone());
                if let Some(sessions_dir) = sessions_dir {
                    let sessions = self.reload_sessions(&sessions_dir);
                    if project_index == self.core.workspace.active_project {
                        self.dispatch(Action::RefreshSessions(sessions), window, cx);
                    }
                }
                self.refresh_session_search_documents();
                let deleted_selected = project_index == self.core.workspace.active_project
                    && same_path(&path, &self.core.session.current);
                if deleted_selected
                    && let Some(runtime) = self.select_or_create_project_draft(project_index, cx)
                {
                    self.select_runtime(runtime, cx);
                    self.input.update(cx, |input, cx| {
                        input.set_placeholder("Describe what you want to build", window, cx);
                        input.focus(window, cx);
                    });
                }
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
        if self.task_active() {
            return;
        }
        self.selected_runtime.update(cx, |runtime, cx| {
            runtime.set_reasoning_effort(effort.clone(), cx)
        });
        self.selected_reasoning_effort = Some(effort);
        cx.notify();
    }

    pub(crate) fn select_model(&mut self, index: usize, cx: &mut Context<Self>) {
        if self.task_active()
            || index >= self.models.len()
            || !self.models[index].model.has_api_key()
            || index == self.selected_model
        {
            return;
        }
        let configured = self.models[index].clone();
        let label = configured.label();
        self.selected_runtime.update(cx, |runtime, cx| {
            runtime.set_model(configured.id.clone(), configured.model.clone(), cx)
        });
        self.selected_model = index;
        self.model = label;
        self.selected_reasoning_effort = configured.model.reasoning_effort().cloned();
        self.dispatch_local(Action::SetComposerMenu(None), cx);
    }

    pub(crate) fn refresh_idle_runtime_models(&mut self, cx: &mut Context<Self>) {
        let updates = self
            .project_runtimes
            .values()
            .flat_map(|project| project.sessions.values())
            .filter_map(|runtime| {
                let snapshot = runtime.read(cx).snapshot();
                let model_id = snapshot.config.model_id?;
                let configured = self.models.iter().find(|model| model.id == model_id)?;
                let mut model = configured.model.clone();
                if let Some(effort) = snapshot
                    .config
                    .reasoning_effort
                    .as_deref()
                    .and_then(parse_reasoning_effort)
                    && model.reasoning_efforts().contains(&effort)
                {
                    model.set_reasoning_effort(effort);
                }
                Some((runtime.clone(), configured.id.clone(), model))
            })
            .collect::<Vec<_>>();
        for (runtime, model_id, model) in updates {
            runtime.update(cx, |runtime, cx| {
                runtime.set_model(model_id, model, cx);
            });
        }
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
        let project_id = self
            .project_store
            .projects()
            .iter()
            .find(|project| project.sessions_dir == sessions_dir)
            .map(|project| project.id.as_str())
            .unwrap_or(kcastle_agent::DEFAULT_PROJECT_ID);
        let (sessions, issues) = match Session::catalog_in_project(sessions_dir, project_id) {
            Ok(catalog) => (catalog.sessions, catalog.issues),
            Err(error) => {
                let sessions = self
                    .project_sessions
                    .get(sessions_dir)
                    .cloned()
                    .unwrap_or_default();
                (
                    sessions,
                    vec![SessionIssue {
                        path: sessions_dir.to_owned(),
                        message: error.to_string(),
                    }],
                )
            }
        };
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
        self.project_session_issues
            .insert(sessions_dir.to_owned(), issues);
        sessions
    }

    fn reload_active_sessions(&mut self) -> Vec<SessionInfo> {
        let sessions_dir = self.core.workspace.sessions_dir.clone();
        self.reload_sessions(&sessions_dir)
    }

    fn refresh_project_session_cache(&mut self) {
        let (sessions, issues) = load_project_sessions(&self.project_store);
        self.session_activity = load_session_activity(&sessions);
        self.project_sessions = sessions;
        self.project_session_issues = issues;
    }

    pub(crate) fn chat_at_bottom(&self) -> bool {
        within_bottom_threshold(self.scroll.max_offset().y, self.scroll.offset().y)
    }

    fn schedule_chat_tail(&mut self, window: &mut Window) {
        let leading_inset = px(self.core.layout.transcript_top_inset);
        let trailing_inset = px(self.core.layout.tail_inset);
        let short_transcript_max = leading_inset + trailing_inset + px(0.5);
        self.chat_tail_alignment.schedule_vertical_end(
            self.scroll.clone(),
            short_transcript_max,
            window,
        );
    }

    pub(crate) fn request_chat_tail(&mut self, window: &mut Window) {
        self.layout_runtime.request_tail_realign();
        self.schedule_chat_tail(window);
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
            if !self.core.follow_chat_tail {
                self.chat_tail_alignment.cancel();
            }
        }
    }

    pub(crate) fn scroll_chat_to_bottom(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);
    }
}

fn is_composer_submit_event(event: &InputEvent) -> bool {
    matches!(event, InputEvent::PressEnter { shift: false, .. })
}

fn message(role: Role, text: String) -> Message {
    Message {
        key: next_message_id(),
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
                key: next_message_id(),
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
    append_interrupted_stream(&mut messages, session.events());
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

fn append_interrupted_stream(messages: &mut Vec<Message>, events: &[SessionEvent]) {
    let mut reasoning = String::new();
    let mut answer = String::new();
    let mut active = false;
    let mut current_input = None;
    let mut admitted = Vec::<(String, String)>::new();
    let mut interruption = None;
    for event in events {
        match event {
            SessionEvent::RunStarted { input } => {
                reasoning.clear();
                answer.clear();
                active = true;
                current_input = Some(input.clone());
                interruption = Some("Run interrupted before completion");
            }
            SessionEvent::ReasoningDelta { delta } if active => reasoning.push_str(delta),
            SessionEvent::TextDelta { delta } if active => answer.push_str(delta),
            SessionEvent::ResponseCommitted => {
                reasoning.clear();
                answer.clear();
                current_input = None;
            }
            SessionEvent::RunFinished => {
                active = false;
                reasoning.clear();
                answer.clear();
                interruption = None;
                current_input = None;
            }
            SessionEvent::RunAborted => {
                active = false;
                interruption = Some("Run stopped; partial output was preserved");
            }
            SessionEvent::RunFailed { .. } => {
                active = false;
                interruption = Some("Run failed; partial output was preserved");
            }
            SessionEvent::InputAdmitted { id, input, .. } => {
                admitted.push((id.clone(), input.clone()));
            }
            SessionEvent::InputConsumed { id } => {
                admitted.retain(|(candidate, _)| candidate != id);
            }
            SessionEvent::ReasoningDelta { .. } | SessionEvent::TextDelta { .. } => {}
        }
    }
    if let Some(input) = current_input
        && !messages
            .last()
            .is_some_and(|message| message.role == Role::User && message.text == input)
    {
        messages.push(restored_message(Role::User, input));
    }
    let had_partial = !reasoning.is_empty() || !answer.is_empty();
    if !reasoning.is_empty() {
        let mut message = restored_message(Role::Reasoning, reasoning);
        message.title = Some("Think · interrupted".into());
        messages.push(message);
    }
    if !answer.is_empty() {
        messages.push(restored_message(Role::Assistant, answer));
    }
    if had_partial && let Some(interruption) = interruption {
        messages.push(restored_message(Role::Notice, interruption.into()));
    }
    for (_, input) in admitted {
        if !messages
            .last()
            .is_some_and(|message| message.role == Role::User && message.text == input)
        {
            messages.push(restored_message(Role::User, input));
        }
        messages.push(restored_message(
            Role::Notice,
            "This queued input was durably accepted but not consumed before the run stopped. Resend it to continue."
                .into(),
        ));
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

fn reasoning_key(effort: &kcastle_agent::ReasoningEffort) -> String {
    serde_json::to_value(effort)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| format!("{effort:?}").to_lowercase())
}

fn parse_reasoning_effort(value: &str) -> Option<kcastle_agent::ReasoningEffort> {
    serde_json::from_value(serde_json::Value::String(value.to_owned())).ok()
}

fn config_for_model(model: &ConfiguredModel, allow_all_tools: bool) -> SessionConfig {
    SessionConfig {
        model_id: Some(model.id.clone()),
        reasoning_effort: model.model.reasoning_effort().map(reasoning_key),
        allow_all_tools,
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

fn resolve_sidebar_session_status(
    status: &SessionRuntimeStatus,
    approval_needed: bool,
    unread: bool,
) -> Option<SidebarSessionStatus> {
    if approval_needed {
        return Some(SidebarSessionStatus::ApprovalNeeded);
    }
    match status {
        SessionRuntimeStatus::Creating | SessionRuntimeStatus::Configuring => {
            Some(SidebarSessionStatus::Preparing)
        }
        SessionRuntimeStatus::Running => Some(SidebarSessionStatus::Running),
        SessionRuntimeStatus::Failed(_) => Some(SidebarSessionStatus::Failed),
        SessionRuntimeStatus::Idle if unread => Some(SidebarSessionStatus::Unread),
        SessionRuntimeStatus::Idle => None,
    }
}

fn has_new_unread_completion(previous: u64, current: u64, selected: bool) -> bool {
    !selected && current > previous
}

fn visible_transcript_update_count(previous: u64, current: u64, selected: bool) -> usize {
    if !selected {
        return 0;
    }
    usize::try_from(current.saturating_sub(previous)).unwrap_or(usize::MAX)
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

fn load_project_sessions(
    project_store: &ProjectStore,
) -> (
    HashMap<PathBuf, Vec<SessionInfo>>,
    HashMap<PathBuf, Vec<SessionIssue>>,
) {
    let mut sessions = HashMap::new();
    let mut issues = HashMap::new();
    for project in project_store.projects() {
        match Session::catalog_in_project(&project.sessions_dir, project.id.as_str()) {
            Ok(catalog) => {
                sessions.insert(project.sessions_dir.clone(), catalog.sessions);
                issues.insert(project.sessions_dir.clone(), catalog.issues);
            }
            Err(error) => {
                sessions.insert(project.sessions_dir.clone(), Vec::new());
                issues.insert(
                    project.sessions_dir.clone(),
                    vec![SessionIssue {
                        path: project.sessions_dir.clone(),
                        message: error.to_string(),
                    }],
                );
            }
        }
    }
    (sessions, issues)
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
    fn composer_submits_only_unshifted_enter_events() {
        assert!(is_composer_submit_event(&InputEvent::PressEnter {
            secondary: false,
            shift: false,
        }));
        assert!(!is_composer_submit_event(&InputEvent::PressEnter {
            secondary: false,
            shift: true,
        }));
        assert!(!is_composer_submit_event(&InputEvent::Change));
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
    fn composer_models_require_configured_credentials() {
        let models = [
            ConfiguredModel::new(
                "deepseek-official",
                ProviderModel::new("deepseek-test", "DeepSeek Test", 10_000, None),
                Model::new("DeepSeek", "", "http://localhost", "deepseek-test", 10_000),
            ),
            ConfiguredModel::new(
                "openai",
                ProviderModel::new("gpt-test", "GPT Test", 10_000, None),
                Model::new("OpenAI", "secret", "http://localhost", "gpt-test", 10_000),
            ),
        ];

        assert_eq!(composer_model_indices(&models).collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            active_model_index(&models, Some("deepseek-official/deepseek-test")),
            Some(1)
        );
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
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), workspace);

        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let app = DesktopApp::new(
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
            );
            window.blur();
            app
        });
        cx.simulate_resize(gpui::size(px(1180.0), px(620.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                this.dispatch(
                    Action::Conversation(Box::new(ConversationAction::SubmitUser(message(
                        Role::User,
                        "hello".into(),
                    )))),
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

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn created_session_path_refreshes_the_sidebar_list(cx: &mut gpui::TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-new-session-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), workspace);

        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let app = DesktopApp::new(
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
            );
            window.blur();
            app
        });

        cx.update(|_, app| {
            view.update(app, |this, cx| {
                let mut snapshot = this.selected_runtime.read(cx).snapshot();
                let path = this.core.workspace.sessions_dir.join("created.jsonl");
                std::fs::create_dir_all(&this.core.workspace.sessions_dir).unwrap();
                std::fs::write(
                    &path,
                    format!(
                        "{{\"record\":\"session\",\"title\":\"Untitled session\",\"created_at\":{}}}\n",
                        snapshot.session.created_at
                    ),
                )
                .unwrap();
                snapshot.session.path = path;
                this.apply_selected_runtime_snapshot(snapshot);
            });
        });

        cx.read_entity(&view, |app, _| {
            assert!(!app.core.session.current.as_os_str().is_empty());
            assert_eq!(app.core.session.sessions.len(), 1);
            assert_eq!(app.core.session.sessions[0].path, app.core.session.current);
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn expanded_streaming_reasoning_stays_open_during_growth(cx: &mut gpui::TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-expanded-reasoning-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), workspace);

        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let app = DesktopApp::new(
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
            );
            window.blur();
            app
        });
        cx.simulate_resize(gpui::size(px(1180.0), px(420.0)));

        let first_chunk = (1..=40)
            .map(|line| format!("reasoning line {line}\n"))
            .collect::<String>();

        cx.update(|_, app| {
            view.update(app, |this, cx| {
                let runtime = this.selected_runtime.clone();
                runtime.update(cx, |runtime, cx| {
                    runtime.apply_test_event(AgentEvent::ReasoningDelta(first_chunk), cx);
                });
            });
        });
        cx.run_until_parked();

        cx.update(|_, app| {
            view.update(app, |this, cx| this.toggle_reasoning(0, cx));
        });
        cx.refresh().unwrap();
        cx.run_until_parked();
        cx.read_entity(&view, |app, _| {
            assert!(app.core.conversation.messages[0].expanded);
        });

        let second_chunk = (41..=80)
            .map(|line| format!("reasoning line {line}\n"))
            .collect::<String>();
        cx.update(|_, app| {
            view.update(app, |this, cx| {
                let runtime = this.selected_runtime.clone();
                runtime.update(cx, |runtime, cx| {
                    runtime.apply_test_event(AgentEvent::ReasoningDelta(second_chunk), cx);
                });
            });
        });
        cx.refresh().unwrap();
        cx.run_until_parked();

        cx.read_entity(&view, |app, _| {
            let reasoning = &app.core.conversation.messages[0];
            assert!(reasoning.text.ends_with("reasoning line 80\n"));
            assert!(reasoning.expanded);
            assert!(
                app.scroll.max_offset().y > px(40.5),
                "expanded reasoning did not overflow the transcript"
            );
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn answer_block_after_collapsed_reasoning_reflows_the_transcript(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-reasoning-tail-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), workspace);

        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let app = DesktopApp::new(
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
            );
            window.blur();
            app
        });
        cx.simulate_resize(gpui::size(px(1180.0), px(720.0)));

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                let runtime = this.selected_runtime.clone();
                runtime.update(cx, |runtime, cx| {
                    runtime.apply_test_event(AgentEvent::RunStarted("line\n".repeat(12)), cx);
                    runtime.apply_test_event(AgentEvent::ReasoningDelta("thinking".into()), cx);
                });
                this.dispatch(Action::Scroll(ScrollIntent::JumpToTail), window, cx);
            });
        });
        cx.refresh().unwrap();
        cx.run_until_parked();

        let max_offset_before = cx.read_entity(&view, |app, _| app.scroll.max_offset().y);
        assert!(
            max_offset_before <= px(40.5),
            "test transcript already overflowed before the answer: max={max_offset_before:?}"
        );

        cx.update(|_, app| {
            view.update(app, |this, cx| {
                let runtime = this.selected_runtime.clone();
                runtime.update(cx, |runtime, cx| {
                    let answer = (1..=100)
                        .map(|line| format!("auto scroll test {line}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    runtime.apply_test_event(AgentEvent::TextDelta(answer), cx);
                });
            });
        });
        cx.refresh().unwrap();
        cx.run_until_parked();

        let max_offset = cx.read_entity(&view, |app, _| app.scroll.max_offset());
        assert!(
            max_offset.y > px(40.5),
            "test answer did not make the transcript overflow: max={max_offset:?}"
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
    fn sidebar_status_prioritizes_actionable_and_live_states_over_unread() {
        assert_eq!(
            resolve_sidebar_session_status(&SessionRuntimeStatus::Running, true, true),
            Some(SidebarSessionStatus::ApprovalNeeded)
        );
        assert_eq!(
            resolve_sidebar_session_status(&SessionRuntimeStatus::Running, false, true),
            Some(SidebarSessionStatus::Running)
        );
        assert_eq!(
            resolve_sidebar_session_status(&SessionRuntimeStatus::Idle, false, true),
            Some(SidebarSessionStatus::Unread)
        );
        assert_eq!(
            resolve_sidebar_session_status(&SessionRuntimeStatus::Idle, false, false),
            None
        );
    }

    #[test]
    fn only_new_background_completions_become_unread() {
        assert!(has_new_unread_completion(0, 1, false));
        assert!(!has_new_unread_completion(0, 1, true));
        assert!(!has_new_unread_completion(1, 1, false));
    }

    #[test]
    fn only_selected_runtime_updates_drive_the_visible_transcript() {
        assert_eq!(visible_transcript_update_count(2, 5, true), 3);
        assert_eq!(visible_transcript_update_count(2, 5, false), 0);
        assert_eq!(visible_transcript_update_count(5, 5, true), 0);
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

    #[test]
    fn interrupted_stream_replay_preserves_durable_input_and_partial_output() {
        let mut messages = Vec::new();
        append_interrupted_stream(
            &mut messages,
            &[
                SessionEvent::RunStarted {
                    input: "question".into(),
                },
                SessionEvent::ReasoningDelta {
                    delta: "partial thought".into(),
                },
                SessionEvent::TextDelta {
                    delta: "partial answer".into(),
                },
                SessionEvent::RunAborted,
            ],
        );
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[0].text, "question");
        assert_eq!(messages[1].role, Role::Reasoning);
        assert_eq!(messages[2].text, "partial answer");
        assert_eq!(messages[3].role, Role::Notice);
    }

    #[test]
    fn consumed_queue_admissions_do_not_create_recovery_notices() {
        let mut messages = Vec::new();
        append_interrupted_stream(
            &mut messages,
            &[
                SessionEvent::InputAdmitted {
                    id: "input-1".into(),
                    input: "later".into(),
                    mode: kcastle_agent::InputMode::Queue,
                },
                SessionEvent::InputConsumed {
                    id: "input-1".into(),
                },
            ],
        );
        assert!(messages.is_empty());
    }
}
