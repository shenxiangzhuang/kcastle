use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpui::{
    AppContext, Bounds, Context, Entity, FocusHandle, PathPromptOptions, Pixels, Point,
    ScrollHandle, ScrollWheelEvent, Subscription, UniformListScrollHandle, Window, point, px,
};
use gpui_component::input::{InputEvent, InputState, TextareaState};
use gpui_component::{Theme, ThemeMode};
use kcastle_agent::{
    ARCHIVE_DIRECTORY, Agent, Model, Session, SessionConfig, SessionId, SessionInfo,
};

use crate::dialogs::Modal;
use crate::domain::session_document::SessionDocument;
use crate::domain::timeline::AxisRange;
use crate::domain::{
    Action, AppState, ComposerMenu, DetailsTab, Effect, Message, Role, RunState, ScrollIntent,
    Surface, TimelineMode, TrajectoryItemId, next_message_id, reduce,
};
use crate::layout::{LayoutInput, ScrollAnchor, ScrollRestore, resolve_scroll_restore};
use crate::platform::NativeTitlebarController;
use crate::platform::gpui::{
    DeferredScrollAlignment, GpuiLayoutRuntime, MeasuredBounds, MessagePresentationStore,
    SessionRuntime, SessionRuntimeSnapshot, SessionRuntimeStatus, run_effects,
};
use crate::project::{ProjectId, ProjectStore};
use crate::settings::{Appearance, EnterBehavior, ProviderModel, SettingsStore};
use crate::trajectory::TimelineModelCache;
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

#[derive(Clone, Debug)]
struct SessionViewState {
    chat_anchor: ScrollAnchor,
    trajectory_offset: Point<Pixels>,
    details_offset: Point<Pixels>,
    selected_trajectory: Option<TrajectoryItemId>,
    details_tab: DetailsTab,
    timeline_mode: TimelineMode,
    timeline_selection: Option<AxisRange>,
    timeline_viewport: Option<AxisRange>,
}

impl Default for SessionViewState {
    fn default() -> Self {
        Self {
            chat_anchor: ScrollAnchor::Tail,
            trajectory_offset: point(px(0.0), px(0.0)),
            details_offset: point(px(0.0), px(0.0)),
            selected_trajectory: None,
            details_tab: DetailsTab::Summary,
            timeline_mode: TimelineMode::Sequence,
            timeline_selection: None,
            timeline_viewport: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SessionSearchDocument {
    pub(crate) searchable: Arc<str>,
    pub(crate) summary: String,
    pub(crate) snippets: Arc<[String]>,
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

type RuntimeKey = (ProjectId, SessionId);
type OpenSessionKey = (ProjectId, PathBuf);

const MAX_CACHED_TERMINAL_RUNTIMES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionOpenCompletion {
    Current,
    WarmCache,
    Reload(u64),
    Ignore,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PendingRuntimeSelection {
    generation: u64,
    project_id: ProjectId,
    path: PathBuf,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RuntimeObservation {
    completed_runs: u64,
    transcript_updates: u64,
    presentation_sequence: u64,
    catalog_synced_revision: u64,
    metadata_generation: u64,
    is_terminal: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct TimelineDragState {
    pub(crate) pan: bool,
    pub(crate) start_value: f64,
    pub(crate) start_x: f32,
    pub(crate) record_id: Option<TrajectoryItemId>,
    pub(crate) initial_viewport: Option<AxisRange>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimelineHoverState {
    pub(crate) fraction: f64,
    pub(crate) record_id: Option<TrajectoryItemId>,
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
    pub(crate) selected_details_raw: Arc<str>,
    selected_details_raw_revision: Option<(TrajectoryItemId, usize, u64)>,
    pub(crate) timeline_bounds: Option<Bounds<Pixels>>,
    pub(crate) timeline_drag: Option<TimelineDragState>,
    pub(crate) timeline_hover: Option<TimelineHoverState>,
    pub(crate) timeline_model_cache: RefCell<Option<TimelineModelCache>>,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) selected_reasoning_effort: Option<kcastle_agent::ReasoningEffort>,
    pub(crate) model: String,
    pub(crate) project_store: ProjectStore,
    pub(crate) settings: SettingsStore,
    pub(crate) selected_started_at: Option<Instant>,
    pub(crate) project_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
    pub(crate) project_archived_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
    pub(crate) session_activity: HashMap<PathBuf, u64>,
    pub(crate) session_search_documents: HashMap<PathBuf, SessionSearchDocument>,
    session_catalog_indices: HashMap<RuntimeKey, usize>,
    pub(crate) available_update: Option<AvailableUpdate>,
    unread_sessions: HashSet<(ProjectId, SessionId)>,
    runtime_observations: HashMap<RuntimeKey, RuntimeObservation>,
    runtime_subscriptions: HashMap<RuntimeKey, Subscription>,
    runtime_recency: HashMap<RuntimeKey, u64>,
    runtime_access_clock: u64,
    open_generation: u64,
    inflight_session_opens: HashMap<OpenSessionKey, u64>,
    pending_runtime_selection: Option<PendingRuntimeSelection>,
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
                SessionDocument::default(),
                runtime_config,
            )
        });
        let runtime_subscription = cx.observe(&runtime, |this, runtime, cx| {
            this.sync_runtime_snapshot(&runtime, cx);
        });
        let initial_runtime_key = (project.id.clone(), current_session_id.clone());
        let mut project_runtimes = HashMap::new();
        project_runtimes.insert(
            project.id.clone(),
            ProjectSessionRuntimes {
                selected: current_session_id.clone(),
                sessions: HashMap::from([(current_session_id, runtime.clone())]),
            },
        );
        let project_sessions = load_project_sessions(&project_store);
        let project_archived_sessions = load_project_archived_sessions(&project_store);
        let session_activity = load_session_activity(&project_sessions);
        let session_catalog_indices =
            build_session_catalog_indices(&project_store, &project_sessions);
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
        let session_search_documents = build_session_search_documents(&project_store);
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
            selected_details_raw: Arc::from(""),
            selected_details_raw_revision: None,
            timeline_bounds: None,
            timeline_drag: None,
            timeline_hover: None,
            timeline_model_cache: RefCell::new(None),
            models,
            selected_model,
            selected_reasoning_effort,
            model,
            project_store,
            settings,
            selected_started_at: None,
            project_sessions,
            project_archived_sessions,
            session_activity,
            session_search_documents,
            session_catalog_indices,
            available_update: None,
            unread_sessions: HashSet::new(),
            runtime_observations: HashMap::new(),
            runtime_subscriptions: HashMap::from([(
                initial_runtime_key.clone(),
                runtime_subscription,
            )]),
            runtime_recency: HashMap::from([(initial_runtime_key, 1)]),
            runtime_access_clock: 1,
            open_generation: 0,
            inflight_session_opens: HashMap::new(),
            pending_runtime_selection: None,
            native_titlebar,
            view_states: HashMap::new(),
            _subscriptions: vec![
                subscription,
                search_subscription,
                trajectory_subscription,
                appearance_subscription,
                bounds_subscription,
            ],
        };
        #[cfg(not(test))]
        app.check_for_updates(window, cx);
        app
    }

    fn sync_runtime_snapshot(&mut self, runtime: &Entity<SessionRuntime>, cx: &mut Context<Self>) {
        let observation = runtime.read(cx).observation();
        let location = self.runtime_location(runtime);
        let selected = runtime.entity_id() == self.selected_runtime.entity_id();
        let mut selected_transcript_updates = 0;
        let mut presentation_update = None;
        let mut became_terminal = false;
        if let Some((project_id, session_id)) = &location {
            let key = (project_id.clone(), session_id.clone());
            let previous_observation = self
                .runtime_observations
                .get(&key)
                .copied()
                .unwrap_or_default();
            let unread_completion = has_new_unread_completion(
                previous_observation.completed_runs,
                observation.completed_runs,
                selected,
            );
            selected_transcript_updates = visible_transcript_update_count(
                previous_observation.transcript_updates,
                observation.transcript_updates,
                selected,
            );
            if unread_completion {
                self.unread_sessions.insert(key.clone());
            }
            if selected {
                self.unread_sessions.remove(&key);
                presentation_update = Some(
                    runtime
                        .read(cx)
                        .presentation_update_since(previous_observation.presentation_sequence),
                );
            }

            let catalog_missing = !observation.session.path.as_os_str().is_empty()
                && !self
                    .project_store
                    .projects()
                    .iter()
                    .find(|project| &project.id == project_id)
                    .and_then(|project| self.project_sessions.get(&project.sessions_dir))
                    .is_some_and(|sessions| {
                        sessions
                            .iter()
                            .any(|info| info.id == observation.session.id)
                    });
            self.upsert_runtime_session_metadata(project_id, &observation.session);
            let catalog_boundary = matches!(
                observation.status,
                SessionRuntimeStatus::Idle
                    | SessionRuntimeStatus::Settling
                    | SessionRuntimeStatus::Failed(_)
            );
            let should_refresh_catalog = !observation.session.path.as_os_str().is_empty()
                && (catalog_missing
                    || observation.metadata_generation != previous_observation.metadata_generation
                    || (catalog_boundary
                        && observation.durable_revision
                            != previous_observation.catalog_synced_revision));
            let catalog_refreshed =
                should_refresh_catalog && self.refresh_project_catalog(project_id);
            let is_terminal = !runtime.read(cx).is_active();
            became_terminal = !previous_observation.is_terminal && is_terminal;
            self.runtime_observations.insert(
                key,
                RuntimeObservation {
                    completed_runs: observation.completed_runs,
                    transcript_updates: observation.transcript_updates,
                    presentation_sequence: observation.presentation_sequence,
                    catalog_synced_revision: if catalog_refreshed {
                        observation.durable_revision
                    } else {
                        previous_observation.catalog_synced_revision
                    },
                    metadata_generation: observation.metadata_generation,
                    is_terminal,
                },
            );
        }
        if location.is_none()
            && !observation.session.path.as_os_str().is_empty()
            && !self
                .project_store
                .projects()
                .iter()
                .find(|project| project.id.as_str() == observation.session.project_id)
                .and_then(|project| self.project_sessions.get(&project.sessions_dir))
                .is_some_and(|sessions| {
                    sessions
                        .iter()
                        .any(|info| info.id == observation.session.id)
                })
        {
            self.refresh_project_session_cache();
            self.refresh_session_search_documents();
        }
        if became_terminal {
            self.evict_terminal_runtimes(cx);
        }
        if !selected {
            cx.notify();
            return;
        }
        let snapshot = runtime.read(cx).snapshot();
        self.apply_selected_runtime_snapshot(snapshot);
        self.refresh_selected_details_raw(cx);
        if let (Some((project_id, session_id)), Some((_, full_reset, messages))) =
            (location.as_ref(), presentation_update)
        {
            let namespace = presentation_namespace(project_id, session_id);
            if full_reset
                || !self.message_presentations.sync_changed(
                    &namespace,
                    messages.iter().map(|message| {
                        (
                            message.key,
                            message.revision,
                            message.text.as_str(),
                            message.role == Role::Assistant,
                        )
                    }),
                )
            {
                self.sync_message_presentations();
            }
        } else if location.is_none() {
            self.sync_message_presentations();
        }
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
        debug_assert_eq!(
            snapshot.view.revision,
            snapshot.view.trajectory.source_revision()
        );
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
        let session_id = snapshot.session.id.clone();
        self.core.session_view = snapshot.view;
        if previous_path != snapshot.session.path {
            self.core.transient_messages.clear();
        }
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
            SessionRuntimeStatus::Settling => RunState::Running {
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
            let needs_catalog_refresh = !snapshot.session.path.as_os_str().is_empty()
                && !self
                    .core
                    .session
                    .sessions
                    .iter()
                    .any(|session| session.id == session_id);
            if needs_catalog_refresh
                && let Some(project_id) = self
                    .project_store
                    .project(self.core.workspace.active_project)
                    .map(|project| project.id.clone())
            {
                self.refresh_project_catalog(&project_id);
            }
        }
    }

    fn register_runtime(
        &mut self,
        project_id: ProjectId,
        runtime: Entity<SessionRuntime>,
        cx: &mut Context<Self>,
    ) {
        let observation = runtime.read(cx).observation();
        let session_id = observation.session.id;
        let key = (project_id.clone(), session_id.clone());
        let subscription = cx.observe(&runtime, |this, runtime, cx| {
            this.sync_runtime_snapshot(&runtime, cx);
        });
        self.runtime_subscriptions.insert(key.clone(), subscription);
        self.runtime_observations
            .entry(key.clone())
            .or_insert(RuntimeObservation {
                completed_runs: observation.completed_runs,
                transcript_updates: observation.transcript_updates,
                presentation_sequence: 0,
                catalog_synced_revision: observation.durable_revision,
                metadata_generation: observation.metadata_generation,
                is_terminal: !runtime.read(cx).is_active(),
            });
        let project =
            self.project_runtimes
                .entry(project_id)
                .or_insert_with(|| ProjectSessionRuntimes {
                    selected: session_id.clone(),
                    sessions: HashMap::new(),
                });
        project.sessions.insert(session_id, runtime);
        self.touch_runtime(&key);
        self.evict_terminal_runtimes(cx);
    }

    fn touch_runtime(&mut self, key: &RuntimeKey) {
        self.runtime_access_clock = self.runtime_access_clock.saturating_add(1);
        self.runtime_recency
            .insert(key.clone(), self.runtime_access_clock);
    }

    /// Keeps a bounded cache of reopenable, inactive runtime documents. Only the globally selected
    /// runtime, active runtimes, and lightweight drafts are protected. A project's remembered
    /// selection is just an identity hint: if its document is evicted, switching back reloads that
    /// session from the store. Eviction drops the GPUI subscription together with the entity so
    /// reopening has exactly one owner and observer.
    fn evict_terminal_runtimes(&mut self, cx: &Context<Self>) {
        let selected_entity = self.selected_runtime.entity_id();
        let mut candidates = Vec::new();
        for (project_id, runtimes) in &self.project_runtimes {
            for (session_id, runtime) in &runtimes.sessions {
                if runtime.entity_id() == selected_entity {
                    continue;
                }
                let observation = runtime.read(cx).observation();
                if observation.session.path.as_os_str().is_empty() || runtime.read(cx).is_active() {
                    continue;
                }
                let key = (project_id.clone(), session_id.clone());
                candidates.push((
                    self.runtime_recency.get(&key).copied().unwrap_or_default(),
                    key,
                ));
            }
        }
        if candidates.len() <= MAX_CACHED_TERMINAL_RUNTIMES {
            return;
        }
        candidates.sort_by_key(|(recency, _)| *recency);
        let remove_count = candidates.len() - MAX_CACHED_TERMINAL_RUNTIMES;
        for (_, key) in candidates.into_iter().take(remove_count) {
            self.remove_cached_runtime(&key);
        }
    }

    fn remove_cached_runtime(&mut self, key: &RuntimeKey) {
        if let Some(runtimes) = self.project_runtimes.get_mut(&key.0) {
            runtimes.sessions.remove(&key.1);
        }
        self.runtime_subscriptions.remove(key);
        self.runtime_recency.remove(key);
        self.runtime_observations.remove(key);
    }

    fn create_runtime(
        &mut self,
        project_index: usize,
        mut session: Session,
        cx: &mut Context<Self>,
    ) -> Option<Entity<SessionRuntime>> {
        let project = self.project_store.project(project_index)?.clone();
        if let Some(runtime) = self
            .project_runtimes
            .get(&project.id)
            .and_then(|runtimes| runtimes.sessions.get(&session.info().id))
            .cloned()
        {
            return Some(runtime);
        }
        let document = match SessionDocument::from_events(session.take_events()) {
            Ok(document) => document,
            Err(_) => return None,
        };
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
                document,
                config,
            )
        });
        self.register_runtime(project.id, runtime.clone(), cx);
        Some(runtime)
    }

    /// Reuses a cached runtime only when its idle Agent, metadata, configuration, and journal
    /// revision all match the snapshot just loaded from SQLite. An active runtime remains the
    /// single owner if an asynchronous open races with a new run.
    fn reconcile_loaded_runtime(
        &mut self,
        project_index: usize,
        session: Session,
        cx: &mut Context<Self>,
    ) -> Option<Entity<SessionRuntime>> {
        let project = self.project_store.project(project_index)?.clone();
        let key = (project.id.clone(), session.info().id.clone());
        if let Some(runtime) = self
            .project_runtimes
            .get(&project.id)
            .and_then(|runtimes| runtimes.sessions.get(&session.info().id))
            .cloned()
        {
            if runtime.read(cx).is_active() || runtime.read(cx).matches_loaded_session(&session) {
                return Some(runtime);
            }
            self.remove_cached_runtime(&key);
        }
        self.create_runtime(project_index, session, cx)
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
        let observation = self
            .project_runtime(project_index, path)?
            .read(cx)
            .observation();
        let unread = self
            .unread_sessions
            .contains(&(project_id, observation.session.id));
        resolve_sidebar_session_status(&observation.status, observation.approval_needed, unread)
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
        let location = self.runtime_location(&runtime);
        if let Some(key) = &location {
            self.unread_sessions.remove(key);
        }
        // Publishing the target entity and retiring the selection capability happen in one GPUI
        // update. No command can observe the new selection while an old loading gate remains.
        self.selected_runtime = runtime.clone();
        self.pending_runtime_selection = None;
        self.timeline_drag = None;
        self.timeline_hover = None;
        let snapshot = runtime.read(cx).snapshot();
        if let Some((project_id, _)) = &location
            && let Some(runtimes) = self.project_runtimes.get_mut(project_id)
        {
            runtimes.selected = snapshot.session.id.clone();
        }
        self.apply_selected_runtime_snapshot(snapshot);
        self.sync_message_presentations();
        if let Some(key) = &location {
            let observation = runtime.read(cx).observation();
            let previous = self.runtime_observations.entry(key.clone()).or_default();
            previous.presentation_sequence = observation.presentation_sequence;
            previous.completed_runs = observation.completed_runs;
            previous.transcript_updates = observation.transcript_updates;
            previous.metadata_generation = observation.metadata_generation;
            self.touch_runtime(key);
        }
        self.restore_current_view_state(cx);
        self.refresh_selected_details_raw(cx);
        self.evict_terminal_runtimes(cx);
        cx.notify();
    }

    /// Advances the single user-selection epoch used by asynchronous session opens.
    ///
    /// Every synchronous user choice (including choosing the current draft again) must advance
    /// this epoch before it can return. An older open may still finish and populate the bounded
    /// runtime cache, but it can no longer replace the user's newer selection.
    fn begin_runtime_selection_intent(&mut self) -> u64 {
        self.open_generation = self.open_generation.saturating_add(1);
        // A newer synchronous choice cancels the preceding loading capability. Async choices set
        // their replacement target before returning to the event loop.
        self.pending_runtime_selection = None;
        self.open_generation
    }

    fn begin_pending_runtime_selection(
        &mut self,
        generation: u64,
        project_id: ProjectId,
        path: PathBuf,
    ) {
        self.pending_runtime_selection = Some(PendingRuntimeSelection {
            generation,
            project_id,
            path,
        });
    }

    pub(crate) fn selection_pending(&self) -> bool {
        self.pending_runtime_selection.is_some()
    }

    fn pending_selection_targets(&self, project_index: usize, path: &Path) -> bool {
        let Some(project_id) = self
            .project_store
            .project(project_index)
            .map(|project| &project.id)
        else {
            return false;
        };
        self.pending_runtime_selection
            .as_ref()
            .is_some_and(|pending| {
                &pending.project_id == project_id && same_path(&pending.path, path)
            })
    }

    fn session_open_matches_current_intent(
        &self,
        requested_generation: u64,
        project_id: &ProjectId,
    ) -> bool {
        requested_generation == self.open_generation
            && self
                .project_store
                .project(self.core.workspace.active_project)
                .is_some_and(|project| &project.id == project_id)
    }

    /// Retires exactly one loader for `key` and decides how its snapshot may be used.
    ///
    /// A later request for the same key takes over the in-flight slot by advancing the requested
    /// generation. The loader that was already running cannot satisfy that later request: its
    /// SQLite snapshot may predate the later click. When that later request is still the current
    /// selection intent, start a fresh load; otherwise discard the superseded result rather than
    /// warming the cache with a snapshot that a newer same-key request explicitly replaced.
    fn finish_session_open_request(
        &mut self,
        key: &OpenSessionKey,
        started_generation: u64,
        project_id: &ProjectId,
    ) -> SessionOpenCompletion {
        let Some(requested_generation) = self.inflight_session_opens.remove(key) else {
            return SessionOpenCompletion::Ignore;
        };
        let is_current = self.session_open_matches_current_intent(requested_generation, project_id);
        if requested_generation != started_generation {
            return if is_current {
                SessionOpenCompletion::Reload(requested_generation)
            } else {
                SessionOpenCompletion::Ignore
            };
        }
        if is_current {
            SessionOpenCompletion::Current
        } else {
            SessionOpenCompletion::WarmCache
        }
    }

    /// Installs the selection capability before a loader can run. Returning `false` means an
    /// existing same-key loader now owns the newer generation and must be allowed to retire into
    /// `Reload`; no second loader should be spawned yet.
    fn start_session_open_request(&mut self, key: &OpenSessionKey, generation: u64) -> bool {
        if !self.session_open_matches_current_intent(generation, &key.0) {
            return false;
        }
        self.begin_pending_runtime_selection(generation, key.0.clone(), key.1.clone());
        if let Some(requested_generation) = self.inflight_session_opens.get_mut(key) {
            *requested_generation = generation;
            return false;
        }
        self.inflight_session_opens.insert(key.clone(), generation);
        true
    }

    /// Resolves a failed current selection without ever exposing another project's runtime under
    /// the target workspace. Same-project failures keep the prior coherent selection; cross-
    /// project failures atomically fall back to the target project's draft.
    fn resolve_failed_runtime_selection(
        &mut self,
        generation: u64,
        project_id: &ProjectId,
        path: &Path,
        project_index: usize,
        cx: &mut Context<Self>,
    ) -> bool {
        let matches_pending = self
            .pending_runtime_selection
            .as_ref()
            .is_some_and(|pending| {
                pending.generation == generation
                    && &pending.project_id == project_id
                    && same_path(&pending.path, path)
            });
        if !matches_pending {
            return false;
        }

        let selected_belongs_to_target = self
            .runtime_location(&self.selected_runtime)
            .is_some_and(|(selected_project_id, _)| &selected_project_id == project_id);
        if selected_belongs_to_target {
            self.pending_runtime_selection = None;
            return false;
        }

        let Some(runtime) = self.select_or_create_project_draft(project_index, cx) else {
            // Retain the gate if the target project disappeared unexpectedly. A newer project
            // selection can supersede it, but no command may fall through to the old runtime.
            return false;
        };
        self.select_runtime(runtime, cx);
        true
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
                        .observation()
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
        let details_may_change = matches!(
            &action,
            Action::SetDetailsTab(_) | Action::SelectDetails(_) | Action::RestoreSessionView { .. }
        );
        let effects = self.transition(action);
        run_effects(self, effects, window, cx);
        if details_may_change {
            self.refresh_selected_details_raw(cx);
        }
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
        let details_may_change = matches!(
            &action,
            Action::SetDetailsTab(_) | Action::SelectDetails(_) | Action::RestoreSessionView { .. }
        );
        for effect in self.transition(action) {
            let Effect::ApplyChatTail = effect;
            self.layout_runtime.request_tail_realign();
        }
        if details_may_change {
            self.refresh_selected_details_raw(cx);
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
        !self.selection_pending() && matches!(self.core.run, RunState::Running { .. })
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
        if self.selection_pending() {
            return;
        }
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
        if self.selection_pending() {
            return;
        }
        self.selected_runtime
            .update(cx, |runtime, cx| runtime.decide(call_id, allow, cx));
    }

    pub(crate) fn abort(&mut self, cx: &mut Context<Self>) {
        if self.selection_pending() {
            return;
        }
        self.selected_runtime
            .update(cx, |runtime, cx| runtime.abort(cx));
    }

    pub(crate) fn notice(&mut self, text: impl Into<String>) {
        let _ = self.transition(Action::AppendTransientNotice(Box::new(message(
            Role::Notice,
            text.into(),
        ))));
        self.sync_message_presentations();
    }

    fn sync_message_presentations(&mut self) {
        let namespace = self
            .runtime_location(&self.selected_runtime)
            .map(|(project_id, session_id)| presentation_namespace(&project_id, &session_id))
            .unwrap_or_else(|| "unregistered-session".to_owned());
        let messages = self
            .core
            .session_view
            .conversation
            .messages
            .iter()
            .chain(self.core.transient_messages.iter());
        self.message_presentations.replace_all(
            namespace,
            messages.map(|message| {
                (
                    message.key,
                    message.revision,
                    message.text.as_str(),
                    message.role == Role::Assistant,
                )
            }),
        );
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
            state.selected_trajectory = self.core.details.selected.clone();
            state.details_tab = self.core.details.tab;
            state.details_offset = self.details_scroll.offset();
            state.timeline_mode = self.core.trajectory.mode;
            state.timeline_selection = self.core.trajectory.selected_range;
            state.timeline_viewport = self.core.trajectory.visible_range;
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
            .cloned()
            .unwrap_or_default();
        if self.core.surface == Surface::Trajectory {
            let _ = self.transition(Action::SetTimelineMode(state.timeline_mode));
            let _ = self.transition(Action::SetTimelineSelection(state.timeline_selection));
            let _ = self.transition(Action::SetTimelineViewport(state.timeline_viewport));
            self.timeline_drag = None;
            self.timeline_hover = None;
            self.trajectory_scroll
                .0
                .borrow()
                .base_handle
                .set_offset(state.trajectory_offset);
            let selected = state.selected_trajectory.filter(|selected| {
                self.core
                    .session_view
                    .trajectory
                    .records
                    .iter()
                    .any(|record| &record.id == selected)
            });
            let _ = self.transition(Action::RestoreSessionView {
                selected: selected.clone(),
                details_tab: state.details_tab,
                follow_chat_tail: matches!(state.chat_anchor, ScrollAnchor::Tail),
            });
            if let Some(selected) = selected
                && let Some(index) = self
                    .core
                    .session_view
                    .trajectory
                    .records
                    .iter()
                    .position(|record| record.id == selected)
            {
                self.scroll_trajectory_to_record(index, cx);
            }
            self.details_scroll.set_offset(state.details_offset);
            self.refresh_selected_details_raw(cx);
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
        if self.selection_pending() {
            return;
        }
        if self.core.session.current.as_os_str().is_empty() {
            self.notice("Start the session before exporting its log");
            cx.notify();
            return;
        }
        let source = self.core.session.current.clone();
        let suggested = format!(
            "{}.jsonl",
            safe_file_name(&self.core.session_view.conversation.title)
        );
        let receiver = cx.prompt_for_new_path(&self.core.workspace.cwd, Some(&suggested));
        cx.spawn_in(window, async move |this, cx| {
            let selection = receiver.await;
            let copied = match selection {
                Ok(Ok(Some(destination))) => tokio::task::spawn_blocking(move || {
                    Session::open_readonly(source)?.export_jsonl(destination)
                })
                .await
                .map_err(|error| error.to_string())
                .and_then(|result| result.map_err(|error| error.to_string()))
                .map(|()| Some("Session log exported".to_owned())),
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
        self.toggle_message_presentation(index, Role::Tool, cx);
    }

    pub(crate) fn toggle_reasoning(&mut self, index: usize, cx: &mut Context<Self>) {
        self.toggle_message_presentation(index, Role::Reasoning, cx);
    }

    fn toggle_message_presentation(&mut self, index: usize, role: Role, cx: &mut Context<Self>) {
        let Some(message_id) = self
            .core
            .session_view
            .conversation
            .messages
            .get(index)
            .filter(|message| message.role == role)
            .map(|message| message.key)
        else {
            return;
        };
        if self
            .message_presentations
            .toggle_expanded(message_id)
            .is_some_and(|expanded| expanded && self.core.follow_chat_tail)
        {
            self.layout_runtime.request_tail_realign();
        }
        cx.notify();
    }

    pub(crate) fn inspect_tool(
        &mut self,
        index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .core
            .session_view
            .conversation
            .messages
            .get(index)
            .is_some_and(|message| message.role == Role::Tool)
        {
            self.save_current_view_state();
            let Some(call_id) = self.core.session_view.conversation.messages[index]
                .tool_call_id
                .as_deref()
            else {
                return;
            };
            let Some((trajectory_index, record)) = self
                .core
                .session_view
                .trajectory
                .records
                .iter()
                .enumerate()
                .find(|(_, record)| record.call_id.as_deref() == Some(call_id))
            else {
                return;
            };
            let message_id = record.id.clone();
            let mut effects = self.transition(Action::ShowTrajectory);
            effects.extend(self.transition(Action::SelectDetails(Some(message_id))));
            run_effects(self, effects, window, cx);
            self.dispatch_local(Action::SetDetailsTab(DetailsTab::Summary), cx);
            self.details_scroll.set_offset(point(px(0.0), px(0.0)));
            self.dispatch_local(Action::ExpandTrajectoryGroups, cx);
            self.scroll_trajectory_to_record(trajectory_index, cx);
            cx.notify();
        }
    }

    pub(crate) fn rate_message(&mut self, index: usize, positive: bool, cx: &mut Context<Self>) {
        let Some(message_id) = self
            .core
            .session_view
            .conversation
            .messages
            .get(index)
            .filter(|message| message.role == Role::Assistant)
            .map(|message| message.key)
        else {
            return;
        };
        if self
            .message_presentations
            .rate(message_id, positive)
            .is_some()
        {
            cx.notify();
        }
    }

    pub(crate) fn refresh_selected_details_raw(&mut self, cx: &Context<Self>) {
        if self.core.details.tab != DetailsTab::Raw {
            self.selected_details_raw = Arc::from("");
            self.selected_details_raw_revision = None;
            return;
        }
        let Some(id) = self.core.details.selected.as_ref() else {
            self.selected_details_raw = Arc::from("");
            self.selected_details_raw_revision = None;
            return;
        };
        let Some((source_count, source_revision)) =
            self.selected_runtime.read(cx).details_raw_revision(id)
        else {
            self.selected_details_raw = Arc::from("");
            self.selected_details_raw_revision = None;
            return;
        };
        let revision = (id.clone(), source_count, source_revision);
        if self.selected_details_raw_revision.as_ref() == Some(&revision) {
            return;
        }
        self.selected_details_raw = Arc::from(self.selected_runtime.read(cx).details_raw(id));
        self.selected_details_raw_revision = Some(revision);
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
        self.cancel_timeline_gesture();
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
        if self.selection_pending() {
            self.dispatch_local(Action::SetComposerMenu(None), cx);
            return;
        }
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

    fn select_new_chat_draft(&mut self, cx: &mut Context<Self>) -> bool {
        self.begin_runtime_selection_intent();
        self.save_current_view_state();
        let Some(runtime) =
            self.select_or_create_project_draft(self.core.workspace.active_project, cx)
        else {
            self.notice("Could not create a session runtime for this project");
            return false;
        };
        self.select_runtime(runtime, cx);
        true
    }

    pub(crate) fn new_chat(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.select_new_chat_draft(cx) {
            return;
        }
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
        if index != self.core.workspace.active_project {
            self.switch_project(index, window, cx);
        }
        if index == self.core.workspace.active_project {
            self.new_chat(window, cx);
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
        let generation = self.begin_runtime_selection_intent();
        self.save_current_view_state();
        let sessions = self
            .project_sessions
            .get(&project.sessions_dir)
            .cloned()
            .unwrap_or_default();
        self.dispatch(
            Action::ActivateWorkspace {
                index,
                cwd: project.path.clone(),
                sessions_dir: project.sessions_dir.clone(),
                sessions,
            },
            window,
            cx,
        );
        self.refresh_session_search_documents();
        let remembered = self
            .project_runtimes
            .get(&project.id)
            .map(|runtimes| runtimes.selected.clone());
        let runtime = remembered.as_ref().and_then(|session_id| {
            self.project_runtimes
                .get(&project.id)
                .and_then(|runtimes| runtimes.sessions.get(session_id))
                .cloned()
        });
        if let Some(runtime) = runtime {
            let observation = runtime.read(cx).observation();
            if runtime.read(cx).is_active() || observation.session.path.as_os_str().is_empty() {
                self.select_runtime(runtime, cx);
            } else {
                self.open_session_async(observation.session.path, generation, window, cx);
            }
        } else if let Some(path) = remembered.and_then(|session_id| {
            self.project_sessions
                .get(&project.sessions_dir)
                .and_then(|sessions| sessions.iter().find(|session| session.id == session_id))
                .map(|session| session.path.clone())
        }) {
            self.open_session_async(path, generation, window, cx);
        } else if let Some(runtime) = self.select_or_create_project_draft(index, cx) {
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
        let runtime_keys = self
            .project_runtimes
            .get(&project.id)
            .into_iter()
            .flat_map(|runtimes| runtimes.sessions.keys())
            .cloned()
            .map(|session_id| (project.id.clone(), session_id))
            .collect::<Vec<_>>();
        let mut presentation_sessions = runtime_keys
            .iter()
            .map(|(_, session_id)| session_id.clone())
            .collect::<HashSet<_>>();
        if let Some(sessions) = self.project_sessions.get(&project.sessions_dir) {
            presentation_sessions.extend(sessions.iter().map(|session| session.id.clone()));
        }
        for key in runtime_keys {
            self.remove_cached_runtime(&key);
        }
        self.project_runtimes.remove(&project.id);
        for session_id in presentation_sessions {
            self.message_presentations
                .remove_session(&presentation_namespace(&project.id, &session_id));
        }
        self.inflight_session_opens
            .retain(|(project_id, _), _| project_id != &project.id);
        self.session_catalog_indices
            .retain(|(project_id, _), _| project_id != &project.id);
        self.unread_sessions
            .retain(|(project_id, _)| project_id != &project.id);
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
        let generation = self.begin_runtime_selection_intent();
        if path == self.core.session.current {
            return;
        }
        self.save_current_view_state();
        if let Some(runtime) = self.active_project_runtime(&path) {
            if runtime.read(cx).is_active() {
                self.select_runtime(runtime, cx);
                self.input.update(cx, |input, cx| {
                    input.set_placeholder("Message the agent", window, cx);
                    input.focus(window, cx);
                });
            } else {
                self.open_session_async(path, generation, window, cx);
            }
            return;
        }
        self.open_session_async(path, generation, window, cx);
    }

    fn open_session_async(
        &mut self,
        path: PathBuf,
        generation: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let project_index = self.core.workspace.active_project;
        let Some(project) = self.project_store.project(project_index).cloned() else {
            return;
        };
        let project_id = project.id.clone();
        let storage_project_id = project.id.as_str().to_owned();
        let key = (project_id.clone(), path.clone());
        if !self.start_session_open_request(&key, generation) {
            return;
        }
        cx.spawn_in(window, async move |this, cx| {
            let session = Session::open_in_project(path, &storage_project_id).await;
            let _ = cx.update(|window, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        let completion =
                            this.finish_session_open_request(&key, generation, &project_id);
                        if let SessionOpenCompletion::Reload(requested_generation) = completion {
                            this.open_session_async(
                                key.1.clone(),
                                requested_generation,
                                window,
                                cx,
                            );
                            return;
                        }
                        if completion == SessionOpenCompletion::Ignore {
                            return;
                        }
                        let resolved_project_index = this
                            .project_store
                            .project(project_index)
                            .filter(|project| project.id == project_id)
                            .map(|_| project_index)
                            .or_else(|| {
                                this.project_store
                                    .projects()
                                    .iter()
                                    .position(|project| project.id == project_id)
                            });
                        let is_current_intent = completion == SessionOpenCompletion::Current;
                        match session {
                            Ok(session) => {
                                if let Some(project_index) = resolved_project_index {
                                    let runtime = if is_current_intent {
                                        this.reconcile_loaded_runtime(project_index, session, cx)
                                    } else {
                                        // A stale completion may warm an empty cache, but it must
                                        // never replace a runtime that the user selected or started
                                        // after this request began.
                                        this.create_runtime(project_index, session, cx)
                                    };
                                    if is_current_intent {
                                        if let Some(runtime) = runtime {
                                            this.select_runtime(runtime, cx);
                                            this.input.update(cx, |input, cx| {
                                                input.set_placeholder(
                                                    "Message the agent",
                                                    window,
                                                    cx,
                                                );
                                                input.focus(window, cx);
                                            });
                                        } else {
                                            let fell_back = this.resolve_failed_runtime_selection(
                                                generation,
                                                &project_id,
                                                &key.1,
                                                project_index,
                                                cx,
                                            );
                                            if fell_back {
                                                this.input.update(cx, |input, cx| {
                                                    input.set_placeholder(
                                                        "Describe what you want to build",
                                                        window,
                                                        cx,
                                                    );
                                                });
                                            }
                                            this.notice(
                                                "Could not open session: its projection is invalid",
                                            );
                                            this.input
                                                .update(cx, |input, cx| input.focus(window, cx));
                                        }
                                    }
                                }
                            }
                            Err(error) if is_current_intent => {
                                if let Some(project_index) = resolved_project_index {
                                    let fell_back = this.resolve_failed_runtime_selection(
                                        generation,
                                        &project_id,
                                        &key.1,
                                        project_index,
                                        cx,
                                    );
                                    if fell_back {
                                        this.input.update(cx, |input, cx| {
                                            input.set_placeholder(
                                                "Describe what you want to build",
                                                window,
                                                cx,
                                            );
                                        });
                                    }
                                }
                                let message = format!("Could not open session: {error}");
                                this.notice(message);
                                this.input.update(cx, |input, cx| input.focus(window, cx));
                            }
                            Err(_) => {}
                        }
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
        if self.pending_selection_targets(project_index, &path) {
            self.notice("Wait for this session to finish opening before renaming it");
            return;
        }
        if path.as_os_str().is_empty() || self.session_is_active(project_index, &path, cx) {
            self.notice("This session cannot be renamed while it is active");
            return;
        }
        let runtime = self.project_runtime(project_index, &path).or_else(|| {
            let project_id = self.project_store.project(project_index)?.id.as_str();
            let session = Session::open_writable_in_project(&path, project_id).ok()?;
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

    pub(crate) fn archive_target_session(
        &mut self,
        project_index: usize,
        path: PathBuf,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.pending_selection_targets(project_index, &path) {
            self.notice("Wait for this session to finish opening before archiving it");
            return;
        }
        if path.as_os_str().is_empty() || self.session_is_active(project_index, &path, cx) {
            self.notice("This session cannot be archived while it is active");
            return;
        }
        let Some(session) = self.target_session_info(project_index, &path) else {
            self.notice("Could not find the target session");
            return;
        };
        match Session::archive(&session) {
            Ok(_) => {
                self.remove_session_projection(project_index, &session, &path, window, cx);
                self.reload_archived_sessions(project_index);
                self.notice(format!("Archived “{}”", session.title));
            }
            Err(error) => self.notice(format!("Could not archive session: {error}")),
        }
        cx.notify();
    }

    pub(crate) fn restore_archived_session(
        &mut self,
        project_index: usize,
        session: SessionInfo,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match Session::restore(&session) {
            Ok(_) => {
                self.reload_archived_sessions(project_index);
                self.reload_project_session_list(project_index, window, cx);
                self.notice(format!("Restored “{}”", session.title));
            }
            Err(error) => self.notice(format!("Could not restore session: {error}")),
        }
        cx.notify();
    }

    pub(crate) fn delete_archived_session(
        &mut self,
        project_index: usize,
        session: SessionInfo,
        cx: &mut Context<Self>,
    ) {
        match Session::delete(&session) {
            Ok(()) => {
                self.reload_archived_sessions(project_index);
                self.notice(format!("Deleted “{}”", session.title));
            }
            Err(error) => self.notice(format!("Could not delete archived session: {error}")),
        }
        cx.notify();
    }

    fn remove_session_projection(
        &mut self,
        project_index: usize,
        session: &SessionInfo,
        previous_path: &Path,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Some(project) = self.project_store.project(project_index) {
            let key = (project.id.clone(), session.id.clone());
            let namespace = presentation_namespace(&project.id, &session.id);
            self.remove_cached_runtime(&key);
            self.message_presentations.remove_session(&namespace);
            self.unread_sessions.remove(&key);
        }
        self.reload_project_session_list(project_index, window, cx);
        let removed_selected = project_index == self.core.workspace.active_project
            && same_path(previous_path, &self.core.session.current);
        if removed_selected {
            self.begin_runtime_selection_intent();
            if let Some(runtime) = self.select_or_create_project_draft(project_index, cx) {
                self.select_runtime(runtime, cx);
                self.input.update(cx, |input, cx| {
                    input.set_placeholder("Describe what you want to build", window, cx);
                    input.focus(window, cx);
                });
            }
        }
    }

    fn reload_project_session_list(
        &mut self,
        project_index: usize,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
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
    }

    fn reload_archived_sessions(&mut self, project_index: usize) {
        let Some(project) = self.project_store.project(project_index) else {
            return;
        };
        let sessions_dir = project.sessions_dir.clone();
        let directory = sessions_dir.join(ARCHIVE_DIRECTORY);
        match Session::catalog_in_project(&directory, project.id.as_str()) {
            Ok(catalog) => {
                self.project_archived_sessions
                    .insert(sessions_dir.clone(), catalog.sessions);
            }
            Err(_) => {
                self.project_archived_sessions
                    .insert(sessions_dir, Vec::new());
            }
        }
    }

    pub(crate) fn set_reasoning_effort(
        &mut self,
        effort: kcastle_agent::ReasoningEffort,
        cx: &mut Context<Self>,
    ) {
        if self.selection_pending() || self.task_active() {
            return;
        }
        self.selected_runtime.update(cx, |runtime, cx| {
            runtime.set_reasoning_effort(effort.clone(), cx)
        });
        self.selected_reasoning_effort = Some(effort);
        cx.notify();
    }

    pub(crate) fn select_model(&mut self, index: usize, cx: &mut Context<Self>) {
        if self.selection_pending()
            || self.task_active()
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
        self.session_search_documents = build_session_search_documents(&self.project_store);
    }

    fn upsert_runtime_session_metadata(&mut self, project_id: &ProjectId, session: &SessionInfo) {
        if session.path.as_os_str().is_empty() {
            return;
        }
        let Some(project) = self
            .project_store
            .projects()
            .iter()
            .find(|project| &project.id == project_id)
        else {
            return;
        };
        let sessions = self
            .project_sessions
            .entry(project.sessions_dir.clone())
            .or_default();
        let key = (project_id.clone(), session.id.clone());
        let index = self
            .session_catalog_indices
            .get(&key)
            .copied()
            .filter(|index| {
                sessions
                    .get(*index)
                    .is_some_and(|existing| existing.id == session.id)
            })
            .or_else(|| {
                sessions
                    .iter()
                    .position(|existing| existing.id == session.id)
            });
        let index = if let Some(index) = index {
            sessions[index] = session.clone();
            index
        } else {
            sessions.push(session.clone());
            sessions.len() - 1
        };
        self.session_catalog_indices.insert(key, index);
        self.session_activity
            .insert(session.path.clone(), session.updated_at);
        if self
            .project_store
            .project(self.core.workspace.active_project)
            .is_some_and(|active| &active.id == project_id)
        {
            if self
                .core
                .session
                .sessions
                .get(index)
                .is_some_and(|existing| existing.id == session.id)
            {
                self.core.session.sessions[index] = session.clone();
            } else if index == self.core.session.sessions.len() {
                self.core.session.sessions.push(session.clone());
            } else {
                self.core.session.sessions.clone_from(sessions);
            }
        }
    }

    /// Refreshes one project's metadata and search projection from one SQLite catalog snapshot.
    /// Invalid/stale rows remain absent because filtering happens in `SessionStore::catalog`.
    fn refresh_project_catalog(&mut self, project_id: &ProjectId) -> bool {
        let Some(project) = self
            .project_store
            .projects()
            .iter()
            .find(|project| &project.id == project_id)
            .cloned()
        else {
            return false;
        };
        let Ok(catalog) = Session::catalog_in_project(&project.sessions_dir, project.id.as_str())
        else {
            return false;
        };

        if let Some(previous) = self.project_sessions.get(&project.sessions_dir) {
            for session in previous {
                self.session_activity.remove(&session.path);
                self.session_search_documents.remove(&session.path);
            }
        }
        for session in &catalog.sessions {
            self.session_activity
                .insert(session.path.clone(), session.updated_at);
        }
        for (path, search) in catalog.search {
            self.session_search_documents.insert(
                path,
                session_search_document(search.values, search.searchable),
            );
        }
        self.project_sessions
            .insert(project.sessions_dir.clone(), catalog.sessions);
        self.session_catalog_indices
            .retain(|(indexed_project_id, _), _| indexed_project_id != project_id);
        if let Some(sessions) = self.project_sessions.get(&project.sessions_dir) {
            self.session_catalog_indices.extend(
                sessions
                    .iter()
                    .enumerate()
                    .map(|(index, session)| ((project_id.clone(), session.id.clone()), index)),
            );
        }
        if self
            .project_store
            .project(self.core.workspace.active_project)
            .is_some_and(|active| active.id == project.id)
        {
            self.core.session.sessions = self
                .project_sessions
                .get(&project.sessions_dir)
                .cloned()
                .unwrap_or_default();
        }
        true
    }

    fn reload_sessions(&mut self, sessions_dir: &Path) -> Vec<SessionInfo> {
        let project_id = self
            .project_store
            .projects()
            .iter()
            .find(|project| project.sessions_dir == sessions_dir)
            .map(|project| project.id.clone());
        let storage_project_id = project_id
            .as_ref()
            .map(ProjectId::as_str)
            .unwrap_or(kcastle_agent::DEFAULT_PROJECT_ID);
        let sessions = match Session::catalog_in_project(sessions_dir, storage_project_id) {
            Ok(catalog) => catalog.sessions,
            Err(_) => self
                .project_sessions
                .get(sessions_dir)
                .cloned()
                .unwrap_or_default(),
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
        if let Some(project_id) = project_id {
            self.session_catalog_indices
                .retain(|(indexed_project_id, _), _| indexed_project_id != &project_id);
            self.session_catalog_indices.extend(
                sessions
                    .iter()
                    .enumerate()
                    .map(|(index, session)| ((project_id.clone(), session.id.clone()), index)),
            );
        }
        sessions
    }

    fn refresh_project_session_cache(&mut self) {
        let sessions = load_project_sessions(&self.project_store);
        let archived_sessions = load_project_archived_sessions(&self.project_store);
        self.session_activity = load_session_activity(&sessions);
        self.project_sessions = sessions;
        self.project_archived_sessions = archived_sessions;
        self.session_catalog_indices =
            build_session_catalog_indices(&self.project_store, &self.project_sessions);
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
        started_at_ms: Some(now_ms()),
        duration_ms: None,
        turn: 0,
        step: 0,
        request_id: None,
        search_text: String::new(),
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
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
        SessionRuntimeStatus::Running | SessionRuntimeStatus::Settling => {
            Some(SidebarSessionStatus::Running)
        }
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

fn presentation_namespace(project_id: &ProjectId, session_id: &SessionId) -> String {
    let project_id = project_id.as_str();
    format!("{}:{project_id}{}", project_id.len(), session_id.as_str())
}

fn build_session_search_documents(
    project_store: &ProjectStore,
) -> HashMap<PathBuf, SessionSearchDocument> {
    let mut documents = HashMap::new();
    for project in project_store.projects() {
        let Ok(catalog) = Session::catalog_in_project(&project.sessions_dir, project.id.as_str())
        else {
            continue;
        };
        for (path, search) in catalog.search {
            documents.insert(
                path,
                session_search_document(search.values, search.searchable),
            );
        }
    }
    documents
}

fn session_search_document(values: Arc<[String]>, searchable: Arc<str>) -> SessionSearchDocument {
    let summary = values
        .iter()
        .find(|value| value.chars().count() >= 4 && !value.starts_with("resp_"))
        .map(|value| truncate_chars(value, 88))
        .unwrap_or_default();
    SessionSearchDocument {
        searchable,
        summary,
        snippets: values,
    }
}

fn load_project_sessions(project_store: &ProjectStore) -> HashMap<PathBuf, Vec<SessionInfo>> {
    let mut sessions = HashMap::new();
    for project in project_store.projects() {
        match Session::catalog_in_project(&project.sessions_dir, project.id.as_str()) {
            Ok(catalog) => {
                sessions.insert(project.sessions_dir.clone(), catalog.sessions);
            }
            Err(_) => {
                sessions.insert(project.sessions_dir.clone(), Vec::new());
            }
        }
    }
    sessions
}

fn build_session_catalog_indices(
    project_store: &ProjectStore,
    project_sessions: &HashMap<PathBuf, Vec<SessionInfo>>,
) -> HashMap<RuntimeKey, usize> {
    let mut indices = HashMap::new();
    for project in project_store.projects() {
        let Some(sessions) = project_sessions.get(&project.sessions_dir) else {
            continue;
        };
        indices.extend(
            sessions
                .iter()
                .enumerate()
                .map(|(index, session)| ((project.id.clone(), session.id.clone()), index)),
        );
    }
    indices
}

fn load_project_archived_sessions(
    project_store: &ProjectStore,
) -> HashMap<PathBuf, Vec<SessionInfo>> {
    let mut sessions = HashMap::new();
    for project in project_store.projects() {
        let directory = project.sessions_dir.join(ARCHIVE_DIRECTORY);
        match Session::catalog_in_project(&directory, project.id.as_str()) {
            Ok(catalog) => {
                sessions.insert(project.sessions_dir.clone(), catalog.sessions);
            }
            Err(_) => {
                sessions.insert(project.sessions_dir.clone(), Vec::new());
            }
        }
    }
    sessions
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
    session.updated_at
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

    fn create_v2_session(
        directory: &Path,
        project_id: &str,
        id: SessionId,
        title: Option<&str>,
    ) -> SessionInfo {
        create_v2_session_with_config(directory, project_id, id, title, SessionConfig::default())
    }

    fn create_v2_session_with_config(
        directory: &Path,
        project_id: &str,
        id: SessionId,
        title: Option<&str>,
        config: SessionConfig,
    ) -> SessionInfo {
        let directory = directory.to_owned();
        let project_id = project_id.to_owned();
        let title = title.map(ToOwned::to_owned);
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async move {
                    let mut session =
                        Session::create_in_project_with_id(directory, project_id, config, id)
                            .await
                            .unwrap();
                    if let Some(title) = title {
                        session.rename(&title).await.unwrap();
                    }
                    session.info().clone()
                })
        })
        .join()
        .unwrap()
    }

    fn text_stream_model(text: &str) -> (Model, std::thread::JoinHandle<()>) {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let text = text.to_owned();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut request = Vec::new();
            let (body_start, content_length) = loop {
                let mut chunk = [0_u8; 4_096];
                let bytes = socket.read(&mut chunk).unwrap();
                assert_ne!(bytes, 0, "client closed before sending request headers");
                request.extend_from_slice(&chunk[..bytes]);
                let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then_some(value.trim())
                    })
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap();
                break (header_end + 4, content_length);
            };
            while request.len() < body_start + content_length {
                let mut chunk = [0_u8; 4_096];
                let bytes = socket.read(&mut chunk).unwrap();
                assert_ne!(bytes, 0, "client closed before sending request body");
                request.extend_from_slice(&chunk[..bytes]);
            }

            let delta = serde_json::json!({
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text.clone(),
            });
            let completed = serde_json::json!({
                "type": "response.completed",
                "sequence_number": 2,
                "response": {
                    "created_at": 0,
                    "id": "resp_external",
                    "model": "test-model",
                    "object": "response",
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "annotations": [],
                            "text": text,
                        }],
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                    }],
                    "status": "completed",
                },
            });
            let body = format!("data: {delta}\n\ndata: {completed}\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).unwrap();
        });
        (
            Model::new(
                "test",
                "key",
                format!("http://{address}"),
                "test-model",
                10_000,
            ),
            server,
        )
    }

    fn commit_external_turn(path: &Path, project_id: &str, input: &str, output: &str) -> u64 {
        let path = path.to_owned();
        let project_id = project_id.to_owned();
        let input = input.to_owned();
        let (model, server) = text_stream_model(output);
        let runner = std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    let session = Session::open_in_project(path, &project_id).await.unwrap();
                    let agent = Agent::new(model, "test", session, ".");
                    let mut active = agent.start(input);
                    while active.next_event().await.is_some() {}
                    active.finish().await.unwrap().session_revision()
                })
        });
        let revision = runner.join().unwrap();
        server.join().unwrap();
        revision
    }

    fn persist_external_config(path: &Path, project_id: &str, config: SessionConfig) {
        let path = path.to_owned();
        let project_id = project_id.to_owned();
        std::thread::spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap()
                .block_on(async move {
                    let session = Session::open_in_project(path, &project_id).await.unwrap();
                    let mut agent = Agent::new(
                        Model::new("test", "key", "http://localhost", "test-model", 10_000),
                        "test",
                        session,
                        ".",
                    );
                    agent.persist_session_config(&config).await.unwrap();
                });
        })
        .join()
        .unwrap();
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
    fn assistant_messages_use_the_semantic_markdown_renderer() {
        let message = message(Role::Assistant, "## Result\n\nbody".into());
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
                let project_id = this
                    .project_store
                    .project(this.core.workspace.active_project)
                    .unwrap()
                    .id
                    .as_str()
                    .to_owned();
                snapshot.session = create_v2_session(
                    &this.core.workspace.sessions_dir,
                    &project_id,
                    snapshot.session.id.clone(),
                    None,
                );
                this.apply_selected_runtime_snapshot(snapshot);
            });
        });

        cx.read_entity(&view, |app, _| {
            assert!(!app.core.session.current.as_os_str().is_empty());
            assert_eq!(
                app.core
                    .session
                    .current
                    .extension()
                    .and_then(|value| value.to_str()),
                Some("session-v2")
            );
            assert!(!app.core.session.current.exists());
            assert!(
                app.core
                    .workspace
                    .sessions_dir
                    .join(kcastle_agent::SESSION_DATABASE_FILE)
                    .is_file()
            );
            assert_eq!(app.core.session.sessions.len(), 1);
            assert_eq!(app.core.session.sessions[0].path, app.core.session.current);
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn archive_and_restore_refresh_both_session_catalogs(cx: &mut gpui::TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-archive-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap();
        let sessions_dir = project.sessions_dir.clone();
        let session = create_v2_session(
            &sessions_dir,
            project.id.as_str(),
            SessionId::from_raw("archive-test"),
            Some("Archive me"),
        );
        let path = session.path;
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

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                this.archive_target_session(active_project, path.clone(), window, cx)
            });
        });
        let archived = cx.read_entity(&view, |app, _| {
            assert!(app.project_sessions[&sessions_dir].is_empty());
            assert_eq!(app.project_archived_sessions[&sessions_dir].len(), 1);
            app.project_archived_sessions[&sessions_dir][0].clone()
        });
        assert_eq!(
            archived.path.parent(),
            Some(sessions_dir.join(ARCHIVE_DIRECTORY).as_path())
        );
        assert!(!archived.path.exists());

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                this.restore_archived_session(active_project, archived.clone(), window, cx)
            });
        });
        cx.read_entity(&view, |app, _| {
            assert_eq!(app.project_sessions[&sessions_dir].len(), 1);
            assert!(app.project_archived_sessions[&sessions_dir].is_empty());
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn stale_session_open_cannot_replace_a_new_chat_draft(cx: &mut gpui::TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-stale-open-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap().clone();
        let stale_session = create_v2_session(
            &project.sessions_dir,
            project.id.as_str(),
            SessionId::from_raw("stale-open"),
            Some("Stale open"),
        );
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
                let session =
                    Session::open_writable_in_project(&stale_session.path, project.id.as_str())
                        .unwrap();
                let stale_runtime = this.create_runtime(active_project, session, cx).unwrap();
                let stale_generation = this.begin_runtime_selection_intent();
                let key = (project.id.clone(), stale_session.path.clone());
                this.inflight_session_opens
                    .insert(key.clone(), stale_generation);

                assert!(this.select_new_chat_draft(cx));

                let completed_generation = this.inflight_session_opens.remove(&key).unwrap();
                if this.session_open_matches_current_intent(completed_generation, &project.id) {
                    this.select_runtime(stale_runtime, cx);
                }
                assert!(
                    this.selected_runtime
                        .read(cx)
                        .observation()
                        .session
                        .path
                        .as_os_str()
                        .is_empty()
                );
                assert!(this.core.session.current.as_os_str().is_empty());
            });
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn pending_session_open_does_not_submit_to_the_previous_runtime(cx: &mut gpui::TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-pending-open-command-gate-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap().clone();
        let target = create_v2_session(
            &project.sessions_dir,
            project.id.as_str(),
            SessionId::from_raw("pending-command-target"),
            Some("Pending command target"),
        );
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

        let loaded_target =
            Session::open_writable_in_project(&target.path, project.id.as_str()).unwrap();
        let (previous_entity, target_entity) = cx.update(|window, app| {
            view.update(app, |this, cx| {
                let previous_entity = this.selected_runtime.entity_id();
                let generation = this.begin_runtime_selection_intent();
                let target_key = (project.id.clone(), target.path.clone());
                assert!(this.start_session_open_request(&target_key, generation));
                assert!(this.selection_pending());
                assert_eq!(this.selected_runtime.entity_id(), previous_entity);

                this.input.update(cx, |input, cx| {
                    input.set_value("must wait for target", window, cx)
                });
                this.submit(window, cx);
                this.set_allow_all_tools(true, cx);

                assert_eq!(this.input.read(cx).value(), "must wait for target");
                let previous = this.selected_runtime.read(cx).snapshot();
                assert_eq!(previous.status, SessionRuntimeStatus::Idle);
                assert!(!previous.allow_all_tools);
                assert!(previous.view.conversation.messages.is_empty());

                let pending = this.pending_runtime_selection.clone().unwrap();
                let key = (project.id.clone(), target.path.clone());
                assert_eq!(
                    this.finish_session_open_request(&key, pending.generation, &project.id,),
                    SessionOpenCompletion::Current
                );
                let runtime = this
                    .reconcile_loaded_runtime(active_project, loaded_target, cx)
                    .unwrap();
                this.select_runtime(runtime, cx);
                assert!(!this.selection_pending());
                assert_ne!(this.selected_runtime.entity_id(), previous_entity);
                assert_eq!(this.core.session.current, target.path);
                assert_eq!(this.input.read(cx).value(), "must wait for target");
                let target_entity = this.selected_runtime.entity_id();

                let missing_path = project
                    .sessions_dir
                    .join("missing-same-project-session.session-v2");
                let generation = this.begin_runtime_selection_intent();
                let missing_key = (project.id.clone(), missing_path.clone());
                assert!(this.start_session_open_request(&missing_key, generation));
                assert!(this.selection_pending());
                let pending = this.pending_runtime_selection.clone().unwrap();
                let key = (project.id.clone(), missing_path.clone());
                assert_eq!(
                    this.finish_session_open_request(&key, pending.generation, &project.id,),
                    SessionOpenCompletion::Current
                );
                assert!(!this.resolve_failed_runtime_selection(
                    pending.generation,
                    &project.id,
                    &missing_path,
                    active_project,
                    cx,
                ));
                assert!(!this.selection_pending());
                assert_eq!(this.selected_runtime.entity_id(), target_entity);
                assert_eq!(this.core.session.current, target.path);
                (previous_entity, target_entity)
            })
        });
        cx.read_entity(&view, |this, _| {
            assert!(!this.selection_pending());
            assert_ne!(target_entity, previous_entity);
            assert_eq!(this.selected_runtime.entity_id(), target_entity);
            assert_eq!(this.core.session.current, target.path);
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn failed_cross_project_open_falls_back_to_the_target_project_draft(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-cross-project-open-failure-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace_a = root.join("workspace-a");
        let workspace_b = root.join("workspace-b");
        std::fs::create_dir_all(&workspace_a).unwrap();
        std::fs::create_dir_all(&workspace_b).unwrap();
        let (mut project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace_a.clone())).unwrap();
        let project_b_index = project_store.add(workspace_b).unwrap();
        let project_b = project_store.project(project_b_index).unwrap().clone();
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), workspace_a);

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

        let (previous_entity, draft_entity) = cx.update(|_window, app| {
            view.update(app, |this, cx| {
                let previous_entity = this.selected_runtime.entity_id();
                let draft = this
                    .select_or_create_project_draft(project_b_index, cx)
                    .expect("target project draft must exist");
                let draft_entity = draft.entity_id();
                let missing_id = SessionId::from_raw("missing-cross-project-session");
                let missing_path = project_b
                    .sessions_dir
                    .join(format!("{missing_id}.session-v2"));
                this.project_sessions
                    .get_mut(&project_b.sessions_dir)
                    .unwrap()
                    .push(SessionInfo {
                        id: missing_id.clone(),
                        project_id: project_b.id.as_str().to_owned(),
                        path: missing_path,
                        title: "Missing session".into(),
                        created_at: 0,
                        updated_at: 0,
                    });
                this.project_runtimes
                    .get_mut(&project_b.id)
                    .unwrap()
                    .selected = missing_id.clone();

                let generation = this.begin_runtime_selection_intent();
                let sessions = this
                    .project_sessions
                    .get(&project_b.sessions_dir)
                    .cloned()
                    .unwrap_or_default();
                let _ = this.transition(Action::ActivateWorkspace {
                    index: project_b_index,
                    cwd: project_b.path.clone(),
                    sessions_dir: project_b.sessions_dir.clone(),
                    sessions,
                });
                let key = (
                    project_b.id.clone(),
                    this.project_sessions[&project_b.sessions_dir]
                        .iter()
                        .find(|session| session.id == missing_id)
                        .unwrap()
                        .path
                        .clone(),
                );
                assert!(this.start_session_open_request(&key, generation));
                assert!(this.selection_pending());
                assert_eq!(this.core.workspace.active_project, project_b_index);
                assert_eq!(this.selected_runtime.entity_id(), previous_entity);
                let pending = this.pending_runtime_selection.clone().unwrap();
                let key = (project_b.id.clone(), pending.path.clone());
                assert_eq!(
                    this.finish_session_open_request(&key, pending.generation, &project_b.id,),
                    SessionOpenCompletion::Current
                );
                assert!(this.resolve_failed_runtime_selection(
                    pending.generation,
                    &project_b.id,
                    &pending.path,
                    project_b_index,
                    cx,
                ));
                (previous_entity, draft_entity)
            })
        });
        cx.read_entity(&view, |this, cx| {
            assert!(!this.selection_pending());
            assert_ne!(this.selected_runtime.entity_id(), previous_entity);
            assert_eq!(this.selected_runtime.entity_id(), draft_entity);
            let (selected_project, _) = this
                .runtime_location(&this.selected_runtime)
                .expect("fallback draft must be registered");
            assert_eq!(selected_project, project_b.id);
            assert_eq!(this.core.workspace.active_project, project_b_index);
            assert!(this.core.session.current.as_os_str().is_empty());
            assert!(
                this.selected_runtime
                    .read(cx)
                    .snapshot()
                    .session
                    .path
                    .as_os_str()
                    .is_empty()
            );
        });

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn reopening_cached_runtime_reloads_external_revision_and_config_drift(
        cx: &mut gpui::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-reopen-cache-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap().clone();
        let initial_config = SessionConfig {
            model_id: Some("test/test-model".into()),
            reasoning_effort: None,
            allow_all_tools: false,
        };
        let session_info = create_v2_session_with_config(
            &project.sessions_dir,
            project.id.as_str(),
            SessionId::from_raw("external-reopen"),
            Some("Stable title"),
            initial_config.clone(),
        );
        let settings = SettingsStore::load(root.join("settings")).unwrap();
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000);
        let profile = ProviderModel::new("test-model", "Test Model", 10_000, None);
        let configured = ConfiguredModel::new("test", profile, model.clone());
        let agent = Agent::new(model, "test", Session::memory(), &workspace);

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

        let stale_entity = cx.update(|_, app| {
            view.update(app, |this, cx| {
                let session =
                    Session::open_writable_in_project(&session_info.path, project.id.as_str())
                        .unwrap();
                let runtime = this.create_runtime(active_project, session, cx).unwrap();
                assert_eq!(runtime.read(cx).observation().durable_revision, 0);
                assert!(
                    runtime
                        .read(cx)
                        .snapshot()
                        .view
                        .conversation
                        .messages
                        .is_empty()
                );
                runtime.entity_id()
            })
        });

        // Hold the first loader's snapshot at a deterministic gate. A later request for the same
        // key must not promote this snapshot after the store advances.
        let stale_loaded =
            Session::open_writable_in_project(&session_info.path, project.id.as_str()).unwrap();
        assert_eq!(stale_loaded.revision(), 0);
        let open_key = (project.id.clone(), session_info.path.clone());
        let first_generation = cx.update(|_, app| {
            view.update(app, |this, _| {
                let generation = this.begin_runtime_selection_intent();
                this.begin_pending_runtime_selection(
                    generation,
                    project.id.clone(),
                    session_info.path.clone(),
                );
                this.inflight_session_opens
                    .insert(open_key.clone(), generation);
                generation
            })
        });

        let external_revision = commit_external_turn(
            &session_info.path,
            project.id.as_str(),
            "external input",
            "external output",
        );
        assert!(external_revision > 0);
        let reload_generation = cx.update(|_, app| {
            view.update(app, |this, _| {
                let generation = this.begin_runtime_selection_intent();
                this.begin_pending_runtime_selection(
                    generation,
                    project.id.clone(),
                    session_info.path.clone(),
                );
                *this.inflight_session_opens.get_mut(&open_key).unwrap() = generation;
                assert_eq!(
                    this.finish_session_open_request(&open_key, first_generation, &project.id,),
                    SessionOpenCompletion::Reload(generation),
                    "the newer same-key intent must force a new SQLite snapshot"
                );
                assert!(!this.inflight_session_opens.contains_key(&open_key));
                assert_eq!(
                    this.pending_runtime_selection,
                    Some(PendingRuntimeSelection {
                        generation,
                        project_id: project.id.clone(),
                        path: session_info.path.clone(),
                    }),
                    "the superseding selection gate must survive until the fresh load completes"
                );
                generation
            })
        });
        drop(stale_loaded);

        let loaded_after_commit =
            Session::open_writable_in_project(&session_info.path, project.id.as_str()).unwrap();
        assert_eq!(loaded_after_commit.revision(), external_revision);
        let reloaded_entity = cx.update(|_, app| {
            view.update(app, |this, cx| {
                this.inflight_session_opens
                    .insert(open_key.clone(), reload_generation);
                assert_eq!(
                    this.finish_session_open_request(&open_key, reload_generation, &project.id,),
                    SessionOpenCompletion::Current
                );
                let runtime = this
                    .reconcile_loaded_runtime(active_project, loaded_after_commit, cx)
                    .unwrap();
                assert!(
                    runtime
                        .read(cx)
                        .snapshot()
                        .view
                        .conversation
                        .messages
                        .iter()
                        .any(|message| message.text.contains("external input"))
                );
                this.select_runtime(runtime.clone(), cx);
                assert!(!this.selection_pending());
                runtime.entity_id()
            })
        });
        assert_ne!(reloaded_entity, stale_entity);

        let changed_config = SessionConfig {
            allow_all_tools: true,
            ..initial_config
        };
        persist_external_config(
            &session_info.path,
            project.id.as_str(),
            changed_config.clone(),
        );
        let loaded_after_config =
            Session::open_writable_in_project(&session_info.path, project.id.as_str()).unwrap();
        assert_eq!(loaded_after_config.revision(), external_revision);
        assert_eq!(loaded_after_config.config(), &changed_config);
        let config_reloaded_entity = cx.update(|_, app| {
            view.update(app, |this, cx| {
                let runtime = this
                    .reconcile_loaded_runtime(active_project, loaded_after_config, cx)
                    .unwrap();
                assert_eq!(runtime.read(cx).snapshot().config, changed_config);
                this.select_runtime(runtime.clone(), cx);
                runtime.entity_id()
            })
        });
        assert_ne!(config_reloaded_entity, reloaded_entity);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui::test]
    fn background_failed_runtimes_share_the_terminal_cache_bound(cx: &mut gpui::TestAppContext) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-settled-runtime-cache-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap().clone();
        let sessions = (0..MAX_CACHED_TERMINAL_RUNTIMES + 2)
            .map(|index| {
                create_v2_session(
                    &project.sessions_dir,
                    project.id.as_str(),
                    SessionId::from_raw(format!("settled-cache-{index}")),
                    None,
                )
            })
            .collect::<Vec<_>>();
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

        let mut evicted_entities = Vec::new();
        cx.update(|_, app| {
            view.update(app, |this, cx| {
                let draft_id = this.selected_runtime.read(cx).observation().session.id;
                let mut runtimes = Vec::new();
                for session_info in &sessions {
                    let session =
                        Session::open_writable_in_project(&session_info.path, project.id.as_str())
                            .unwrap();
                    let runtime = this.create_runtime(active_project, session, cx).unwrap();
                    runtime.update(cx, |runtime, _| {
                        runtime
                            .mark_failed_for_test(format!("failed-{}", session_info.id.as_str()));
                    });
                    runtimes.push((session_info.id.clone(), runtime));
                }

                // Recreate the app-layer observation immediately before many background runtimes
                // become terminal failures. Prior observations deliberately record their last
                // visible state as active so one synchronization exercises the transition-triggered
                // eviction path, not just registration-time eviction.
                for (index, (session_id, runtime)) in runtimes.iter().enumerate() {
                    let key = (project.id.clone(), session_id.clone());
                    let subscription = cx.observe(runtime, |this, runtime, cx| {
                        this.sync_runtime_snapshot(&runtime, cx);
                    });
                    this.project_runtimes
                        .get_mut(&project.id)
                        .unwrap()
                        .sessions
                        .insert(session_id.clone(), runtime.clone());
                    this.runtime_subscriptions.insert(key.clone(), subscription);
                    this.runtime_recency.insert(key.clone(), index as u64 + 1);
                    let observation = runtime.read(cx).observation();
                    this.runtime_observations.insert(
                        key,
                        RuntimeObservation {
                            completed_runs: observation.completed_runs,
                            transcript_updates: observation.transcript_updates,
                            presentation_sequence: observation.presentation_sequence,
                            catalog_synced_revision: observation.durable_revision,
                            metadata_generation: observation.metadata_generation,
                            is_terminal: false,
                        },
                    );
                }
                // A remembered per-project selection is an identity hint, not a cache pin. Make
                // the oldest full runtime the remembered selection and prove it is still evicted.
                this.project_runtimes.get_mut(&project.id).unwrap().selected =
                    runtimes[0].0.clone();

                assert_eq!(
                    runtimes
                        .iter()
                        .filter(|(session_id, _)| this.project_runtimes[&project.id]
                            .sessions
                            .contains_key(session_id))
                        .count(),
                    MAX_CACHED_TERMINAL_RUNTIMES + 2
                );
                evicted_entities.extend(
                    runtimes
                        .iter()
                        .take(2)
                        .map(|(_, runtime)| runtime.downgrade()),
                );
                this.sync_runtime_snapshot(&runtimes.last().unwrap().1, cx);

                let retained = runtimes
                    .iter()
                    .filter(|(session_id, _)| {
                        this.project_runtimes[&project.id]
                            .sessions
                            .contains_key(session_id)
                    })
                    .count();
                assert_eq!(retained, MAX_CACHED_TERMINAL_RUNTIMES);
                for (session_id, _) in runtimes.iter().take(2) {
                    let key = (project.id.clone(), session_id.clone());
                    assert!(
                        !this.project_runtimes[&project.id]
                            .sessions
                            .contains_key(session_id)
                    );
                    assert!(!this.runtime_subscriptions.contains_key(&key));
                    assert!(!this.runtime_recency.contains_key(&key));
                    assert!(!this.runtime_observations.contains_key(&key));
                }
                assert_ne!(
                    this.project_runtimes[&project.id].selected, draft_id,
                    "the test must exercise a remembered persisted selection"
                );
                drop(runtimes);
            });
        });
        cx.run_until_parked();
        assert!(
            evicted_entities
                .iter()
                .all(|runtime| runtime.upgrade().is_none()),
            "eviction must release the runtime entity and its owned document/view"
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
