use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpui_kit::component::input::{InputEvent, InputState, TextareaState};
use gpui_kit::component::{Theme, ThemeMode};
use gpui_kit::{
    AppContext, Bounds, Context, Entity, FocusHandle, ListAlignment, ListOffset, ListState,
    PathPromptOptions, Pixels, Point, ScrollHandle, ScrollWheelEvent, Subscription, Window, point,
    px,
};
use kcastle_agent::{Agent, Session, SessionConfig, SessionError, SessionId, SessionInfo};
#[cfg(test)]
use kcastle_agent::{Model, SessionCatalog, SessionModelConfig, SessionStoreError};

use crate::agent_config::ConfiguredModel;
use crate::application::session_catalog::{
    SessionCatalogCache, SessionSearchDocument, apply_project_catalog_result,
    load_project_archived_sessions, load_session_catalog_cache, matching_search_snippet,
    remove_project_catalog_members, remove_session_catalog_entry, should_clear_catalog_after_error,
};
#[cfg(test)]
use crate::application::session_catalog::{
    clear_project_catalog_cache, load_session_catalog_cache_with, session_search_document,
    truncate_chars,
};
use crate::dialogs::Modal;
use crate::domain::session_document::SessionDocument;
use crate::domain::timeline::{AxisId, AxisRange, DomainRange};
use crate::domain::{
    Action, AppState, ComposerMenu, DetailsSelection, DetailsTab, Effect, LayoutGeneration,
    Message, Role, RunState, ScrollIntent, Surface, TimelineMode, TrajectoryItemId,
    TrajectoryRequestKey, next_message_id, reduce,
};
use crate::layout::{LayoutInput, ScrollAnchor, ScrollRestore, resolve_scroll_restore};
use crate::platform::NativeTitlebarController;
use crate::platform::gpui::{
    DeferredScrollAlignment, GpuiLayoutRuntime, MeasuredBounds, MessagePresentationStore,
    SessionRuntime, SessionRuntimeSnapshot, SessionRuntimeStatus, run_effects,
};
use crate::project::{ProjectId, ProjectStore};
#[cfg(test)]
use crate::settings::ProviderModel;
use crate::settings::{Appearance, EnterBehavior, SettingsStore};
use crate::trajectory::{
    TimelineModelCache, TrajectoryDetailsLayoutState, TrajectoryDetailsMarkdownCache,
};
use crate::updater::AvailableUpdate;

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
    surface: Surface,
    chat_anchor: ScrollAnchor,
    trajectory_offset: Option<ListOffset>,
    trajectory_follow_tail: bool,
    trajectory_query: String,
    details_offset: Point<Pixels>,
    selected_details: Option<DetailsSelection>,
    details_tab_history: Vec<DetailsTab>,
    collapsed_turns: HashSet<u32>,
    collapsed_assistants: HashSet<TrajectoryItemId>,
    timeline_selection: Option<SavedTimelineRange>,
    timeline_viewport: Option<SavedTimelineRange>,
}

/// A timeline range saved by meaning rather than by one in-memory projection identity.
///
/// `AxisRange` deliberately carries a projection lineage so a stale live range cannot be applied
/// to another document. Session view state outlives the bounded runtime/document cache, though, so
/// persisting that lineage would also reject a legitimate range when the same session is replayed
/// into a fresh projection. Saving the mode plus domain coordinates lets restore issue a new,
/// correctly stamped range for the current canonical projection.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SavedTimelineRange {
    mode: TimelineMode,
    range: DomainRange,
}

impl SavedTimelineRange {
    fn capture(range: AxisRange) -> Self {
        Self {
            mode: range.axis.mode,
            range: range.range,
        }
    }

    fn restore(
        self,
        mode: TimelineMode,
        document_generation: u64,
        geometry_revision: u64,
    ) -> Option<AxisRange> {
        (self.mode == mode).then_some(AxisRange {
            axis: AxisId {
                document_generation,
                geometry_revision,
                mode,
            },
            range: self.range,
        })
    }
}

impl Default for SessionViewState {
    fn default() -> Self {
        Self {
            surface: Surface::Chat,
            chat_anchor: ScrollAnchor::Tail,
            trajectory_offset: None,
            trajectory_follow_tail: true,
            trajectory_query: String::new(),
            details_offset: point(px(0.0), px(0.0)),
            selected_details: None,
            details_tab_history: vec![DetailsTab::Summary],
            collapsed_turns: HashSet::new(),
            collapsed_assistants: HashSet::new(),
            timeline_selection: None,
            timeline_viewport: None,
        }
    }
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

fn invalid_session_open_error(error: &SessionError) -> bool {
    should_clear_catalog_after_error(error)
}

fn session_open_error_notice(error: &SessionError) -> Option<String> {
    (!invalid_session_open_error(error)).then(|| format!("Could not open session: {error}"))
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
    catalog_synced_revision: u64,
    metadata_generation: u64,
    is_terminal: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct TimelineDragState {
    pub(crate) pan: bool,
    pub(crate) start_value: f64,
    pub(crate) current_value: f64,
    pub(crate) start_x: f32,
    pub(crate) record_id: Option<TrajectoryItemId>,
    pub(crate) initial_viewport: AxisRange,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimelineHoverState {
    pub(crate) axis: AxisId,
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
    pub(crate) message_presentations: RefCell<MessagePresentationStore>,
    pub(crate) selected_runtime: Entity<SessionRuntime>,
    project_runtimes: HashMap<ProjectId, ProjectSessionRuntimes>,
    pub(crate) input: Entity<TextareaState>,
    pub(crate) session_search: Entity<InputState>,
    pub(crate) trajectory_search: Entity<InputState>,
    trajectory_query_value: String,
    pub(crate) modal: Option<Modal>,
    pub(crate) modal_focus: FocusHandle,
    pub(crate) composer_menu_focus: FocusHandle,
    pub(crate) scroll: ScrollHandle,
    chat_tail_alignment: DeferredScrollAlignment,
    pub(crate) trajectory_scroll: ListState,
    pub(crate) trajectory_scroll_restore: Cell<Option<ListOffset>>,
    pub(crate) trajectory_follow_tail: Cell<bool>,
    pending_trajectory_query_restore: RefCell<Option<String>>,
    pub(crate) trajectory_list_structure: Cell<Option<(u64, u64, u64, bool)>>,
    pub(crate) details_scroll: ScrollHandle,
    pub(crate) timeline_bounds: Option<Bounds<Pixels>>,
    pub(crate) trajectory_ledger_width: Option<(LayoutGeneration, Pixels)>,
    pub(crate) timeline_drag: Option<TimelineDragState>,
    pub(crate) timeline_hover: Option<TimelineHoverState>,
    pub(crate) request_marker_hover: Option<TrajectoryRequestKey>,
    pub(crate) timeline_model_cache: RefCell<Option<TimelineModelCache>>,
    pub(crate) trajectory_details_layout: TrajectoryDetailsLayoutState,
    pub(crate) trajectory_details_markdown: RefCell<TrajectoryDetailsMarkdownCache>,
    pub(crate) models: Vec<ConfiguredModel>,
    pub(crate) selected_model: usize,
    pub(crate) selected_reasoning_effort: Option<kcastle_agent::ReasoningEffort>,
    pub(crate) model: String,
    pub(crate) project_store: ProjectStore,
    pub(crate) settings: SettingsStore,
    pub(crate) selected_started_at: Option<Instant>,
    pub(crate) project_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
    pub(crate) project_archived_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
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
    view_states: HashMap<RuntimeKey, SessionViewState>,
    _subscriptions: Vec<Subscription>,
}

impl DesktopApp {
    #[allow(
        clippy::expect_used,
        reason = "DesktopStartup contains the ProjectStore-validated active project"
    )]
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
            .model
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
        let SessionCatalogCache {
            project_sessions,
            session_search_documents,
            session_catalog_indices,
        } = load_session_catalog_cache(&project_store);
        let project_archived_sessions = load_project_archived_sessions(&project_store);
        let viewport = window.viewport_size();
        let mut core = AppState::new(LayoutInput {
            viewport_width: f32::from(viewport.width),
            viewport_height: f32::from(viewport.height),
            rem_size: f32::from(window.rem_size()),
            ..LayoutInput::default()
        });
        core.trajectory.mode = if settings.trajectory_actual_duration() {
            TimelineMode::Duration
        } else {
            TimelineMode::Sequence
        };
        core.workspace.cwd = project.path.clone();
        core.workspace.active_project = active_project;
        core.workspace.expanded_projects = HashSet::from([project.path.clone()]);
        core.workspace.sessions_dir = project.sessions_dir.clone();
        core.session.current = current_session.clone();
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
            |this, search, event: &InputEvent, _, cx| {
                if matches!(event, InputEvent::Change) {
                    this.trajectory_query_value = search.read(cx).value().to_string();
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
        // Keep short trajectories anchored below the overview like DSH. Tail following is an
        // independent scroll policy and still moves overflowing ledgers to their newest item.
        let trajectory_scroll = ListState::new(0, ListAlignment::Top, px(1_000.0));
        let trajectory_follow_tail = Cell::new(true);
        let trajectory_scroll_owner = cx.entity().downgrade();
        trajectory_scroll.set_scroll_handler({
            move |event, _, cx| {
                let owner = trajectory_scroll_owner.clone();
                let (is_scrolled, visible_end, count) =
                    (event.is_scrolled, event.visible_range.end, event.count);
                // GPUI holds ListState's mutable borrow throughout this callback.
                cx.defer(move |cx| {
                    let _ = owner.update(cx, |this, _| {
                        let follows = if !is_scrolled {
                            true
                        } else if count > 0 && visible_end == count {
                            this.trajectory_scroll
                                .bounds_for_item(count - 1)
                                .is_some_and(|last| {
                                    let remaining = (last.bottom()
                                        - this.trajectory_scroll.viewport_bounds().bottom())
                                    .max(px(0.0));
                                    remaining <= px(2.0)
                                })
                        } else {
                            false
                        };
                        this.trajectory_follow_tail.set(follows);
                    });
                });
            }
        });
        let app = Self {
            core,
            layout_runtime: GpuiLayoutRuntime::default(),
            message_presentations: RefCell::new(MessagePresentationStore::default()),
            selected_runtime: runtime,
            project_runtimes,
            input,
            session_search,
            trajectory_search,
            trajectory_query_value: String::new(),
            modal: None,
            modal_focus: cx.focus_handle(),
            composer_menu_focus: cx.focus_handle(),
            scroll: ScrollHandle::new(),
            chat_tail_alignment: DeferredScrollAlignment::default(),
            trajectory_scroll,
            trajectory_scroll_restore: Cell::new(None),
            trajectory_follow_tail,
            pending_trajectory_query_restore: RefCell::new(None),
            trajectory_list_structure: Cell::new(None),
            details_scroll: ScrollHandle::new(),
            timeline_bounds: None,
            trajectory_ledger_width: None,
            timeline_drag: None,
            timeline_hover: None,
            request_marker_hover: None,
            timeline_model_cache: RefCell::new(None),
            trajectory_details_layout: TrajectoryDetailsLayoutState::default(),
            trajectory_details_markdown: RefCell::new(TrajectoryDetailsMarkdownCache::default()),
            models,
            selected_model,
            selected_reasoning_effort,
            model,
            project_store,
            settings,
            selected_started_at: None,
            project_sessions,
            project_archived_sessions,
            session_search_documents,
            session_catalog_indices,
            available_update: None,
            unread_sessions: HashSet::new(),
            runtime_observations: HashMap::new(),
            runtime_subscriptions: HashMap::from([(
                initial_runtime_key.clone(),
                runtime_subscription,
            )]),
            runtime_recency: HashMap::from([(initial_runtime_key.clone(), 1)]),
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
        app.message_presentations
            .borrow_mut()
            .activate(presentation_namespace(
                &initial_runtime_key.0,
                &initial_runtime_key.1,
            ));
        #[cfg(not(test))]
        app.check_for_updates(window, cx);
        app
    }

    fn sync_runtime_snapshot(&mut self, runtime: &Entity<SessionRuntime>, cx: &mut Context<Self>) {
        let observation = runtime.read(cx).observation();
        let location = self.runtime_location(runtime);
        let selected = runtime.entity_id() == self.selected_runtime.entity_id();
        let mut selected_transcript_updates = 0;
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
        if location.is_none() && !observation.session.path.as_os_str().is_empty() {
            let project = self
                .project_store
                .projects()
                .iter()
                .find(|project| project.id.as_str() == observation.session.project_id)
                .cloned();
            if let Some(project) = project
                && !self
                    .project_sessions
                    .get(&project.sessions_dir)
                    .is_some_and(|sessions| {
                        sessions
                            .iter()
                            .any(|info| info.id == observation.session.id)
                    })
            {
                self.refresh_project_catalog(&project.id);
            }
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
        if let Some(model_id) = snapshot.config.model.model_id.as_deref()
            && let Some(index) = self.models.iter().position(|model| model.id == model_id)
            && self.models[index].model.has_api_key()
        {
            self.selected_model = index;
            self.model = self.models[index].label();
        }
        self.selected_reasoning_effort = snapshot
            .config
            .model
            .reasoning_effort
            .as_deref()
            .and_then(parse_reasoning_effort);
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
            SessionRuntimeStatus::Failed(failure) => RunState::Failed { failure },
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
                    .project_store
                    .project(self.core.workspace.active_project)
                    .and_then(|project| self.project_sessions.get(&project.sessions_dir))
                    .is_some_and(|sessions| {
                        sessions.iter().any(|session| session.id == session_id)
                    });
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
        let config = session.config().clone();
        let model_index = active_model_index(&self.models, config.model.model_id.as_deref())
            .unwrap_or(self.selected_model);
        let configured = &self.models[model_index];
        let agent = Agent::new(
            configured.model.clone(),
            crate::INSTRUCTIONS,
            session,
            project.path.clone(),
        );
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
        let changing_session = runtime.entity_id() != self.selected_runtime.entity_id();
        if changing_session {
            // A session load is asynchronous, so the user may keep changing search, tabs, tail
            // following, or scroll positions after the click that started it. Capture once more
            // at the atomic handoff; the earlier eager capture remains useful for a failed load,
            // while this one guarantees a successful load cannot drop those later edits.
            self.save_current_view_state();
        }
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
        self.request_marker_hover = None;
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
            previous.completed_runs = observation.completed_runs;
            previous.transcript_updates = observation.transcript_updates;
            previous.metadata_generation = observation.metadata_generation;
            self.touch_runtime(key);
        }
        if changing_session {
            let surface = location
                .as_ref()
                .and_then(|key| self.view_states.get(key))
                .map(|state| state.surface)
                .unwrap_or_default();
            let _ = self.transition(match surface {
                Surface::Chat => Action::ShowChat,
                Surface::Trajectory => Action::ShowTrajectory,
            });
        }
        self.restore_current_view_state(cx);
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
    }

    fn sync_message_presentations(&mut self) {
        let namespace = self
            .runtime_location(&self.selected_runtime)
            .map(|(project_id, session_id)| presentation_namespace(&project_id, &session_id))
            .unwrap_or_else(|| "unregistered-session".to_owned());
        self.message_presentations.get_mut().activate(namespace);
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

    fn view_state_key(&self) -> Option<RuntimeKey> {
        // Derive the identity from the selected runtime, not the mutable workspace/session
        // projection in `core`. During a cross-project async open the workspace already names the
        // target while the selected runtime still owns the source session; combining those fields
        // would save a chimeric key and lose every edit made while loading.
        self.runtime_location(&self.selected_runtime)
    }

    fn save_current_view_state(&mut self) {
        let Some(key) = self.view_state_key() else {
            return;
        };
        let state = self.view_states.entry(key).or_default();
        state.surface = self.core.surface;
        if self.core.surface == Surface::Trajectory {
            state.trajectory_follow_tail = self.trajectory_follow_tail.get();
            state.trajectory_offset = (!state.trajectory_follow_tail).then(|| {
                self.trajectory_scroll_restore
                    .get()
                    .unwrap_or_else(|| self.trajectory_scroll.logical_scroll_top())
            });
            state
                .trajectory_query
                .clone_from(&self.trajectory_query_value);
            state.selected_details = self.core.details.selected.clone();
            state
                .details_tab_history
                .clone_from(&self.core.details.tab_history);
            state.details_offset = self.details_scroll.offset();
            state.collapsed_turns = self.core.trajectory.collapsed_turns.clone();
            state.collapsed_assistants = self.core.trajectory.collapsed_assistants.clone();
            state.timeline_selection = self
                .core
                .trajectory
                .selected_range
                .map(SavedTimelineRange::capture);
            state.timeline_viewport = self
                .core
                .trajectory
                .visible_range
                .map(SavedTimelineRange::capture);
        } else {
            state.chat_anchor = self
                .layout_runtime
                .capture_chat_anchor(self.core.layout_generation, self.core.follow_chat_tail);
        }
    }

    fn restore_current_view_state(&mut self, _cx: &mut Context<Self>) {
        let state = self
            .view_state_key()
            .and_then(|key| self.view_states.get(&key))
            .cloned()
            .unwrap_or_default();
        if self.core.surface == Surface::Trajectory {
            let _ = self.transition(Action::SetTrajectoryTurnsCollapsed(
                state.collapsed_turns.clone(),
            ));
            let _ = self.transition(Action::SetTrajectoryAssistantsCollapsed(
                state.collapsed_assistants.clone(),
            ));
            let projection = &self.core.session_view.trajectory;
            let mode = self.core.trajectory.mode;
            let lineage = projection.projection_lineage();
            let revision = projection.revision();
            let selection = state
                .timeline_selection
                .and_then(|range| range.restore(mode, lineage, revision));
            let viewport = state
                .timeline_viewport
                .and_then(|range| range.restore(mode, lineage, revision));
            let _ = self.transition(Action::SetTimelineSelection(selection));
            let _ = self.transition(Action::SetTimelineViewport(viewport));
            self.timeline_drag = None;
            self.timeline_hover = None;
            self.request_marker_hover = None;
            self.trajectory_follow_tail
                .set(state.trajectory_follow_tail);
            *self.pending_trajectory_query_restore.borrow_mut() =
                Some(state.trajectory_query.clone());
            self.trajectory_query_value = state.trajectory_query.clone();
            // The new session's row count is not known until its ledger projection is rendered.
            // Applying the offset before then would clamp it against the previous session's list.
            self.trajectory_scroll_restore.set(state.trajectory_offset);
            self.trajectory_list_structure.set(None);
            let selected = state.selected_details.filter(|selected| {
                let trajectory = &self.core.session_view.trajectory;
                match selected {
                    DetailsSelection::Record(id) => trajectory.record_index(id).is_some(),
                    DetailsSelection::Request(key) => trajectory.request_index(key).is_some(),
                }
            });
            let _ = self.transition(Action::RestoreSessionView {
                selected: selected.clone(),
                details_tab_history: state.details_tab_history.clone(),
                follow_chat_tail: matches!(state.chat_anchor, ScrollAnchor::Tail),
            });
            self.details_scroll.set_offset(state.details_offset);
        } else {
            let _ = self.transition(Action::RestoreSessionView {
                selected: None,
                details_tab_history: state.details_tab_history,
                follow_chat_tail: matches!(state.chat_anchor, ScrollAnchor::Tail),
            });
            self.layout_runtime.pending_chat_anchor =
                Some((self.core.layout_generation, state.chat_anchor));
            self.layout_runtime.restore_scheduled = false;
        }
    }

    pub(crate) fn apply_pending_trajectory_query_restore(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(query) = self.pending_trajectory_query_restore.borrow_mut().take() else {
            return;
        };
        if self.trajectory_search.read(cx).value().as_ref() != query.as_str() {
            self.trajectory_search
                .update(cx, |search, cx| search.set_value(query, window, cx));
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
            .get_mut()
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
                .find(|(_, record)| {
                    matches!(
                        &record.id,
                        TrajectoryItemId::Tool(record_call_id)
                            if record_call_id.as_str() == call_id
                    )
                })
            else {
                return;
            };
            let message_id = record.id.clone();
            let mut effects = self.transition(Action::ShowTrajectory);
            effects.extend(
                self.transition(Action::SelectDetails(Some(DetailsSelection::Record(
                    message_id,
                )))),
            );
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
            .get_mut()
            .rate(message_id, positive)
            .is_some()
        {
            cx.notify();
        }
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
        event: &gpui_kit::KeyDownEvent,
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
                                                if let Some(project) =
                                                    this.project_store.project(index).cloned()
                                                {
                                                    this.project_sessions
                                                        .entry(project.sessions_dir.clone())
                                                        .or_default();
                                                    this.project_archived_sessions
                                                        .entry(project.sessions_dir.clone())
                                                        .or_default();
                                                    this.reload_archived_sessions(index);
                                                    if index == this.core.workspace.active_project {
                                                        this.refresh_project_catalog(&project.id);
                                                    } else {
                                                        this.switch_project(index, window, cx);
                                                    }
                                                }
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
        self.dispatch(
            Action::ActivateWorkspace {
                index,
                cwd: project.path.clone(),
                sessions_dir: project.sessions_dir.clone(),
            },
            window,
            cx,
        );
        self.refresh_project_catalog(&project.id);
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
                .get_mut()
                .remove_session(&presentation_namespace(&project.id, &session_id));
        }
        self.inflight_session_opens
            .retain(|(project_id, _), _| project_id != &project.id);
        self.unread_sessions
            .retain(|(project_id, _)| project_id != &project.id);
        remove_project_catalog_members(
            &project.id,
            &project.sessions_dir,
            &self.project_sessions,
            &mut self.session_search_documents,
            &mut self.session_catalog_indices,
        );
        self.project_sessions.remove(&project.sessions_dir);
        self.project_archived_sessions.remove(&project.sessions_dir);
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
                                            this.discard_invalid_session_catalog_entry(
                                                project_index,
                                                &key.1,
                                            );
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
                                            this.input
                                                .update(cx, |input, cx| input.focus(window, cx));
                                        }
                                    } else if runtime.is_none() {
                                        this.discard_invalid_session_catalog_entry(
                                            project_index,
                                            &key.1,
                                        );
                                    }
                                }
                            }
                            Err(error) => {
                                let invalid = invalid_session_open_error(&error);
                                if let Some(project_index) = resolved_project_index {
                                    if invalid {
                                        this.discard_invalid_session_catalog_entry(
                                            project_index,
                                            &key.1,
                                        );
                                    }
                                    if is_current_intent {
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
                                }
                                if is_current_intent {
                                    if let Some(message) = session_open_error_notice(&error) {
                                        this.notice(message);
                                    }
                                    this.input.update(cx, |input, cx| input.focus(window, cx));
                                }
                            }
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
        cx: &mut Context<Self>,
    ) {
        match Session::restore(&session) {
            Ok(_) => {
                self.reload_archived_sessions(project_index);
                self.reload_project_session_list(project_index);
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
            self.message_presentations
                .get_mut()
                .remove_session(&namespace);
            self.unread_sessions.remove(&key);
        }
        self.reload_project_session_list(project_index);
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

    fn reload_project_session_list(&mut self, project_index: usize) {
        let project_id = self
            .project_store
            .project(project_index)
            .map(|project| project.id.clone());
        if let Some(project_id) = project_id {
            self.refresh_project_catalog(&project_id);
        }
    }

    fn discard_invalid_session_catalog_entry(&mut self, project_index: usize, path: &Path) {
        let Some(project) = self.project_store.project(project_index).cloned() else {
            return;
        };
        if let Some(session_id) = remove_session_catalog_entry(
            &project.id,
            &project.sessions_dir,
            path,
            &mut self.project_sessions,
            &mut self.session_search_documents,
            &mut self.session_catalog_indices,
        ) {
            self.unread_sessions.remove(&(project.id, session_id));
        }
    }

    fn reload_archived_sessions(&mut self, project_index: usize) {
        let Some(project) = self.project_store.project(project_index) else {
            return;
        };
        let sessions_dir = project.sessions_dir.clone();
        match Session::archived_catalog_in_project(&sessions_dir, project.id.as_str()) {
            Ok(catalog) => {
                self.project_archived_sessions
                    .insert(sessions_dir.clone(), catalog.sessions);
            }
            Err(error) if should_clear_catalog_after_error(&error) => {
                self.project_archived_sessions
                    .insert(sessions_dir, Vec::new());
            }
            Err(_) => {}
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
        self.selected_runtime
            .update(cx, |runtime, cx| runtime.select_model(&configured, cx));
        self.selected_model = index;
        self.model = label;
        self.selected_reasoning_effort = configured.reasoning_effort;
        self.dispatch_local(Action::SetComposerMenu(None), cx);
    }

    pub(crate) fn refresh_idle_runtime_models(&mut self, cx: &mut Context<Self>) {
        let updates = self
            .project_runtimes
            .values()
            .flat_map(|project| project.sessions.values())
            .filter_map(|runtime| {
                let snapshot = runtime.read(cx).snapshot();
                let model_id = snapshot.config.model.model_id?;
                let configured = self.models.iter().find(|model| model.id == model_id)?;
                Some((runtime.clone(), configured.clone()))
            })
            .collect::<Vec<_>>();
        for (runtime, configured) in updates {
            runtime.update(cx, |runtime, cx| {
                runtime.refresh_model(&configured, cx);
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

    pub(crate) fn set_trajectory_actual_duration(
        &mut self,
        enabled: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if let Err(error) = self.settings.set_trajectory_actual_duration(enabled) {
            self.notice(format!(
                "Could not save trajectory duration preference: {error}"
            ));
        }
        for state in self.view_states.values_mut() {
            state.timeline_selection = None;
            state.timeline_viewport = None;
        }
        self.dispatch(
            Action::SetTimelineMode(if enabled {
                TimelineMode::Duration
            } else {
                TimelineMode::Sequence
            }),
            window,
            cx,
        );
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
        session.updated_at
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
        apply_project_catalog_result(
            &project.id,
            &project.sessions_dir,
            Session::catalog_in_project(&project.sessions_dir, project.id.as_str()),
            &mut self.project_sessions,
            &mut self.session_search_documents,
            &mut self.session_catalog_indices,
        )
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
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn parse_reasoning_effort(value: &str) -> Option<kcastle_agent::ReasoningEffort> {
    serde_json::from_value(serde_json::Value::String(value.to_owned())).ok()
}

fn config_for_model(model: &ConfiguredModel, allow_all_tools: bool) -> SessionConfig {
    SessionConfig {
        model: model.session_model_config(),
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

fn within_bottom_threshold(max_offset: gpui_kit::Pixels, offset_y: gpui_kit::Pixels) -> bool {
    max_offset + offset_y <= px(24.0)
}

#[cfg(test)]
fn update_chat_follow_on_scroll(
    delta_y: gpui_kit::Pixels,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn close_test_window(view: Entity<DesktopApp>, cx: &mut gpui_kit::VisualTestContext) {
        let weak_view = view.downgrade();
        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        assert!(
            weak_view.upgrade().is_none(),
            "closing the test window must release its app and session database handles"
        );
    }

    #[gpui_kit::test]
    fn trajectory_scroll_callback_can_leave_and_rejoin_tail(cx: &mut gpui_kit::TestAppContext) {
        use gpui_kit::{IntoElement, ScrollDelta, Styled, div, list, size};

        struct TestList(ListState);
        impl gpui_kit::Render for TestList {
            fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
                list(self.0.clone(), |_, _, _| {
                    div().h(px(20.0)).w_full().into_any_element()
                })
                .size_full()
            }
        }

        let root = std::env::temp_dir().join(format!(
            "kcastle-trajectory-scroll-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let (startup, _) = crate::desktop_startup(root.clone()).unwrap();
        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let app = DesktopApp::new(startup, window, cx);
            window.blur(cx);
            app
        });
        let state = cx.read_entity(&view, |app, _| app.trajectory_scroll.clone());
        state.reset_with_uniform_height(5, px(20.0));
        state.scroll_to(ListOffset {
            item_ix: 2,
            offset_in_item: px(0.0),
        });

        for (delta, follows) in [(10.0, false), (-10.0, true)] {
            cx.draw(
                point(px(0.0), px(0.0)),
                size(px(100.0), px(60.0)),
                |_, cx| cx.new(|_| TestList(state.clone())).into_any_element(),
            );
            // Exercise GPUI's real callback while its ListState is mutably borrowed.
            cx.simulate_event(ScrollWheelEvent {
                position: point(px(50.0), px(30.0)),
                delta: ScrollDelta::Pixels(point(px(0.0), px(delta))),
                ..Default::default()
            });
            cx.read_entity(&view, |app, _| {
                assert_eq!(app.trajectory_follow_tail.get(), follows);
            });
        }

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

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
        presentations.activate("test-session");
        assert!(
            !presentations
                .sync_message(message.key, 1, message.revision, &message.text, true)
                .markdown
                .tail_blocks()
                .is_empty()
        );
    }

    #[test]
    fn invalid_session_open_errors_are_silent_but_operational_errors_are_reported() {
        let invalid_locator = SessionError::Invalid("bad locator".into());
        let corrupt_history = SessionError::Store(SessionStoreError::Corrupt("bad tail".into()));
        let missing_database = SessionError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "missing database",
        ));
        assert!(session_open_error_notice(&invalid_locator).is_none());
        assert!(session_open_error_notice(&corrupt_history).is_none());
        assert!(session_open_error_notice(&missing_database).is_none());

        let writer_busy = SessionError::Store(SessionStoreError::WriterBusy {
            session_id: SessionId::from_raw("busy-session"),
        });
        assert!(
            session_open_error_notice(&writer_busy)
                .is_some_and(|notice| notice.contains("already has an active writer"))
        );
    }

    #[test]
    fn catalog_cache_policy_clears_invalid_data_but_retains_transient_failures() {
        let invalid = SessionError::Store(SessionStoreError::UnsupportedSchemaVersion {
            found: 5,
            expected: 6,
        });
        let transient = SessionError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "database temporarily unavailable",
        ));
        let operational = SessionError::Store(SessionStoreError::ReadonlyStore);
        assert!(should_clear_catalog_after_error(&invalid));
        assert!(!should_clear_catalog_after_error(&transient));
        assert!(!should_clear_catalog_after_error(&operational));
    }

    #[test]
    fn transient_and_operational_catalog_failures_retain_every_cached_projection() {
        let project_id = ProjectId::default_project();
        let sessions_dir = PathBuf::from("sessions");
        let path = sessions_dir.join("retained.session-v2");
        let session_id = SessionId::from_raw("retained");
        let session = SessionInfo {
            id: session_id.clone(),
            project_id: project_id.as_str().into(),
            path: path.clone(),
            title: "Retained".into(),
            created_at: 0,
            updated_at: 0,
        };
        let errors = [
            SessionError::Io(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "database temporarily unavailable",
            )),
            SessionError::Store(SessionStoreError::ReadonlyStore),
        ];

        for error in errors {
            let mut sessions = HashMap::from([(sessions_dir.clone(), vec![session.clone()])]);
            let mut documents = HashMap::from([(
                path.clone(),
                session_search_document(Arc::from(["retained".into()])),
            )]);
            let mut indices = HashMap::from([((project_id.clone(), session_id.clone()), 0)]);

            assert!(!apply_project_catalog_result(
                &project_id,
                &sessions_dir,
                Err(error),
                &mut sessions,
                &mut documents,
                &mut indices,
            ));

            assert_eq!(sessions[&sessions_dir], vec![session.clone()]);
            assert_eq!(documents[&path].searchable.as_ref(), "retained");
            assert_eq!(indices[&(project_id.clone(), session_id.clone())], 0);
        }
    }

    #[test]
    fn startup_catalog_loader_reads_once_and_builds_consistent_linear_projections() {
        const SESSIONS_PER_PROJECT: usize = 3_334;

        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-linear-catalog-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace_a = root.join("workspace-a");
        let workspace_b = root.join("workspace-b");
        std::fs::create_dir_all(&workspace_a).unwrap();
        std::fs::create_dir_all(&workspace_b).unwrap();
        let (mut project_store, _) =
            ProjectStore::load(root.join("state"), Some(workspace_a)).unwrap();
        project_store.add(workspace_b).unwrap();
        let project_count = project_store.projects().len();
        let mut reads = HashMap::<String, usize>::new();

        let cache = load_session_catalog_cache_with(
            &project_store,
            |directory: &Path, project_id: &str| {
                *reads.entry(project_id.to_owned()).or_default() += 1;
                let mut catalog = SessionCatalog {
                    sessions: Vec::with_capacity(SESSIONS_PER_PROJECT),
                    search_values: HashMap::with_capacity(SESSIONS_PER_PROJECT),
                };
                for index in 0..SESSIONS_PER_PROJECT {
                    let raw_id = format!("{project_id}-{index}");
                    let id = SessionId::from_raw(raw_id.clone());
                    let title = format!("Session {raw_id}");
                    let path = directory.join(format!("{raw_id}.session-v2"));
                    catalog.sessions.push(SessionInfo {
                        id,
                        project_id: project_id.into(),
                        path: path.clone(),
                        title: title.clone(),
                        created_at: index as u64,
                        updated_at: index as u64,
                    });
                    catalog
                        .search_values
                        .insert(path, Arc::from([title.clone()]));
                }
                Ok(catalog)
            },
        );

        let expected_sessions = project_count * SESSIONS_PER_PROJECT;
        assert!(expected_sessions >= 10_000);
        assert_eq!(reads.len(), project_count);
        assert!(reads.values().all(|reads| *reads == 1));
        assert_eq!(cache.project_sessions.len(), project_count);
        assert_eq!(cache.session_search_documents.len(), expected_sessions);
        assert_eq!(cache.session_catalog_indices.len(), expected_sessions);
        for project in project_store.projects() {
            let sessions = &cache.project_sessions[&project.sessions_dir];
            assert_eq!(sessions.len(), SESSIONS_PER_PROJECT);
            for (index, session) in sessions.iter().enumerate() {
                assert_eq!(session.project_id, project.id.as_str());
                assert_eq!(
                    cache
                        .session_catalog_indices
                        .get(&(project.id.clone(), session.id.clone())),
                    Some(&index)
                );
                assert!(cache.session_search_documents.contains_key(&session.path));
            }
        }

        drop(project_store);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clearing_an_invalid_project_catalog_removes_all_project_projections_only() {
        let project_id = ProjectId::default_project();
        let other_project_id: ProjectId = serde_json::from_str("\"other-project\"").unwrap();
        let sessions_dir = PathBuf::from("sessions");
        let other_sessions_dir = PathBuf::from("other-sessions");
        let invalid_path = sessions_dir.join("invalid.session-v2");
        let other_path = other_sessions_dir.join("valid.session-v2");
        let invalid_id = SessionId::from_raw("invalid");
        let other_id = SessionId::from_raw("valid");
        let session = |id: SessionId, project: &ProjectId, path: PathBuf| SessionInfo {
            id,
            project_id: project.as_str().into(),
            path,
            title: "Session".into(),
            created_at: 0,
            updated_at: 0,
        };
        let mut sessions = HashMap::from([
            (
                sessions_dir.clone(),
                vec![session(
                    invalid_id.clone(),
                    &project_id,
                    invalid_path.clone(),
                )],
            ),
            (
                other_sessions_dir.clone(),
                vec![session(
                    other_id.clone(),
                    &other_project_id,
                    other_path.clone(),
                )],
            ),
        ]);
        let document = || session_search_document(Arc::from(["needle".into()]));
        let mut documents = HashMap::from([
            (invalid_path.clone(), document()),
            (other_path.clone(), document()),
        ]);
        let mut indices = HashMap::from([
            ((project_id.clone(), invalid_id), 0),
            ((other_project_id.clone(), other_id), 0),
        ]);

        clear_project_catalog_cache(
            &project_id,
            &sessions_dir,
            &mut sessions,
            &mut documents,
            &mut indices,
        );

        assert!(sessions[&sessions_dir].is_empty());
        assert_eq!(sessions[&other_sessions_dir].len(), 1);
        assert!(!documents.contains_key(&invalid_path));
        assert!(documents.contains_key(&other_path));
        assert!(!indices.keys().any(|(project, _)| project == &project_id));
        assert!(
            indices
                .keys()
                .any(|(project, _)| project == &other_project_id)
        );
    }

    #[gpui_kit::test]
    fn refreshing_an_invalid_catalog_removes_stale_sidebar_search_and_index_rows(
        cx: &mut gpui_kit::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-invalid-switch-{}-{}",
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
        let invalid_id = SessionId::from_raw("stale-invalid-switch");
        let invalid_path = project_b
            .sessions_dir
            .join(format!("{invalid_id}.session-v2"));
        let invalid = SessionInfo {
            id: invalid_id.clone(),
            project_id: project_b.id.as_str().into(),
            path: invalid_path.clone(),
            title: "Stale invalid session".into(),
            created_at: 0,
            updated_at: 0,
        };
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
            window.blur(cx);
            app
        });

        std::fs::write(
            project_b
                .sessions_dir
                .join(kcastle_agent::SESSION_DATABASE_FILE),
            b"not a sqlite database",
        )
        .unwrap();
        cx.update(|_, app| {
            view.update(app, |this, _| {
                this.project_sessions
                    .get_mut(&project_b.sessions_dir)
                    .unwrap()
                    .push(invalid.clone());
                this.session_search_documents.insert(
                    invalid_path.clone(),
                    session_search_document(Arc::from(["stale".into()])),
                );
                this.session_catalog_indices
                    .insert((project_b.id.clone(), invalid_id.clone()), 0);

                assert!(!this.refresh_project_catalog(&project_b.id));
            });
        });

        cx.read_entity(&view, |app, _| {
            assert!(app.project_sessions[&project_b.sessions_dir].is_empty());
            assert!(!app.session_search_documents.contains_key(&invalid_path));
            assert!(
                !app.session_catalog_indices
                    .contains_key(&(project_b.id.clone(), invalid_id))
            );
        });

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn removing_an_invalid_catalog_entry_reindexes_remaining_entries() {
        let project_id = ProjectId::default_project();
        let sessions_dir = PathBuf::from("sessions");
        let first_path = sessions_dir.join("first.session-v2");
        let second_path = sessions_dir.join("second.session-v2");
        let first = SessionInfo {
            id: SessionId::from_raw("first"),
            project_id: project_id.as_str().into(),
            path: first_path.clone(),
            title: "First".into(),
            created_at: 0,
            updated_at: 0,
        };
        let second = SessionInfo {
            id: SessionId::from_raw("second"),
            project_id: project_id.as_str().into(),
            path: second_path.clone(),
            title: "Second".into(),
            created_at: 0,
            updated_at: 0,
        };
        let mut sessions =
            HashMap::from([(sessions_dir.clone(), vec![first.clone(), second.clone()])]);
        let mut documents = HashMap::from([
            (
                first_path.clone(),
                session_search_document(Arc::from(["first".into()])),
            ),
            (
                second_path.clone(),
                session_search_document(Arc::from(["second".into()])),
            ),
        ]);
        let mut indices = HashMap::from([
            ((project_id.clone(), first.id.clone()), 0),
            ((project_id.clone(), second.id.clone()), 1),
        ]);

        assert_eq!(
            remove_session_catalog_entry(
                &project_id,
                &sessions_dir,
                &first_path,
                &mut sessions,
                &mut documents,
                &mut indices,
            ),
            Some(first.id)
        );
        assert_eq!(sessions[&sessions_dir], vec![second.clone()]);
        assert!(!documents.contains_key(&first_path));
        assert!(documents.contains_key(&second_path));
        assert_eq!(indices.get(&(project_id, second.id)), Some(&0));
    }

    #[gpui_kit::test]
    fn created_session_path_refreshes_the_sidebar_list(cx: &mut gpui_kit::TestAppContext) {
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
            window.blur(cx);
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
            assert_eq!(
                app.project_sessions[&app.core.workspace.sessions_dir].len(),
                1
            );
            assert_eq!(
                app.project_sessions[&app.core.workspace.sessions_dir][0].path,
                app.core.session.current
            );
        });

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn opening_an_invalid_session_silently_removes_its_catalog_entry(
        cx: &mut gpui_kit::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-invalid-open-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap().clone();
        let invalid = SessionInfo {
            id: SessionId::from_raw("invalid-open"),
            project_id: project.id.as_str().into(),
            path: project.sessions_dir.join("invalid-open.not-a-session"),
            title: "Invalid".into(),
            created_at: 0,
            updated_at: 0,
        };
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
            window.blur(cx);
            app
        });

        cx.update(|window, app| {
            view.update(app, |this, cx| {
                this.project_sessions
                    .get_mut(&project.sessions_dir)
                    .unwrap()
                    .push(invalid.clone());
                this.session_catalog_indices
                    .insert((project.id.clone(), invalid.id.clone()), 0);
                this.session_search_documents.insert(
                    invalid.path.clone(),
                    session_search_document(Arc::from(["invalid".into()])),
                );
                let generation = this.begin_runtime_selection_intent();
                let key = (project.id.clone(), invalid.path.clone());
                assert!(this.start_session_open_request(&key, generation));
                assert_eq!(
                    this.finish_session_open_request(&key, generation, &project.id),
                    SessionOpenCompletion::Current
                );
                // Exercise the same atomic error branch as `open_session_async` without asking a
                // GPUI test window to perform the real-platform focus operation at its end.
                this.discard_invalid_session_catalog_entry(active_project, &invalid.path);
                assert!(!this.resolve_failed_runtime_selection(
                    generation,
                    &project.id,
                    &invalid.path,
                    active_project,
                    cx,
                ));
                let _ = window;
            });
        });
        cx.run_until_parked();

        cx.read_entity(&view, |app, _| {
            assert!(!app.selection_pending());
            assert!(app.project_sessions[&project.sessions_dir].is_empty());
            assert!(!app.session_search_documents.contains_key(&invalid.path));
            assert!(app.core.transient_messages.is_empty());
            assert!(app.core.session.current.as_os_str().is_empty());
        });

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn archive_and_restore_refresh_both_session_catalogs(cx: &mut gpui_kit::TestAppContext) {
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
            window.blur(cx);
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
        assert!(!archived.path.exists());

        cx.update(|_window, app| {
            view.update(app, |this, cx| {
                this.restore_archived_session(active_project, archived.clone(), cx)
            });
        });
        cx.read_entity(&view, |app, _| {
            assert_eq!(app.project_sessions[&sessions_dir].len(), 1);
            assert!(app.project_archived_sessions[&sessions_dir].is_empty());
        });

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn stale_session_open_cannot_replace_a_new_chat_draft(cx: &mut gpui_kit::TestAppContext) {
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
            window.blur(cx);
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

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn pending_session_open_does_not_submit_to_the_previous_runtime(
        cx: &mut gpui_kit::TestAppContext,
    ) {
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
            window.blur(cx);
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

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn successful_session_open_captures_view_edits_made_while_loading(
        cx: &mut gpui_kit::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-pending-open-view-state-{}-{}",
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
            SessionId::from_raw("pending-view-state-target"),
            Some("Pending view-state target"),
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
            window.blur(cx);
            app
        });

        let loaded_target =
            Session::open_writable_in_project(&target.path, project.id.as_str()).unwrap();
        cx.update(|window, app| {
            view.update(app, |this, cx| {
                let source_key = this
                    .runtime_location(&this.selected_runtime)
                    .expect("the source draft is registered");
                let source = this.selected_runtime.clone();
                let generation = this.begin_runtime_selection_intent();
                let target_key = (project.id.clone(), target.path.clone());
                assert!(this.start_session_open_request(&target_key, generation));

                // These changes happen after the eager capture performed by the click handler.
                // The successful handoff must capture them once more before replacing the runtime.
                this.set_trajectory(true, window, cx);
                this.trajectory_query_value = "edited while loading".into();
                this.trajectory_follow_tail.set(false);
                let offset = ListOffset {
                    item_ix: 7,
                    offset_in_item: px(3.0),
                };
                this.trajectory_scroll_restore.set(Some(offset));
                this.core.details.tab_history = vec![DetailsTab::Raw, DetailsTab::Timing];
                this.details_scroll.set_offset(point(px(0.0), px(-42.0)));
                let projection = &this.core.session_view.trajectory;
                let axis = AxisId {
                    document_generation: projection.projection_lineage(),
                    geometry_revision: projection.revision(),
                    mode: this.core.trajectory.mode,
                };
                let selected = AxisRange {
                    axis,
                    range: DomainRange::new(2.0, 5.0),
                };
                let viewport = AxisRange {
                    axis,
                    range: DomainRange::new(1.0, 8.0),
                };
                this.core.trajectory.selected_range = Some(selected);
                this.core.trajectory.visible_range = Some(viewport);

                assert_eq!(
                    this.finish_session_open_request(&target_key, generation, &project.id),
                    SessionOpenCompletion::Current
                );
                let runtime = this
                    .reconcile_loaded_runtime(active_project, loaded_target, cx)
                    .unwrap();
                this.select_runtime(runtime, cx);

                let saved = &this.view_states[&source_key];
                assert_eq!(saved.trajectory_query, "edited while loading");
                assert!(!saved.trajectory_follow_tail);
                let saved_offset = saved
                    .trajectory_offset
                    .expect("the loading-time ledger offset should be saved");
                assert_eq!(saved_offset.item_ix, offset.item_ix);
                assert_eq!(saved_offset.offset_in_item, offset.offset_in_item);
                assert_eq!(
                    saved.details_tab_history,
                    vec![DetailsTab::Raw, DetailsTab::Timing]
                );
                assert_eq!(saved.details_offset, point(px(0.0), px(-42.0)));
                assert_eq!(
                    saved.timeline_selection,
                    Some(SavedTimelineRange::capture(selected))
                );
                assert_eq!(
                    saved.timeline_viewport,
                    Some(SavedTimelineRange::capture(viewport))
                );
                assert_eq!(this.core.session.current, target.path);
                assert_eq!(this.core.surface, Surface::Chat);
                assert!(!this.core.layout_input.trajectory_visible);

                this.select_runtime(source, cx);
                assert_eq!(this.core.surface, Surface::Trajectory);
                assert!(this.core.layout_input.trajectory_visible);
                assert_eq!(this.trajectory_query_value, "edited while loading");
            });
        });

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn failed_cross_project_open_falls_back_to_the_target_project_draft(
        cx: &mut gpui_kit::TestAppContext,
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
            window.blur(cx);
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
                let _ = this.transition(Action::ActivateWorkspace {
                    index: project_b_index,
                    cwd: project_b.path.clone(),
                    sessions_dir: project_b.sessions_dir.clone(),
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

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn reopening_cached_runtime_reloads_external_revision_and_config_drift(
        cx: &mut gpui_kit::TestAppContext,
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
            model: SessionModelConfig {
                model_id: Some("test/test-model".into()),
                reasoning_effort: None,
            },
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
            window.blur(cx);
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

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[gpui_kit::test]
    fn background_failed_runtimes_share_the_terminal_cache_bound(
        cx: &mut gpui_kit::TestAppContext,
    ) {
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
            window.blur(cx);
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

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn message_render_keys_do_not_alias_across_session_reloads() {
        let first = message(Role::Assistant, "first".into());
        let second = message(Role::Assistant, "second".into());
        assert_ne!(first.key, second.key);
    }

    #[test]
    fn trajectory_folds_are_isolated_by_session_view_state() {
        let assistant = TrajectoryItemId::Assistant(kcastle_agent::RequestId::from("request-a"));
        let mut first = SessionViewState {
            trajectory_offset: Some(ListOffset {
                item_ix: 17,
                offset_in_item: px(6.0),
            }),
            trajectory_follow_tail: false,
            trajectory_query: "alpha crane".into(),
            ..SessionViewState::default()
        };
        first.collapsed_turns.insert(1);
        first.collapsed_assistants.insert(assistant.clone());
        let second = SessionViewState::default();

        assert_eq!(first.trajectory_offset.unwrap().item_ix, 17);
        assert_eq!(first.trajectory_offset.unwrap().offset_in_item, px(6.0));
        assert_eq!(first.trajectory_query, "alpha crane");
        assert!(!first.trajectory_follow_tail);
        assert!(second.trajectory_offset.is_none());
        assert_eq!(second.trajectory_query, "");
        assert!(second.trajectory_follow_tail);
        assert_eq!(first.collapsed_turns, HashSet::from([1]));
        assert_eq!(first.collapsed_assistants, HashSet::from([assistant]));
        assert!(second.collapsed_turns.is_empty());
        assert!(second.collapsed_assistants.is_empty());
    }

    #[test]
    fn saved_timeline_ranges_rebase_to_a_replayed_session_projection() {
        let original = AxisRange {
            axis: AxisId {
                document_generation: 17,
                geometry_revision: 4,
                mode: TimelineMode::Duration,
            },
            range: DomainRange::new(125.0, 875.0),
        };
        let saved = SavedTimelineRange::capture(original);

        let restored = saved
            .restore(TimelineMode::Duration, 91, 12)
            .expect("the same mode should restore");
        assert_eq!(restored.range, original.range);
        assert_eq!(restored.axis.document_generation, 91);
        assert_eq!(restored.axis.geometry_revision, 12);
        assert_eq!(restored.axis.mode, TimelineMode::Duration);
        assert_ne!(restored.axis, original.axis);
        assert!(saved.restore(TimelineMode::Sequence, 91, 12).is_none());
    }

    #[gpui_kit::test]
    fn evicted_runtime_restores_surface_and_ranges_on_the_fresh_projection_lineage(
        cx: &mut gpui_kit::TestAppContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-desktop-evicted-timeline-state-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        let (project_store, active_project) =
            ProjectStore::load(root.join("state"), Some(workspace.clone())).unwrap();
        let project = project_store.project(active_project).unwrap().clone();
        let session_info = create_v2_session(
            &project.sessions_dir,
            project.id.as_str(),
            SessionId::from_raw("evicted-timeline-state"),
            Some("Evicted timeline state"),
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
            window.blur(cx);
            app
        });

        let first_session =
            Session::open_writable_in_project(&session_info.path, project.id.as_str()).unwrap();
        let (old_lineage, evicted_runtime) = cx.update(|window, app| {
            view.update(app, |this, cx| {
                let draft = this.selected_runtime.clone();
                let runtime = this
                    .create_runtime(active_project, first_session, cx)
                    .unwrap();
                let key = (project.id.clone(), session_info.id.clone());
                this.select_runtime(runtime.clone(), cx);
                this.set_trajectory(true, window, cx);
                assert_eq!(this.core.surface, Surface::Trajectory);
                this.select_runtime(draft.clone(), cx);
                assert_eq!(this.core.surface, Surface::Chat);
                this.select_runtime(runtime.clone(), cx);
                assert_eq!(this.core.surface, Surface::Trajectory);

                this.set_trajectory(false, window, cx);
                assert_eq!(this.core.surface, Surface::Chat);
                this.select_runtime(draft.clone(), cx);
                this.select_runtime(runtime.clone(), cx);
                assert_eq!(this.core.surface, Surface::Chat);
                this.set_trajectory(true, window, cx);
                this.select_runtime(runtime.clone(), cx);
                assert_eq!(this.core.surface, Surface::Trajectory);

                let projection = &this.core.session_view.trajectory;
                let old_lineage = projection.projection_lineage();
                let axis = AxisId {
                    document_generation: old_lineage,
                    geometry_revision: projection.revision(),
                    mode: this.core.trajectory.mode,
                };
                this.core.trajectory.selected_range = Some(AxisRange {
                    axis,
                    range: DomainRange::new(2.0, 4.0),
                });
                this.core.trajectory.visible_range = Some(AxisRange {
                    axis,
                    range: DomainRange::new(1.0, 8.0),
                });

                // Switching away captures the semantic ranges. Removing the terminal runtime then
                // drops the document and its projection identity exactly like LRU eviction.
                this.select_runtime(draft, cx);
                assert_eq!(this.core.surface, Surface::Chat);
                this.remove_cached_runtime(&key);
                (old_lineage, runtime.downgrade())
            })
        });
        cx.run_until_parked();
        assert!(
            evicted_runtime.upgrade().is_none(),
            "the test must drop the original document projection"
        );

        let replayed_session =
            Session::open_writable_in_project(&session_info.path, project.id.as_str()).unwrap();
        cx.update(|_, app| {
            view.update(app, |this, cx| {
                let runtime = this
                    .create_runtime(active_project, replayed_session, cx)
                    .unwrap();
                let new_lineage = runtime
                    .read(cx)
                    .snapshot()
                    .view
                    .trajectory
                    .projection_lineage();
                assert_ne!(new_lineage, old_lineage);

                this.select_runtime(runtime, cx);
                assert_eq!(this.core.surface, Surface::Trajectory);
                assert!(this.core.layout_input.trajectory_visible);
                let selection = this
                    .core
                    .trajectory
                    .selected_range
                    .expect("selection should survive replay");
                let viewport = this
                    .core
                    .trajectory
                    .visible_range
                    .expect("viewport should survive replay");
                assert_eq!(selection.axis.document_generation, new_lineage);
                assert_eq!(viewport.axis.document_generation, new_lineage);
                assert_eq!(selection.range, DomainRange::new(2.0, 4.0));
                assert_eq!(viewport.range, DomainRange::new(1.0, 8.0));
            });
        });

        close_test_window(view, cx);
        std::fs::remove_dir_all(root).unwrap();
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
