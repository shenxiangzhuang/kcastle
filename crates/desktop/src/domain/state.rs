use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use crate::domain::timeline::{AxisRange, TimelineMode};
use crate::domain::{
    LayoutGeneration, Message, RunId, SessionView, TrajectoryItemId, TrajectoryRequestKey,
};
use crate::layout::{LayoutInput, LayoutPlan, resolve_layout};
use im::Vector;
use kcastle_agent::RunFailure;

pub(crate) const INITIAL_SESSION_LIMIT: usize = 5;
pub(crate) const SESSION_PAGE_SIZE: usize = 10;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Surface {
    #[default]
    Chat,
    Trajectory,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ComposerState {
    pub(crate) menu: Option<ComposerMenu>,
    pub(crate) highlighted_item: usize,
}

#[derive(Debug)]
pub(crate) struct SidebarState {
    pub(crate) search_sessions: bool,
    pub(crate) options_open: bool,
    pub(crate) group_by_workspace: bool,
    pub(crate) sort_by_recent: bool,
    pub(crate) visible_sessions_by_project: HashMap<PathBuf, usize>,
}

impl Default for SidebarState {
    fn default() -> Self {
        Self {
            search_sessions: false,
            options_open: false,
            group_by_workspace: true,
            sort_by_recent: true,
            visible_sessions_by_project: HashMap::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DetailsState {
    pub(crate) selected: Option<DetailsSelection>,
    /// Most-recently-used tab order. The inspector resolves the first tab that is valid for the
    /// selected entity, so changing between heterogeneous records cannot leave an impossible tab
    /// selected and returning to an entity restores the user's last meaningful tab.
    pub(crate) tab_history: Vec<DetailsTab>,
    /// A details-local display preference, reset whenever the selected entity or tab changes.
    pub(crate) unix_time: bool,
}

impl Default for DetailsState {
    fn default() -> Self {
        Self {
            selected: None,
            tab_history: vec![DetailsTab::Summary],
            unix_time: false,
        }
    }
}

impl DetailsState {
    pub(crate) fn active_tab(&self, available: &[DetailsTab]) -> DetailsTab {
        self.tab_history
            .iter()
            .rev()
            .copied()
            .find(|tab| available.contains(tab))
            .or_else(|| available.first().copied())
            .unwrap_or(DetailsTab::Summary)
    }

    pub(crate) fn activate_tab(&mut self, tab: DetailsTab) {
        self.tab_history.retain(|candidate| *candidate != tab);
        self.tab_history.push(tab);
        // The displayed clock preference belongs to Timing. Re-activating the effective Timing
        // tab after an entity change can move it to the MRU tail even though it was already the
        // resolved tab; that must not silently change the clock. Leaving Timing still resets it.
        if tab != DetailsTab::Timing {
            self.unix_time = false;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DetailsSelection {
    Record(TrajectoryItemId),
    Request(TrajectoryRequestKey),
}

impl DetailsSelection {
    pub(crate) fn record(&self) -> Option<&TrajectoryItemId> {
        match self {
            Self::Record(id) => Some(id),
            Self::Request(_) => None,
        }
    }

    pub(crate) fn request(&self) -> Option<&TrajectoryRequestKey> {
        match self {
            Self::Record(_) => None,
            Self::Request(key) => Some(key),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ApprovalState {
    pub(crate) call_id: String,
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum DetailsTab {
    #[default]
    Summary,
    SystemPrompt,
    Diff,
    Tools,
    Preview,
    Raw,
    Payload,
    Result,
    Schema,
    Options,
    Usage,
    Timing,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum RunState {
    #[default]
    Idle,
    Preparing,
    Running {
        run: RunId,
    },
    Failed {
        failure: RunFailure,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct TrajectoryState {
    pub(crate) collapsed_turns: HashSet<u32>,
    pub(crate) collapsed_assistants: HashSet<TrajectoryItemId>,
    /// Monotonic presentation revision used by render caches to avoid comparing large fold sets
    /// on every GPUI notification.
    pub(crate) fold_revision: u64,
    pub(crate) mode: TimelineMode,
    /// Ranges are meaningful only in the projection that created them. Keeping
    /// the axis alongside the coordinates makes stale view state unrepresentable
    /// as a range on a newer document revision or another timeline mode.
    pub(crate) selected_range: Option<AxisRange>,
    pub(crate) visible_range: Option<AxisRange>,
}

#[derive(Debug, Default)]
pub(crate) struct WorkspaceState {
    pub(crate) cwd: PathBuf,
    pub(crate) active_project: usize,
    pub(crate) expanded_projects: HashSet<PathBuf>,
    pub(crate) sessions_dir: PathBuf,
}

#[derive(Debug, Default)]
pub(crate) struct SessionState {
    pub(crate) current: PathBuf,
}

#[derive(Debug)]
pub(crate) struct AppState {
    /// One revision-stamped aggregate published from one canonical document.
    /// Conversation and trajectory can therefore never come from different commits.
    pub(crate) session_view: Arc<SessionView>,
    /// Bounded UI feedback that is intentionally excluded from the durable document.
    pub(crate) transient_messages: Vector<Arc<Message>>,
    pub(crate) composer: ComposerState,
    pub(crate) sidebar: SidebarState,
    pub(crate) sidebar_requested: bool,
    pub(crate) surface: Surface,
    pub(crate) details: DetailsState,
    pub(crate) approval: Option<ApprovalState>,
    pub(crate) trajectory: TrajectoryState,
    pub(crate) workspace: WorkspaceState,
    pub(crate) session: SessionState,
    pub(crate) follow_chat_tail: bool,
    pub(crate) unread_stream_updates: usize,
    pub(crate) run: RunState,
    pub(crate) layout_generation: LayoutGeneration,
    pub(crate) layout_input: LayoutInput,
    pub(crate) layout: LayoutPlan,
}

impl AppState {
    pub(crate) fn new(layout_input: LayoutInput) -> Self {
        Self {
            session_view: Arc::new(SessionView::default()),
            transient_messages: Vector::new(),
            composer: ComposerState::default(),
            sidebar: SidebarState::default(),
            sidebar_requested: true,
            surface: Surface::Chat,
            details: DetailsState::default(),
            approval: None,
            trajectory: TrajectoryState::default(),
            workspace: WorkspaceState::default(),
            session: SessionState::default(),
            follow_chat_tail: true,
            unread_stream_updates: 0,
            run: RunState::Idle,
            layout_generation: LayoutGeneration::default(),
            layout: resolve_layout(layout_input),
            layout_input,
        }
    }
}
