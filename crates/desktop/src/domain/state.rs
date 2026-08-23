use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use im::Vector;
use kcastle_agent::SessionInfo;

use crate::domain::timeline::{AxisRange, TimelineMode};
use crate::domain::{LayoutGeneration, Message, RunId, SessionView, TrajectoryItemId};
use crate::layout::{LayoutInput, LayoutPlan, resolve_layout};

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DetailsState {
    pub(crate) selected: Option<TrajectoryItemId>,
    pub(crate) tab: DetailsTab,
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
    Preview,
    Raw,
    Payload,
    Result,
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
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TrajectoryState {
    pub(crate) collapsed_turns: bool,
    pub(crate) collapsed_calls: bool,
    pub(crate) mode: TimelineMode,
    /// Ranges are meaningful only in the projection that created them. Keeping
    /// the axis alongside the coordinates makes stale view state unrepresentable
    /// as a range on a newer document revision or another timeline mode.
    pub(crate) selected_range: Option<AxisRange>,
    pub(crate) visible_range: Option<AxisRange>,
    pub(crate) unix_time: bool,
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
    pub(crate) sessions: Vec<SessionInfo>,
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
