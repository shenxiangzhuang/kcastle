use std::collections::HashSet;
use std::path::PathBuf;

use kcastle_agent::SessionInfo;

use crate::domain::{ConversationState, LayoutGeneration, MessageId, OperationId, RunId};
use crate::layout::{LayoutInput, LayoutPlan, resolve_layout};

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
    pub(crate) session_action_target: Option<PathBuf>,
    pub(crate) group_by_workspace: bool,
    pub(crate) sort_by_recent: bool,
}

impl Default for SidebarState {
    fn default() -> Self {
        Self {
            search_sessions: false,
            options_open: false,
            session_action_target: None,
            group_by_workspace: true,
            sort_by_recent: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DetailsState {
    pub(crate) selected: Option<MessageId>,
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
    Schema,
    Timing,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum RunState {
    #[default]
    Idle,
    CreatingSession {
        operation: OperationId,
        input: String,
    },
    Running {
        run: RunId,
    },
    Failed {
        operation: Option<OperationId>,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SessionOperationKind {
    Open { path: PathBuf },
    Rename { path: PathBuf, title: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PendingSessionOperation {
    pub(crate) operation: OperationId,
    pub(crate) kind: SessionOperationKind,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TrajectoryState {
    pub(crate) collapsed_turns: bool,
    pub(crate) collapsed_calls: bool,
    pub(crate) show_duration: bool,
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
    pub(crate) conversation: ConversationState,
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
    pub(crate) pending_session_operation: Option<PendingSessionOperation>,
    pub(crate) last_error: Option<String>,
    pub(crate) next_operation: OperationId,
    pub(crate) next_run: RunId,
    pub(crate) layout_generation: LayoutGeneration,
    pub(crate) layout_input: LayoutInput,
    pub(crate) layout: LayoutPlan,
}

impl AppState {
    pub(crate) fn new(layout_input: LayoutInput) -> Self {
        Self {
            conversation: ConversationState::default(),
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
            pending_session_operation: None,
            last_error: None,
            next_operation: OperationId::default(),
            next_run: RunId::default(),
            layout_generation: LayoutGeneration::default(),
            layout: resolve_layout(layout_input),
            layout_input,
        }
    }
}
