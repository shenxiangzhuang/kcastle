use std::path::PathBuf;

use kcastle_agent::SessionInfo;

use crate::domain::{
    ApprovalState, ComposerMenu, ConversationAction, ConversationState, DetailsTab, MessageId,
    OperationId, RunId,
};
use crate::layout::LayoutInput;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScrollIntent {
    Away,
    Toward { at_tail: bool },
    JumpToTail,
}

#[derive(Debug)]
pub(crate) enum Action {
    ToggleSidebar,
    ShowChat,
    ShowTrajectory,
    SelectDetails(Option<MessageId>),
    SetDetailsTab(DetailsTab),
    SetApproval(Option<ApprovalState>),
    SetComposerMenu(Option<ComposerMenu>),
    MoveComposerHighlight {
        delta: isize,
        item_count: usize,
    },
    ToggleSessionSearch,
    ToggleSidebarOptions,
    CloseTransientOverlays,
    DismissTransient,
    SetSidebarGrouping(bool),
    SetSidebarSort(bool),
    SetSessionActionTarget(Option<PathBuf>),
    ToggleProjectExpanded(PathBuf),
    ExpandProject(PathBuf),
    ToggleTrajectoryDuration,
    ToggleTrajectoryTurns,
    ToggleTrajectoryCalls,
    ExpandTrajectoryGroups,
    RestoreSessionView {
        selected: Option<MessageId>,
        details_tab: DetailsTab,
        follow_chat_tail: bool,
    },
    ActivateWorkspace {
        index: usize,
        cwd: PathBuf,
        sessions_dir: PathBuf,
        sessions: Vec<SessionInfo>,
    },
    SetActiveProject(usize),
    RefreshSessions(Vec<SessionInfo>),
    SetCurrentSession(PathBuf),
    BeginOpenSession(PathBuf),
    SessionOpened {
        operation: OperationId,
        conversation: ConversationState,
        current_session: PathBuf,
        sessions: Vec<SessionInfo>,
    },
    BeginRenameSession(String),
    SessionRenamed {
        operation: OperationId,
        title: String,
        sessions: Vec<SessionInfo>,
    },
    SessionOperationFailed {
        operation: OperationId,
        message: String,
    },
    ReplaceConversation {
        conversation: ConversationState,
        current_session: PathBuf,
        sessions: Vec<SessionInfo>,
    },
    ResetConversation,
    Conversation(ConversationAction),
    Scroll(ScrollIntent),
    StreamDeltasReceived(usize),
    LayoutInputChanged(LayoutInput),
    BeginSessionCreation(String),
    SessionCreationFailed {
        operation: OperationId,
        message: String,
    },
    SessionCreated {
        operation: OperationId,
        current_session: PathBuf,
        sessions: Vec<SessionInfo>,
    },
    BeginRun(String),
    RunStartFailed {
        run: RunId,
        message: String,
    },
    RunFinished(RunId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    ApplyChatTail,
    CreateSession {
        operation: OperationId,
        input: String,
    },
    StartRun {
        run: RunId,
        input: String,
    },
    OpenSession {
        operation: OperationId,
        path: PathBuf,
    },
    RenameSession {
        operation: OperationId,
        title: String,
    },
}
