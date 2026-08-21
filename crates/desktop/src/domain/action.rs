use std::path::PathBuf;

use kcastle_agent::SessionInfo;

use crate::domain::{ComposerMenu, ConversationAction, DetailsTab, MessageId, TimelineMode};
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
    ShowMoreSessions(PathBuf),
    ToggleProjectExpanded(PathBuf),
    ExpandProject(PathBuf),
    SetTimelineMode(TimelineMode),
    SetTimelineSelection(Option<(f64, f64)>),
    SetTimelineViewport(Option<(f64, f64)>),
    ToggleTimelineUnixTime,
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
    Conversation(Box<ConversationAction>),
    Scroll(ScrollIntent),
    #[cfg_attr(not(test), allow(dead_code))]
    StreamDeltasReceived(usize),
    LayoutInputChanged(LayoutInput),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    ApplyChatTail,
}
