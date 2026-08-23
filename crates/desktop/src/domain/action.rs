use std::path::PathBuf;

use kcastle_agent::SessionInfo;

use crate::domain::timeline::AxisRange;
use crate::domain::{ComposerMenu, DetailsTab, Message, TimelineMode, TrajectoryItemId};
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
    SelectDetails(Option<TrajectoryItemId>),
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
    SetTimelineSelection(Option<AxisRange>),
    SetTimelineViewport(Option<AxisRange>),
    ToggleTimelineUnixTime,
    ToggleTrajectoryTurns,
    ToggleTrajectoryCalls,
    ExpandTrajectoryGroups,
    RestoreSessionView {
        selected: Option<TrajectoryItemId>,
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
    AppendTransientNotice(Box<Message>),
    Scroll(ScrollIntent),
    #[cfg_attr(not(test), allow(dead_code))]
    StreamDeltasReceived(usize),
    LayoutInputChanged(LayoutInput),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Effect {
    ApplyChatTail,
}
