use std::collections::HashSet;
use std::path::PathBuf;

use crate::domain::timeline::AxisRange;
use crate::domain::{
    ComposerMenu, DetailsSelection, DetailsTab, Message, TimelineMode, TrajectoryItemId,
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
    SelectDetails(Option<DetailsSelection>),
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
    ToggleDetailsUnixTime,
    ToggleTrajectoryTurn(u32),
    ToggleTrajectoryAssistant(TrajectoryItemId),
    SetTrajectoryTurnsCollapsed(HashSet<u32>),
    SetTrajectoryAssistantsCollapsed(HashSet<TrajectoryItemId>),
    ExpandTrajectoryGroups,
    RestoreSessionView {
        selected: Option<DetailsSelection>,
        details_tab_history: Vec<DetailsTab>,
        follow_chat_tail: bool,
    },
    ActivateWorkspace {
        index: usize,
        cwd: PathBuf,
        sessions_dir: PathBuf,
    },
    SetActiveProject(usize),
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
