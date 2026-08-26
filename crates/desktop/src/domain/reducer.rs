use std::sync::Arc;

use crate::domain::{Action, AppState, Effect, ScrollIntent, Surface};
use crate::layout::resolve_layout;

pub(crate) fn reduce(state: &mut AppState, action: Action) -> Vec<Effect> {
    let mut effects = Vec::new();
    match action {
        Action::ToggleSidebar => {
            state.sidebar_requested = !state.sidebar_requested;
            state.layout_input.sidebar_requested = state.sidebar_requested;
            recompute_layout(state, &mut effects);
        }
        Action::ShowChat => {
            state.surface = Surface::Chat;
            state.layout_input.trajectory_visible = false;
            recompute_layout(state, &mut effects);
        }
        Action::ShowTrajectory => {
            state.surface = Surface::Trajectory;
            state.layout_input.trajectory_visible = true;
            recompute_layout(state, &mut effects);
        }
        Action::SelectDetails(selected) => {
            if state.details.selected != selected {
                state.details.unix_time = false;
            }
            state.details.selected = selected.clone();
            state.layout_input.details_visible = selected.is_some();
            recompute_layout(state, &mut effects);
        }
        Action::SetDetailsTab(tab) => state.details.activate_tab(tab),
        Action::SetComposerMenu(menu) => {
            state.composer.menu = menu;
            state.composer.highlighted_item = 0;
            state.sidebar.options_open = false;
        }
        Action::MoveComposerHighlight { delta, item_count } => {
            if item_count > 0 {
                state.composer.highlighted_item = (state.composer.highlighted_item as isize + delta)
                    .rem_euclid(item_count as isize)
                    as usize;
            }
        }
        Action::ToggleSessionSearch => {
            state.sidebar.search_sessions = !state.sidebar.search_sessions;
            if state.sidebar.search_sessions {
                state.sidebar.options_open = false;
            }
        }
        Action::ToggleSidebarOptions => {
            state.sidebar.options_open = !state.sidebar.options_open;
            if state.sidebar.options_open {
                state.composer.menu = None;
            }
        }
        Action::CloseTransientOverlays => {
            state.composer.menu = None;
            state.sidebar.options_open = false;
        }
        Action::DismissTransient => {
            state.composer.menu = match state.composer.menu {
                Some(crate::domain::ComposerMenu::Models | crate::domain::ComposerMenu::Effort) => {
                    Some(crate::domain::ComposerMenu::Model)
                }
                _ => None,
            };
            state.sidebar.options_open = false;
            state.sidebar.search_sessions = false;
            state.trajectory.selected_range = None;
        }
        Action::SetSidebarGrouping(group_by_workspace) => {
            state.sidebar.group_by_workspace = group_by_workspace;
            state.sidebar.options_open = false;
        }
        Action::SetSidebarSort(sort_by_recent) => {
            state.sidebar.sort_by_recent = sort_by_recent;
            state.sidebar.options_open = false;
        }
        Action::ShowMoreSessions(path) => {
            let visible = state
                .sidebar
                .visible_sessions_by_project
                .entry(path)
                .or_insert(crate::domain::INITIAL_SESSION_LIMIT);
            *visible = visible.saturating_add(crate::domain::SESSION_PAGE_SIZE);
        }
        Action::ToggleProjectExpanded(path) => {
            if !state.workspace.expanded_projects.remove(&path) {
                state.workspace.expanded_projects.insert(path);
            }
        }
        Action::ExpandProject(path) => {
            state.workspace.expanded_projects.insert(path);
        }
        Action::SetTimelineMode(mode) => {
            state.trajectory.mode = mode;
            state.trajectory.selected_range = None;
            state.trajectory.visible_range = None;
        }
        Action::SetTimelineSelection(range) => state.trajectory.selected_range = range,
        Action::SetTimelineViewport(range) => state.trajectory.visible_range = range,
        Action::ToggleDetailsUnixTime => {
            state.details.unix_time = !state.details.unix_time;
        }
        Action::ToggleTrajectoryTurn(turn) => {
            if !state.trajectory.collapsed_turns.remove(&turn) {
                state.trajectory.collapsed_turns.insert(turn);
            }
            state.trajectory.fold_revision = state.trajectory.fold_revision.saturating_add(1);
        }
        Action::ToggleTrajectoryAssistant(assistant) => {
            if !state.trajectory.collapsed_assistants.remove(&assistant) {
                state.trajectory.collapsed_assistants.insert(assistant);
            }
            state.trajectory.fold_revision = state.trajectory.fold_revision.saturating_add(1);
        }
        Action::SetTrajectoryTurnsCollapsed(turns) => {
            if state.trajectory.collapsed_turns != turns {
                state.trajectory.collapsed_turns = turns;
                state.trajectory.fold_revision = state.trajectory.fold_revision.saturating_add(1);
            }
        }
        Action::SetTrajectoryAssistantsCollapsed(assistants) => {
            if state.trajectory.collapsed_assistants != assistants {
                state.trajectory.collapsed_assistants = assistants;
                state.trajectory.fold_revision = state.trajectory.fold_revision.saturating_add(1);
            }
        }
        Action::ExpandTrajectoryGroups => {
            if !state.trajectory.collapsed_turns.is_empty()
                || !state.trajectory.collapsed_assistants.is_empty()
            {
                state.trajectory.collapsed_turns.clear();
                state.trajectory.collapsed_assistants.clear();
                state.trajectory.fold_revision = state.trajectory.fold_revision.saturating_add(1);
            }
        }
        Action::RestoreSessionView {
            selected,
            details_tab_history,
            follow_chat_tail,
        } => {
            state.details.selected = selected.clone();
            state.details.tab_history = if details_tab_history.is_empty() {
                vec![crate::domain::DetailsTab::Summary]
            } else {
                details_tab_history
            };
            state.details.unix_time = false;
            state.follow_chat_tail = follow_chat_tail;
            if follow_chat_tail {
                state.unread_stream_updates = 0;
            }
            state.layout_input.details_visible = selected.is_some();
            recompute_layout(state, &mut effects);
        }
        Action::ActivateWorkspace {
            index,
            cwd,
            sessions_dir,
        } => {
            state.workspace.active_project = index;
            state.workspace.expanded_projects.insert(cwd.clone());
            state.workspace.cwd = cwd;
            state.workspace.sessions_dir = sessions_dir;
        }
        Action::SetActiveProject(index) => state.workspace.active_project = index,
        Action::AppendTransientNotice(message) => {
            let mut message = *message;
            if let Some(previous) = state
                .transient_messages
                .back()
                .or_else(|| state.session_view.conversation.messages.back())
            {
                message.turn = previous.turn;
                message.step = previous.step;
                if message.request_id.is_none() {
                    message.request_id.clone_from(&previous.request_id);
                }
            }
            const MAX_TRANSIENT_NOTICES: usize = 8;
            if state.transient_messages.len() == MAX_TRANSIENT_NOTICES {
                state.transient_messages.pop_front();
            }
            state.transient_messages.push_back(Arc::new(message));
            if state.follow_chat_tail {
                effects.push(Effect::ApplyChatTail);
            }
        }
        Action::Scroll(intent) => match intent {
            ScrollIntent::Away => state.follow_chat_tail = false,
            ScrollIntent::Toward { at_tail } if at_tail => {
                state.follow_chat_tail = true;
                state.unread_stream_updates = 0;
            }
            ScrollIntent::Toward { .. } => {}
            ScrollIntent::JumpToTail => {
                state.follow_chat_tail = true;
                state.unread_stream_updates = 0;
                effects.push(Effect::ApplyChatTail);
            }
        },
        Action::StreamDeltasReceived(count) => {
            if count == 0 {
                return effects;
            }
            if state.follow_chat_tail {
                effects.push(Effect::ApplyChatTail);
            } else {
                state.unread_stream_updates = state.unread_stream_updates.saturating_add(count);
            }
        }
        Action::LayoutInputChanged(input) if input != state.layout_input => {
            state.layout_generation = state.layout_generation.next();
            state.layout_input = input;
            state.layout = resolve_layout(input);
            if state.follow_chat_tail {
                effects.push(Effect::ApplyChatTail);
            }
        }
        Action::LayoutInputChanged(_) => {}
    }
    effects
}

fn recompute_layout(state: &mut AppState, effects: &mut Vec<Effect>) {
    let layout = resolve_layout(state.layout_input);
    if layout == state.layout {
        return;
    }
    state.layout_generation = state.layout_generation.next();
    state.layout = layout;
    if state.follow_chat_tail {
        effects.push(Effect::ApplyChatTail);
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use kcastle_agent::RequestId;

    use super::*;
    use crate::domain::timeline::{AxisId, AxisRange, DomainRange, TimelineMode};
    use crate::domain::{DetailsSelection, DetailsTab, Message, MessageId, Role, TrajectoryItemId};
    use crate::layout::LayoutInput;

    #[test]
    fn show_more_sessions_adds_ten_to_the_initial_five() {
        let mut state = AppState::new(LayoutInput::default());
        let project = PathBuf::from("project-a");

        reduce(&mut state, Action::ShowMoreSessions(project.clone()));
        assert_eq!(
            state.sidebar.visible_sessions_by_project.get(&project),
            Some(&15)
        );

        reduce(&mut state, Action::ShowMoreSessions(project.clone()));
        assert_eq!(
            state.sidebar.visible_sessions_by_project.get(&project),
            Some(&25)
        );
    }

    #[test]
    fn dismissing_with_escape_clears_selection_without_losing_zoom() {
        let mut state = AppState::new(LayoutInput::default());
        let range = AxisRange {
            axis: AxisId {
                document_generation: 1,
                geometry_revision: 2,
                mode: TimelineMode::Duration,
            },
            range: DomainRange::new(10.0, 20.0),
        };
        state.trajectory.selected_range = Some(range);
        state.trajectory.visible_range = Some(range);

        reduce(&mut state, Action::DismissTransient);

        assert_eq!(state.trajectory.selected_range, None);
        assert_eq!(state.trajectory.visible_range, Some(range));
    }

    #[test]
    fn details_selection_uses_mru_valid_tabs_and_resets_local_clock_format() {
        let mut state = AppState::new(LayoutInput::default());
        let record =
            DetailsSelection::Record(TrajectoryItemId::Assistant(RequestId::from("request-1")));
        reduce(&mut state, Action::SelectDetails(Some(record)));
        reduce(&mut state, Action::SetDetailsTab(DetailsTab::Timing));
        reduce(&mut state, Action::SetDetailsTab(DetailsTab::Preview));
        assert_eq!(
            state
                .details
                .active_tab(&[DetailsTab::Summary, DetailsTab::Preview]),
            DetailsTab::Preview
        );
        assert_eq!(
            state.details.active_tab(&[
                DetailsTab::Summary,
                DetailsTab::Options,
                DetailsTab::Usage,
                DetailsTab::Timing,
            ]),
            DetailsTab::Timing
        );

        reduce(&mut state, Action::ToggleDetailsUnixTime);
        assert!(state.details.unix_time);
        reduce(&mut state, Action::SetDetailsTab(DetailsTab::Timing));
        assert!(state.details.unix_time);
        reduce(&mut state, Action::SetDetailsTab(DetailsTab::Summary));
        assert!(!state.details.unix_time);
    }

    #[test]
    fn restoring_an_empty_details_history_falls_back_to_summary() {
        let mut state = AppState::new(LayoutInput::default());
        reduce(
            &mut state,
            Action::RestoreSessionView {
                selected: None,
                details_tab_history: Vec::new(),
                follow_chat_tail: true,
            },
        );
        assert_eq!(state.details.tab_history, vec![DetailsTab::Summary]);
    }

    #[test]
    fn transient_notices_never_mutate_the_canonical_session_view() {
        let mut state = AppState::new(LayoutInput::default());
        let canonical = Arc::clone(&state.session_view);

        for index in 0..10 {
            reduce(
                &mut state,
                Action::AppendTransientNotice(Box::new(Message {
                    key: MessageId(index + 1),
                    revision: 0,
                    role: Role::Notice,
                    tool_call_id: None,
                    title: None,
                    text: format!("notice {index}"),
                    payload: None,
                    schema: None,
                    pending: false,
                    failed: false,
                    started_at_ms: None,
                    duration_ms: None,
                    turn: 0,
                    step: 0,
                    request_id: None,
                })),
            );
        }

        assert!(Arc::ptr_eq(&state.session_view, &canonical));
        assert_eq!(state.transient_messages.len(), 8);
        assert_eq!(state.transient_messages.front().unwrap().text, "notice 2");
        assert_eq!(state.transient_messages.back().unwrap().text, "notice 9");
    }
}
