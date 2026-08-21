use std::sync::Arc;

use crate::domain::{Action, AppState, Effect, ScrollIntent, Surface, reduce_conversation};
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
            state.details.selected = selected;
            state.layout_input.details_visible = selected.is_some();
            recompute_layout(state, &mut effects);
        }
        Action::SetDetailsTab(tab) => state.details.tab = tab,
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
        Action::ToggleTimelineUnixTime => {
            state.trajectory.unix_time = !state.trajectory.unix_time;
        }
        Action::ToggleTrajectoryTurns => {
            state.trajectory.collapsed_turns = !state.trajectory.collapsed_turns;
        }
        Action::ToggleTrajectoryCalls => {
            state.trajectory.collapsed_calls = !state.trajectory.collapsed_calls;
        }
        Action::ExpandTrajectoryGroups => {
            state.trajectory.collapsed_turns = false;
            state.trajectory.collapsed_calls = false;
        }
        Action::RestoreSessionView {
            selected,
            details_tab,
            follow_chat_tail,
        } => {
            state.details.selected = selected;
            state.details.tab = details_tab;
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
            sessions,
        } => {
            state.workspace.active_project = index;
            state.workspace.expanded_projects.insert(cwd.clone());
            state.workspace.cwd = cwd;
            state.workspace.sessions_dir = sessions_dir;
            state.session.sessions = sessions;
        }
        Action::SetActiveProject(index) => state.workspace.active_project = index,
        Action::RefreshSessions(sessions) => state.session.sessions = sessions,
        Action::Conversation(action) => {
            let expanded = reduce_conversation(Arc::make_mut(&mut state.conversation), *action);
            if expanded && state.follow_chat_tail {
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

    use super::*;
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
}
