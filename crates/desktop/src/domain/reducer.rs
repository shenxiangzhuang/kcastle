use crate::domain::{
    Action, AppState, Effect, PendingSessionOperation, RunState, ScrollIntent,
    SessionOperationKind, Surface, reduce_conversation,
};
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
        Action::SetApproval(approval) => state.approval = approval,
        Action::SetComposerMenu(menu) => {
            state.composer.menu = menu;
            state.composer.highlighted_item = 0;
            state.sidebar.options_open = false;
            state.sidebar.session_action_target = None;
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
                state.sidebar.session_action_target = None;
            }
        }
        Action::ToggleSidebarOptions => {
            state.sidebar.options_open = !state.sidebar.options_open;
            state.sidebar.session_action_target = None;
            if state.sidebar.options_open {
                state.composer.menu = None;
            }
        }
        Action::CloseTransientOverlays => {
            state.composer.menu = None;
            state.sidebar.options_open = false;
            state.sidebar.session_action_target = None;
        }
        Action::DismissTransient => {
            state.composer.menu = match state.composer.menu {
                Some(crate::domain::ComposerMenu::Models | crate::domain::ComposerMenu::Effort) => {
                    Some(crate::domain::ComposerMenu::Model)
                }
                _ => None,
            };
            state.sidebar.options_open = false;
            state.sidebar.session_action_target = None;
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
        Action::SetSessionActionTarget(target) => {
            state.sidebar.session_action_target = target;
        }
        Action::ToggleProjectExpanded(path) => {
            if !state.workspace.expanded_projects.remove(&path) {
                state.workspace.expanded_projects.insert(path);
            }
        }
        Action::ExpandProject(path) => {
            state.workspace.expanded_projects.insert(path);
        }
        Action::ToggleTrajectoryDuration => {
            state.trajectory.show_duration = !state.trajectory.show_duration;
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
        Action::SetCurrentSession(path) => state.session.current = path,
        Action::BeginOpenSession(path) => {
            if state.pending_session_operation.is_none()
                && !matches!(
                    state.run,
                    RunState::CreatingSession { .. } | RunState::Running { .. }
                )
                && path != state.session.current
            {
                let operation = state.next_operation.next();
                state.next_operation = operation;
                state.last_error = None;
                state.pending_session_operation = Some(PendingSessionOperation {
                    operation,
                    kind: SessionOperationKind::Open { path: path.clone() },
                });
                effects.push(Effect::OpenSession { operation, path });
            }
        }
        Action::SessionOpened {
            operation,
            conversation,
            current_session,
            sessions,
        } => {
            let accepted = matches!(
                state.pending_session_operation.as_ref(),
                Some(PendingSessionOperation {
                    operation: active,
                    kind: SessionOperationKind::Open { path },
                }) if *active == operation && *path == current_session
            );
            if accepted {
                state.pending_session_operation = None;
                state.conversation = conversation;
                state.session.current = current_session;
                state.session.sessions = sessions;
                reset_navigation(state, &mut effects);
                invalidate_measurements(state, &mut effects);
            }
        }
        Action::BeginRenameSession(title) => {
            if state.pending_session_operation.is_none()
                && !state.session.current.as_os_str().is_empty()
                && !matches!(
                    state.run,
                    RunState::CreatingSession { .. } | RunState::Running { .. }
                )
            {
                let operation = state.next_operation.next();
                state.next_operation = operation;
                state.last_error = None;
                state.pending_session_operation = Some(PendingSessionOperation {
                    operation,
                    kind: SessionOperationKind::Rename {
                        path: state.session.current.clone(),
                        title: title.clone(),
                    },
                });
                effects.push(Effect::RenameSession { operation, title });
            }
        }
        Action::SessionRenamed {
            operation,
            title,
            sessions,
        } => {
            let accepted = matches!(
                state.pending_session_operation.as_ref(),
                Some(PendingSessionOperation {
                    operation: active,
                    kind: SessionOperationKind::Rename { .. },
                }) if *active == operation
            );
            if accepted {
                state.pending_session_operation = None;
                state.conversation.title = title;
                state.session.sessions = sessions;
            }
        }
        Action::SessionOperationFailed { operation, message } => {
            if state
                .pending_session_operation
                .as_ref()
                .is_some_and(|pending| pending.operation == operation)
            {
                state.pending_session_operation = None;
                state.last_error = Some(message);
            }
        }
        Action::ReplaceConversation {
            conversation,
            current_session,
            sessions,
        } => {
            state.conversation = conversation;
            state.session.current = current_session;
            state.session.sessions = sessions;
            reset_navigation(state, &mut effects);
            invalidate_measurements(state, &mut effects);
        }
        Action::ResetConversation => {
            state.conversation = Default::default();
            state.session.current.clear();
            reset_navigation(state, &mut effects);
            invalidate_measurements(state, &mut effects);
        }
        Action::Conversation(action) => {
            let expanded = reduce_conversation(&mut state.conversation, action);
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
        Action::BeginSessionCreation(input) => {
            let operation = state.next_operation.next();
            state.next_operation = operation;
            state.run = RunState::CreatingSession {
                operation,
                input: input.clone(),
            };
            effects.push(Effect::CreateSession { operation, input });
        }
        Action::SessionCreationFailed { operation, message } => {
            if matches!(state.run, RunState::CreatingSession { operation: active, .. } if active == operation)
            {
                state.run = RunState::Failed {
                    operation: Some(operation),
                    message,
                };
            }
        }
        Action::SessionCreated {
            operation,
            current_session,
            sessions,
        } => {
            let input = match &state.run {
                RunState::CreatingSession {
                    operation: active,
                    input,
                } if *active == operation => Some(input.clone()),
                _ => None,
            };
            if let Some(input) = input {
                state.session.current = current_session;
                state.session.sessions = sessions;
                let run = state.next_run.next();
                state.next_run = run;
                state.run = RunState::Running { run };
                effects.push(Effect::StartRun { run, input });
            }
        }
        Action::BeginRun(input) => {
            if matches!(state.run, RunState::Idle | RunState::Failed { .. }) {
                let run = state.next_run.next();
                state.next_run = run;
                state.run = RunState::Running { run };
                effects.push(Effect::StartRun { run, input });
            }
        }
        Action::RunStartFailed { run, message } => {
            if matches!(state.run, RunState::Running { run: active } if active == run) {
                state.run = RunState::Failed {
                    operation: None,
                    message,
                };
            }
        }
        Action::RunFinished(run) => {
            if matches!(state.run, RunState::Running { run: active } if active == run) {
                state.run = RunState::Idle;
                state.approval = None;
            }
        }
    }
    effects
}

fn reset_navigation(state: &mut AppState, effects: &mut Vec<Effect>) {
    state.composer = Default::default();
    state.surface = Surface::Chat;
    state.details = Default::default();
    state.approval = None;
    state.trajectory = Default::default();
    state.follow_chat_tail = true;
    state.unread_stream_updates = 0;
    state.layout_input.trajectory_visible = false;
    state.layout_input.details_visible = false;
    recompute_layout(state, effects);
}

fn invalidate_measurements(state: &mut AppState, effects: &mut Vec<Effect>) {
    state.layout_generation = state.layout_generation.next();
    if state.follow_chat_tail && !effects.contains(&Effect::ApplyChatTail) {
        effects.push(Effect::ApplyChatTail);
    }
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
    use super::*;
    use crate::domain::{ConversationState, OperationId, RunId};
    use crate::layout::LayoutInput;
    use proptest::prelude::*;

    #[test]
    fn stale_session_failure_cannot_replace_current_state() {
        let mut state = AppState::new(LayoutInput::default());
        let effects = reduce(&mut state, Action::BeginSessionCreation("hello".into()));
        let operation = match effects.as_slice() {
            [Effect::CreateSession { operation, .. }] => *operation,
            effects => panic!("unexpected effects: {effects:?}"),
        };
        reduce(
            &mut state,
            Action::SessionCreationFailed {
                operation: OperationId(operation.0 + 1),
                message: "stale".into(),
            },
        );
        assert!(matches!(state.run, RunState::CreatingSession { .. }));
    }

    #[test]
    fn only_the_matching_session_creation_can_start_its_run() {
        let mut state = AppState::new(LayoutInput::default());
        let effects = reduce(&mut state, Action::BeginSessionCreation("hello".into()));
        let operation = match effects.as_slice() {
            [Effect::CreateSession { operation, .. }] => *operation,
            effects => panic!("unexpected effects: {effects:?}"),
        };
        let stale_effects = reduce(
            &mut state,
            Action::SessionCreated {
                operation: OperationId(operation.0 + 1),
                current_session: "stale.jsonl".into(),
                sessions: Vec::new(),
            },
        );
        assert!(stale_effects.is_empty());
        assert!(state.session.current.as_os_str().is_empty());
        assert!(matches!(state.run, RunState::CreatingSession { .. }));

        let effects = reduce(
            &mut state,
            Action::SessionCreated {
                operation,
                current_session: "current.jsonl".into(),
                sessions: Vec::new(),
            },
        );
        assert!(matches!(effects.as_slice(), [Effect::StartRun { input, .. }] if input == "hello"));
        assert_eq!(state.session.current, std::path::Path::new("current.jsonl"));
        assert!(matches!(state.run, RunState::Running { .. }));
    }

    #[test]
    fn stale_run_completion_is_ignored() {
        let mut state = AppState::new(LayoutInput::default());
        let effects = reduce(&mut state, Action::BeginRun("hello".into()));
        let run = match effects.as_slice() {
            [Effect::StartRun { run, .. }] => *run,
            effects => panic!("unexpected effects: {effects:?}"),
        };
        reduce(&mut state, Action::RunFinished(RunId(run.0 + 1)));
        assert_eq!(state.run, RunState::Running { run });
        reduce(&mut state, Action::RunFinished(run));
        assert_eq!(state.run, RunState::Idle);
    }

    #[test]
    fn user_scroll_disables_follow_until_tail_is_reached() {
        let mut state = AppState::new(LayoutInput::default());
        reduce(&mut state, Action::Scroll(ScrollIntent::Away));
        assert!(!state.follow_chat_tail);
        assert!(reduce(&mut state, Action::StreamDeltasReceived(3)).is_empty());
        assert_eq!(state.unread_stream_updates, 3);
        reduce(
            &mut state,
            Action::Scroll(ScrollIntent::Toward { at_tail: true }),
        );
        assert!(state.follow_chat_tail);
        assert_eq!(state.unread_stream_updates, 0);
    }

    #[test]
    fn stale_session_open_result_cannot_replace_the_current_conversation() {
        let mut state = AppState::new(LayoutInput::default());
        let effects = reduce(&mut state, Action::BeginOpenSession("wanted.jsonl".into()));
        let operation = match effects.as_slice() {
            [Effect::OpenSession { operation, .. }] => *operation,
            effects => panic!("unexpected effects: {effects:?}"),
        };
        let stale = ConversationState {
            title: "stale".into(),
            ..ConversationState::default()
        };
        reduce(
            &mut state,
            Action::SessionOpened {
                operation: OperationId(operation.0 + 1),
                conversation: stale,
                current_session: "stale.jsonl".into(),
                sessions: Vec::new(),
            },
        );
        assert_eq!(state.conversation.title, "New chat");
        assert!(state.pending_session_operation.is_some());
    }

    #[test]
    fn accepted_session_open_invalidates_old_layout_measurements() {
        let mut state = AppState::new(LayoutInput::default());
        let effects = reduce(&mut state, Action::BeginOpenSession("wanted.jsonl".into()));
        let operation = match effects.as_slice() {
            [Effect::OpenSession { operation, .. }] => *operation,
            effects => panic!("unexpected effects: {effects:?}"),
        };
        let generation = state.layout_generation;
        reduce(
            &mut state,
            Action::SessionOpened {
                operation,
                conversation: ConversationState::default(),
                current_session: "wanted.jsonl".into(),
                sessions: Vec::new(),
            },
        );
        assert!(state.pending_session_operation.is_none());
        assert_ne!(state.layout_generation, generation);
    }

    #[test]
    fn conversation_replacement_requests_exactly_one_tail_restore() {
        let mut state = AppState::new(LayoutInput::default());
        reduce(&mut state, Action::ShowTrajectory);
        reduce(
            &mut state,
            Action::SelectDetails(Some(crate::domain::MessageId(7))),
        );
        let effects = reduce(
            &mut state,
            Action::ReplaceConversation {
                conversation: ConversationState::default(),
                current_session: "next.jsonl".into(),
                sessions: Vec::new(),
            },
        );
        assert_eq!(
            effects
                .iter()
                .filter(|effect| matches!(effect, Effect::ApplyChatTail))
                .count(),
            1
        );
    }

    #[test]
    fn measurement_invalidation_requests_tail_when_layout_did_not_change() {
        let mut state = AppState::new(LayoutInput::default());
        let effects = reduce(
            &mut state,
            Action::ReplaceConversation {
                conversation: ConversationState::default(),
                current_session: "next.jsonl".into(),
                sessions: Vec::new(),
            },
        );
        assert_eq!(effects, vec![Effect::ApplyChatTail]);
    }

    proptest! {
        #[test]
        fn arbitrary_ui_action_sequences_preserve_state_layout_invariants(
            actions in prop::collection::vec((0u8..12, any::<bool>(), 0u16..4_000), 0..500)
        ) {
            let mut state = AppState::new(LayoutInput::default());
            for (kind, flag, value) in actions {
                let action = match kind {
                    0 => Action::ToggleSidebar,
                    1 => if flag { Action::ShowTrajectory } else { Action::ShowChat },
                    2 => Action::SelectDetails(flag.then_some(crate::domain::MessageId(value as u64))),
                    3 => Action::Scroll(if flag { ScrollIntent::Away } else { ScrollIntent::JumpToTail }),
                    4 => Action::ToggleSessionSearch,
                    5 => Action::ToggleSidebarOptions,
                    6 => Action::ToggleTrajectoryDuration,
                    7 => Action::ToggleTrajectoryTurns,
                    8 => Action::ToggleTrajectoryCalls,
                    9 => Action::SetSidebarGrouping(flag),
                    10 => Action::SetSidebarSort(flag),
                    _ => {
                        let mut input = state.layout_input;
                        input.viewport_width = value as f32;
                        input.viewport_height = (value / 2) as f32;
                        Action::LayoutInputChanged(input)
                    }
                };
                reduce(&mut state, action);
                prop_assert_eq!(state.layout_input.sidebar_requested, state.sidebar_requested);
                prop_assert_eq!(state.layout_input.trajectory_visible, state.surface == Surface::Trajectory);
                prop_assert_eq!(state.layout_input.details_visible, state.details.selected.is_some());
                prop_assert!(state.layout.sidebar_width >= 0.0);
                prop_assert!(state.layout.main_width >= 0.0);
                prop_assert!(state.layout.sidebar_width + state.layout.main_width <= state.layout_input.viewport_width.max(0.0) + f32::EPSILON);
            }
        }
    }
}
