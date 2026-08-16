use std::collections::HashSet;

use crate::domain::{AppState, Message};
use crate::layout::HeightMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConversationViewModel<'a> {
    pub(crate) empty: bool,
    pub(crate) title: &'a str,
    pub(crate) turns: usize,
    pub(crate) steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EmptyConversationViewModel {
    pub(crate) show_intro: bool,
    pub(crate) show_workspace: bool,
    pub(crate) show_composer: bool,
}

pub(crate) fn conversation_view_model(state: &AppState) -> ConversationViewModel<'_> {
    ConversationViewModel {
        empty: state.conversation.messages.is_empty(),
        title: &state.conversation.title,
        turns: state.conversation.turns,
        steps: step_count(&state.conversation.messages),
    }
}

pub(crate) fn empty_conversation_view_model(state: &AppState) -> EmptyConversationViewModel {
    EmptyConversationViewModel {
        show_intro: state.layout.height != HeightMode::Compact,
        show_workspace: true,
        show_composer: true,
    }
}

pub(crate) fn composer_status(state: &AppState) -> String {
    let view = conversation_view_model(state);
    format!(
        "{} turns · {} steps   |   input {} · cached {} · output {} tokens",
        view.turns,
        view.steps,
        compact_number(state.conversation.input_tokens),
        compact_number(state.conversation.cached_tokens),
        compact_number(state.conversation.output_tokens),
    )
}

pub(crate) fn step_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|message| (message.turn, message.step))
        .filter(|(_, step)| *step > 0)
        .collect::<HashSet<_>>()
        .len()
}

fn compact_number(value: u32) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f32 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f32 / 1_000.0)
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutInput;

    #[test]
    fn empty_state_projects_to_stable_view_data() {
        let state = AppState::new(LayoutInput::default());
        let view = conversation_view_model(&state);
        assert!(view.empty);
        assert_eq!(view.title, "New chat");
        assert_eq!(
            composer_status(&state),
            "0 turns · 0 steps   |   input 0 · cached 0 · output 0 tokens"
        );
    }

    #[test]
    fn compact_empty_state_keeps_the_workspace_and_composer_visible() {
        let state = AppState::new(LayoutInput {
            viewport_height: 620.0,
            ..LayoutInput::default()
        });
        let view = empty_conversation_view_model(&state);
        assert!(!view.show_intro);
        assert!(view.show_workspace);
        assert!(view.show_composer);
    }

    #[test]
    fn token_counts_are_compact() {
        assert_eq!(compact_number(999), "999");
        assert_eq!(compact_number(1_250), "1.2K");
        assert_eq!(compact_number(2_500_000), "2.5M");
    }
}
