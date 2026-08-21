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
    let stats = state.trajectory_data.stats();
    let mut groups = vec![format!(
        "{} {} · {} {}",
        stats.turns,
        if stats.turns == 1 { "turn" } else { "turns" },
        stats.steps,
        if stats.steps == 1 { "step" } else { "steps" },
    )];
    let mut durations = Vec::new();
    if stats.llm_ns > 0 {
        durations.push(format!("LLM {}", compact_duration(stats.llm_ns)));
    }
    if stats.tool_ns > 0 {
        durations.push(format!("Tools {}", compact_duration(stats.tool_ns)));
    }
    if !durations.is_empty() {
        groups.push(durations.join(" · "));
    }
    let mut speeds = Vec::new();
    if stats.ttft_steps > 0 {
        speeds.push(format!(
            "Avg TTFT {}",
            compact_duration(stats.ttft_ns / stats.ttft_steps as u64)
        ));
    }
    if stats.decode_ns > 0 {
        speeds.push(format!(
            "{} tok/s",
            compact_rate(stats.decode_tokens as f64 / (stats.decode_ns as f64 / 1_000_000_000.0))
        ));
    }
    if !speeds.is_empty() {
        groups.push(speeds.join(" · "));
    }
    if stats.input_tokens > 0 || stats.output_tokens > 0 {
        if let Some(percent) = stats
            .cached_tokens
            .saturating_mul(100)
            .saturating_add(stats.input_tokens / 2)
            .checked_div(stats.input_tokens)
        {
            groups.push(format!("Cache hit {percent}%"));
        }
        groups.push(format!(
            "Input {} tok · Output {} tok",
            compact_number(stats.input_tokens),
            compact_number(stats.output_tokens),
        ));
    }
    groups.join(" | ")
}

pub(crate) fn step_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|message| (message.turn, message.step))
        .filter(|(_, step)| *step > 0)
        .collect::<HashSet<_>>()
        .len()
}

fn compact_number(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f32 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f32 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn compact_duration(nanoseconds: u64) -> String {
    let seconds = nanoseconds as f64 / 1_000_000_000.0;
    if seconds < 60.0 {
        return format!("{:.1}s", (seconds * 10.0).round() / 10.0);
    }
    let seconds = seconds.round() as u64;
    format!("{}m{}s", seconds / 60, seconds % 60)
}

fn compact_rate(value: f64) -> String {
    if value >= 100.0 {
        format!("{value:.0}")
    } else {
        format!("{:.1}", (value * 10.0).round() / 10.0)
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
        assert_eq!(composer_status(&state), "0 turns · 0 steps");
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
