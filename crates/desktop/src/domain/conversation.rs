use crate::domain::MessageId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    User,
    Reasoning,
    Assistant,
    Tool,
    Notice,
}

#[derive(Clone, Debug)]
pub(crate) struct Message {
    pub(crate) key: MessageId,
    pub(crate) revision: u64,
    pub(crate) role: Role,
    pub(crate) tool_call_id: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) text: String,
    pub(crate) payload: Option<String>,
    pub(crate) schema: Option<String>,
    pub(crate) pending: bool,
    pub(crate) failed: bool,
    pub(crate) expanded: bool,
    pub(crate) rating: Option<bool>,
    pub(crate) started_at_ms: Option<u128>,
    pub(crate) duration_ms: Option<u128>,
    pub(crate) turn: usize,
    pub(crate) step: usize,
    pub(crate) request_id: Option<String>,
    pub(crate) search_text: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ConversationState {
    pub(crate) messages: Vec<Message>,
    pub(crate) title: String,
    pub(crate) turns: usize,
    pub(crate) tool_calls: usize,
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) cached_tokens: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UsageSnapshot {
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) cached_tokens: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum ConversationAction {
    SubmitUser(Message),
    RollbackSubmittedUser,
    AppendNotice(Message),
    TextDelta {
        delta: String,
        new_message: Message,
    },
    ReasoningDelta {
        delta: String,
        new_message: Message,
    },
    ToolStarted(Message),
    ToolFinished {
        call_id: String,
        output: String,
        is_error: bool,
        duration_ms: Option<u128>,
    },
    RunFinished {
        response_id: String,
        usage: Option<UsageSnapshot>,
    },
    FinishReasoning,
    RefreshLiveSearch,
    ToggleExpanded {
        index: usize,
        role: Role,
    },
    RateAssistant {
        index: usize,
        positive: bool,
    },
}

impl Default for ConversationState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            title: "New chat".into(),
            turns: 0,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
        }
    }
}

pub(crate) fn reduce_conversation(
    state: &mut ConversationState,
    action: ConversationAction,
) -> bool {
    let mut request_tail = false;
    match action {
        ConversationAction::SubmitUser(message) => {
            let title = (state.title == "New chat").then(|| short_title(&message.text));
            state.messages.push(message);
            reindex_messages(&mut state.messages);
            state.turns = state
                .messages
                .iter()
                .filter(|message| message.role == Role::User)
                .count();
            if let Some(title) = title {
                state.title = title;
            }
        }
        ConversationAction::RollbackSubmittedUser => {
            if state
                .messages
                .last()
                .is_some_and(|message| message.role == Role::User)
            {
                state.messages.pop();
                reindex_messages(&mut state.messages);
                state.turns = state
                    .messages
                    .iter()
                    .filter(|message| message.role == Role::User)
                    .count();
            }
            if state.messages.is_empty() {
                state.title = "New chat".into();
            }
        }
        ConversationAction::AppendNotice(message) => {
            state.messages.push(message);
            reindex_messages(&mut state.messages);
        }
        ConversationAction::TextDelta {
            delta,
            mut new_message,
        } => {
            if let Some(reasoning) = state
                .messages
                .last_mut()
                .filter(|message| message.role == Role::Reasoning)
            {
                reasoning.pending = false;
            }
            if let Some(message) = state
                .messages
                .last_mut()
                .filter(|message| message.role == Role::Assistant)
            {
                message.text.push_str(&delta);
                message.revision = message.revision.saturating_add(1);
            } else {
                new_message.pending = true;
                state.messages.push(new_message);
            }
        }
        ConversationAction::ReasoningDelta {
            delta,
            mut new_message,
        } => {
            if let Some(assistant) = state
                .messages
                .last_mut()
                .filter(|message| message.role == Role::Assistant)
            {
                assistant.pending = false;
            }
            if let Some(message) = state
                .messages
                .last_mut()
                .filter(|message| message.role == Role::Reasoning)
            {
                message.text.push_str(&delta);
                message.revision = message.revision.saturating_add(1);
            } else {
                new_message.title = Some("Think".into());
                new_message.pending = true;
                state.messages.push(new_message);
            }
        }
        ConversationAction::ToolStarted(message) => {
            settle_active_response_message(&mut state.messages);
            state.tool_calls = state.tool_calls.saturating_add(1);
            state.messages.push(message);
        }
        ConversationAction::ToolFinished {
            call_id,
            output,
            is_error,
            duration_ms,
        } => {
            if let Some(message) = state
                .messages
                .iter_mut()
                .rev()
                .find(|message| message.tool_call_id.as_deref() == Some(&call_id))
            {
                message.text = output;
                message.revision = message.revision.saturating_add(1);
                message.pending = false;
                message.failed = is_error;
                message.duration_ms = duration_ms;
                refresh_message_search_text(message);
            }
        }
        ConversationAction::RunFinished { response_id, usage } => {
            for message in state.messages.iter_mut().rev() {
                if message.role == Role::User {
                    break;
                }
                if message.request_id.is_none()
                    || message
                        .request_id
                        .as_deref()
                        .is_some_and(|id| id.starts_with("turn-"))
                {
                    message.request_id = Some(response_id.clone());
                }
                if matches!(message.role, Role::Reasoning | Role::Assistant) {
                    message.pending = false;
                }
            }
            if let Some(usage) = usage {
                state.input_tokens = usage.input_tokens;
                state.output_tokens = usage.output_tokens;
                state.cached_tokens = usage.cached_tokens;
            }
        }
        ConversationAction::FinishReasoning => {
            for message in state.messages.iter_mut().rev() {
                if message.role == Role::User {
                    break;
                }
                if matches!(message.role, Role::Reasoning | Role::Assistant) {
                    message.pending = false;
                }
            }
        }
        ConversationAction::RefreshLiveSearch => refresh_live_search(&mut state.messages),
        ConversationAction::ToggleExpanded { index, role } => {
            if let Some(message) = state.messages.get_mut(index)
                && message.role == role
            {
                message.expanded = !message.expanded;
                request_tail = message.expanded;
            }
        }
        ConversationAction::RateAssistant { index, positive } => {
            if let Some(message) = state.messages.get_mut(index)
                && message.role == Role::Assistant
            {
                message.rating = (message.rating != Some(positive)).then_some(positive);
            }
        }
    }
    request_tail
}

fn settle_active_response_message(messages: &mut [Message]) {
    if let Some(message) = messages
        .last_mut()
        .filter(|message| matches!(message.role, Role::Reasoning | Role::Assistant))
    {
        message.pending = false;
    }
}

fn refresh_live_search(messages: &mut [Message]) {
    for message in messages.iter_mut().rev() {
        if message.role == Role::User {
            break;
        }
        if matches!(message.role, Role::Reasoning | Role::Assistant) {
            refresh_message_search_text(message);
        }
    }
}

pub(crate) fn refresh_message_search_text(message: &mut Message) {
    message.search_text = [
        message.title.as_deref().unwrap_or_default(),
        message.payload.as_deref().unwrap_or_default(),
        message.schema.as_deref().unwrap_or_default(),
        message.text.as_str(),
    ]
    .join("\n")
    .to_lowercase();
}

pub(crate) fn reindex_messages(messages: &mut [Message]) {
    let mut turn = 0;
    let mut step = 0;
    let mut assistant_phase = false;
    for message in messages {
        match message.role {
            Role::User => {
                turn += 1;
                step = 0;
                assistant_phase = false;
            }
            Role::Reasoning | Role::Assistant => {
                if !assistant_phase {
                    step += 1;
                    assistant_phase = true;
                }
            }
            Role::Tool | Role::Notice => assistant_phase = false,
        }
        message.turn = turn;
        message.step = step;
        if message.request_id.is_none() && turn > 0 {
            message.request_id = Some(if step > 0 {
                format!("turn-{turn}-step-{step}")
            } else {
                format!("turn-{turn}")
            });
        }
        refresh_message_search_text(message);
    }
}

fn short_title(value: &str) -> String {
    const LIMIT: usize = 42;
    let mut chars = value.chars();
    let title: String = chars.by_ref().take(LIMIT).collect();
    if chars.next().is_some() {
        format!("{}…", title.trim_end())
    } else {
        title
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn message(id: u64, role: Role, text: impl Into<String>) -> Message {
        Message {
            key: MessageId(id),
            revision: 0,
            role,
            tool_call_id: None,
            title: None,
            text: text.into(),
            payload: None,
            schema: None,
            pending: false,
            failed: false,
            expanded: false,
            rating: None,
            started_at_ms: None,
            duration_ms: None,
            turn: 0,
            step: 0,
            request_id: None,
            search_text: String::new(),
        }
    }

    proptest! {
        #[test]
        fn text_stream_is_independent_of_chunk_boundaries(chunks in prop::collection::vec("[^\\PC]*", 1..40)) {
            let chunk_count = chunks.len();
            let expected = chunks.concat();
            let mut state = ConversationState::default();
            for (index, chunk) in chunks.into_iter().enumerate() {
                reduce_conversation(
                    &mut state,
                    ConversationAction::TextDelta {
                        delta: chunk.clone(),
                        new_message: message(index as u64 + 1, Role::Assistant, chunk),
                    },
                );
            }
            prop_assert_eq!(state.messages.len(), 1);
            prop_assert_eq!(&state.messages[0].text, &expected);
            prop_assert_eq!(state.messages[0].revision as usize, chunk_count - 1);
        }

        #[test]
        fn reindexing_never_moves_turns_backwards(roles in prop::collection::vec(0u8..5, 0..200)) {
            let mut messages = roles
                .into_iter()
                .enumerate()
                .map(|(index, role)| message(index as u64 + 1, match role {
                    0 => Role::User,
                    1 => Role::Reasoning,
                    2 => Role::Assistant,
                    3 => Role::Tool,
                    _ => Role::Notice,
                }, "x"))
                .collect::<Vec<_>>();
            reindex_messages(&mut messages);
            prop_assert!(messages.windows(2).all(|pair| pair[0].turn <= pair[1].turn));
            prop_assert!(messages.iter().all(|message| message.turn > 0 || message.request_id.is_none()));
        }
    }
}
