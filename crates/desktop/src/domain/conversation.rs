use std::sync::Arc;

use im::Vector;

use crate::domain::MessageId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    User,
    Reasoning,
    Assistant,
    Tool,
    Notice,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    pub(crate) started_at_ms: Option<u128>,
    pub(crate) duration_ms: Option<u128>,
    pub(crate) turn: usize,
    pub(crate) step: usize,
    pub(crate) request_id: Option<String>,
    pub(crate) search_text: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ConversationState {
    pub(crate) messages: Vector<Arc<Message>>,
    pub(crate) title: String,
    pub(crate) turns: usize,
    pub(crate) tool_calls: usize,
}

impl Default for ConversationState {
    fn default() -> Self {
        Self {
            messages: Vector::new(),
            title: "New chat".into(),
            turns: 0,
            tool_calls: 0,
        }
    }
}

pub(crate) fn refresh_message_search_text(message: &mut Message) {
    // Conversation search has no consumer; trajectory search owns its own allocation-free
    // selectors. Keep the compatibility field empty instead of duplicating a growing streamed
    // response (and a temporary `join`) on every publication.
    message.search_text.clear();
}

#[cfg(test)]
fn reindex_messages(messages: &mut [Message]) {
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
