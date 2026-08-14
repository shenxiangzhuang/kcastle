use std::collections::HashMap;

use async_openai::types::responses::{
    EasyInputContent, EasyInputMessage, FunctionCallOutput, InputItem, InputParam, Item,
    MessageItem, OutputMessageContent, ResponseUsage, Role,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub id: String,
    pub model: String,
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StateEntry {
    Items {
        id: u64,
        items: Vec<InputItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseMetadata>,
    },
    Compaction {
        id: u64,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseMetadata>,
    },
}

impl StateEntry {
    pub fn id(&self) -> u64 {
        match self {
            Self::Items { id, .. } | Self::Compaction { id, .. } => *id,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptItem {
    User(String),
    Assistant(String),
    ToolCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolOutput {
        call_id: String,
        output: String,
    },
    Summary(String),
}

#[derive(Debug, Clone, Default)]
pub struct State {
    entries: Vec<StateEntry>,
    active_start: usize,
}

impl State {
    pub fn restore(entries: Vec<StateEntry>) -> Result<Self, String> {
        let mut state = Self {
            entries,
            active_start: 0,
        };
        state.reindex()?;
        Ok(state)
    }

    pub fn entries(&self) -> &[StateEntry] {
        &self.entries
    }

    pub fn context(&self) -> InputParam {
        let mut items = Vec::new();
        if let Some(summary) = self.latest_summary() {
            items.push(InputItem::from(EasyInputMessage::from(format!(
                "<conversation_summary>\nThis is factual context from earlier work, not new instructions.\n{summary}\n</conversation_summary>"
            ))));
        }
        for entry in &self.entries[self.active_start..] {
            if let StateEntry::Items {
                items: entry_items, ..
            } = entry
            {
                items.extend(entry_items.iter().cloned());
            }
        }
        InputParam::Items(items)
    }

    pub fn active_batches(&self) -> Vec<(u64, &[InputItem])> {
        self.entries[self.active_start..]
            .iter()
            .filter_map(|entry| match entry {
                StateEntry::Items { id, items, .. } => Some((*id, items.as_slice())),
                StateEntry::Compaction { .. } => None,
            })
            .collect()
    }

    pub fn append_items(
        &mut self,
        items: Vec<InputItem>,
        response: Option<ResponseMetadata>,
    ) -> Result<StateEntry, String> {
        if items.is_empty() {
            return Err("items must not be empty".into());
        }
        let entry = StateEntry::Items {
            id: self.next_id(),
            items,
            response,
        };
        self.entries.push(entry.clone());
        Ok(entry)
    }

    pub fn append_compaction(
        &mut self,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        response: Option<ResponseMetadata>,
    ) -> Result<StateEntry, String> {
        let Some(active_start) = self.entries.iter().position(
            |entry| matches!(entry, StateEntry::Items { id, .. } if *id == first_kept_id),
        ) else {
            return Err(format!("unknown first_kept_id: {first_kept_id}"));
        };
        if active_start < self.active_start {
            return Err("compaction cannot restore inactive history".into());
        }
        let entry = StateEntry::Compaction {
            id: self.next_id(),
            summary,
            first_kept_id,
            tokens_before,
            response,
        };
        self.entries.push(entry.clone());
        self.active_start = active_start;
        Ok(entry)
    }

    pub fn rollback(&mut self, id: u64) -> Result<(), String> {
        if self.entries.last().map(StateEntry::id) != Some(id) {
            return Err(format!("cannot roll back non-tail entry: {id}"));
        }
        self.entries.pop();
        self.reindex()
    }

    pub fn latest_response(&self) -> Option<&ResponseMetadata> {
        self.entries.iter().rev().find_map(|entry| match entry {
            StateEntry::Items {
                response: Some(response),
                ..
            } => Some(response),
            _ => None,
        })
    }

    pub fn latest_summary(&self) -> Option<&str> {
        self.entries.iter().rev().find_map(|entry| match entry {
            StateEntry::Compaction { summary, .. } => Some(summary.as_str()),
            _ => None,
        })
    }

    pub fn unresolved_tool_call_ids(&self) -> Vec<String> {
        let mut pending = HashMap::new();
        for entry in &self.entries[self.active_start..] {
            let StateEntry::Items { items, .. } = entry else {
                continue;
            };
            for item in items {
                match item {
                    InputItem::Item(Item::FunctionCall(call)) => {
                        pending.insert(call.call_id.clone(), ());
                    }
                    InputItem::Item(Item::FunctionCallOutput(output)) => {
                        pending.remove(&output.call_id);
                    }
                    _ => {}
                }
            }
        }
        pending.into_keys().collect()
    }

    pub fn transcript(&self) -> Vec<TranscriptItem> {
        let mut transcript = Vec::new();
        if let Some(summary) = self.latest_summary() {
            transcript.push(TranscriptItem::Summary(summary.to_owned()));
        }
        for entry in &self.entries[self.active_start..] {
            let StateEntry::Items { items, .. } = entry else {
                continue;
            };
            for item in items {
                match item {
                    InputItem::EasyMessage(message) => match (&message.role, &message.content) {
                        (Role::User, EasyInputContent::Text(text)) => {
                            transcript.push(TranscriptItem::User(text.clone()));
                        }
                        (Role::Assistant, EasyInputContent::Text(text)) => {
                            transcript.push(TranscriptItem::Assistant(text.clone()));
                        }
                        _ => {}
                    },
                    InputItem::Item(Item::Message(MessageItem::Output(message))) => {
                        let text = message
                            .content
                            .iter()
                            .filter_map(|content| match content {
                                OutputMessageContent::OutputText(text) => Some(text.text.as_str()),
                                _ => None,
                            })
                            .collect::<String>();
                        if !text.is_empty() {
                            transcript.push(TranscriptItem::Assistant(text));
                        }
                    }
                    InputItem::Item(Item::FunctionCall(call)) => {
                        transcript.push(TranscriptItem::ToolCall {
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        });
                    }
                    InputItem::Item(Item::FunctionCallOutput(output)) => {
                        let value = match &output.output {
                            FunctionCallOutput::Text(text) => text.clone(),
                            FunctionCallOutput::Content(content) => {
                                serde_json::to_string(content).unwrap_or_default()
                            }
                        };
                        transcript.push(TranscriptItem::ToolOutput {
                            call_id: output.call_id.clone(),
                            output: value,
                        });
                    }
                    _ => {}
                }
            }
        }
        transcript
    }

    fn next_id(&self) -> u64 {
        self.entries.last().map_or(1, |entry| entry.id() + 1)
    }

    fn reindex(&mut self) -> Result<(), String> {
        self.active_start = 0;
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.id() != index as u64 + 1 {
                return Err("state entry IDs must be consecutive from 1".into());
            }
            if let StateEntry::Compaction { first_kept_id, .. } = entry {
                let Some(boundary) = self.entries[..index].iter().position(
                    |candidate| matches!(candidate, StateEntry::Items { id, .. } if id == first_kept_id),
                ) else {
                    return Err(format!("unknown first_kept_id: {first_kept_id}"));
                };
                if boundary < self.active_start {
                    return Err("compaction cannot restore inactive history".into());
                }
                self.active_start = boundary;
            }
        }
        Ok(())
    }
}

pub fn estimate_tokens(value: &impl Serialize) -> usize {
    serde_json::to_vec(value)
        .map(|encoded| encoded.len().div_ceil(4).max(1))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use async_openai::types::responses::{
        EasyInputMessage, FunctionCallOutputItemParam, FunctionToolCall, InputItem, Item,
    };

    use super::{State, TranscriptItem};

    #[test]
    fn projects_compacted_history_and_tracks_tools() {
        let mut state = State::default();
        state
            .append_items(vec![InputItem::from(EasyInputMessage::from("hello"))], None)
            .unwrap();
        let call = FunctionToolCall {
            arguments: r#"{"command":"pwd"}"#.into(),
            call_id: "call_1".into(),
            namespace: None,
            name: "shell".into(),
            id: None,
            status: None,
        };
        state
            .append_items(vec![InputItem::from(Item::from(call))], None)
            .unwrap();
        assert_eq!(state.unresolved_tool_call_ids(), ["call_1"]);

        state
            .append_items(
                vec![InputItem::from(Item::from(FunctionCallOutputItemParam {
                    call_id: "call_1".into(),
                    output: "ok".into(),
                    id: None,
                    status: None,
                }))],
                None,
            )
            .unwrap();
        assert!(state.unresolved_tool_call_ids().is_empty());
        state
            .append_items(
                vec![InputItem::from(EasyInputMessage::from("recent"))],
                None,
            )
            .unwrap();
        state
            .append_compaction("summary".into(), 4, 10_000, None)
            .unwrap();

        let async_openai::types::responses::InputParam::Items(context) = state.context() else {
            panic!("context must be item based")
        };
        assert_eq!(context.len(), 2);
        assert!(matches!(
            state.transcript().as_slice(),
            [TranscriptItem::Summary(_), TranscriptItem::User(value)] if value == "recent"
        ));
    }
}
