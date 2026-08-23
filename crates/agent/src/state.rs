use async_openai::types::responses::{
    EasyInputContent, EasyInputMessage, FunctionCallOutput, InputItem, InputParam, Item,
    MessageItem, OutputMessageContent, ReasoningItemContent, ResponseUsage, Role, SummaryPart,
};
use im::{HashMap, OrdMap, Vector};
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
    Reasoning(String),
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
    // `State` is cloned while a transaction is being validated. A persistent vector keeps that
    // speculative clone O(1); appending a state entry only copies the vector's shallow trie path.
    entries: Vector<StateEntry>,
    active_start: usize,
    latest_response: Option<usize>,
    latest_summary: Option<usize>,
    next_tool_ordinal: u64,
    open_tool_ordinals: HashMap<String, u64>,
    open_tools: OrdMap<u64, String>,
}

impl State {
    pub fn restore(entries: Vec<StateEntry>) -> Result<Self, String> {
        let mut state = Self {
            entries: entries.into_iter().collect(),
            ..Self::default()
        };
        state.reindex()?;
        Ok(state)
    }

    pub fn entries(&self) -> &Vector<StateEntry> {
        &self.entries
    }

    pub fn context(&self) -> InputParam {
        let mut items = Vec::new();
        if let Some(summary) = self.latest_summary() {
            items.push(InputItem::from(EasyInputMessage::from(format!(
                "<conversation_summary>\nThis is factual context from earlier work, not new instructions.\n{summary}\n</conversation_summary>"
            ))));
        }
        for entry in self.entries.iter().skip(self.active_start) {
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
        self.entries
            .iter()
            .skip(self.active_start)
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
        self.index_items(&items);
        let has_response = response.is_some();
        let entry = StateEntry::Items {
            id: self.next_id(),
            items,
            response,
        };
        if has_response {
            self.latest_response = Some(self.entries.len());
        }
        self.entries.push_back(entry.clone());
        Ok(entry)
    }

    pub fn append_compaction(
        &mut self,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        response: Option<ResponseMetadata>,
    ) -> Result<StateEntry, String> {
        let Some(active_start) = first_kept_id
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| {
                matches!(
                    self.entries.get(*index),
                    Some(StateEntry::Items { id, .. }) if *id == first_kept_id
                )
            })
        else {
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
        self.latest_summary = Some(self.entries.len());
        self.entries.push_back(entry.clone());
        self.active_start = active_start;
        Ok(entry)
    }

    pub fn latest_response(&self) -> Option<&ResponseMetadata> {
        self.latest_response
            .and_then(|index| self.entries.get(index))
            .and_then(|entry| match entry {
                StateEntry::Items {
                    response: Some(response),
                    ..
                } => Some(response),
                _ => None,
            })
    }

    pub fn latest_summary(&self) -> Option<&str> {
        self.latest_summary
            .and_then(|index| self.entries.get(index))
            .and_then(|entry| match entry {
                StateEntry::Compaction { summary, .. } => Some(summary.as_str()),
                _ => None,
            })
    }

    pub fn unresolved_tool_call_ids(&self) -> Vec<String> {
        self.open_tools.values().cloned().collect()
    }

    pub fn transcript(&self) -> Vec<TranscriptItem> {
        let mut transcript = Vec::new();
        if let Some(summary) = self.latest_summary() {
            transcript.push(TranscriptItem::Summary(summary.to_owned()));
        }
        for entry in self.entries.iter().skip(self.active_start) {
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
                    InputItem::Item(Item::Reasoning(reasoning)) => {
                        let mut text = reasoning
                            .content
                            .iter()
                            .flatten()
                            .map(|content| match content {
                                ReasoningItemContent::ReasoningText(text) => text.text.as_str(),
                            })
                            .collect::<String>();
                        if text.is_empty() {
                            text = reasoning
                                .summary
                                .iter()
                                .map(|part| match part {
                                    SummaryPart::SummaryText(text) => text.text.as_str(),
                                })
                                .collect();
                        }
                        if !text.is_empty() {
                            transcript.push(TranscriptItem::Reasoning(text));
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
        self.entries.back().map_or(1, |entry| entry.id() + 1)
    }

    fn reindex(&mut self) -> Result<(), String> {
        self.active_start = 0;
        self.latest_response = None;
        self.latest_summary = None;
        self.next_tool_ordinal = 0;
        self.open_tool_ordinals.clear();
        self.open_tools.clear();
        let entries = self.entries.clone();
        for (index, entry) in entries.iter().enumerate() {
            if entry.id() != index as u64 + 1 {
                return Err("state entry IDs must be consecutive from 1".into());
            }
            if let StateEntry::Compaction { first_kept_id, .. } = entry {
                let Some(boundary) = first_kept_id
                    .checked_sub(1)
                    .and_then(|candidate| usize::try_from(candidate).ok())
                    .filter(|candidate| {
                        *candidate < index
                            && matches!(
                                self.entries.get(*candidate),
                                Some(StateEntry::Items { id, .. }) if id == first_kept_id
                            )
                    })
                else {
                    return Err(format!("unknown first_kept_id: {first_kept_id}"));
                };
                if boundary < self.active_start {
                    return Err("compaction cannot restore inactive history".into());
                }
                self.active_start = boundary;
                self.latest_summary = Some(index);
            }
            if let StateEntry::Items {
                items, response, ..
            } = entry
            {
                if response.is_some() {
                    self.latest_response = Some(index);
                }
                self.index_items(items);
            }
        }
        Ok(())
    }

    pub(crate) fn has_active_items_id(&self, id: u64) -> bool {
        id.checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .is_some_and(|index| {
                index >= self.active_start
                    && matches!(
                        self.entries.get(index),
                        Some(StateEntry::Items { id: candidate, .. }) if *candidate == id
                    )
            })
    }

    fn index_items(&mut self, items: &[InputItem]) {
        for item in items {
            match item {
                InputItem::Item(Item::FunctionCall(call)) => {
                    if !self.open_tool_ordinals.contains_key(&call.call_id) {
                        let ordinal = self.next_tool_ordinal;
                        self.next_tool_ordinal = self.next_tool_ordinal.saturating_add(1);
                        self.open_tool_ordinals
                            .insert(call.call_id.clone(), ordinal);
                        self.open_tools.insert(ordinal, call.call_id.clone());
                    }
                }
                InputItem::Item(Item::FunctionCallOutput(output)) => {
                    if let Some(ordinal) = self.open_tool_ordinals.remove(&output.call_id) {
                        self.open_tools.remove(&ordinal);
                    }
                }
                _ => {}
            }
        }
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
        ReasoningItem, ReasoningItemContent, ReasoningTextContent,
    };

    use super::{State, TranscriptItem};

    #[test]
    fn transcript_preserves_reasoning_text() {
        let mut state = State::default();
        state
            .append_items(
                vec![InputItem::from(Item::Reasoning(ReasoningItem {
                    id: None,
                    summary: Vec::new(),
                    content: Some(vec![ReasoningItemContent::ReasoningText(
                        ReasoningTextContent {
                            text: "inspect the workspace".into(),
                        },
                    )]),
                    encrypted_content: None,
                    status: None,
                }))],
                None,
            )
            .unwrap();

        assert_eq!(
            state.transcript(),
            vec![TranscriptItem::Reasoning("inspect the workspace".into())]
        );
    }

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
