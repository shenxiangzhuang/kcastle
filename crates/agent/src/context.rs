use async_openai::types::responses::{
    EasyInputMessage, InputItem, InputParam, Item, ResponseUsage,
};
use im::{HashMap, OrdMap, Vector};
use serde::{Deserialize, Serialize};

pub(crate) mod compaction;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub id: String,
    pub model: String,
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextEntry {
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

impl ContextEntry {
    pub fn id(&self) -> u64 {
        match self {
            Self::Items { id, .. } | Self::Compaction { id, .. } => *id,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ContextState {
    // `ContextState` is cloned while a transaction is being validated. A persistent vector keeps that
    // speculative clone O(1); appending a state entry only copies the vector's shallow trie path.
    entries: Vector<ContextEntry>,
    active_start: usize,
    latest_summary: Option<usize>,
    next_tool_ordinal: u64,
    open_tool_ordinals: HashMap<String, u64>,
    open_tools: OrdMap<u64, String>,
}

impl ContextState {
    #[cfg(test)]
    pub fn entries(&self) -> &Vector<ContextEntry> {
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
            if let ContextEntry::Items {
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
                ContextEntry::Items { id, items, .. } => Some((*id, items.as_slice())),
                ContextEntry::Compaction { .. } => None,
            })
            .collect()
    }

    pub fn append_items(
        &mut self,
        items: Vec<InputItem>,
        response: Option<ResponseMetadata>,
    ) -> Result<ContextEntry, String> {
        if items.is_empty() {
            return Err("items must not be empty".into());
        }
        self.index_items(&items);
        let entry = ContextEntry::Items {
            id: self.next_id(),
            items,
            response,
        };
        self.entries.push_back(entry.clone());
        Ok(entry)
    }

    pub fn append_compaction(
        &mut self,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        response: Option<ResponseMetadata>,
    ) -> Result<ContextEntry, String> {
        let Some(active_start) = first_kept_id
            .checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .filter(|index| {
                matches!(
                    self.entries.get(*index),
                    Some(ContextEntry::Items { id, .. }) if *id == first_kept_id
                )
            })
        else {
            return Err(format!("unknown first_kept_id: {first_kept_id}"));
        };
        if active_start < self.active_start {
            return Err("compaction cannot restore inactive history".into());
        }
        let entry = ContextEntry::Compaction {
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

    pub fn latest_summary(&self) -> Option<&str> {
        self.latest_summary
            .and_then(|index| self.entries.get(index))
            .and_then(|entry| match entry {
                ContextEntry::Compaction { summary, .. } => Some(summary.as_str()),
                _ => None,
            })
    }

    #[cfg(test)]
    pub fn unresolved_tool_call_ids(&self) -> Vec<String> {
        self.open_tools.values().cloned().collect()
    }

    fn next_id(&self) -> u64 {
        self.entries.back().map_or(1, |entry| entry.id() + 1)
    }

    pub(crate) fn has_active_items_id(&self, id: u64) -> bool {
        id.checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .is_some_and(|index| {
                index >= self.active_start
                    && matches!(
                        self.entries.get(index),
                        Some(ContextEntry::Items { id: candidate, .. }) if *candidate == id
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
    };

    use super::ContextState;

    #[test]
    fn projects_compacted_history_and_tracks_tools() {
        let mut state = ContextState::default();
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
        assert_eq!(state.latest_summary(), Some("summary"));
    }
}
