use std::collections::{HashMap, HashSet};

use gpui::{ScrollHandle, SharedString, Window};

use crate::domain::MessageId;
use crate::streaming_markdown::StreamingMarkdownState;

use super::frame_clock::FrameThrottledScroll;

#[derive(Debug)]
pub(crate) struct MessagePresentation {
    pub(crate) render_text: SharedString,
    pub(crate) markdown: StreamingMarkdownState,
    reasoning_summary_scroll: FrameThrottledScroll,
    source: String,
}

impl MessagePresentation {
    fn new(source: &str, markdown: bool) -> Self {
        let mut presentation = Self {
            render_text: source.to_owned().into(),
            markdown: StreamingMarkdownState::default(),
            reasoning_summary_scroll: FrameThrottledScroll::default(),
            source: source.to_owned(),
        };
        if markdown {
            presentation.markdown.update(source);
        }
        presentation
    }

    fn update(&mut self, source: &str, markdown: bool) {
        if self.source == source {
            return;
        }
        self.source.clear();
        self.source.push_str(source);
        self.render_text = source.to_owned().into();
        if markdown {
            self.markdown.update(source);
        }
    }

    pub(crate) fn reasoning_summary_scroll(&self) -> ScrollHandle {
        self.reasoning_summary_scroll.handle()
    }

    pub(crate) fn align_reasoning_summary(
        &self,
        follow_end: bool,
        revision: u64,
        window: &mut Window,
    ) {
        if follow_end {
            self.reasoning_summary_scroll.follow_end(revision, window);
        } else {
            self.reasoning_summary_scroll.cancel_and_reset();
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct MessagePresentationStore {
    entries: HashMap<MessageId, MessagePresentation>,
}

impl MessagePresentationStore {
    pub(crate) fn sync<'a>(
        &mut self,
        messages: impl IntoIterator<Item = (MessageId, &'a str, bool)>,
    ) {
        let mut live = HashSet::new();
        for (id, source, markdown) in messages {
            live.insert(id);
            self.entries
                .entry(id)
                .and_modify(|presentation| presentation.update(source, markdown))
                .or_insert_with(|| MessagePresentation::new(source, markdown));
        }
        self.entries.retain(|id, _| live.contains(id));
    }

    pub(crate) fn get(&self, id: MessageId) -> &MessagePresentation {
        self.entries
            .get(&id)
            .expect("message presentation must be synchronized before rendering")
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_updates_only_changed_sources_and_removes_stale_messages() {
        let mut store = MessagePresentationStore::default();
        store.sync([(MessageId(1), "one", true), (MessageId(2), "two", false)]);
        let revision = store.get(MessageId(1)).markdown.revision();
        store.sync([(MessageId(1), "one", true)]);
        assert_eq!(store.get(MessageId(1)).markdown.revision(), revision);
        assert!(!store.entries.contains_key(&MessageId(2)));
    }
}
