use std::{cell::RefCell, collections::HashMap};

use gpui::{ScrollHandle, SharedString, Window};

use crate::domain::MessageId;
use crate::streaming_markdown::StreamingMarkdownState;

use super::frame_clock::FrameThrottledScroll;

#[derive(Debug)]
pub(crate) struct MessagePresentation {
    pub(crate) render_text: SharedString,
    pub(crate) markdown: StreamingMarkdownState,
    reasoning_summary_scroll: FrameThrottledScroll,
    source_generation: u64,
    source_revision: u64,
    markdown_enabled: bool,
    selection: RefCell<Option<super::MessageSelection>>,
    expanded: bool,
    rating: Option<bool>,
}

impl MessagePresentation {
    fn new(
        source: &str,
        source_generation: u64,
        source_revision: u64,
        markdown: bool,
        overlay: PresentationOverlay,
    ) -> Self {
        let mut presentation = Self {
            render_text: source.to_owned().into(),
            markdown: StreamingMarkdownState::default(),
            reasoning_summary_scroll: FrameThrottledScroll::default(),
            source_generation,
            source_revision,
            markdown_enabled: markdown,
            selection: RefCell::new(None),
            expanded: overlay.expanded,
            rating: overlay.rating,
        };
        if markdown {
            presentation.markdown.update(source);
        }
        presentation
    }

    fn update(
        &mut self,
        source: &str,
        source_generation: u64,
        source_revision: u64,
        markdown: bool,
    ) {
        if self.source_generation == source_generation
            && self.source_revision == source_revision
            && self.markdown_enabled == markdown
        {
            return;
        }
        self.source_generation = source_generation;
        self.source_revision = source_revision;
        self.markdown_enabled = markdown;
        self.render_text = source.to_owned().into();
        if markdown {
            self.markdown.update(source);
        }
    }

    pub(crate) fn reasoning_summary_scroll(&self) -> ScrollHandle {
        self.reasoning_summary_scroll.handle()
    }

    pub(crate) fn selection(
        &self,
        order: u64,
        window: &Window,
        cx: &mut gpui::App,
    ) -> super::SelectionFrame {
        self.selection
            .borrow_mut()
            .get_or_insert_with(|| super::MessageSelection::new(window, cx))
            .frame(order)
    }

    pub(crate) fn expanded(&self) -> bool {
        self.expanded
    }

    pub(crate) fn rating(&self) -> Option<bool> {
        self.rating
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PresentationOverlay {
    expanded: bool,
    rating: Option<bool>,
}

#[derive(Debug, Default)]
pub(crate) struct MessagePresentationStore {
    entries: HashMap<MessageId, MessagePresentation>,
    active_session: Option<String>,
    overlays: HashMap<String, HashMap<MessageId, PresentationOverlay>>,
}

impl MessagePresentationStore {
    /// Switches the presentation namespace without touching canonical messages. Visible rows
    /// synchronize themselves lazily by stable ID and message revision during rendering.
    pub(crate) fn activate(&mut self, session: impl Into<String>) {
        let session = session.into();
        if self.active_session.as_deref() != Some(session.as_str()) {
            self.entries.clear();
            self.active_session = Some(session);
        }
    }

    #[allow(
        clippy::expect_used,
        reason = "rendering activates its presentation namespace before syncing messages"
    )]
    pub(crate) fn sync_message(
        &mut self,
        id: MessageId,
        generation: u64,
        revision: u64,
        source: &str,
        markdown: bool,
    ) -> &MessagePresentation {
        let session = self
            .active_session
            .as_ref()
            .expect("message presentation namespace must be activated before rendering");
        let overlay = self
            .overlays
            .get(session)
            .and_then(|overlays| overlays.get(&id))
            .copied()
            .unwrap_or_default();
        self.entries
            .entry(id)
            .and_modify(|presentation| presentation.update(source, generation, revision, markdown))
            .or_insert_with(|| {
                MessagePresentation::new(source, generation, revision, markdown, overlay)
            })
    }

    pub(crate) fn toggle_expanded(&mut self, id: MessageId) -> Option<bool> {
        let session = self.active_session.clone()?;
        let presentation = self.entries.get_mut(&id)?;
        presentation.expanded = !presentation.expanded;
        let expanded = presentation.expanded;
        self.overlays
            .entry(session)
            .or_default()
            .entry(id)
            .or_default()
            .expanded = expanded;
        Some(expanded)
    }

    pub(crate) fn rate(&mut self, id: MessageId, positive: bool) -> Option<Option<bool>> {
        let session = self.active_session.clone()?;
        let presentation = self.entries.get_mut(&id)?;
        presentation.rating = (presentation.rating != Some(positive)).then_some(positive);
        let rating = presentation.rating;
        self.overlays
            .entry(session)
            .or_default()
            .entry(id)
            .or_default()
            .rating = rating;
        Some(rating)
    }

    pub(crate) fn remove_session(&mut self, session: &str) {
        self.overlays.remove(session);
        if self.active_session.as_deref() == Some(session) {
            self.active_session = None;
            self.entries.clear();
        }
    }

    #[cfg(test)]
    pub(crate) fn selection_initialized(&self, id: MessageId) -> bool {
        self.entries
            .get(&id)
            .is_some_and(|presentation| presentation.selection.borrow().is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_revision_controls_lazy_markdown_updates() {
        let mut store = MessagePresentationStore::default();
        store.activate("session");
        let revision = store
            .sync_message(MessageId(1), 1, 0, "one", true)
            .markdown
            .revision();
        assert_eq!(
            store
                .sync_message(MessageId(1), 1, 0, "one", true)
                .markdown
                .revision(),
            revision
        );
        let updated = store.sync_message(MessageId(1), 1, 1, "two", true);
        assert_eq!(updated.render_text.as_ref(), "two");
        assert!(updated.markdown.revision() > revision);
    }

    #[test]
    fn overlays_are_scoped_per_session_and_survive_switches() {
        let mut store = MessagePresentationStore::default();
        store.activate("one");
        store.sync_message(MessageId(1), 1, 0, "first", true);
        assert_eq!(store.toggle_expanded(MessageId(1)), Some(true));
        assert_eq!(store.rate(MessageId(1), true), Some(Some(true)));

        store.activate("two");
        assert!(
            !store
                .sync_message(MessageId(1), 1, 0, "second", true)
                .expanded()
        );
        assert_eq!(
            store
                .sync_message(MessageId(1), 1, 0, "second", true)
                .rating(),
            None
        );

        store.activate("one");
        let presentation = store.sync_message(MessageId(1), 1, 0, "first", true);
        assert!(presentation.expanded());
        assert_eq!(presentation.rating(), Some(true));
    }

    #[test]
    fn same_text_switching_to_assistant_initializes_markdown() {
        let mut store = MessagePresentationStore::default();
        store.activate("session");
        let revision = store
            .sync_message(MessageId(1), 1, 0, "same", false)
            .markdown
            .revision();

        let presentation = store.sync_message(MessageId(1), 1, 0, "same", true);

        assert!(presentation.markdown.revision() > revision);
    }

    #[test]
    fn new_projection_generation_refreshes_reset_message_revisions() {
        let mut store = MessagePresentationStore::default();
        store.activate("session");
        store.sync_message(MessageId(1), 1, 0, "old", true);

        let presentation = store.sync_message(MessageId(1), 2, 0, "new", true);

        assert_eq!(presentation.render_text.as_ref(), "new");
    }
}
