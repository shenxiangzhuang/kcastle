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
    source_revision: u64,
    expanded: bool,
    rating: Option<bool>,
}

impl MessagePresentation {
    fn new(
        source: &str,
        source_revision: u64,
        markdown: bool,
        overlay: PresentationOverlay,
    ) -> Self {
        let mut presentation = Self {
            render_text: source.to_owned().into(),
            markdown: StreamingMarkdownState::default(),
            reasoning_summary_scroll: FrameThrottledScroll::default(),
            source: source.to_owned(),
            source_revision,
            expanded: overlay.expanded,
            rating: overlay.rating,
        };
        if markdown {
            presentation.markdown.update(source);
        }
        presentation
    }

    fn update(&mut self, source: &str, source_revision: u64, markdown: bool) {
        if self.source_revision == source_revision && self.source == source {
            return;
        }
        self.source_revision = source_revision;
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
    /// Fully synchronizes a selected session. Existing render state is reused only when the
    /// namespace matches; presentation overlays are retained independently per session.
    pub(crate) fn replace_all<'a>(
        &mut self,
        session: impl Into<String>,
        messages: impl IntoIterator<Item = (MessageId, u64, &'a str, bool)>,
    ) {
        let session = session.into();
        if self.active_session.as_deref() != Some(session.as_str()) {
            self.entries.clear();
            self.active_session = Some(session.clone());
        }
        let mut live = HashSet::new();
        for (id, revision, source, markdown) in messages {
            live.insert(id);
            let overlay = self
                .overlays
                .get(&session)
                .and_then(|overlays| overlays.get(&id))
                .copied()
                .unwrap_or_default();
            self.entries
                .entry(id)
                .and_modify(|presentation| presentation.update(source, revision, markdown))
                .or_insert_with(|| MessagePresentation::new(source, revision, markdown, overlay));
        }
        self.entries.retain(|id, _| live.contains(id));
        if let Some(overlays) = self.overlays.get_mut(&session) {
            overlays.retain(|id, _| live.contains(id));
        }
    }

    /// Applies only messages named by the runtime's committed projection delta.
    ///
    /// Returns false when the caller supplied a delta for a session that is not active, in which
    /// case the caller must fall back to `replace_all`.
    pub(crate) fn sync_changed<'a>(
        &mut self,
        session: &str,
        messages: impl IntoIterator<Item = (MessageId, u64, &'a str, bool)>,
    ) -> bool {
        if self.active_session.as_deref() != Some(session) {
            return false;
        }
        for (id, revision, source, markdown) in messages {
            let overlay = self
                .overlays
                .get(session)
                .and_then(|overlays| overlays.get(&id))
                .copied()
                .unwrap_or_default();
            self.entries
                .entry(id)
                .and_modify(|presentation| presentation.update(source, revision, markdown))
                .or_insert_with(|| MessagePresentation::new(source, revision, markdown, overlay));
        }
        true
    }

    /// Compatibility helper for focused tests and one-off transient rows.
    #[cfg(test)]
    pub(crate) fn sync<'a>(
        &mut self,
        messages: impl IntoIterator<Item = (MessageId, &'a str, bool)>,
    ) {
        self.replace_all(
            "__compatibility__",
            messages
                .into_iter()
                .map(|(id, source, markdown)| (id, 0, source, markdown)),
        );
    }

    pub(crate) fn get(&self, id: MessageId) -> &MessagePresentation {
        self.entries
            .get(&id)
            .expect("message presentation must be synchronized before rendering")
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

    #[test]
    fn overlays_are_scoped_per_session_and_survive_switches() {
        let mut store = MessagePresentationStore::default();
        store.replace_all("one", [(MessageId(1), 0, "first", true)]);
        assert_eq!(store.toggle_expanded(MessageId(1)), Some(true));
        assert_eq!(store.rate(MessageId(1), true), Some(Some(true)));

        store.replace_all("two", [(MessageId(1), 0, "second", true)]);
        assert!(!store.get(MessageId(1)).expanded());
        assert_eq!(store.get(MessageId(1)).rating(), None);

        store.replace_all("one", [(MessageId(1), 0, "first", true)]);
        assert!(store.get(MessageId(1)).expanded());
        assert_eq!(store.get(MessageId(1)).rating(), Some(true));
    }

    #[test]
    fn changed_sync_does_not_remove_untouched_presentations() {
        let mut store = MessagePresentationStore::default();
        store.replace_all(
            "session",
            [
                (MessageId(1), 0, "one", true),
                (MessageId(2), 0, "two", false),
            ],
        );
        assert!(store.sync_changed("session", [(MessageId(2), 1, "updated", false)]));
        assert_eq!(store.get(MessageId(1)).render_text.as_ref(), "one");
        assert_eq!(store.get(MessageId(2)).render_text.as_ref(), "updated");
    }
}
