use super::frame_clock::FrameThrottledScroll;
use crate::domain::MessageId;
use gpui_kit::{ScrollHandle, Window};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct MessagePresentation {
    reasoning_summary_scroll: FrameThrottledScroll,
    overlay: PresentationOverlay,
}
impl MessagePresentation {
    pub(crate) fn reasoning_summary_scroll(&self) -> ScrollHandle {
        self.reasoning_summary_scroll.handle()
    }
    pub(crate) fn expanded(&self) -> bool {
        self.overlay.expanded
    }
    pub(crate) fn rating(&self) -> Option<bool> {
        self.overlay.rating
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
#[derive(Clone, Copy, Debug, Default)]
struct PresentationOverlay {
    expanded: bool,
    rating: Option<bool>,
}

/// Lightweight interaction state outlives the viewport; text, ASTs and selections do not live here.
#[derive(Debug, Default)]
pub(crate) struct MessagePresentationStore {
    entries: HashMap<MessageId, MessagePresentation>,
    active_session: String,
    overlays: HashMap<String, HashMap<MessageId, PresentationOverlay>>,
    revision: u64,
}
impl MessagePresentationStore {
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }
    pub(crate) fn expanded(&self, id: MessageId) -> bool {
        self.overlays
            .get(&self.active_session)
            .and_then(|overlays| overlays.get(&id))
            .is_some_and(|overlay| overlay.expanded)
    }
    pub(crate) fn retain_messages(&mut self, messages: &HashSet<MessageId>) {
        self.entries.retain(|id, _| messages.contains(id));
    }
    pub(crate) fn activate(&mut self, session: impl Into<String>) {
        let session = session.into();
        if self.active_session != session {
            self.entries.clear();
            self.active_session = session;
            self.revision = self.revision.wrapping_add(1);
        }
    }
    pub(crate) fn sync_message(&mut self, id: MessageId) -> &MessagePresentation {
        let overlay = self
            .overlays
            .get(&self.active_session)
            .and_then(|overlays| overlays.get(&id))
            .copied()
            .unwrap_or_default();
        self.entries
            .entry(id)
            .or_insert_with(|| MessagePresentation {
                reasoning_summary_scroll: FrameThrottledScroll::default(),
                overlay,
            })
    }
    pub(crate) fn toggle_expanded(&mut self, id: MessageId) -> Option<bool> {
        let presentation = self.entries.get_mut(&id)?;
        presentation.overlay.expanded = !presentation.overlay.expanded;
        let expanded = presentation.overlay.expanded;
        self.overlays
            .entry(self.active_session.clone())
            .or_default()
            .insert(id, presentation.overlay);
        self.revision = self.revision.wrapping_add(1);
        Some(expanded)
    }
    pub(crate) fn rate(&mut self, id: MessageId, positive: bool) -> Option<Option<bool>> {
        let presentation = self.entries.get_mut(&id)?;
        presentation.overlay.rating =
            (presentation.overlay.rating != Some(positive)).then_some(positive);
        self.overlays
            .entry(self.active_session.clone())
            .or_default()
            .insert(id, presentation.overlay);
        Some(presentation.overlay.rating)
    }
    pub(crate) fn remove_session(&mut self, session: &str) {
        self.overlays.remove(session);
        if self.active_session == session {
            self.entries.clear();
            self.active_session.clear();
        }
    }
    #[cfg(test)]
    pub(crate) fn retained_messages(&self) -> usize {
        self.entries.len()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eviction_and_session_switches_preserve_only_interaction_state() {
        let mut store = MessagePresentationStore::default();
        store.activate("one");
        store.sync_message(MessageId(1));
        assert_eq!(store.toggle_expanded(MessageId(1)), Some(true));
        assert_eq!(store.rate(MessageId(1), true), Some(Some(true)));
        store.retain_messages(&HashSet::new());
        assert_eq!(store.retained_messages(), 0);
        store.activate("two");
        assert!(!store.sync_message(MessageId(1)).expanded());
        store.activate("one");
        let presentation = store.sync_message(MessageId(1));
        assert!(presentation.expanded());
        assert_eq!(presentation.rating(), Some(true));
        store.remove_session("one");
        store.activate("one");
        assert!(!store.sync_message(MessageId(1)).expanded());
    }
}
