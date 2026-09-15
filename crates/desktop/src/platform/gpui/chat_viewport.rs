//! The chat list owns only the visible/overscan presentation working set. Journal/runtime
//! ownership stays in SessionRuntime; rows are cheap source locators, never parsed documents.
use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use gpui_kit::{Context, ListAlignment, ListOffset, ListState, Task, Window, px};
use im::Vector;

use super::{MessagePresentationStore, MessageSelection, SelectionFrame};
use crate::{
    app::DesktopApp,
    domain::{Message, MessageId, Role},
    dsh_markdown::{self, PreparedMarkdown},
    layout::ScrollAnchor,
    ui_theme::markdown_highlight_theme,
};

const CHUNK_BYTES: usize = 2048;
const CHUNK_LINES: usize = 24;
const MAX_PROSE_BYTES: usize = 16 * 1024;
const PRESENTATION_BYTES: usize = 8 * 1024 * 1024;
const MAX_CHUNK_PRESENTATION_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct RowKey {
    pub(crate) message: MessageId,
    pub(crate) field: u8,
    pub(crate) start: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SourceChunk {
    pub(crate) range: Range<usize>,
    body: Range<usize>,
    fence: Option<String>,
    literal: bool,
}

/// Scan block boundaries without building an AST. Oversized blocks are bounded fragments;
/// fenced-code fragments carry their language so scrolling into a fence remains code.
fn source_chunks(source: &str) -> Vec<SourceChunk> {
    if source.len() <= CHUNK_BYTES {
        return vec![SourceChunk {
            range: 0..source.len(),
            body: 0..source.len(),
            fence: None,
            literal: false,
        }];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut end = 0;
    let mut lines = 0;
    let mut fence: Option<(u8, usize, String)> = None;
    let mut body_start = 0;
    let mut chunk_fence = None;
    let mut literal = false;
    for line in source.split_inclusive('\n') {
        let trimmed = line.trim_start_matches(' ');
        let indent = line.len() - trimmed.len();
        let marker = trimmed.as_bytes().first().copied().unwrap_or_default();
        let marker_len = trimmed.bytes().take_while(|byte| *byte == marker).count();
        let opening = fence.is_none()
            && line.len() <= CHUNK_BYTES
            && (3..=64).contains(&marker_len)
            && indent <= 3
            && matches!(marker, b'`' | b'~');
        let closing = fence.as_ref().is_some_and(|(byte, count, _)| {
            indent <= 3
                && marker == *byte
                && marker_len >= *count
                && trimmed[marker_len..].trim().is_empty()
        });
        if opening && end > start {
            chunks.push(SourceChunk {
                range: start..end,
                body: body_start..end,
                fence: chunk_fence.take(),
                literal,
            });
            start = end;
            lines = 0;
        }
        if opening {
            // Keep fence context bounded even for pathological info strings.
            let language = trimmed[marker_len..]
                .split_whitespace()
                .next()
                .unwrap_or_default();
            let language = &language[..language.floor_char_boundary(64.min(language.len()))];
            let header = format!(
                "{}{}",
                (marker as char).to_string().repeat(marker_len.min(64)),
                language
            );
            fence = Some((marker, marker_len, header.clone()));
            chunk_fence = Some(header);
            body_start = end + line.len();
        }
        if closing {
            let body_end = end;
            end += line.len();
            chunks.push(SourceChunk {
                range: start..end,
                body: body_start..body_end,
                fence: chunk_fence.take(),
                literal,
            });
            start = end;
            body_start = end;
            fence = None;
            lines = 0;
            continue;
        }
        // A single unbroken line is also bounded; UTF-8 boundaries never depend on slicing bytes blindly.
        let line_end = end + line.len();
        let limit = if fence.is_some() {
            CHUNK_BYTES
        } else {
            MAX_PROSE_BYTES
        };
        while line_end.saturating_sub(body_start) > limit {
            // ponytail: an oversized non-code block stays lossless plain text. Structural
            // slicing of arbitrary nested Markdown is needed only beyond this 16 KiB block limit.
            literal = fence.is_none();
            let split = source.floor_char_boundary(body_start + limit);
            if split <= body_start {
                break;
            }
            chunks.push(SourceChunk {
                range: start..split,
                body: body_start..split,
                fence: chunk_fence.clone(),
                literal,
            });
            start = split;
            body_start = split;
            lines = 0;
        }
        end = line_end;
        lines += 1;
        if !opening
            && (fence.is_some() && lines >= CHUNK_LINES
                || fence.is_none() && line.trim().is_empty())
        {
            chunks.push(SourceChunk {
                range: start..end,
                body: body_start..end,
                fence: chunk_fence.clone(),
                literal,
            });
            start = end;
            body_start = end;
            lines = 0;
            literal = false;
        }
    }
    if end > start || chunks.is_empty() {
        chunks.push(SourceChunk {
            range: start..end,
            body: body_start.min(end)..end,
            fence: chunk_fence,
            literal,
        });
    }
    chunks
}

fn text_chunks(source: &str) -> Vec<SourceChunk> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < source.len() {
        let end = source.floor_char_boundary((start + CHUNK_BYTES).min(source.len()));
        chunks.push(SourceChunk {
            range: start..end,
            body: start..end,
            fence: None,
            literal: false,
        });
        start = end;
    }
    chunks
}

#[derive(Clone, Debug)]
pub(crate) struct ChatRow {
    pub(crate) key: RowKey,
    pub(crate) message: Arc<Message>,
    pub(crate) message_index: usize,
    revision: u64,
    pub(crate) chunk: Option<SourceChunk>,
}
impl ChatRow {
    fn source(&self) -> &str {
        if self.key.field == 2 {
            self.message.payload.as_deref().unwrap_or_default()
        } else {
            &self.message.text
        }
    }
    pub(crate) fn plain(&self) -> &str {
        self.chunk
            .as_ref()
            .map_or("", |chunk| &self.source()[chunk.range.clone()])
    }
    fn markdown_source(&self) -> String {
        let Some(chunk) = &self.chunk else {
            return String::new();
        };
        let source = &self.source()[chunk.body.clone()];
        if self.message.role == Role::Tool {
            let count = source
                .split(|ch| ch != '`')
                .map(str::len)
                .max()
                .unwrap_or(0)
                .max(2)
                + 1;
            let marker = "`".repeat(count);
            let title = self
                .message
                .title
                .as_deref()
                .unwrap_or_default()
                .to_ascii_lowercase();
            let language = if self.key.field == 2 {
                "json"
            } else if title.contains("shell")
                || title.contains("bash")
                || title.contains("terminal")
            {
                "bash"
            } else if title.contains("json") {
                "json"
            } else {
                "text"
            };
            return format!("{marker}{language}\n{source}\n{marker}");
        }
        let fence = chunk.fence.as_deref();
        if let Some(fence) = fence {
            let marker = fence.chars().next().unwrap_or('`');
            let count = fence.chars().take_while(|ch| *ch == marker).count();
            format!("{fence}\n{source}\n{}", marker.to_string().repeat(count))
        } else {
            source.to_owned()
        }
    }
    fn rich(&self) -> bool {
        matches!(self.message.role, Role::Assistant | Role::Tool)
            && self.chunk.as_ref().is_some_and(|chunk| !chunk.literal)
    }
}

struct Presentation {
    revision: u64,
    dark: bool,
    selection: MessageSelection,
    prepared: Option<Arc<PreparedMarkdown>>,
    settled: bool,
}
struct InFlight {
    key: RowKey,
    epoch: u64,
    revision: u64,
    dark: bool,
    cancel: Arc<AtomicBool>,
    _task: Task<()>,
}

pub(crate) struct ChatViewport {
    pub(crate) list: ListState,
    pub(crate) rows: Vec<ChatRow>,
    messages: Vector<Arc<Message>>,
    notices: Vector<Arc<Message>>,
    overlays_revision: u64,
    namespace: String,
    lineage: u64,
    epoch: u64,
    presentations: HashMap<RowKey, Presentation>,
    requested: HashMap<RowKey, usize>,
    pub(crate) demand_scheduled: bool,
    in_flight: Option<InFlight>,
    pub(crate) pending_anchor: Option<ScrollAnchor>,
    dark: bool,
    #[cfg(test)]
    pub(crate) worker_gate: Option<tokio::sync::oneshot::Receiver<()>>,
    #[cfg(test)]
    pub(crate) worker_starts: Arc<std::sync::atomic::AtomicUsize>,
}

impl Default for ChatViewport {
    fn default() -> Self {
        let list = ListState::new(0, ListAlignment::Top, px(600.0));
        list.set_follow_mode(gpui_kit::FollowMode::Tail);
        Self {
            list,
            rows: Vec::new(),
            messages: Vector::new(),
            notices: Vector::new(),
            overlays_revision: u64::MAX,
            namespace: String::new(),
            lineage: 0,
            epoch: 0,
            presentations: HashMap::new(),
            requested: HashMap::new(),
            demand_scheduled: false,
            in_flight: None,
            pending_anchor: Some(ScrollAnchor::Tail),
            dark: false,
            #[cfg(test)]
            worker_gate: None,
            #[cfg(test)]
            worker_starts: Arc::default(),
        }
    }
}
impl ChatViewport {
    pub(crate) fn activate(&mut self, namespace: String) {
        if self.namespace != namespace {
            self.namespace = namespace;
            self.release();
            self.messages = Vector::new();
            self.notices = Vector::new();
            self.rows.clear();
            self.overlays_revision = u64::MAX;
            self.list.reset(0);
        }
    }
    pub(crate) fn release(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.presentations.clear();
        self.list.remeasure();
        self.requested.clear();
        if let Some(work) = &self.in_flight {
            work.cancel.store(true, Ordering::Relaxed);
        }
    }
    pub(crate) fn sync(
        &mut self,
        messages: &Vector<Arc<Message>>,
        notices: &Vector<Arc<Message>>,
        overlays: &MessagePresentationStore,
        lineage: u64,
        dark: bool,
    ) {
        let lineage_changed = self.lineage != lineage;
        if self.dark != dark || lineage_changed {
            self.release();
            self.list.remeasure();
            self.dark = dark;
            self.lineage = lineage;
        }
        if self.messages.ptr_eq(messages)
            && self.notices.ptr_eq(notices)
            && self.overlays_revision == overlays.revision()
        {
            self.restore_pending();
            return;
        }
        let anchor = self.pending_anchor.take().unwrap_or_else(|| self.anchor());
        let previous = std::mem::take(&mut self.rows);
        let old_rows = previous
            .iter()
            .map(|row| (row.key, row.revision))
            .collect::<Vec<_>>();
        // Keep only cheap source locators for history. Unchanged messages share their existing index.
        let mut indexed = HashMap::<MessageId, Vec<ChatRow>>::new();
        for row in previous {
            indexed.entry(row.key.message).or_default().push(row);
        }
        for (index, message) in messages.iter().chain(notices).enumerate() {
            let previous_rows = indexed.remove(&message.key).unwrap_or_default();
            if !lineage_changed
                && self.overlays_revision == overlays.revision()
                && previous_rows
                    .first()
                    .is_some_and(|row| row.message.revision == message.revision)
            {
                self.rows.extend(previous_rows.into_iter().map(|mut row| {
                    row.message_index = index;
                    row
                }));
                continue;
            }
            let row_start = self.rows.len();
            let chrome = ChatRow {
                key: RowKey {
                    message: message.key,
                    field: 0,
                    start: 0,
                },
                message: message.clone(),
                message_index: index,
                revision: message.revision,
                chunk: None,
            };
            if matches!(message.role, Role::Reasoning | Role::Tool | Role::Notice) {
                self.rows.push(chrome.clone());
            }
            if matches!(message.role, Role::Assistant | Role::User)
                || overlays.expanded(message.key)
            {
                if message.role == Role::Tool
                    && let Some(payload) = &message.payload
                {
                    self.rows
                        .extend(text_chunks(payload).into_iter().map(|chunk| ChatRow {
                            key: RowKey {
                                field: 2,
                                start: chunk.range.start,
                                ..chrome.key
                            },
                            chunk: Some(chunk),
                            ..chrome.clone()
                        }));
                }
                let chunks = if message.role == Role::Tool {
                    text_chunks(&message.text)
                } else {
                    source_chunks(&message.text)
                };
                self.rows.extend(chunks.into_iter().map(|chunk| ChatRow {
                    key: RowKey {
                        field: 1,
                        start: chunk.range.start,
                        ..chrome.key
                    },
                    chunk: Some(chunk),
                    ..chrome.clone()
                }));
            }
            if matches!(message.role, Role::Assistant | Role::User) {
                self.rows.push(chrome);
            }
            // Streaming only invalidates changed fragments, including the unfinished tail.
            let previous_rows = previous_rows
                .iter()
                .map(|row| (row.key, row))
                .collect::<HashMap<_, _>>();
            for row in &mut self.rows[row_start..] {
                if row.chunk.is_some()
                    && let Some(old) = previous_rows.get(&row.key)
                    && old.chunk == row.chunk
                    && old.plain() == row.plain()
                {
                    row.revision = old.revision;
                }
            }
        }
        if self.rows.is_empty() {
            self.release();
        }
        self.messages = messages.clone();
        self.notices = notices.clone();
        self.overlays_revision = overlays.revision();
        let new_rows = self
            .rows
            .iter()
            .map(|row| (row.key, row.revision))
            .collect::<Vec<_>>();
        let prefix = old_rows
            .iter()
            .zip(&new_rows)
            .take_while(|(old, new)| old == new)
            .count();
        let suffix = old_rows[prefix..]
            .iter()
            .rev()
            .zip(new_rows[prefix..].iter().rev())
            .take_while(|(old, new)| old == new)
            .count();
        self.list.splice(
            prefix..old_rows.len() - suffix,
            new_rows.len() - prefix - suffix,
        );
        self.pending_anchor = Some(anchor);
        self.restore_pending();
    }
    pub(crate) fn anchor(&self) -> ScrollAnchor {
        if self.list.is_following_tail() {
            return ScrollAnchor::Tail;
        }
        let offset = self.list.logical_scroll_top();
        self.rows
            .get(offset.item_ix)
            .map_or(ScrollAnchor::Tail, |row| ScrollAnchor::Block {
                id: row.key.message,
                field: row.key.field,
                source_offset: row.key.start,
                local_offset: f32::from(offset.offset_in_item),
            })
    }
    fn restore_pending(&mut self) {
        let Some(anchor) = self.pending_anchor.take() else {
            return;
        };
        match anchor {
            ScrollAnchor::Tail => self.list.set_follow_mode(gpui_kit::FollowMode::Tail),
            ScrollAnchor::Block {
                id,
                field,
                source_offset,
                local_offset,
            } => {
                let row = self
                    .rows
                    .iter()
                    .rposition(|row| {
                        row.key.message == id
                            && row.key.field == field
                            && row.key.start <= source_offset
                    })
                    .or_else(|| self.rows.iter().position(|row| row.key.message == id));
                if let Some(item_ix) = row {
                    self.list.set_follow_mode(gpui_kit::FollowMode::Normal);
                    self.list.scroll_to(ListOffset {
                        item_ix,
                        offset_in_item: px(local_offset.max(0.0)),
                    });
                } else {
                    self.list.scroll_to_end();
                }
            }
        }
    }
    pub(crate) fn begin_frame(&mut self) {
        self.requested.clear();
    }
    pub(crate) fn row(
        &mut self,
        index: usize,
        window: &Window,
        cx: &mut gpui_kit::App,
    ) -> Option<(
        ChatRow,
        Option<SelectionFrame>,
        Option<Arc<PreparedMarkdown>>,
    )> {
        let row = self.rows.get(index)?.clone();
        self.requested.insert(row.key, index);
        if row.chunk.is_none() {
            return Some((row, None, None));
        }
        let entry = self
            .presentations
            .entry(row.key)
            .or_insert_with(|| Presentation {
                revision: row.revision,
                dark: self.dark,
                selection: MessageSelection::new(window, cx),
                prepared: None,
                settled: !row.rich(),
            });
        if entry.revision != row.revision || entry.dark != self.dark {
            entry.revision = row.revision;
            entry.dark = self.dark;
            entry.prepared = None;
            entry.settled = !row.rich();
        }
        Some((
            row,
            Some(entry.selection.frame(index as u64)),
            entry.prepared.clone(),
        ))
    }
    fn current_result(
        &self,
        key: RowKey,
        epoch: u64,
        revision: u64,
        dark: bool,
        cancelled: bool,
    ) -> Option<usize> {
        if cancelled || self.epoch != epoch || self.dark != dark {
            return None;
        }
        self.requested.get(&key).copied().filter(|index| {
            self.rows
                .get(*index)
                .is_some_and(|row| row.key == key && row.revision == revision)
        })
    }
    #[cfg(test)]
    pub(crate) fn prepared_chunks(&self) -> usize {
        self.presentations
            .values()
            .filter(|entry| entry.prepared.is_some())
            .count()
    }
    #[cfg(test)]
    pub(crate) fn retained_chunks(&self) -> usize {
        self.presentations.len()
    }
    #[cfg(test)]
    pub(crate) fn selection_initialized(&self, id: MessageId) -> bool {
        self.presentations.keys().any(|key| key.message == id)
    }
}

impl DesktopApp {
    /// Called after list layout, never while GPUI holds ListState's mutable borrow.
    pub(crate) fn finish_chat_frame(&mut self, cx: &mut Context<Self>) {
        // GPUI has now applied wheel/scrollbar movement and measured the visible rows.
        // Reading here also avoids the mutable ListState borrow held by scroll handlers.
        let follows = {
            let chat = self.chat.borrow();
            chat.list.is_following_tail()
                || chat
                    .rows
                    .len()
                    .checked_sub(1)
                    .and_then(|last| chat.list.bounds_for_item(last))
                    .is_some_and(|last| {
                        last.bottom() <= chat.list.viewport_bounds().bottom() + px(2.0)
                    })
        };
        if follows != self.core.follow_chat_tail {
            if follows {
                self.chat
                    .borrow()
                    .list
                    .set_follow_mode(gpui_kit::FollowMode::Tail);
            }
            self.dispatch_local(
                crate::domain::Action::Scroll(if follows {
                    crate::domain::ScrollIntent::Toward { at_tail: true }
                } else {
                    crate::domain::ScrollIntent::Away
                }),
                cx,
            );
        }
        let mut chat = self.chat.borrow_mut();
        chat.demand_scheduled = false;
        let requested = chat.requested.clone();
        chat.presentations
            .retain(|key, _| requested.contains_key(key));
        self.message_presentations
            .borrow_mut()
            .retain_messages(&requested.keys().map(|key| key.message).collect());
        if let Some(work) = &chat.in_flight {
            if !requested.contains_key(&work.key)
                || work.epoch != chat.epoch
                || work.dark != chat.dark
                || !requested
                    .get(&work.key)
                    .and_then(|index| chat.rows.get(*index))
                    .is_some_and(|row| row.revision == work.revision)
            {
                work.cancel.store(true, Ordering::Relaxed);
            }
            return;
        }
        let top = chat.list.logical_scroll_top().item_ix;
        let viewport = chat.list.viewport_bounds();
        let row = requested
            .iter()
            .filter(|(key, _)| {
                chat.presentations
                    .get(key)
                    .is_some_and(|entry| !entry.settled)
            })
            .min_by_key(|(_, index)| {
                let visible = chat.list.bounds_for_item(**index).is_some_and(|bounds| {
                    bounds.bottom() > viewport.origin.y && bounds.origin.y < viewport.bottom()
                });
                (!visible, index.abs_diff(top))
            })
            .and_then(|(_, index)| chat.rows.get(*index))
            .cloned();
        let Some(row) = row else {
            return;
        };
        let key = row.key;
        let epoch = chat.epoch;
        let revision = row.revision;
        let dark = chat.dark;
        let source = row.markdown_source();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let completion_cancel = cancel.clone();
        let theme = markdown_highlight_theme(dark).clone();
        let executor = cx.background_executor().clone();
        #[cfg(test)]
        let (gate, starts, dispatcher) = (
            chat.worker_gate.take(),
            chat.worker_starts.clone(),
            executor.clone(),
        );
        let task = cx.spawn(async move |this, cx| {
            let prepared = executor
                .spawn(async move {
                    #[cfg(test)]
                    {
                        assert!(
                            !dispatcher.is_main_thread(),
                            "Markdown preparation must run in the background"
                        );
                        starts.fetch_add(1, Ordering::Relaxed);
                        if let Some(gate) = gate {
                            let _ = gate.await;
                        }
                    }
                    dsh_markdown::prepare_markdown(&source, &theme, &worker_cancel)
                })
                .await;
            let _ = this.update(cx, |this, cx| {
                let mut chat = this.chat.borrow_mut();
                chat.in_flight = None;
                if let Some(index) = chat.current_result(
                    key,
                    epoch,
                    revision,
                    dark,
                    completion_cancel.load(Ordering::Relaxed),
                ) {
                    let current_bytes: usize = chat
                        .presentations
                        .values()
                        .filter_map(|entry| entry.prepared.as_ref())
                        .map(|value| value.bytes())
                        .sum();
                    if let Some(entry) = chat.presentations.get_mut(&key) {
                        entry.prepared = prepared
                            .filter(|value| {
                                value.bytes() <= MAX_CHUNK_PRESENTATION_BYTES
                                    && current_bytes + value.bytes() <= PRESENTATION_BYTES
                            })
                            .map(Arc::new);
                        entry.settled = true;
                        chat.list.remeasure_items(index..index + 1);
                    }
                }
                cx.notify();
            });
        });
        chat.in_flight = Some(InFlight {
            key,
            epoch,
            revision,
            dark,
            cancel,
            _task: task,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn chunks_bound_long_messages_and_preserve_source_and_fences() {
        for source in [
            "`".repeat(100000),
            "文🦀字".repeat(10000),
            format!(
                "Intro\n\n```haskell\n{}\n```\nEnd",
                "main = print 42\n".repeat(1000)
            ),
        ] {
            let chunks = source_chunks(&source);
            assert!(chunks.len() > 2);
            assert_eq!(
                chunks
                    .iter()
                    .map(|chunk| &source[chunk.range.clone()])
                    .collect::<String>(),
                source
            );
            for chunk in &chunks {
                assert!(chunk.range.len() <= MAX_PROSE_BYTES + CHUNK_BYTES);
                assert!(
                    chunk.body.len()
                        <= if chunk.fence.is_some() {
                            CHUNK_BYTES
                        } else {
                            MAX_PROSE_BYTES
                        }
                );
                assert!(
                    source.is_char_boundary(chunk.body.start)
                        && source.is_char_boundary(chunk.body.end)
                );
            }
            if source.contains("haskell") {
                assert!(
                    chunks
                        .iter()
                        .filter(|chunk| chunk.fence.as_deref() == Some("```haskell"))
                        .count()
                        > 10
                );
            }
        }
    }
    fn message(text: String, revision: u64) -> Arc<Message> {
        Arc::new(Message {
            key: MessageId(1),
            revision,
            role: Role::Assistant,
            text,
            tool_call_id: None,
            title: None,
            payload: None,
            schema: None,
            pending: false,
            failed: false,
            started_at_ms: None,
            duration_ms: None,
            turn: 0,
            step: 0,
            request_id: None,
        })
    }

    #[test]
    fn streaming_keeps_prefix_and_restores_source_anchor_and_rejects_stale_work() {
        let mut chat = ChatViewport::default();
        let overlays = MessagePresentationStore::default();
        let source = "A paragraph with **emphasis**.\n\n".repeat(200);
        chat.sync(
            &Vector::unit(message(source.clone(), 1)),
            &Vector::new(),
            &overlays,
            1,
            false,
        );
        let key = chat.rows[50].key;
        chat.list.set_follow_mode(gpui_kit::FollowMode::Normal);
        chat.list.scroll_to(ListOffset {
            item_ix: 50,
            offset_in_item: px(12.0),
        });
        let anchor = chat.anchor();
        let epoch = chat.epoch;
        chat.requested.insert(key, 50);
        assert_eq!(chat.current_result(key, epoch, 1, false, false), Some(50));
        assert_eq!(chat.current_result(key, epoch, 1, false, true), None);
        chat.sync(
            &Vector::unit(message(format!("{source}New tail"), 2)),
            &Vector::new(),
            &overlays,
            1,
            false,
        );
        assert_eq!(chat.rows[50].revision, 1);
        assert_eq!(chat.anchor(), anchor);
        assert_eq!(chat.current_result(key, epoch, 2, false, false), None);
        chat.begin_frame();
        assert_eq!(chat.current_result(key, epoch, 1, false, false), None);
        chat.requested.insert(key, 50);
        chat.sync(
            &Vector::unit(message(format!("{source}New tail"), 2)),
            &Vector::new(),
            &overlays,
            1,
            true,
        );
        assert_eq!(chat.current_result(key, epoch, 1, false, false), None);
        chat.activate("other session".into());
        assert!(chat.rows.is_empty());
        chat.activate("original session".into());
        chat.pending_anchor = Some(anchor);
        chat.sync(
            &Vector::unit(message(source, 1)),
            &Vector::new(),
            &overlays,
            1,
            false,
        );
        assert_eq!(chat.anchor(), anchor);
        assert_eq!(chat.current_result(key, epoch, 1, false, false), None);
    }

    #[test]
    fn ordinary_tables_and_multiline_markup_are_not_split_at_line_limits() {
        let source = format!(
            "| Heading | Value |\n| --- | --- |\n{}",
            "| **label** | $x$ |\n".repeat(100)
        );
        let chunks = source_chunks(&source);
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].literal);
    }
}
