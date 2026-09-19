//! The chat list owns only the visible/overscan presentation working set. Journal/runtime
//! ownership stays in SessionRuntime; rows are cheap source locators, never parsed documents.
use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

mod cache;
use cache::{CacheKey, Cached, IDLE_TTL, PreparationCache};

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
const OVERSCAN: f32 = 600.0;
pub(crate) const MAX_INDEX_SOURCE_BYTES: usize = 1024 * 1024;
pub(crate) const MAX_CODE_SOURCE_BYTES: usize = 256 * 1024;

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
    pub(crate) gap_before: Option<u8>,
    code: Option<CodeSlice>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CodeSlice {
    source: Range<usize>,
    visible: Range<usize>,
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
            gap_before: None,
            code: None,
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
                gap_before: None,
                code: None,
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
                gap_before: None,
                code: None,
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
                gap_before: None,
                code: None,
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
                gap_before: None,
                code: None,
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
            gap_before: None,
            code: None,
        });
    }
    chunks
}

/// The worker discovers real block boundaries. Only byte ranges survive publication.
fn semantic_chunks(source: &str, cancel: &AtomicBool) -> Option<Vec<SourceChunk>> {
    if cancel.load(Ordering::Relaxed) {
        return None;
    }
    let mut state = crate::streaming_markdown::StreamingMarkdownState::default();
    state.update(source);
    let blocks = state
        .frozen()
        .iter()
        .chain(state.tail_blocks())
        .collect::<Vec<_>>();
    fn has_definition(node: &markdown::mdast::Node) -> bool {
        matches!(node, markdown::mdast::Node::Definition(_))
            || node
                .children()
                .is_some_and(|children| children.iter().any(has_definition))
    }
    // Keep the original small-message parsing boundary when references depend on it.
    // ponytail: definitions across larger fragments still require shared document context.
    if source.len() <= CHUNK_BYTES && blocks.iter().any(|block| has_definition(&block.node)) {
        let mut chunks = source_chunks(source);
        chunks[0].gap_before = Some(0);
        return Some(chunks);
    }
    // A fenced node's position starts after its indentation. Keep the entire line
    // in both the source partition and reparse input so code offsets stay identical.
    let block_start = |block: &crate::streaming_markdown::MarkdownBlock| {
        if matches!(block.node, markdown::mdast::Node::Code(_)) {
            source[..block.key].rfind(['\n', '\r']).map_or(0, |i| i + 1)
        } else {
            block.key
        }
    };
    let mut chunks = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return None;
        }
        let gap = dsh_markdown::block_gap(
            index.checked_sub(1).map(|i| &blocks[i].node),
            &block.node,
            blocks.get(index + 1).map(|b| &b.node),
        ) as u8;
        let end = blocks
            .get(index + 1)
            .map_or(source.len(), |b| block_start(b));
        if let markdown::mdast::Node::Code(code) = &block.node {
            let code_start = block_start(block);
            // A browser consumes a document, not independently highlighted line slices.
            if crate::html_preview::is_html(code.lang.as_deref().unwrap_or_default()) {
                chunks.push(SourceChunk {
                    range: if index == 0 { 0 } else { code_start }..end,
                    body: code_start..block.key + block.source.len(),
                    fence: None,
                    literal: false,
                    gap_before: Some(gap),
                    code: Some(CodeSlice {
                        source: code_start..block.key + block.source.len(),
                        visible: 0..code.value.len(),
                    }),
                });
                continue;
            }
            let mut start = 0;
            let mut pieces = Vec::new();
            let mut offset = 0;
            let mut lines = 0;
            for line in code.value.split_inclusive('\n') {
                offset += line.len();
                lines += 1;
                while offset - start > CHUNK_BYTES {
                    let end = code.value.floor_char_boundary(start + CHUNK_BYTES);
                    pieces.push(start..end);
                    start = end;
                    lines = 0;
                }
                if lines >= CHUNK_LINES || offset - start >= CHUNK_BYTES {
                    pieces.push(start..offset);
                    start = offset;
                    lines = 0;
                }
            }
            if start < code.value.len() || pieces.is_empty() {
                pieces.push(start..code.value.len());
            }
            let mut range_start = if index == 0 { 0 } else { code_start };
            let count = pieces.len();
            for (i, visible) in pieces.into_iter().enumerate() {
                let range_end = if i + 1 == count {
                    end
                } else {
                    source.floor_char_boundary(block.key + visible.end)
                };
                chunks.push(SourceChunk {
                    range: range_start..range_end,
                    body: range_start..range_end,
                    fence: None,
                    literal: false,
                    gap_before: Some(if i == 0 { gap } else { 0 }),
                    code: Some(CodeSlice {
                        source: code_start..block.key + block.source.len(),
                        visible,
                    }),
                });
                range_start = range_end;
            }
        } else if block.source.len() <= MAX_PROSE_BYTES {
            chunks.push(SourceChunk {
                range: if index == 0 { 0 } else { block.key }..end,
                body: block.key..block.key + block.source.len(),
                fence: None,
                literal: false,
                gap_before: Some(gap),
                code: None,
            });
        } else {
            // ponytail: oversized blocks retain the existing bounded fallback; the
            // semantic index still prevents neighbouring lists/quotes from being severed.
            let mut pieces = source_chunks(&source[block.key..end]);
            for (piece_index, piece) in pieces.iter_mut().enumerate() {
                piece.range = piece.range.start + block.key..piece.range.end + block.key;
                piece.body = piece.body.start + block.key..piece.body.end + block.key;
                piece.gap_before = Some(if piece_index == 0 { gap } else { 0 });
                piece.literal = true;
                piece.fence = None;
                piece.body = piece.range.clone();
            }
            chunks.extend(pieces);
        }
    }
    if chunks.is_empty() {
        chunks = source_chunks(source);
        for chunk in &mut chunks {
            chunk.gap_before = Some(0);
        }
    }
    Some(chunks)
}

fn paragraph_only(source: &str) -> bool {
    source.len() > 64 * 1024
        && source.lines().all(|line| {
            line.is_empty()
                || (line.chars().next().is_some_and(char::is_alphabetic) && !line.contains('|'))
        })
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
            gap_before: None,
            code: None,
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
    fn indexed(&self, chunks: Vec<SourceChunk>, previous: &[ChatRow]) -> Vec<ChatRow> {
        let previous = previous
            .iter()
            .map(|row| (row.key, row))
            .collect::<HashMap<_, _>>();
        chunks
            .into_iter()
            .map(|chunk| {
                let mut row = Self {
                    key: RowKey {
                        start: chunk.range.start,
                        ..self.key
                    },
                    chunk: Some(chunk),
                    revision: self.message.revision,
                    ..self.clone()
                };
                if let Some(old) = previous.get(&row.key)
                    && old.chunk == row.chunk
                    && old.plain() == row.plain()
                {
                    row.revision = old.revision;
                }
                row
            })
            .collect()
    }
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
        if let Some(code) = &chunk.code {
            return self.source()[code.source.clone()].to_owned();
        }
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
    pub(crate) fn code_visible(&self) -> Option<Range<usize>> {
        self.chunk
            .as_ref()?
            .code
            .as_ref()
            .map(|code| code.visible.clone())
    }
    pub(crate) fn preparation_range(&self) -> Option<Range<usize>> {
        self.chunk.as_ref().map(|c| {
            c.code
                .as_ref()
                .map_or_else(|| c.body.clone(), |code| code.source.clone())
        })
    }
    fn rich(&self) -> bool {
        matches!(self.message.role, Role::Assistant | Role::Tool)
            && self.chunk.as_ref().is_some_and(|chunk| !chunk.literal)
    }
}

struct Presentation {
    source: Option<Range<usize>>,
    revision: u64,
    dark: bool,
    selection: MessageSelection,
    prepared: Option<Arc<PreparedMarkdown>>,
    settled: bool,
}
struct InFlight {
    source_revision: Option<u64>,
    append_source: Option<Arc<Message>>,
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
    cache: PreparationCache,
    heights: HashMap<RowKey, f32>,
    rejected: HashSet<CacheKey>,
    visible: HashSet<RowKey>,
    cache_sweeper: Option<Task<()>>,
    last_frame: Option<Instant>,
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
        let list = ListState::new(0, ListAlignment::Top, px(OVERSCAN));
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
            cache: PreparationCache::new(PRESENTATION_BYTES),
            heights: HashMap::new(),
            rejected: HashSet::new(),
            visible: HashSet::new(),
            cache_sweeper: None,
            last_frame: None,
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
    pub(crate) fn namespace(&self) -> &str {
        &self.namespace
    }

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
        self.heights.clear();
        self.rejected.clear();
        self.visible.clear();
        self.cache.priorities.clear();
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
                // Assistant content does not expand/collapse. Keep its semantic row
                // boundaries and matching presentations when another message toggles.
                && (self.overlays_revision == overlays.revision() || message.role == Role::Assistant)
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
            if !lineage_changed
                && message.role == Role::Assistant
                && message.text.len() <= MAX_INDEX_SOURCE_BYTES
                && previous_rows
                    .first()
                    .is_some_and(|old| message.text.starts_with(&old.message.text))
                && previous_rows.iter().any(|row| {
                    self.presentations
                        .get(&row.key)
                        .is_some_and(|p| p.prepared.is_some())
                })
            {
                // Keep source locators and prepared content from the same snapshot.
                // The index worker will publish the next demanded presentation atomically.
                self.rows.extend(previous_rows.into_iter().map(|mut row| {
                    row.message_index = index;
                    if row.chunk.is_none() {
                        row.message = message.clone();
                        row.revision = message.revision;
                    }
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
                let mut chunks = if message.role == Role::Tool {
                    text_chunks(&message.text)
                } else {
                    source_chunks(&message.text)
                };
                if !lineage_changed
                    && message.role == Role::Assistant
                    && previous_rows
                        .first()
                        .is_some_and(|old| message.text.starts_with(&old.message.text))
                {
                    let mut stable = previous_rows
                        .iter()
                        .filter_map(|row| row.chunk.as_ref())
                        .take_while(|chunk| chunk.gap_before.is_some())
                        .cloned()
                        .collect::<Vec<_>>();
                    // Match StreamingMarkdownState: the final two logical blocks
                    // can still absorb new list items, fences or paragraph text.
                    let mut starts = stable
                        .iter()
                        .map(|c| {
                            c.code
                                .as_ref()
                                .map_or(c.body.start, |code| code.source.start)
                        })
                        .collect::<Vec<_>>();
                    starts.dedup();
                    if let Some(tail_start) = starts.len().checked_sub(2).map(|i| starts[i]) {
                        stable.retain(|c| c.range.end <= tail_start);
                        let mut tail = source_chunks(&message.text[tail_start..]);
                        for chunk in &mut tail {
                            chunk.range =
                                chunk.range.start + tail_start..chunk.range.end + tail_start;
                            chunk.body = chunk.body.start + tail_start..chunk.body.end + tail_start;
                        }
                        stable.extend(tail);
                        chunks = stable;
                    }
                }
                // A large document containing only paragraph lines needs no global AST.
                // Reject every possible container opener/setext/table delimiter before
                // taking this fast path; mixed Markdown still uses the semantic worker.
                let paragraphs = message.role == Role::Assistant && paragraph_only(&message.text);
                let oversized = message.text.len() > MAX_INDEX_SOURCE_BYTES && !paragraphs;
                if message.role == Role::Assistant && (paragraphs || oversized) {
                    for (i, chunk) in chunks.iter_mut().enumerate() {
                        chunk.gap_before = Some(if i == 0 { 0 } else { 16 });
                        // Bound the input to whole-document parsing, not only the retained result.
                        chunk.literal |= oversized;
                    }
                }
                if message.role == Role::Assistant
                    && let Some(Cached::Index(cached)) =
                        self.cache.peek(&self.cache_key(&chrome, true))
                {
                    chunks = cached.clone();
                }
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
    fn install_index(&mut self, row: &ChatRow, chunks: Vec<SourceChunk>) {
        let anchor = self.anchor();
        let Some(start) = self
            .rows
            .iter()
            .position(|r| r.key.message == row.key.message && r.chunk.is_some())
        else {
            return;
        };
        let end = start
            + self.rows[start..]
                .iter()
                .take_while(|r| r.key.message == row.key.message && r.chunk.is_some())
                .count();
        let count = chunks.len();
        let replacement = row.indexed(chunks, &self.rows[start..end]);
        let revisions = replacement
            .iter()
            .map(|r| (r.key, r.revision))
            .collect::<HashMap<_, _>>();
        self.rows.splice(start..end, replacement);
        self.presentations.retain(|key, entry| {
            key.message != row.key.message || revisions.get(key) == Some(&entry.revision)
        });
        self.requested.clear();
        self.list.splice(start..end, count);
        self.pending_anchor = Some(anchor);
        self.restore_pending();
    }
    fn latest_message<'a>(&'a self, row: &'a ChatRow) -> &'a Arc<Message> {
        self.messages
            .get(row.message_index)
            .filter(|message| message.key == row.key.message)
            .unwrap_or(&row.message)
    }
    fn pending_index(&self, row: &ChatRow) -> bool {
        row.message.role == Role::Assistant
            && row.message.revision != self.latest_message(row).revision
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

    fn cache_key(&self, row: &ChatRow, index: bool) -> CacheKey {
        CacheKey {
            namespace: self.namespace.clone(),
            lineage: self.lineage,
            message: row.key.message,
            revision: if index {
                row.message.revision
            } else {
                row.revision
            },
            field: if index { 1 } else { row.key.field },
            source: if index {
                0..row.message.text.len()
            } else {
                row.preparation_range().unwrap_or_default()
            },
            dark: !index && self.dark,
            index,
        }
    }

    /// Native overdraw may reuse heights without calling our renderer. Declare demand
    /// from geometry as well, so measured neighbours still get prepared and retained.
    fn refresh_demand(&mut self, now: Instant) {
        let top = self.list.logical_scroll_top();
        let viewport = self.list.viewport_bounds();
        let previous_visible = std::mem::take(&mut self.visible);
        let mut y = -f32::from(top.offset_in_item);
        for index in top.item_ix..self.rows.len() {
            if y >= f32::from(viewport.size.height) + OVERSCAN {
                break;
            }
            let row = &self.rows[index];
            let height = if let Some(bounds) = self.list.bounds_for_item(index) {
                let height = f32::from(bounds.size.height).max(1.0);
                self.heights.insert(row.key, height);
                height
            } else {
                self.estimated_height(row)
            };
            self.requested.insert(row.key, index);
            if y < f32::from(viewport.size.height) && y + height > 0.0 {
                self.visible.insert(row.key);
            }
            y += height;
        }
        let mut above = f32::from(top.offset_in_item);
        for index in (0..top.item_ix.min(self.rows.len())).rev() {
            if above >= OVERSCAN {
                break;
            }
            let row = &self.rows[index];
            above += self.estimated_height(row);
            self.requested.insert(row.key, index);
        }
        self.heights
            .retain(|key, _| self.requested.contains_key(key));
        let wanted = self
            .requested
            .values()
            .map(|i| self.cache_key(&self.rows[*i], false))
            .collect::<HashSet<_>>();
        let entering = self
            .visible
            .difference(&previous_visible)
            .filter_map(|key| self.requested.get(key))
            .map(|index| self.cache_key(&self.rows[*index], false))
            .collect::<HashSet<_>>();
        // A prefetch rejected under pressure gets another chance on entering the
        // viewport, where its admission priority is higher.
        self.rejected
            .retain(|key| wanted.contains(key) && !entering.contains(key));
        self.cache.priorities.clear();
        for index in self.requested.values() {
            let row = &self.rows[*index];
            let key = self.cache_key(row, false);
            let priority = if self.visible.contains(&row.key) {
                2
            } else {
                1
            };
            for key in [key, self.cache_key(row, true)] {
                let entry = self.cache.priorities.entry(key).or_default();
                *entry = (*entry).max(priority);
            }
            self.cache.get(&self.cache_key(row, true), now);
            self.cache.get(&self.cache_key(row, false), now);
        }
    }

    fn estimated_height(&self, row: &ChatRow) -> f32 {
        // GPUI exposes no bounds above the scroll top. Reuse the last observed height;
        // unseen rows use a bounded source-line estimate until native layout measures them.
        self.heights.get(&row.key).copied().unwrap_or_else(|| {
            (row.plain().lines().take(CHUNK_LINES).count().max(1) as f32 * 26.0).max(24.0)
        })
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
        let cache_key = self.cache_key(&row, false);
        let rejected = self.rejected.contains(&cache_key);
        let cached = match self.cache.get(&cache_key, cx.background_executor().now()) {
            Some(Cached::Markdown(prepared)) => Some(prepared.clone()),
            _ => None,
        };
        let shared = cached.or_else(|| {
            self.presentations.iter().find_map(|(key, entry)| {
                (key.message == row.key.message
                    && entry.source == row.preparation_range()
                    && entry.revision == row.revision
                    && entry.dark == self.dark)
                    .then(|| entry.prepared.clone())
                    .flatten()
            })
        });
        let entry = self
            .presentations
            .entry(row.key)
            .or_insert_with(|| Presentation {
                source: row.preparation_range(),
                revision: row.revision,
                dark: self.dark,
                selection: MessageSelection::new(window, cx),
                prepared: None,
                settled: !row.rich(),
            });
        if entry.revision != row.revision || entry.dark != self.dark {
            entry.source = row.preparation_range();
            entry.revision = row.revision;
            entry.dark = self.dark;
            entry.prepared = None;
            entry.settled = !row.rich();
        }
        if entry.prepared.is_none() && shared.is_some() {
            entry.prepared = shared;
            entry.settled = true;
        }
        entry.settled |= rejected;
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
    pub(crate) fn unsettled_chunks(&self) -> usize {
        self.presentations
            .values()
            .filter(|entry| !entry.settled)
            .count()
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

    #[cfg(test)]
    pub(crate) fn cache_bytes(&self) -> usize {
        self.cache.bytes
    }

    #[cfg(test)]
    pub(crate) fn set_cache_budget(&mut self, bytes: usize) {
        self.release();
        self.cache = PreparationCache::new(bytes);
    }

    #[cfg(test)]
    pub(crate) fn demanded_unprepared(&self) -> usize {
        self.requested
            .values()
            .filter(|i| {
                let row = &self.rows[**i];
                let key = self.cache_key(row, false);
                row.rich() && self.cache.peek(&key).is_none() && !self.rejected.contains(&key)
            })
            .count()
    }
}

impl DesktopApp {
    /// Called after list layout, never while GPUI holds ListState's mutable borrow.
    pub(crate) fn finish_chat_frame(&mut self, cx: &mut Context<Self>) {
        let executor = cx.background_executor().clone();
        let now = executor.now();
        if self.chat.borrow().cache_sweeper.is_none() {
            self.chat.borrow_mut().cache_sweeper = Some(cx.spawn(async move |this, cx| {
                loop {
                    executor.timer(Duration::from_secs(30)).await;
                    if this
                        .update(cx, |this, _| {
                            let mut chat = this.chat.borrow_mut();
                            let now = executor.now();
                            // Idle windows keep the text being read, but release offscreen
                            // entities too so they cannot pin cache entries indefinitely.
                            if chat
                                .last_frame
                                .is_none_or(|last| now.saturating_duration_since(last) >= IDLE_TTL)
                            {
                                let visible = chat.visible.clone();
                                chat.presentations.retain(|key, _| visible.contains(key));
                                chat.requested.retain(|key, _| visible.contains(key));
                                chat.cache.priorities.retain(|_, priority| *priority == 2);
                            }
                            chat.cache.expire(now);
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }));
        }
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
        // Completion also drains the queue through this method. Only a native frame
        // changes geometry or counts as viewport activity.
        if chat.demand_scheduled {
            chat.last_frame = Some(now);
            chat.refresh_demand(now);
        }
        chat.demand_scheduled = false;
        let requested = chat.requested.clone();
        chat.presentations
            .retain(|key, _| requested.contains_key(key));
        let visible = chat.visible.clone();
        for (key, entry) in &mut chat.presentations {
            if !visible.contains(key) {
                entry.prepared = None;
            }
        }
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
                    .is_some_and(|row| {
                        row.revision == work.revision
                            && work.source_revision.is_none_or(|revision| {
                                chat.latest_message(row).revision == revision
                                    || work.append_source.as_ref().is_some_and(|source| {
                                        chat.latest_message(row).text.starts_with(&source.text)
                                    })
                            })
                    })
            {
                work.cancel.store(true, Ordering::Relaxed);
            }
            return;
        }
        let top = chat.list.logical_scroll_top().item_ix;
        let viewport = chat.list.viewport_bounds();
        let row = requested
            .iter()
            .filter(|(_, index)| {
                let row = &chat.rows[**index];
                let cache_key = chat.cache_key(row, false);
                chat.pending_index(row)
                    || (row.rich()
                        && !chat.rejected.contains(&cache_key)
                        && chat.cache.peek(&cache_key).is_none())
            })
            .min_by_key(|(_, index)| {
                let visible = chat.list.bounds_for_item(**index).is_some_and(|bounds| {
                    bounds.bottom() > viewport.origin.y && bounds.origin.y < viewport.bottom()
                });
                (!visible, index.abs_diff(top))
            })
            .and_then(|(_, index)| chat.rows.get(*index))
            .cloned();
        let Some(mut row) = row else {
            return;
        };
        let key = row.key;
        let epoch = chat.epoch;
        let revision = row.revision;
        let dark = chat.dark;
        let updating = chat.pending_index(&row);
        let needs_index = updating
            || (row.message.role == Role::Assistant
                && row
                    .chunk
                    .as_ref()
                    .is_some_and(|chunk| chunk.gap_before.is_none()));
        let previous = if updating {
            chat.rows
                .iter()
                .filter(|old| old.key.message == key.message)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut demand = row
            .chunk
            .as_ref()
            .map(|chunk| chunk.range.clone())
            .unwrap_or_default();
        let mut reusable = HashMap::new();
        if updating {
            for old in previous
                .iter()
                .filter(|old| requested.contains_key(&old.key))
            {
                if let Some(chunk) = &old.chunk {
                    demand.start = demand.start.min(chunk.range.start);
                    demand.end = demand.end.max(chunk.range.end);
                    if let Some(Cached::Markdown(prepared)) =
                        chat.cache.peek(&chat.cache_key(old, false))
                    {
                        reusable.insert((old.revision, old.preparation_range()), prepared.clone());
                    }
                }
            }
            if demand.end == row.message.text.len() {
                demand.end = chat.latest_message(&row).text.len();
            }
            row.message = chat.latest_message(&row).clone();
        }
        let cache_key = chat.cache_key(&row, needs_index);
        // Do not parse an arbitrarily large logical code block just to reject its
        // result afterwards. All visible slices fall back to readable source.
        if !needs_index
            && row.code_visible().is_some()
            && row
                .preparation_range()
                .is_some_and(|range| range.len() > MAX_CODE_SOURCE_BYTES)
        {
            chat.rejected.insert(cache_key);
            for (_, entry) in chat.presentations.iter_mut().filter(|(key, entry)| {
                key.message == row.key.message && entry.source == row.preparation_range()
            }) {
                entry.settled = true;
            }
            drop(chat);
            self.finish_chat_frame(cx);
            return;
        }
        let source = if needs_index {
            row.message.text.clone()
        } else {
            row.markdown_source()
        };
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
        let source_revision = needs_index.then_some(row.message.revision);
        let append_source = updating.then(|| row.message.clone());
        let task = cx.spawn(async move |this, cx| {
            let worker_row = row.clone();
            let (indexed, prepared, replacements) = executor
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
                    if needs_index {
                        let indexed = semantic_chunks(&source, &worker_cancel);
                        let mut replacements = Vec::new();
                        let mut bytes = 0;
                        if updating && let Some(chunks) = &indexed {
                            for next in worker_row.indexed(chunks.clone(), &previous) {
                                if worker_cancel.load(Ordering::Relaxed) {
                                    break;
                                }
                                let Some(chunk) = &next.chunk else { continue };
                                if chunk.range.end <= demand.start
                                    || chunk.range.start >= demand.end
                                    || !next.rich()
                                {
                                    continue;
                                }
                                let range = next.preparation_range();
                                let limit = if next.code_visible().is_some() {
                                    PRESENTATION_BYTES
                                } else {
                                    MAX_CHUNK_PRESENTATION_BYTES
                                };
                                let prepared = reusable
                                    .get(&(next.revision, range.clone()))
                                    .cloned()
                                    .or_else(|| {
                                        if next.code_visible().is_some()
                                            && range.as_ref().is_some_and(|range| {
                                                range.len() > MAX_CODE_SOURCE_BYTES
                                            })
                                        {
                                            return None;
                                        }
                                        let prepared = dsh_markdown::prepare_markdown(
                                            &next.markdown_source(),
                                            &theme,
                                            &worker_cancel,
                                        )?;
                                        if prepared.bytes() > limit
                                            || bytes + prepared.bytes() > PRESENTATION_BYTES
                                        {
                                            return None;
                                        }
                                        bytes += prepared.bytes();
                                        let prepared = Arc::new(prepared);
                                        reusable.insert((next.revision, range), prepared.clone());
                                        Some(prepared)
                                    });
                                replacements.push((next, prepared));
                            }
                        }
                        (indexed, None, replacements)
                    } else {
                        (
                            None,
                            dsh_markdown::prepare_markdown(&source, &theme, &worker_cancel),
                            Vec::new(),
                        )
                    }
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
                    if needs_index
                        && chat.latest_message(&chat.rows[index]).revision != row.message.revision
                        && !(updating
                            && chat
                                .latest_message(&chat.rows[index])
                                .text
                                .starts_with(&row.message.text))
                    {
                        cx.notify();
                        return;
                    }
                    if let Some(chunks) = indexed {
                        let visible_ranges = chat
                            .rows
                            .iter()
                            .filter(|old| {
                                old.key.message == key.message && chat.visible.contains(&old.key)
                            })
                            .filter_map(|old| old.chunk.as_ref().map(|chunk| chunk.range.clone()))
                            .collect::<Vec<_>>();
                        if updating {
                            // Superseded keys must not protect an obsolete revision against
                            // admission of the replacement. Live frame leases remain protected.
                            let namespace = chat.namespace.clone();
                            chat.cache.priorities.retain(|cached, _| {
                                cached.namespace != namespace || cached.message != key.message
                            });
                        }
                        chat.cache.insert(
                            cache_key.clone(),
                            Cached::Index(chunks.clone()),
                            cx.background_executor().now(),
                        );
                        chat.install_index(&row, chunks);
                        for (replacement, prepared) in replacements {
                            let key = chat.cache_key(&replacement, false);
                            let visible = replacement.chunk.as_ref().is_some_and(|chunk| {
                                visible_ranges.iter().any(|range| {
                                    range.start < chunk.range.end && chunk.range.start < range.end
                                })
                            });
                            chat.cache
                                .priorities
                                .insert(key.clone(), if visible { 2 } else { 1 });
                            if !prepared.is_some_and(|prepared| {
                                chat.cache.insert(
                                    key.clone(),
                                    Cached::Markdown(prepared),
                                    cx.background_executor().now(),
                                )
                            }) {
                                chat.rejected.insert(key);
                            }
                        }
                        cx.notify();
                        return;
                    }
                    let prepared = prepared
                        .filter(|value| {
                            value.bytes()
                                <= if row.code_visible().is_some() {
                                    PRESENTATION_BYTES
                                } else {
                                    MAX_CHUNK_PRESENTATION_BYTES
                                }
                        })
                        .map(Arc::new);
                    let prepared = prepared.filter(|value| {
                        chat.cache.insert(
                            cache_key.clone(),
                            Cached::Markdown(value.clone()),
                            cx.background_executor().now(),
                        )
                    });
                    if prepared.is_none() {
                        chat.rejected.insert(cache_key);
                    }
                    if let Some(entry) = chat.presentations.get_mut(&key) {
                        entry.prepared = prepared.clone();
                        entry.settled = true;
                    }
                    chat.list.remeasure_items(index..index + 1);
                    // Publish shared code to all demanded slices before advancing the
                    // queue; they must not each start a whole-block syntax parse.
                    if row.code_visible().is_some() {
                        let shared = prepared;
                        let siblings = chat
                            .requested
                            .iter()
                            .filter_map(|(candidate, index)| {
                                let other = chat.rows.get(*index)?;
                                (candidate != &key
                                    && candidate.message == key.message
                                    && other.revision == revision
                                    && other.preparation_range() == row.preparation_range())
                                .then_some((*candidate, *index))
                            })
                            .collect::<Vec<_>>();
                        for (sibling, index) in siblings {
                            if let Some(entry) = chat.presentations.get_mut(&sibling) {
                                entry.prepared = shared.clone();
                                entry.settled = true;
                            }
                            chat.list.remeasure_items(index..index + 1);
                        }
                    }
                }
                // A cached/throttled native window need not redraw immediately on
                // notify. Drain already-declared demand without waiting for another
                // render callback (or for the user to resize the window).
                drop(chat);
                this.finish_chat_frame(cx);
                cx.notify();
            });
        });
        chat.in_flight = Some(InFlight {
            source_revision,
            append_source,
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

    fn indexed_row(source: &str, chunk: SourceChunk) -> ChatRow {
        ChatRow {
            key: RowKey {
                message: MessageId(1),
                field: 1,
                start: chunk.range.start,
            },
            message: message(source.to_owned(), 1),
            message_index: 0,
            revision: 1,
            chunk: Some(chunk),
        }
    }

    #[test]
    fn html_documents_are_atomic_and_keep_distinct_row_identities() {
        let html = format!(
            "<style>body {{ color: blue }}</style>\n{}<script>let x = 1;</script>",
            "<p>中文</p>\n".repeat(300)
        );
        let source =
            format!("Before\n\n```html\n{html}\n```\n\nBetween\n\n```HTML\n{html}\n```\n\nAfter");
        let chunks = semantic_chunks(&source, &AtomicBool::new(false)).unwrap();
        let rows = chunks
            .into_iter()
            .map(|c| indexed_row(&source, c))
            .collect::<Vec<_>>();
        assert_eq!(rows.iter().map(ChatRow::plain).collect::<String>(), source);
        let documents = rows
            .iter()
            .filter(|r| r.code_visible().is_some())
            .collect::<Vec<_>>();
        assert_eq!(documents.len(), 2);
        assert_ne!(documents[0].key, documents[1].key);
        for row in documents {
            let prepared = dsh_markdown::prepare_markdown(
                &row.markdown_source(),
                markdown_highlight_theme(false),
                &AtomicBool::new(false),
            )
            .unwrap();
            assert_eq!(prepared.html_document(), Some(html.as_str()));
        }
    }

    #[test]
    fn semantic_code_ranges_match_reparsed_indented_fences() {
        for indent in 0..=3 {
            let pad = " ".repeat(indent);
            for value in ["中文\n".to_owned(), "let 中文 = 1;\n".repeat(80)] {
                let source = format!(
                    "Introduction.\n\n{pad}```rust\n{}{pad}```\n\nEnding.",
                    value
                        .lines()
                        .map(|line| format!("{pad}{line}\n"))
                        .collect::<String>()
                );
                let chunks = semantic_chunks(&source, &AtomicBool::new(false)).unwrap();
                let mut chat = ChatViewport::default();
                let overlays = MessagePresentationStore::default();
                chat.sync(
                    &Vector::unit(message(source.clone(), 1)),
                    &Vector::new(),
                    &overlays,
                    1,
                    false,
                );
                let row = chat.rows[0].clone();
                chat.install_index(&row, chunks.clone());
                let updated = format!("{source} More text.");
                chat.sync(
                    &Vector::unit(message(updated.clone(), 2)),
                    &Vector::new(),
                    &overlays,
                    1,
                    false,
                );
                assert_eq!(
                    chat.rows.iter().map(ChatRow::plain).collect::<String>(),
                    updated,
                    "streaming must retain the paragraph before an indented code block"
                );
                let mut displayed = String::new();
                for chunk in chunks {
                    let row = indexed_row(&source, chunk);
                    let Some(visible) = row.code_visible() else {
                        continue;
                    };
                    let mut parsed = crate::streaming_markdown::StreamingMarkdownState::default();
                    parsed.update(&row.markdown_source());
                    let markdown::mdast::Node::Code(code) = &parsed.tail_blocks()[0].node else {
                        panic!("code fragment must remain code");
                    };
                    assert_eq!(code.value, value.trim_end_matches('\n'), "indent={indent}");
                    displayed.push_str(&code.value[visible]);
                }
                assert_eq!(displayed, value.trim_end_matches('\n'));
            }
        }
    }

    #[test]
    fn semantic_short_messages_preserve_reference_context() {
        fn references(node: &markdown::mdast::Node) -> usize {
            usize::from(matches!(
                node,
                markdown::mdast::Node::LinkReference(_) | markdown::mdast::Node::ImageReference(_)
            )) + node
                .children()
                .map_or(0, |children| children.iter().map(references).sum())
        }
        for source in [
            "[label][id]\n\n[id]: https://example.com",
            "[label][id]\n\n> [id]: https://example.com",
            "![image][id]\n\n[id]: https://example.com/image.png",
        ] {
            let count: usize = semantic_chunks(source, &AtomicBool::new(false))
                .unwrap()
                .into_iter()
                .map(|chunk| {
                    let row = indexed_row(source, chunk);
                    let mut parsed = crate::streaming_markdown::StreamingMarkdownState::default();
                    parsed.update(&row.markdown_source());
                    parsed
                        .frozen()
                        .iter()
                        .chain(parsed.tail_blocks())
                        .map(|block| references(&block.node))
                        .sum::<usize>()
                })
                .sum();
            assert_eq!(count, 1, "{source}");
        }
    }

    #[test]
    fn semantic_chunks_preserve_loose_lists_and_section_spacing() {
        let source = "**5. Browser**\n- first\n\n**6. Ecosystem**\n- second\n\n- third\n\n    nested paragraph\n\n> quote\n>\n> continuation";
        let chunks = semantic_chunks(source, &AtomicBool::new(false)).unwrap();
        assert_eq!(chunks.len(), 5);
        assert_eq!(
            chunks
                .iter()
                .map(|c| c.gap_before.unwrap())
                .collect::<Vec<_>>(),
            vec![0, 8, 24, 8, 16]
        );
        assert!(source[chunks[3].body.clone()].contains("nested paragraph"));
        assert!(source[chunks[4].body.clone()].contains("continuation"));
        let padded = format!("{}\n\n{source}", "intro ".repeat(400));
        let indexed = semantic_chunks(&padded, &AtomicBool::new(false)).unwrap();
        assert_eq!(indexed.len(), chunks.len() + 1);
        for (original, shifted) in chunks.iter().zip(&indexed[1..]) {
            assert_eq!(
                &source[original.body.clone()],
                &padded[shifted.body.clone()]
            );
        }
        assert_eq!(
            indexed[2..]
                .iter()
                .map(|c| c.gap_before)
                .collect::<Vec<_>>(),
            chunks[1..].iter().map(|c| c.gap_before).collect::<Vec<_>>()
        );
    }
    #[test]
    fn streaming_retains_completed_semantic_rows() {
        let source = "A paragraph with **emphasis**.\n\n".repeat(200);
        let mut chat = ChatViewport::default();
        let overlays = MessagePresentationStore::default();
        chat.sync(
            &Vector::unit(message(source.clone(), 1)),
            &Vector::new(),
            &overlays,
            1,
            false,
        );
        let row = chat.rows[0].clone();
        chat.install_index(
            &row,
            semantic_chunks(&source, &AtomicBool::new(false)).unwrap(),
        );
        let stable = chat.rows[50].chunk.clone();
        let updated = format!("{source}New tail");
        chat.sync(
            &Vector::unit(message(updated.clone(), 2)),
            &Vector::new(),
            &overlays,
            1,
            false,
        );
        assert_eq!(
            chat.rows[50].chunk, stable,
            "streaming must not turn settled prose back into raw Markdown"
        );
        let row = chat
            .rows
            .iter()
            .find(|r| r.chunk.as_ref().is_some_and(|c| c.gap_before.is_none()))
            .unwrap()
            .clone();
        chat.install_index(
            &row,
            semantic_chunks(&updated, &AtomicBool::new(false)).unwrap(),
        );
        assert_eq!(
            chat.rows[50].revision, 1,
            "publishing the new index must preserve unchanged presentations"
        );
    }

    #[test]
    fn semantic_code_slices_share_the_full_source_and_preserve_bytes() {
        let value = format!(
            "/* open\n{}close */\nlet value = 42;",
            "中文 comment\n".repeat(80)
        );
        let source = format!("```rust\n{value}\n```");
        let chunks = semantic_chunks(&source, &AtomicBool::new(false)).unwrap();
        assert!(chunks.len() > 2);
        assert_eq!(
            chunks
                .iter()
                .map(|c| &source[c.range.clone()])
                .collect::<String>(),
            source
        );
        assert_eq!(
            chunks
                .iter()
                .map(|c| &value[c.code.as_ref().unwrap().visible.clone()])
                .collect::<String>(),
            value
        );
        for chunk in chunks {
            let row = indexed_row(&source, chunk);
            assert_eq!(row.markdown_source(), source);
        }
        let paragraphs = "A **long** paragraph.\n\n".repeat(4000);
        assert!(paragraph_only(&paragraphs));
        for container in [
            "- item",
            "> quote",
            "    code",
            "---",
            "1. item",
            "| cell |",
            "```",
            "[id]: /url",
        ] {
            assert!(!paragraph_only(&format!("{paragraphs}{container}")));
        }
    }

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
