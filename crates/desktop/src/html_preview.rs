//! Window-owned live HTML documents. Virtual list rows only mount their visible rectangles.
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use gpui_kit::component::ActiveTheme;
use gpui_kit::component::IconName;
use gpui_kit::component::button::{Button, ButtonVariants};
use gpui_kit::{
    AnyElement, App, Bounds, Context, Element, ElementId, GlobalElementId, InspectorElementId,
    InteractiveElement, IntoElement, LayoutId, ParentElement, Pixels, StatefulInteractiveElement,
    Styled, Task, Window, accesskit::Role, div, prelude::FluentBuilder, px,
};
use serde::Deserialize;
use tokio::sync::mpsc;

#[cfg(any(test, debug_assertions))]
use crate::domain::MessageId;
use crate::{app::DesktopApp, platform::gpui::RowKey, ui_theme::palette};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(not(target_os = "macos"))]
mod occlusion;

const INITIAL_HEIGHT: f32 = 240.0;
const MAX_HEIGHT: f32 = 480.0;
type Snapshot = tokio::sync::oneshot::Receiver<Result<Vec<u8>, String>>;

pub(crate) fn is_html(language: &str) -> bool {
    language.eq_ignore_ascii_case("html") || language.eq_ignore_ascii_case("htm")
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    full: Bounds<Pixels>,
    clip: Bounds<Pixels>,
    occlusion: Option<Bounds<Pixels>>,
}

impl Placement {
    fn new(full: Bounds<Pixels>, mask: Bounds<Pixels>) -> Option<Self> {
        let clip = full.intersect(&mask);
        (clip.size.width > px(0.0) && clip.size.height > px(0.0)).then_some(Self {
            full,
            clip,
            occlusion: None,
        })
    }

    fn contains(self, point: gpui_kit::Point<Pixels>) -> bool {
        self.clip.contains(&point)
            && !self
                .occlusion
                .is_some_and(|hole| pill_contains(hole, point))
    }

    fn needs_layout(self, previous: Option<Self>, source_mode: bool) -> bool {
        // AppKit moves/clips the native view without changing the iframe viewport.
        // Source mode alone needs its inset synchronized with the visible clip.
        previous.is_none_or(|old| {
            old.full.size != self.full.size
                || ((!cfg!(target_os = "macos") || source_mode)
                    && (old.full != self.full || old.clip != self.clip))
        })
    }
}

fn pill_contains(bounds: Bounds<Pixels>, point: gpui_kit::Point<Pixels>) -> bool {
    let radius = bounds.size.height.min(bounds.size.width) / 2.0;
    let center_x = point
        .x
        .clamp(bounds.left() + radius, bounds.right() - radius);
    let center_y = point
        .y
        .clamp(bounds.top() + radius, bounds.bottom() - radius);
    let dx = f32::from(point.x - center_x);
    let dy = f32::from(point.y - center_y);
    dx * dx + dy * dy <= f32::from(radius).powi(2)
}

struct Preview {
    source: String,
    source_prefix: String,
    generation: u64,
    height: f32,
    source_mode: bool,
    placement: Option<Placement>,
    applied: Option<Placement>,
    browser: Option<NativeBrowser>,
    error: Option<String>,
    loaded: bool,
    dark: bool,
    dirty: bool,
    applied_mode: Option<(bool, bool)>,
    source_revision: Option<u64>,
    source_task: Option<Task<()>>,
    #[cfg(test)]
    source_gate: Option<tokio::sync::oneshot::Receiver<()>>,
}
impl Preview {
    fn new(source: String, source_prefix: String, generation: u64, dark: bool) -> Self {
        Self {
            source,
            source_prefix,
            generation,
            height: INITIAL_HEIGHT,
            source_mode: false,
            placement: None,
            applied: None,
            browser: None,
            error: None,
            loaded: false,
            dark,
            dirty: false,
            applied_mode: None,
            source_revision: None,
            source_task: None,
            #[cfg(test)]
            source_gate: None,
        }
    }

    fn replace_source(&mut self, source: String, prefix: String, generation: u64) {
        self.source = source;
        self.source_prefix = prefix;
        self.generation = generation;
        self.error = None;
        self.dirty = true;
    }
}
impl Drop for Preview {
    fn drop(&mut self) {
        if let Some(browser) = &self.browser {
            let _ = browser.set_visible(false);
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "kind", rename_all = "camelCase")]
enum BrowserEvent {
    Ready,
    Action { action: PreviewAction },
    Height { height: f32 },
    Wheel { x: f32, y: f32, dx: f32, dy: f32 },
    Error { message: String },
}
#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
enum PreviewAction {
    Download,
    Expand,
    Source,
    Dismiss,
}

struct Envelope {
    key: RowKey,
    generation: u64,
    event: BrowserEvent,
}
#[derive(Deserialize)]
struct BrowserMessage {
    token: String,
    generation: u64,
    #[serde(flatten)]
    event: BrowserEvent,
}
struct Store {
    namespace: String,
    lineage: u64,
    next_generation: u64,
    #[cfg(target_os = "macos")]
    accessibility: Option<macos::Accessibility>,
    #[cfg(target_os = "macos")]
    cursor: Option<macos::CursorOwner>,
    entries: HashMap<RowKey, Preview>,
    expanded: Option<RowKey>,
    sidebar: Option<(RowKey, Preview)>,
    button_occlusion: Option<Bounds<Pixels>>,
    sender: mpsc::Sender<Envelope>,
}

impl Store {
    fn preview_mut(&mut self, key: RowKey, generation: u64) -> Option<(&mut Preview, bool)> {
        if let Some((sidebar_key, preview)) = &mut self.sidebar
            && *sidebar_key == key
            && preview.generation == generation
        {
            return Some((preview, true));
        }
        self.entries
            .get_mut(&key)
            .filter(|p| p.generation == generation)
            .map(|p| (p, false))
    }
}

pub(crate) struct HtmlPreviews {
    store: Rc<RefCell<Store>>,
    _events: Task<()>,
    #[cfg(target_os = "linux")]
    _gtk_events: Task<()>,
}

impl HtmlPreviews {
    pub(crate) fn new(window: &Window, cx: &mut Context<DesktopApp>) -> Self {
        let (sender, mut receiver) = mpsc::channel::<Envelope>(256);
        let events = cx.spawn_in(window, async move |owner, cx| {
            while let Some(message) = receiver.recv().await {
                if owner
                    .update_in(cx, |app, window, cx| {
                        let mut store = app.html_previews.store.borrow_mut();
                        let Some((preview, enlarged)) =
                            store.preview_mut(message.key, message.generation)
                        else {
                            return;
                        };
                        match message.event {
                            BrowserEvent::Ready => {
                                preview.loaded = true;
                                preview.applied = None;
                                preview.applied_mode = None;
                                window.refresh();
                            }
                            BrowserEvent::Action { action } => {
                                match action {
                                    PreviewAction::Download => {
                                        drop(store);
                                        app.export_html_preview(
                                            message.key,
                                            message.generation,
                                            window,
                                            cx,
                                        );
                                    }
                                    PreviewAction::Source => {
                                        preview.source_mode = !preview.source_mode;
                                    }
                                    PreviewAction::Expand => {
                                        store.expanded = (store.expanded != Some(message.key))
                                            .then_some(message.key);
                                    }
                                    PreviewAction::Dismiss => {
                                        if store.expanded == Some(message.key) {
                                            store.expanded = None;
                                        }
                                    }
                                }
                                window.refresh();
                            }
                            BrowserEvent::Height { height } => {
                                if !enlarged && height.is_finite() {
                                    let height = height.ceil().clamp(64.0, MAX_HEIGHT);
                                    if (height - preview.height).abs() >= 1.0 {
                                        preview.height = height;
                                        let chat = app.chat.borrow();
                                        if let Some(index) =
                                            chat.rows.iter().position(|r| r.key == message.key)
                                        {
                                            chat.list.remeasure_items(index..index + 1);
                                        }
                                        cx.notify();
                                    }
                                }
                            }
                            BrowserEvent::Wheel { x, y, dx, dy } => {
                                // A sidebar owns its scrolling, including at both boundaries.
                                if !enlarged
                                    && preview.applied.is_some()
                                    && [x, y, dx, dy].iter().all(|v| v.is_finite())
                                {
                                    // The row may have moved since the browser sent this event.
                                    // Route to its owner instead of hit-testing stale coordinates.
                                    let position =
                                        app.chat.borrow().list.viewport_bounds().center();
                                    let event = gpui_kit::ScrollWheelEvent {
                                        position,
                                        delta: gpui_kit::ScrollDelta::Pixels(gpui_kit::point(
                                            px(-dx.clamp(-2000.0, 2000.0)),
                                            px(-dy.clamp(-2000.0, 2000.0)),
                                        )),
                                        ..Default::default()
                                    };
                                    drop(store);
                                    window.defer(cx, move |window, cx| {
                                        window.dispatch_event(
                                            gpui_kit::PlatformInput::ScrollWheel(event),
                                            cx,
                                        );
                                    });
                                }
                            }
                            BrowserEvent::Error { message: error } => {
                                preview.error = Some(error.chars().take(240).collect());
                                let chat = app.chat.borrow();
                                if let Some(index) =
                                    chat.rows.iter().position(|r| r.key == message.key)
                                {
                                    chat.list.remeasure_items(index..index + 1);
                                }
                                cx.notify();
                            }
                        }
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
        #[cfg(target_os = "linux")]
        let gtk_events = cx.spawn(async move |_, cx| {
            loop {
                cx.background_executor()
                    .timer(std::time::Duration::from_millis(8))
                    .await;
                if gtk::is_initialized_main_thread() {
                    let context = gtk::glib::MainContext::default();
                    for _ in 0..32 {
                        if !context.pending() {
                            break;
                        }
                        context.iteration(false);
                    }
                }
            }
        });
        Self {
            store: Rc::new(RefCell::new(Store {
                namespace: String::new(),
                lineage: 0,
                next_generation: 0,
                #[cfg(target_os = "macos")]
                accessibility: None,
                #[cfg(target_os = "macos")]
                cursor: None,
                entries: HashMap::new(),
                expanded: None,
                sidebar: None,
                button_occlusion: None,
                sender,
            })),
            _events: events,
            #[cfg(target_os = "linux")]
            _gtk_events: gtk_events,
        }
    }

    pub(crate) fn sync<'a>(
        &self,
        namespace: &str,
        lineage: u64,
        messages: impl Iterator<Item = &'a crate::domain::Message>,
    ) {
        let mut store = self.store.borrow_mut();
        if store.namespace != namespace || store.lineage != lineage {
            store.sidebar = None;
            store.entries.clear();
            store.expanded = None;
            store.namespace = namespace.to_owned();
            store.lineage = lineage;
        }
        if store.entries.is_empty() {
            return;
        }
        let messages: HashMap<_, _> = messages.map(|message| (message.key, message)).collect();
        store.entries.retain(|key, preview| {
            messages
                .get(&key.message)
                .and_then(|message| message.text.get(key.start..))
                .is_some_and(|source| source.starts_with(&preview.source_prefix))
        });
        if store
            .expanded
            .is_some_and(|key| !store.entries.contains_key(&key))
        {
            store.expanded = None;
        }
    }

    pub(crate) fn render(
        &self,
        key: RowKey,
        html: &str,
        source_prefix: &str,
        selection: &crate::platform::gpui::SelectionFrame,
        cx: &mut App,
    ) -> AnyElement {
        let colors = palette(cx);
        let dark = cx.theme().is_dark();
        let mut store = self.store.borrow_mut();
        if store.entries.get(&key).is_none_or(|p| p.source != html) {
            store.next_generation += 1;
            let generation = store.next_generation;
            if let Some(preview) = store.entries.get_mut(&key) {
                preview.replace_source(
                    html.to_owned(),
                    source_prefix.trim_end().to_owned(),
                    generation,
                );
            } else {
                store.entries.insert(
                    key,
                    Preview::new(
                        html.to_owned(),
                        source_prefix.trim_end().to_owned(),
                        generation,
                        dark,
                    ),
                );
            }
        }
        let Some(preview) = store.entries.get_mut(&key) else {
            return div().into_any_element();
        };
        if preview.dark != dark {
            preview.dark = dark;
            if let Some(browser) = &preview.browser {
                let _ = browser.evaluate_script(&format!("window.previewTheme({dark})"));
            }
        }
        let height = preview.height;
        let error = preview.error.clone();
        let unavailable = error.is_some() && preview.browser.is_none();
        drop(store);
        div()
            .w_full()
            .min_w(px(0.0))
            .rounded(px(12.0))
            .overflow_hidden()
            .when_some(error, |body, error| {
                body.child(
                    div()
                        .px_4()
                        .py_2()
                        .text_xs()
                        .text_color(colors.danger)
                        .child(format!("Preview: {error}")),
                )
            })
            .child(if unavailable {
                div()
                    .p_4()
                    .text_sm()
                    .font_family(cx.theme().mono_font_family.clone())
                    .child(crate::dsh_markdown::plain_text(
                        html.to_owned().into(),
                        Some(selection),
                    ))
                    .into_any_element()
            } else {
                div()
                    .h(px(height))
                    .w_full()
                    .child(PreviewMount {
                        key,
                        expanded: false,
                        store: self.store.clone(),
                    })
                    .into_any_element()
            })
            .into_any_element()
    }

    pub(crate) fn retained_source(&self, key: RowKey) -> Option<String> {
        self.store
            .borrow()
            .entries
            .get(&key)
            .map(|entry| entry.source.clone())
    }

    pub(crate) fn sidebar(
        &self,
        messages: &im::Vector<std::sync::Arc<crate::domain::Message>>,
        cx: &mut Context<DesktopApp>,
    ) -> Option<AnyElement> {
        let mut state = self.store.borrow_mut();
        let Some(key) = state.expanded else {
            state.sidebar = None;
            return None;
        };
        let inline = state.entries.get(&key)?;
        if state.sidebar.as_ref().is_none_or(|(old, _)| *old != key) {
            let source = inline.source.clone();
            let prefix = inline.source_prefix.clone();
            state.next_generation += 1;
            state.sidebar = Some((
                key,
                Preview::new(source, prefix, state.next_generation, cx.theme().is_dark()),
            ));
        }
        let message = messages.iter().find(|message| message.key == key.message)?;
        let namespace = state.namespace.clone();
        let lineage = state.lineage;
        let (_, preview) = state.sidebar.as_mut()?;
        if preview.source_revision != Some(message.revision) && preview.source_task.is_none() {
            let generation = preview.generation;
            preview.source_revision = Some(message.revision);
            let message = message.clone();
            let executor = cx.background_executor().clone();
            #[cfg(test)]
            let dispatcher = executor.clone();
            #[cfg(test)]
            let gate = preview.source_gate.take();
            let task =
                cx.spawn(async move |owner, cx| {
                    let input = message.clone();
                    let result = executor
                        .spawn(async move {
                            #[cfg(test)]
                            if let Some(gate) = gate {
                                let _ = gate.await;
                            }
                            #[cfg(test)]
                            assert!(
                                !dispatcher.is_main_thread(),
                                "sidebar parsing must stay off the UI thread"
                            );
                            sidebar_document(input.text.get(key.start..)?)
                        })
                        .await;
                    let _ =
                        owner.update(cx, |app, cx| {
                            let mut store = app.html_previews.store.borrow_mut();
                            let Some((preview, true)) = store.preview_mut(key, generation) else {
                                return;
                            };
                            preview.source_task = None;
                            // A completed append may advance the sidebar while newer input is queued.
                            // Rewrites, session changes and replacement sidebar instances cannot.
                            if app.chat.borrow().namespace() != namespace
                                || app.core.session_view.trajectory.projection_lineage() != lineage
                                || !app.core.session_view.conversation.messages.iter().any(
                                    |current| {
                                        current.key == key.message
                                            && current.text.starts_with(&message.text)
                                    },
                                )
                            {
                                preview.source_revision = None;
                                cx.notify();
                                return;
                            }
                            let Some((source, prefix)) = result else {
                                store.sidebar = None;
                                store.expanded = None;
                                cx.notify();
                                return;
                            };
                            if preview.source != source {
                                store.next_generation += 1;
                                let next_generation = store.next_generation;
                                if let Some((preview, true)) = store.preview_mut(key, generation) {
                                    preview.replace_source(source, prefix, next_generation);
                                }
                            } else {
                                preview.source_prefix = prefix;
                            }
                            cx.notify();
                        });
                });
            preview.source_task = Some(task);
        }
        drop(state);
        let colors = palette(cx);
        let store = self.store.clone();
        Some(
            div()
                .id("html-preview-sidebar")
                .role(Role::Complementary)
                .aria_label("HTML preview")
                .flex()
                .flex_col()
                .size_full()
                .min_w(px(0.0))
                .border_l_1()
                .border_color(colors.border)
                .bg(colors.surface)
                .child(
                    div()
                        .flex()
                        .flex_none()
                        .items_center()
                        .justify_between()
                        .h(px(crate::ui_theme::metrics::TITLEBAR_HEIGHT))
                        .px_3()
                        .border_b_1()
                        .border_color(colors.border)
                        .text_sm()
                        .child("HTML preview")
                        .child(
                            Button::new("close-html-preview")
                                .icon(IconName::Close)
                                .accessibility_label("Close preview sidebar")
                                .ghost()
                                .compact()
                                .tooltip("Close preview sidebar")
                                .on_click(move |_, window, _| {
                                    store.borrow_mut().expanded = None;
                                    window.refresh();
                                }),
                        ),
                )
                .child(
                    div()
                        .flex_1()
                        .min_h(px(0.0))
                        .overflow_hidden()
                        .child(PreviewMount {
                            key,
                            expanded: true,
                            store: self.store.clone(),
                        }),
                )
                .into_any_element(),
        )
    }

    pub(crate) fn frame(&self, child: impl IntoElement, covered: bool) -> impl IntoElement {
        PreviewFrame {
            child: child.into_any_element(),
            store: self.store.clone(),
            covered,
        }
    }

    pub(crate) fn button_occlusion(&self) -> impl IntoElement {
        let store = self.store.clone();
        gpui_kit::canvas(
            |_, _, _| {},
            move |bounds, _, _, _| store.borrow_mut().button_occlusion = Some(bounds),
        )
        .absolute()
        .size_full()
    }
}

fn sidebar_document(source: &str) -> Option<(String, String)> {
    use crate::platform::gpui::{MAX_CODE_SOURCE_BYTES, MAX_INDEX_SOURCE_BYTES};
    if source.len() > MAX_INDEX_SOURCE_BYTES {
        return None;
    }
    let mut parsed = crate::streaming_markdown::StreamingMarkdownState::default();
    parsed.update(source);
    let block = parsed.frozen().iter().chain(parsed.tail_blocks()).next()?;
    if let markdown::mdast::Node::Code(code) = &block.node
        && is_html(code.lang.as_deref().unwrap_or_default())
        && block.source.len() <= MAX_CODE_SOURCE_BYTES
    {
        return Some((
            code.value.clone(),
            source[..block.key + block.source.len()]
                .trim_end()
                .to_owned(),
        ));
    }
    None
}

impl DesktopApp {
    fn export_html_preview(
        &mut self,
        key: RowKey,
        generation: u64,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let snapshot = self
            .html_previews
            .store
            .borrow_mut()
            .preview_mut(key, generation)
            .and_then(|(entry, _)| entry.browser.as_ref())
            .map(|browser| browser.snapshot());
        let Some(snapshot) = snapshot else { return };
        cx.spawn_in(window, async move |owner, cx| {
            let result = snapshot
                .await
                .map_err(|error| error.to_string())
                .and_then(|result| result);
            let selection = owner
                .update_in(cx, |app, _, cx| {
                    if let Some(entry) = app
                        .html_previews
                        .store
                        .borrow_mut()
                        .preview_mut(key, generation)
                        .map(|(entry, _)| entry)
                        && let Some(browser) = &entry.browser
                    {
                        let _ = browser.evaluate_script("window.previewCaptureFinished()");
                    }
                    match &result {
                        Ok(_) => Some(cx.prompt_for_new_path(
                            &app.core.workspace.cwd,
                            Some("html-preview.png"),
                        )),
                        Err(error) => {
                            app.notice(format!("Could not save preview: {error}"));
                            cx.notify();
                            None
                        }
                    }
                })
                .ok()
                .flatten();
            let (Ok(bytes), Some(selection)) = (result, selection) else {
                return;
            };
            let saved = match selection.await {
                Ok(Ok(Some(path))) => {
                    tokio::task::spawn_blocking(move || std::fs::write(path, bytes))
                        .await
                        .map_err(|error| error.to_string())
                        .and_then(|result| result.map_err(|error| error.to_string()))
                }
                Ok(Ok(None)) => return,
                Ok(Err(error)) => Err(error.to_string()),
                Err(error) => Err(error.to_string()),
            };
            if let Err(error) = saved {
                let _ = owner.update_in(cx, |app, _, cx| {
                    app.notice(format!("Could not save preview: {error}"));
                    cx.notify();
                });
            }
        })
        .detach();
    }
}

#[allow(clippy::expect_used, reason = "serializing a string cannot fail")]
fn json_script(value: &str) -> String {
    serde_json::to_string(value)
        .expect("strings serialize")
        .replace('<', "\\u003c")
}

fn document(source: &str, dark: bool, generation: u64, token: &str) -> String {
    let boot = include_str!("html_preview/document.js");
    let html = format!(
        "<!doctype html><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'\"><style>html{{color-scheme:{}}}body{{margin:16px;font:14px/1.5 system-ui;overflow-wrap:anywhere}}img,svg,canvas{{max-width:100%}}</style><script>{boot}</script>{source}",
        if dark { "dark" } else { "light" }
    );
    include_str!("html_preview/host.html")
        .replace("__GENERATION__", &generation.to_string())
        .replace("__TOKEN__", &json_script(token))
        .replace("__DARK__", if dark { "true" } else { "false" })
        .replace("__DOCUMENT__", &json_script(&html))
        .replacen("__SOURCE__", &json_script(source), 1)
}

/// Load a local Markdown fixture in the complete debug app, with no provider or journal writes.
#[cfg(debug_assertions)]
pub(crate) fn native_fixture(mut app: DesktopApp) -> DesktopApp {
    let Some(path) = std::env::var_os("KCASTLE_PREVIEW_MARKDOWN") else {
        return app;
    };
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("HTML preview fixture: {error}");
            return app;
        }
    };
    let mut snapshot = crate::domain::SessionView::default();
    snapshot
        .conversation
        .messages
        .push_back(std::sync::Arc::new(crate::domain::Message {
            key: MessageId(u64::MAX),
            revision: 0,
            role: crate::domain::Role::Assistant,
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
        }));
    app.core.session_view = std::sync::Arc::new(snapshot);
    app.chat.get_mut().pending_anchor = Some(crate::layout::ScrollAnchor::Block {
        id: MessageId(u64::MAX),
        field: 1,
        source_offset: 0,
        local_offset: 0.0,
    });
    app
}

fn create_browser(
    source: &str,
    dark: bool,
    key: RowKey,
    generation: u64,
    sender: mpsc::Sender<Envelope>,
    window: &Window,
) -> Result<NativeBrowser, String> {
    #[cfg(target_os = "linux")]
    let parent = linux_parent(window)?;
    #[cfg(target_os = "macos")]
    let clip = macos::ClipView::new(window)?;
    // Native WebKit/GTK bridges can also be exposed inside subframes. Only the
    // opaque sandbox's trusted parent knows this per-browser capability token.
    let token = kcastle_agent::SessionId::new().to_string();
    let expected_token = token.clone();
    let builder = wry::WebViewBuilder::new()
        .with_html(document(source, dark, generation, &token))
        .with_visible(false)
        .with_focused(false)
        .with_incognito(true)
        .with_devtools(false)
        .with_clipboard(false)
        .with_accept_first_mouse(true)
        .with_navigation_handler(|url| {
            matches!(url.split('#').next(), Some("about:blank" | "about:srcdoc"))
        })
        .with_new_window_req_handler(|_, _| wry::NewWindowResponse::Deny)
        .with_download_started_handler(|_, _| false)
        .with_drag_drop_handler(|_| true)
        .with_ipc_handler(move |request| {
            if request.body().len() > 4096 {
                return;
            }
            if let Ok(message) = serde_json::from_str::<BrowserMessage>(request.body())
                && message.token == expected_token
            {
                let _ = sender.try_send(Envelope {
                    key,
                    generation: message.generation,
                    event: message.event,
                });
            }
        });
    #[cfg(target_os = "macos")]
    let view = builder.build_as_child(&clip);
    #[cfg(target_os = "linux")]
    let view = builder.build_as_child(&parent);
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let view = builder.build_as_child(window);
    Ok(NativeBrowser {
        token,
        view: view.map_err(|error| error.to_string())?,
        #[cfg(target_os = "macos")]
        clip,
    })
}

#[cfg(target_os = "linux")]
fn linux_parent(window: &Window) -> Result<wry::raw_window_handle::WindowHandle<'_>, String> {
    use gtk::prelude::Cast;
    use wry::raw_window_handle::{
        HasWindowHandle, RawWindowHandle, WindowHandle, XlibWindowHandle,
    };
    let raw = HasWindowHandle::window_handle(window)
        .map_err(|error| error.to_string())?
        .as_raw();
    let raw =
        match raw {
            // GPUI uses XCB; Wry accepts the same server-owned window ID through Xlib.
            RawWindowHandle::Xcb(handle) => {
                RawWindowHandle::Xlib(XlibWindowHandle::new(handle.window.get().into()))
            }
            RawWindowHandle::Xlib(_) => raw,
            _ => return Err(
                "Inline HTML currently requires an X11 window on Linux; source remains available."
                    .into(),
            ),
        };
    if !gtk::is_initialized() {
        gtk::gdk::set_allowed_backends("x11");
        gtk::init().map_err(|error| error.to_string())?;
    }
    if gtk::gdk::Display::default()
        .and_then(|display| display.downcast::<gdkx11::X11Display>().ok())
        .is_none()
    {
        return Err("Inline HTML requires the GTK X11 display backend.".into());
    }
    // Changing the API wrapper does not change the XID or its lifetime, owned by GPUI's window.
    Ok(unsafe { WindowHandle::borrow_raw(raw) })
}

struct NativeBrowser {
    token: String,
    // Drop WebKit before removing its retained native parent.
    view: wry::WebView,
    #[cfg(target_os = "macos")]
    clip: macos::ClipView,
}
impl std::ops::Deref for NativeBrowser {
    type Target = wry::WebView;
    fn deref(&self) -> &Self::Target {
        &self.view
    }
}
#[cfg(target_os = "macos")]
impl Drop for NativeBrowser {
    fn drop(&mut self) {
        self.release_focus();
    }
}
impl NativeBrowser {
    fn snapshot(&self) -> Snapshot {
        #[cfg(target_os = "macos")]
        {
            macos::snapshot(&self.view)
        }
        #[cfg(target_os = "linux")]
        {
            use webkit2gtk::{SnapshotOptions, SnapshotRegion, WebViewExt};
            use wry::WebViewExtUnix;
            let (sender, receiver) = tokio::sync::oneshot::channel();
            self.view.webview().snapshot(
                SnapshotRegion::FullDocument,
                SnapshotOptions::empty(),
                None::<&gtk::gio::Cancellable>,
                move |result| {
                    let result = result
                        .map_err(|error| error.to_string())
                        .and_then(|surface| {
                            let mut bytes = Vec::new();
                            surface
                                .write_to_png(&mut bytes)
                                .map_err(|error| error.to_string())?;
                            Ok(bytes)
                        });
                    let _ = sender.send(result);
                },
            );
            receiver
        }
        #[cfg(target_os = "windows")]
        {
            use webview2_com::{
                CapturePreviewCompletedHandler,
                Microsoft::Web::WebView2::Win32::COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG,
            };
            use windows::Win32::{
                System::Com::{STATFLAG_NONAME, STATSTG, STREAM_SEEK_SET},
                UI::Shell::SHCreateMemStream,
            };
            use wry::WebViewExtWindows;
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let sender = Rc::new(RefCell::new(Some(sender)));
            let result = (|| -> Result<(), String> {
                // COM owns an in-memory PNG stream for the duration of the callback.
                let stream =
                    unsafe { SHCreateMemStream(None) }.ok_or("Could not create image stream")?;
                let output = stream.clone();
                let completion = sender.clone();
                let callback = CapturePreviewCompletedHandler::create(Box::new(move |result| {
                    let result = result.map_err(|error| error.to_string()).and_then(|()| {
                        let mut stat = STATSTG::default();
                        unsafe { output.Stat(&mut stat, STATFLAG_NONAME) }
                            .map_err(|error| error.to_string())?;
                        if stat.cbSize > 64 * 1024 * 1024 {
                            return Err("Preview image is too large".into());
                        }
                        let mut bytes = vec![0u8; stat.cbSize as usize];
                        let mut read = 0;
                        unsafe {
                            output
                                .Seek(0, STREAM_SEEK_SET, None)
                                .map_err(|error| error.to_string())?;
                            output
                                .Read(
                                    bytes.as_mut_ptr().cast(),
                                    bytes.len() as u32,
                                    Some(&mut read),
                                )
                                .ok()
                                .map_err(|error| error.to_string())?;
                        }
                        if read as usize != bytes.len() {
                            return Err("Incomplete preview image".into());
                        }
                        Ok(bytes)
                    });
                    if let Some(sender) = completion.borrow_mut().take() {
                        let _ = sender.send(result);
                    }
                    Ok(())
                }));
                unsafe {
                    self.view.webview().CapturePreview(
                        COREWEBVIEW2_CAPTURE_PREVIEW_IMAGE_FORMAT_PNG,
                        &stream,
                        &callback,
                    )
                }
                .map_err(|error| error.to_string())
            })();
            if let Err(error) = result
                && let Some(sender) = sender.borrow_mut().take()
            {
                let _ = sender.send(Err(error));
            }
            receiver
        }
    }
    fn release_focus(&self) {
        #[cfg(target_os = "macos")]
        self.clip.release_focus();
        #[cfg(not(target_os = "macos"))]
        let _ = self.view.focus_parent();
    }

    fn place(
        &self,
        placement: Placement,
        previous: Option<Placement>,
        source_mode: bool,
    ) -> Result<(), wry::Error> {
        let full = placement.full;
        let clip = placement.clip;
        #[cfg(target_os = "macos")]
        let (bounds, x, y) = {
            self.clip.place(placement);
            (Bounds::new(full.origin - clip.origin, full.size), 0.0, 0.0)
        };
        #[cfg(not(target_os = "macos"))]
        let (bounds, x, y) = (
            clip,
            f32::from(full.origin.x - clip.origin.x),
            f32::from(full.origin.y - clip.origin.y),
        );
        if placement.needs_layout(previous, source_mode) {
            let (top, bottom) = if cfg!(target_os = "macos") {
                (
                    f32::from(clip.top() - full.top()),
                    f32::from(full.bottom() - clip.bottom()),
                )
            } else {
                (0.0, 0.0)
            };
            self.view.evaluate_script(&format!(
                "window.previewLayout({},{},{x},{y},{top},{bottom})",
                f32::from(full.size.width),
                f32::from(full.size.height)
            ))?;
        }
        self.view.set_bounds(wry::Rect {
            position: wry::dpi::LogicalPosition::new(
                f64::from(bounds.origin.x),
                f64::from(bounds.origin.y),
            )
            .into(),
            size: wry::dpi::LogicalSize::new(
                f64::from(bounds.size.width),
                f64::from(bounds.size.height),
            )
            .into(),
        })?;
        #[cfg(not(target_os = "macos"))]
        occlusion::apply(&self.view, placement)?;
        self.view.set_visible(true)
    }
}

struct PreviewMount {
    key: RowKey,
    expanded: bool,
    store: Rc<RefCell<Store>>,
}
impl IntoElement for PreviewMount {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}
impl Element for PreviewMount {
    type RequestLayoutState = ();
    type PrepaintState = ();
    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }
    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (
            window.request_layout(
                gpui_kit::Style {
                    size: gpui_kit::Size::full(),
                    ..Default::default()
                },
                [],
                cx,
            ),
            (),
        )
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut Window,
        _: &mut App,
    ) {
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        _: &mut App,
    ) {
        let mut store = self.store.borrow_mut();
        let entry = if self.expanded {
            store
                .sidebar
                .as_mut()
                .filter(|(key, _)| *key == self.key)
                .map(|(_, p)| p)
        } else {
            store.entries.get_mut(&self.key)
        };
        if let Some(entry) = entry {
            entry.placement = Placement::new(bounds, window.content_mask().bounds);
        }
    }
}

struct PreviewFrame {
    child: AnyElement,
    store: Rc<RefCell<Store>>,
    covered: bool,
}
impl IntoElement for PreviewFrame {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}
impl Element for PreviewFrame {
    type RequestLayoutState = ();
    type PrepaintState = ();
    fn id(&self) -> Option<ElementId> {
        None
    }
    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }
    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        (self.child.request_layout(window, cx), ())
    }
    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.prepaint(window, cx);
    }
    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        {
            let mut store = self.store.borrow_mut();
            store.button_occlusion = None;
            let Store {
                entries, sidebar, ..
            } = &mut *store;
            for entry in entries
                .values_mut()
                .chain(sidebar.iter_mut().map(|(_, p)| p))
            {
                entry.placement = None;
            }
        }
        self.child.paint(window, cx);
        let events = self.store.clone();
        window.on_mouse_event(move |event: &gpui_kit::MouseDownEvent, phase, _, _| {
            if phase != gpui_kit::DispatchPhase::Capture {
                return;
            }
            let store = events.borrow();
            if !store
                .entries
                .values()
                .chain(store.sidebar.iter().map(|(_, p)| p))
                .any(|entry| entry.applied.is_some_and(|p| p.contains(event.position)))
            {
                for browser in store
                    .entries
                    .values()
                    .chain(store.sidebar.iter().map(|(_, p)| p))
                    .filter_map(|entry| entry.browser.as_ref())
                {
                    browser.release_focus();
                }
            }
        });
        #[cfg(target_os = "macos")]
        if !cfg!(test) {
            let store = self.store.clone();
            window.defer(cx, move |_, _| {
                let mut store = store.borrow_mut();
                let Store {
                    entries,
                    sidebar,
                    accessibility,
                    cursor,
                    ..
                } = &mut *store;
                macos::Accessibility::update(
                    accessibility,
                    entries.values().chain(sidebar.iter().map(|(_, p)| p)),
                );
                macos::CursorOwner::update(
                    cursor,
                    entries.values().chain(sidebar.iter().map(|(_, p)| p)),
                );
            });
        }
        let mut store = self.store.borrow_mut();
        let sender = store.sender.clone();
        let button_occlusion = store.button_occlusion;
        let Store {
            entries, sidebar, ..
        } = &mut *store;
        for (key, entry, enlarged) in entries
            .iter_mut()
            .map(|(&key, p)| (key, p, false))
            .chain(sidebar.iter_mut().map(|(key, p)| (*key, p, true)))
        {
            entry.placement = entry.placement.map(|mut placement| {
                placement.occlusion = button_occlusion.filter(|hole| {
                    let overlap = placement.clip.intersect(hole);
                    overlap.size.width > px(0.0) && overlap.size.height > px(0.0)
                });
                placement
            });
            let placement = if self.covered { None } else { entry.placement };
            let Some(placement) = placement else {
                if entry.applied.take().is_some()
                    && let Some(browser) = &entry.browser
                {
                    #[cfg(target_os = "macos")]
                    browser.release_focus();
                    let _ = browser.set_visible(false);
                }
                continue;
            };
            // Headless GPUI windows have no native handle; geometry still exercises this path.
            if !cfg!(test) && entry.browser.is_none() && entry.error.is_none() {
                match create_browser(
                    &entry.source,
                    entry.dark,
                    key,
                    entry.generation,
                    sender.clone(),
                    window,
                ) {
                    Ok(browser) => {
                        entry.browser = Some(browser);
                        entry.dirty = false;
                    }
                    Err(error) => {
                        entry.error = Some(error);
                        window.defer(cx, |window, _| window.refresh());
                    }
                }
            }
            let Some(browser) = &entry.browser else {
                continue;
            };
            if entry.dirty {
                entry.dirty = false;
                entry.loaded = false;
                entry.applied = None;
                browser.release_focus();
                let _ = browser.set_visible(false);
                if let Err(error) = browser.load_html(&document(
                    &entry.source,
                    entry.dark,
                    entry.generation,
                    &browser.token,
                )) {
                    entry.error = Some(error.to_string());
                    window.defer(cx, |window, _| window.refresh());
                }
            }
            let dark = cx.theme().is_dark();
            if entry.dark != dark {
                entry.dark = dark;
                let _ = browser.evaluate_script(&format!("window.previewTheme({dark})"));
            }
            let mode = (enlarged, entry.source_mode);
            if entry.loaded && entry.applied_mode != Some(mode) {
                let _ =
                    browser.evaluate_script(&format!("window.previewMode({},{})", mode.0, mode.1));
                entry.applied_mode = Some(mode);
                // Entering source mode needs current clip insets even without a resize.
                entry.applied = None;
            }
            if !entry.loaded || entry.applied == Some(placement) {
                continue;
            }
            if let Err(error) = browser.place(placement, entry.applied, entry.source_mode) {
                entry.error = Some(error.to_string());
                window.defer(cx, |window, _| window.refresh());
            } else {
                entry.applied = Some(placement);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui_kit::{TestAppContext, size};

    #[cfg(target_os = "macos")]
    #[test]
    fn transcript_scroll_does_not_send_browser_layout_commands() {
        let mask = Bounds::new(
            gpui_kit::point(px(0.0), px(80.0)),
            size(px(800.0), px(500.0)),
        );
        let mut full = Bounds::new(
            gpui_kit::point(px(20.0), px(120.0)),
            size(px(600.0), px(462.0)),
        );
        let initial = Placement::new(full, mask).unwrap();
        assert!(initial.needs_layout(None, false));
        full.origin.y -= px(180.0);
        let scrolled = Placement::new(full, mask).unwrap();
        assert!(
            !scrolled.needs_layout(Some(initial), false),
            "native clipping must not enqueue JS layout work on every wheel frame"
        );
        assert!(
            scrolled.needs_layout(Some(initial), true),
            "source mode still follows the clip"
        );
        full.size.width -= px(80.0);
        assert!(
            Placement::new(full, mask)
                .unwrap()
                .needs_layout(Some(scrolled), false)
        );
    }

    #[gpui_kit::test]
    fn multiple_previews_survive_virtualization_and_reject_old_callbacks(cx: &mut TestAppContext) {
        let root =
            std::env::temp_dir().join(format!("kcastle-html-{}", kcastle_agent::SessionId::new()));
        let (startup, _) = crate::desktop_startup(root.clone()).unwrap();
        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let mut app = DesktopApp::new(startup, window, cx);
            let mut snapshot = crate::domain::SessionView::default();
            snapshot
                .conversation
                .messages
                .push_back(std::sync::Arc::new(crate::domain::Message {
                    key: MessageId(80000),
                    revision: 0,
                    role: crate::domain::Role::Assistant,
                    text: include_str!("../tests/fixtures/html-previews.md").into(),
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
                }));
            app.core.session_view = std::sync::Arc::new(snapshot);
            app.chat.get_mut().pending_anchor = Some(crate::layout::ScrollAnchor::Block {
                id: MessageId(80000),
                field: 1,
                source_offset: 0,
                local_offset: 0.0,
            });
            window.blur(cx);
            app
        });
        cx.simulate_resize(size(px(1180.0), px(900.0)));
        cx.run_until_parked();
        let (keys, sender) = view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            assert_eq!(store.entries.len(), 2);
            assert!(
                store
                    .entries
                    .values()
                    .all(|entry| entry.placement.is_some())
            );
            (
                store
                    .entries
                    .iter()
                    .map(|(&key, entry)| (key, entry.generation))
                    .collect::<Vec<_>>(),
                store.sender.clone(),
            )
        });
        for action in [PreviewAction::Expand, PreviewAction::Source] {
            sender
                .try_send(Envelope {
                    key: keys[0].0,
                    generation: keys[0].1,
                    event: BrowserEvent::Action { action },
                })
                .unwrap();
        }
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            assert_eq!(store.expanded, Some(keys[0].0));
            assert!(store.entries[&keys[0].0].source_mode);
            let sidebar = store.sidebar.as_ref().unwrap().1.placement.unwrap().full;
            let inline = store.entries[&keys[1].0].placement.unwrap().full;
            assert_eq!(sidebar.right(), px(1180.0));
            assert_eq!(sidebar.top(), px(40.0));
            assert_eq!(sidebar.bottom(), px(900.0));
            assert!(inline.right() < sidebar.left());
            let original = store.entries[&keys[0].0].placement.unwrap().full;
            assert!(
                original.right() < sidebar.left(),
                "the original stays visible beside its sidebar copy"
            );
            assert_ne!(store.sidebar.as_ref().unwrap().1.generation, keys[0].1);
            assert_eq!(
                store.entries.len(),
                2,
                "enlarging must retain the other document"
            );
        });
        let sidebar_generation = view.read_with(cx, |app, _| {
            app.html_previews
                .store
                .borrow()
                .sidebar
                .as_ref()
                .unwrap()
                .1
                .generation
        });
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: sidebar_generation,
                event: BrowserEvent::Height { height: 1500.0 },
            })
            .unwrap();
        cx.run_until_parked();
        view.update(cx, |app, cx| {
            assert_eq!(
                app.html_previews.store.borrow().entries[&keys[0].0].height,
                INITIAL_HEIGHT,
                "sidebar measurements must not resize the inline preview"
            );
            app.chat.borrow().list.scroll_to_end();
            cx.notify();
        });
        for width in [720.0, 1400.0, 1180.0] {
            cx.simulate_resize(size(px(width), px(900.0)));
            cx.run_until_parked();
            view.read_with(cx, |app, _| {
                let store = app.html_previews.store.borrow();
                let sidebar = store.sidebar.as_ref().unwrap().1.placement.unwrap().full;
                assert_eq!(sidebar.right(), px(width));
                assert!(sidebar.size.width >= px(280.0));
                assert!(sidebar.left() >= px(320.0));
                assert_eq!(store.entries[&keys[0].0].generation, keys[0].1);
            });
        }
        // Selecting another document reuses the same sidebar; a late dismissal from
        // the previous document must not close its replacement.
        let before = view.update(cx, |app, _| {
            let mut store = app.html_previews.store.borrow_mut();
            let preview = &mut store.sidebar.as_mut().unwrap().1;
            // Headless windows have no native browser to publish the applied rectangle.
            preview.applied = preview.placement;
            app.chat.borrow().list.logical_scroll_top()
        });
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: sidebar_generation,
                event: BrowserEvent::Wheel {
                    x: 50.0,
                    y: 50.0,
                    dx: 0.0,
                    dy: -160.0,
                },
            })
            .unwrap();
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let after = app.chat.borrow().list.logical_scroll_top();
            assert_eq!(
                (after.item_ix, after.offset_in_item),
                (before.item_ix, before.offset_in_item),
                "a sidebar boundary wheel must not scroll the transcript"
            );
        });
        for (key, action) in [
            (keys[1], PreviewAction::Expand),
            (keys[0], PreviewAction::Dismiss),
        ] {
            sender
                .try_send(Envelope {
                    key: key.0,
                    generation: key.1,
                    event: BrowserEvent::Action { action },
                })
                .unwrap();
        }
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            assert_eq!(store.expanded, Some(keys[1].0));
            assert_eq!(
                store
                    .sidebar
                    .as_ref()
                    .unwrap()
                    .1
                    .placement
                    .unwrap()
                    .full
                    .right(),
                px(1180.0)
            );
        });
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: keys[0].1,
                event: BrowserEvent::Action {
                    action: PreviewAction::Expand,
                },
            })
            .unwrap();
        view.update(cx, |app, cx| {
            app.chat
                .borrow()
                .list
                .scroll_to(gpui_kit::ListOffset::default());
            cx.notify();
        });
        cx.run_until_parked();
        // Reopening the same row must not accept the retired sidebar's callbacks.
        for event in [
            BrowserEvent::Height { height: 999.0 },
            BrowserEvent::Action {
                action: PreviewAction::Dismiss,
            },
        ] {
            sender
                .try_send(Envelope {
                    key: keys[0].0,
                    generation: sidebar_generation,
                    event,
                })
                .unwrap();
        }
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            assert_eq!(store.expanded, Some(keys[0].0));
            assert_ne!(
                store.sidebar.as_ref().unwrap().1.generation,
                sidebar_generation
            );
            assert_eq!(store.entries[&keys[0].0].height, INITIAL_HEIGHT);
        });
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: keys[0].1,
                event: BrowserEvent::Action {
                    action: PreviewAction::Dismiss,
                },
            })
            .unwrap();
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            assert_eq!(store.expanded, None);
            assert!(store.entries[&keys[0].0].source_mode);
            assert_eq!(store.entries[&keys[0].0].generation, keys[0].1);
        });
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: keys[0].1,
                event: BrowserEvent::Height { height: 1500.0 },
            })
            .unwrap();
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert_eq!(
                app.html_previews.store.borrow().entries[&keys[0].0].height,
                480.0,
                "long inline content must have a bounded scrolling viewport"
            );
        });
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: keys[0].1,
                event: BrowserEvent::Height { height: 360.0 },
            })
            .unwrap();
        cx.run_until_parked();
        view.update(cx, |app, cx| {
            assert_eq!(
                app.html_previews.store.borrow().entries[&keys[0].0]
                    .placement
                    .unwrap()
                    .full
                    .size
                    .height,
                px(360.0),
                "a height callback must resize the mounted native document"
            );
            assert_eq!(
                app.html_previews.store.borrow().entries[&keys[0].0].height,
                360.0
            );
            app.chat.borrow().list.scroll_to_end();
            cx.notify();
        });
        cx.run_until_parked();
        view.update(cx, |app, cx| {
            let store = app.html_previews.store.borrow();
            for &(key, generation) in &keys {
                assert_eq!(store.entries[&key].generation, generation);
            }
            drop(store);
            app.chat
                .borrow()
                .list
                .scroll_to(gpui_kit::ListOffset::default());
            cx.notify();
        });
        cx.run_until_parked();
        view.update(cx, |app, cx| {
            assert_eq!(
                app.html_previews.store.borrow().entries[&keys[0].0].height,
                360.0
            );
            app.chat.get_mut().activate("different-session".into());
            app.chat.get_mut().pending_anchor = Some(crate::layout::ScrollAnchor::Block {
                id: MessageId(80000),
                field: 1,
                source_offset: 0,
                local_offset: 0.0,
            });
            cx.notify();
        });
        cx.run_until_parked();
        sender
            .try_send(Envelope {
                key: keys[0].0,
                generation: keys[0].1,
                event: BrowserEvent::Height { height: 999.0 },
            })
            .unwrap();
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            assert!(store.entries[&keys[0].0].generation > keys[0].1);
            assert_eq!(store.entries[&keys[0].0].height, INITIAL_HEIGHT);
            assert_eq!(store.expanded, None);
        });
        view.update(cx, |app, cx| {
            app.chat.get_mut().set_cache_budget(0);
            cx.notify();
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert!(
                app.html_previews
                    .store
                    .borrow()
                    .entries
                    .values()
                    .all(|entry| entry.placement.is_some()),
                "a live document must not revert to source when the Markdown cache is evicted"
            );
        });
        // Append-only streaming replaces document contents, not the preview allocation or layout.
        for revision in 0..5 {
            view.update(cx, |app, cx| {
                if revision == 0 {
                    app.chat.get_mut().set_cache_budget(8 * 1024 * 1024);
                    app.chat.get_mut().activate("streaming".into());
                }
                let mut snapshot = (*app.core.session_view).clone();
                let mut message = (**snapshot.conversation.messages.front().unwrap()).clone();
                message.revision += 1;
                message.text = format!(
                    "```html\n<h3>Streaming</h3>{}",
                    "<p>Next</p>".repeat(revision)
                );
                if revision == 4 {
                    message.text.push_str("\n```");
                }
                snapshot.conversation.messages.clear();
                snapshot
                    .conversation
                    .messages
                    .push_back(std::sync::Arc::new(message));
                app.core.session_view = std::sync::Arc::new(snapshot);
                app.chat.get_mut().pending_anchor = Some(crate::layout::ScrollAnchor::Block {
                    id: MessageId(80000),
                    field: 1,
                    source_offset: 0,
                    local_offset: 0.0,
                });
                cx.notify();
            });
            cx.run_until_parked();
            view.update(cx, |app, _| {
                let mut store = app.html_previews.store.borrow_mut();
                assert_eq!(store.entries.len(), 1);
                let entry = store.entries.values_mut().next().unwrap();
                if revision == 0 {
                    entry.height = 360.0;
                    entry.source_mode = true;
                } else {
                    assert_eq!(
                        entry.height, 360.0,
                        "streaming must preserve measured height"
                    );
                    assert!(
                        entry.source_mode,
                        "streaming must retain the preview allocation"
                    );
                    assert!(entry.dirty, "only document content needs reloading");
                    assert!(entry.source.ends_with(&"<p>Next</p>".repeat(revision)));
                }
            });
        }
        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clipping_preserves_document_size_and_scroll_offset() {
        let full = Bounds::new(
            gpui_kit::point(px(30.0), px(-120.0)),
            gpui_kit::size(px(600.0), px(800.0)),
        );
        let mask = Bounds::new(
            gpui_kit::point(px(0.0), px(80.0)),
            gpui_kit::size(px(1000.0), px(400.0)),
        );
        let placement = Placement::new(full, mask).unwrap();
        assert_eq!(placement.full, full);
        assert_eq!(placement.clip.origin.y, px(80.0));
        assert_eq!(placement.clip.size.height, px(400.0));
        assert_eq!(
            placement.full.origin.y - placement.clip.origin.y,
            px(-200.0)
        );
        assert!(
            Placement::new(
                full,
                Bounds::new(gpui_kit::point(px(0.0), px(900.0)), mask.size)
            )
            .is_none()
        );
    }
    #[test]
    fn document_is_an_opaque_sandbox_and_cannot_escape_host_script() {
        let source = "</script><script>parent.document.body.textContent='escape __SOURCE__ __DOCUMENT__'</script>";
        let html = document(source, true, 1, "test-capability");
        assert!(html.contains("sandbox=\"allow-scripts\""));
        assert_eq!(
            html.matches("test-capability").count(),
            1,
            "only the trusted host receives the native action capability"
        );
        assert!(
            serde_json::from_str::<BrowserMessage>(
                r#"{"kind":"action","action":"download","generation":1}"#
            )
            .is_err()
        );
        assert!(!html.contains("allow-same-origin"));
        assert!(!html.contains(source));
        assert!(html.contains("connect-src 'none'"));
        assert!(html.contains("frame-src 'none'"));
        assert_eq!(
            html.matches("escape __SOURCE__ __DOCUMENT__").count(),
            2,
            "template-like user content must survive both source and preview serialization"
        );
    }
    fn preview_app(
        cx: &mut TestAppContext,
        text: String,
    ) -> (
        std::path::PathBuf,
        gpui_kit::Entity<DesktopApp>,
        &mut gpui_kit::VisualTestContext,
    ) {
        let root = std::env::temp_dir().join(format!(
            "kcastle-preview-{}",
            kcastle_agent::SessionId::new()
        ));
        let (startup, _) = crate::desktop_startup(root.clone()).unwrap();
        cx.update(crate::init_ui);
        let (view, cx) = cx.add_window_view(|window, cx| {
            let mut app = DesktopApp::new(startup, window, cx);
            let mut snapshot = crate::domain::SessionView::default();
            snapshot
                .conversation
                .messages
                .push_back(std::sync::Arc::new(crate::domain::Message {
                    key: MessageId(88000),
                    revision: 0,
                    role: crate::domain::Role::Assistant,
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
                }));
            app.core.session_view = std::sync::Arc::new(snapshot);
            app.core.follow_chat_tail = false;
            app.chat.get_mut().pending_anchor = Some(crate::layout::ScrollAnchor::Block {
                id: MessageId(88000),
                field: 1,
                source_offset: 0,
                local_offset: 0.0,
            });
            window.blur(cx);
            app
        });
        cx.simulate_resize(size(px(1180.0), px(620.0)));
        cx.run_until_parked();
        (root, view, cx)
    }

    #[gpui_kit::test]
    fn back_to_bottom_floats_over_native_preview_without_layout_gap(cx: &mut TestAppContext) {
        let (root, view, cx) = preview_app(
            cx,
            format!(
                "```html\n<div style='height:2000px'>Long preview</div>\n```\n\n{}",
                "Text after preview.\n\n".repeat(100),
            ),
        );
        view.update(cx, |app, cx| {
            app.core.unread_stream_updates = 649;
            for preview in app.html_previews.store.borrow_mut().entries.values_mut() {
                preview.height = MAX_HEIGHT;
            }
            app.chat.borrow().list.remeasure();
            app.chat
                .borrow()
                .list
                .set_follow_mode(gpui_kit::FollowMode::Normal);
            app.chat
                .borrow()
                .list
                .scroll_to(gpui_kit::ListOffset::default());
            cx.notify();
        });
        cx.run_until_parked();
        let button = cx.debug_bounds("back-to-bottom").unwrap();
        let viewport = view.read_with(cx, |app, _| app.chat.borrow().list.viewport_bounds());
        assert!(
            button.bottom() <= viewport.bottom() - px(8.0),
            "Back to bottom floats inside the transcript without a reserved footer"
        );
        assert!(
            (button.center().x - viewport.center().x).abs() < px(1.0),
            "Back to bottom must be centered in the chat column"
        );
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            let placements = store
                .entries
                .values()
                .filter_map(|p| p.placement)
                .collect::<Vec<_>>();
            assert!(
                placements.iter().any(|p| p.clip.contains(&button.center())),
                "exercise a native preview behind the floating button"
            );
            assert!(
                placements.iter().all(|p| !p.contains(button.center())),
                "Back to bottom is covered by a native browser clip"
            );
            let beside = gpui_kit::point(button.left() - px(4.0), button.center().y);
            assert!(
                placements.iter().any(|p| p.contains(beside)),
                "HTML beside the button must remain visible and interactive"
            );
        });
        cx.simulate_click(button.center(), gpui_kit::Modifiers::default());
        cx.run_until_parked();
        view.read_with(cx, |app, _| assert!(app.chat_at_bottom()));
        assert!(cx.debug_bounds("back-to-bottom").is_none());
        view.read_with(cx, |app, _| {
            assert_eq!(
                app.chat.borrow().list.viewport_bounds(),
                viewport,
                "hiding the button must not resize the transcript"
            );
            assert!(app.html_previews.store.borrow().button_occlusion.is_none());
        });
        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn floating_button_only_excludes_its_pill_without_reflow() {
        let full = Bounds::new(
            gpui_kit::point(px(0.0), px(0.0)),
            size(px(600.0), px(480.0)),
        );
        let before = Placement::new(full, full).unwrap();
        let hole = Bounds::new(
            gpui_kit::point(px(230.0), px(430.0)),
            size(px(140.0), px(32.0)),
        );
        let placed = Placement {
            occlusion: Some(hole),
            ..before
        };
        assert!(!placed.contains(hole.center()));
        assert!(
            placed.contains(hole.origin),
            "rounded corners retain HTML pixels"
        );
        assert!(placed.contains(gpui_kit::point(hole.left() - px(1.0), hole.center().y)));
        assert!(!placed.needs_layout(Some(before), false));
        assert!(!placed.needs_layout(Some(before), true));
        assert!(
            before.contains(hole.center()),
            "removing the button restores native input"
        );
    }

    #[gpui_kit::test]
    fn sidebar_follows_stream_while_inline_is_hidden(cx: &mut TestAppContext) {
        let (root, view, cx) = preview_app(cx, "```html\n<h1>Start</h1>".into());
        view.update(cx, |app, cx| {
            let mut store = app.html_previews.store.borrow_mut();
            let key = *store.entries.keys().next().unwrap();
            store.expanded = Some(key);
            app.core.surface = crate::domain::Surface::Trajectory;
            cx.notify();
        });
        cx.run_until_parked();
        let mut generation = view.update(cx, |app, _| {
            let mut store = app.html_previews.store.borrow_mut();
            let sidebar = &mut store.sidebar.as_mut().unwrap().1;
            sidebar.source_mode = true;
            sidebar.generation
        });
        for (suffix, changed) in [
            ("<h2>Finished</h2>\n```", true),
            (
                "\n\nAfter the document.\n\n```html\n<p>Second document</p>\n```",
                false,
            ),
        ] {
            view.update(cx, |app, cx| {
                let mut snapshot = (*app.core.session_view).clone();
                let mut message = (**snapshot.conversation.messages.front().unwrap()).clone();
                message.revision += 1;
                message.text.push_str(suffix);
                snapshot.conversation.messages.clear();
                snapshot
                    .conversation
                    .messages
                    .push_back(std::sync::Arc::new(message));
                app.core.session_view = std::sync::Arc::new(snapshot);
                cx.notify();
            });
            cx.run_until_parked();
            generation = view.read_with(cx, |app, _| {
                let store = app.html_previews.store.borrow();
                assert!(store.entries.values().all(|p| p.placement.is_none()));
                let sidebar = &store.sidebar.as_ref().unwrap().1;
                assert!(sidebar.placement.is_some());
                assert!(sidebar.source_mode, "streaming retains source-view mode");
                assert!(sidebar.source_task.is_none());
                assert_eq!(sidebar.source, "<h1>Start</h1><h2>Finished</h2>");
                assert_eq!(sidebar.generation != generation, changed);
                sidebar.generation
            });
        }
        view.update(cx, |app, cx| {
            app.core.surface = crate::domain::Surface::Chat;
            cx.notify();
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            let (key, sidebar) = store.sidebar.as_ref().unwrap();
            assert_eq!(
                sidebar.generation, generation,
                "remounting inline must not reload the sidebar"
            );
            assert_eq!(store.entries[key].source, sidebar.source);
        });
        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn sidebar_extracts_only_the_selected_fence_with_chat_limits() {
        let source = "\n  ~~~HTM\n  <p>Selected</p>\n  ~~~\n\n```html\n<p>Next</p>\n```";
        let (html, prefix) = sidebar_document(source).unwrap();
        assert_eq!(html, "<p>Selected</p>");
        assert_eq!(prefix, "\n  ~~~HTM\n  <p>Selected</p>\n  ~~~");
        assert!(sidebar_document("```rust\nlet x = 1;\n```").is_none());
        let oversized = format!(
            "```html\n{}\n```",
            "x".repeat(crate::platform::gpui::MAX_CODE_SOURCE_BYTES)
        );
        assert!(sidebar_document(&oversized).is_none());
        assert!(
            sidebar_document(&"x".repeat(crate::platform::gpui::MAX_INDEX_SOURCE_BYTES + 1))
                .is_none()
        );
    }
    #[gpui_kit::test]
    fn retired_sidebar_worker_cannot_replace_reopened_document(cx: &mut TestAppContext) {
        let (root, view, cx) = preview_app(cx, "```html\n<h1>Start</h1>".into());
        view.update(cx, |app, cx| {
            let mut store = app.html_previews.store.borrow_mut();
            store.expanded = store.entries.keys().next().copied();
            app.core.surface = crate::domain::Surface::Trajectory;
            cx.notify();
        });
        cx.run_until_parked();
        let (release, gate) = tokio::sync::oneshot::channel();
        view.update(cx, |app, cx| {
            let mut store = app.html_previews.store.borrow_mut();
            let preview = &mut store.sidebar.as_mut().unwrap().1;
            preview.source_revision = None;
            preview.source_gate = Some(gate);
            cx.notify();
        });
        cx.run_until_parked();
        let retired = view.update(cx, |app, cx| {
            // Hold the old task alive to exercise late publication even after replacement.
            let task = app
                .html_previews
                .store
                .borrow_mut()
                .sidebar
                .take()
                .unwrap()
                .1
                .source_task
                .take()
                .unwrap();
            let mut snapshot = (*app.core.session_view).clone();
            let mut message = (**snapshot.conversation.messages.front().unwrap()).clone();
            message.revision += 1;
            message.text.push_str("<p>Latest</p>\n```");
            snapshot.conversation.messages.clear();
            snapshot
                .conversation
                .messages
                .push_back(std::sync::Arc::new(message));
            app.core.session_view = std::sync::Arc::new(snapshot);
            cx.notify();
            task
        });
        cx.run_until_parked();
        let generation = view.read_with(cx, |app, _| {
            app.html_previews
                .store
                .borrow()
                .sidebar
                .as_ref()
                .unwrap()
                .1
                .generation
        });
        release.send(()).unwrap();
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let store = app.html_previews.store.borrow();
            let preview = &store.sidebar.as_ref().unwrap().1;
            assert_eq!(preview.source, "<h1>Start</h1><p>Latest</p>");
            assert_eq!(preview.generation, generation);
        });
        drop(retired);
        drop(view);
        cx.update(|window, _| window.remove_window());
        cx.run_until_parked();
        std::fs::remove_dir_all(root).unwrap();
    }
}
