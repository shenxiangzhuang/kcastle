//! Adapts our mixed Markdown renderer to gpui-base's window selection.
//! A message is one participant; text fragments and formula atoms contribute
//! runs in reading order, without introducing line breaks between flex items.
use std::{cell::RefCell, collections::HashMap, fmt, ops::Range, rc::Rc, sync::Arc};

use gpui_kit::base::{
    TextSelection, TextSelectionCoverage, TextSelectionEvent, TextSelectionHandle,
    TextSelectionRegistration, TextSelectionRun, TextSelectionSnapshot,
};
use gpui_kit::component::input::Copy;
use gpui_kit::component::{
    highlighter::{HighlightTheme, SyntaxHighlighter},
    input::Rope,
};
use gpui_kit::{
    AnyElement, App, Bounds, Element, ElementId, FocusHandle, GlobalElementId, Hsla,
    InspectorElementId, InteractiveElement, IntoElement, LayoutId, ParentElement, Pixels,
    SharedString, Styled, Subscription, TextLayout, Window, div, fill, point,
};

type CodeStyles = Vec<(Range<usize>, gpui_kit::HighlightStyle)>;
type CodeCache = HashMap<String, (String, Arc<HighlightTheme>, CodeStyles)>;

#[derive(Default)]
struct SelectionMemory {
    text: String,
    omitted: Vec<Range<usize>>,
    range: Option<Range<usize>>,
    snapshot: Option<TextSelectionSnapshot>,
}

impl SelectionMemory {
    fn invalidate_unless(&mut self, snapshot: Option<TextSelectionSnapshot>) {
        if same_logical_selection(snapshot, self.snapshot) {
            return;
        }
        self.text.clear();
        self.omitted.clear();
        self.range = None;
    }
}

#[derive(Clone)]
pub(crate) struct MessageSelection {
    handle: TextSelectionHandle,
    focus: FocusHandle,
    memory: Rc<RefCell<SelectionMemory>>,
    _subscriptions: Rc<[Subscription; 2]>,
    code: Rc<RefCell<CodeCache>>,
}

impl fmt::Debug for MessageSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessageSelection").finish_non_exhaustive()
    }
}

impl MessageSelection {
    pub(crate) fn new(window: &Window, cx: &mut App) -> Self {
        let handle = TextSelectionHandle::new("", cx);
        let focus = cx.focus_handle();
        let focus_for_selection = focus.clone();
        handle.focus_with(move |window, cx| focus_for_selection.focus(window, cx), cx);
        let memory = Rc::new(RefCell::new(SelectionMemory::default()));
        let copy = memory.clone();
        handle.copy_with(
            move |_| {
                let memory = copy.borrow();
                let Some(range) = memory.range.clone() else {
                    return String::new();
                };
                let mut copied = String::new();
                let mut start = range.start;
                for omitted in &memory.omitted {
                    if omitted.end <= start || omitted.start >= range.end {
                        continue;
                    }
                    copied.push_str(&memory.text[start..omitted.start.max(start)]);
                    start = omitted.end.min(range.end);
                }
                copied.push_str(&memory.text[start..range.end]);
                copied
            },
            cx,
        );
        let clear = memory.clone();
        handle.clear_with(
            move |_| *clear.borrow_mut() = SelectionMemory::default(),
            cx,
        );
        let refresh = handle.refresh_window_on_change(window, cx);
        let invalidate = memory.clone();
        let invalidation = handle.subscribe(
            move |event, _| {
                if let TextSelectionEvent::SelectionChanged(snapshot) = event {
                    invalidate.borrow_mut().invalidate_unless(*snapshot);
                }
            },
            cx,
        );
        Self {
            handle,
            focus,
            memory,
            _subscriptions: Rc::new([refresh, invalidation]),
            code: Rc::default(),
        }
    }

    pub(crate) fn frame(&self, order: u64) -> SelectionFrame {
        SelectionFrame {
            selection: self.clone(),
            order,
            content: Rc::default(),
        }
    }
}

#[derive(Default)]
struct FrameContent {
    text: String,
    omitted: Vec<Range<usize>>,
    runs: Vec<Run>,
    separator: &'static str,
    selected: Option<Range<usize>>,
}

struct Run {
    range: Range<usize>,
    text: SharedString,
    layout: Option<TextLayout>,
    bounds: Option<Bounds<Pixels>>,
    atom: bool,
}

#[derive(Clone)]
pub(crate) struct SelectionFrame {
    selection: MessageSelection,
    order: u64,
    content: Rc<RefCell<FrameContent>>,
}

impl SelectionFrame {
    pub(crate) fn code_styles(
        &self,
        key: &str,
        language: &str,
        source: &str,
        theme: &Arc<HighlightTheme>,
    ) -> CodeStyles {
        let mut cache = self.selection.code.borrow_mut();
        if let Some((text, previous_theme, styles)) = cache.get(key)
            && text == source
            && Arc::ptr_eq(previous_theme, theme)
        {
            return styles.clone();
        }
        // Settled code used TextView's unbounded parse; keep that contract so a
        // timed-out empty or stale tree can never enter this permanent cache.
        let mut highlighter = SyntaxHighlighter::new(language);
        let completed = highlighter.update(None, &Rope::from(source), None);
        debug_assert!(completed, "an unbounded syntax parse always completes");
        let styles = highlighter.styles(&(0..source.len()), theme.as_ref());
        cache.insert(
            key.to_owned(),
            (source.to_owned(), theme.clone(), styles.clone()),
        );
        styles
    }
    /// Separators belong to document structure, never to visual line wrapping.
    pub(crate) fn separate(&self, separator: &'static str) {
        let mut content = self.content.borrow_mut();
        if separator.len() > content.separator.len() {
            content.separator = separator;
        }
    }

    pub(crate) fn text(&self, text: SharedString) -> SelectionFragment {
        self.fragment(text, false)
    }

    fn fragment(&self, text: SharedString, atom: bool) -> SelectionFragment {
        let mut content = self.content.borrow_mut();
        let separator = content.separator;
        if !content.text.is_empty() {
            content.text.push_str(separator);
        }
        content.separator = "";
        let start = content.text.len();
        content.text.push_str(&text);
        let end = content.text.len();
        let index = content.runs.len();
        content.runs.push(Run {
            range: start..end,
            text,
            layout: None,
            bounds: None,
            atom,
        });
        SelectionFragment {
            content: self.content.clone(),
            index,
        }
    }

    pub(crate) fn atom(
        &self,
        text: impl Into<SharedString>,
        child: impl IntoElement,
    ) -> AnyElement {
        SelectionAtom {
            fragment: self.fragment(text.into(), true),
            child: child.into_any_element(),
        }
        .into_any_element()
    }

    pub(crate) fn wrap(self, child: impl IntoElement) -> AnyElement {
        let content = div()
            .id("selectable-message")
            .track_focus(&self.selection.focus)
            .key_context("Root")
            .cursor_text()
            .on_action(|_: &Copy, window, cx| {
                // Root's default handler trims whitespace, which corrupts selected code.
                let text = TextSelection::selected_text(window, cx);
                if text.is_empty() {
                    cx.propagate();
                } else {
                    cx.write_to_clipboard(gpui_kit::ClipboardItem::new_string(text));
                }
            })
            .on_key_down(|event, window, cx| {
                if event.keystroke.key == "escape" {
                    TextSelection::clear(window, cx);
                }
            })
            .w_full()
            .child(child);
        SelectionGroup {
            frame: self,
            child: content.into_any_element(),
        }
        .into_any_element()
    }

    fn project(&self, cx: &mut App) {
        let snapshot = self.selection.handle.snapshot(cx);
        let mut content = self.content.borrow_mut();
        let mut memory = self.selection.memory.borrow_mut();
        let runs = content
            .runs
            .iter()
            .filter_map(|run| {
                Some(TextSelectionRun::new(
                    run.text.clone(),
                    run.layout.clone()?,
                    run.bounds?,
                ))
            })
            .collect::<Vec<_>>();
        let projection = self.selection.handle.update_runs(&runs, cx);
        let stable = same_logical_selection(snapshot, memory.snapshot);
        let range = if snapshot.is_none() {
            None
        } else if stable {
            // Keep logical bytes across reflow/scroll and append-only streaming.
            // If earlier rendered text changes, drop the range rather than copy stale text.
            memory
                .range
                .clone()
                .filter(|range| content.text.get(..range.end) == memory.text.get(..range.end))
        } else {
            let mut projected = projection.ranges().iter();
            let mut range: Option<Range<usize>> = None;
            for run in &content.runs {
                let selected = if run.atom {
                    snapshot
                        .and_then(|snapshot| snapshot.window_points())
                        .and_then(|points| {
                            let bounds = run.bounds?;
                            atom_in_selection(bounds, points.anchor(), points.cursor())
                                .then_some(0..run.text.len())
                        })
                } else if run.layout.is_some() {
                    projected.next().cloned().flatten()
                } else {
                    None
                };
                if let Some(selected) = selected.filter(|range| !range.is_empty()) {
                    let selected =
                        (run.range.start + selected.start)..(run.range.start + selected.end);
                    if let Some(range) = &mut range {
                        range.end = selected.end;
                    } else {
                        range = Some(selected);
                    }
                }
            }
            match snapshot.map(|snapshot| snapshot.coverage()) {
                Some(TextSelectionCoverage::Full) => Some(0..content.text.len()),
                Some(TextSelectionCoverage::FromStart) => range.map(|range| 0..range.end),
                Some(TextSelectionCoverage::ToEnd) => {
                    range.map(|range| range.start..content.text.len())
                }
                _ => range,
            }
        };
        content.selected = range.clone();
        memory.range = range;
        if memory.range.is_some() {
            memory.text.clone_from(&content.text);
            memory.omitted.clone_from(&content.omitted);
        } else {
            memory.text.clear();
            memory.omitted.clear();
        }
        memory.snapshot = snapshot;
    }

    #[cfg(test)]
    pub(crate) fn text_position(&self, needle: &str, end: bool) -> gpui_kit::Point<Pixels> {
        let content = self.content.borrow();
        let start = content
            .text
            .find(needle)
            .expect("selection test text exists");
        let offset = start + if end { needle.len() } else { 0 };
        let run = content
            .runs
            .iter()
            .find(|run| {
                run.range.start <= offset && offset <= run.range.end && run.layout.is_some()
            })
            .expect("test text has a layout");
        let layout = run.layout.as_ref().unwrap();
        let position = layout.position_for_index(offset - run.range.start).unwrap();
        point(position.x, position.y + layout.line_height() / 2.0)
    }
}

fn same_logical_selection(
    current: Option<TextSelectionSnapshot>,
    projected: Option<TextSelectionSnapshot>,
) -> bool {
    current.zip(projected).is_some_and(|(current, projected)| {
        current.anchor() == projected.anchor()
            && current.cursor() == projected.cursor()
            && current.coverage() == projected.coverage()
    })
}

fn atom_in_selection(
    bounds: Bounds<Pixels>,
    a: gpui_kit::Point<Pixels>,
    b: gpui_kit::Point<Pixels>,
) -> bool {
    let (a, b) = if (a.y, a.x) <= (b.y, b.x) {
        (a, b)
    } else {
        (b, a)
    };
    let center = bounds.center();
    bounds.bottom() > a.y
        && bounds.top() <= b.y
        && (a.y < bounds.top() || center.x >= a.x)
        && (b.y >= bounds.bottom() || center.x <= b.x)
}

#[derive(Clone)]
pub(crate) struct SelectionFragment {
    content: Rc<RefCell<FrameContent>>,
    index: usize,
}

impl SelectionFragment {
    pub(crate) fn omit(&self, ranges: &[Range<usize>]) {
        let mut content = self.content.borrow_mut();
        let start = content.runs[self.index].range.start;
        content.omitted.extend(
            ranges
                .iter()
                .map(|range| (start + range.start)..(start + range.end)),
        );
    }
    pub(crate) fn layout(&self, layout: TextLayout, bounds: Bounds<Pixels>) {
        let mut content = self.content.borrow_mut();
        content.runs[self.index].layout = Some(layout);
        content.runs[self.index].bounds = Some(bounds);
    }

    pub(crate) fn paint(&self, color: Hsla, window: &mut Window) {
        let content = self.content.borrow();
        let run = &content.runs[self.index];
        let Some(range) = content.selected.as_ref() else {
            return;
        };
        let start = range.start.max(run.range.start);
        let end = range.end.min(run.range.end);
        if start >= end {
            return;
        }
        if let Some(layout) = &run.layout {
            paint_text_selection(
                layout,
                (start - run.range.start)..(end - run.range.start),
                color,
                window,
            );
        } else if let Some(bounds) = run.bounds {
            window.paint_quad(fill(bounds, color));
        }
    }
}

fn paint_text_selection(
    layout: &TextLayout,
    range: Range<usize>,
    color: Hsla,
    window: &mut Window,
) {
    let (Some(start), Some(end)) = (
        layout.position_for_index(range.start),
        layout.position_for_index(range.end),
    ) else {
        return;
    };
    let bounds = layout.bounds();
    let height = layout.line_height();
    if start.y == end.y {
        window.paint_quad(fill(
            Bounds::from_corners(start, point(end.x, end.y + height)),
            color,
        ));
    } else {
        window.paint_quad(fill(
            Bounds::from_corners(start, point(bounds.right(), start.y + height)),
            color,
        ));
        if end.y > start.y + height {
            window.paint_quad(fill(
                Bounds::from_corners(
                    point(bounds.left(), start.y + height),
                    point(bounds.right(), end.y),
                ),
                color,
            ));
        }
        window.paint_quad(fill(
            Bounds::from_corners(point(bounds.left(), end.y), point(end.x, end.y + height)),
            color,
        ));
    }
}

struct SelectionGroup {
    frame: SelectionFrame,
    child: AnyElement,
}

impl IntoElement for SelectionGroup {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}
impl Element for SelectionGroup {
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
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let hitbox = window.insert_hitbox(bounds, gpui_kit::HitboxBehavior::Normal);
        self.child.prepaint(window, cx);
        let text_bounds = self
            .frame
            .content
            .borrow()
            .runs
            .iter()
            .filter_map(|run| run.bounds)
            .collect();
        self.frame.selection.handle.register(
            TextSelectionRegistration::new(hitbox, bounds)
                .with_document_order(self.frame.order)
                .with_text_bounds(text_bounds),
            window,
            cx,
        );
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
        self.frame.project(cx);
        self.child.paint(window, cx);
    }
}

struct SelectionAtom {
    fragment: SelectionFragment,
    child: AnyElement,
}
impl IntoElement for SelectionAtom {
    type Element = Self;
    fn into_element(self) -> Self {
        self
    }
}
impl Element for SelectionAtom {
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
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        self.child.prepaint(window, cx);
        self.fragment.content.borrow_mut().runs[self.fragment.index].bounds = Some(bounds);
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
        self.fragment
            .paint(gpui_kit::hsla(0.58, 0.8, 0.6, 0.3), window);
        self.child.paint(window, cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui_kit::{
        Context, Entity, Modifiers, MouseButton, Render, ScrollHandle, StatefulInteractiveElement,
        TestAppContext, VisualTestContext, px, size,
    };

    struct Transcript {
        selections: [MessageSelection; 2],
        frames: Vec<SelectionFrame>,
        scroll: ScrollHandle,
    }

    impl Render for Transcript {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.frames.clear();
            let mut body = div()
                .id("transcript")
                .size_full()
                .overflow_y_scroll()
                .track_scroll(&self.scroll);
            for (index, text) in ["first selection 中文", "second tail"]
                .into_iter()
                .enumerate()
            {
                let frame = self.selections[index].frame(index as u64);
                let content = crate::dsh_markdown::plain_text(text.into(), Some(&frame));
                body = body.child(
                    div()
                        .id(("message", index))
                        .h(px(120.0))
                        .child(frame.clone().wrap(content)),
                );
                self.frames.push(frame);
            }
            div()
                .size_full()
                .child(gpui_kit::base::TextSelectionLayer)
                .child(body)
        }
    }

    fn setup(cx: &mut TestAppContext) -> (Entity<Transcript>, &mut VisualTestContext) {
        cx.update(gpui_kit::component::init);
        let (view, cx) = cx.add_window_view(|window, cx| Transcript {
            selections: [
                MessageSelection::new(window, cx),
                MessageSelection::new(window, cx),
            ],
            frames: Vec::new(),
            scroll: ScrollHandle::new(),
        });
        cx.simulate_resize(size(px(400.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();
        (view, cx)
    }

    #[gpui_kit::test]
    fn cross_message_selection_survives_scrolling_and_clears_with_session(cx: &mut TestAppContext) {
        let (view, cx) = setup(cx);
        let (start, end) = view.read_with(cx, |view, _| {
            (
                view.frames[0].text_position("selection", false),
                view.frames[1].text_position("tail", true),
            )
        });
        cx.simulate_mouse_down(start, MouseButton::Left, Modifiers::default());
        cx.simulate_mouse_move(end, Some(MouseButton::Left), Modifiers::default());
        cx.simulate_mouse_up(end, MouseButton::Left, Modifiers::default());
        cx.run_until_parked();
        assert_eq!(
            cx.update(TextSelection::selected_text),
            "selection 中文\nsecond tail"
        );
        view.update(cx, |view, cx| {
            view.scroll.set_offset(point(px(0.0), px(-50.0)));
            cx.notify();
        });
        cx.run_until_parked();
        assert_eq!(
            cx.update(TextSelection::selected_text),
            "selection 中文\nsecond tail"
        );
        cx.update(|window, cx| {
            view.update(cx, |view, cx| {
                view.selections = [
                    MessageSelection::new(window, cx),
                    MessageSelection::new(window, cx),
                ];
                cx.notify();
            })
        });
        cx.run_until_parked();
        assert!(cx.update(TextSelection::selected_text).is_empty());
    }

    #[gpui_kit::test]
    fn changed_selection_invalidates_copy_projection(cx: &mut TestAppContext) {
        let (view, cx) = setup(cx);
        let (start, end) = view.read_with(cx, |view, _| {
            (
                view.frames[0].text_position("selection", false),
                view.frames[0].text_position("中文", true),
            )
        });
        cx.simulate_mouse_down(start, MouseButton::Left, Modifiers::default());
        cx.simulate_mouse_move(end, Some(MouseButton::Left), Modifiers::default());
        cx.simulate_mouse_up(end, MouseButton::Left, Modifiers::default());
        cx.run_until_parked();
        assert_eq!(cx.update(TextSelection::selected_text), "selection 中文");

        view.read_with(cx, |view, _| {
            view.selections[0]
                .memory
                .borrow_mut()
                .invalidate_unless(None);
        });
        assert!(cx.update(TextSelection::selected_text).is_empty());
    }

    #[gpui_kit::test]
    fn inactive_messages_do_not_cache_copy_buffers(cx: &mut TestAppContext) {
        let (view, cx) = setup(cx);
        view.read_with(cx, |view, _| {
            for selection in &view.selections {
                let memory = selection.memory.borrow();
                assert!(memory.text.is_empty());
                assert!(memory.omitted.is_empty());
                assert!(memory.range.is_none());
            }
        });
    }
}
