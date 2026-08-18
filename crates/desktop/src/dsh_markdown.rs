use std::{
    collections::HashMap,
    ops::Range,
    sync::{Mutex, OnceLock},
};

use gpui::{
    AnyElement, App, Bounds, Context, Element, ElementId, FontStyle, FontWeight, GlobalElementId,
    HighlightStyle, Hsla, InspectorElementId, IntoElement, LayoutId, ParentElement, Pixels,
    SharedString, StrikethroughStyle, StyleRefinement, Styled, StyledText, Window, div, fill,
    point, prelude::FluentBuilder, px, rems, size, svg,
};
use gpui_component::ActiveTheme;
use gpui_component::clipboard::Clipboard;
use gpui_component::scroll::ScrollableElement;
use gpui_component::text::{TextView, TextViewStyle};
use markdown::mdast::Node;

use crate::app::DesktopApp;
use crate::assets::register_generated_asset;
use crate::layout::{ColumnSpec, allocate_columns, list_marker_width};
use crate::streaming_markdown::{MarkdownBlock, StreamingMarkdownState};
use crate::ui_theme::{UiPalette, metrics, palette};

pub(crate) fn render_markdown(
    message_key: u64,
    state: &StreamingMarkdownState,
    streaming: bool,
    fallback: &SharedString,
    available_width: f32,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> AnyElement {
    let colors = palette(cx);
    let blocks = state
        .frozen()
        .iter()
        .map(|block| (block, None))
        .chain(
            state
                .tail_blocks()
                .iter()
                .map(|block| (block, Some(state.revision()))),
        )
        .collect::<Vec<_>>();

    if blocks.is_empty() {
        return div()
            .w_full()
            .text_color(colors.markdown_text)
            .text_size(px(16.0))
            .line_height(px(metrics::MESSAGE_LINE_HEIGHT))
            .child(fallback.clone())
            .into_any_element();
    }

    let mut root = div()
        .flex()
        .flex_col()
        .w_full()
        .min_w(px(0.0))
        .text_color(colors.markdown_text)
        .text_size(px(16.0))
        .line_height(px(metrics::MESSAGE_LINE_HEIGHT));
    for (index, (block, revision)) in blocks.iter().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|index| blocks.get(index))
            .map(|(block, _)| &block.node);
        let gap = root_block_gap(previous, &block.node, index == 0);
        let context = BlockContext {
            message_key,
            generation: state.generation(),
            revision: *revision,
            block,
            streaming,
            colors,
            available_width,
        };
        root = root.child(
            div()
                .w_full()
                .min_w(px(0.0))
                .when(gap > 0.0, |element| element.mt(px(gap)))
                .child(render_node(&block.node, &context, "root", window, cx)),
        );
    }
    root.into_any_element()
}

#[derive(Clone, Copy)]
struct BlockContext<'a> {
    message_key: u64,
    generation: u64,
    revision: Option<u64>,
    block: &'a MarkdownBlock,
    streaming: bool,
    colors: UiPalette,
    available_width: f32,
}

fn render_node(
    node: &Node,
    context: &BlockContext<'_>,
    path: &str,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> AnyElement {
    match node {
        Node::Paragraph(paragraph) => inline_block(
            &paragraph.children,
            16.0,
            metrics::MESSAGE_LINE_HEIGHT,
            FontWeight::NORMAL,
            context,
        ),
        Node::Heading(heading) => {
            let (size, line_height, weight) = heading_style(heading.depth);
            inline_block(&heading.children, size, line_height, weight, context)
        }
        Node::List(list) => render_list(list, context, path, window, cx),
        Node::Blockquote(quote) => {
            let mut body = div()
                .flex()
                .flex_col()
                .w_full()
                .min_w(px(0.0))
                .border_l_2()
                .border_color(context.colors.markdown_quote)
                .pl(px(14.0));
            for (index, child) in quote.children.iter().enumerate() {
                body = body.child(
                    div()
                        .w_full()
                        .when(index > 0, |element| {
                            element.mt(px(metrics::MARKDOWN_BLOCK_GAP))
                        })
                        .child(render_node(
                            child,
                            context,
                            &format!("{path}-quote-{index}"),
                            window,
                            cx,
                        )),
                );
            }
            body.into_any_element()
        }
        Node::Code(code) => render_code_block(
            code.lang.as_deref().unwrap_or_default(),
            &code.value,
            context,
            path,
            window,
            cx,
        ),
        Node::Math(math) => render_math(&math.value, true, context)
            .unwrap_or_else(|| render_code_block("math", &math.value, context, path, window, cx)),
        Node::Table(table) => render_table(table, context, path, window, cx),
        Node::ThematicBreak(_) => div()
            .w_full()
            .h(px(1.0))
            .bg(context.colors.border)
            .into_any_element(),
        Node::Html(html) => div()
            .w_full()
            .text_color(context.colors.muted_text)
            .child(html.value.clone())
            .into_any_element(),
        Node::Root(root) => {
            let mut body = div().flex().flex_col().w_full().min_w(px(0.0));
            for (index, child) in root.children.iter().enumerate() {
                body = body.child(
                    div()
                        .w_full()
                        .when(index > 0, |element| {
                            element.mt(px(metrics::MARKDOWN_BLOCK_GAP))
                        })
                        .child(render_node(
                            child,
                            context,
                            &format!("{path}-root-{index}"),
                            window,
                            cx,
                        )),
                );
            }
            body.into_any_element()
        }
        _ => inline_block(
            std::slice::from_ref(node),
            16.0,
            metrics::MESSAGE_LINE_HEIGHT,
            FontWeight::NORMAL,
            context,
        ),
    }
}

fn render_list(
    list: &markdown::mdast::List,
    context: &BlockContext<'_>,
    path: &str,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> AnyElement {
    let start = list.start.unwrap_or(1);
    let mut body = div().flex().flex_col().w_full().min_w(px(0.0));
    for (index, item) in list.children.iter().enumerate() {
        let Node::ListItem(item) = item else {
            continue;
        };
        let marker = if let Some(checked) = item.checked {
            if checked {
                "☑".to_owned()
            } else {
                "☐".to_owned()
            }
        } else if list.ordered {
            format!("{}.", start + index as u32)
        } else {
            "•".to_owned()
        };
        let marker_width = list_marker_width(list.ordered, start, list.children.len());
        let mut item_body = div().flex().flex_col().flex_1().min_w(px(0.0));
        for (child_index, child) in item.children.iter().enumerate() {
            let child_gap = if matches!(child, Node::List(_)) {
                4.0
            } else {
                8.0
            };
            item_body = item_body.child(
                div()
                    .w_full()
                    .when(child_index > 0, |element| element.mt(px(child_gap)))
                    .child(render_node(
                        child,
                        context,
                        &format!("{path}-item-{index}-{child_index}"),
                        window,
                        cx,
                    )),
            );
        }
        body = body.child(
            div()
                .flex()
                .items_start()
                .w_full()
                .min_w(px(0.0))
                .when(index > 0, |element| element.mt(px(6.0)))
                .child(
                    div()
                        .flex_none()
                        .w(px(marker_width))
                        .pr(px(8.0))
                        .text_right()
                        .whitespace_nowrap()
                        .line_height(px(metrics::MESSAGE_LINE_HEIGHT))
                        .text_color(context.colors.muted_text)
                        .child(marker),
                )
                .child(item_body),
        );
    }
    body.into_any_element()
}

fn render_table(
    table: &markdown::mdast::Table,
    context: &BlockContext<'_>,
    _path: &str,
    _window: &mut Window,
    _cx: &mut Context<DesktopApp>,
) -> AnyElement {
    let columns = table
        .children
        .iter()
        .filter_map(|row| match row {
            Node::TableRow(row) => Some(row.children.len()),
            _ => None,
        })
        .max()
        .unwrap_or(1);
    let preferred = (context.available_width / columns as f32).clamp(120.0, 240.0);
    let specs = vec![
        ColumnSpec {
            min: 100.0,
            preferred,
            max: Some(320.0),
            weight: 1.0,
        };
        columns
    ];
    let table_layout = allocate_columns(context.available_width, &specs);
    let mut grid = div()
        .flex()
        .flex_col()
        .flex_none()
        .w(px(table_layout.content_width));
    for (row_index, row) in table.children.iter().enumerate() {
        let Node::TableRow(row) = row else {
            continue;
        };
        let mut row_view = div()
            .flex()
            .w_full()
            .border_b_1()
            .border_color(if row_index == 0 {
                context.colors.markdown_quote
            } else {
                context.colors.border
            });
        for (cell_index, cell) in row.children.iter().enumerate() {
            let Node::TableCell(cell) = cell else {
                continue;
            };
            let width = table_layout
                .tracks
                .get(cell_index)
                .copied()
                .unwrap_or(100.0);
            row_view = row_view.child(
                div()
                    .flex_none()
                    .w(px(width))
                    .min_w(px(0.0))
                    .px_4()
                    .py(px(10.0))
                    .when(cell_index == 0, |element| element.pl(px(0.0)))
                    .when(cell_index + 1 == row.children.len(), |element| {
                        element.pr(px(0.0))
                    })
                    .text_size(px(15.0))
                    .line_height(px(25.0))
                    .font_weight(if row_index == 0 {
                        FontWeight::MEDIUM
                    } else {
                        FontWeight::NORMAL
                    })
                    .child(inline_block(
                        &cell.children,
                        15.0,
                        25.0,
                        if row_index == 0 {
                            FontWeight::MEDIUM
                        } else {
                            FontWeight::NORMAL
                        },
                        context,
                    )),
            );
        }
        grid = grid.child(row_view);
    }
    div()
        .w_full()
        .overflow_x_scrollbar()
        .child(grid)
        .into_any_element()
}

fn render_code_block(
    language: &str,
    code: &str,
    context: &BlockContext<'_>,
    path: &str,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) -> AnyElement {
    let clipboard_id = SharedString::from(format!(
        "dsh-md-copy-{}-{}-{path}",
        context.message_key, context.block.key
    ));
    let mut body = div()
        .flex()
        .flex_col()
        .w_full()
        .min_w(px(0.0))
        .overflow_hidden()
        .rounded(px(12.0))
        .bg(context.colors.markdown_code_block)
        .child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .w_full()
                .px(px(14.0))
                .py(px(9.0))
                .bg(context.colors.markdown_code_banner)
                .font_family("SF Mono")
                .text_size(px(12.0))
                .line_height(px(18.0))
                .child(if language.is_empty() {
                    "text".to_owned()
                } else {
                    language.to_owned()
                })
                .child(Clipboard::new(clipboard_id).value(code.to_owned())),
        );

    if context.streaming {
        body = body.child(
            div().w_full().overflow_x_scrollbar().child(
                div()
                    .min_w_full()
                    .p_4()
                    .whitespace_nowrap()
                    .font_family("SF Mono")
                    .text_size(px(13.0))
                    .line_height(px(22.0))
                    .child(SharedString::from(code.to_owned())),
            ),
        );
    } else {
        let fence = if code.contains("```") { "````" } else { "```" };
        let markdown = format!("{fence}{language}\n{code}\n{fence}");
        let mut style = TextViewStyle::default().paragraph_gap(rems(0.0));
        style.highlight_theme = cx.theme().highlight_theme.clone();
        style.is_dark = cx.theme().is_dark();
        style.code_block = StyleRefinement::default()
            .w_full()
            .p_4()
            .rounded(px(0.0))
            .bg(context.colors.markdown_code_block)
            .font_family("SF Mono")
            .text_size(px(13.0))
            .line_height(px(22.0));
        body = body.child(
            TextView::markdown(render_id(context, path), markdown, window, cx)
                .style(style)
                .selectable(true),
        );
    }
    body.into_any_element()
}

fn render_id(context: &BlockContext<'_>, path: &str) -> SharedString {
    SharedString::from(format!(
        "dsh-md-{}-{}-{}-{path}-{}",
        context.message_key,
        context.generation,
        context.block.key,
        context
            .revision
            .map(|revision| revision.to_string())
            .unwrap_or_else(|| "frozen".into())
    ))
}

fn root_block_gap(previous: Option<&Node>, node: &Node, first: bool) -> f32 {
    if first {
        return 0.0;
    }
    if matches!(previous, Some(Node::ThematicBreak(_))) || matches!(node, Node::ThematicBreak(_)) {
        return metrics::MARKDOWN_SECTION_GAP;
    }
    if matches!(node, Node::Heading(heading) if heading.depth <= 3) {
        return metrics::MARKDOWN_SECTION_GAP;
    }
    if matches!(node, Node::List(_))
        && matches!(previous, Some(Node::Heading(heading)) if heading.depth >= 4)
    {
        return 8.0;
    }
    metrics::MARKDOWN_BLOCK_GAP
}

fn heading_style(depth: u8) -> (f32, f32, FontWeight) {
    match depth {
        // CoreText renders the system font's 700 face more heavily than the
        // browser stack used by DeepSeek Harness. Use 600 for the equivalent
        // native visual weight while retaining the reference type scale.
        1 => (24.0, 34.0, FontWeight::SEMIBOLD),
        2 => (22.0, 32.0, FontWeight::SEMIBOLD),
        3 => (20.0, 30.0, FontWeight::SEMIBOLD),
        _ => (16.0, 28.0, FontWeight::SEMIBOLD),
    }
}

fn inline_block(
    nodes: &[Node],
    size: f32,
    line_height: f32,
    weight: FontWeight,
    context: &BlockContext<'_>,
) -> AnyElement {
    if !nodes.iter().any(|node| matches!(node, Node::InlineMath(_))) {
        return div()
            .w_full()
            .min_w(px(0.0))
            .whitespace_normal()
            .text_size(px(size))
            .line_height(px(line_height))
            .font_weight(weight)
            .child(inline_text(nodes, context.colors))
            .into_any_element();
    }

    let mut body = div()
        .flex()
        .flex_wrap()
        .items_baseline()
        .w_full()
        .min_w(px(0.0))
        .whitespace_normal()
        .text_size(px(size))
        .line_height(px(line_height))
        .font_weight(weight);
    let mut text_start = 0;
    for (index, node) in nodes.iter().enumerate() {
        let Node::InlineMath(math) = node else {
            continue;
        };
        if text_start < index {
            body = body.child(
                div()
                    .min_w(px(0.0))
                    .flex_shrink()
                    .child(inline_text(&nodes[text_start..index], context.colors)),
            );
        }
        body = body.child(render_math(&math.value, false, context).unwrap_or_else(|| {
            div()
                .child(inline_text(std::slice::from_ref(node), context.colors))
                .into_any_element()
        }));
        text_start = index + 1;
    }
    if text_start < nodes.len() {
        body = body.child(
            div()
                .min_w(px(0.0))
                .flex_shrink()
                .child(inline_text(&nodes[text_start..], context.colors)),
        );
    }
    body.into_any_element()
}

#[derive(Clone, Debug)]
struct RenderedMath {
    asset: SharedString,
    width: f32,
    height: f32,
    baseline_offset: f32,
}

type MathCache = HashMap<(String, bool), Result<RenderedMath, String>>;
static MATH_CACHE: OnceLock<Mutex<MathCache>> = OnceLock::new();

fn render_math(source: &str, display: bool, context: &BlockContext<'_>) -> Option<AnyElement> {
    let rendered = cached_math(source, display).ok()?;
    let width = rendered.width;
    let formula = svg()
        .path(rendered.asset)
        .flex_none()
        .w(px(width))
        .h(px(rendered.height))
        .text_color(context.colors.markdown_text)
        .when(!display, |formula| {
            formula.relative().top(px(rendered.baseline_offset))
        });
    if display {
        Some(
            div()
                .w_full()
                .overflow_x_scrollbar()
                .child(
                    div()
                        .flex()
                        .justify_center()
                        .min_w_full()
                        .w(px(width))
                        .child(formula),
                )
                .into_any_element(),
        )
    } else {
        Some(formula.into_any_element())
    }
}

fn cached_math(source: &str, display: bool) -> Result<RenderedMath, String> {
    let key = (source.to_owned(), display);
    let cache = MATH_CACHE.get_or_init(Default::default);
    if let Some(rendered) = cache
        .lock()
        .expect("math cache poisoned")
        .get(&key)
        .cloned()
    {
        return rendered;
    }

    let rendered = build_math(source, display);
    // ponytail: process-wide cache; add eviction only if long sessions show material growth.
    cache
        .lock()
        .expect("math cache poisoned")
        .insert(key, rendered.clone());
    rendered
}

fn build_math(source: &str, display: bool) -> Result<RenderedMath, String> {
    use ratex_layout::{LayoutOptions, layout, to_display_list};
    use ratex_svg::{SvgOptions, render_to_svg};
    use ratex_types::math_style::MathStyle;

    let ast = ratex_parser::parse(source).map_err(|error| error.to_string())?;
    let style = if display {
        MathStyle::Display
    } else {
        MathStyle::Text
    };
    let layout = layout(&ast, &LayoutOptions::default().with_style(style));
    let display_list = to_display_list(&layout);
    let font_size = if display { 20.0 } else { 16.0 };
    let padding = if display { 2.0 } else { 1.0 };
    let width = (display_list.width * font_size + padding * 2.0) as f32;
    let height = ((display_list.height + display_list.depth) * font_size + padding * 2.0) as f32;
    if width <= 0.0 || height <= 0.0 {
        return Err("formula has no visible bounds".to_owned());
    }

    let svg = render_to_svg(
        &display_list,
        &SvgOptions {
            font_size,
            padding,
            stroke_width: 1.0,
            embed_glyphs: true,
            font_dir: String::new(),
        },
    );
    let asset = register_generated_asset(svg.into_bytes());
    Ok(RenderedMath {
        asset,
        width,
        height,
        baseline_offset: (display_list.depth * font_size + padding) as f32,
    })
}

fn inline_text(nodes: &[Node], colors: UiPalette) -> InlineText {
    let mut output = InlineOutput::default();
    append_inlines(nodes, InlineStyle::default(), colors, &mut output);
    InlineText::new(output)
}

#[derive(Clone, Copy, Default)]
struct InlineStyle {
    strong: bool,
    emphasis: bool,
    deleted: bool,
    code: bool,
    link: bool,
}

#[derive(Default)]
struct InlineOutput {
    text: String,
    highlights: Vec<(Range<usize>, HighlightStyle)>,
    backgrounds: Vec<(Range<usize>, Hsla)>,
}

// GPUI highlight backgrounds occupy the entire line height, which makes code on
// adjacent lines merge visually. Paint code backgrounds separately so they can
// be inset without changing the text shaping or wrapping behavior.
struct InlineText {
    text: SharedString,
    styled: StyledText,
    backgrounds: Vec<(Range<usize>, Hsla)>,
}

impl InlineText {
    const CODE_BACKGROUND_INSET: Pixels = px(2.0);
    const CODE_BACKGROUND_RADIUS: Pixels = px(4.0);

    fn new(output: InlineOutput) -> Self {
        let text = SharedString::from(output.text);
        Self {
            styled: StyledText::new(text.clone()).with_highlights(output.highlights),
            text,
            backgrounds: output.backgrounds,
        }
    }

    fn paint_backgrounds(&self, window: &mut Window) {
        let text_layout = self.styled.layout();
        let line_height = text_layout.line_height();
        let background_height = line_height - Self::CODE_BACKGROUND_INSET * 2.0;
        if background_height <= px(0.0) {
            return;
        }

        for (range, color) in &self.backgrounds {
            let mut segment_start = range.start;
            while segment_start < range.end {
                let line_start = self.text[..segment_start]
                    .rfind('\n')
                    .map_or(0, |index| index + 1);
                let line_end = self.text[segment_start..]
                    .find('\n')
                    .map_or(self.text.len(), |index| segment_start + index);
                let segment_end = range.end.min(line_end);

                if segment_start < segment_end {
                    self.paint_line_background(
                        line_start,
                        segment_start..segment_end,
                        *color,
                        window,
                    );
                }

                if line_end == self.text.len() {
                    break;
                }
                segment_start = line_end + 1;
            }
        }
    }

    fn paint_line_background(
        &self,
        line_start: usize,
        range: Range<usize>,
        color: Hsla,
        window: &mut Window,
    ) {
        let text_layout = self.styled.layout();
        let Some(layout) = text_layout.line_layout_for_index(range.start) else {
            return;
        };
        let Some(line_origin) = text_layout.position_for_index(line_start) else {
            return;
        };
        let line_height = text_layout.line_height();
        let unwrapped = &layout.unwrapped_layout;
        let local_range = (range.start - line_start)..(range.end - line_start);
        let mut visual_line_start = 0;

        for (visual_line, visual_line_end) in layout
            .wrap_boundaries()
            .iter()
            .map(|boundary| unwrapped.runs[boundary.run_ix].glyphs[boundary.glyph_ix].index)
            .chain([layout.len()])
            .enumerate()
        {
            let start = local_range.start.max(visual_line_start);
            let end = local_range.end.min(visual_line_end);
            if start < end {
                let visual_line_x = unwrapped.x_for_index(visual_line_start);
                let left = unwrapped.x_for_index(start) - visual_line_x;
                let right = unwrapped.x_for_index(end) - visual_line_x;
                if right > left {
                    let bounds = Bounds::new(
                        point(
                            line_origin.x + left,
                            line_origin.y + line_height * visual_line + Self::CODE_BACKGROUND_INSET,
                        ),
                        size(
                            right - left,
                            line_height - Self::CODE_BACKGROUND_INSET * 2.0,
                        ),
                    );
                    window
                        .paint_quad(fill(bounds, color).corner_radii(Self::CODE_BACKGROUND_RADIUS));
                }
            }
            if visual_line_end >= local_range.end {
                break;
            }
            visual_line_start = visual_line_end;
        }
    }
}

impl Element for InlineText {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        self.styled.request_layout(id, inspector_id, window, cx)
    }

    fn prepaint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.styled
            .prepaint(id, inspector_id, bounds, state, window, cx);
    }

    fn paint(
        &mut self,
        id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        state: &mut Self::RequestLayoutState,
        prepaint: &mut Self::PrepaintState,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.paint_backgrounds(window);
        self.styled
            .paint(id, inspector_id, bounds, state, prepaint, window, cx);
    }
}

impl IntoElement for InlineText {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

fn append_inlines(
    nodes: &[Node],
    style: InlineStyle,
    colors: UiPalette,
    output: &mut InlineOutput,
) {
    for node in nodes {
        match node {
            Node::Text(text) => append_inline_text(&text.value, style, colors, output),
            Node::InlineCode(code) => {
                let mut code_style = style;
                code_style.code = true;
                append_inline_text(
                    &format!("\u{a0}{}\u{a0}", code.value),
                    code_style,
                    colors,
                    output,
                );
            }
            Node::InlineMath(math) => {
                let mut code_style = style;
                code_style.code = true;
                append_inline_text(
                    &format!("\u{a0}{}\u{a0}", math.value),
                    code_style,
                    colors,
                    output,
                );
            }
            Node::Strong(strong) => {
                let mut nested = style;
                nested.strong = true;
                append_inlines(&strong.children, nested, colors, output);
            }
            Node::Emphasis(emphasis) => {
                let mut nested = style;
                nested.emphasis = true;
                append_inlines(&emphasis.children, nested, colors, output);
            }
            Node::Delete(deleted) => {
                let mut nested = style;
                nested.deleted = true;
                append_inlines(&deleted.children, nested, colors, output);
            }
            Node::Link(link) => {
                let mut nested = style;
                nested.link = true;
                append_inlines(&link.children, nested, colors, output);
            }
            Node::LinkReference(link) => {
                let mut nested = style;
                nested.link = true;
                append_inlines(&link.children, nested, colors, output);
            }
            Node::Break(_) => append_inline_text("\n", style, colors, output),
            Node::Image(image) => append_inline_text(
                if image.alt.is_empty() {
                    "image"
                } else {
                    image.alt.as_str()
                },
                InlineStyle {
                    emphasis: true,
                    ..style
                },
                colors,
                output,
            ),
            Node::FootnoteReference(reference) => append_inline_text(
                &format!("[{}]", reference.identifier),
                InlineStyle {
                    link: true,
                    ..style
                },
                colors,
                output,
            ),
            Node::Html(html) => append_inline_text(&html.value, style, colors, output),
            Node::Paragraph(paragraph) => {
                append_inlines(&paragraph.children, style, colors, output)
            }
            Node::TableCell(cell) => append_inlines(&cell.children, style, colors, output),
            _ => {}
        }
    }
}

fn append_inline_text(
    value: &str,
    style: InlineStyle,
    colors: UiPalette,
    output: &mut InlineOutput,
) {
    if value.is_empty() {
        return;
    }
    let start = output.text.len();
    output.text.push_str(value);
    let end = output.text.len();
    let mut highlight = HighlightStyle::default();
    if style.strong {
        highlight.font_weight = Some(FontWeight::SEMIBOLD);
    }
    if style.emphasis {
        highlight.font_style = Some(FontStyle::Italic);
    }
    if style.deleted {
        highlight.strikethrough = Some(StrikethroughStyle {
            thickness: px(1.0),
            ..Default::default()
        });
    }
    if style.code {
        output
            .backgrounds
            .push((start..end, colors.markdown_inline_code));
    }
    if style.link {
        highlight.color = Some(colors.markdown_link);
    }
    if highlight != HighlightStyle::default() {
        output.highlights.push((start..end, highlight));
    }
}

#[cfg(test)]
mod tests {
    use gpui::{AssetSource, FontWeight, Hsla, px, rgb};
    use markdown::{ParseOptions, mdast::Node};

    use super::{InlineText, build_math, heading_style, inline_text, root_block_gap};
    use crate::assets::DesktopAssets;
    use crate::ui_theme::{UiPalette, metrics};

    fn blocks(source: &str) -> Vec<Node> {
        match markdown::to_mdast(source, &ParseOptions::gfm()).unwrap() {
            Node::Root(root) => root.children,
            _ => unreachable!(),
        }
    }

    fn test_palette() -> UiPalette {
        let color: Hsla = rgb(0x123456).into();
        UiPalette {
            canvas: color,
            surface: color,
            sidebar: color,
            border: color,
            text: color,
            muted_text: color,
            subtle: color,
            hover: color,
            selected: color,
            primary: color,
            overlay: color,
            danger: color,
            warning: color,
            assistant: color,
            user_bubble: color,
            markdown_text: color,
            markdown_inline_code: color,
            markdown_code_block: color,
            markdown_code_banner: color,
            markdown_link: color,
            markdown_quote: color,
        }
    }

    #[test]
    fn dsh_headings_use_explicit_size_and_line_height_pairs() {
        assert_eq!(heading_style(1).0, 24.0);
        assert_eq!(heading_style(1).1, 34.0);
        assert_eq!(heading_style(1).2, FontWeight::SEMIBOLD);
        assert_eq!(heading_style(2).0, 22.0);
        assert_eq!(heading_style(2).2, FontWeight::SEMIBOLD);
        assert_eq!(heading_style(3).1, 30.0);
        assert_eq!(heading_style(3).2, FontWeight::SEMIBOLD);
        assert_eq!(heading_style(4).0, 16.0);
    }

    #[test]
    fn block_rhythm_preserves_sections_and_tight_heading_lists() {
        let nodes = blocks("paragraph\n\n## section\n\ntext");
        assert_eq!(root_block_gap(None, &nodes[0], true), 0.0);
        assert_eq!(
            root_block_gap(Some(&nodes[0]), &nodes[1], false),
            metrics::MARKDOWN_SECTION_GAP
        );
        assert_eq!(
            root_block_gap(Some(&nodes[1]), &nodes[2], false),
            metrics::MARKDOWN_BLOCK_GAP
        );

        let nodes = blocks("#### details\n\n- one\n- two");
        assert_eq!(root_block_gap(Some(&nodes[0]), &nodes[1], false), 8.0);
    }

    #[test]
    fn inline_code_backgrounds_are_separate_inset_shapes() {
        let nodes = blocks("`agent::Model`\n`Agent::set_model`");
        let Node::Paragraph(paragraph) = &nodes[0] else {
            unreachable!();
        };
        let inline = inline_text(&paragraph.children, test_palette());

        assert_eq!(inline.backgrounds.len(), 2);
        assert!(InlineText::CODE_BACKGROUND_INSET > px(0.0));
        assert_eq!(
            &inline.text[inline.backgrounds[0].0.clone()],
            "\u{a0}agent::Model\u{a0}"
        );
        assert_eq!(
            &inline.text[inline.backgrounds[1].0.clone()],
            "\u{a0}Agent::set_model\u{a0}"
        );
    }

    #[test]
    fn latex_is_rendered_to_a_nonempty_embedded_svg() {
        let rendered = build_math(r"\frac{-b \pm \sqrt{b^2-4ac}}{2a}", true).unwrap();
        let asset = DesktopAssets
            .load(rendered.asset.as_ref())
            .unwrap()
            .unwrap();

        assert!(rendered.width > 0.0 && rendered.height > 0.0);
        assert!(asset.starts_with(b"<svg "));
        assert!(build_math(r"\frac{", true).is_err());
    }

    #[test]
    fn inline_math_tracks_its_svg_baseline() {
        let capitals = build_math("K,V", false).unwrap();
        let fraction = build_math(r"\frac{1}{2}", false).unwrap();

        assert!(capitals.baseline_offset > 1.0);
        assert!(fraction.baseline_offset > capitals.baseline_offset);
        assert!(fraction.baseline_offset < fraction.height);
    }
}
