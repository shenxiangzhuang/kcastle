use std::{
    collections::HashMap,
    ops::Range,
    sync::{Mutex, OnceLock},
};

use gpui::{
    AnyElement, App, Bounds, Context, Element, ElementId, FontStyle, FontWeight, GlobalElementId,
    HighlightStyle, Hsla, InspectorElementId, InteractiveElement, IntoElement, LayoutId,
    ParentElement, Pixels, SharedString, StrikethroughStyle, StyleRefinement, Styled, StyledText,
    Window, div, fill, point, prelude::FluentBuilder, px, rems, size, svg,
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
            window,
        ),
        Node::Heading(heading) => {
            let (size, line_height, weight) = heading_style(heading.depth);
            inline_block(
                &heading.children,
                size,
                line_height,
                weight,
                context,
                window,
            )
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
        Node::Math(math) => render_math(&math.value, None, context)
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
            window,
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
    window: &mut Window,
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
                        window,
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
    window: &Window,
) -> AnyElement {
    if !contains_inline_math(nodes) {
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
        .items_center()
        .w_full()
        .min_w(px(0.0))
        .whitespace_normal()
        .text_size(px(size))
        .line_height(px(line_height))
        .font_weight(weight)
        .when(cfg!(test), |element| {
            element.debug_selector(|| "inline-math-flow".to_owned())
        });
    let text_baseline = shaped_text_baseline(nodes, size, line_height, weight, context, window);
    for piece in inline_pieces(nodes, context.colors) {
        match piece {
            InlinePiece::Text(output) => {
                for element in inline_flow_text(output) {
                    body = body.child(element);
                }
            }
            InlinePiece::Math { source, style } => {
                body = body.child(
                    render_math(&source, Some((text_baseline, line_height, size)), context)
                        .unwrap_or_else(|| inline_math_fallback(&source, style, context.colors)),
                );
            }
        }
    }
    body.into_any_element()
}

fn contains_inline_math(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match node {
        Node::InlineMath(_) => true,
        Node::Strong(strong) => contains_inline_math(&strong.children),
        Node::Emphasis(emphasis) => contains_inline_math(&emphasis.children),
        Node::Delete(deleted) => contains_inline_math(&deleted.children),
        Node::Link(link) => contains_inline_math(&link.children),
        Node::LinkReference(link) => contains_inline_math(&link.children),
        Node::Paragraph(paragraph) => contains_inline_math(&paragraph.children),
        Node::TableCell(cell) => contains_inline_math(&cell.children),
        _ => false,
    })
}

#[derive(Clone, Debug)]
struct RenderedMath {
    asset: SharedString,
    width: f32,
    height: f32,
    baseline: f32,
}

type MathCache = HashMap<(String, bool, u32), Result<RenderedMath, String>>;
static MATH_CACHE: OnceLock<Mutex<MathCache>> = OnceLock::new();

fn render_math(
    source: &str,
    inline_text_metrics: Option<(f32, f32, f32)>,
    context: &BlockContext<'_>,
) -> Option<AnyElement> {
    let display = inline_text_metrics.is_none();
    let font_size = inline_text_metrics
        .map(|(_, _, font_size)| font_size)
        .unwrap_or(20.0);
    let rendered = cached_math(source, display, font_size).ok()?;
    let width = rendered.width;
    let inline_offset = inline_text_metrics.map(|(text_baseline, line_height, _)| {
        inline_math_offset(&rendered, text_baseline, line_height)
    });
    // GPUI exposes no baseline for these flex items; asymmetric margins move the
    // centered SVG by half their difference while preserving the line's bounds.
    let (margin_top, margin_bottom) = inline_offset.map(inline_math_margins).unwrap_or_default();
    let formula = svg()
        .path(rendered.asset)
        .flex_none()
        .w(px(width))
        .h(px(rendered.height))
        .mt(px(margin_top))
        .mb(px(margin_bottom))
        .text_color(context.colors.markdown_text)
        .when(cfg!(test), |element| {
            let source = source.to_owned();
            element.debug_selector(move || format!("math:{source}"))
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

fn inline_math_offset(rendered: &RenderedMath, text_baseline: f32, line_height: f32) -> f32 {
    text_baseline - line_height / 2.0 - (rendered.baseline - rendered.height / 2.0)
}

fn shaped_text_baseline(
    nodes: &[Node],
    size: f32,
    line_height: f32,
    weight: FontWeight,
    context: &BlockContext<'_>,
    window: &Window,
) -> f32 {
    let mut output = InlineOutput::default();
    append_inlines(nodes, InlineStyle::default(), context.colors, &mut output);
    let text: SharedString = output.text.replace('\n', " ").into();
    let mut style = window.text_style();
    style.font_weight = weight;
    let shaped =
        window
            .text_system()
            .shape_line(text.clone(), px(size), &[style.to_run(text.len())], None);
    let ascent = f32::from(shaped.ascent);
    let descent = f32::from(shaped.descent);
    (line_height - ascent - descent) / 2.0 + ascent
}

fn inline_math_margins(offset: f32) -> (f32, f32) {
    (offset.max(0.0) * 2.0, (-offset).max(0.0) * 2.0)
}

fn cached_math(source: &str, display: bool, font_size: f32) -> Result<RenderedMath, String> {
    let key = (source.to_owned(), display, font_size.to_bits());
    let cache = MATH_CACHE.get_or_init(Default::default);
    if let Some(rendered) = cache
        .lock()
        .expect("math cache poisoned")
        .get(&key)
        .cloned()
    {
        return rendered;
    }

    let rendered = build_math_at_size(source, display, font_size);
    // ponytail: process-wide cache; add eviction only if long sessions show material growth.
    cache
        .lock()
        .expect("math cache poisoned")
        .insert(key, rendered.clone());
    rendered
}

#[cfg(test)]
fn build_math(source: &str, display: bool) -> Result<RenderedMath, String> {
    build_math_at_size(source, display, if display { 20.0 } else { 16.0 })
}

fn build_math_at_size(source: &str, display: bool, font_size: f32) -> Result<RenderedMath, String> {
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
    if !font_size.is_finite() || font_size <= 0.0 {
        return Err("formula font size must be positive and finite".to_owned());
    }
    let font_size = f64::from(font_size);
    let padding = if display {
        font_size / 10.0
    } else {
        font_size / 16.0
    };
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
        baseline: (display_list.height * font_size + padding) as f32,
    })
}

fn inline_text(nodes: &[Node], colors: UiPalette) -> InlineText {
    let mut output = InlineOutput::default();
    append_inlines(nodes, InlineStyle::default(), colors, &mut output);
    InlineText::new(output)
}

fn inline_wrap_ranges(text: &str) -> Vec<Range<usize>> {
    let mut start = 0;
    unicode_linebreak::linebreaks(text)
        .filter_map(|(end, _)| {
            let range = (start < end).then_some(start..end);
            start = end;
            range
        })
        .collect()
}

fn inline_flow_text(output: InlineOutput) -> Vec<AnyElement> {
    inline_wrap_ranges(&output.text)
        .into_iter()
        .flat_map(|range| {
            let explicit_break = output.text[range.clone()].ends_with('\n');
            let mut content = range.clone();
            if explicit_break {
                content.end -= 1;
                if content.start < content.end && output.text[content.clone()].ends_with('\r') {
                    content.end -= 1;
                }
            }

            let mut elements = Vec::with_capacity(2);
            if content.start < content.end {
                let chunk = output.slice(content);
                let selector = chunk.text.clone();
                elements.push(
                    div()
                        .min_w(px(0.0))
                        .max_w_full()
                        .flex_initial()
                        .whitespace_normal()
                        .when(cfg!(test), |element| {
                            element.debug_selector(move || format!("inline-text:{selector}"))
                        })
                        .child(InlineText::new(chunk))
                        .into_any_element(),
                );
            }
            if explicit_break {
                elements.push(div().w_full().h(px(0.0)).into_any_element());
            }
            elements
        })
        .collect()
}

enum InlinePiece {
    Text(InlineOutput),
    Math { source: String, style: InlineStyle },
}

fn inline_pieces(nodes: &[Node], colors: UiPalette) -> Vec<InlinePiece> {
    let mut pieces = Vec::new();
    append_inline_pieces(nodes, InlineStyle::default(), colors, &mut pieces);
    pieces
}

fn append_inline_pieces(
    nodes: &[Node],
    style: InlineStyle,
    colors: UiPalette,
    pieces: &mut Vec<InlinePiece>,
) {
    for node in nodes {
        match node {
            Node::InlineMath(math) => pieces.push(InlinePiece::Math {
                source: math.value.clone(),
                style,
            }),
            Node::Strong(strong) => append_inline_pieces(
                &strong.children,
                InlineStyle {
                    strong: true,
                    ..style
                },
                colors,
                pieces,
            ),
            Node::Emphasis(emphasis) => append_inline_pieces(
                &emphasis.children,
                InlineStyle {
                    emphasis: true,
                    ..style
                },
                colors,
                pieces,
            ),
            Node::Delete(deleted) => append_inline_pieces(
                &deleted.children,
                InlineStyle {
                    deleted: true,
                    ..style
                },
                colors,
                pieces,
            ),
            Node::Link(link) => append_inline_pieces(
                &link.children,
                InlineStyle {
                    link: true,
                    ..style
                },
                colors,
                pieces,
            ),
            Node::LinkReference(link) => append_inline_pieces(
                &link.children,
                InlineStyle {
                    link: true,
                    ..style
                },
                colors,
                pieces,
            ),
            Node::Paragraph(paragraph) => {
                append_inline_pieces(&paragraph.children, style, colors, pieces);
            }
            Node::TableCell(cell) => {
                append_inline_pieces(&cell.children, style, colors, pieces);
            }
            _ => {
                if !matches!(pieces.last(), Some(InlinePiece::Text(_))) {
                    pieces.push(InlinePiece::Text(InlineOutput::default()));
                }
                let Some(InlinePiece::Text(output)) = pieces.last_mut() else {
                    unreachable!();
                };
                append_inlines(std::slice::from_ref(node), style, colors, output);
            }
        }
    }
}

fn inline_math_fallback(source: &str, style: InlineStyle, colors: UiPalette) -> AnyElement {
    let mut output = InlineOutput::default();
    append_inline_text(
        &format!("\u{a0}{source}\u{a0}"),
        InlineStyle {
            code: true,
            ..style
        },
        colors,
        &mut output,
    );
    div()
        .when(cfg!(test), |element| {
            let source = source.to_owned();
            element.debug_selector(move || format!("math-fallback:{source}"))
        })
        .child(InlineText::new(output))
        .into_any_element()
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

impl InlineOutput {
    fn slice(&self, range: Range<usize>) -> Self {
        let adjust = |candidate: &Range<usize>| {
            let start = candidate.start.max(range.start);
            let end = candidate.end.min(range.end);
            (start < end).then_some((start - range.start)..(end - range.start))
        };

        Self {
            text: self.text[range.clone()].to_owned(),
            highlights: self
                .highlights
                .iter()
                .filter_map(|(candidate, style)| adjust(candidate).map(|range| (range, *style)))
                .collect(),
            backgrounds: self
                .backgrounds
                .iter()
                .filter_map(|(candidate, color)| adjust(candidate).map(|range| (range, *color)))
                .collect(),
        }
    }
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
    use gpui::{
        AssetSource, Context, FontWeight, Hsla, IntoElement, ParentElement, Render, Styled,
        TestAppContext, Window, div, px, rgb, size,
    };
    use markdown::{ParseOptions, mdast::Node};
    use proptest::prelude::*;

    use super::{InlineText, build_math, heading_style, inline_text, root_block_gap};
    use crate::assets::DesktopAssets;
    use crate::streaming_markdown::MarkdownBlock;
    use crate::ui_theme::{UiPalette, metrics};

    fn blocks(source: &str) -> Vec<Node> {
        match markdown::to_mdast(source, &ParseOptions::gfm()).unwrap() {
            Node::Root(root) => root.children,
            _ => unreachable!(),
        }
    }

    fn math_blocks(source: &str) -> Vec<Node> {
        let mut options = ParseOptions::gfm();
        options.constructs.math_text = true;
        match markdown::to_mdast(source, &options).unwrap() {
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

    struct InlineLayoutHarness {
        block: MarkdownBlock,
        text_size: f32,
        line_height: f32,
    }

    impl InlineLayoutHarness {
        fn new(source: &str) -> Self {
            let mut nodes = math_blocks(source);
            assert_eq!(nodes.len(), 1);
            Self {
                block: MarkdownBlock {
                    key: 0,
                    source: source.to_owned().into(),
                    node: nodes.remove(0),
                },
                text_size: 16.0,
                line_height: metrics::MESSAGE_LINE_HEIGHT,
            }
        }
    }

    impl Render for InlineLayoutHarness {
        fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let Node::Paragraph(paragraph) = &self.block.node else {
                unreachable!();
            };
            let available_width = f32::from(window.viewport_size().width);
            let context = super::BlockContext {
                message_key: 0,
                generation: 0,
                revision: None,
                block: &self.block,
                streaming: false,
                colors: test_palette(),
                available_width,
            };
            div().size_full().child(super::inline_block(
                &paragraph.children,
                self.text_size,
                self.line_height,
                FontWeight::NORMAL,
                &context,
                window,
            ))
        }
    }

    struct DisplayMathHarness {
        source: String,
        block: MarkdownBlock,
    }

    impl DisplayMathHarness {
        fn new(source: &str) -> Self {
            Self {
                source: source.to_owned(),
                block: MarkdownBlock {
                    key: 0,
                    source: source.to_owned().into(),
                    node: Node::Paragraph(markdown::mdast::Paragraph {
                        children: Vec::new(),
                        position: None,
                    }),
                },
            }
        }
    }

    impl Render for DisplayMathHarness {
        fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let context = super::BlockContext {
                message_key: 0,
                generation: 0,
                revision: None,
                block: &self.block,
                streaming: false,
                colors: test_palette(),
                available_width: f32::from(window.viewport_size().width),
            };
            div()
                .size_full()
                .child(super::render_math(&self.source, None, &context).unwrap())
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
    fn common_latex_constructs_render_inline_and_display() {
        let sources = [
            "x",
            "x^2",
            "x_i",
            "x_i^2",
            r"\alpha + \beta = \gamma",
            r"\Gamma \Delta \Omega",
            r"\frac{a+b}{c}",
            r"\sqrt{x}",
            r"\sqrt[n]{x}",
            r"\sum_{i=1}^{n} i",
            r"\int_0^1 x\,dx",
            r"\lim_{n \to \infty} a_n",
            r"\left(\frac{a}{b}\right)",
            r"a \le b",
            r"a \ne b",
            r"x \in \mathbb{R}",
            r"\mathbf{x}",
            r"\mathrm{softmax}(x)",
            r"\text{draft tokens}",
            r"\hat{x} + \bar{y} + \vec{z}",
            r"\overline{AB}",
            r"\begin{matrix} a & b \\ c & d \end{matrix}",
            r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}",
            r"\begin{cases} x & x > 0 \\ -x & x \le 0 \end{cases}",
            r"a \cdot b \times c \approx d",
            r"t_{\text{step}}",
        ];

        for source in sources {
            for display in [false, true] {
                let rendered = build_math(source, display)
                    .unwrap_or_else(|error| panic!("failed to render {source:?}: {error}"));
                assert!(
                    rendered.width.is_finite() && rendered.width > 0.0,
                    "{source}"
                );
                assert!(
                    rendered.height.is_finite() && rendered.height > 0.0,
                    "{source}"
                );
                let asset = DesktopAssets
                    .load(rendered.asset.as_ref())
                    .unwrap()
                    .unwrap();
                assert!(asset.starts_with(b"<svg "), "{source}");
            }
        }
    }

    #[test]
    fn inline_math_aligns_svg_and_text_baselines() {
        let line_height = 28.0;
        let text_baseline = 20.0;
        for source in ["q_t", "K,V", r"\gamma", r"\frac{1}{2}"] {
            let rendered = build_math(source, false).unwrap();
            let offset = super::inline_math_offset(&rendered, text_baseline, line_height);
            let (margin_top, margin_bottom) = super::inline_math_margins(offset);
            let aligned_baseline = -(rendered.height + margin_top + margin_bottom) / 2.0
                + margin_top
                + rendered.baseline;
            let expected_baseline = text_baseline - line_height / 2.0;

            assert!((aligned_baseline - expected_baseline).abs() < f32::EPSILON);
        }
    }

    #[gpui::test]
    fn inline_math_uses_remaining_width_before_wrapping(cx: &mut TestAppContext) {
        let source = "回顾上轮结论：decode 阶段计算强度 $I \\approx 1$甲后续文本足够长，用于验证公式后内容不会整块移到下一行。";
        let (_, cx) = cx.add_window_view(|_, _| InlineLayoutHarness::new(source));
        cx.simulate_resize(size(px(760.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let formula = cx.debug_bounds(r"math:I \approx 1").unwrap();
        let following = cx.debug_bounds("inline-text:甲").unwrap();
        let vertical_overlap = (formula.origin.y + formula.size.height)
            .min(following.origin.y + following.size.height)
            - formula.origin.y.max(following.origin.y);

        assert!(vertical_overlap > px(0.0), "{formula:?} {following:?}");
        assert!(following.origin.x >= formula.origin.x + formula.size.width - px(1.0));
    }

    #[gpui::test]
    fn inline_math_scales_with_the_surrounding_text(cx: &mut TestAppContext) {
        let (view, cx) = cx.add_window_view(|_, _| InlineLayoutHarness::new("heading $\\gamma$"));
        cx.simulate_resize(size(px(520.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();
        let body_formula = cx.debug_bounds(r"math:\gamma").unwrap();

        view.update(cx, |harness, cx| {
            harness.text_size = 24.0;
            harness.line_height = 34.0;
            cx.notify();
        });
        cx.refresh().unwrap();
        cx.run_until_parked();
        let heading_formula = cx.debug_bounds(r"math:\gamma").unwrap();

        assert!(
            heading_formula.size.height > body_formula.size.height * 1.25,
            "body={body_formula:?} heading={heading_formula:?}"
        );
    }

    #[gpui::test]
    fn inline_math_nested_in_bold_text_reaches_the_svg_renderer(cx: &mut TestAppContext) {
        let source = "**一次前向生成全部 $\\gamma$ 个草稿 token**";
        let (_, cx) = cx.add_window_view(|_, _| InlineLayoutHarness::new(source));
        cx.simulate_resize(size(px(620.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        assert!(cx.debug_bounds(r"math:\gamma").is_some());
        assert!(cx.debug_bounds(r"math-fallback:\gamma").is_none());
    }

    #[gpui::test]
    fn tall_inline_fraction_is_contained_and_does_not_overlap_text(cx: &mut TestAppContext) {
        let source = r"left$\frac{1}{2}$right";
        let (_, cx) = cx.add_window_view(|_, _| InlineLayoutHarness::new(source));
        cx.simulate_resize(size(px(520.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let flow = cx.debug_bounds("inline-math-flow").unwrap();
        let left = cx.debug_bounds("inline-text:left").unwrap();
        let formula = cx.debug_bounds(r"math:\frac{1}{2}").unwrap();
        let right = cx.debug_bounds("inline-text:right").unwrap();

        assert!(formula.origin.y >= flow.origin.y - px(1.0));
        assert!(
            formula.origin.y + formula.size.height <= flow.origin.y + flow.size.height + px(1.0),
            "flow={flow:?} formula={formula:?}"
        );
        assert!(left.origin.x + left.size.width <= formula.origin.x + px(1.0));
        assert!(formula.origin.x + formula.size.width <= right.origin.x + px(1.0));
    }

    #[gpui::test]
    fn display_math_is_centered_when_it_fits(cx: &mut TestAppContext) {
        let source = r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}";
        cx.update(gpui_component::init);
        let (_, cx) = cx.add_window_view(|_, _| DisplayMathHarness::new(source));
        cx.simulate_resize(size(px(520.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let formula = cx
            .debug_bounds(r"math:\frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
            .unwrap();
        let formula_center = formula.origin.x + formula.size.width / 2.0;

        assert!((formula_center - px(260.0)).abs() < px(1.0), "{formula:?}");
    }

    #[test]
    fn inline_math_prose_exposes_unicode_wrap_opportunities() {
        let text = "回顾上轮结论：decode 阶段计算强度，深度访存受限。";
        let ranges = super::inline_wrap_ranges(text);
        let pieces = ranges
            .iter()
            .map(|range| &text[range.clone()])
            .collect::<Vec<_>>();

        assert!(pieces.len() > 3);
        assert_eq!(pieces.concat(), text);
        assert!(pieces.contains(&"decode "));
    }

    #[test]
    fn inline_math_nested_in_strong_text_is_rendered_as_math() {
        let nodes = math_blocks("**一次前向生成全部 $\\gamma$ 个草稿 token**");
        let Node::Paragraph(paragraph) = &nodes[0] else {
            unreachable!();
        };
        assert!(super::contains_inline_math(&paragraph.children));

        let pieces = super::inline_pieces(&paragraph.children, test_palette());
        assert!(matches!(
            pieces.as_slice(),
            [
                super::InlinePiece::Text(before),
                super::InlinePiece::Math { source, style },
                super::InlinePiece::Text(after),
            ] if before.text == "一次前向生成全部 "
                && source == r"\gamma"
                && style.strong
                && after.text == " 个草稿 token"
        ));
    }

    #[test]
    fn inline_math_survives_common_markdown_style_nesting() {
        let cases = [
            ("**before $x$ after**", (true, false, false, false)),
            ("*before $x$ after*", (false, true, false, false)),
            ("~~before $x$ after~~", (false, false, true, false)),
            (
                "[before $x$ after](https://example.com)",
                (false, false, false, true),
            ),
            ("***before $x$ after***", (true, true, false, false)),
        ];

        for (source, expected) in cases {
            let nodes = math_blocks(source);
            let Node::Paragraph(paragraph) = &nodes[0] else {
                panic!("expected paragraph for {source:?}");
            };
            let pieces = super::inline_pieces(&paragraph.children, test_palette());
            let formulas = pieces
                .iter()
                .filter_map(|piece| match piece {
                    super::InlinePiece::Math { source, style } => Some((source, style)),
                    super::InlinePiece::Text(_) => None,
                })
                .collect::<Vec<_>>();

            assert_eq!(formulas.len(), 1, "{source}");
            assert_eq!(formulas[0].0, "x", "{source}");
            let style = formulas[0].1;
            assert_eq!(
                (style.strong, style.emphasis, style.deleted, style.link),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn multiple_inline_formulas_preserve_order_and_surrounding_text() {
        let nodes = math_blocks("速度 $S$，当 $\\gamma$ 增大时，$t_{step}$ 不应消失。");
        let Node::Paragraph(paragraph) = &nodes[0] else {
            unreachable!();
        };
        let pieces = super::inline_pieces(&paragraph.children, test_palette());
        let formulas = pieces
            .iter()
            .filter_map(|piece| match piece {
                super::InlinePiece::Math { source, .. } => Some(source.as_str()),
                super::InlinePiece::Text(_) => None,
            })
            .collect::<Vec<_>>();
        let text = pieces
            .iter()
            .filter_map(|piece| match piece {
                super::InlinePiece::Text(output) => Some(output.text.as_str()),
                super::InlinePiece::Math { .. } => None,
            })
            .collect::<String>();

        assert_eq!(formulas, ["S", r"\gamma", "t_{step}"]);
        assert_eq!(text, "速度 ，当  增大时， 不应消失。");
    }

    #[test]
    fn consecutive_inline_formulas_remain_two_atomic_formulas() {
        let nodes = math_blocks("before $x$ $y$ after");
        let Node::Paragraph(paragraph) = &nodes[0] else {
            unreachable!();
        };
        let pieces = super::inline_pieces(&paragraph.children, test_palette());
        let formulas = pieces
            .iter()
            .filter_map(|piece| match piece {
                super::InlinePiece::Math { source, .. } => Some(source.as_str()),
                super::InlinePiece::Text(_) => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(formulas, ["x", "y"]);
    }

    #[gpui::test]
    fn hard_break_after_inline_math_starts_the_following_text_on_a_new_line(
        cx: &mut TestAppContext,
    ) {
        let (_, cx) =
            cx.add_window_view(|_, _| InlineLayoutHarness::new("before $x$  \nafterbreak"));
        cx.simulate_resize(size(px(520.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let formula = cx.debug_bounds("math:x").unwrap();
        let following = cx.debug_bounds("inline-text:afterbreak").unwrap();

        assert!(
            following.origin.y >= formula.origin.y + formula.size.height,
            "{formula:?} {following:?}"
        );
    }

    #[gpui::test]
    fn invalid_inline_math_fallback_remains_visible(cx: &mut TestAppContext) {
        let (_, cx) = cx.add_window_view(|_, _| InlineLayoutHarness::new(r"before $\frac{$ after"));
        cx.simulate_resize(size(px(520.0), px(180.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();

        let fallback = cx.debug_bounds(r"math-fallback:\frac{").unwrap();
        assert!(fallback.size.width > px(0.0));
        assert!(fallback.size.height > px(0.0));
    }

    #[test]
    fn inline_wrap_slices_preserve_text_styles_and_code_backgrounds() {
        let nodes = blocks("**decode phase** `token`");
        let Node::Paragraph(paragraph) = &nodes[0] else {
            unreachable!();
        };
        let mut output = super::InlineOutput::default();
        super::append_inlines(
            &paragraph.children,
            super::InlineStyle::default(),
            test_palette(),
            &mut output,
        );
        let start = output.text.find("phase").unwrap();
        let end = output.text.find("token").unwrap() + "token".len();
        let slice = output.slice(start..end);

        assert_eq!(&slice.text[slice.highlights[0].0.clone()], "phase");
        assert_eq!(&slice.text[slice.backgrounds[0].0.clone()], "\u{a0}token");
    }

    proptest! {
        #[test]
        fn unicode_wrap_ranges_are_a_lossless_utf8_partition(
            characters in proptest::collection::vec(any::<char>(), 0..96)
        ) {
            let text = characters.into_iter().collect::<String>();
            let ranges = super::inline_wrap_ranges(&text);
            let mut cursor = 0;
            let mut rebuilt = String::new();

            for range in ranges {
                prop_assert_eq!(range.start, cursor);
                prop_assert!(range.start < range.end);
                prop_assert!(text.is_char_boundary(range.start));
                prop_assert!(text.is_char_boundary(range.end));
                rebuilt.push_str(&text[range.clone()]);
                cursor = range.end;
            }

            prop_assert_eq!(cursor, text.len());
            prop_assert_eq!(rebuilt, text);
        }
    }
}
