use std::ops::Range;

use gpui::{
    AnyElement, Context, FontStyle, FontWeight, HighlightStyle, IntoElement, ParentElement,
    SharedString, StrikethroughStyle, StyleRefinement, Styled, StyledText, Window, div,
    prelude::FluentBuilder, px, rems,
};
use gpui_component::ActiveTheme;
use gpui_component::clipboard::Clipboard;
use gpui_component::scroll::ScrollableElement;
use gpui_component::text::{TextView, TextViewStyle};
use markdown::mdast::Node;

use crate::app::DesktopApp;
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
            context.colors,
            16.0,
            metrics::MESSAGE_LINE_HEIGHT,
            FontWeight::NORMAL,
        ),
        Node::Heading(heading) => {
            let (size, line_height, weight) = heading_style(heading.depth);
            inline_block(&heading.children, context.colors, size, line_height, weight)
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
        Node::Math(math) => render_code_block("math", &math.value, context, path, window, cx),
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
            context.colors,
            16.0,
            metrics::MESSAGE_LINE_HEIGHT,
            FontWeight::NORMAL,
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
                    .child(inline_text(&cell.children, context.colors)),
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
        1 => (24.0, 34.0, FontWeight::BOLD),
        2 => (22.0, 32.0, FontWeight::BOLD),
        3 => (20.0, 30.0, FontWeight::BOLD),
        _ => (16.0, 28.0, FontWeight::SEMIBOLD),
    }
}

fn inline_block(
    nodes: &[Node],
    colors: UiPalette,
    size: f32,
    line_height: f32,
    weight: FontWeight,
) -> AnyElement {
    div()
        .w_full()
        .min_w(px(0.0))
        .whitespace_normal()
        .text_size(px(size))
        .line_height(px(line_height))
        .font_weight(weight)
        .child(inline_text(nodes, colors))
        .into_any_element()
}

fn inline_text(nodes: &[Node], colors: UiPalette) -> StyledText {
    let mut output = InlineOutput::default();
    append_inlines(nodes, InlineStyle::default(), colors, &mut output);
    StyledText::new(output.text).with_highlights(output.highlights)
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
        highlight.background_color = Some(colors.markdown_inline_code);
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
    use markdown::{ParseOptions, mdast::Node};

    use super::{heading_style, root_block_gap};
    use crate::ui_theme::metrics;

    fn blocks(source: &str) -> Vec<Node> {
        match markdown::to_mdast(source, &ParseOptions::gfm()).unwrap() {
            Node::Root(root) => root.children,
            _ => unreachable!(),
        }
    }

    #[test]
    fn dsh_headings_use_explicit_size_and_line_height_pairs() {
        assert_eq!(heading_style(1).0, 24.0);
        assert_eq!(heading_style(1).1, 34.0);
        assert_eq!(heading_style(2).0, 22.0);
        assert_eq!(heading_style(3).1, 30.0);
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
}
