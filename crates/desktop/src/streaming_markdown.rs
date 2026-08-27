use gpui::SharedString;
use markdown::{
    ParseOptions,
    mdast::{Math, Node, Paragraph},
    unist::Position,
};

const UNSTABLE_TAIL_BLOCKS: usize = 2;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MarkdownBlock {
    pub(crate) key: usize,
    pub(crate) source: SharedString,
    pub(crate) node: Node,
}

#[derive(Debug, Default)]
pub(crate) struct StreamingMarkdownState {
    previous_text: String,
    tail_start: usize,
    frozen: Vec<MarkdownBlock>,
    tail: Vec<MarkdownBlock>,
    generation: u64,
    revision: u64,
    last_parse_start: usize,
}

impl StreamingMarkdownState {
    pub(crate) fn update(&mut self, text: &str) {
        if text == self.previous_text {
            return;
        }
        if !text.starts_with(&self.previous_text) {
            self.previous_text.clear();
            self.tail_start = 0;
            self.frozen.clear();
            self.tail.clear();
            self.generation += 1;
        }

        let base = self.tail_start;
        self.last_parse_start = base;
        let source = &text[base..];
        let normalized = normalize_latex_math_delimiters(source);
        let mut parsed = markdown::to_mdast(&normalized, &math_parse_options());
        if let Ok(node) = &mut parsed {
            promote_display_math(node, source);
        }
        let Ok(Node::Root(root)) = parsed else {
            self.previous_text = text.to_owned();
            self.tail_start = 0;
            self.frozen.clear();
            self.tail.clear();
            self.generation += 1;
            self.revision += 1;
            return;
        };

        let first_unstable = root.children.len().saturating_sub(UNSTABLE_TAIL_BLOCKS);
        if first_unstable > 0
            && let Some(cut_end) = root.children[first_unstable - 1]
                .position()
                .map(|position| position.end.offset)
            && root.children[..first_unstable]
                .iter()
                .all(|node| node.position().is_some())
        {
            for node in &root.children[..first_unstable] {
                let Some(position) = node.position() else {
                    continue;
                };
                let start = base + position.start.offset;
                let end = base + position.end.offset;
                self.frozen.push(MarkdownBlock {
                    key: start,
                    source: text[start..end].to_owned().into(),
                    node: node.clone(),
                });
            }
            self.tail_start = base + cut_end;
        }

        self.tail = root.children[first_unstable..]
            .iter()
            .filter_map(|node| {
                let position = node.position()?;
                let start = base + position.start.offset;
                let end = base + position.end.offset;
                Some(MarkdownBlock {
                    key: start,
                    source: text[start..end].to_owned().into(),
                    node: node.clone(),
                })
            })
            .collect();

        self.previous_text = text.to_owned();
        self.revision += 1;
    }

    pub(crate) fn frozen(&self) -> &[MarkdownBlock] {
        &self.frozen
    }

    pub(crate) fn tail_blocks(&self) -> &[MarkdownBlock] {
        &self.tail
    }

    #[cfg(test)]
    fn tail<'a>(&self, text: &'a str) -> &'a str {
        &text[self.tail_start.min(text.len())..]
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    #[cfg(test)]
    fn parsed_bytes(&self) -> usize {
        self.previous_text
            .len()
            .saturating_sub(self.last_parse_start)
    }
}

fn math_parse_options() -> ParseOptions {
    let mut options = ParseOptions::gfm();
    options.constructs.math_text = true;
    options
}

fn normalize_latex_math_delimiters(source: &str) -> String {
    let mut protected = Vec::new();
    if let Ok(root) = markdown::to_mdast(source, &ParseOptions::gfm()) {
        collect_code_ranges(&root, &mut protected);
        protected.sort_by_key(|range| range.start);
    }

    let replacements = paired_latex_math_delimiters(source, &protected);

    let mut output = String::with_capacity(source.len());
    let mut index = 0;
    let mut protected = protected.into_iter().peekable();
    let mut replacements = replacements.into_iter().peekable();
    while index < source.len() {
        if let Some(range) = protected.next_if(|range| range.start == index) {
            output.push_str(&source[range.clone()]);
            index = range.end;
            continue;
        }

        if replacements.peek() == Some(&index) {
            output.push_str("$$");
            replacements.next();
            index += 2;
            continue;
        }

        let rest = &source[index..];
        let Some(character) = rest.chars().next() else {
            break;
        };
        output.push(character);
        index += character.len_utf8();
    }
    output
}

fn paired_latex_math_delimiters(source: &str, protected: &[std::ops::Range<usize>]) -> Vec<usize> {
    let mut replacements = Vec::new();
    let mut open: Option<(u8, usize)> = None;
    let mut index = 0;
    let mut protected = protected.iter().peekable();
    while index < source.len() {
        if let Some(range) = protected.next_if(|range| range.start == index) {
            index = range.end;
            continue;
        }

        let rest = &source[index..];
        let delimiter = rest.as_bytes().get(1).copied().filter(|delimiter| {
            rest.starts_with('\\')
                && matches!(delimiter, b'(' | b')' | b'[' | b']')
                && preceding_backslashes(source, index).is_multiple_of(2)
        });
        match (open, delimiter) {
            (None, Some(delimiter @ (b'(' | b'['))) => open = Some((delimiter, index)),
            (Some((b'(', start)), Some(b')')) | (Some((b'[', start)), Some(b']')) => {
                replacements.extend([start, index]);
                open = None;
            }
            _ => {}
        }
        if delimiter.is_some() {
            index += 2;
        } else {
            let Some(character) = rest.chars().next() else {
                break;
            };
            index += character.len_utf8();
        }
    }
    replacements.sort_unstable();
    replacements
}

fn collect_code_ranges(node: &Node, ranges: &mut Vec<std::ops::Range<usize>>) {
    if matches!(node, Node::Code(_) | Node::InlineCode(_)) {
        if let Some(position) = node.position() {
            ranges.push(position.start.offset..position.end.offset);
        }
        return;
    }
    if let Some(children) = node.children() {
        for child in children {
            collect_code_ranges(child, ranges);
        }
    }
}

fn preceding_backslashes(source: &str, index: usize) -> usize {
    source[..index]
        .bytes()
        .rev()
        .take_while(|byte| *byte == b'\\')
        .count()
}

fn promote_display_math(node: &mut Node, source: &str) {
    if let Some(children) = node.children_mut() {
        let mut promoted = Vec::with_capacity(children.len());
        for mut child in std::mem::take(children) {
            if let Node::Paragraph(paragraph) = child {
                promoted.extend(split_display_math_paragraph(paragraph, source));
            } else {
                promote_display_math(&mut child, source);
                promoted.push(child);
            }
        }
        *children = promoted;
    }
}

fn split_display_math_paragraph(paragraph: Paragraph, source: &str) -> Vec<Node> {
    let fallback_position = paragraph.position;
    let mut output = Vec::new();
    let mut inline = Vec::new();
    for child in paragraph.children {
        match child {
            Node::InlineMath(math)
                if math.position.as_ref().is_some_and(|position| {
                    is_display_delimiter(source, position) && starts_new_line(&inline)
                }) =>
            {
                push_paragraph(&mut output, &mut inline, fallback_position.as_ref());
                output.push(Node::Math(Math {
                    value: math.value.trim().to_owned(),
                    position: math.position,
                    meta: None,
                }));
            }
            child => inline.push(child),
        }
    }
    if output.is_empty() {
        return vec![Node::Paragraph(Paragraph {
            children: inline,
            position: fallback_position,
        })];
    }
    push_paragraph(&mut output, &mut inline, fallback_position.as_ref());
    output
}

fn starts_new_line(inline: &[Node]) -> bool {
    let Some(last) = inline.last() else {
        return true;
    };
    if inline
        .iter()
        .all(|node| matches!(node, Node::Text(text) if text.value.trim().is_empty()))
    {
        return true;
    }
    match last {
        Node::Break(_) => true,
        Node::Text(text) => text
            .value
            .rsplit_once('\n')
            .is_some_and(|(_, suffix)| suffix.trim().is_empty()),
        _ => false,
    }
}

fn is_display_delimiter(source: &str, position: &Position) -> bool {
    source
        .get(position.start.offset..position.end.offset)
        .is_some_and(|original| original.starts_with("$$") || original.starts_with("\\["))
}

fn push_paragraph(
    output: &mut Vec<Node>,
    children: &mut Vec<Node>,
    fallback_position: Option<&Position>,
) {
    if children
        .iter()
        .all(|child| matches!(child, Node::Text(text) if text.value.trim().is_empty()))
    {
        children.clear();
        return;
    }

    let position = children
        .first()
        .and_then(Node::position)
        .zip(children.last().and_then(Node::position))
        .map(|(first, last)| Position {
            start: first.start.clone(),
            end: last.end.clone(),
        })
        .or_else(|| fallback_position.cloned());
    output.push(Node::Paragraph(Paragraph {
        children: std::mem::take(children),
        position,
    }));
}

#[cfg(test)]
mod tests {
    use markdown::mdast::Node;

    use super::StreamingMarkdownState;

    fn contains_math(node: &Node, display: bool) -> bool {
        matches!(
            (node, display),
            (Node::Math(_), true) | (Node::InlineMath(_), false)
        ) || node
            .children()
            .is_some_and(|children| children.iter().any(|child| contains_math(child, display)))
    }

    #[test]
    fn parses_dollar_and_latex_math_delimiters() {
        for (source, display) in [
            ("before $x^2$ after", false),
            ("$$\n\\frac{1}{2}\n$$", true),
            ("before \\(x^2\\) after", false),
            ("\\[\\frac{1}{2}\\]", true),
        ] {
            let mut state = StreamingMarkdownState::default();
            state.update(source);
            assert!(
                state
                    .tail_blocks()
                    .iter()
                    .any(|block| contains_math(&block.node, display)),
                "did not parse {source:?} as math"
            );
        }
    }

    #[test]
    fn inline_formula_is_promoted_only_after_its_delimiter_closes() {
        let mut state = StreamingMarkdownState::default();
        state.update(r"before $\gamma");
        assert!(
            state
                .tail_blocks()
                .iter()
                .all(|block| !contains_math(&block.node, false))
        );

        state.update(r"before $\gamma$ after");
        assert!(
            state
                .tail_blocks()
                .iter()
                .any(|block| contains_math(&block.node, false))
        );
    }

    #[test]
    fn formulas_inside_markdown_styles_remain_formula_nodes() {
        for source in [
            r"**bold $\gamma$ text**",
            r"*italic $x^2$ text*",
            r"~~deleted $x_i$ text~~",
            r"[linked $\alpha$ text](https://example.com)",
        ] {
            let mut state = StreamingMarkdownState::default();
            state.update(source);
            assert!(
                state
                    .tail_blocks()
                    .iter()
                    .any(|block| contains_math(&block.node, false)),
                "did not retain nested formula for {source:?}"
            );
        }
    }

    #[test]
    fn display_math_does_not_consume_following_markdown() {
        for opening in ["", "\n"] {
            let source = format!(
                "intro\n\n$${opening}V_{{1:t}} = \\begin{{bmatrix}} v_1 \\\\ v_t \\end{{bmatrix}}$$\n\nafter\n\n$$q_{{t+1}} = x_{{t+1}} W_Q$$\n\nend"
            );
            let mut state = StreamingMarkdownState::default();
            state.update(&source);
            let blocks = state
                .frozen()
                .iter()
                .chain(state.tail_blocks())
                .map(|block| &block.node)
                .collect::<Vec<_>>();

            assert_eq!(blocks.len(), 5);
            assert!(matches!(&blocks[0], Node::Paragraph(_)));
            let Node::Math(first) = &blocks[1] else {
                panic!("expected first display formula, got {:?}", blocks[1]);
            };
            assert_eq!(
                first.value,
                r"V_{1:t} = \begin{bmatrix} v_1 \\ v_t \end{bmatrix}"
            );
            assert!(matches!(&blocks[2], Node::Paragraph(_)));
            let Node::Math(second) = &blocks[3] else {
                panic!("expected second display formula, got {:?}", blocks[3]);
            };
            assert_eq!(second.value, r"q_{t+1} = x_{t+1} W_Q");
            assert!(matches!(&blocks[4], Node::Paragraph(_)));
        }
    }

    #[test]
    fn text_followed_by_display_math_without_blank_line_is_split() {
        let source = "强度：\n$$\n\\boxed{I = \\frac{2s}{3}}\n$$";
        let mut state = StreamingMarkdownState::default();
        state.update(source);
        let blocks = state
            .frozen()
            .iter()
            .chain(state.tail_blocks())
            .map(|block| &block.node)
            .collect::<Vec<_>>();

        assert_eq!(blocks.len(), 2, "{blocks:#?}");
        assert!(matches!(&blocks[0], Node::Paragraph(_)));
        let Node::Math(math) = &blocks[1] else {
            panic!("expected display formula, got {:?}", blocks[1]);
        };
        assert_eq!(math.value, r"\boxed{I = \frac{2s}{3}}");
    }

    #[test]
    fn line_delimited_display_math_splits_surrounding_text() {
        let source = "before\n$$x^2$$\nafter";
        let mut state = StreamingMarkdownState::default();
        state.update(source);
        let blocks = state
            .frozen()
            .iter()
            .chain(state.tail_blocks())
            .map(|block| &block.node)
            .collect::<Vec<_>>();

        assert_eq!(blocks.len(), 3, "{blocks:#?}");
        assert!(matches!(&blocks[0], Node::Paragraph(_)));
        assert!(matches!(&blocks[1], Node::Math(_)));
        assert!(matches!(&blocks[2], Node::Paragraph(_)));
    }

    #[test]
    fn double_dollar_math_embedded_in_prose_stays_inline() {
        let source = "before $$x^2$$ after";
        let mut state = StreamingMarkdownState::default();
        state.update(source);
        let blocks = state
            .frozen()
            .iter()
            .chain(state.tail_blocks())
            .map(|block| &block.node)
            .collect::<Vec<_>>();

        assert_eq!(blocks.len(), 1);
        assert!(matches!(&blocks[0], Node::Paragraph(_)));
        assert!(contains_math(blocks[0], false));
        assert!(!contains_math(blocks[0], true));
    }

    #[test]
    fn latex_delimiters_inside_code_or_escaped_stay_literal() {
        let source = "`\\(inline\\)`\n\n```tex\n\\[block\\]\n```\n\nliteral \\\\(text\\\\)";
        assert_eq!(super::normalize_latex_math_delimiters(source), source);
    }

    #[test]
    fn incomplete_latex_math_stays_literal_while_streaming() {
        for source in ["before \\(x^2", "\\[\\frac{1}{2}"] {
            assert_eq!(super::normalize_latex_math_delimiters(source), source);
        }
    }

    #[test]
    fn freezes_stable_blocks_and_reparses_only_the_tail() {
        let mut state = StreamingMarkdownState::default();
        let first = "# 标题\n\nfirst\n\nsecond\n\nthird\n\nfourth";
        state.update(first);

        assert_eq!(
            state
                .frozen()
                .iter()
                .map(|block| block.source.as_ref())
                .collect::<Vec<_>>(),
            ["# 标题", "first", "second"],
        );
        let stable_keys = state
            .frozen()
            .iter()
            .map(|block| block.key)
            .collect::<Vec<_>>();
        assert_eq!(state.tail(first), "\n\nthird\n\nfourth");

        let second = format!("{first}\n\nfifth");
        state.update(&second);

        assert_eq!(
            &state
                .frozen()
                .iter()
                .map(|block| block.key)
                .collect::<Vec<_>>()[..stable_keys.len()],
            stable_keys,
        );
        assert_eq!(state.frozen().last().unwrap().source.as_ref(), "third");
        assert_eq!(state.tail(&second), "\n\nfourth\n\nfifth");
        assert!(state.parsed_bytes() < second.len());
    }

    #[test]
    fn identical_input_is_free_and_non_append_input_resets_the_generation() {
        let mut state = StreamingMarkdownState::default();
        state.update("one\n\ntwo\n\nthree");
        let revision = state.revision();
        state.update("one\n\ntwo\n\nthree");
        assert_eq!(state.revision(), revision);

        state.update("replacement");
        assert_eq!(state.generation(), 1);
        assert!(state.frozen().is_empty());
        assert_eq!(state.tail("replacement"), "replacement");
    }

    #[test]
    fn long_stream_keeps_per_frame_parse_work_bounded_to_the_frontier() {
        let mut state = StreamingMarkdownState::default();
        let mut text = (0..500)
            .map(|index| format!("paragraph {index}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        state.update(&text);

        for _ in 0..100 {
            text.push_str(" more");
            state.update(&text);
            assert!(state.parsed_bytes() < 1024);
        }
    }

    #[test]
    fn total_parse_work_stays_linear_for_a_many_block_stream() {
        let mut state = StreamingMarkdownState::default();
        let mut text = String::new();
        let mut parsed = 0;
        for index in 0..80 {
            text.push_str(&format!("paragraph {index} with some words.\n\n"));
            state.update(&text);
            parsed += state.parsed_bytes();
        }
        assert!(parsed < text.len() * 5);
    }

    #[test]
    fn frontier_sensitive_fence_and_list_stay_unfrozen_until_closed() {
        let mut fence = StreamingMarkdownState::default();
        let mut fenced = "p1.\n\np2.\n\np3.\n\n```ts\n".to_owned();
        fence.update(&fenced);
        let frozen_at_open = fence.frozen().len();
        for line in ["const a = 1\n", "\n", "- still code\n"] {
            fenced.push_str(line);
            fence.update(&fenced);
            assert_eq!(fence.frozen().len(), frozen_at_open);
        }
        fenced.push_str("```\n\nafter one.\n\nafter two.\n");
        fence.update(&fenced);
        assert!(fence.frozen().len() > frozen_at_open);
        assert!(
            fence
                .frozen()
                .iter()
                .any(|block| block.source.as_ref().contains("still code"))
        );

        let mut list = StreamingMarkdownState::default();
        let mut listed = "intro.\n\nsecond.\n\nthird.\n\n- item a\n- item b\n".to_owned();
        list.update(&listed);
        let frozen_before = list.frozen().len();
        listed.push_str("\n- item c\n");
        list.update(&listed);
        assert_eq!(list.frozen().len(), frozen_before);
        listed.push_str("\nafter.\n\nmore.\n\nend.\n");
        list.update(&listed);
        assert!(
            list.frozen()
                .iter()
                .any(|block| block.source.as_ref().contains("item c"))
        );
    }

    #[test]
    fn open_code_fences_remain_in_the_mutable_tail() {
        let mut state = StreamingMarkdownState::default();
        let source = "intro\n\n```rust\nfn main() {\n";
        state.update(source);
        assert!(state.tail_blocks().iter().any(|block| {
            matches!(&block.node, Node::Code(code) if code.lang.as_deref() == Some("rust") && code.value == "fn main() {")
        }));

        let closed = format!("{source}```\n");
        state.update(&closed);
        assert!(state.tail_blocks().iter().any(|block| {
            matches!(&block.node, Node::Code(code) if code.lang.as_deref() == Some("rust") && code.value == "fn main() {")
        }));
    }

    #[test]
    fn malformed_streaming_markdown_never_panics() {
        let mut state = StreamingMarkdownState::default();
        for source in [
            "\\(",
            "\\[\\frac{",
            "```rust\nfn main() {",
            "| incomplete | table\n| ---",
            "- [x",
            "中文 **unfinished",
        ] {
            state.update(source);
            assert!(!state.tail_blocks().is_empty());
        }
    }
}
