use gpui::SharedString;
use markdown::{ParseOptions, mdast::Node};

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
        let parsed = markdown::to_mdast(&text[base..], &ParseOptions::gfm());
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
                let position = node.position().expect("positions checked above");
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

#[cfg(test)]
mod tests {
    use markdown::mdast::Node;

    use super::StreamingMarkdownState;

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
}
