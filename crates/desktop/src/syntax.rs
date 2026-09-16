//! Source-free highlighters shared by background preparation and settled rendering.
use std::{
    ops::Range,
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use gpui_kit::{
    HighlightStyle,
    component::{
        highlighter::{HighlightTheme, LanguageConfig, LanguageRegistry, SyntaxHighlighter},
        input::Rope,
    },
};

const IDLE_LIMIT: usize = 4;
const MAX_REUSABLE_SOURCE_BYTES: usize = 256 * 1024;

#[derive(Default)]
struct Highlighters {
    idle: Mutex<Vec<(LanguageConfig, SyntaxHighlighter)>>,
    #[cfg(test)]
    builds: std::sync::atomic::AtomicUsize,
}

static HIGHLIGHTERS: Highlighters = Highlighters {
    idle: Mutex::new(Vec::new()),
    #[cfg(test)]
    builds: std::sync::atomic::AtomicUsize::new(0),
};

pub(crate) fn highlight_code(
    language: &str,
    source: &str,
    theme: &HighlightTheme,
    cancelled: &AtomicBool,
) -> Option<Vec<(Range<usize>, HighlightStyle)>> {
    HIGHLIGHTERS.highlight(language, source, theme, cancelled)
}

impl Highlighters {
    fn highlight(
        &self,
        language: &str,
        source: &str,
        theme: &HighlightTheme,
        cancelled: &AtomicBool,
    ) -> Option<Vec<(Range<usize>, HighlightStyle)>> {
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }
        let Some(config) = LanguageRegistry::singleton()
            .language(language)
            .filter(LanguageConfig::has_grammar)
        else {
            return Some(Vec::new());
        };
        let mut highlighter = {
            let mut idle = self
                .idle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            idle.iter()
                .position(|(previous, _)| *previous == config)
                .map(|index| idle.remove(index).1)
        }
        .unwrap_or_else(|| {
            #[cfg(test)]
            self.builds.fetch_add(1, Ordering::Relaxed);
            SyntaxHighlighter::new(language)
        });
        if cancelled.load(Ordering::Relaxed) {
            return None;
        }
        let completed = highlighter.update(None, &Rope::from(source), None);
        debug_assert!(completed, "an unbounded syntax parse always completes");
        let styles = (!cancelled.load(Ordering::Relaxed))
            .then(|| highlighter.styles(&(0..source.len()), theme));
        if source.len() <= MAX_REUSABLE_SOURCE_BYTES {
            // A full deletion is required: update(None, empty) describes an insertion,
            // not replacement of an existing document. Drop source and injection trees.
            let cleared = highlighter.update(
                Some(tree_sitter::InputEdit {
                    start_byte: 0,
                    old_end_byte: source.len(),
                    new_end_byte: 0,
                    start_position: tree_sitter::Point::new(0, 0),
                    old_end_position: tree_sitter::Point::new(
                        source.bytes().filter(|b| *b == b'\n').count(),
                        source.rsplit('\n').next().map_or(0, str::len),
                    ),
                    new_end_position: tree_sitter::Point::new(0, 0),
                }),
                &Rope::new(),
                None,
            );
            debug_assert!(cleared);
            let mut idle = self
                .idle
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !idle.iter().any(|(previous, _)| *previous == config) {
                // ponytail: four recent language objects; no per-session pool or source cache.
                if idle.len() == IDLE_LIMIT {
                    idle.remove(0);
                }
                idle.push((config, highlighter));
            }
        }
        styles.filter(|_| !cancelled.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_empty_highlighters_without_cross_document_or_theme_styles() {
        crate::register_syntax_languages();
        let pool = Highlighters::default();
        let cancelled = AtomicBool::new(false);
        for (language, source) in [
            ("rust", "/* unfinished comment\n多字节"),
            ("rust", "fn next() { let text = \"你好\"; }\n"),
            ("rust", ""),
            ("rust", "let n = 42;"),
            ("html", "<script>const value = 42;</script>"),
            ("html", "<p>different document</p>"),
            ("hs", "main = putStrLn \"hello\""),
            ("haskell", "value :: Int\nvalue = 42"),
        ] {
            for dark in [false, true] {
                let theme = crate::ui_theme::markdown_highlight_theme(dark);
                let mut fresh = SyntaxHighlighter::new(language);
                assert!(fresh.update(None, &Rope::from(source), None));
                assert_eq!(
                    pool.highlight(language, source, theme, &cancelled).unwrap(),
                    fresh.styles(&(0..source.len()), theme.as_ref())
                );
                let idle = pool.idle.lock().unwrap();
                assert!(!idle.is_empty(), "rules should survive each document");
                assert!(idle.len() <= IDLE_LIMIT);
                for (_, highlighter) in idle.iter() {
                    assert!(
                        highlighter.is_empty(),
                        "idle objects must not retain source"
                    );
                    assert!(
                        highlighter
                            .tree()
                            .is_none_or(|tree| tree.root_node().end_byte() == 0)
                    );
                }
            }
        }
        assert_eq!(
            pool.idle.lock().unwrap().len(),
            3,
            "aliases share one entry"
        );
        assert_eq!(
            pool.builds.load(Ordering::Relaxed),
            3,
            "compile once per language, not per document or theme"
        );
    }

    #[test]
    fn bounds_idle_languages_and_drops_oversized_or_cancelled_input() {
        let pool = Highlighters::default();
        let theme = crate::ui_theme::markdown_highlight_theme(false);
        for language in ["rust", "python", "javascript", "json", "html", "css"] {
            pool.highlight(language, "42", theme, &AtomicBool::new(false))
                .unwrap();
        }
        assert_eq!(pool.idle.lock().unwrap().len(), IDLE_LIMIT);
        let builds = pool.builds.load(Ordering::Relaxed);
        assert!(
            pool.highlight("rust", "let value = 42;", theme, &AtomicBool::new(true))
                .is_none()
        );
        assert!(
            pool.highlight(
                "unknown-profile-language",
                "42",
                theme,
                &AtomicBool::new(false)
            )
            .unwrap()
            .is_empty()
        );
        assert_eq!(pool.builds.load(Ordering::Relaxed), builds);
        let source = " ".repeat(MAX_REUSABLE_SOURCE_BYTES + 1);
        pool.highlight("rust", &source, theme, &AtomicBool::new(false))
            .unwrap();
        assert!(
            pool.idle
                .lock()
                .unwrap()
                .iter()
                .all(|(config, _)| config.name.as_ref() != "rust")
        );
    }
}
