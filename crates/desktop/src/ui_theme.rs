use std::time::Duration;

use gpui::{App, Hsla, rgb, rgba};
use gpui_component::ActiveTheme;

#[derive(Clone, Copy)]
pub(crate) struct UiPalette {
    pub(crate) canvas: Hsla,
    pub(crate) surface: Hsla,
    pub(crate) sidebar: Hsla,
    pub(crate) border: Hsla,
    pub(crate) text: Hsla,
    pub(crate) muted_text: Hsla,
    pub(crate) subtle: Hsla,
    pub(crate) hover: Hsla,
    pub(crate) selected: Hsla,
    pub(crate) primary: Hsla,
    pub(crate) overlay: Hsla,
    pub(crate) danger: Hsla,
    pub(crate) warning: Hsla,
    pub(crate) assistant: Hsla,
    pub(crate) user_bubble: Hsla,
    pub(crate) markdown_inline_code: Hsla,
    pub(crate) markdown_code_block: Hsla,
    pub(crate) markdown_code_banner: Hsla,
    pub(crate) markdown_link: Hsla,
    pub(crate) markdown_quote: Hsla,
}

#[derive(Clone, Copy)]
pub(crate) struct TrajectoryPalette {
    pub(crate) background: Hsla,
    pub(crate) code_background: Hsla,
    pub(crate) border_l1: Hsla,
    pub(crate) border_l2: Hsla,
    pub(crate) label_primary: Hsla,
    pub(crate) label_secondary: Hsla,
    pub(crate) label_tertiary: Hsla,
    pub(crate) label_caption: Hsla,
    pub(crate) hover: Hsla,
    pub(crate) active: Hsla,
    pub(crate) primary: Hsla,
    pub(crate) user_foreground: Hsla,
    pub(crate) user_background: Hsla,
    pub(crate) assistant_foreground: Hsla,
    pub(crate) assistant_background: Hsla,
    pub(crate) tool_foreground: Hsla,
    pub(crate) tool_background: Hsla,
    pub(crate) context_foreground: Hsla,
    pub(crate) context_background: Hsla,
    pub(crate) json_property: Hsla,
    pub(crate) json_string: Hsla,
    pub(crate) json_keyword: Hsla,
    pub(crate) json_punctuation: Hsla,
    pub(crate) error: Hsla,
}

pub(crate) mod metrics {
    pub(crate) const SIDEBAR_WIDTH: f32 = 280.0;
    pub(crate) const SIDEBAR_RAIL_WIDTH: f32 = 56.0;
    pub(crate) const CONTENT_WIDTH: f32 = 748.0;
    pub(crate) const COMPOSER_WIDTH: f32 = 780.0;
    pub(crate) const CHAT_SIDE_PADDING: f32 = 16.0;
    // Keep the terminal Markdown frontier and its action row scrollable above the docked composer.
    pub(crate) const CHAT_BOTTOM_CLEARANCE: f32 = 256.0;
    pub(crate) const COMPOSER_RADIUS: f32 = 22.0;
    pub(crate) const WORKSPACE_ROW_HEIGHT: f32 = 34.0;
    pub(crate) const SESSION_ROW_HEIGHT: f32 = 32.0;
    pub(crate) const SIDEBAR_ICON_SLOT: f32 = 16.0;
    pub(crate) const SESSION_ROW_INDENT: f32 = 28.0;
    pub(crate) const SESSION_TRAILING_SLOT: f32 = 32.0;
    pub(crate) const SIDEBAR_LABEL_UNITS: usize = 24;
    pub(crate) const TAB_HEIGHT: f32 = 42.0;
    pub(crate) const DETAILS_HEADER_HEIGHT: f32 = 54.0;
    pub(crate) const LEDGER_ROW_HEIGHT: f32 = 30.0;
    pub(crate) const BODY_LINE_HEIGHT: f32 = 24.0;
    pub(crate) const MESSAGE_LINE_HEIGHT: f32 = 28.0;
    pub(crate) const MARKDOWN_BLOCK_GAP: f32 = 16.0;
    pub(crate) const MARKDOWN_SECTION_GAP: f32 = 32.0;
    pub(crate) const RESPONSIVE_SIDEBAR_BREAKPOINT: f32 = 1024.0;
}

pub(crate) mod motion {
    use super::Duration;

    pub(crate) const FAST: Duration = Duration::from_millis(120);
    pub(crate) const STANDARD: Duration = Duration::from_millis(150);
}

pub(crate) fn palette(cx: &App) -> UiPalette {
    let theme = cx.theme();
    let mut palette = UiPalette {
        canvas: theme.background,
        surface: theme.popover,
        sidebar: theme.sidebar,
        border: theme.border,
        text: theme.foreground,
        muted_text: theme.muted_foreground,
        subtle: theme.muted,
        hover: theme.accent,
        selected: theme.list_active,
        primary: theme.primary,
        overlay: theme.overlay,
        danger: theme.danger,
        warning: theme.warning,
        assistant: theme.chart_4,
        user_bubble: rgb(0xedf3fe).into(),
        markdown_inline_code: rgb(0xebeef2).into(),
        markdown_code_block: rgb(0xf9fafb).into(),
        markdown_code_banner: rgb(0xf9fafb).into(),
        markdown_link: rgb(0x4176e6).into(),
        markdown_quote: rgb(0xadb2b8).into(),
    };
    if theme.is_dark() {
        palette.user_bubble = rgb(0x2c2c2e).into();
        palette.markdown_inline_code = rgb(0x2c2c2e).into();
        palette.markdown_code_block = rgb(0x1b1b1c).into();
        palette.markdown_code_banner = rgb(0x2c2c2e).into();
        palette.markdown_link = rgb(0x679efe).into();
        palette.markdown_quote = rgb(0x81858c).into();
    }
    palette
}

pub(crate) fn trajectory_palette(cx: &App) -> TrajectoryPalette {
    if cx.theme().is_dark() {
        TrajectoryPalette {
            background: rgb(0x232324).into(),
            code_background: rgb(0x1b1b1c).into(),
            border_l1: rgba(0xffffff0f).into(),
            border_l2: rgba(0xffffff1f).into(),
            label_primary: rgb(0xf9fafb).into(),
            label_secondary: rgb(0xcfd3d6).into(),
            label_tertiary: rgb(0xadb2b8).into(),
            label_caption: rgb(0x81858c).into(),
            hover: rgba(0xffffff14).into(),
            active: rgba(0xffffff24).into(),
            primary: rgb(0x679efe).into(),
            user_foreground: rgb(0x679efe).into(),
            user_background: rgb(0x34415b).into(),
            assistant_foreground: rgb(0x9474bc).into(),
            assistant_background: rgb(0x342f3b).into(),
            tool_foreground: rgb(0xdd8629).into(),
            tool_background: rgb(0x27241f).into(),
            context_foreground: rgb(0x59c984).into(),
            context_background: rgb(0x233c2c).into(),
            json_property: rgb(0x5db0d7).into(),
            json_string: rgb(0xf28b82).into(),
            json_keyword: rgb(0x99c8ff).into(),
            json_punctuation: rgb(0xe8eaed).into(),
            error: rgb(0xf25a5a).into(),
        }
    } else {
        TrajectoryPalette {
            background: rgb(0xffffff).into(),
            code_background: rgb(0xf9fafb).into(),
            border_l1: rgba(0x0000000a).into(),
            border_l2: rgba(0x0000001a).into(),
            label_primary: rgb(0x0f1115).into(),
            label_secondary: rgb(0x61666b).into(),
            label_tertiary: rgb(0x81858c).into(),
            label_caption: rgb(0xadb2b8).into(),
            hover: rgba(0x2631480f).into(),
            active: rgba(0x2631481a).into(),
            primary: rgb(0x4176e6).into(),
            user_foreground: rgb(0x4176e6).into(),
            user_background: rgb(0xe4edfd).into(),
            assistant_foreground: rgb(0x886bae).into(),
            assistant_background: rgb(0xede9f3).into(),
            tool_foreground: rgb(0xdd8629).into(),
            tool_background: rgb(0xfef5e7).into(),
            context_foreground: rgb(0x36a763).into(),
            context_background: rgb(0xe6faed).into(),
            json_property: rgb(0x881391).into(),
            json_string: rgb(0xc41a16).into(),
            json_keyword: rgb(0x1c00cf).into(),
            json_punctuation: rgb(0x202124).into(),
            error: rgb(0xec1313).into(),
        }
    }
}
