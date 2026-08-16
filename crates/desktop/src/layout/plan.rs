use crate::layout::{ContainerInput, resolve_container};

const SIDEBAR_EXPANDED_WIDTH: f32 = 280.0;
const MAIN_MIN_WIDTH: f32 = 720.0;
const LEDGER_MIN_WIDTH: f32 = 500.0;
const DETAILS_MIN_WIDTH: f32 = 380.0;
const PANE_GUTTER: f32 = 1.0;
const CONTENT_MAX_WIDTH_REM: f32 = 46.75;
const COMPOSER_MAX_WIDTH_REM: f32 = 48.75;
const CHAT_SIDE_PADDING_REM: f32 = 1.0;
const STATUS_BAR_HEIGHT_REM: f32 = 1.75;
const TAIL_COMFORT_GAP_REM: f32 = 1.0;
const COMPACT_HEIGHT: f32 = 680.0;
const SPACIOUS_HEIGHT: f32 = 920.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayoutInput {
    pub(crate) viewport_width: f32,
    pub(crate) viewport_height: f32,
    pub(crate) rem_size: f32,
    pub(crate) sidebar_requested: bool,
    pub(crate) trajectory_visible: bool,
    pub(crate) details_visible: bool,
    pub(crate) composer_height: f32,
    pub(crate) safe_area_bottom: f32,
    pub(crate) measured_main_width: f32,
}

impl Default for LayoutInput {
    fn default() -> Self {
        Self {
            viewport_width: 1180.0,
            viewport_height: 720.0,
            rem_size: 16.0,
            sidebar_requested: true,
            trajectory_visible: false,
            details_visible: false,
            composer_height: 100.0,
            safe_area_bottom: 0.0,
            measured_main_width: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SidebarMode {
    Rail,
    Expanded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryMode {
    Ledger,
    Split,
    Overlay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeightMode {
    Compact,
    Regular,
    Spacious,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayoutPlan {
    pub(crate) sidebar: SidebarMode,
    pub(crate) sidebar_width: f32,
    pub(crate) main_width: f32,
    pub(crate) trajectory: TrajectoryMode,
    pub(crate) height: HeightMode,
    pub(crate) show_status_bar: bool,
    pub(crate) content_max_width: f32,
    pub(crate) composer_max_width: f32,
    pub(crate) chat_side_padding: f32,
    pub(crate) tail_inset: f32,
}

pub(crate) fn resolve_layout(input: LayoutInput) -> LayoutPlan {
    let viewport_width = finite_non_negative(input.viewport_width);
    let viewport_height = finite_non_negative(input.viewport_height);
    let rem = if input.rem_size.is_finite() && input.rem_size > 0.0 {
        input.rem_size
    } else {
        16.0
    };
    let required_main_width = if input.trajectory_visible && input.details_visible {
        (LEDGER_MIN_WIDTH + DETAILS_MIN_WIDTH + PANE_GUTTER).max(MAIN_MIN_WIDTH)
    } else {
        MAIN_MIN_WIDTH
    };
    let sidebar = if input.sidebar_requested
        && viewport_width >= SIDEBAR_EXPANDED_WIDTH + required_main_width
    {
        SidebarMode::Expanded
    } else {
        SidebarMode::Rail
    };
    let sidebar_width = match sidebar {
        SidebarMode::Rail => 0.0,
        SidebarMode::Expanded => SIDEBAR_EXPANDED_WIDTH,
    }
    .min(viewport_width);
    let derived_main_width = (viewport_width - sidebar_width).max(0.0);
    let measured_main_width = finite_non_negative(input.measured_main_width);
    let main_width = if measured_main_width > 0.0 {
        measured_main_width.min(derived_main_width)
    } else {
        derived_main_width
    };
    let trajectory = if !input.trajectory_visible || !input.details_visible {
        TrajectoryMode::Ledger
    } else if main_width >= LEDGER_MIN_WIDTH + DETAILS_MIN_WIDTH + PANE_GUTTER {
        TrajectoryMode::Split
    } else {
        TrajectoryMode::Overlay
    };
    let reading = resolve_container(ContainerInput {
        available_width: main_width,
        rem_size: rem,
        max_width_rem: CONTENT_MAX_WIDTH_REM,
        preferred_padding_rem: CHAT_SIDE_PADDING_REM,
    });
    let composer = resolve_container(ContainerInput {
        available_width: main_width,
        rem_size: rem,
        max_width_rem: COMPOSER_MAX_WIDTH_REM,
        preferred_padding_rem: CHAT_SIDE_PADDING_REM,
    });
    let chat_side_padding = reading.inline_padding;
    let composer_height = finite_non_negative(input.composer_height);
    let safe_area_bottom = finite_non_negative(input.safe_area_bottom);
    let height = if viewport_height < COMPACT_HEIGHT {
        HeightMode::Compact
    } else if viewport_height >= SPACIOUS_HEIGHT {
        HeightMode::Spacious
    } else {
        HeightMode::Regular
    };
    let show_status_bar = height != HeightMode::Compact;
    let status_height = if show_status_bar {
        STATUS_BAR_HEIGHT_REM * rem
    } else {
        0.0
    };

    LayoutPlan {
        sidebar,
        sidebar_width,
        main_width,
        trajectory,
        height,
        show_status_bar,
        content_max_width: reading.content_width,
        composer_max_width: composer.content_width,
        chat_side_padding,
        tail_inset: composer_height + safe_area_bottom + status_height + TAIL_COMFORT_GAP_REM * rem,
    }
}

fn finite_non_negative(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn layout_modes_follow_available_main_width() {
        let narrow = resolve_layout(LayoutInput {
            viewport_width: 850.0,
            trajectory_visible: true,
            details_visible: true,
            ..LayoutInput::default()
        });
        assert_eq!(narrow.sidebar, SidebarMode::Rail);
        assert_eq!(narrow.trajectory, TrajectoryMode::Overlay);

        let wide = resolve_layout(LayoutInput {
            viewport_width: 1180.0,
            trajectory_visible: true,
            details_visible: true,
            ..LayoutInput::default()
        });
        assert_eq!(wide.sidebar, SidebarMode::Expanded);
        assert_eq!(wide.trajectory, TrajectoryMode::Split);
    }

    #[test]
    fn collapsed_sidebar_is_an_overlay_and_does_not_reserve_main_width() {
        let plan = resolve_layout(LayoutInput {
            viewport_width: 1180.0,
            sidebar_requested: false,
            ..LayoutInput::default()
        });
        assert_eq!(plan.sidebar, SidebarMode::Rail);
        assert_eq!(plan.sidebar_width, 0.0);
        assert_eq!(plan.main_width, 1180.0);
    }

    #[test]
    fn tail_inset_is_derived_from_measured_composer_height() {
        let one_line = resolve_layout(LayoutInput {
            composer_height: 88.0,
            ..LayoutInput::default()
        });
        let fourteen_lines = resolve_layout(LayoutInput {
            composer_height: 360.0,
            ..LayoutInput::default()
        });
        assert_eq!(fourteen_lines.tail_inset - one_line.tail_inset, 272.0);
    }

    #[test]
    fn short_windows_drop_nonessential_status_chrome() {
        let compact = resolve_layout(LayoutInput {
            viewport_height: 620.0,
            ..LayoutInput::default()
        });
        let regular = resolve_layout(LayoutInput {
            viewport_height: 720.0,
            ..LayoutInput::default()
        });
        assert_eq!(compact.height, HeightMode::Compact);
        assert!(!compact.show_status_bar);
        assert!(regular.show_status_bar);
        assert!(regular.tail_inset > compact.tail_inset);
    }

    #[test]
    fn invalid_platform_measurements_never_escape_into_the_plan() {
        let plan = resolve_layout(LayoutInput {
            viewport_width: f32::NAN,
            composer_height: f32::INFINITY,
            safe_area_bottom: f32::NEG_INFINITY,
            ..LayoutInput::default()
        });
        for value in [
            plan.sidebar_width,
            plan.main_width,
            plan.content_max_width,
            plan.composer_max_width,
            plan.tail_inset,
        ] {
            assert!(value.is_finite());
            assert!(value >= 0.0);
        }
    }

    proptest! {
        #[test]
        fn arbitrary_finite_measurements_produce_valid_geometry(
            width in 0.0f32..10_000.0,
            height in 0.0f32..10_000.0,
            rem_size in 8.0f32..40.0,
            composer_height in 1.0f32..2_000.0,
            safe_area in 0.0f32..200.0,
            sidebar_requested in any::<bool>(),
            trajectory_visible in any::<bool>(),
            details_visible in any::<bool>(),
            measured_main_width in 0.0f32..10_000.0,
        ) {
            let plan = resolve_layout(LayoutInput {
                viewport_width: width,
                viewport_height: height,
                rem_size,
                sidebar_requested,
                trajectory_visible,
                details_visible,
                composer_height,
                safe_area_bottom: safe_area,
                measured_main_width,
            });
            for value in [
                plan.sidebar_width,
                plan.main_width,
                plan.content_max_width,
                plan.composer_max_width,
                plan.chat_side_padding,
                plan.tail_inset,
            ] {
                prop_assert!(value.is_finite());
                prop_assert!(value >= 0.0);
            }
            prop_assert!(plan.sidebar_width + plan.main_width <= width + f32::EPSILON);
            prop_assert!(plan.content_max_width <= plan.main_width + f32::EPSILON);
            prop_assert!(plan.composer_max_width <= plan.main_width + f32::EPSILON);
        }

        #[test]
        fn wider_windows_never_regress_to_a_more_constrained_mode(
            first in 0.0f32..4_000.0,
            extra in 0.0f32..4_000.0,
        ) {
            let input = LayoutInput {
                viewport_width: first,
                trajectory_visible: true,
                details_visible: true,
                ..LayoutInput::default()
            };
            let narrow = resolve_layout(input);
            let wide = resolve_layout(LayoutInput {
                viewport_width: first + extra,
                ..input
            });
            let sidebar_rank = |mode| match mode {
                SidebarMode::Rail => 0,
                SidebarMode::Expanded => 1,
            };
            let trajectory_rank = |mode| match mode {
                TrajectoryMode::Overlay => 0,
                TrajectoryMode::Ledger => 1,
                TrajectoryMode::Split => 2,
            };
            prop_assert!(sidebar_rank(wide.sidebar) >= sidebar_rank(narrow.sidebar));
            prop_assert!(trajectory_rank(wide.trajectory) >= trajectory_rank(narrow.trajectory));
        }
    }
}
