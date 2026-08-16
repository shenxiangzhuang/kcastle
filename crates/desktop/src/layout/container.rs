#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ContainerInput {
    pub(crate) available_width: f32,
    pub(crate) rem_size: f32,
    pub(crate) max_width_rem: f32,
    pub(crate) preferred_padding_rem: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ContainerPlan {
    pub(crate) content_width: f32,
    pub(crate) inline_padding: f32,
}

pub(crate) fn resolve_container(input: ContainerInput) -> ContainerPlan {
    let available = finite_non_negative(input.available_width);
    let rem = if input.rem_size.is_finite() && input.rem_size > 0.0 {
        input.rem_size
    } else {
        16.0
    };
    let max_width = finite_non_negative(input.max_width_rem) * rem;
    let preferred_padding = finite_non_negative(input.preferred_padding_rem) * rem;
    let inline_padding = preferred_padding.min(available / 2.0);
    ContainerPlan {
        content_width: (available - inline_padding * 2.0).max(0.0).min(max_width),
        inline_padding,
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

    proptest! {
        #[test]
        fn container_content_never_exceeds_its_measured_parent(
            width in 0.0f32..10_000.0,
            rem in 8.0f32..40.0,
            max_width_rem in 1.0f32..100.0,
            padding_rem in 0.0f32..10.0,
        ) {
            let plan = resolve_container(ContainerInput {
                available_width: width,
                rem_size: rem,
                max_width_rem,
                preferred_padding_rem: padding_rem,
            });
            prop_assert!(plan.content_width >= 0.0);
            prop_assert!(plan.inline_padding >= 0.0);
            prop_assert!(plan.content_width + 2.0 * plan.inline_padding <= width + 0.01);
        }
    }
}
