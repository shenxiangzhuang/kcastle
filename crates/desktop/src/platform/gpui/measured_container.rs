use gpui::{Context, IntoElement, Styled, WeakEntity, Window, canvas};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MeasuredBounds {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) width: f32,
    pub(crate) height: f32,
}

pub(crate) fn measured_container<T: 'static>(
    owner: WeakEntity<T>,
    on_measure: impl 'static + Fn(MeasuredBounds, &mut T, &mut Context<T>) -> bool,
    after_layout: impl 'static + Fn(&mut T, &mut Window, &mut Context<T>),
) -> impl IntoElement {
    canvas(
        {
            let owner = owner.clone();
            move |bounds, window, cx| {
                let changed = owner
                    .update(cx, |state, cx| {
                        on_measure(
                            MeasuredBounds {
                                x: f32::from(bounds.origin.x),
                                y: f32::from(bounds.origin.y),
                                width: f32::from(bounds.size.width),
                                height: f32::from(bounds.size.height),
                            },
                            state,
                            cx,
                        )
                    })
                    .unwrap_or(false);
                if changed {
                    let owner = owner.clone();
                    window.on_next_frame(move |window, cx| {
                        let _ = owner.update(cx, |state, cx| after_layout(state, window, cx));
                    });
                    window.refresh();
                }
            }
        },
        |_, _, _, _| {},
    )
    .absolute()
    .size_full()
}

#[cfg(test)]
mod tests {
    use gpui::{
        AppContext, Context, IntoElement, ParentElement, Render, Styled, TestAppContext, Window,
        div, px, size,
    };

    use super::*;

    #[derive(Default)]
    struct MeasurementHarness {
        width: f32,
    }

    impl Render for MeasurementHarness {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let owner = cx.entity().downgrade();
            div().relative().size_full().child(measured_container(
                owner,
                |bounds, harness: &mut MeasurementHarness, _| {
                    if (harness.width - bounds.width).abs() < 0.5 {
                        return false;
                    }
                    harness.width = bounds.width;
                    true
                },
                |_: &mut MeasurementHarness, _, _| {},
            ))
        }
    }

    #[gpui::test]
    fn measurement_updates_during_prepaint_and_survives_resize(cx: &mut TestAppContext) {
        let (view, cx) = cx.add_window_view(|_, _| MeasurementHarness::default());
        cx.simulate_resize(size(px(320.0), px(400.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();
        let width = cx.read_entity(&view, |harness, _| harness.width);
        assert!((width - 320.0).abs() < 1.0);

        cx.simulate_resize(size(px(640.0), px(400.0)));
        cx.refresh().unwrap();
        cx.run_until_parked();
        let width = cx.read_entity(&view, |harness, _| harness.width);
        assert!((width - 640.0).abs() < 1.0);
    }
}
