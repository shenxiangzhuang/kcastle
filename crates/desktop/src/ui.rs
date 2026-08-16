use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement, Render, Styled, Window,
    div, prelude::FluentBuilder, px,
};

use crate::app::DesktopApp;
use crate::application::conversation_view_model;
use crate::platform::gpui::measured_container;
use crate::ui_theme::palette;

impl Render for DesktopApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let empty = conversation_view_model(&self.core).empty;
        let colors = palette(cx);
        let measurement_owner = cx.entity().downgrade();
        div()
            .flex()
            .size_full()
            .capture_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                this.handle_root_key(event, window, cx);
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, cx| {
                    if this.core.composer.menu.is_some() {
                        this.dispatch(crate::domain::Action::SetComposerMenu(None), window, cx);
                    }
                    if this.core.sidebar.options_open
                        || this.core.sidebar.session_action_target.is_some()
                    {
                        this.dispatch(crate::domain::Action::CloseTransientOverlays, window, cx);
                    }
                }),
            )
            .bg(colors.canvas)
            .text_color(colors.text)
            .child(self.sidebar(window, cx))
            .child(
                div()
                    .relative()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .h_full()
                    .min_w(px(0.0))
                    .child(measured_container(
                        measurement_owner,
                        |bounds, this: &mut DesktopApp, cx| {
                            this.update_main_measurement(bounds.width, cx)
                        },
                        |this: &mut DesktopApp, cx| {
                            if this.core.follow_chat_tail {
                                this.restore_chat_tail_after_layout(cx);
                            }
                        },
                    ))
                    .when(empty, |main| main.child(self.empty_conversation(cx)))
                    .when(!empty, |main| {
                        main.child(self.conversation_header(cx))
                            .child(self.conversation_body(window, cx))
                            .children(self.approval_card(cx))
                            .child(self.docked_composer(cx))
                    }),
            )
            .children(self.modal_view(window, cx))
    }
}
