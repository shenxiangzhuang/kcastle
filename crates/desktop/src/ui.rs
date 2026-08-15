use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement, Render, Styled, Window,
    div, prelude::FluentBuilder, px,
};

use crate::app::DesktopApp;
use crate::ui_theme::palette;

impl Render for DesktopApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let empty = self.messages.is_empty();
        let colors = palette(cx);
        div()
            .flex()
            .size_full()
            .capture_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, window, cx| {
                this.handle_root_key(event, window, cx);
            }))
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, _, cx| {
                    if this.composer_menu.is_some() {
                        this.composer_menu = None;
                        cx.notify();
                    }
                    if this.show_sidebar_options || this.session_action_target.is_some() {
                        this.show_sidebar_options = false;
                        this.session_action_target = None;
                        cx.notify();
                    }
                }),
            )
            .bg(colors.canvas)
            .text_color(colors.text)
            .child(self.sidebar(window, cx))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .h_full()
                    .min_w(px(0.0))
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
