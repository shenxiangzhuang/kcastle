use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement, Render, Styled, Window,
    div, prelude::FluentBuilder, px,
};
use gpui_component::resizable::{h_resizable, resizable_panel};

use crate::app::DesktopApp;
use crate::application::conversation_view_model;
use crate::layout::{SidebarMode, sidebar_max_width};
use crate::platform::gpui::measured_container;
use crate::ui_theme::palette;

impl Render for DesktopApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let empty = conversation_view_model(&self.core).empty;
        let sidebar_mode = self.core.layout.sidebar;
        let colors = palette(cx);
        let measurement_owner = cx.entity().downgrade();
        let sidebar_max_width = sidebar_max_width(f32::from(window.viewport_size().width));
        let main = div()
            .relative()
            .flex()
            .flex_col()
            .flex_1()
            .h_full()
            .min_w(px(0.0))
            .child(measured_container(
                measurement_owner,
                |bounds, this: &mut DesktopApp, cx| this.update_main_measurement(bounds.width, cx),
                |this: &mut DesktopApp, window, cx| {
                    if this.core.follow_chat_tail {
                        this.restore_chat_tail_after_layout(window, cx);
                    }
                },
            ))
            .when(empty, |main| main.child(self.empty_conversation(cx)))
            .when(!empty, |main| {
                main.child(self.conversation_header(cx))
                    .child(self.conversation_body(window, cx))
                    .children(self.approval_card(cx))
                    .child(self.docked_composer(window, cx))
            });
        let content = match sidebar_mode {
            SidebarMode::Expanded => h_resizable("app-layout")
                .child(
                    resizable_panel()
                        .size(px(crate::ui_theme::metrics::SIDEBAR_WIDTH))
                        .size_range(
                            px(crate::ui_theme::metrics::SIDEBAR_WIDTH)..px(sidebar_max_width),
                        )
                        .flex_none()
                        .child(self.sidebar(window, cx)),
                )
                .child(resizable_panel().child(main))
                .into_any_element(),
            SidebarMode::Rail => main.into_any_element(),
        };
        div()
            .relative()
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
                    if this.core.sidebar.options_open {
                        this.dispatch(crate::domain::Action::CloseTransientOverlays, window, cx);
                    }
                }),
            )
            .bg(colors.canvas)
            .text_color(colors.text)
            .child(content)
            .when(sidebar_mode == SidebarMode::Rail, |root| {
                root.child(self.sidebar(window, cx))
            })
            .children(self.modal_view(window, cx))
    }
}
