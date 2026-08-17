use std::path::PathBuf;

use crate::app::{DesktopApp, SidebarSessionStatus, same_path, session_age};
use gpui::{
    Context, InteractiveElement, IntoElement, MouseButton, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, WindowControlArea, deferred, div,
    prelude::FluentBuilder, px, relative,
};
use gpui_component::button::{Button, ButtonVariants};
use gpui_component::input::Input;
use gpui_component::scroll::ScrollableElement;
use gpui_component::spinner::Spinner;
use gpui_component::tooltip::Tooltip;
use gpui_component::{Icon, IconName, Sizable};

use crate::assets::DesktopIconName;
use crate::domain::Action;
use crate::layout::SidebarMode;
use crate::ui_theme::{UiPalette, metrics, palette};

impl DesktopApp {
    pub(crate) fn sidebar(&self, window: &Window, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = palette(cx);
        let toggle_leading = if window.is_fullscreen() {
            metrics::SIDEBAR_TOGGLE_FULLSCREEN_LEADING
        } else {
            metrics::SIDEBAR_TOGGLE_WINDOWED_LEADING
        };
        if self.core.layout.sidebar == SidebarMode::Rail {
            return self.sidebar_rail(toggle_leading, cx).into_any_element();
        }
        let panel = div()
            .flex()
            .flex_col()
            .relative()
            .w(px(metrics::SIDEBAR_WIDTH))
            .h_full()
            .flex_none()
            .bg(colors.sidebar)
            .border_r_1()
            .border_color(colors.border)
            .child(self.sidebar_titlebar(toggle_leading, cx))
            .child(self.sidebar_primary_navigation(cx))
            .child(self.workspace_header(cx))
            .child(self.workspace_tree(cx))
            .children(
                self.core
                    .sidebar
                    .options_open
                    .then(|| self.sidebar_options(cx)),
            )
            .child(
                div().flex_none().p_3().child(
                    div()
                        .flex()
                        .items_center()
                        .gap_1()
                        .child(
                            Button::new("settings")
                                .icon(IconName::Settings)
                                .label("Settings")
                                .ghost()
                                .flex_1()
                                .justify_start()
                                .px_2()
                                .on_click(cx.listener(|this, _, window, cx| {
                                    this.open_settings_dialog(window, cx)
                                })),
                        )
                        .children(self.available_update.as_ref().map(|update| {
                            Button::new("update")
                                .icon(IconName::ArrowUp)
                                .label("Update")
                                .ghost()
                                .flex_none()
                                .px_2()
                                .tooltip(format!("Update to v{}", update.version))
                                .on_click(
                                    cx.listener(|this, _, _, cx| this.open_available_update(cx)),
                                )
                        })),
                ),
            );
        panel.into_any_element()
    }

    fn sidebar_rail(&self, toggle_leading: f32, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = palette(cx);
        let panel = div()
            .absolute()
            .top_0()
            .left_0()
            .flex()
            .items_center()
            .w(px(metrics::COLLAPSED_TITLEBAR_CONTROLS_WIDTH))
            .h(px(metrics::TITLEBAR_HEIGHT))
            .pl(px(toggle_leading))
            .child(
                Button::new("open-sidebar")
                    .icon(IconName::PanelLeftOpen)
                    .ghost()
                    .compact()
                    .tooltip("Toggle sidebar (⌘B)")
                    .on_click(cx.listener(|this, _, window, cx| this.toggle_sidebar(window, cx))),
            )
            .child(
                Button::new("collapsed-new-chat")
                    .icon(DesktopIconName::SquarePen)
                    .ghost()
                    .compact()
                    .tooltip("New session")
                    .on_click(cx.listener(|this, _, window, cx| this.new_chat(window, cx))),
            )
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .window_control_area(WindowControlArea::Drag),
            )
            .child(div().h(px(24.0)).border_l_1().border_color(colors.border));
        panel.into_any_element()
    }

    fn sidebar_titlebar(&self, toggle_leading: f32, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .h(px(metrics::TITLEBAR_HEIGHT))
            .pl(px(toggle_leading))
            .pr_3()
            .child(
                Button::new("hide-sidebar")
                    .icon(IconName::PanelLeftClose)
                    .ghost()
                    .compact()
                    .tooltip("Toggle sidebar (⌘B)")
                    .on_click(cx.listener(|this, _, window, cx| this.toggle_sidebar(window, cx))),
            )
            .child(
                div()
                    .flex_1()
                    .h_full()
                    .window_control_area(WindowControlArea::Drag),
            )
    }

    fn sidebar_primary_navigation(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .px_3()
            .pb_2()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .h(px(40.0))
                    .px_2()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .child("kcastle")
                    .child(
                        Button::new("search-sessions")
                            .icon(if self.core.sidebar.search_sessions {
                                IconName::Close
                            } else {
                                IconName::Search
                            })
                            .ghost()
                            .compact()
                            .tooltip(if self.core.sidebar.search_sessions {
                                "Close search"
                            } else {
                                "Search sessions"
                            })
                            .on_click(cx.listener(|this, _, window, cx| {
                                this.toggle_session_search(window, cx)
                            })),
                    ),
            )
            .child(
                Button::new("new-chat")
                    .icon(DesktopIconName::SquarePen)
                    .label("New Session")
                    .ghost()
                    .w_full()
                    .h(px(40.0))
                    .justify_start()
                    .px_2()
                    .on_click(cx.listener(|this, _, window, cx| this.new_chat(window, cx))),
            )
    }

    fn workspace_header(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        div().flex().relative().px_3().child(
            div()
                .flex()
                .items_center()
                .justify_between()
                .h(px(44.0))
                .w_full()
                .px_1()
                .text_sm()
                .text_color(colors.muted_text)
                .when(!self.core.sidebar.search_sessions, |row| {
                    row.child("Workspaces")
                })
                .when(!self.core.sidebar.search_sessions, |row| {
                    row.child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .child(
                                Button::new("sort-sessions")
                                    .icon(IconName::Settings2)
                                    .ghost()
                                    .compact()
                                    .tooltip("View options")
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.toggle_sidebar_options(cx)
                                    })),
                            )
                            .child(
                                Button::new("add-project")
                                    .icon(IconName::FolderOpen)
                                    .ghost()
                                    .compact()
                                    .tooltip("Add workspace")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.add_project(window, cx)
                                    })),
                            ),
                    )
                })
                .when(self.core.sidebar.search_sessions, |row| {
                    row.child(
                        div()
                            .flex()
                            .items_center()
                            .w_full()
                            .gap_1()
                            .child(
                                div().flex_1().min_w(px(0.0)).child(
                                    Input::new(&self.session_search).small().cleanable(true),
                                ),
                            )
                            .child(
                                Button::new("close-session-search")
                                    .icon(IconName::Close)
                                    .ghost()
                                    .compact()
                                    .tooltip("Close search")
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        this.toggle_session_search(window, cx)
                                    })),
                            ),
                    )
                }),
        )
    }

    fn sidebar_options(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let colors = palette(cx);
        div()
            .absolute()
            .top(px(176.0))
            .right(px(36.0))
            .w(px(210.0))
            .p_2()
            .rounded_xl()
            .border_1()
            .border_color(colors.border)
            .bg(colors.surface)
            .shadow_lg()
            .occlude()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(
                div()
                    .px_2()
                    .py_1()
                    .text_xs()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(colors.muted_text)
                    .child("Group"),
            )
            .child(sidebar_option(
                "group-by-workspace",
                "By workspace",
                self.core.sidebar.group_by_workspace,
                colors,
                cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::SetSidebarGrouping(true), window, cx);
                }),
            ))
            .child(sidebar_option(
                "group-all-sessions",
                "All sessions",
                !self.core.sidebar.group_by_workspace,
                colors,
                cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::SetSidebarGrouping(false), window, cx);
                }),
            ))
            .child(
                div()
                    .mt_1()
                    .px_2()
                    .py_1()
                    .border_t_1()
                    .border_color(colors.border)
                    .text_xs()
                    .font_weight(gpui::FontWeight::SEMIBOLD)
                    .text_color(colors.muted_text)
                    .child("Order"),
            )
            .child(sidebar_option(
                "order-newest",
                "Last updated",
                self.core.sidebar.sort_by_recent,
                colors,
                cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::SetSidebarSort(true), window, cx);
                }),
            ))
            .child(sidebar_option(
                "order-oldest",
                "Oldest first",
                !self.core.sidebar.sort_by_recent,
                colors,
                cx.listener(|this, _, window, cx| {
                    this.dispatch(Action::SetSidebarSort(false), window, cx);
                }),
            ))
    }

    fn workspace_tree(&self, cx: &mut Context<Self>) -> gpui::AnyElement {
        let colors = palette(cx);
        let query = self.session_search.read(cx).value().trim().to_lowercase();
        if !self.core.sidebar.group_by_workspace {
            return self.flat_session_tree(&query, colors, cx);
        }
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .px_3()
            .overflow_y_scrollbar()
            .children(self.project_store.projects().iter().enumerate().map(|(index, project)| {
                let active = index == self.core.workspace.active_project;
                let expanded = self
                    .core
                    .workspace
                    .expanded_projects
                    .contains(&project.path)
                    || !query.is_empty();
                let mut sessions = if active {
                    self.core.session.sessions.clone()
                } else if expanded {
                    self.project_sessions
                        .get(&project.sessions_dir)
                        .cloned()
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                sessions.sort_by_key(|session| self.session_modified_at(session));
                if self.core.sidebar.sort_by_recent {
                    sessions.reverse();
                }
                let project_name =
                    sidebar_label(&project.name, metrics::SIDEBAR_LABEL_UNITS);
                let issue_count = self
                    .project_session_issues
                    .get(&project.sessions_dir)
                    .map_or(0, Vec::len);
                let project_missing = project.missing;
                let project_group = SharedString::from(format!("workspace-{index}"));
                div()
                    .flex()
                    .flex_col()
                    .child(
                        div()
                            .id(("workspace", index))
                            .flex()
                            .items_center()
                            .justify_between()
                            .h(px(metrics::WORKSPACE_ROW_HEIGHT))
                            .px_2()
                            .rounded_lg()
                            .cursor_pointer()
                            .tab_index(0)
                            .hover(move |element| element.bg(colors.hover))
                            .group(project_group.clone())
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.toggle_project(index, window, cx)
                            }))
                            .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                                if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                    this.toggle_project(index, window, cx);
                                }
                            }))
                            .child(
                                div()
                                    .flex()
                                    .flex_1()
                                    .items_center()
                                    .min_w(px(0.0))
                                    .gap_1()
                                    .child(
                                        div()
                                            .relative()
                                            .flex_none()
                                            .size(px(metrics::SIDEBAR_ICON_SLOT))
                                            .child(
                                                div()
                                                    .absolute()
                                                    .top_0()
                                                    .left_0()
                                                    .group_hover(project_group.clone(), |icon| icon.invisible())
                                                    .child(
                                                        Icon::new(IconName::Folder)
                                                            .size_4()
                                                            .text_color(if active { colors.primary } else { colors.muted_text }),
                                                    ),
                                            )
                                            .child(
                                                div()
                                                    .absolute()
                                                    .top_0()
                                                    .left_0()
                                                    .invisible()
                                                    .group_hover(project_group.clone(), |icon| icon.visible())
                                                    .child(
                                                        Icon::new(if expanded { IconName::ChevronDown } else { IconName::ChevronRight })
                                                            .size_4()
                                                            .text_color(if active { colors.primary } else { colors.muted_text }),
                                                    ),
                                            ),
                                    )
                                    .child(
                                        div()
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .truncate()
                                            .text_sm()
                                            .child(project_name),
                                    )
                                    .children(project_missing.then(|| {
                                        div().text_xs().text_color(colors.danger).child("Missing")
                                    }))
                                    .children((issue_count > 0).then(|| {
                                        div()
                                            .text_xs()
                                            .text_color(colors.danger)
                                            .child(format!("{issue_count} unreadable"))
                                    })),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .flex_none()
                                    .invisible()
                                    .group_hover(project_group, |element| element.visible())
                                    .children(project_missing.then(|| {
                                        Button::new(("relocate-workspace", index))
                                            .icon(IconName::FolderOpen)
                                            .ghost()
                                            .compact()
                                            .tooltip("Relocate workspace")
                                            .on_click(cx.listener(move |this, _, window, cx| {
                                                cx.stop_propagation();
                                                this.relocate_project(index, window, cx)
                                            }))
                                    }))
                                    .children((!project.is_default() && !self.project_has_active_sessions(index, cx)).then(|| {
                                        Button::new(("remove-workspace", index))
                                            .icon(IconName::Ellipsis)
                                            .ghost()
                                            .compact()
                                            .tooltip("Remove workspace")
                                            .on_click(cx.listener(move |this, _, window, cx| {
                                                cx.stop_propagation();
                                                this.open_remove_project_dialog(index, window, cx)
                                            }))
                                    }))
                                    .child(
                                        Button::new(("new-workspace-session", index))
                                            .icon(DesktopIconName::SquarePen)
                                            .ghost()
                                            .compact()
                                            .tooltip("New session in workspace")
                                            .on_click(cx.listener(move |this, _, window, cx| {
                                                cx.stop_propagation();
                                                this.new_chat_in_project(index, window, cx)
                                            })),
                                    ),
                            ),
                    )
                    .children(expanded.then(|| {
                        div()
                            .flex()
                            .flex_col()
                            .children((issue_count > 0).then(|| {
                                div()
                                    .px_7()
                                    .py_1()
                                    .text_xs()
                                    .text_color(colors.danger)
                                    .child("Some session files could not be read; valid sessions remain available.")
                            }))
                            .children(sessions.into_iter().enumerate().filter_map(|(session_index, session)| {
                                let path = session.path.clone();
                                let keyboard_path = path.clone();
                                let selected =
                                    active && same_path(&path, &self.core.session.current);
                                let title = if selected
                                    && self.core.conversation.title != "New chat"
                                {
                                    self.core.conversation.title.clone()
                                } else if session.title == "Untitled session" {
                                    "New Session".into()
                                } else {
                                    session.title.clone()
                                };
                                let title_matches = title.to_lowercase().contains(&query);
                                let content_matches = self.session_document_matches(&path, &query);
                                if !query.is_empty() && !title_matches && !content_matches {
                                    return None;
                                }
                                let display_title = if !query.is_empty() && !title_matches {
                                    self.session_document_summary(&path, &query)
                                        .map(|summary| format!("{title} · {summary}"))
                                        .unwrap_or_else(|| title.clone())
                                } else {
                                    title.clone()
                                };
                                let action_title = title;
                                let group = SharedString::from(format!("session-{index}-{session_index}"));
                                let action_path = path.clone();
                                let action_open = self
                                    .core
                                    .sidebar
                                    .session_action_target
                                    .as_ref()
                                    .is_some_and(|target| same_path(target, &path));
                                let target_active = self.session_is_active(index, &path, cx);
                                let age = session_age(self.session_modified_at(&session));
                                let status = self.session_status_indicator(index, &path, cx);
                                let action = Button::new(SharedString::from(format!("session-actions-{index}-{session_index}")))
                                    .icon(IconName::Ellipsis)
                                    .ghost()
                                    .compact()
                                    .tooltip("Session actions")
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        cx.stop_propagation();
                                        if this
                                            .core
                                            .sidebar
                                            .session_action_target
                                            .as_ref()
                                            .is_some_and(|target| same_path(target, &action_path))
                                        {
                                            this.dispatch(
                                                Action::SetSessionActionTarget(None),
                                                window,
                                                cx,
                                            );
                                        } else {
                                            this.dispatch(
                                                Action::SetSessionActionTarget(Some(action_path.clone())),
                                                window,
                                                cx,
                                            );
                                        }
                                        cx.notify();
                                    }))
                                    .into_any_element();
                                let open_path = path.clone();
                                Some(
                                    session_row(
                                        group.clone(),
                                        group.clone(),
                                        display_title,
                                        SessionRowTrailing {
                                            age,
                                            status,
                                            reduce_motion: self.settings.reduce_motion(),
                                        },
                                        selected,
                                        colors,
                                        Some(action),
                                    )
                                        .on_click(cx.listener(move |this, _, window, cx| {
                                            this.dispatch(
                                                Action::SetSessionActionTarget(None),
                                                window,
                                                cx,
                                            );
                                            this.open_project_session(index, open_path.clone(), window, cx)
                                        }))
                                        .on_key_down(cx.listener(move |this, event: &gpui::KeyDownEvent, window, cx| {
                                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                                this.open_project_session(index, keyboard_path.clone(), window, cx);
                                            }
                                        }))
                                        .children((action_open && !target_active).then(|| {
                                            session_actions_popover(
                                                SharedString::from(format!("{index}-{session_index}")),
                                                index,
                                                path.clone(),
                                                action_title,
                                                colors,
                                                cx,
                                            )
                                        })),
                                )
                            }))
                    }))
            }))
            .into_any_element()
    }

    fn flat_session_tree(
        &self,
        query: &str,
        colors: UiPalette,
        cx: &mut Context<Self>,
    ) -> gpui::AnyElement {
        let mut sessions = self
            .project_store
            .projects()
            .iter()
            .enumerate()
            .flat_map(|(project_index, project)| {
                self.project_sessions
                    .get(&project.sessions_dir)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(move |session| (project_index, project.name.clone(), session))
            })
            .filter_map(|(project_index, project_name, session)| {
                let selected = project_index == self.core.workspace.active_project
                    && same_path(&session.path, &self.core.session.current);
                let title = if selected && self.core.conversation.title != "New chat" {
                    self.core.conversation.title.clone()
                } else if session.title == "Untitled session" {
                    "New Session".into()
                } else {
                    session.title.clone()
                };
                if !query.is_empty()
                    && !title.to_lowercase().contains(query)
                    && !self.session_document_matches(&session.path, query)
                {
                    return None;
                }
                Some((
                    project_index,
                    project_name,
                    self.session_modified_at(&session),
                    session.path,
                    title,
                    selected,
                ))
            })
            .collect::<Vec<_>>();
        sessions.sort_by_key(|(_, _, modified, _, _, _)| *modified);
        if self.core.sidebar.sort_by_recent {
            sessions.reverse();
        }
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .px_3()
            .overflow_y_scrollbar()
            .children(sessions.into_iter().enumerate().map(
                |(row_index, (project_index, project_name, modified, path, title, selected))| {
                    let keyboard_path = path.clone();
                    let action_path = path.clone();
                    let action_title = title.clone();
                    let action_open = self
                        .core
                        .sidebar
                        .session_action_target
                        .as_ref()
                        .is_some_and(|target| same_path(target, &path));
                    let target_active = self.session_is_active(project_index, &path, cx);
                    let age = session_age(modified);
                    let status = self.session_status_indicator(project_index, &path, cx);
                    let action = Button::new(SharedString::from(format!(
                        "flat-session-actions-{row_index}"
                    )))
                    .icon(IconName::Ellipsis)
                    .ghost()
                    .compact()
                    .tooltip("Session actions")
                    .on_click(cx.listener(move |this, _, window, cx| {
                        cx.stop_propagation();
                        if this
                            .core
                            .sidebar
                            .session_action_target
                            .as_ref()
                            .is_some_and(|target| same_path(target, &action_path))
                        {
                            this.dispatch(Action::SetSessionActionTarget(None), window, cx);
                        } else {
                            this.dispatch(
                                Action::SetSessionActionTarget(Some(action_path.clone())),
                                window,
                                cx,
                            );
                        }
                        cx.notify();
                    }))
                    .into_any_element();
                    let open_path = path.clone();
                    session_row(
                        ("flat-session", row_index),
                        SharedString::from(format!("flat-session-{row_index}")),
                        format!("{title} · {project_name}"),
                        SessionRowTrailing {
                            age,
                            status,
                            reduce_motion: self.settings.reduce_motion(),
                        },
                        selected,
                        colors,
                        Some(action),
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.dispatch(Action::SetSessionActionTarget(None), window, cx);
                        this.open_project_session(project_index, open_path.clone(), window, cx)
                    }))
                    .on_key_down(cx.listener(
                        move |this, event: &gpui::KeyDownEvent, window, cx| {
                            if matches!(event.keystroke.key.as_str(), "enter" | "space") {
                                this.open_project_session(
                                    project_index,
                                    keyboard_path.clone(),
                                    window,
                                    cx,
                                );
                            }
                        },
                    ))
                    .children((action_open && !target_active).then(|| {
                        session_actions_popover(
                            SharedString::from(format!("flat-{row_index}")),
                            project_index,
                            path.clone(),
                            action_title,
                            colors,
                            cx,
                        )
                    }))
                },
            ))
            .into_any_element()
    }
}

fn sidebar_option(
    id: &'static str,
    label: &'static str,
    selected: bool,
    colors: UiPalette,
    on_click: impl Fn(&gpui::ClickEvent, &mut gpui::Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .id(id)
        .flex()
        .items_center()
        .justify_between()
        .h(px(34.0))
        .px_2()
        .rounded_lg()
        .cursor_pointer()
        .tab_index(0)
        .hover(move |item| item.bg(colors.hover))
        .on_click(on_click)
        .child(div().text_sm().child(label))
        .children(selected.then(|| Icon::new(IconName::Check).size_4()))
}

struct SessionRowTrailing {
    age: String,
    status: Option<SidebarSessionStatus>,
    reduce_motion: bool,
}

fn session_row(
    id: impl Into<gpui::ElementId>,
    group: impl Into<SharedString>,
    title: String,
    trailing: SessionRowTrailing,
    selected: bool,
    colors: UiPalette,
    action: Option<gpui::AnyElement>,
) -> gpui::Stateful<gpui::Div> {
    let group = group.into();
    let SessionRowTrailing {
        age,
        status,
        reduce_motion,
    } = trailing;
    div()
        .id(id)
        .group(group.clone())
        .relative()
        .flex()
        .items_center()
        .gap_2()
        .h(px(metrics::SESSION_ROW_HEIGHT))
        .pl(px(metrics::SESSION_ROW_INDENT))
        .pr_2()
        .rounded_lg()
        .cursor_pointer()
        .tab_index(0)
        .when(selected, |element| element.bg(colors.selected))
        .hover(move |element| element.bg(colors.hover))
        .child(
            div()
                .flex_1()
                .min_w(px(0.0))
                .truncate()
                .text_sm()
                .child(sidebar_label(&title, metrics::SIDEBAR_LABEL_UNITS)),
        )
        .child(
            div()
                .relative()
                .flex()
                .flex_none()
                .items_center()
                .justify_end()
                .w(px(metrics::SESSION_TRAILING_SLOT))
                .h_full()
                .child(
                    div()
                        .w_full()
                        .flex()
                        .items_center()
                        .justify_end()
                        .gap_1()
                        .text_right()
                        .text_xs()
                        .text_color(colors.muted_text)
                        .when(action.is_some(), |age| {
                            age.group_hover(group.clone(), |age| age.invisible())
                        })
                        .children(status.map(|status| {
                            session_status_icon(status, reduce_motion, colors, group.clone())
                        }))
                        .child(age),
                )
                .children(action.map(|action| {
                    div()
                        .absolute()
                        .top_0()
                        .right_0()
                        .flex()
                        .items_center()
                        .justify_end()
                        .h_full()
                        .invisible()
                        .group_hover(group, |element| element.visible())
                        .child(action)
                })),
        )
}

fn session_status_icon(
    status: SidebarSessionStatus,
    reduce_motion: bool,
    colors: UiPalette,
    group: SharedString,
) -> gpui::AnyElement {
    let label = match status {
        SidebarSessionStatus::Preparing => "Preparing",
        SidebarSessionStatus::Running => "Running",
        SidebarSessionStatus::ApprovalNeeded => "Needs approval",
        SidebarSessionStatus::Failed => "Failed",
        SidebarSessionStatus::Unread => "Unread response",
    };
    let icon = match status {
        SidebarSessionStatus::Preparing | SidebarSessionStatus::Running if !reduce_motion => {
            Spinner::new()
                .small()
                .icon(IconName::LoaderCircle)
                .color(colors.primary)
                .into_any_element()
        }
        SidebarSessionStatus::Preparing | SidebarSessionStatus::Running => {
            Icon::new(IconName::LoaderCircle)
                .size_4()
                .text_color(colors.primary)
                .into_any_element()
        }
        SidebarSessionStatus::ApprovalNeeded => Icon::new(IconName::TriangleAlert)
            .size_4()
            .text_color(colors.warning)
            .into_any_element(),
        SidebarSessionStatus::Failed => Icon::new(IconName::CircleX)
            .size_4()
            .text_color(colors.danger)
            .into_any_element(),
        SidebarSessionStatus::Unread => div()
            .size(px(6.0))
            .rounded_full()
            .bg(colors.primary)
            .into_any_element(),
    };
    div()
        .id(SharedString::from(format!("session-status-{group}")))
        .flex()
        .flex_none()
        .items_center()
        .justify_center()
        .size(px(metrics::SIDEBAR_ICON_SLOT))
        .tooltip(move |window, cx| Tooltip::new(label).build(window, cx))
        .child(icon)
        .into_any_element()
}

fn session_actions_popover(
    key: SharedString,
    project_index: usize,
    path: PathBuf,
    title: String,
    colors: UiPalette,
    cx: &mut Context<DesktopApp>,
) -> gpui::AnyElement {
    deferred(
        div()
            .absolute()
            .top_0()
            .left(relative(1.0))
            .ml_2()
            .w(px(184.0))
            .p_1()
            .rounded(px(12.0))
            .border_1()
            .border_color(colors.border)
            .bg(colors.sidebar)
            .shadow_lg()
            .occlude()
            .on_mouse_down(MouseButton::Left, |_, _, cx| cx.stop_propagation())
            .child(
                Button::new(SharedString::from(format!("rename-session-{key}")))
                    .label("Rename")
                    .ghost()
                    .w_full()
                    .h(px(metrics::WORKSPACE_ROW_HEIGHT))
                    .px_2()
                    .rounded(px(8.0))
                    .justify_start()
                    .text_sm()
                    .on_click(cx.listener({
                        let path = path.clone();
                        let title = title.clone();
                        move |this, _, window, cx| {
                            cx.stop_propagation();
                            this.dispatch(Action::SetSessionActionTarget(None), window, cx);
                            this.open_target_rename_session_dialog(
                                project_index,
                                path.clone(),
                                title.clone(),
                                window,
                                cx,
                            )
                        }
                    })),
            )
            .child(
                Button::new(SharedString::from(format!("delete-session-{key}")))
                    .label("Delete")
                    .ghost()
                    .w_full()
                    .h(px(metrics::WORKSPACE_ROW_HEIGHT))
                    .px_2()
                    .rounded(px(8.0))
                    .justify_start()
                    .text_sm()
                    .text_color(colors.danger)
                    .on_click(cx.listener(move |this, _, window, cx| {
                        cx.stop_propagation();
                        this.dispatch(Action::SetSessionActionTarget(None), window, cx);
                        this.open_target_delete_session_dialog(
                            project_index,
                            path.clone(),
                            title.clone(),
                            window,
                            cx,
                        )
                    })),
            ),
    )
    .with_priority(1)
    .into_any_element()
}

fn sidebar_label(value: &str, max_units: usize) -> String {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut units = 0;
    let mut output = String::new();
    let mut truncated = false;
    for character in value.chars() {
        let character_units = if character.is_ascii() { 1 } else { 2 };
        if units + character_units > max_units {
            truncated = true;
            break;
        }
        units += character_units;
        output.push(character);
    }
    if truncated {
        output.push('…');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::sidebar_label;

    #[test]
    fn sidebar_labels_are_width_aware_and_show_truncation() {
        assert_eq!(sidebar_label("  short   title  ", 24), "short title");
        assert_eq!(
            sidebar_label("write forty numbered short lines", 24),
            "write forty numbered sho…"
        );
        assert_eq!(
            sidebar_label("你好，这是一个很长的会话标题", 12),
            "你好，这是一…"
        );
    }
}
