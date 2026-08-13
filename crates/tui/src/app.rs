use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use kcastle_agent::{AgentEvent, Model, SessionInfo, TranscriptItem};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_textarea::TextArea;
use time::OffsetDateTime;
use time::macros::format_description;

const ACCENT: Color = Color::Rgb(196, 181, 253);

#[derive(Debug)]
enum Entry {
    User(String),
    Assistant(String),
    Tool {
        call_id: String,
        name: String,
        arguments: String,
        output: Option<String>,
        failed: bool,
        expanded: bool,
    },
    Notice(String),
}

#[derive(Debug)]
enum Modal {
    Approval {
        call_id: String,
        name: String,
        arguments: String,
    },
    Sessions {
        sessions: Vec<SessionInfo>,
        selected: usize,
    },
    Models {
        names: Vec<String>,
        selected: usize,
    },
    Commands {
        items: Vec<CommandItem>,
        query: String,
        selected: usize,
    },
    AllowAll {
        pending_call_id: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct CommandItem {
    command: String,
    description: String,
    prefill: bool,
}

pub enum UiAction {
    None,
    Submit(String),
    Approve {
        call_id: String,
        allow: bool,
    },
    Resume(PathBuf),
    SelectModel(usize),
    SetPermissions {
        allow_all: bool,
        pending_call_id: Option<String>,
    },
    Prefill(String),
    Abort,
    Exit,
}

pub struct App {
    entries: Vec<Entry>,
    input: TextArea<'static>,
    activity: String,
    model: String,
    session: String,
    cwd: PathBuf,
    context_window: usize,
    cached_tokens: u32,
    used_tokens: u32,
    modal: Option<Modal>,
    tool_rows: Vec<(u16, usize)>,
    selected_tool: Option<usize>,
    follow: bool,
    scroll: u16,
    max_scroll: u16,
    allow_all: bool,
    should_exit: bool,
}

impl App {
    pub fn new(
        model: &Model,
        session: &SessionInfo,
        transcript: Vec<TranscriptItem>,
        cwd: &Path,
        usage: Option<(u32, u32)>,
        allow_all: bool,
    ) -> Self {
        let mut input = TextArea::default();
        input.set_cursor_line_style(Style::default());
        input.set_placeholder_text("Message K…  / for commands");
        let mut app = Self {
            entries: Vec::new(),
            input,
            activity: "idle".into(),
            model: format!("{} · {}", model.name(), model.model()),
            session: session.title.clone(),
            cwd: cwd.to_path_buf(),
            context_window: model.context_window(),
            cached_tokens: 0,
            used_tokens: 0,
            modal: None,
            tool_rows: Vec::new(),
            selected_tool: None,
            follow: true,
            scroll: 0,
            max_scroll: 0,
            allow_all,
            should_exit: false,
        };
        app.set_usage(usage);
        app.load_transcript(transcript);
        app
    }

    pub fn render(&mut self, frame: &mut Frame<'_>, running: bool) {
        let area = frame.area();
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(4),
                Constraint::Length(1),
                Constraint::Length(3),
            ])
            .split(area);

        let (lines, tool_lines) = self.transcript_lines(layout[0].width.saturating_sub(2) as usize);
        let viewport_height = layout[0].height.saturating_sub(2);
        let content_height = lines.len();
        let transcript = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!(" K CASTLE v{} ", env!("CARGO_PKG_VERSION")),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )))
            .wrap(Wrap { trim: false });
        self.max_scroll = content_height
            .saturating_sub(viewport_height as usize)
            .min(u16::MAX as usize) as u16;
        self.scroll = if self.follow {
            self.max_scroll
        } else {
            self.scroll.min(self.max_scroll)
        };
        let transcript = transcript.scroll((self.scroll, 0));
        frame.render_widget(transcript, layout[0]);

        self.tool_rows = tool_lines
            .into_iter()
            .filter_map(|(line, entry)| {
                let row = line as u16;
                (row >= self.scroll && row < self.scroll.saturating_add(viewport_height))
                    .then_some((layout[0].y + 1 + row - self.scroll, entry))
            })
            .collect();

        frame.render_widget(Paragraph::new(self.status_line(running)), layout[1]);

        self.input.set_block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(if running { Color::DarkGray } else { ACCENT })),
        );
        frame.render_widget(&self.input, layout[2]);

        if let Some(modal) = &self.modal {
            render_modal(frame, modal, centered(area, 72, 60));
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent, running: bool) -> UiAction {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            return UiAction::Exit;
        }
        if let Some(action) = self.handle_modal_key(key) {
            return action;
        }
        match key.code {
            KeyCode::Esc if running => UiAction::Abort,
            KeyCode::Char('/') if self.input.lines().iter().all(String::is_empty) => {
                self.show_commands(running);
                UiAction::None
            }
            KeyCode::PageUp => {
                self.follow = false;
                self.scroll = self.scroll.saturating_sub(10);
                UiAction::None
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(10).min(self.max_scroll);
                self.follow = self.scroll == self.max_scroll;
                UiAction::None
            }
            KeyCode::End => {
                self.follow = true;
                UiAction::None
            }
            KeyCode::Tab => {
                let tools = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter_map(|(index, entry)| {
                        matches!(entry, Entry::Tool { .. }).then_some(index)
                    })
                    .collect::<Vec<_>>();
                if !tools.is_empty() {
                    let position = self
                        .selected_tool
                        .and_then(|selected| tools.iter().position(|index| *index == selected))
                        .map_or(0, |position| (position + 1) % tools.len());
                    self.selected_tool = Some(tools[position]);
                }
                UiAction::None
            }
            KeyCode::Enter => {
                let value = self.input.lines().join("\n");
                if value.trim().is_empty() {
                    if let Some(index) = self.selected_tool
                        && let Some(Entry::Tool { expanded, .. }) = self.entries.get_mut(index)
                    {
                        *expanded = !*expanded;
                    }
                    return UiAction::None;
                }
                self.input = TextArea::default();
                self.input.set_cursor_line_style(Style::default());
                self.input
                    .set_placeholder_text("Message K…  / for commands");
                UiAction::Submit(value)
            }
            _ => {
                self.input.input(key);
                UiAction::None
            }
        }
    }

    pub fn handle_mouse(&mut self, event: MouseEvent) {
        match event.kind {
            MouseEventKind::ScrollUp => {
                self.follow = false;
                self.scroll = self.scroll.saturating_sub(3);
            }
            MouseEventKind::ScrollDown => {
                self.scroll = self.scroll.saturating_add(3).min(self.max_scroll);
                self.follow = self.scroll == self.max_scroll;
            }
            MouseEventKind::Down(MouseButton::Left) => {
                if let Some((_, entry)) = self.tool_rows.iter().find(|(row, _)| *row == event.row)
                    && let Some(Entry::Tool { expanded, .. }) = self.entries.get_mut(*entry)
                {
                    self.selected_tool = Some(*entry);
                    *expanded = !*expanded;
                }
            }
            _ => {}
        }
    }

    pub fn paste(&mut self, value: &str) {
        self.input.insert_str(value);
    }

    pub fn apply_event(&mut self, event: AgentEvent) -> Option<(String, bool)> {
        match event {
            AgentEvent::RunStarted(_) => self.activity = "thinking".into(),
            AgentEvent::ModelStarted(turn) => {
                self.activity = format!("model turn {turn}");
                self.entries.push(Entry::Assistant(String::new()));
                self.follow = true;
            }
            AgentEvent::TextDelta(delta) => {
                if let Some(Entry::Assistant(text)) = self.entries.last_mut() {
                    text.push_str(&delta);
                } else {
                    self.entries.push(Entry::Assistant(delta));
                }
                self.follow = true;
            }
            AgentEvent::ApprovalRequired(call) => {
                if self.allow_all {
                    return Some((call.call_id, true));
                }
                self.modal = Some(Modal::Approval {
                    call_id: call.call_id,
                    name: call.name,
                    arguments: call.arguments,
                });
            }
            AgentEvent::ToolStarted(call) => {
                self.entries.push(Entry::Tool {
                    call_id: call.call_id,
                    name: call.name,
                    arguments: call.arguments,
                    output: None,
                    failed: false,
                    expanded: false,
                });
            }
            AgentEvent::ToolFinished { call, result } => {
                if let Some(Entry::Tool { output, failed, .. }) = self.entries.iter_mut().rev().find(
                    |entry| matches!(entry, Entry::Tool { call_id, .. } if call_id == &call.call_id),
                ) {
                    *output = Some(result.output);
                    *failed = result.is_error;
                }
            }
            AgentEvent::CompactionStarted { tokens_before } => {
                self.activity = format!("compacting {tokens_before} estimated tokens");
            }
            AgentEvent::CompactionFinished { .. } => {
                self.activity = "compaction finished".into();
            }
            AgentEvent::RunFinished(summary) => {
                self.activity = "idle".into();
                self.set_usage(
                    summary.usage.as_ref().map(|usage| {
                        (usage.input_tokens_details.cached_tokens, usage.total_tokens)
                    }),
                );
            }
            AgentEvent::RunAborted => self.activity = "aborted".into(),
            AgentEvent::RunFailed(error) => {
                self.activity = "failed".into();
                self.entries.push(Entry::Notice(format!("Error: {error}")));
            }
        }
        None
    }

    pub fn push_user(&mut self, value: String) {
        self.entries.push(Entry::User(value));
        self.follow = true;
    }

    pub fn notice(&mut self, value: impl Into<String>) {
        self.entries.push(Entry::Notice(value.into()));
        self.follow = true;
    }

    pub fn set_identity(&mut self, model: &Model, session: &SessionInfo) {
        self.model = format!("{} · {}", model.name(), model.model());
        self.context_window = model.context_window();
        self.session = session.title.clone();
    }

    pub fn set_usage_values(&mut self, usage: Option<(u32, u32)>) {
        self.set_usage(usage);
    }

    pub fn load_transcript(&mut self, transcript: Vec<TranscriptItem>) {
        self.entries.clear();
        for item in transcript {
            match item {
                TranscriptItem::User(text) => self.entries.push(Entry::User(text)),
                TranscriptItem::Assistant(text) => self.entries.push(Entry::Assistant(text)),
                TranscriptItem::ToolCall {
                    call_id,
                    name,
                    arguments,
                } => self.entries.push(Entry::Tool {
                    call_id,
                    name,
                    arguments,
                    output: None,
                    failed: false,
                    expanded: false,
                }),
                TranscriptItem::ToolOutput { call_id, output } => {
                    if let Some(Entry::Tool {
                        output: tool_output,
                        ..
                    }) = self.entries.iter_mut().rev().find(
                        |entry| matches!(entry, Entry::Tool { call_id: id, .. } if id == &call_id),
                    ) {
                        *tool_output = Some(output);
                    }
                }
                TranscriptItem::Summary(text) => {
                    self.entries
                        .push(Entry::Notice(format!("Earlier context: {text}")));
                }
            }
        }
        self.follow = true;
    }

    pub fn show_sessions(&mut self, sessions: Vec<SessionInfo>) {
        if sessions.is_empty() {
            self.notice("No saved sessions");
        } else {
            self.modal = Some(Modal::Sessions {
                sessions,
                selected: 0,
            });
        }
    }

    pub fn show_models(&mut self, models: &[Model], selected: usize) {
        self.modal = Some(Modal::Models {
            names: models
                .iter()
                .map(|model| format!("{} · {}", model.name(), model.model()))
                .collect(),
            selected,
        });
    }

    pub fn show_commands(&mut self, running: bool) {
        self.modal = Some(Modal::Commands {
            items: command_items(running),
            query: String::new(),
            selected: 0,
        });
    }

    pub fn request_permission_toggle(&mut self) -> Option<bool> {
        if self.allow_all {
            self.allow_all = false;
            Some(false)
        } else {
            self.modal = Some(Modal::AllowAll {
                pending_call_id: None,
            });
            None
        }
    }

    pub fn set_permission_mode(&mut self, allow_all: bool) {
        self.allow_all = allow_all;
        self.notice(format!(
            "Tool permissions: {}",
            if allow_all { "allow all" } else { "ask" }
        ));
    }

    pub fn prefill(&mut self, value: &str) {
        self.input = TextArea::default();
        self.input.set_cursor_line_style(Style::default());
        self.input
            .set_placeholder_text("Message K…  / for commands");
        self.input.insert_str(value);
    }

    pub fn request_exit(&mut self) {
        self.should_exit = true;
    }

    pub fn should_exit(&self) -> bool {
        self.should_exit
    }

    fn set_usage(&mut self, usage: Option<(u32, u32)>) {
        (self.cached_tokens, self.used_tokens) = usage.unwrap_or((0, 0));
    }

    fn status_line(&self, running: bool) -> Line<'static> {
        let activity = if running { &self.activity } else { "idle" };
        Line::from(vec![
            Span::styled(format!(" {activity} "), Style::default().fg(ACCENT)),
            Span::raw(format!(
                "permissions: {} · {} · context {}/{}/{} · {}",
                if self.allow_all { "allow all" } else { "ask" },
                self.model,
                self.cached_tokens,
                self.used_tokens,
                self.context_window,
                self.session,
            )),
        ])
    }

    fn handle_modal_key(&mut self, key: KeyEvent) -> Option<UiAction> {
        let modal = self.modal.as_mut()?;
        let action = match modal {
            Modal::Approval { call_id, .. } => match key.code {
                KeyCode::Char('y') | KeyCode::Enter => Some(UiAction::Approve {
                    call_id: call_id.clone(),
                    allow: true,
                }),
                KeyCode::Char('a') => {
                    self.modal = Some(Modal::AllowAll {
                        pending_call_id: Some(call_id.clone()),
                    });
                    return Some(UiAction::None);
                }
                KeyCode::Char('n') | KeyCode::Esc => Some(UiAction::Approve {
                    call_id: call_id.clone(),
                    allow: false,
                }),
                _ => None,
            },
            Modal::Sessions { sessions, selected } => match key.code {
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    None
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(sessions.len() - 1);
                    None
                }
                KeyCode::Enter => Some(UiAction::Resume(sessions[*selected].path.clone())),
                KeyCode::Esc => Some(UiAction::None),
                _ => None,
            },
            Modal::Models { names, selected } => match key.code {
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    None
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(names.len() - 1);
                    None
                }
                KeyCode::Enter => Some(UiAction::SelectModel(*selected)),
                KeyCode::Esc => Some(UiAction::None),
                _ => None,
            },
            Modal::Commands {
                items,
                query,
                selected,
            } => {
                let visible = filtered_commands(items, query);
                match key.code {
                    KeyCode::Up => {
                        *selected = selected.saturating_sub(1);
                        None
                    }
                    KeyCode::Down => {
                        *selected = (*selected + 1).min(visible.len().saturating_sub(1));
                        None
                    }
                    KeyCode::Backspace => {
                        query.pop();
                        *selected = 0;
                        None
                    }
                    KeyCode::Char(character) => {
                        query.push(character);
                        *selected = 0;
                        None
                    }
                    KeyCode::Enter => visible.get(*selected).map(|item| {
                        if item.prefill {
                            UiAction::Prefill(format!("{} ", item.command))
                        } else {
                            UiAction::Submit(item.command.clone())
                        }
                    }),
                    KeyCode::Esc => Some(UiAction::None),
                    _ => None,
                }
            }
            Modal::AllowAll { pending_call_id } => match key.code {
                KeyCode::Char('y') | KeyCode::Enter => Some(UiAction::SetPermissions {
                    allow_all: true,
                    pending_call_id: pending_call_id.clone(),
                }),
                KeyCode::Char('n') | KeyCode::Esc => Some(UiAction::None),
                _ => None,
            },
        };
        if action.is_some() {
            self.modal = None;
        }
        action.or(Some(UiAction::None))
    }

    fn transcript_lines(&self, width: usize) -> (Vec<Line<'static>>, Vec<(usize, usize)>) {
        if self.entries.is_empty() {
            return (
                vec![
                    Line::from(""),
                    Line::from(Span::styled(
                        format!("  {}", self.cwd.display()),
                        Style::default().fg(Color::DarkGray),
                    )),
                    Line::from(Span::styled(
                        "  Type / for commands.",
                        Style::default().fg(Color::DarkGray),
                    )),
                ],
                Vec::new(),
            );
        }
        let mut lines = Vec::new();
        let mut tool_lines = Vec::new();
        for (entry_index, entry) in self.entries.iter().enumerate() {
            match entry {
                Entry::User(text) => push_text(
                    &mut lines,
                    "YOU",
                    text,
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Entry::Assistant(text) if !text.is_empty() => {
                    lines.push(Line::from(Span::styled(
                        "K",
                        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                    )));
                    lines.extend(
                        markdown_ratatui::render_with(
                            &markdown_stream::parse(text),
                            &markdown_ratatui::Theme::default(),
                            width,
                        )
                        .lines,
                    );
                    lines.push(Line::from(""));
                }
                Entry::Tool {
                    name,
                    arguments,
                    output,
                    failed,
                    expanded,
                    ..
                } => {
                    let mark = if *failed {
                        "!"
                    } else if output.is_some() {
                        "✓"
                    } else {
                        "·"
                    };
                    let color = if *failed { Color::Red } else { Color::Cyan };
                    tool_lines.push((lines.len(), entry_index));
                    let mut line = Line::from(vec![
                        Span::styled(format!("{mark} {name}"), Style::default().fg(color)),
                        Span::styled(
                            format!("  {}", compact(arguments, 100)),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]);
                    if self.selected_tool == Some(entry_index) {
                        line = line.style(Style::default().bg(Color::DarkGray));
                    }
                    lines.push(line);
                    if *expanded {
                        lines.push(Line::from(Span::styled(
                            format!("  Arguments: {arguments}"),
                            Style::default().fg(Color::DarkGray),
                        )));
                        if let Some(output) = output {
                            for line in output.lines() {
                                lines.push(Line::from(Span::styled(
                                    format!("  {line}"),
                                    Style::default().fg(Color::DarkGray),
                                )));
                            }
                        }
                    }
                    lines.push(Line::from(""));
                }
                Entry::Notice(text) => {
                    lines.push(Line::from(Span::styled(
                        text.clone(),
                        Style::default().fg(Color::Yellow),
                    )));
                    lines.push(Line::from(""));
                }
                Entry::Assistant(_) => {}
            }
        }
        (lines, tool_lines)
    }
}

fn push_text(lines: &mut Vec<Line<'static>>, label: &str, text: &str, style: Style) {
    lines.push(Line::from(Span::styled(
        label.to_owned(),
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
    )));
    for line in text.lines() {
        lines.push(Line::from(Span::styled(line.to_owned(), style)));
    }
    lines.push(Line::from(""));
}

fn compact(value: &str, limit: usize) -> String {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if value.chars().count() <= limit {
        value
    } else {
        format!("{}…", value.chars().take(limit).collect::<String>())
    }
}

fn command_items(running: bool) -> Vec<CommandItem> {
    let mut items = vec![
        CommandItem {
            command: "/resume".into(),
            description: "Switch to a saved session".into(),
            prefill: false,
        },
        CommandItem {
            command: "/model".into(),
            description: "Switch model backend".into(),
            prefill: false,
        },
        CommandItem {
            command: "/compact".into(),
            description: "Compact the active context".into(),
            prefill: false,
        },
        CommandItem {
            command: "/permissions".into(),
            description: "Switch tool approval policy".into(),
            prefill: false,
        },
        CommandItem {
            command: "/help".into(),
            description: "Show command help".into(),
            prefill: false,
        },
        CommandItem {
            command: "/exit".into(),
            description: "Exit K".into(),
            prefill: false,
        },
    ];
    if running {
        items.insert(
            4,
            CommandItem {
                command: "/queue".into(),
                description: "Queue a message after the active task".into(),
                prefill: true,
            },
        );
    }
    items
}

fn filtered_commands<'a>(items: &'a [CommandItem], query: &str) -> Vec<&'a CommandItem> {
    let query = query.trim_start_matches('/').to_ascii_lowercase();
    let mut matches = items
        .iter()
        .filter(|item| {
            query.is_empty()
                || item.command.to_ascii_lowercase().contains(&query)
                || item.description.to_ascii_lowercase().contains(&query)
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|item| {
        let command = item.command.trim_start_matches('/').to_ascii_lowercase();
        (!command.starts_with(&query), !command.contains(&query))
    });
    matches
}

fn session_label(session: &SessionInfo) -> String {
    let timestamp = i64::try_from(session.created_at)
        .ok()
        .and_then(|timestamp| OffsetDateTime::from_unix_timestamp(timestamp).ok())
        .and_then(|time| {
            time.format(format_description!(
                "[year]-[month]-[day] [hour]:[minute] UTC"
            ))
            .ok()
        })
        .unwrap_or_else(|| session.created_at.to_string());
    format!("{} · {timestamp}", session.title)
}

fn render_modal(frame: &mut Frame<'_>, modal: &Modal, area: Rect) {
    frame.render_widget(Clear, area);
    match modal {
        Modal::Approval {
            name, arguments, ..
        } => {
            let content = format!(
                "Tool: {name}\n\n{arguments}\n\n[y/Enter] allow   [n/Esc] deny   [a] allow all"
            );
            frame.render_widget(
                Paragraph::new(content)
                    .block(Block::default().borders(Borders::ALL).title(" Permission "))
                    .wrap(Wrap { trim: false }),
                area,
            );
        }
        Modal::Sessions { sessions, selected } => {
            let items = sessions
                .iter()
                .map(|session| ListItem::new(session_label(session)))
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(*selected));
            frame.render_stateful_widget(
                List::new(items)
                    .block(Block::default().borders(Borders::ALL).title(" Sessions "))
                    .highlight_style(Style::default().bg(Color::DarkGray).fg(ACCENT)),
                area,
                &mut state,
            );
        }
        Modal::Models { names, selected } => {
            let items = names
                .iter()
                .map(|name| ListItem::new(name.clone()))
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(*selected));
            frame.render_stateful_widget(
                List::new(items)
                    .block(Block::default().borders(Borders::ALL).title(" Models "))
                    .highlight_style(Style::default().bg(Color::DarkGray).fg(ACCENT)),
                area,
                &mut state,
            );
        }
        Modal::Commands {
            items,
            query,
            selected,
        } => {
            let visible = filtered_commands(items, query);
            let rows = visible
                .iter()
                .map(|item| {
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{:<14}", item.command), Style::default().fg(ACCENT)),
                        Span::raw(&item.description),
                    ]))
                })
                .collect::<Vec<_>>();
            let mut state =
                ListState::default().with_selected((!rows.is_empty()).then_some(*selected));
            frame.render_stateful_widget(
                List::new(rows)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!(" Commands /{query} ")),
                    )
                    .highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White)),
                area,
                &mut state,
            );
        }
        Modal::AllowAll { .. } => {
            frame.render_widget(
                Paragraph::new(
                    "Allow all can execute any command without asking again.\n\nOnly continue if you trust this session.\n\n[y/Enter] enable   [n/Esc] cancel",
                )
                .block(Block::default().borders(Borders::ALL).title(" Enable allow all? "))
                .wrap(Wrap { trim: false }),
                area,
            );
        }
    }
}

fn centered(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);
    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(vertical[1])[1]
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
    use kcastle_agent::{Model, SessionInfo};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    use super::{App, Entry, Modal, UiAction, filtered_commands, session_label};

    fn app() -> App {
        App::new(
            &Model::new("test", "key", "http://localhost", "test-model", 10_000),
            &SessionInfo {
                path: PathBuf::from("session.jsonl"),
                title: "session".into(),
                created_at: 0,
            },
            Vec::new(),
            std::path::Path::new("/work"),
            Some((12, 34)),
            false,
        )
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn slash_palette_filters_and_selects_commands() {
        let mut app = app();
        assert!(matches!(
            app.handle_key(key(KeyCode::Char('/')), false),
            UiAction::None
        ));
        let Modal::Commands { items, query, .. } = app.modal.as_ref().unwrap() else {
            panic!("command palette not opened")
        };
        assert_eq!(filtered_commands(items, query).len(), 6);

        app.handle_key(key(KeyCode::Char('m')), false);
        let Modal::Commands { items, query, .. } = app.modal.as_ref().unwrap() else {
            panic!("command palette closed")
        };
        assert_eq!(filtered_commands(items, query)[0].command, "/model");
        assert!(matches!(
            app.handle_key(key(KeyCode::Enter), false),
            UiAction::Submit(value) if value == "/model"
        ));
    }

    #[test]
    fn allow_all_requires_confirmation() {
        let mut app = app();
        assert_eq!(app.request_permission_toggle(), None);
        assert!(matches!(app.modal, Some(Modal::AllowAll { .. })));
        assert!(matches!(
            app.handle_key(key(KeyCode::Enter), false),
            UiAction::SetPermissions {
                allow_all: true,
                ..
            }
        ));
        assert!(!app.allow_all);
    }

    #[test]
    fn markdown_and_full_tool_details_are_rendered() {
        let mut app = app();
        app.entries.push(Entry::Assistant("**bold**".into()));
        app.entries.push(Entry::Tool {
            call_id: "call".into(),
            name: "shell".into(),
            arguments: r#"{"command":"printf hello"}"#.into(),
            output: Some("exit_code=0\nfull output".into()),
            failed: false,
            expanded: true,
        });
        let (lines, tools) = app.transcript_lines(80);
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content == "bold" && span.style.add_modifier.contains(Modifier::BOLD)
        }));
        let rendered = lines
            .iter()
            .flat_map(|line| &line.spans)
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(rendered.contains("full output"));
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn status_contains_permissions_usage_and_context_window() {
        let app = app();
        let status = app
            .status_line(false)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(status.contains("permissions: ask"));
        assert!(status.contains("context 12/34/10000"));
        assert_eq!(
            session_label(&SessionInfo {
                path: PathBuf::new(),
                title: "saved".into(),
                created_at: 0,
            }),
            "saved · 1970-01-01 00:00 UTC"
        );
    }

    #[test]
    fn transcript_scrolls_and_stays_within_content() {
        let mut app = app();
        for line in 0..30 {
            app.notice(format!("line {line}"));
        }
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        assert!(app.max_scroll > 0);
        assert_eq!(app.scroll, app.max_scroll);

        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollUp,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        });
        assert_eq!(app.scroll, app.max_scroll - 3);
        assert!(!app.follow);

        for _ in 0..20 {
            app.handle_mouse(MouseEvent {
                kind: MouseEventKind::ScrollDown,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::NONE,
            });
        }
        assert_eq!(app.scroll, app.max_scroll);
        assert!(app.follow);
    }
}
