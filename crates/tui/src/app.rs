use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use kcastle_agent::{AgentEvent, Model, SessionInfo, TranscriptItem};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_textarea::TextArea;

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
}

pub enum UiAction {
    None,
    Submit(String),
    Approve { call_id: String, allow: bool },
    Resume(PathBuf),
    SelectModel(usize),
    Abort,
    Exit,
}

pub struct App {
    entries: Vec<Entry>,
    input: TextArea<'static>,
    status: String,
    modal: Option<Modal>,
    follow: bool,
    scroll: u16,
    allow_all: bool,
    should_exit: bool,
}

impl App {
    pub fn new(model: &Model, session: &SessionInfo, transcript: Vec<TranscriptItem>) -> Self {
        let mut input = TextArea::default();
        input.set_cursor_line_style(Style::default());
        input.set_placeholder_text("Message K…");
        let mut app = Self {
            entries: Vec::new(),
            input,
            status: format!("{} · {} · {}", model.name(), model.model(), session.title),
            modal: None,
            follow: true,
            scroll: 0,
            allow_all: false,
            should_exit: false,
        };
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

        let lines = self.transcript_lines();
        let viewport_height = layout[0].height.saturating_sub(2);
        if self.follow {
            self.scroll = (lines.len() as u16).saturating_sub(viewport_height);
        }
        let transcript = Paragraph::new(Text::from(lines))
            .block(Block::default().borders(Borders::ALL).title(Span::styled(
                format!(" K CASTLE v{} ", env!("CARGO_PKG_VERSION")),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )))
            .wrap(Wrap { trim: false })
            .scroll((self.scroll, 0));
        frame.render_widget(transcript, layout[0]);

        let activity = if running { "running" } else { "idle" };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(format!(" {activity} "), Style::default().fg(ACCENT)),
                Span::raw(&self.status),
            ])),
            layout[1],
        );

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
            KeyCode::PageUp => {
                self.follow = false;
                self.scroll = self.scroll.saturating_sub(10);
                UiAction::None
            }
            KeyCode::PageDown => {
                self.scroll = self.scroll.saturating_add(10);
                self.follow = false;
                UiAction::None
            }
            KeyCode::End => {
                self.follow = true;
                UiAction::None
            }
            KeyCode::Enter => {
                let value = self.input.lines().join("\n");
                if value.trim().is_empty() {
                    return UiAction::None;
                }
                self.input = TextArea::default();
                self.input.set_cursor_line_style(Style::default());
                self.input.set_placeholder_text("Message K…");
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
                self.follow = false;
                self.scroll = self.scroll.saturating_add(3);
            }
            _ => {}
        }
    }

    pub fn paste(&mut self, value: &str) {
        self.input.insert_str(value);
    }

    pub fn apply_event(&mut self, event: AgentEvent) -> Option<(String, bool)> {
        match event {
            AgentEvent::RunStarted(_) => self.status = "thinking".into(),
            AgentEvent::ModelStarted(turn) => {
                self.status = format!("model turn {turn}");
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
                self.status = format!("compacting {tokens_before} estimated tokens");
            }
            AgentEvent::CompactionFinished { .. } => {
                self.status = "compaction finished".into();
            }
            AgentEvent::RunFinished(summary) => {
                self.status = summary.usage.map_or_else(
                    || "finished".into(),
                    |usage| format!("{} tokens", usage.total_tokens),
                );
            }
            AgentEvent::RunAborted => self.status = "aborted".into(),
            AgentEvent::RunFailed(error) => {
                self.status = "failed".into();
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
        self.status = format!("{} · {} · {}", model.name(), model.model(), session.title);
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

    pub fn toggle_permissions(&mut self) {
        self.allow_all = !self.allow_all;
        let mode = if self.allow_all { "allow all" } else { "ask" };
        self.notice(format!("Tool permissions: {mode}"));
    }

    pub fn request_exit(&mut self) {
        self.should_exit = true;
    }

    pub fn should_exit(&self) -> bool {
        self.should_exit
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
                    self.allow_all = true;
                    Some(UiAction::Approve {
                        call_id: call_id.clone(),
                        allow: true,
                    })
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
        };
        if action.is_some() {
            self.modal = None;
        }
        action.or(Some(UiAction::None))
    }

    fn transcript_lines(&self) -> Vec<Line<'static>> {
        if self.entries.is_empty() {
            return vec![
                Line::from(""),
                Line::from(Span::styled(
                    "  A minimal native agent harness.",
                    Style::default().fg(Color::DarkGray),
                )),
                Line::from(Span::styled(
                    "  Type /help for commands.",
                    Style::default().fg(Color::DarkGray),
                )),
            ];
        }
        let mut lines = Vec::new();
        for entry in &self.entries {
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
                    push_text(&mut lines, "K", text, Style::default())
                }
                Entry::Tool {
                    name,
                    arguments,
                    output,
                    failed,
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
                    lines.push(Line::from(vec![
                        Span::styled(format!("{mark} {name}"), Style::default().fg(color)),
                        Span::styled(
                            format!("  {}", compact(arguments, 100)),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]));
                    if let Some(output) = output {
                        lines.push(Line::from(Span::styled(
                            format!("  {}", compact(output, 180)),
                            Style::default().fg(Color::DarkGray),
                        )));
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
        lines
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
                .map(|session| ListItem::new(session.title.clone()))
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
