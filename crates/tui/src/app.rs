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
const USER_BACKGROUND: Color = Color::Rgb(65, 69, 77);

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
    show_startup: bool,
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
            show_startup: false,
            should_exit: false,
        };
        app.set_usage(usage);
        app.load_transcript(transcript);
        app.show_startup = true;
        app
    }

    pub fn render(&mut self, frame: &mut Frame<'_>, running: bool) {
        let area = frame.area();
        let compact = area.height < 8;
        let input_height = if area.height >= 4 { 3 } else { 1 };
        let status_height = u16::from(!compact);
        let command_height = match &self.modal {
            Some(Modal::Commands { items, query, .. }) => {
                filtered_commands(items, query).len().min(u16::MAX as usize) as u16
            }
            _ => 0,
        }
        .min(
            area.height
                .saturating_sub(input_height)
                .saturating_sub(status_height),
        );
        let content_width = area.width;
        let (lines, tool_lines) = self.transcript_lines(content_width as usize);
        let transcript = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
        let content_height = transcript.line_count(content_width).min(u16::MAX as usize) as u16;
        let transcript_height = if self.show_startup && self.entries.is_empty() {
            9
        } else {
            content_height
        }
        .min(
            area.height
                .saturating_sub(input_height)
                .saturating_sub(command_height)
                .saturating_sub(status_height),
        );
        let layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(transcript_height),
                Constraint::Length(input_height),
                Constraint::Length(command_height),
                Constraint::Length(status_height),
                Constraint::Min(0),
            ])
            .split(area);

        let viewport_height = layout[0].height;
        self.max_scroll = usize::from(content_height)
            .saturating_sub(layout[0].height as usize)
            .min(u16::MAX as usize) as u16;
        self.scroll = if self.follow {
            self.max_scroll
        } else {
            self.scroll.min(self.max_scroll)
        };
        let transcript = transcript.scroll((self.scroll, 0));
        frame.render_widget(transcript, layout[0]);

        if self.show_startup && self.entries.is_empty() {
            let margin = u16::from(layout[0].width > 4);
            let top = u16::from(layout[0].height > 7);
            let banner_area = Rect::new(
                layout[0].x + margin,
                layout[0].y + top,
                layout[0].width.saturating_sub(margin * 2),
                7.min(layout[0].height.saturating_sub(top)),
            );
            let permissions = if self.allow_all { "allow all" } else { "ask" };
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled(">_  ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            "K CASTLE ",
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("(v{})", env!("CARGO_PKG_VERSION")),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ]),
                    Line::from(""),
                    Line::from(vec![
                        Span::styled("model:       ", Style::default().fg(Color::DarkGray)),
                        Span::raw(self.model.clone()),
                    ]),
                    Line::from(vec![
                        Span::styled("directory:   ", Style::default().fg(Color::DarkGray)),
                        Span::raw(self.cwd.display().to_string()),
                    ]),
                    Line::from(vec![
                        Span::styled("permissions: ", Style::default().fg(Color::DarkGray)),
                        Span::styled(permissions, Style::default().fg(ACCENT)),
                    ]),
                ])
                .block(Block::default().borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
                banner_area,
            );
        }

        self.tool_rows = tool_lines
            .into_iter()
            .filter_map(|(line, entry)| {
                let row = line as u16;
                (row >= self.scroll && row < self.scroll.saturating_add(viewport_height))
                    .then(|| (layout[0].y + row - self.scroll, entry))
            })
            .collect();

        let input_surface = layout[1];
        frame.render_widget(
            Block::default().style(Style::default().bg(USER_BACKGROUND)),
            input_surface,
        );
        let input_line = Rect::new(
            input_surface.x.saturating_add(2),
            input_surface
                .y
                .saturating_add(u16::from(input_surface.height >= 3)),
            input_surface.width.saturating_sub(2),
            1.min(input_surface.height),
        );
        frame.render_widget(
            Paragraph::new(Span::styled(
                "›",
                Style::default().fg(if running { Color::DarkGray } else { ACCENT }),
            )),
            Rect::new(input_surface.x, input_line.y, 1, input_line.height),
        );
        self.input.set_block(Block::default());
        let command_query = match &self.modal {
            Some(Modal::Commands { query, .. }) => Some(format!("/{query}")),
            _ => None,
        };
        if let Some(query) = &command_query {
            frame.render_widget(Paragraph::new(query.as_str()), input_line);
        } else {
            frame.render_widget(&self.input, input_line);
        }

        if !compact {
            frame.render_widget(Paragraph::new(self.status_line(running)), layout[3]);
        }

        match &self.modal {
            Some(Modal::Commands {
                items,
                query,
                selected,
            }) => {
                render_commands(frame, items, query, *selected, layout[2]);
                let query_width = Span::raw(command_query.as_deref().unwrap_or("/")).width();
                frame.set_cursor_position((
                    input_line.x
                        + query_width.min(input_line.width.saturating_sub(1) as usize) as u16,
                    input_line.y,
                ));
            }
            Some(modal) => render_modal(frame, modal, centered(area, 72, 60)),
            None => {
                let cursor = {
                    let buffer = frame.buffer_mut();
                    input_line.rows().find_map(|row| {
                        row.columns().find(|position| {
                            buffer[*position].modifier.contains(Modifier::REVERSED)
                        })
                    })
                };
                if let Some(position) = cursor {
                    frame.set_cursor_position(position);
                }
            }
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
        self.show_startup = false;
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
        self.show_startup = false;
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
                    KeyCode::Backspace if query.is_empty() => Some(UiAction::None),
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
            return (Vec::new(), Vec::new());
        }
        let mut lines = Vec::new();
        let mut tool_lines = Vec::new();
        for (entry_index, entry) in self.entries.iter().enumerate() {
            match entry {
                Entry::User(text) => push_user(&mut lines, text, width),
                Entry::Assistant(text) if !text.is_empty() => {
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
                    let label = format!("{mark} {name}");
                    let summary = tool_summary(
                        name,
                        arguments,
                        width.saturating_sub(Span::raw(&label).width() + 2),
                    );
                    let mut spans = vec![Span::styled(label, Style::default().fg(color))];
                    if !summary.is_empty() {
                        spans.push(Span::styled(
                            format!("  {summary}"),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    let mut line = Line::from(spans);
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
                    if !matches!(self.entries.get(entry_index + 1), Some(Entry::Tool { .. })) {
                        lines.push(Line::from(""));
                    }
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

fn push_user(lines: &mut Vec<Line<'static>>, text: &str, width: usize) {
    let style = Style::default()
        .fg(Color::White)
        .bg(USER_BACKGROUND)
        .add_modifier(Modifier::BOLD);
    let inner_width = width.saturating_sub(4).max(1);
    lines.push(Line::from(""));
    lines.push(Line::from(" ".repeat(width)).style(style));
    for source in text.split('\n') {
        let mut row = String::new();
        let mut row_width = 0;
        for character in source.chars() {
            let character_width = Span::raw(character.to_string()).width();
            if row_width + character_width > inner_width && !row.is_empty() {
                push_user_row(lines, &row, row_width, width, style);
                row.clear();
                row_width = 0;
            }
            row.push(character);
            row_width += character_width;
        }
        push_user_row(lines, &row, row_width, width, style);
    }
    lines.push(Line::from(" ".repeat(width)).style(style));
    lines.push(Line::from(""));
}

fn push_user_row(
    lines: &mut Vec<Line<'static>>,
    row: &str,
    row_width: usize,
    width: usize,
    style: Style,
) {
    lines.push(
        Line::from(format!(
            "  {row}{}",
            " ".repeat(width.saturating_sub(row_width + 2))
        ))
        .style(style),
    );
}

fn compact(value: &str, limit: usize) -> String {
    let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if Span::raw(&value).width() <= limit {
        return value;
    }
    if limit == 0 {
        return String::new();
    }

    let mut result = String::new();
    let mut width = 0;
    for character in value.chars() {
        let character_width = Span::raw(character.to_string()).width();
        if width + character_width + 1 > limit {
            break;
        }
        result.push(character);
        width += character_width;
    }
    result.push('…');
    result
}

fn tool_summary(name: &str, arguments: &str, limit: usize) -> String {
    let summary = if name == "shell" {
        serde_json::from_str::<serde_json::Value>(arguments)
            .ok()
            .and_then(|value| value.get("command")?.as_str().map(str::to_owned))
            .unwrap_or_else(|| arguments.to_owned())
    } else {
        arguments.to_owned()
    };
    compact(&summary, limit)
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
        } => render_commands(frame, items, query, *selected, area),
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

fn render_commands(
    frame: &mut Frame<'_>,
    items: &[CommandItem],
    query: &str,
    selected: usize,
    area: Rect,
) {
    let rows = filtered_commands(items, query)
        .into_iter()
        .map(|item| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{:<14}", item.command),
                    Style::default().fg(Color::White),
                ),
                Span::styled(&item.description, Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect::<Vec<_>>();
    let mut state = ListState::default().with_selected((!rows.is_empty()).then_some(selected));
    frame.render_stateful_widget(
        List::new(rows).highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)),
        area,
        &mut state,
    );
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
    use kcastle_agent::{AgentEvent, Model, SessionInfo};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Position;
    use ratatui::style::Modifier;

    use super::{App, Entry, Modal, USER_BACKGROUND, UiAction, filtered_commands, session_label};

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

        app.handle_key(key(KeyCode::Char('/')), false);
        app.handle_key(key(KeyCode::Backspace), false);
        assert!(app.modal.is_none());
    }

    #[test]
    fn command_palette_opens_below_input_and_moves_it_up() {
        let mut app = app();
        app.entries
            .extend((0..10).map(|line| Entry::Assistant(format!("line {line}"))));
        let mut terminal = Terminal::new(TestBackend::new(40, 16)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let prompt_row = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .position(|cell| cell.symbol() == "›")
            .unwrap()
            / 40;

        app.handle_key(key(KeyCode::Char('/')), false);
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rows = terminal
            .backend()
            .buffer()
            .content
            .chunks(40)
            .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
            .collect::<Vec<_>>();
        let moved_prompt_row = rows.iter().position(|row| row.contains('›')).unwrap();
        let command_row = rows.iter().position(|row| row.contains("/model")).unwrap();

        assert!(moved_prompt_row < prompt_row);
        assert!(rows[moved_prompt_row].contains("› /"));
        assert!(command_row > moved_prompt_row);
    }

    #[test]
    fn startup_banner_is_ephemeral_and_transcript_has_no_border() {
        let mut app = app();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains(&format!("(v{})", env!("CARGO_PKG_VERSION"))));
        assert_eq!(terminal.backend().buffer()[(0, 0)].symbol(), " ");

        app.push_user("hello".into());
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(!rendered.contains(env!("CARGO_PKG_VERSION")));
        assert!(matches!(app.entries.as_slice(), [Entry::User(value)] if value == "hello"));
    }

    #[test]
    fn borderless_input_follows_short_transcript_and_keeps_real_cursor() {
        let mut app = app();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();

        assert!(terminal.backend().cursor_visible());
        assert_eq!(terminal.backend().cursor_position(), Position::new(2, 10));
        assert_eq!(terminal.backend().buffer()[(0, 10)].symbol(), "›");
        assert_eq!(terminal.backend().buffer()[(0, 9)].bg, USER_BACKGROUND);
        assert_eq!(terminal.backend().buffer()[(0, 9)].symbol(), " ");
        assert_eq!(terminal.backend().buffer()[(0, 11)].symbol(), " ");

        app.input.insert_str("中文");
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        assert_eq!(terminal.backend().cursor_position(), Position::new(6, 10));
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
    fn tool_rows_are_semantic_compact_and_grouped() {
        let mut app = app();
        app.entries.push(Entry::Tool {
            call_id: "one".into(),
            name: "shell".into(),
            arguments: r#"{"command":"printf a-very-long-command"}"#.into(),
            output: Some("exit_code=0".into()),
            failed: false,
            expanded: false,
        });
        app.entries.push(Entry::Tool {
            call_id: "two".into(),
            name: "read_file".into(),
            arguments: r#"{"path":"README.md"}"#.into(),
            output: Some("contents".into()),
            failed: false,
            expanded: false,
        });

        let (lines, tools) = app.transcript_lines(24);
        let first = lines[0]
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();

        assert!(first.contains("printf"));
        assert!(!first.contains("command"));
        assert!(lines[..2].iter().all(|line| line.width() <= 24));
        assert_eq!(tools, vec![(0, 0), (1, 1)]);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn assistant_message_has_no_role_marker() {
        let mut app = app();
        app.entries.push(Entry::Assistant("answer".into()));
        let (lines, _) = app.transcript_lines(20);
        let rendered = lines
            .iter()
            .flat_map(|line| &line.spans)
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(rendered, "answer");
    }

    #[test]
    fn user_message_is_a_full_width_highlight_without_a_marker() {
        let mut app = app();
        app.push_user("hi".into());
        let (lines, _) = app.transcript_lines(20);
        let highlighted = lines
            .iter()
            .filter(|line| line.style.bg == Some(USER_BACKGROUND))
            .collect::<Vec<_>>();
        assert_eq!(highlighted.len(), 3);
        assert!(highlighted.iter().all(|line| line.width() == 20));
        assert_eq!(lines.last().and_then(|line| line.style.bg), None);
        let rendered = lines
            .iter()
            .flat_map(|line| &line.spans)
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(rendered.contains("hi"));
        assert!(!rendered.contains("YOU"));
        assert!(!rendered.contains('›'));

        let mut terminal = Terminal::new(TestBackend::new(20, 12)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        assert_eq!(terminal.backend().buffer()[(0, 3)].bg, USER_BACKGROUND);
        assert_ne!(terminal.backend().buffer()[(0, 4)].bg, USER_BACKGROUND);
        assert_eq!(terminal.backend().buffer()[(0, 5)].bg, USER_BACKGROUND);
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
        app.apply_event(AgentEvent::TextDelta("more".into()));
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

    #[test]
    fn follow_reaches_tail_of_wrapped_transcript_content() {
        let mut app = app();
        app.entries.push(Entry::Tool {
            call_id: "call".into(),
            name: "shell".into(),
            arguments: "{}".into(),
            output: Some(format!("{} TAIL", "wrapped words ".repeat(20))),
            failed: false,
            expanded: true,
        });
        let mut terminal = Terminal::new(TestBackend::new(40, 8)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("TAIL"));
    }

    #[test]
    fn resize_to_small_terminal_keeps_tail_visible() {
        let mut app = app();
        app.entries
            .push(Entry::Assistant(format!("{}TAIL", "line\n".repeat(20))));
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        terminal.backend_mut().resize(30, 6);
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("TAIL"));
        assert_eq!(terminal.backend().buffer()[(0, 4)].symbol(), "›");
        assert_eq!(terminal.backend().buffer()[(0, 5)].bg, USER_BACKGROUND);
    }
}
