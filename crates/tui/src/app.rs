use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
use kcastle_agent::{AgentEvent, Model, ReasoningEffort, SessionInfo, TranscriptItem};
use markdown_stream::{Alignment, BlockKind, Event, InlineStyle};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui_textarea::{CursorMove, TextArea, WrapMode};
use time::macros::format_description;
use time::{OffsetDateTime, UtcOffset};

const ACCENT: Color = Color::Rgb(196, 181, 253);
const TOOL_ACCENT: Color = Color::Rgb(251, 146, 60);
const INPUT_ACCENT: Color = Color::Rgb(96, 165, 250);
const OUTPUT_ACCENT: Color = Color::Rgb(74, 222, 128);
const USER_BACKGROUND: Color = Color::Rgb(65, 69, 77);
const MAX_COMPOSER_HEIGHT: u16 = 8;
const STREAMING_TAIL_LINES: usize = 4;

#[derive(Debug)]
enum Entry {
    User(String),
    Assistant {
        text: String,
        committed_lines: usize,
    },
    Tools(Vec<usize>),
    Notice(String),
}

#[derive(Debug)]
struct ToolRecord {
    call_id: String,
    name: String,
    arguments: String,
    output: Option<String>,
    failed: Option<bool>,
    started_at: Option<OffsetDateTime>,
    started_instant: Option<Instant>,
    duration: Option<Duration>,
}

#[derive(Debug)]
enum Modal {
    Approval {
        call_id: String,
        name: String,
        arguments: String,
        cwd: PathBuf,
        scroll: u16,
        max_scroll: u16,
    },
    Sessions {
        sessions: Vec<SessionInfo>,
        selected: usize,
        current_path: PathBuf,
        allow_delete: bool,
        confirm_delete: bool,
        notice: Option<String>,
    },
    Models {
        names: Vec<String>,
        selected: usize,
    },
    Reasoning {
        model: usize,
        name: String,
        choices: Vec<String>,
        selected: usize,
    },
    Commands {
        items: Vec<CommandItem>,
        query: String,
        selected: usize,
    },
    AllowAll {
        pending_call_id: Option<String>,
        return_to: Option<Box<Modal>>,
    },
    Tools {
        selected: usize,
        expanded: bool,
        scroll: u16,
        max_scroll: u16,
        page_size: u16,
    },
}

#[derive(Debug, Clone)]
struct CommandItem {
    command: String,
    description: String,
    prefill: bool,
}

struct TranscriptCache {
    width: usize,
    entries: usize,
    lines: Vec<Line<'static>>,
}

pub enum UiAction {
    None,
    Submit(String),
    Approve {
        call_id: String,
        allow: bool,
    },
    Resume(PathBuf),
    DeleteSession {
        session: SessionInfo,
        selected: usize,
    },
    SelectModel(usize),
    SelectReasoning {
        model: usize,
        effort: usize,
    },
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
    model: String,
    cwd: PathBuf,
    context_window: usize,
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
    cached_tokens: u32,
    modal: Option<Modal>,
    transcript_cache: Option<TranscriptCache>,
    streaming_entry: Option<usize>,
    tools: Vec<ToolRecord>,
    active_tool_row: Option<usize>,
    input_history: Vec<String>,
    history_index: Option<usize>,
    history_draft: Option<String>,
    allow_all: bool,
    show_startup: bool,
    should_exit: bool,
}

impl App {
    pub fn new(
        model: &Model,
        transcript: Vec<TranscriptItem>,
        cwd: &Path,
        usage: Option<(u32, u32, u32, u32)>,
        allow_all: bool,
    ) -> Self {
        let mut app = Self {
            entries: Vec::new(),
            input: new_textarea(),
            model: format!("{} · {}", model.name(), model.model()),
            cwd: cwd.to_path_buf(),
            context_window: model.context_window(),
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_tokens: 0,
            modal: None,
            transcript_cache: None,
            streaming_entry: None,
            tools: Vec::new(),
            active_tool_row: None,
            input_history: Vec::new(),
            history_index: None,
            history_draft: None,
            allow_all,
            show_startup: false,
            should_exit: false,
        };
        app.set_identity(model);
        app.set_usage(usage);
        app.load_transcript(transcript);
        app.show_startup = true;
        app
    }

    pub fn render(&mut self, frame: &mut Frame<'_>, running: bool) {
        let area = frame.area();
        let compact = area.height < 8;
        let status_height = u16::from(!compact);
        let command_height = match &self.modal {
            Some(Modal::Commands { items, query, .. }) => {
                filtered_commands(items, query).len().min(u16::MAX as usize) as u16
            }
            _ => 0,
        }
        .min(area.height.saturating_sub(status_height));
        let input_width = area.width.saturating_sub(2).max(1);
        let input_text = self.input.lines().join("\n");
        let wrapped_input_height = Paragraph::new(input_text)
            .wrap(Wrap { trim: false })
            .line_count(input_width)
            .max(1)
            .min(u16::MAX as usize) as u16;
        let input_height = wrapped_input_height
            .saturating_add(2)
            .min(MAX_COMPOSER_HEIGHT)
            .min(
                area.height
                    .saturating_sub(status_height)
                    .saturating_sub(command_height),
            );
        let content_width = area.width;
        let lines = self.transcript_lines(content_width as usize);
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

        let tail_scroll = content_height.saturating_sub(layout[0].height);
        frame.render_widget(transcript.scroll((tail_scroll, 0)), layout[0]);

        if self.show_startup && self.entries.is_empty() {
            let margin = u16::from(layout[0].width > 4);
            let top = u16::from(layout[0].height > 7);
            let banner_area = Rect::new(
                layout[0].x + margin,
                layout[0].y + top,
                layout[0].width.saturating_sub(margin * 2),
                7.min(layout[0].height.saturating_sub(top)),
            );
            frame.render_widget(
                Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled(">_  ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            "Kcastle ",
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
                ])
                .block(Block::default().borders(Borders::ALL))
                .wrap(Wrap { trim: false }),
                banner_area,
            );
        }

        let input_surface = layout[1];
        frame.render_widget(
            Block::default().style(Style::default().bg(USER_BACKGROUND)),
            input_surface,
        );
        let input_line = Rect::new(
            input_surface.x.saturating_add(2),
            input_surface.y.saturating_add(1),
            input_surface.width.saturating_sub(2),
            input_surface.height.saturating_sub(2),
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
            frame.render_widget(Paragraph::new(self.status_line(area.width)), layout[3]);
        }

        match &mut self.modal {
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
            Some(modal) => {
                let modal_area = if matches!(modal, Modal::Tools { .. }) {
                    tool_modal_area(area)
                } else {
                    centered(area, 92, 90)
                };
                render_modal(frame, modal, &self.tools, modal_area);
            }
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
            KeyCode::Enter if !key.modifiers.is_empty() => {
                self.input.insert_newline();
                self.reset_history_navigation();
                UiAction::None
            }
            KeyCode::Up if self.input.cursor().0 == 0 && !self.input_history.is_empty() => {
                self.recall_older_input();
                UiAction::None
            }
            KeyCode::Down
                if self.input.cursor().0 + 1 == self.input.lines().len()
                    && self.history_index.is_some() =>
            {
                self.recall_newer_input();
                UiAction::None
            }
            KeyCode::Enter => {
                let value = self.input.lines().join("\n");
                if value.trim().is_empty() {
                    return UiAction::None;
                }
                if self.input_history.last() != Some(&value) {
                    self.input_history.push(value.clone());
                }
                self.input = new_textarea();
                self.reset_history_navigation();
                UiAction::Submit(value)
            }
            _ => {
                self.input.input(key);
                self.reset_history_navigation();
                UiAction::None
            }
        }
    }

    pub fn paste(&mut self, value: &str) {
        self.input
            .insert_str(value.replace("\r\n", "\n").replace('\r', "\n"));
        self.reset_history_navigation();
    }

    pub fn apply_event(&mut self, event: AgentEvent) -> Option<(String, bool)> {
        match event {
            AgentEvent::SessionEvent(_) => {}
            AgentEvent::RunStarted(input) => self.push_user(input),
            AgentEvent::InputAdmitted { input, .. } => self.push_user(input),
            AgentEvent::ModelStarted(_) => {}
            AgentEvent::ReasoningDelta(delta) => {
                if let Some(Entry::Assistant { text, .. }) = self
                    .streaming_entry
                    .and_then(|index| self.entries.get_mut(index))
                    .filter(|entry| {
                        matches!(entry, Entry::Assistant { text, .. } if text.starts_with("Think · "))
                    })
                {
                    text.push_str(&delta);
                } else {
                    self.active_tool_row = None;
                    self.entries.push(Entry::Assistant {
                        text: format!("Think · {delta}"),
                        committed_lines: 0,
                    });
                    self.streaming_entry = Some(self.entries.len() - 1);
                }
            }
            AgentEvent::TextDelta(delta) => {
                if let Some(Entry::Assistant { text, .. }) = self
                    .streaming_entry
                    .and_then(|index| self.entries.get_mut(index))
                    .filter(|entry| {
                        matches!(entry, Entry::Assistant { text, .. } if !text.starts_with("Think · "))
                    })
                {
                    text.push_str(&delta);
                } else {
                    self.active_tool_row = None;
                    self.entries.push(Entry::Assistant {
                        text: delta,
                        committed_lines: 0,
                    });
                    self.streaming_entry = Some(self.entries.len() - 1);
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
                    cwd: self.cwd.clone(),
                    scroll: 0,
                    max_scroll: 0,
                });
            }
            AgentEvent::ToolStarted(call) => {
                let tool_index = self.tools.len();
                self.tools.push(ToolRecord {
                    call_id: call.call_id,
                    name: call.name,
                    arguments: call.arguments,
                    output: None,
                    failed: None,
                    started_at: Some(OffsetDateTime::now_utc()),
                    started_instant: Some(Instant::now()),
                    duration: None,
                });
                let row = self.active_tool_row.and_then(|index| {
                    matches!(self.entries.get(index), Some(Entry::Tools(_))).then_some(index)
                });
                let row = if let Some(index) = row {
                    if let Entry::Tools(tools) = &mut self.entries[index] {
                        tools.push(tool_index);
                    }
                    index
                } else {
                    self.entries.push(Entry::Tools(vec![tool_index]));
                    self.entries.len() - 1
                };
                self.active_tool_row = Some(row);
                self.streaming_entry = Some(row);
                self.invalidate_transcript();
            }
            AgentEvent::ToolFinished { call, result } => {
                if let Some(ToolRecord {
                    output,
                    failed,
                    started_instant,
                    duration,
                    ..
                }) = self
                    .tools
                    .iter_mut()
                    .rev()
                    .find(|tool| tool.call_id == call.call_id)
                {
                    *duration = started_instant.map(|started| started.elapsed());
                    *started_instant = None;
                    *output = Some(result.output);
                    *failed = Some(result.is_error);
                    self.invalidate_transcript();
                }
            }
            AgentEvent::CompactionStarted { .. } | AgentEvent::CompactionFinished { .. } => {}
            AgentEvent::RunFinished(summary) => {
                self.streaming_entry = None;
                self.active_tool_row = None;
                self.set_usage(summary.usage.as_ref().map(|usage| {
                    (
                        usage.input_tokens,
                        usage.output_tokens,
                        usage.total_tokens,
                        usage.input_tokens_details.cached_tokens,
                    )
                }));
            }
            AgentEvent::RunAborted => {
                self.streaming_entry = None;
                self.active_tool_row = None;
            }
            AgentEvent::RunFailed(error) => {
                self.streaming_entry = None;
                self.active_tool_row = None;
                self.entries.push(Entry::Notice(format!("Error: {error}")));
                self.invalidate_transcript();
            }
        }
        None
    }

    pub fn push_user(&mut self, value: String) {
        self.show_startup = false;
        self.active_tool_row = None;
        self.entries.push(Entry::User(value));
        self.invalidate_transcript();
    }

    pub fn notice(&mut self, value: impl Into<String>) {
        self.entries.push(Entry::Notice(value.into()));
        self.invalidate_transcript();
    }

    pub fn set_identity(&mut self, model: &Model) {
        self.model = format!("{} · {}", model.name(), model.model());
        if let Some(effort) = model.reasoning_effort() {
            self.model
                .push_str(&format!(" · {}", reasoning_effort_value(effort)));
        }
        self.context_window = model.context_window();
    }

    pub fn set_usage_values(&mut self, usage: Option<(u32, u32, u32, u32)>) {
        self.set_usage(usage);
    }

    pub fn load_transcript(&mut self, transcript: Vec<TranscriptItem>) {
        self.show_startup = false;
        self.entries.clear();
        self.streaming_entry = None;
        self.tools.clear();
        self.active_tool_row = None;
        let mut tool_row = None;
        for item in transcript {
            match item {
                TranscriptItem::User(text) => {
                    tool_row = None;
                    self.entries.push(Entry::User(text));
                }
                TranscriptItem::Reasoning(text) => {
                    tool_row = None;
                    self.entries.push(Entry::Assistant {
                        text: format!("Think · {text}"),
                        committed_lines: 0,
                    });
                }
                TranscriptItem::Assistant(text) => {
                    if !text.is_empty() {
                        tool_row = None;
                    }
                    self.entries.push(Entry::Assistant {
                        text,
                        committed_lines: 0,
                    });
                }
                TranscriptItem::ToolCall {
                    call_id,
                    name,
                    arguments,
                } => {
                    let tool_index = self.tools.len();
                    self.tools.push(ToolRecord {
                        call_id,
                        name,
                        arguments,
                        output: None,
                        failed: None,
                        started_at: None,
                        started_instant: None,
                        duration: None,
                    });
                    if let Some(index) = tool_row
                        && let Entry::Tools(tools) = &mut self.entries[index]
                    {
                        tools.push(tool_index);
                    } else {
                        self.entries.push(Entry::Tools(vec![tool_index]));
                        tool_row = Some(self.entries.len() - 1);
                    }
                }
                TranscriptItem::ToolOutput { call_id, output } => {
                    if let Some(tool) = self
                        .tools
                        .iter_mut()
                        .rev()
                        .find(|tool| tool.call_id == call_id)
                    {
                        tool.failed = output_failed(&output);
                        tool.output = Some(output);
                    }
                }
                TranscriptItem::Summary(text) => {
                    tool_row = None;
                    self.entries
                        .push(Entry::Notice(format!("Earlier context: {text}")));
                }
            }
        }
        self.invalidate_transcript();
    }

    pub fn show_sessions(
        &mut self,
        sessions: Vec<SessionInfo>,
        current_path: &Path,
        allow_delete: bool,
        selected: Option<usize>,
        notice: Option<String>,
    ) {
        if sessions.is_empty() {
            self.notice(notice.unwrap_or_else(|| "No saved sessions".into()));
        } else {
            let selected = selected
                .unwrap_or_else(|| {
                    sessions
                        .iter()
                        .position(|session| session.path == current_path)
                        .unwrap_or(0)
                })
                .min(sessions.len() - 1);
            self.modal = Some(Modal::Sessions {
                sessions,
                selected,
                current_path: current_path.to_path_buf(),
                allow_delete,
                confirm_delete: false,
                notice,
            });
        }
    }

    pub fn show_models(&mut self, models: &[Model], selected: usize) {
        self.modal = Some(Modal::Models {
            names: models
                .iter()
                .map(|model| {
                    let effort = model
                        .reasoning_effort()
                        .map(reasoning_effort_value)
                        .unwrap_or("—");
                    format!("{} · {} · {effort}", model.name(), model.model())
                })
                .collect(),
            selected,
        });
    }

    pub fn show_reasoning(&mut self, model: &Model, model_index: usize) {
        let efforts = model.reasoning_efforts();
        let selected = model
            .reasoning_effort()
            .and_then(|current| efforts.iter().position(|effort| effort == current))
            .unwrap_or(0);
        self.modal = Some(Modal::Reasoning {
            model: model_index,
            name: model.model().into(),
            choices: efforts
                .iter()
                .map(|effort| {
                    format!(
                        "{:<10} {}",
                        reasoning_effort_label(effort),
                        reasoning_effort_description(effort)
                    )
                })
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
                return_to: None,
            });
            None
        }
    }

    pub fn set_permission_mode(&mut self, allow_all: bool) {
        self.allow_all = allow_all;
    }

    pub fn prefill(&mut self, value: &str) {
        self.input = new_textarea();
        self.input.insert_str(value);
        self.reset_history_navigation();
    }

    pub fn request_exit(&mut self) {
        self.should_exit = true;
    }

    pub fn should_exit(&self) -> bool {
        self.should_exit
    }

    pub fn take_committed_lines(&mut self, width: usize) -> Vec<Line<'static>> {
        let stable = self
            .entries
            .iter()
            .enumerate()
            .position(|(index, _)| self.streaming_entry == Some(index))
            .unwrap_or(self.entries.len());
        let mut lines = render_entries(&self.entries, &self.tools, 0..stable, None, width);
        if stable > 0 {
            self.entries.drain(..stable);
            self.streaming_entry = self.streaming_entry.map(|index| index - stable);
            self.active_tool_row = self
                .active_tool_row
                .and_then(|index| (index >= stable).then_some(index - stable));
            self.invalidate_transcript();
        }
        let Some(index) = self.streaming_entry else {
            return lines;
        };
        let Entry::Assistant {
            text,
            committed_lines,
        } = &self.entries[index]
        else {
            return lines;
        };
        let rendered = render_assistant(text, true, width);
        let commit_to = rendered
            .len()
            .saturating_sub(STREAMING_TAIL_LINES.saturating_add(1));
        if commit_to > *committed_lines {
            lines.extend(rendered[*committed_lines..commit_to].iter().cloned());
            if let Entry::Assistant {
                committed_lines, ..
            } = &mut self.entries[index]
            {
                *committed_lines = commit_to;
            }
            self.invalidate_transcript();
        }
        lines
    }

    pub fn show_tools(&mut self) {
        if self.tools.is_empty() {
            self.notice("No tool calls");
        } else {
            self.modal = Some(Modal::Tools {
                selected: 0,
                expanded: false,
                scroll: 0,
                max_scroll: 0,
                page_size: 1,
            });
        }
    }

    pub fn fullscreen_modal_open(&self) -> bool {
        matches!(
            self.modal,
            Some(Modal::Sessions { .. } | Modal::Tools { .. })
        )
    }

    pub fn handle_mouse(&mut self, mouse: MouseEvent) {
        let Some(Modal::Tools {
            expanded: true,
            scroll,
            max_scroll,
            ..
        }) = self.modal.as_mut()
        else {
            return;
        };
        match mouse.kind {
            MouseEventKind::ScrollUp => *scroll = scroll.saturating_sub(3),
            MouseEventKind::ScrollDown => {
                *scroll = scroll.saturating_add(3).min(*max_scroll);
            }
            _ => {}
        }
    }

    fn set_usage(&mut self, usage: Option<(u32, u32, u32, u32)>) {
        (
            self.input_tokens,
            self.output_tokens,
            self.total_tokens,
            self.cached_tokens,
        ) = usage.unwrap_or((0, 0, 0, 0));
    }

    fn status_line(&self, width: u16) -> Line<'static> {
        let context_used = precise_percentage(self.total_tokens as usize, self.context_window);
        let cache_hit = percentage(self.cached_tokens as usize, self.input_tokens as usize);
        let base = format!(" {}", self.model);
        let compact = format!("{base} · ctx {context_used:.2}% · cache {cache_hit}%");
        let full = format!(
            "{base} · in {} · out {} · ctx {context_used:.2}% · cache {cache_hit}%",
            format_tokens(self.input_tokens),
            format_tokens(self.output_tokens),
        );
        let width = usize::from(width);
        let text = if Span::raw(&full).width() <= width {
            full
        } else if Span::raw(&compact).width() <= width {
            compact
        } else {
            base
        };
        Line::from(Span::styled(text, Style::default().fg(ACCENT)))
    }

    fn reset_history_navigation(&mut self) {
        self.history_index = None;
        self.history_draft = None;
    }

    fn recall_older_input(&mut self) {
        let index = match self.history_index {
            Some(index) => index.saturating_sub(1),
            None => {
                self.history_draft = Some(self.input.lines().join("\n"));
                self.input_history.len() - 1
            }
        };
        self.history_index = Some(index);
        self.replace_input(self.input_history[index].clone());
    }

    fn recall_newer_input(&mut self) {
        let Some(index) = self.history_index else {
            return;
        };
        if index + 1 < self.input_history.len() {
            self.history_index = Some(index + 1);
            self.replace_input(self.input_history[index + 1].clone());
        } else {
            let draft = self.history_draft.take().unwrap_or_default();
            self.replace_input(draft);
            self.history_index = None;
        }
    }

    fn replace_input(&mut self, value: String) {
        self.input = new_textarea();
        self.input.insert_str(&value);
        self.input.move_cursor(CursorMove::Bottom);
        self.input.move_cursor(CursorMove::End);
    }

    fn handle_modal_key(&mut self, key: KeyEvent) -> Option<UiAction> {
        let modal = self.modal.as_mut()?;
        let action = match modal {
            Modal::Approval {
                call_id,
                name,
                arguments,
                cwd,
                scroll,
                max_scroll,
                ..
            } => match key.code {
                KeyCode::PageUp => {
                    *scroll = scroll.saturating_sub(10);
                    None
                }
                KeyCode::PageDown => {
                    *scroll = scroll.saturating_add(10).min(*max_scroll);
                    None
                }
                KeyCode::Home => {
                    *scroll = 0;
                    None
                }
                KeyCode::End => {
                    *scroll = *max_scroll;
                    None
                }
                KeyCode::Char('y') | KeyCode::Enter => Some(UiAction::Approve {
                    call_id: call_id.clone(),
                    allow: true,
                }),
                KeyCode::Char('a') => {
                    let return_to = Modal::Approval {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        cwd: cwd.clone(),
                        scroll: *scroll,
                        max_scroll: *max_scroll,
                    };
                    self.modal = Some(Modal::AllowAll {
                        pending_call_id: Some(call_id.clone()),
                        return_to: Some(Box::new(return_to)),
                    });
                    return Some(UiAction::None);
                }
                KeyCode::Char('n') | KeyCode::Esc => Some(UiAction::Approve {
                    call_id: call_id.clone(),
                    allow: false,
                }),
                _ => None,
            },
            Modal::Tools {
                selected,
                expanded,
                scroll,
                max_scroll,
                page_size,
            } => {
                if *expanded {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
                        KeyCode::Down | KeyCode::Char('j') => {
                            *scroll = scroll.saturating_add(1).min(*max_scroll);
                        }
                        KeyCode::PageUp => *scroll = scroll.saturating_sub(*page_size),
                        KeyCode::PageDown | KeyCode::Char(' ') => {
                            *scroll = scroll.saturating_add(*page_size).min(*max_scroll);
                        }
                        KeyCode::Home => *scroll = 0,
                        KeyCode::End => *scroll = *max_scroll,
                        KeyCode::Left | KeyCode::Backspace | KeyCode::Esc | KeyCode::Enter => {
                            *expanded = false;
                            *scroll = 0;
                        }
                        _ => {}
                    }
                    None
                } else {
                    match key.code {
                        KeyCode::Tab | KeyCode::Down => {
                            *selected = (*selected + 1) % self.tools.len();
                            None
                        }
                        KeyCode::BackTab | KeyCode::Up => {
                            *selected = selected
                                .checked_sub(1)
                                .unwrap_or(self.tools.len().saturating_sub(1));
                            None
                        }
                        KeyCode::Enter => {
                            *expanded = true;
                            *scroll = 0;
                            None
                        }
                        KeyCode::Esc => Some(UiAction::None),
                        _ => None,
                    }
                }
            }
            Modal::Sessions {
                sessions,
                selected,
                current_path,
                allow_delete,
                confirm_delete,
                notice,
            } => {
                if *confirm_delete {
                    match key.code {
                        KeyCode::Char('y' | 'Y') | KeyCode::Enter => {
                            Some(UiAction::DeleteSession {
                                session: sessions[*selected].clone(),
                                selected: *selected,
                            })
                        }
                        KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                            *confirm_delete = false;
                            None
                        }
                        _ => None,
                    }
                } else {
                    match key.code {
                        KeyCode::Tab | KeyCode::Down | KeyCode::Right => {
                            *selected = (*selected + 1) % sessions.len();
                            *notice = None;
                            None
                        }
                        KeyCode::BackTab | KeyCode::Up | KeyCode::Left => {
                            *selected = selected
                                .checked_sub(1)
                                .unwrap_or(sessions.len().saturating_sub(1));
                            *notice = None;
                            None
                        }
                        KeyCode::Char('d' | 'D') if *allow_delete => {
                            if sessions[*selected].path == *current_path {
                                *notice = Some("Current session cannot be deleted".into());
                            } else {
                                *confirm_delete = true;
                                *notice = None;
                            }
                            None
                        }
                        KeyCode::Enter => Some(UiAction::Resume(sessions[*selected].path.clone())),
                        KeyCode::Esc => Some(UiAction::None),
                        _ => None,
                    }
                }
            }
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
            Modal::Reasoning {
                model,
                choices,
                selected,
                ..
            } => match key.code {
                KeyCode::Up => {
                    *selected = selected.saturating_sub(1);
                    None
                }
                KeyCode::Down => {
                    *selected = (*selected + 1).min(choices.len() - 1);
                    None
                }
                KeyCode::Enter => Some(UiAction::SelectReasoning {
                    model: *model,
                    effort: *selected,
                }),
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
            Modal::AllowAll {
                pending_call_id,
                return_to,
            } => match key.code {
                KeyCode::Char('y') | KeyCode::Enter => Some(UiAction::SetPermissions {
                    allow_all: true,
                    pending_call_id: pending_call_id.clone(),
                }),
                KeyCode::Char('n') | KeyCode::Esc if return_to.is_some() => {
                    self.modal = return_to.take().map(|modal| *modal);
                    return Some(UiAction::None);
                }
                KeyCode::Char('n') | KeyCode::Esc => Some(UiAction::None),
                _ => None,
            },
        };
        if action.is_some() {
            self.modal = None;
        }
        action.or(Some(UiAction::None))
    }

    fn transcript_lines(&mut self, width: usize) -> Vec<Line<'static>> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let stable_entries = self.streaming_entry.unwrap_or(self.entries.len());
        if self
            .transcript_cache
            .as_ref()
            .is_none_or(|cache| cache.width != width || cache.entries != stable_entries)
        {
            let lines = render_entries(&self.entries, &self.tools, 0..stable_entries, None, width);
            self.transcript_cache = Some(TranscriptCache {
                width,
                entries: stable_entries,
                lines,
            });
        }
        let cache = self.transcript_cache.as_ref().expect("cache initialized");
        let mut lines = cache.lines.clone();
        if stable_entries < self.entries.len() {
            let tail = render_entries(
                &self.entries,
                &self.tools,
                stable_entries..self.entries.len(),
                self.streaming_entry,
                width,
            );
            lines.extend(tail);
        }
        lines
    }

    fn invalidate_transcript(&mut self) {
        self.transcript_cache = None;
    }
}

fn render_entries(
    entries: &[Entry],
    tools: &[ToolRecord],
    range: std::ops::Range<usize>,
    plain_assistant: Option<usize>,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for entry_index in range {
        let entry = &entries[entry_index];
        match entry {
            Entry::User(text) => push_user(&mut lines, text, width),
            Entry::Assistant {
                text,
                committed_lines,
            } if !text.is_empty() => {
                lines.extend(
                    render_assistant(text, plain_assistant == Some(entry_index), width)
                        .into_iter()
                        .skip(*committed_lines),
                );
            }
            Entry::Tools(indices) => {
                let tools = indices
                    .iter()
                    .map(|&index| (tool_icon(&tools[index].name), tool_color(&tools[index])))
                    .collect::<Vec<_>>();
                lines.push(render_tool_line(
                    &tools,
                    width.min(u16::MAX as usize) as u16,
                ));
                lines.push(Line::default());
            }
            Entry::Notice(text) => {
                lines.push(Line::from(Span::styled(
                    text.clone(),
                    Style::default().fg(Color::Yellow),
                )));
                lines.push(Line::from(""));
            }
            Entry::Assistant { .. } => {}
        }
    }
    lines
}

fn render_assistant(text: &str, streaming: bool, width: usize) -> Vec<Line<'static>> {
    let mut lines = if streaming {
        if let Some(newline) = text.rfind('\n') {
            let mut lines = render_markdown(&text[..newline], width).lines;
            lines.extend(render_plain(&text[newline + 1..], width));
            lines
        } else {
            render_plain(text, width)
        }
    } else {
        render_markdown(text, width).lines
    };
    lines.push(Line::from(""));
    lines
}

fn render_plain(text: &str, width: usize) -> Vec<Line<'static>> {
    markdown_ratatui::render_with(
        &[
            Event::enter(BlockKind::Paragraph),
            Event::Text {
                text: text.to_owned(),
                style: InlineStyle::default(),
                span: Default::default(),
            },
            Event::exit(BlockKind::Paragraph),
        ],
        &markdown_ratatui::Theme::default(),
        width,
    )
    .lines
}

fn render_markdown(text: &str, width: usize) -> Text<'static> {
    let events = markdown_stream::parse(text);
    let theme = markdown_ratatui::Theme::default();
    let mut lines = Vec::new();
    let mut chunk_start = 0;
    let mut index = 0;
    while index < events.len() {
        if matches!(
            events[index],
            Event::EnterBlock {
                block: BlockKind::Table,
                ..
            }
        ) && let Some(end) = events[index + 1..].iter().position(|event| {
            matches!(
                event,
                Event::ExitBlock {
                    block: BlockKind::Table,
                    ..
                }
            )
        }) {
            append_markdown_lines(
                &mut lines,
                markdown_ratatui::render_with(&events[chunk_start..index], &theme, width).lines,
            );
            let end = index + end + 1;
            append_markdown_lines(
                &mut lines,
                render_table(&events[index..=end], width, &theme),
            );
            index = end + 1;
            chunk_start = index;
        } else {
            index += 1;
        }
    }
    append_markdown_lines(
        &mut lines,
        markdown_ratatui::render_with(&events[chunk_start..], &theme, width).lines,
    );
    Text::from(lines)
}

type MarkdownCell = Vec<Event>;

fn append_markdown_lines(lines: &mut Vec<Line<'static>>, rendered: Vec<Line<'static>>) {
    if rendered.is_empty() {
        return;
    }
    if lines.last().is_some_and(|line| !line.spans.is_empty()) {
        lines.push(Line::default());
    }
    lines.extend(rendered);
}

fn table_rows(events: &[Event]) -> Vec<Vec<MarkdownCell>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut cell = None;
    for event in events {
        match event {
            Event::EnterBlock {
                block: BlockKind::TableRow,
                ..
            } => row.clear(),
            Event::EnterBlock {
                block: BlockKind::TableCell,
                ..
            } => cell = Some(Vec::new()),
            Event::Text { .. } => {
                if let Some(cell) = &mut cell {
                    cell.push(event.clone());
                }
            }
            Event::SoftBreak => {
                if let Some(cell) = &mut cell {
                    cell.push(Event::text(" "));
                }
            }
            Event::LineBreak => {
                if let Some(cell) = &mut cell {
                    cell.push(Event::LineBreak);
                }
            }
            Event::ExitBlock {
                block: BlockKind::TableCell,
                ..
            } => row.push(cell.take().unwrap_or_default()),
            Event::ExitBlock {
                block: BlockKind::TableRow,
                ..
            } => rows.push(std::mem::take(&mut row)),
            _ => {}
        }
    }
    rows
}

fn render_table(
    events: &[Event],
    width: usize,
    theme: &markdown_ratatui::Theme,
) -> Vec<Line<'static>> {
    const CELL_PADDING: usize = 1;
    const COLUMN_GAP: usize = 2;

    let rows = table_rows(events);
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    if columns == 0 {
        return Vec::new();
    }
    let alignments = events
        .iter()
        .find_map(|event| match event {
            Event::EnterBlock {
                block: BlockKind::Table,
                data,
                ..
            } => Some(data.alignment.as_slice()),
            _ => None,
        })
        .unwrap_or_default();
    let reserved = columns * CELL_PADDING * 2 + columns.saturating_sub(1) * COLUMN_GAP;
    let column_widths = table_column_widths(&rows, width.saturating_sub(reserved), columns);
    let mut lines = Vec::new();
    for (row_index, row) in rows.iter().enumerate() {
        lines.extend(render_table_row(
            row,
            &column_widths,
            alignments,
            theme,
            row_index == 0,
        ));
        if row_index + 1 < rows.len() {
            let character = if row_index == 0 { '━' } else { '─' };
            lines.push(Line::from(Span::styled(
                column_widths
                    .iter()
                    .map(|width| character.to_string().repeat(width + CELL_PADDING * 2))
                    .collect::<Vec<_>>()
                    .join(&" ".repeat(COLUMN_GAP)),
                theme.muted,
            )));
        }
    }
    lines
}

fn table_column_widths(rows: &[Vec<MarkdownCell>], available: usize, columns: usize) -> Vec<usize> {
    let minimum = if available < columns * 3 { 1 } else { 3 };
    let mut widths = vec![minimum; columns];
    let mut floors = vec![minimum; columns];
    for column in 0..columns {
        let cells = rows.iter().filter_map(|row| row.get(column));
        for cell in cells {
            let text = cell_text(cell);
            let cell_width = Span::raw(&text).width();
            widths[column] = widths[column].max(cell_width);
            let preferred = if cell_width >= 28 {
                16
            } else {
                text.split_whitespace()
                    .map(|token| Span::raw(token).width())
                    .max()
                    .unwrap_or(minimum)
                    .min(16)
            };
            floors[column] = floors[column].max(preferred.min(cell_width));
        }
    }
    shrink_columns(&mut floors, &vec![minimum; columns], available);
    shrink_columns(&mut widths, &floors, available);
    widths
}

fn shrink_columns(widths: &mut [usize], floors: &[usize], available: usize) {
    let mut excess = widths.iter().sum::<usize>().saturating_sub(available);
    while excess > 0 {
        let Some((column, _)) = widths
            .iter()
            .enumerate()
            .filter(|(column, width)| **width > floors[*column])
            .max_by_key(|(column, width)| **width - floors[*column])
        else {
            break;
        };
        widths[column] -= 1;
        excess -= 1;
    }
}

fn render_table_row(
    row: &[MarkdownCell],
    widths: &[usize],
    alignments: &[Alignment],
    theme: &markdown_ratatui::Theme,
    header: bool,
) -> Vec<Line<'static>> {
    const CELL_PADDING: usize = 1;
    const COLUMN_GAP: usize = 2;

    let cells = widths
        .iter()
        .enumerate()
        .map(|(column, width)| {
            wrap_table_cell(
                row.get(column).map_or(&[], Vec::as_slice),
                *width,
                theme,
                header,
            )
        })
        .collect::<Vec<_>>();
    let height = cells.iter().map(Vec::len).max().unwrap_or(1);
    let mut lines = Vec::with_capacity(height);
    for line_index in 0..height {
        let mut spans = Vec::new();
        for (column, width) in widths.iter().enumerate() {
            let mut content = cells[column].get(line_index).cloned().unwrap_or_default();
            let content_width = content.iter().map(|span| span.width()).sum::<usize>();
            let remaining = width.saturating_sub(content_width);
            let (left, right) = match alignments.get(column).copied().unwrap_or(Alignment::None) {
                Alignment::Right => (remaining, 0),
                Alignment::Center => (remaining / 2, remaining - remaining / 2),
                Alignment::None | Alignment::Left => (0, remaining),
            };
            spans.push(Span::raw(" ".repeat(CELL_PADDING + left)));
            spans.append(&mut content);
            spans.push(Span::raw(" ".repeat(right + CELL_PADDING)));
            if column + 1 < widths.len() {
                spans.push(Span::raw(" ".repeat(COLUMN_GAP)));
            }
        }
        lines.push(Line::from(spans));
    }
    lines
}

fn wrap_table_cell(
    cell: &[Event],
    width: usize,
    theme: &markdown_ratatui::Theme,
    header: bool,
) -> Vec<Vec<Span<'static>>> {
    let mut lines = vec![Vec::<Span<'static>>::new()];
    let mut line_width = 0;
    for event in cell {
        if matches!(event, Event::LineBreak) {
            lines.push(Vec::new());
            line_width = 0;
            continue;
        }
        let Event::Text { text, style, .. } = event else {
            continue;
        };
        let mut style = markdown_inline_style(style, theme);
        if header {
            style = style.patch(theme.bold);
        }
        for character in text.chars() {
            let character_width = Span::raw(character.to_string()).width();
            if line_width > 0 && line_width + character_width > width {
                lines.push(Vec::new());
                line_width = 0;
            }
            if line_width == 0 && character.is_whitespace() {
                continue;
            }
            let line = lines.last_mut().expect("table cell has one line");
            if let Some(last) = line.last_mut()
                && last.style == style
            {
                last.content.to_mut().push(character);
            } else {
                line.push(Span::styled(character.to_string(), style));
            }
            line_width += character_width;
        }
    }
    lines
}

fn markdown_inline_style(inline: &InlineStyle, theme: &markdown_ratatui::Theme) -> Style {
    let mut style = Style::default();
    if inline.strong {
        style = style.patch(theme.bold);
    }
    if inline.emphasis {
        style = style.patch(theme.italic);
    }
    if inline.strikethrough {
        style = style.patch(theme.strike);
    }
    if inline.code {
        style = style.patch(theme.code);
    }
    if inline.link.is_some() {
        style = style.patch(theme.link);
    }
    style
}

fn cell_text(cell: &[Event]) -> String {
    cell.iter()
        .filter_map(|event| match event {
            Event::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
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

fn format_tokens(tokens: u32) -> String {
    if tokens >= 1_000_000 {
        format!("{:.1}m", tokens as f64 / 1_000_000.0)
    } else if tokens >= 1_000 {
        format!("{:.1}k", tokens as f64 / 1_000.0)
    } else {
        tokens.to_string()
    }
}

fn percentage(part: usize, total: usize) -> usize {
    part.saturating_mul(100)
        .checked_div(total)
        .unwrap_or(0)
        .min(100)
}

fn precise_percentage(part: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        (part as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
    }
}

fn new_textarea() -> TextArea<'static> {
    let mut input = TextArea::default();
    input.set_cursor_line_style(Style::default());
    input.set_placeholder_text("Message Kcastle…  / for commands");
    input.set_wrap_mode(WrapMode::WordOrGlyph);
    input
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

fn tool_icon(name: &str) -> &'static str {
    match name {
        "shell" => "$",
        "read" | "read_file" => "R",
        "write" | "write_file" => "W",
        "edit" | "edit_file" => "E",
        _ => "•",
    }
}

fn render_tool_line(tools: &[(&'static str, Color)], width: u16) -> Line<'static> {
    let show_guide = width >= 5;
    let mut spans = Vec::new();
    if show_guide {
        spans.push(Span::styled("╰─  ", Style::default().fg(Color::DarkGray)));
    }
    let available = usize::from(width).saturating_sub(if show_guide { 4 } else { 0 });
    let capacity = available.saturating_add(1) / 2;
    let show_ellipsis = tools.len() > capacity && available >= 3;
    let icon_width = available.saturating_sub(usize::from(show_ellipsis) * 2);
    let visible = (icon_width.saturating_add(1) / 2).min(tools.len());
    if show_ellipsis {
        spans.push(Span::styled("… ", Style::default().fg(Color::DarkGray)));
    }
    for (index, (icon, color)) in tools[tools.len() - visible..].iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(" "));
        }
        spans.push(Span::styled(*icon, Style::default().fg(*color)));
    }
    Line::from(spans)
}

fn output_failed(output: &str) -> Option<bool> {
    output
        .lines()
        .next()
        .and_then(|line| line.strip_prefix("exit_code="))
        .and_then(|code| code.parse::<i32>().ok())
        .map(|code| code != 0)
}

fn tool_color(tool: &ToolRecord) -> Color {
    match (tool.output.is_some(), tool.failed) {
        (false, _) => Color::Yellow,
        (true, Some(false)) => Color::Green,
        (true, Some(true)) => Color::Red,
        (true, None) => Color::Gray,
    }
}

fn tool_start_time(tool: &ToolRecord) -> String {
    tool.started_at
        .and_then(|started| {
            local_datetime(started)
                .format(format_description!(
                    "[hour]:[minute]:[second] [offset_hour sign:mandatory]:[offset_minute]"
                ))
                .ok()
        })
        .unwrap_or_else(|| "—".into())
}

fn local_datetime(datetime: OffsetDateTime) -> OffsetDateTime {
    datetime.to_offset(UtcOffset::local_offset_at(datetime).unwrap_or(UtcOffset::UTC))
}

fn tool_duration(tool: &ToolRecord) -> String {
    tool.duration
        .or_else(|| tool.started_instant.map(|started| started.elapsed()))
        .map(|duration| format!("{:.1}s", duration.as_secs_f64()))
        .unwrap_or_else(|| "—".into())
}

fn command_items(running: bool) -> Vec<CommandItem> {
    let mut items = vec![
        CommandItem {
            command: "/session".into(),
            description: "Manage saved sessions".into(),
            prefill: false,
        },
        CommandItem {
            command: "/resume".into(),
            description: "Resume a saved session".into(),
            prefill: false,
        },
        CommandItem {
            command: "/model".into(),
            description: "Select model and reasoning level".into(),
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
            command: "/tool".into(),
            description: "Browse tool calls".into(),
            prefill: false,
        },
        CommandItem {
            command: "/help".into(),
            description: "Show command help".into(),
            prefill: false,
        },
        CommandItem {
            command: "/exit".into(),
            description: "Exit Kcastle".into(),
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
            local_datetime(time)
                .format(format_description!(
                    "[year]-[month]-[day] [hour]:[minute] [offset_hour sign:mandatory]:[offset_minute]"
                ))
                .ok()
        })
        .unwrap_or_else(|| session.created_at.to_string());
    format!("{} · {timestamp}", session.title)
}

fn reasoning_effort_label(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "None",
        ReasoningEffort::Minimal => "Minimal",
        ReasoningEffort::Low => "Low",
        ReasoningEffort::Medium => "Medium",
        ReasoningEffort::High => "High",
        ReasoningEffort::Xhigh => "Extra high",
    }
}

fn reasoning_effort_value(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Minimal => "minimal",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
    }
}

fn reasoning_effort_description(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "Fastest responses without reasoning",
        ReasoningEffort::Minimal => "Minimal reasoning for simple tasks",
        ReasoningEffort::Low => "Fast responses with lighter reasoning",
        ReasoningEffort::Medium => "Balances speed and reasoning depth",
        ReasoningEffort::High => "Greater reasoning depth for complex tasks",
        ReasoningEffort::Xhigh => "Extra reasoning depth for difficult tasks",
    }
}

fn render_modal(frame: &mut Frame<'_>, modal: &mut Modal, tools: &[ToolRecord], area: Rect) {
    frame.render_widget(Clear, area);
    match modal {
        Modal::Approval {
            name,
            arguments,
            cwd,
            scroll,
            max_scroll,
            ..
        } => {
            let details = approval_details(name, arguments, Some(cwd));
            let content = format!("{details}\n\n[y/Enter] allow   [n/Esc] deny   [a] allow all");
            let paragraph = Paragraph::new(content)
                .block(Block::default().borders(Borders::ALL).title(" Permission "))
                .wrap(Wrap { trim: false });
            *max_scroll = paragraph
                .line_count(area.width)
                .saturating_sub(area.height as usize)
                .min(u16::MAX as usize) as u16;
            *scroll = (*scroll).min(*max_scroll);
            frame.render_widget(paragraph.scroll((*scroll, 0)), area);
        }
        Modal::Sessions {
            sessions,
            selected,
            current_path,
            allow_delete,
            confirm_delete,
            notice,
        } => {
            let items = sessions
                .iter()
                .map(|session| {
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            if session.path == *current_path {
                                "● "
                            } else {
                                "  "
                            },
                            Style::default().fg(ACCENT),
                        ),
                        Span::raw(session_label(session)),
                    ]))
                })
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(*selected));
            let selected_session = &sessions[*selected];
            let footer = if *confirm_delete {
                Line::from(vec![
                    Span::styled(
                        " Delete selected session permanently? ",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "Y/Enter confirm · N/Esc cancel ",
                        Style::default().fg(Color::Gray),
                    ),
                ])
            } else if let Some(notice) = notice {
                Line::from(Span::styled(
                    format!(" {notice} "),
                    Style::default().fg(Color::Yellow),
                ))
            } else if selected_session.path == *current_path {
                Line::from(Span::styled(
                    " Current session · Tab/↑↓ move · Esc close ",
                    Style::default().fg(Color::DarkGray),
                ))
            } else if *allow_delete {
                Line::from(Span::styled(
                    " Enter open · D delete · Tab/↑↓ move · Esc close ",
                    Style::default().fg(Color::DarkGray),
                ))
            } else {
                Line::from(Span::styled(
                    " Enter open · Tab/↑↓ move · Esc close ",
                    Style::default().fg(Color::DarkGray),
                ))
            };
            let block = Block::default()
                .borders(Borders::ALL)
                .title(format!(" Sessions · {} ", sessions.len()))
                .title_bottom(footer)
                .border_style(if *confirm_delete {
                    Style::default().fg(Color::Red)
                } else {
                    Style::default()
                });
            frame.render_stateful_widget(
                List::new(items)
                    .block(block)
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
        Modal::Reasoning {
            name,
            choices,
            selected,
            ..
        } => {
            let items = choices
                .iter()
                .map(|choice| ListItem::new(choice.clone()))
                .collect::<Vec<_>>();
            let mut state = ListState::default().with_selected(Some(*selected));
            frame.render_stateful_widget(
                List::new(items)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(format!(" Reasoning level for {name} "))
                            .title_bottom(" Enter confirm · Esc cancel "),
                    )
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
        Modal::Tools {
            selected,
            expanded,
            scroll,
            max_scroll,
            page_size,
        } => {
            let tool = &tools[tools.len() - 1 - *selected];
            if *expanded {
                let mut content = vec![Line::from(Span::styled(
                    "INPUT",
                    Style::default()
                        .fg(INPUT_ACCENT)
                        .add_modifier(Modifier::BOLD),
                ))];
                content.extend(
                    approval_details(&tool.name, &tool.arguments, None)
                        .lines()
                        .map(|line| {
                            let style = if matches!(
                                line,
                                "Command" | "Directory" | "Timeout" | "Tool" | "Arguments"
                            ) {
                                Style::default().fg(INPUT_ACCENT)
                            } else {
                                Style::default()
                            };
                            Line::styled(line.to_owned(), style)
                        }),
                );
                content.push(Line::default());
                content.push(Line::from(Span::styled(
                    "OUTPUT",
                    Style::default()
                        .fg(OUTPUT_ACCENT)
                        .add_modifier(Modifier::BOLD),
                )));
                content.extend(
                    tool.output
                        .as_deref()
                        .unwrap_or("  Running…")
                        .lines()
                        .map(|line| Line::raw(line.to_owned())),
                );
                let paragraph = Paragraph::new(Text::from(content)).wrap(Wrap { trim: false });
                *page_size = area.height.saturating_sub(2).max(1);
                *max_scroll = paragraph
                    .line_count(area.width.saturating_sub(2).max(1))
                    .saturating_sub(*page_size as usize)
                    .min(u16::MAX as usize) as u16;
                *scroll = (*scroll).min(*max_scroll);
                let title = Line::from(vec![
                    Span::styled(
                        " TOOL ",
                        Style::default()
                            .fg(TOOL_ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("· {} ", tool.name),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(
                            "· {}/{} · ↑↓/wheel scroll · Space page · ←/Esc/Enter back ",
                            scroll.saturating_add(1),
                            max_scroll.saturating_add(1),
                        ),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]);
                frame.render_widget(
                    paragraph
                        .block(Block::default().borders(Borders::ALL).title(title))
                        .scroll((*scroll, 0)),
                    area,
                );
            } else {
                let items = tools
                    .iter()
                    .rev()
                    .map(|tool| {
                        ListItem::new(Line::from(vec![
                            Span::styled("● ", Style::default().fg(tool_color(tool))),
                            Span::styled(
                                format!("{}  {}  ", tool_start_time(tool), tool_duration(tool)),
                                Style::default().fg(Color::DarkGray),
                            ),
                            Span::styled(
                                format!("{} ", tool.name),
                                Style::default().add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                tool_summary(&tool.name, &tool.arguments, 80),
                                Style::default().fg(Color::Gray),
                            ),
                        ]))
                    })
                    .collect::<Vec<_>>();
                let mut state = ListState::default().with_selected(Some(*selected));
                frame.render_stateful_widget(
                    List::new(items)
                        .block(
                            Block::default()
                                .borders(Borders::ALL)
                                .title(" Tools · Tab move · Enter details "),
                        )
                        .highlight_style(Style::default().bg(Color::DarkGray).fg(Color::White)),
                    area,
                    &mut state,
                );
            }
        }
    }
}

fn approval_details(name: &str, arguments: &str, cwd: Option<&PathBuf>) -> String {
    if name == "shell"
        && let Ok(value) = serde_json::from_str::<serde_json::Value>(arguments)
        && let Some(command) = value.get("command").and_then(serde_json::Value::as_str)
    {
        let timeout = value
            .get("timeout")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(120.0);
        let cwd = cwd
            .map(|path| format!("\n\nDirectory\n  {}", path.display()))
            .unwrap_or_default();
        return format!("Command\n  {command}{cwd}\n\nTimeout\n  {timeout}s");
    }
    format!("Tool\n  {name}\n\nArguments\n  {arguments}")
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

fn tool_modal_area(area: Rect) -> Rect {
    let horizontal = if area.width >= 20 {
        2
    } else {
        u16::from(area.width >= 4)
    };
    let vertical = if area.height >= 16 {
        2
    } else {
        u16::from(area.height >= 8)
    };
    Rect::new(
        area.x.saturating_add(horizontal),
        area.y.saturating_add(vertical),
        area.width.saturating_sub(horizontal.saturating_mul(2)),
        area.height.saturating_sub(vertical.saturating_mul(2)),
    )
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseEvent, MouseEventKind};
    use kcastle_agent::{
        AgentEvent, Model, ReasoningEffort, RunSummary, SessionInfo, TranscriptItem,
    };
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::{Position, Rect};
    use ratatui::style::{Color, Modifier};
    use ratatui::text::Span;
    use time::macros::format_description;
    use time::{OffsetDateTime, UtcOffset};

    use super::{
        App, Entry, INPUT_ACCENT, Modal, OUTPUT_ACCENT, STREAMING_TAIL_LINES, ToolRecord,
        USER_BACKGROUND, UiAction, filtered_commands, local_datetime, render_markdown,
        render_tool_line, session_label, tool_icon, tool_modal_area, tool_start_time,
    };

    fn app() -> App {
        App::new(
            &Model::new("test", "key", "http://localhost", "test-model", 10_000),
            Vec::new(),
            std::path::Path::new("/work"),
            Some((40, 1, 41, 38)),
            false,
        )
    }

    fn assistant(text: impl Into<String>) -> Entry {
        Entry::Assistant {
            text: text.into(),
            committed_lines: 0,
        }
    }

    fn start_tool(app: &mut App, call_id: &str) {
        let arguments = serde_json::json!({ "command": "true" }).to_string();
        app.apply_event(AgentEvent::ToolStarted(
            serde_json::from_value(serde_json::json!({
                "arguments": arguments,
                "call_id": call_id,
                "name": "shell"
            }))
            .unwrap(),
        ));
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
        assert_eq!(filtered_commands(items, query).len(), 8);
        assert!(items.iter().any(|item| item.command == "/resume"));

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
    fn model_selection_continues_to_reasoning_selection() {
        const EFFORTS: &[ReasoningEffort] = &[
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ];
        let model = Model::new("test", "key", "http://localhost", "test-model", 10_000)
            .with_reasoning(EFFORTS, ReasoningEffort::Medium);
        let mut app = App::new(
            &model,
            Vec::new(),
            std::path::Path::new("/work"),
            None,
            false,
        );

        app.show_models(std::slice::from_ref(&model), 0);
        assert!(matches!(
            app.handle_key(key(KeyCode::Enter), false),
            UiAction::SelectModel(0)
        ));

        app.show_reasoning(&model, 0);
        let Modal::Reasoning {
            choices, selected, ..
        } = app.modal.as_ref().unwrap()
        else {
            panic!("reasoning picker not opened")
        };
        assert_eq!(*selected, 1);
        assert!(choices[1].starts_with("Medium"));
        app.handle_key(key(KeyCode::Down), false);
        assert!(matches!(
            app.handle_key(key(KeyCode::Enter), false),
            UiAction::SelectReasoning {
                model: 0,
                effort: 2
            }
        ));
    }

    #[test]
    fn command_palette_opens_below_input_and_moves_it_up() {
        let mut app = app();
        app.entries
            .extend((0..10).map(|line| assistant(format!("line {line}"))));
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
    fn session_manager_navigates_and_confirms_before_delete() {
        let mut app = app();
        let current = SessionInfo::legacy(PathBuf::from("current.jsonl"), "Current", 2);
        let saved = SessionInfo::legacy(PathBuf::from("saved.jsonl"), "Saved", 1);
        app.show_sessions(
            vec![saved.clone(), current.clone()],
            &current.path,
            true,
            None,
            None,
        );

        assert!(matches!(
            app.handle_key(key(KeyCode::Char('D')), false),
            UiAction::None
        ));
        assert!(matches!(
            app.modal,
            Some(Modal::Sessions {
                confirm_delete: false,
                notice: Some(_),
                ..
            })
        ));
        app.handle_key(key(KeyCode::Tab), false);
        app.handle_key(key(KeyCode::Up), false);
        app.handle_key(key(KeyCode::Down), false);
        app.handle_key(key(KeyCode::Char('d')), false);
        assert!(matches!(
            app.modal,
            Some(Modal::Sessions {
                selected: 0,
                confirm_delete: true,
                ..
            })
        ));

        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("Delete selected session permanently?"));

        app.handle_key(key(KeyCode::Esc), false);
        assert!(matches!(
            app.modal,
            Some(Modal::Sessions {
                confirm_delete: false,
                ..
            })
        ));
        app.handle_key(key(KeyCode::Char('D')), false);
        assert!(matches!(
            app.handle_key(key(KeyCode::Enter), false),
            UiAction::DeleteSession { session, selected: 0 }
                if session == saved
        ));
        assert!(app.modal.is_none());
    }

    #[test]
    fn resume_session_list_does_not_delete() {
        let mut app = app();
        let saved = SessionInfo::legacy(PathBuf::from("saved.jsonl"), "Saved", 1);
        app.show_sessions(
            vec![saved],
            &PathBuf::from("current.jsonl"),
            false,
            None,
            None,
        );

        assert!(matches!(
            app.handle_key(key(KeyCode::Char('d')), false),
            UiAction::None
        ));
        assert!(matches!(
            app.modal,
            Some(Modal::Sessions {
                allow_delete: false,
                confirm_delete: false,
                ..
            })
        ));
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
        assert!(!rendered.contains("permissions"));
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
    fn cancelling_allow_all_returns_to_the_pending_approval() {
        let mut app = app();
        app.apply_event(AgentEvent::ApprovalRequired(
            serde_json::from_value(serde_json::json!({
                "arguments": r#"{"command":"true"}"#,
                "call_id": "call-1",
                "name": "shell"
            }))
            .unwrap(),
        ));

        app.handle_key(key(KeyCode::Char('a')), true);
        assert!(matches!(app.modal, Some(Modal::AllowAll { .. })));

        app.handle_key(key(KeyCode::Esc), true);
        assert!(matches!(
            app.modal,
            Some(Modal::Approval { ref call_id, .. }) if call_id == "call-1"
        ));
    }

    #[test]
    fn markdown_is_rendered_without_tool_output_in_history() {
        let mut app = app();
        app.entries.push(assistant("**bold**"));
        let lines = app.transcript_lines(80);
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content == "bold" && span.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn wide_markdown_table_wraps_cells_and_keeps_columns_aligned() {
        let text = "| Project | Detail | State |\n| --- | --- | --- |\n| AG2 | A long agent framework description | Active |";
        let rendered = render_markdown(text, 32);
        let rows = rendered
            .lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        let separator = rows.iter().position(|row| row.contains('━')).unwrap();
        let header = rows[..separator]
            .join("")
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let body = rows[separator + 1..]
            .join("")
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(header.contains("Project"));
        assert!(header.contains("Detail"));
        assert!(header.contains("State"));
        assert!(body.contains("frameworkdescription"));
        assert!(rows.iter().all(|row| Span::raw(row).width() == 32));
        assert!(!rows.iter().any(|row| row.contains("Project:")));
    }

    #[test]
    fn long_approval_details_can_scroll_to_the_end() {
        let mut app = app();
        app.modal = Some(Modal::Approval {
            arguments: format!("{}VISIBLE_TAIL", "hidden line\n".repeat(30)),
            call_id: "call".into(),
            name: "shell".into(),
            cwd: PathBuf::from("/work"),
            scroll: 0,
            max_scroll: 0,
        });
        let mut terminal = Terminal::new(TestBackend::new(40, 12)).unwrap();
        terminal.draw(|frame| app.render(frame, true)).unwrap();
        app.handle_key(key(KeyCode::End), true);
        terminal.draw(|frame| app.render(frame, true)).unwrap();

        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("VISIBLE_TAIL"));
    }

    #[test]
    fn tool_line_and_modal_keep_details_out_of_history() {
        let mut app = app();
        app.tools.push(ToolRecord {
            call_id: "one".into(),
            name: "shell".into(),
            arguments: r#"{"command":"printf a-very-long-command"}"#.into(),
            output: Some("exit_code=0\nFIRST_OUTPUT".into()),
            failed: Some(false),
            started_at: Some(OffsetDateTime::from_unix_timestamp(1).unwrap()),
            started_instant: None,
            duration: Some(Duration::from_millis(400)),
        });
        app.tools.push(ToolRecord {
            call_id: "two".into(),
            name: "shell".into(),
            arguments: r#"{"command":"exit 7"}"#.into(),
            output: Some("exit_code=7\nSECOND_OUTPUT".into()),
            failed: Some(true),
            started_at: Some(OffsetDateTime::from_unix_timestamp(2).unwrap()),
            started_instant: None,
            duration: Some(Duration::from_millis(800)),
        });
        app.entries.push(Entry::Tools(vec![0, 1]));
        let history = app.transcript_lines(80);
        assert_eq!(
            history
                .iter()
                .flat_map(|line| &line.spans)
                .filter(|span| span.content == "$")
                .count(),
            2
        );
        assert!(
            history
                .iter()
                .all(|line| !line.to_string().contains("OUTPUT"))
        );
        assert_eq!(
            history.last().map(|line| line.to_string()).as_deref(),
            Some("")
        );

        let second_start = tool_start_time(&app.tools[1]);
        app.show_tools();
        let mut terminal = Terminal::new(TestBackend::new(80, 24)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.find("exit 7").unwrap() < rendered.find("printf a-very").unwrap());
        assert!(rendered.contains(&second_start));
        assert!(rendered.contains("0.8s"));

        app.handle_key(key(KeyCode::Tab), false);
        app.handle_key(key(KeyCode::Enter), false);
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("FIRST_OUTPUT"));
        assert!(rendered.contains("INPUT"));
        assert!(rendered.contains("OUTPUT"));
        assert!(!rendered.contains("Started"));
        assert!(!rendered.contains("Duration"));
        assert!(!rendered.contains(&tool_start_time(&app.tools[0])));
        assert_eq!(
            terminal.backend().buffer()[Position::new(3, 3)].fg,
            INPUT_ACCENT
        );
        assert_eq!(
            terminal.backend().buffer()[Position::new(3, 10)].fg,
            OUTPUT_ACCENT
        );
    }

    #[test]
    fn tool_modal_uses_most_of_large_and_small_terminals() {
        assert_eq!(
            tool_modal_area(Rect::new(0, 0, 120, 40)),
            Rect::new(2, 2, 116, 36)
        );
        assert_eq!(
            tool_modal_area(Rect::new(0, 0, 12, 6)),
            Rect::new(1, 0, 10, 6)
        );
    }

    #[test]
    fn tool_details_scroll_with_arrows_and_reach_the_end() {
        let mut app = app();
        app.tools.push(ToolRecord {
            call_id: "long".into(),
            name: "shell".into(),
            arguments: r#"{"command":"printf long"}"#.into(),
            output: Some(format!("{}VISIBLE_END", "output line\n".repeat(40))),
            failed: Some(false),
            started_at: Some(OffsetDateTime::from_unix_timestamp(1).unwrap()),
            started_instant: None,
            duration: Some(Duration::from_secs(1)),
        });
        app.show_tools();
        app.handle_key(key(KeyCode::Enter), false);
        let mut terminal = Terminal::new(TestBackend::new(40, 10)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();

        app.handle_key(key(KeyCode::Down), false);
        assert!(matches!(app.modal, Some(Modal::Tools { scroll: 1, .. })));
        app.handle_key(key(KeyCode::Char(' ')), false);
        assert!(matches!(
            app.modal,
            Some(Modal::Tools {
                scroll,
                page_size,
                ..
            }) if scroll == page_size.saturating_add(1)
        ));
        app.handle_mouse(MouseEvent {
            kind: MouseEventKind::ScrollDown,
            column: 0,
            row: 0,
            modifiers: KeyModifiers::NONE,
        });
        assert!(matches!(
            app.modal,
            Some(Modal::Tools {
                scroll,
                page_size,
                ..
            }) if scroll == page_size.saturating_add(4)
        ));
        app.handle_key(key(KeyCode::End), false);
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains("VISIBLE_END"));
        app.handle_key(key(KeyCode::Left), false);
        assert!(matches!(
            app.modal,
            Some(Modal::Tools {
                expanded: false,
                ..
            })
        ));
    }

    #[test]
    fn text_output_starts_a_new_inline_tool_row() {
        let mut app = app();
        app.apply_event(AgentEvent::ModelStarted(1));
        app.apply_event(AgentEvent::TextDelta("before".into()));
        start_tool(&mut app, "one");
        start_tool(&mut app, "two");
        app.apply_event(AgentEvent::ModelStarted(2));
        start_tool(&mut app, "three");
        app.apply_event(AgentEvent::ModelStarted(3));
        app.apply_event(AgentEvent::TextDelta("after".into()));
        start_tool(&mut app, "four");

        assert!(matches!(
            app.entries.as_slice(),
            [
                Entry::Assistant { text: before, .. },
                Entry::Tools(first),
                Entry::Assistant { text: after, .. },
                Entry::Tools(second),
            ] if before == "before" && first == &[0, 1, 2] && after == "after" && second == &[3]
        ));
        app.show_tools();
        assert!(matches!(app.modal, Some(Modal::Tools { .. })));
    }

    #[test]
    fn reasoning_stream_does_not_merge_the_final_answer() {
        let mut app = app();
        app.apply_event(AgentEvent::ReasoningDelta("inspect ".into()));
        app.apply_event(AgentEvent::ReasoningDelta("workspace".into()));
        app.apply_event(AgentEvent::TextDelta("done".into()));

        assert!(matches!(
            app.entries.as_slice(),
            [
                Entry::Assistant { text: reasoning, .. },
                Entry::Assistant { text: answer, .. }
            ] if reasoning == "Think · inspect workspace" && answer == "done"
        ));
    }

    #[test]
    fn resumed_tools_restore_their_inline_row() {
        let mut app = app();
        app.load_transcript(vec![
            TranscriptItem::ToolCall {
                call_id: "old".into(),
                name: "shell".into(),
                arguments: r#"{"command":"true"}"#.into(),
            },
            TranscriptItem::ToolOutput {
                call_id: "old".into(),
                output: "exit_code=0".into(),
            },
        ]);

        assert_eq!(app.tools.len(), 1);
        assert!(app.tools[0].started_at.is_none());
        assert!(app.tools[0].duration.is_none());
        assert_eq!(app.tools[0].failed, Some(false));
        assert!(
            app.transcript_lines(80)
                .iter()
                .flat_map(|line| &line.spans)
                .any(|span| span.content == "$" && span.style.fg == Some(Color::Green))
        );
    }

    #[test]
    fn inline_tool_row_is_left_aligned_and_never_exceeds_the_viewport() {
        let tools = [
            (tool_icon("shell"), Color::Green),
            (tool_icon("read_file"), Color::Yellow),
            (tool_icon("write"), Color::Red),
            (tool_icon("edit_file"), Color::Green),
            (tool_icon("other"), Color::Gray),
        ];
        assert_eq!(render_tool_line(&tools, 80).to_string(), "╰─  $ R W E •");
        for width in 0..16 {
            let line = render_tool_line(&tools, width);
            assert!(Span::raw(line.to_string()).width() <= usize::from(width));
        }
    }

    #[test]
    fn assistant_message_has_no_role_marker() {
        let mut app = app();
        app.entries.push(assistant("answer"));
        let lines = app.transcript_lines(20);
        let rendered = lines
            .iter()
            .flat_map(|line| &line.spans)
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert_eq!(rendered, "answer");
    }

    #[test]
    fn steering_does_not_split_the_streaming_assistant() {
        let mut app = app();
        app.apply_event(AgentEvent::ModelStarted(1));
        app.apply_event(AgentEvent::TextDelta("**hel".into()));
        app.push_user("steer".into());
        app.apply_event(AgentEvent::TextDelta("lo**".into()));

        assert!(matches!(
            app.entries.as_slice(),
            [Entry::Assistant { text: answer, .. }, Entry::User(message)]
                if answer == "**hello**" && message == "steer"
        ));
        app.apply_event(AgentEvent::RunFinished(RunSummary {
            output: "hello".into(),
            response_id: "response".into(),
            usage: None,
        }));
        let lines = app.transcript_lines(40);
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content == "hello" && span.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn streaming_markdown_formats_only_complete_lines() {
        let mut app = app();
        app.apply_event(AgentEvent::ModelStarted(1));
        app.apply_event(AgentEvent::TextDelta("**done**\n**partial".into()));
        let lines = app.transcript_lines(40);
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content == "done" && span.style.add_modifier.contains(Modifier::BOLD)
        }));
        assert!(lines.iter().flat_map(|line| &line.spans).any(|span| {
            span.content == "**partial" && !span.style.add_modifier.contains(Modifier::BOLD)
        }));
    }

    #[test]
    fn user_message_is_a_full_width_highlight_without_a_marker() {
        let mut app = app();
        app.push_user("hi".into());
        let lines = app.transcript_lines(20);
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
    }

    #[test]
    fn status_contains_usage_but_not_permissions() {
        let mut app = app();
        app.set_permission_mode(true);
        let status = app
            .status_line(80)
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        assert!(!status.contains("permissions"));
        assert!(!status.contains("allow all"));
        assert!(app.entries.is_empty());
        assert_eq!(
            status,
            " test · test-model · in 40 · out 1 · ctx 0.41% · cache 95%"
        );
        assert_eq!(app.status_line(40).to_string(), " test · test-model");
        let created_at = local_datetime(OffsetDateTime::UNIX_EPOCH)
            .format(format_description!(
                "[year]-[month]-[day] [hour]:[minute] [offset_hour sign:mandatory]:[offset_minute]"
            ))
            .unwrap();
        assert_eq!(
            session_label(&SessionInfo::legacy(PathBuf::new(), "saved", 0)),
            format!("saved · {created_at}")
        );
    }

    #[test]
    fn datetime_display_uses_the_system_offset_at_that_instant() {
        let datetime = OffsetDateTime::from_unix_timestamp(1_700_000_000).unwrap();
        let expected = UtcOffset::local_offset_at(datetime).unwrap_or(UtcOffset::UTC);
        let localized = local_datetime(datetime);

        assert_eq!(localized.offset(), expected);
        assert_eq!(localized.unix_timestamp(), datetime.unix_timestamp());
    }

    #[test]
    fn stable_history_is_drained_for_terminal_scrollback() {
        let mut app = app();
        app.notice("old");
        app.apply_event(AgentEvent::ModelStarted(1));
        app.apply_event(AgentEvent::TextDelta("live".into()));
        let history = app.take_committed_lines(40);
        assert!(history.iter().any(|line| line.to_string().contains("old")));
        assert!(matches!(
            app.entries.as_slice(),
            [Entry::Assistant { text, .. }] if text == "live"
        ));
    }

    #[test]
    fn streaming_output_drains_wrapped_prefix_into_terminal_history() {
        let mut app = app();
        app.apply_event(AgentEvent::ModelStarted(1));
        app.apply_event(AgentEvent::TextDelta("word ".repeat(80)));

        let history = app.take_committed_lines(20);

        assert!(!history.is_empty());
        assert!(app.transcript_lines(20).len() <= STREAMING_TAIL_LINES + 1);
        assert!(matches!(
            app.entries.as_slice(),
            [Entry::Assistant {
                committed_lines,
                ..
            }] if *committed_lines > 0
        ));
    }

    #[test]
    fn composer_supports_modified_enter_paste_and_history() {
        let mut app = app();
        app.paste("one\r\ntwo");
        app.handle_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT), false);
        app.handle_key(key(KeyCode::Char('x')), false);
        assert_eq!(app.input.lines(), ["one", "two", "x"]);
        assert!(matches!(
            app.handle_key(key(KeyCode::Enter), false),
            UiAction::Submit(_)
        ));
        app.handle_key(key(KeyCode::Up), false);
        assert_eq!(app.input.lines(), ["one", "two", "x"]);
    }

    #[test]
    fn small_viewport_keeps_tool_status_line_visible() {
        let mut app = app();
        app.tools.push(ToolRecord {
            call_id: "call".into(),
            name: "shell".into(),
            arguments: "{}".into(),
            output: None,
            failed: None,
            started_at: None,
            started_instant: Some(Instant::now()),
            duration: None,
        });
        app.entries.push(Entry::Tools(vec![0]));
        let mut terminal = Terminal::new(TestBackend::new(40, 8)).unwrap();
        terminal.draw(|frame| app.render(frame, false)).unwrap();
        let rendered = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(rendered.contains('$'));
    }

    #[test]
    fn resize_to_small_terminal_keeps_tail_visible() {
        let mut app = app();
        app.entries
            .push(assistant(format!("{}TAIL", "line\n".repeat(20))));
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
