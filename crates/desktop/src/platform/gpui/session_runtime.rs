use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use gpui::{Context, Window};
use kcastle_agent::{
    Agent, AgentEvent, Model, ReasoningEffort, RunControl, Session, SessionConfig, SessionInfo,
    ToolResult,
};

use crate::application::{MAX_EVENTS_PER_FRAME, StreamBatch, is_frame_stream_event};
use crate::domain::{
    ApprovalState, ConversationAction, ConversationState, Message, Role, RunId, UsageSnapshot,
    next_message_id, reduce_conversation, reindex_messages,
};
use crate::platform::gpui::arm_next_frame;
use crate::settings::EnterBehavior;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SessionRuntimeStatus {
    Idle,
    Creating,
    Configuring,
    Running,
    Failed(String),
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRuntimeSnapshot {
    pub(crate) session: SessionInfo,
    pub(crate) conversation: ConversationState,
    pub(crate) status: SessionRuntimeStatus,
    pub(crate) approval: Option<ApprovalState>,
    pub(crate) started_at: Option<Instant>,
    pub(crate) allow_all_tools: bool,
    pub(crate) config: SessionConfig,
    pub(crate) active_run: Option<RunId>,
    pub(crate) completed_runs: u64,
    pub(crate) transcript_updates: u64,
}

/// The complete GPUI-owned execution boundary for one session.
///
/// Agent events never leave this entity. Observers receive only immutable snapshots, so
/// selecting another session cannot reroute or interrupt this runtime.
pub(crate) struct SessionRuntime {
    session: SessionInfo,
    project_id: String,
    sessions_dir: PathBuf,
    agent: Option<Agent>,
    control: Option<RunControl>,
    active_run: Option<RunId>,
    next_run: RunId,
    completed_runs: u64,
    transcript_updates: u64,
    conversation: ConversationState,
    approval: Option<ApprovalState>,
    status: SessionRuntimeStatus,
    started_at: Option<Instant>,
    allow_all_tools: bool,
    config: SessionConfig,
    tool_schemas: HashMap<String, String>,
}

impl SessionRuntime {
    pub(crate) fn new(
        agent: Agent,
        project_id: String,
        sessions_dir: PathBuf,
        conversation: ConversationState,
        config: SessionConfig,
    ) -> Self {
        let session = agent.session_info().clone();
        let tool_schemas = agent
            .tool_schemas()
            .into_iter()
            .filter_map(|schema| {
                let value = serde_json::to_value(schema).ok()?;
                let function = value.get("function").unwrap_or(&value);
                let name = function.get("name")?.as_str()?.to_owned();
                let display = serde_json::to_string_pretty(function).ok()?;
                Some((name, display))
            })
            .collect();
        Self {
            session,
            project_id,
            sessions_dir,
            agent: Some(agent),
            control: None,
            active_run: None,
            next_run: RunId::default(),
            completed_runs: 0,
            transcript_updates: 0,
            conversation,
            approval: None,
            status: SessionRuntimeStatus::Idle,
            started_at: None,
            allow_all_tools: config.allow_all_tools,
            config,
            tool_schemas,
        }
    }

    pub(crate) fn snapshot(&self) -> SessionRuntimeSnapshot {
        SessionRuntimeSnapshot {
            session: self.session.clone(),
            conversation: self.conversation.clone(),
            status: self.status.clone(),
            approval: self.approval.clone(),
            started_at: self.started_at,
            allow_all_tools: self.allow_all_tools,
            config: self.config.clone(),
            active_run: self.active_run,
            completed_runs: self.completed_runs,
            transcript_updates: self.transcript_updates,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        matches!(
            self.status,
            SessionRuntimeStatus::Creating
                | SessionRuntimeStatus::Configuring
                | SessionRuntimeStatus::Running
        )
    }

    pub(crate) fn set_allow_all_tools(&mut self, allow: bool, cx: &mut Context<Self>) -> bool {
        if allow == self.allow_all_tools {
            return true;
        }
        let mut config = self.config.clone();
        config.allow_all_tools = allow;
        self.update_config(config, None, cx)
    }

    pub(crate) fn set_model(
        &mut self,
        model_id: String,
        model: Model,
        cx: &mut Context<Self>,
    ) -> bool {
        let mut config = self.config.clone();
        config.model_id = Some(model_id);
        config.reasoning_effort = model.reasoning_effort().map(reasoning_key);
        self.update_config(config, Some(model), cx)
    }

    pub(crate) fn set_reasoning_effort(
        &mut self,
        effort: ReasoningEffort,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(mut model) = self.agent.as_ref().map(|agent| agent.model().clone()) else {
            return false;
        };
        model.set_reasoning_effort(effort.clone());
        let mut config = self.config.clone();
        config.reasoning_effort = Some(reasoning_key(&effort));
        self.update_config(config, Some(model), cx)
    }

    fn update_config(
        &mut self,
        config: SessionConfig,
        model: Option<Model>,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.is_active() {
            return false;
        }
        if config == self.config {
            if let (Some(agent), Some(model)) = (&mut self.agent, model) {
                agent.set_model(model);
                cx.notify();
                return true;
            }
            return false;
        }
        if self.session.path.as_os_str().is_empty() {
            if let (Some(agent), Some(model)) = (&mut self.agent, model) {
                agent.set_model(model);
            }
            self.allow_all_tools = config.allow_all_tools;
            self.config = config;
            cx.notify();
            return true;
        }
        let Some(mut agent) = self.agent.take() else {
            return false;
        };
        self.status = SessionRuntimeStatus::Configuring;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = agent.persist_session_config(&config).await;
            let _ = this.update(cx, |runtime, cx| {
                match result {
                    Ok(()) => {
                        if let Some(model) = model {
                            agent.set_model(model);
                        }
                        runtime.allow_all_tools = config.allow_all_tools;
                        runtime.config = config;
                        runtime.status = SessionRuntimeStatus::Idle;
                    }
                    Err(error) => {
                        runtime.status = SessionRuntimeStatus::Failed(error.to_string());
                        runtime.notice(format!("Could not save session configuration: {error}"));
                    }
                }
                agent.release_session_writer();
                runtime.agent = Some(agent);
                cx.notify();
            });
        })
        .detach();
        true
    }

    pub(crate) fn rename(&mut self, title: String, cx: &mut Context<Self>) -> bool {
        if self.is_active() || self.session.path.as_os_str().is_empty() {
            return false;
        }
        let Some(mut agent) = self.agent.take() else {
            return false;
        };
        cx.spawn(async move |this, cx| {
            let result = agent.rename_session(&title).await;
            let _ = this.update(cx, |runtime, cx| {
                match result {
                    Ok(()) => {
                        runtime.session = agent.session_info().clone();
                        runtime.conversation.title = runtime.session.title.clone();
                        runtime.status = SessionRuntimeStatus::Idle;
                    }
                    Err(error) => {
                        runtime.status = SessionRuntimeStatus::Failed(error.to_string());
                        runtime.notice(format!("Could not rename session: {error}"));
                    }
                }
                agent.release_session_writer();
                runtime.agent = Some(agent);
                cx.notify();
            });
        })
        .detach();
        true
    }

    pub(crate) fn submit(
        &mut self,
        input: String,
        behavior: EnterBehavior,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if input.trim().is_empty() || matches!(self.status, SessionRuntimeStatus::Creating) {
            return;
        }
        if let Some(control) = &self.control {
            let result = match behavior {
                EnterBehavior::Steer => control.steer(input),
                EnterBehavior::Queue => control.queue(input),
            };
            if let Err(error) = result {
                self.notice(error.to_string());
            }
            cx.notify();
            return;
        }

        if self.session.path.as_os_str().is_empty() {
            self.create_and_start(input, window, cx);
        } else {
            self.start(input, window, cx);
        }
    }

    fn create_and_start(&mut self, input: String, window: &mut Window, cx: &mut Context<Self>) {
        self.status = SessionRuntimeStatus::Creating;
        let sessions_dir = self.sessions_dir.clone();
        let project_id = self.project_id.clone();
        let runtime_config = self.config.clone();
        let session_id = self.session.id.clone();
        cx.notify();
        cx.spawn_in(window, async move |this, cx| {
            let result = Session::create_in_project_with_id(
                sessions_dir,
                project_id,
                runtime_config,
                session_id,
            )
            .await;
            let _ = cx.update(|window, app| {
                this.update(app, |runtime, cx| match result {
                    Ok(session) => {
                        runtime.session = session.info().clone();
                        if let Some(agent) = &mut runtime.agent {
                            agent.set_session(session);
                        }
                        runtime.start(input, window, cx);
                    }
                    Err(error) => {
                        runtime.status = SessionRuntimeStatus::Failed(error.to_string());
                        runtime.notice(format!("Could not create session: {error}"));
                        cx.notify();
                    }
                })
            });
        })
        .detach();
    }

    fn start(&mut self, input: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(agent) = self.agent.take() else {
            self.status = SessionRuntimeStatus::Failed("Agent is unavailable".into());
            self.notice("Agent is unavailable");
            cx.notify();
            return;
        };
        let run = self.next_run.next();
        self.next_run = run;
        let mut active = agent.start(input);
        self.control = Some(active.control());
        self.active_run = Some(run);
        self.status = SessionRuntimeStatus::Running;
        self.started_at = Some(Instant::now());
        cx.notify();

        cx.spawn_in(window, async move |this, cx| {
            let mut stream_ended = false;
            while !stream_ended {
                let Some(first) = active.next_event().await else {
                    break;
                };
                let collect_frame = is_frame_stream_event(&first);
                let mut batch = StreamBatch::new(first);
                if collect_frame {
                    let (frame_tx, mut frame_rx) = tokio::sync::oneshot::channel();
                    cx.update(|window, _| arm_next_frame(window, frame_tx)).ok();
                    let mut reached_frame = false;
                    let mut reached_structure = false;
                    while batch.len() < MAX_EVENTS_PER_FRAME {
                        tokio::select! {
                            biased;
                            _ = &mut frame_rx => {
                                reached_frame = true;
                                break;
                            }
                            event = active.next_event() => match event {
                                Some(event) => {
                                    let structural = !is_frame_stream_event(&event);
                                    batch.push(event);
                                    if structural {
                                        reached_structure = true;
                                        break;
                                    }
                                }
                                None => {
                                    stream_ended = true;
                                    break;
                                }
                            },
                        }
                    }
                    if !reached_frame && !reached_structure && !stream_ended {
                        let _ = frame_rx.await;
                    }
                }
                let _ = cx.update(|_, app| {
                    this.update(app, |runtime, cx| {
                        if runtime.active_run != Some(run) {
                            return;
                        }
                        for event in batch.into_events() {
                            runtime.apply_event(event);
                        }
                        cx.notify();
                    })
                });
            }
            let result = active.finish().await;
            let _ = cx.update(|_, app| {
                this.update(app, |runtime, cx| {
                    if runtime.active_run != Some(run) {
                        return;
                    }
                    runtime.active_run = None;
                    runtime.control = None;
                    runtime.started_at = None;
                    match result {
                        Ok(mut agent) => {
                            runtime.session = agent.session_info().clone();
                            agent.release_session_writer();
                            runtime.agent = Some(agent);
                            if matches!(runtime.status, SessionRuntimeStatus::Running) {
                                runtime.status = SessionRuntimeStatus::Idle;
                            }
                        }
                        Err(error) => {
                            runtime.status = SessionRuntimeStatus::Failed(error.to_string());
                            runtime.notice(error.to_string());
                        }
                    }
                    cx.notify();
                })
            });
        })
        .detach();
    }

    pub(crate) fn abort(&mut self, cx: &mut Context<Self>) {
        if let Some(control) = &self.control {
            control.abort();
            self.notice("Stopping…");
            cx.notify();
        }
    }

    pub(crate) fn decide(&mut self, call_id: String, allow: bool, cx: &mut Context<Self>) {
        if let Some(control) = &self.control
            && let Err(error) = control.approve(call_id, allow)
        {
            self.notice(error.to_string());
        }
        self.approval = None;
        self.notice(if allow { "Tool allowed" } else { "Tool denied" });
        cx.notify();
    }

    fn apply_event(&mut self, event: AgentEvent) {
        let changes_transcript = matches!(
            &event,
            AgentEvent::ReasoningDelta(_)
                | AgentEvent::TextDelta(_)
                | AgentEvent::ToolStarted(_)
                | AgentEvent::ToolFinished { .. }
                | AgentEvent::RunFinished(_)
                | AgentEvent::RunStarted(_)
                | AgentEvent::InputAdmitted { .. }
        );
        match event {
            AgentEvent::ReasoningDelta(delta) => {
                reduce_conversation(
                    &mut self.conversation,
                    ConversationAction::ReasoningDelta {
                        new_message: message(Role::Reasoning, delta.clone()),
                        delta,
                    },
                );
            }
            AgentEvent::TextDelta(delta) => {
                reduce_conversation(
                    &mut self.conversation,
                    ConversationAction::TextDelta {
                        new_message: message(Role::Assistant, delta.clone()),
                        delta,
                    },
                );
            }
            AgentEvent::ApprovalRequired(call) => {
                if self.allow_all_tools {
                    if let Some(control) = &self.control
                        && let Err(error) = control.approve(call.call_id, true)
                    {
                        self.notice(error.to_string());
                    }
                } else {
                    self.approval = Some(ApprovalState {
                        call_id: call.call_id,
                        name: call.name,
                        arguments: call.arguments,
                    });
                }
            }
            AgentEvent::ToolStarted(call) => {
                let schema = self.tool_schemas.get(&call.name).cloned();
                reduce_conversation(
                    &mut self.conversation,
                    ConversationAction::ToolStarted(Message {
                        key: next_message_id(),
                        revision: 0,
                        role: Role::Tool,
                        tool_call_id: Some(call.call_id),
                        title: Some(call.name),
                        text: String::new(),
                        payload: Some(call.arguments),
                        schema,
                        pending: true,
                        failed: false,
                        expanded: false,
                        rating: None,
                        started_at_ms: Some(now_ms()),
                        duration_ms: None,
                        turn: 0,
                        step: 0,
                        request_id: None,
                        search_text: String::new(),
                    }),
                );
            }
            AgentEvent::ToolFinished { call, result } => self.tool_result(&call.call_id, result),
            AgentEvent::RunFinished(summary) => {
                let usage = summary.usage.map(|usage| UsageSnapshot {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cached_tokens: usage.input_tokens_details.cached_tokens,
                });
                reduce_conversation(
                    &mut self.conversation,
                    ConversationAction::RunFinished {
                        response_id: summary.response_id,
                        usage,
                    },
                );
                self.completed_runs = self.completed_runs.saturating_add(1);
                self.status = SessionRuntimeStatus::Idle;
            }
            AgentEvent::RunFailed(error) => {
                self.finish_reasoning();
                self.status = SessionRuntimeStatus::Failed(error.clone());
                self.notice(error);
            }
            AgentEvent::RunAborted => {
                self.finish_reasoning();
                self.status = SessionRuntimeStatus::Idle;
                self.notice("Stopped");
            }
            AgentEvent::CompactionStarted { .. } => self.notice("Compacting context…"),
            AgentEvent::CompactionFinished { .. } => self.notice("Context compacted"),
            AgentEvent::RunStarted(input) => {
                reduce_conversation(
                    &mut self.conversation,
                    ConversationAction::SubmitUser(message(Role::User, input)),
                );
            }
            AgentEvent::InputAdmitted { input, .. } => {
                reduce_conversation(
                    &mut self.conversation,
                    ConversationAction::SubmitUser(message(Role::User, input)),
                );
            }
            AgentEvent::ModelStarted(_) => {}
        }
        if changes_transcript {
            self.transcript_updates = self.transcript_updates.saturating_add(1);
        }
        reindex_messages(&mut self.conversation.messages);
    }

    #[cfg(test)]
    pub(crate) fn apply_test_event(&mut self, event: AgentEvent, cx: &mut Context<Self>) {
        self.apply_event(event);
        cx.notify();
    }

    fn finish_reasoning(&mut self) {
        reduce_conversation(&mut self.conversation, ConversationAction::FinishReasoning);
    }

    fn tool_result(&mut self, call_id: &str, result: ToolResult) {
        let duration_ms = self
            .conversation
            .messages
            .iter()
            .rev()
            .find(|message| message.tool_call_id.as_deref() == Some(call_id))
            .and_then(|message| {
                message
                    .started_at_ms
                    .map(|started| now_ms().saturating_sub(started))
            });
        reduce_conversation(
            &mut self.conversation,
            ConversationAction::ToolFinished {
                call_id: call_id.to_owned(),
                output: result.output,
                is_error: result.is_error,
                duration_ms,
            },
        );
    }

    fn notice(&mut self, text: impl Into<String>) {
        reduce_conversation(
            &mut self.conversation,
            ConversationAction::AppendNotice(message(Role::Notice, text.into())),
        );
        self.transcript_updates = self.transcript_updates.saturating_add(1);
    }
}

impl Drop for SessionRuntime {
    fn drop(&mut self) {
        if let Some(control) = &self.control {
            control.abort();
        }
    }
}

fn message(role: Role, text: String) -> Message {
    Message {
        key: next_message_id(),
        revision: 0,
        role,
        tool_call_id: None,
        title: None,
        text,
        payload: None,
        schema: None,
        pending: false,
        failed: false,
        expanded: false,
        rating: None,
        started_at_ms: None,
        duration_ms: None,
        turn: 0,
        step: 0,
        request_id: None,
        search_text: String::new(),
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn reasoning_key(effort: &ReasoningEffort) -> String {
    serde_json::to_value(effort)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| format!("{effort:?}").to_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> SessionRuntime {
        let agent = Agent::new(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            Session::memory(),
            ".",
        );
        SessionRuntime::new(
            agent,
            "default".into(),
            PathBuf::from("sessions"),
            ConversationState::default(),
            SessionConfig::default(),
        )
    }

    #[test]
    fn agent_events_mutate_only_the_runtime_that_owns_the_channel() {
        let mut first = runtime();
        let mut second = runtime();

        first.apply_event(AgentEvent::TextDelta("first".into()));
        second.apply_event(AgentEvent::RunFailed("second failed".into()));

        assert_eq!(first.conversation.messages.len(), 1);
        assert_eq!(first.conversation.messages[0].text, "first");
        assert!(matches!(first.status, SessionRuntimeStatus::Idle));
        assert_eq!(second.conversation.messages.len(), 1);
        assert_eq!(second.conversation.messages[0].text, "second failed");
        assert!(matches!(
            second.status,
            SessionRuntimeStatus::Failed(ref message) if message == "second failed"
        ));
    }

    #[test]
    fn local_run_generations_cannot_alias_between_runtimes() {
        let mut first = runtime();
        let mut second = runtime();
        first.active_run = Some(RunId(1));
        second.active_run = Some(RunId(1));

        first.apply_event(AgentEvent::TextDelta("only first".into()));

        assert_eq!(first.conversation.messages.len(), 1);
        assert!(second.conversation.messages.is_empty());
        assert_eq!(second.active_run, Some(RunId(1)));
    }

    #[test]
    fn completion_generation_advances_only_for_a_finished_response() {
        let mut runtime = runtime();

        runtime.apply_event(AgentEvent::RunAborted);
        assert_eq!(runtime.completed_runs, 0);
        assert_eq!(runtime.transcript_updates, 1);

        runtime.apply_event(AgentEvent::RunFinished(kcastle_agent::RunSummary {
            output: "done".into(),
            response_id: "response-1".into(),
            usage: None,
        }));
        assert_eq!(runtime.completed_runs, 1);
        assert_eq!(runtime.transcript_updates, 2);
    }
}
