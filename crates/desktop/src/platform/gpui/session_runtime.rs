use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gpui::{Context, Window};
use kcastle_agent::{
    Agent, AgentEvent, Model, ReasoningEffort, RunControl, Session, SessionConfig, SessionInfo,
    ToolResult,
};

use crate::application::{MAX_EVENTS_PER_FRAME, StreamBatch, is_frame_stream_event};
use crate::domain::{
    ApprovalState, ConversationAction, ConversationState, Message, Role, RunId,
    TrajectoryProjection, UsageSnapshot, next_message_id, reduce_conversation, reindex_messages,
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
    pub(crate) conversation: Arc<ConversationState>,
    pub(crate) trajectory: Arc<TrajectoryProjection>,
    pub(crate) status: SessionRuntimeStatus,
    pub(crate) approval: Option<ApprovalState>,
    pub(crate) started_at: Option<Instant>,
    pub(crate) allow_all_tools: bool,
    pub(crate) config: SessionConfig,
    pub(crate) active_run: Option<RunId>,
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRuntimeObservation {
    pub(crate) session: SessionInfo,
    pub(crate) status: SessionRuntimeStatus,
    pub(crate) approval_needed: bool,
    pub(crate) completed_runs: u64,
    pub(crate) transcript_updates: u64,
}

enum PendingConversationDelta {
    Reasoning(String),
    Text(String),
}

fn merge_pending_delta(
    pending: &mut Option<PendingConversationDelta>,
    next: PendingConversationDelta,
    runtime: &mut SessionRuntime,
    conversation_changed: &mut bool,
) {
    match (pending.as_mut(), &next) {
        (
            Some(PendingConversationDelta::Reasoning(previous)),
            PendingConversationDelta::Reasoning(next),
        )
        | (Some(PendingConversationDelta::Text(previous)), PendingConversationDelta::Text(next)) => {
            previous.push_str(next)
        }
        _ => {
            if let Some(previous) = pending.take() {
                runtime.apply_pending_delta(previous);
                *conversation_changed = true;
            }
            *pending = Some(next);
        }
    }
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
    conversation: Arc<ConversationState>,
    trajectory: Arc<TrajectoryProjection>,
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
        trajectory: TrajectoryProjection,
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
            conversation: Arc::new(conversation),
            trajectory: Arc::new(trajectory),
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
            trajectory: self.trajectory.clone(),
            status: self.status.clone(),
            approval: self.approval.clone(),
            started_at: self.started_at,
            allow_all_tools: self.allow_all_tools,
            config: self.config.clone(),
            active_run: self.active_run,
        }
    }

    pub(crate) fn observation(&self) -> SessionRuntimeObservation {
        SessionRuntimeObservation {
            session: self.session.clone(),
            status: self.status.clone(),
            approval_needed: self.approval.is_some(),
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

    pub(crate) fn set_message_expanded(
        &mut self,
        index: usize,
        role: Role,
        expanded: bool,
        cx: &mut Context<Self>,
    ) {
        let Some(message) = self.conversation.messages.get(index) else {
            return;
        };
        if message.role != role || message.expanded == expanded {
            return;
        }
        reduce_conversation(
            Arc::make_mut(&mut self.conversation),
            ConversationAction::ToggleExpanded { index, role },
        );
        cx.notify();
    }

    pub(crate) fn set_allow_all_tools(&mut self, allow: bool, cx: &mut Context<Self>) -> bool {
        if allow == self.allow_all_tools && allow == self.config.allow_all_tools {
            return true;
        }
        if self.is_active() {
            self.allow_all_tools = allow;
            if allow
                && let Some(approval) = self.approval.take()
                && let Some(control) = &self.control
                && let Err(error) = control.approve(approval.call_id, true)
            {
                self.notice(error.to_string());
            }
            cx.notify();
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
        let previous_allow_all_tools = self.allow_all_tools;
        self.allow_all_tools = config.allow_all_tools;
        let settled_status = self.status.clone();
        self.status = SessionRuntimeStatus::Configuring;
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = agent.persist_session_config(&config).await;
            let _ = this.update(cx, |runtime, cx| {
                let persisted = match result {
                    Ok(()) => {
                        if let Some(model) = model {
                            agent.set_model(model);
                        }
                        runtime.config = config;
                        runtime.status = settled_status;
                        true
                    }
                    Err(error) => {
                        if runtime.allow_all_tools == config.allow_all_tools {
                            runtime.allow_all_tools = previous_allow_all_tools;
                        }
                        runtime.status = SessionRuntimeStatus::Failed(error.to_string());
                        runtime.notice(format!("Could not save session configuration: {error}"));
                        false
                    }
                };
                agent.release_session_writer();
                runtime.agent = Some(agent);
                let allow_all_tools = runtime.allow_all_tools;
                if persisted && runtime.config.allow_all_tools != allow_all_tools {
                    runtime.set_allow_all_tools(allow_all_tools, cx);
                }
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
                        Arc::make_mut(&mut runtime.conversation).title =
                            runtime.session.title.clone();
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
                        runtime.apply_events(batch.into_events());
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
                    let allow_all_tools = runtime.allow_all_tools;
                    if runtime.agent.is_some() && runtime.config.allow_all_tools != allow_all_tools
                    {
                        runtime.set_allow_all_tools(allow_all_tools, cx);
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

    fn apply_events(&mut self, events: Vec<AgentEvent>) {
        let mut pending_delta = None;
        let mut conversation_changed = false;
        let mut transcript_updates = 0_u64;
        for event in events {
            match event {
                AgentEvent::SessionEvent(recorded) => {
                    Arc::make_mut(&mut self.trajectory).apply(&recorded);
                }
                AgentEvent::ReasoningDelta(delta) => {
                    merge_pending_delta(
                        &mut pending_delta,
                        PendingConversationDelta::Reasoning(delta),
                        self,
                        &mut conversation_changed,
                    );
                    transcript_updates = transcript_updates.saturating_add(1);
                }
                AgentEvent::TextDelta(delta) => {
                    merge_pending_delta(
                        &mut pending_delta,
                        PendingConversationDelta::Text(delta),
                        self,
                        &mut conversation_changed,
                    );
                    transcript_updates = transcript_updates.saturating_add(1);
                }
                event => {
                    if let Some(delta) = pending_delta.take() {
                        self.apply_pending_delta(delta);
                        conversation_changed = true;
                    }
                    let (changed, transcript) = self.apply_structural_event(event);
                    conversation_changed |= changed;
                    transcript_updates = transcript_updates.saturating_add(u64::from(transcript));
                }
            }
        }
        if let Some(delta) = pending_delta {
            self.apply_pending_delta(delta);
            conversation_changed = true;
        }
        self.transcript_updates = self.transcript_updates.saturating_add(transcript_updates);
        if conversation_changed {
            reindex_messages(&mut Arc::make_mut(&mut self.conversation).messages);
        }
    }

    fn apply_pending_delta(&mut self, delta: PendingConversationDelta) {
        let conversation = Arc::make_mut(&mut self.conversation);
        match delta {
            PendingConversationDelta::Reasoning(delta) => {
                reduce_conversation(
                    conversation,
                    ConversationAction::ReasoningDelta {
                        new_message: message(Role::Reasoning, delta.clone()),
                        delta,
                    },
                );
            }
            PendingConversationDelta::Text(delta) => {
                reduce_conversation(
                    conversation,
                    ConversationAction::TextDelta {
                        new_message: message(Role::Assistant, delta.clone()),
                        delta,
                    },
                );
            }
        }
    }

    fn apply_structural_event(&mut self, event: AgentEvent) -> (bool, bool) {
        let changes_transcript = matches!(
            &event,
            AgentEvent::ToolStarted(_)
                | AgentEvent::ToolFinished { .. }
                | AgentEvent::RunFinished(_)
                | AgentEvent::RunStarted(_)
                | AgentEvent::InputAdmitted { .. }
        );
        let mut conversation_changed = true;
        match event {
            AgentEvent::ApprovalRequired(call) => {
                conversation_changed = false;
                if self.allow_all_tools {
                    if let Some(control) = &self.control
                        && let Err(error) = control.approve(call.call_id, true)
                    {
                        self.notice(error.to_string());
                        conversation_changed = true;
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
                    Arc::make_mut(&mut self.conversation),
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
                        started_at_ms: None,
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
                    Arc::make_mut(&mut self.conversation),
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
                    Arc::make_mut(&mut self.conversation),
                    ConversationAction::SubmitUser(message(Role::User, input)),
                );
            }
            AgentEvent::InputAdmitted { input, .. } => {
                reduce_conversation(
                    Arc::make_mut(&mut self.conversation),
                    ConversationAction::SubmitUser(message(Role::User, input)),
                );
            }
            AgentEvent::ModelStarted(_) => conversation_changed = false,
            AgentEvent::SessionEvent(_)
            | AgentEvent::ReasoningDelta(_)
            | AgentEvent::TextDelta(_) => {
                unreachable!("stream events are handled before structural reduction")
            }
        }
        (conversation_changed, changes_transcript)
    }

    #[cfg(test)]
    pub(crate) fn apply_test_event(&mut self, event: AgentEvent, cx: &mut Context<Self>) {
        self.apply_events(vec![event]);
        cx.notify();
    }

    fn finish_reasoning(&mut self) {
        reduce_conversation(
            Arc::make_mut(&mut self.conversation),
            ConversationAction::FinishReasoning,
        );
    }

    fn tool_result(&mut self, call_id: &str, result: ToolResult) {
        reduce_conversation(
            Arc::make_mut(&mut self.conversation),
            ConversationAction::ToolFinished {
                call_id: call_id.to_owned(),
                output: result.output,
                is_error: result.is_error,
                duration_ms: None,
            },
        );
    }

    fn notice(&mut self, text: impl Into<String>) {
        reduce_conversation(
            Arc::make_mut(&mut self.conversation),
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

fn reasoning_key(effort: &ReasoningEffort) -> String {
    serde_json::to_value(effort)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| format!("{effort:?}").to_lowercase())
}

#[cfg(test)]
mod tests {
    use gpui::AppContext;

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
            TrajectoryProjection::default(),
            SessionConfig::default(),
        )
    }

    #[test]
    fn agent_events_mutate_only_the_runtime_that_owns_the_channel() {
        let mut first = runtime();
        let mut second = runtime();

        first.apply_events(vec![AgentEvent::TextDelta("first".into())]);
        second.apply_events(vec![AgentEvent::RunFailed("second failed".into())]);

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

        first.apply_events(vec![AgentEvent::TextDelta("only first".into())]);

        assert_eq!(first.conversation.messages.len(), 1);
        assert!(second.conversation.messages.is_empty());
        assert_eq!(second.active_run, Some(RunId(1)));
    }

    #[test]
    fn one_frame_merges_visible_deltas_across_durable_chunk_events() {
        let mut runtime = runtime();
        runtime.apply_events(vec![
            AgentEvent::TextDelta("hello ".into()),
            AgentEvent::SessionEvent(kcastle_agent::RecordedEvent {
                seq: 0,
                time: kcastle_agent::EventTime {
                    wall_time_ms: 1,
                    clock_id: "runtime-batch".into(),
                    monotonic_ns: 1,
                },
                source_event_seqs: Vec::new(),
                surface_op: None,
                event: kcastle_agent::SessionEvent::AssistantChunk {
                    turn: 1,
                    step: 1,
                    chunk: kcastle_agent::AssistantChunk::OutputTextDelta {
                        delta: "hello ".into(),
                    },
                },
            }),
            AgentEvent::TextDelta("world".into()),
        ]);

        assert_eq!(runtime.conversation.messages.len(), 1);
        assert_eq!(runtime.conversation.messages[0].text, "hello world");
        assert_eq!(runtime.transcript_updates, 2);
        assert_eq!(runtime.trajectory.revision(), 1);
    }

    #[test]
    fn snapshots_share_projection_storage_until_the_runtime_mutates() {
        let mut runtime = runtime();
        let before = runtime.snapshot();
        assert!(Arc::ptr_eq(&before.conversation, &runtime.conversation));
        assert!(Arc::ptr_eq(&before.trajectory, &runtime.trajectory));

        runtime.apply_events(vec![AgentEvent::TextDelta("new".into())]);
        let after = runtime.snapshot();
        assert!(before.conversation.messages.is_empty());
        assert_eq!(after.conversation.messages[0].text, "new");
        assert!(!Arc::ptr_eq(&before.conversation, &after.conversation));
        assert!(Arc::ptr_eq(&before.trajectory, &after.trajectory));
    }

    #[test]
    fn completion_generation_advances_only_for_a_finished_response() {
        let mut runtime = runtime();

        runtime.apply_events(vec![AgentEvent::RunAborted]);
        assert_eq!(runtime.completed_runs, 0);
        assert_eq!(runtime.transcript_updates, 1);

        runtime.apply_events(vec![AgentEvent::RunFinished(kcastle_agent::RunSummary {
            output: "done".into(),
            response_id: "response-1".into(),
            usage: None,
        })]);
        assert_eq!(runtime.completed_runs, 1);
        assert_eq!(runtime.transcript_updates, 2);
    }

    #[gpui::test]
    fn allow_all_applies_during_a_run_and_clears_pending_approval(cx: &mut gpui::TestAppContext) {
        let runtime = cx.new(|_| {
            let mut runtime = runtime();
            runtime.status = SessionRuntimeStatus::Running;
            runtime.approval = Some(ApprovalState {
                call_id: "call-1".into(),
                name: "shell".into(),
                arguments: "{}".into(),
            });
            runtime
        });

        runtime.update(cx, |runtime, cx| {
            assert!(runtime.set_allow_all_tools(true, cx));
        });

        let snapshot = cx.read_entity(&runtime, |runtime, _| runtime.snapshot());
        assert!(snapshot.allow_all_tools);
        assert!(snapshot.approval.is_none());

        runtime.update(cx, |runtime, _| {
            runtime.apply_events(vec![AgentEvent::ApprovalRequired(
                serde_json::from_value(serde_json::json!({
                    "arguments": "{}",
                    "call_id": "call-2",
                    "name": "shell"
                }))
                .unwrap(),
            )]);
        });
        assert!(cx.read_entity(&runtime, |runtime, _| runtime.approval.is_none()));
    }
}
