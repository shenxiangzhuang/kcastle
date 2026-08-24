use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
pub use async_openai::types::responses::ReasoningEffort;
use async_openai::types::responses::{
    CreateResponseArgs, EasyInputMessage, FunctionCallOutputItemParam, FunctionToolCall, InputItem,
    Item, OutputItem, Reasoning, Response, ResponseStreamEvent, ResponseUsage, Tool,
};
use futures_util::StreamExt;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

use crate::compaction::{
    CompactionConfig, SUMMARY_INSTRUCTIONS, context_tokens, prepare_compaction,
};
use crate::session::{Session, SessionConfig, SessionError, SessionInfo, SessionParts};
use crate::session_event::{
    AssistantChunk, CallId, CompactionId, EventDraft, EventTime, InputId, InputOrigin, RequestId,
    ResponseInfo, RunId, RunOutcome, SessionEvent, StepId, StepOutcome, ToolAuthorizationDecision,
    ToolExecutionOutcome, ToolResultStatus, TurnEndReason, TurnId, TxId,
};
use crate::session_machine::{PlannedBatch, SessionMachine, SessionMachineError};
use crate::session_store::{
    AppendTx, CommitReceipt, MetadataUpdate, SessionStore, SessionStoreError, SessionWriterPermit,
};
use crate::state::TranscriptItem;
use crate::tool::{AgentTool, Env, ShellTool, ToolResult};

const DEFAULT_MAX_TURNS: usize = 100;
const STREAM_COMMIT_INTERVAL: Duration = Duration::from_millis(32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPreset {
    pub id: &'static str,
    pub display_name: &'static str,
    pub context_window: usize,
}

pub const DEEPSEEK_PROVIDER_ID: &str = "deepseek-official";
pub const OPENAI_PROVIDER_ID: &str = "openai";

pub const DEEPSEEK_MODEL_PRESETS: &[ModelPreset] = &[
    ModelPreset {
        id: "deepseek-v4-flash",
        display_name: "DeepSeek-V4-Flash",
        context_window: 1_000_000,
    },
    ModelPreset {
        id: "deepseek-v4-pro",
        display_name: "DeepSeek-V4-Pro",
        context_window: 1_000_000,
    },
];

pub const OPENAI_MODEL_PRESETS: &[ModelPreset] = &[
    ModelPreset {
        id: "gpt-5.6-sol",
        display_name: "GPT-5.6 Sol",
        context_window: 1_050_000,
    },
    ModelPreset {
        id: "gpt-5.6-terra",
        display_name: "GPT-5.6 Terra",
        context_window: 1_050_000,
    },
    ModelPreset {
        id: "gpt-5.6-luna",
        display_name: "GPT-5.6 Luna",
        context_window: 1_050_000,
    },
];

const DEEPSEEK_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::None,
    ReasoningEffort::Low,
    ReasoningEffort::High,
];
const OPENAI_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::None,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::Xhigh,
];

#[derive(Clone)]
pub struct Model {
    name: String,
    api_key: String,
    api_base: String,
    client: Client<OpenAIConfig>,
    model: String,
    context_window: usize,
    max_output_tokens: Option<u32>,
    reasoning_efforts: &'static [ReasoningEffort],
    reasoning_effort: Option<ReasoningEffort>,
}

impl Model {
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        api_base: impl Into<String>,
        model: impl Into<String>,
        context_window: usize,
    ) -> Self {
        let api_key = api_key.into();
        let api_base = api_base.into();
        let config = OpenAIConfig::new()
            .with_api_key(api_key.clone())
            .with_api_base(api_base.clone());
        Self {
            name: name.into(),
            api_key,
            api_base,
            client: Client::with_config(config),
            model: model.into(),
            context_window,
            max_output_tokens: None,
            reasoning_efforts: &[],
            reasoning_effort: None,
        }
    }

    pub fn with_reasoning(
        mut self,
        reasoning_efforts: &'static [ReasoningEffort],
        reasoning_effort: ReasoningEffort,
    ) -> Self {
        assert!(reasoning_efforts.contains(&reasoning_effort));
        self.reasoning_efforts = reasoning_efforts;
        self.reasoning_effort = Some(reasoning_effort);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    pub fn has_api_key(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    pub fn context_window(&self) -> usize {
        self.context_window
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    pub fn with_max_output_tokens(mut self, max_output_tokens: Option<u32>) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    pub fn with_provider_reasoning(self, provider_id: &str) -> Self {
        match provider_id {
            DEEPSEEK_PROVIDER_ID | "deepseek" => {
                self.with_reasoning(DEEPSEEK_REASONING_EFFORTS, ReasoningEffort::High)
            }
            OPENAI_PROVIDER_ID => {
                self.with_reasoning(OPENAI_REASONING_EFFORTS, ReasoningEffort::Medium)
            }
            _ => self,
        }
    }

    pub fn reasoning_efforts(&self) -> &'static [ReasoningEffort] {
        self.reasoning_efforts
    }

    pub fn reasoning_effort(&self) -> Option<&ReasoningEffort> {
        self.reasoning_effort.as_ref()
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: ReasoningEffort) {
        assert!(self.reasoning_efforts.contains(&reasoning_effort));
        self.reasoning_effort = Some(reasoning_effort);
    }

    pub fn reconfigured(
        &self,
        name: impl Into<String>,
        api_key: Option<String>,
        api_base: impl Into<String>,
        model: impl Into<String>,
        context_window: usize,
    ) -> Self {
        let mut configured = Self::new(
            name,
            api_key.unwrap_or_else(|| self.api_key.clone()),
            api_base,
            model,
            context_window,
        );
        configured.reasoning_efforts = self.reasoning_efforts;
        configured.reasoning_effort = self.reasoning_effort.clone();
        configured.max_output_tokens = self.max_output_tokens;
        configured
    }
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("agent input must not be empty")]
    EmptyInput,
    #[error("agent exceeded {0} model turns")]
    MaxTurns(usize),
    #[error("Responses stream ended without a completed response")]
    MissingResponse,
    #[error("model response failed: {0}")]
    ModelResponse(String),
    #[error("not enough history to compact")]
    NothingToCompact,
    #[error("agent operation was aborted")]
    Aborted,
    #[error("session machine failed: {0}")]
    Machine(#[from] SessionMachineError),
    #[error(transparent)]
    OpenAI(#[from] OpenAIError),
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error(transparent)]
    Store(#[from] SessionStoreError),
    #[error("agent task failed: {0}")]
    Task(String),
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub output: String,
    pub response_id: String,
    pub usage: Option<ResponseUsage>,
}

/// Durable UI content is published exclusively through `SessionCommitted`.
/// The remaining variants are transient controls or completion notices.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    SessionCommitted(CommitReceipt),
    RunStarted(String),
    ModelStarted(usize),
    ApprovalRequired(FunctionToolCall),
    CompactionStarted {
        tokens_before: usize,
    },
    CompactionFinished {
        tokens_before: usize,
        first_kept_id: u64,
        summary: String,
    },
    RunFinished(RunSummary),
    RunAborted,
    RunFailed(String),
}

type EventSink = mpsc::UnboundedSender<AgentEvent>;

pub struct Agent {
    model: Model,
    instructions: String,
    machine: SessionMachine,
    store: SessionStore,
    revision: u64,
    writer: Option<SessionWriterPermit>,
    info: SessionInfo,
    session_config: SessionConfig,
    env: Env,
    tools: Vec<Arc<dyn AgentTool>>,
    compaction: Option<CompactionConfig>,
    max_turns: usize,
    clock: EventClock,
}

impl Agent {
    pub fn new(
        model: Model,
        instructions: impl Into<String>,
        session: Session,
        cwd: impl Into<PathBuf>,
    ) -> Self {
        let context_window = model.context_window();
        Self::from_session_parts(
            model,
            instructions.into(),
            session.into_parts(),
            cwd.into(),
            vec![Arc::new(ShellTool)],
            Some(CompactionConfig::new(context_window)),
            DEFAULT_MAX_TURNS,
        )
        .expect("default max turns is valid")
    }

    fn from_session_parts(
        model: Model,
        instructions: String,
        parts: SessionParts,
        cwd: PathBuf,
        tools: Vec<Arc<dyn AgentTool>>,
        compaction: Option<CompactionConfig>,
        max_turns: usize,
    ) -> Result<Self, AgentError> {
        if max_turns == 0 {
            return Err(AgentError::MaxTurns(0));
        }
        Ok(Self {
            model,
            instructions,
            machine: parts.machine,
            store: parts.store,
            revision: parts.revision,
            writer: None,
            info: parts.info,
            session_config: parts.config,
            env: Env { cwd },
            tools,
            compaction,
            max_turns,
            clock: EventClock::new(),
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn session_info(&self) -> &SessionInfo {
        &self.info
    }

    /// Revision of the canonical session snapshot currently owned by this agent.
    ///
    /// UI runtimes use this together with [`Self::session_config`] to decide whether an idle,
    /// cached agent still represents the current store snapshot before reusing it.
    pub fn session_revision(&self) -> u64 {
        self.revision
    }

    /// Durable configuration loaded into the session snapshot currently owned by this agent.
    pub fn session_config(&self) -> &SessionConfig {
        &self.session_config
    }

    pub fn transcript(&self) -> Vec<TranscriptItem> {
        self.machine.state().transcript()
    }

    pub fn latest_usage(&self) -> Option<&ResponseUsage> {
        self.machine
            .state()
            .latest_response()
            .and_then(|response| response.usage.as_ref())
    }

    pub fn set_model(&mut self, model: Model) {
        self.compaction = Some(CompactionConfig::new(model.context_window()));
        self.model = model;
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: ReasoningEffort) {
        self.model.set_reasoning_effort(reasoning_effort);
    }

    pub fn set_session(&mut self, session: Session) {
        let parts = session.into_parts();
        self.machine = parts.machine;
        self.store = parts.store;
        self.revision = parts.revision;
        self.writer = None;
        self.info = parts.info;
        self.session_config = parts.config;
        self.clock = EventClock::new();
    }

    pub fn set_cwd(&mut self, cwd: impl Into<PathBuf>) {
        self.env.cwd = cwd.into();
    }

    pub async fn rename_session(&mut self, title: &str) -> Result<(), AgentError> {
        let store = self.store.clone();
        let id = self.info.id.clone();
        let title = title.to_owned();
        let writer = self.acquire_or_clone_writer().await?;
        let metadata = tokio::task::spawn_blocking(move || {
            store.update_metadata(
                &id,
                MetadataUpdate {
                    title: Some(title),
                    ..MetadataUpdate::default()
                },
                &writer,
            )
        })
        .await
        .map_err(|error| AgentError::Task(error.to_string()))??;
        self.info.title = metadata.title;
        self.info.updated_at = millis_to_seconds(metadata.updated_at_ms);
        Ok(())
    }

    pub async fn persist_session_config(
        &mut self,
        config: &SessionConfig,
    ) -> Result<(), AgentError> {
        let store = self.store.clone();
        let id = self.info.id.clone();
        let new_config = config.clone();
        let writer = self.acquire_or_clone_writer().await?;
        let metadata = tokio::task::spawn_blocking(move || {
            store.update_metadata(
                &id,
                MetadataUpdate {
                    config: Some(new_config),
                    ..MetadataUpdate::default()
                },
                &writer,
            )
        })
        .await
        .map_err(|error| AgentError::Task(error.to_string()))??;
        self.session_config = config.clone();
        self.info.updated_at = millis_to_seconds(metadata.updated_at_ms);
        Ok(())
    }

    pub fn tool_schemas(&self) -> Vec<Tool> {
        self.tools.iter().map(|tool| tool.schema()).collect()
    }

    pub fn start(self, input: impl Into<String>) -> ActiveAgent {
        let input = input.into();
        self.spawn(move |agent, channels, events| async move {
            agent.run(input, channels, events).await
        })
    }

    pub fn start_compaction(self, instructions: Option<String>) -> ActiveAgent {
        self.spawn(move |agent, channels, events| async move {
            agent
                .run_manual_compaction(instructions.as_deref(), channels, events)
                .await
        })
    }

    fn spawn<F, Fut>(self, operation: F) -> ActiveAgent
    where
        F: FnOnce(Agent, RunChannels, EventSink) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Agent, Box<(Agent, AgentError)>>> + Send + 'static,
    {
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (approvals_tx, approvals_rx) = mpsc::unbounded_channel();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let cancel = CancellationToken::new();
        let control = RunControl {
            commands: commands_tx,
            approvals: approvals_tx,
            cancel: cancel.clone(),
        };
        let channels = RunChannels {
            commands: commands_rx,
            approvals: approvals_rx,
            cancel,
        };
        let task = tokio::spawn(async move {
            let mut agent = self;
            if let Err(error) = agent.acquire_writer_and_reload(&events_tx).await {
                publish(&events_tx, AgentEvent::RunFailed(error.to_string()));
                return agent;
            }
            let mut agent = match operation(agent, channels, events_tx.clone()).await {
                Ok(agent) => agent,
                Err(error) => {
                    let (mut agent, error) = *error;
                    let aborted = matches!(error, AgentError::Aborted);
                    let cleanup = agent
                        .terminate_after_error(aborted, &error, &events_tx)
                        .await;
                    match (aborted, cleanup) {
                        (true, Ok(())) => publish(&events_tx, AgentEvent::RunAborted),
                        (false, Ok(())) => {
                            publish(&events_tx, AgentEvent::RunFailed(error.to_string()));
                        }
                        (_, Err(cleanup)) => publish(
                            &events_tx,
                            AgentEvent::RunFailed(format!(
                                "{}; terminal commit failed: {cleanup}",
                                error
                            )),
                        ),
                    }
                    agent
                }
            };
            // Returning an idle Agent drops the last run-scoped writer capability. Readers and a
            // future run may now acquire the session; OS ownership is also released on crash.
            agent.writer = None;
            agent
        });
        ActiveAgent {
            control,
            events: events_rx,
            task,
        }
    }

    async fn run(
        mut self,
        input: String,
        mut channels: RunChannels,
        events: EventSink,
    ) -> Result<Self, Box<(Self, AgentError)>> {
        if input.trim().is_empty() {
            return Err(Box::new((self, AgentError::EmptyInput)));
        }
        if let Err(error) = self.recover_interrupted(&events).await {
            return Err(Box::new((self, error)));
        }

        let run_id = RunId::random();
        let turn_id = TurnId::random();
        let step_id = StepId::random();
        let input_id = InputId::random();
        let items = user_items(input.clone());
        let initial = vec![
            SessionEvent::InputSubmitted {
                input_id: input_id.clone(),
                input: input.clone(),
                origin: InputOrigin::Initial,
            },
            SessionEvent::RunStarted {
                run_id: run_id.clone(),
            },
            SessionEvent::TurnStarted {
                run_id: run_id.clone(),
                turn_id: turn_id.clone(),
            },
            SessionEvent::StepStarted {
                turn_id: turn_id.clone(),
                step_id: step_id.clone(),
            },
            SessionEvent::InputAttached {
                input_id,
                step_id: step_id.clone(),
                items,
            },
        ];
        if let Err(error) = self.commit_now(initial, &events).await {
            return Err(Box::new((self, error)));
        }
        self.set_initial_title_best_effort(&input).await;
        publish(&events, AgentEvent::RunStarted(input));

        let result = self
            .run_loop(run_id, turn_id, step_id, &mut channels, &events)
            .await;
        match result {
            Ok(summary) => {
                publish(&events, AgentEvent::RunFinished(summary));
                Ok(self)
            }
            Err(error) => Err(Box::new((self, error))),
        }
    }

    async fn run_loop(
        &mut self,
        run_id: RunId,
        mut turn_id: TurnId,
        mut step_id: StepId,
        channels: &mut RunChannels,
        events: &EventSink,
    ) -> Result<RunSummary, AgentError> {
        let mut request_count = 0_usize;
        loop {
            if request_count >= self.max_turns {
                return Err(AgentError::MaxTurns(self.max_turns));
            }
            self.compact_once(false, None, channels, events).await?;
            request_count += 1;
            publish(events, AgentEvent::ModelStarted(request_count));

            let response = self.request_model(&step_id, channels, events).await?;
            let calls = response
                .output
                .iter()
                .filter_map(|item| match item {
                    OutputItem::FunctionCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let summary = RunSummary {
                output: response.output_text().unwrap_or_default(),
                response_id: response.id.clone(),
                usage: response.usage.clone(),
            };

            if !calls.is_empty() {
                self.execute_tools(&calls, channels, events).await?;
            }
            self.drain_inputs(channels, events).await?;

            if let Some(input) = self.pending_input(InputOrigin::Steer) {
                let next_step = StepId::random();
                self.commit_now(
                    vec![
                        SessionEvent::StepTerminated {
                            step_id: step_id.clone(),
                            outcome: StepOutcome::Completed,
                            error: None,
                        },
                        SessionEvent::StepStarted {
                            turn_id: turn_id.clone(),
                            step_id: next_step.clone(),
                        },
                        SessionEvent::InputAttached {
                            input_id: input.input_id,
                            step_id: next_step.clone(),
                            items: user_items(input.input),
                        },
                    ],
                    events,
                )
                .await?;
                step_id = next_step;
                continue;
            }

            if !calls.is_empty() {
                let next_step = StepId::random();
                self.commit_now(
                    vec![
                        SessionEvent::StepTerminated {
                            step_id: step_id.clone(),
                            outcome: StepOutcome::Completed,
                            error: None,
                        },
                        SessionEvent::StepStarted {
                            turn_id: turn_id.clone(),
                            step_id: next_step.clone(),
                        },
                    ],
                    events,
                )
                .await?;
                step_id = next_step;
                continue;
            }

            if let Some(input) = self.pending_input(InputOrigin::Queue) {
                let next_turn = TurnId::random();
                let next_step = StepId::random();
                self.commit_now(
                    vec![
                        SessionEvent::StepTerminated {
                            step_id: step_id.clone(),
                            outcome: StepOutcome::Completed,
                            error: None,
                        },
                        SessionEvent::TurnTerminated {
                            turn_id: turn_id.clone(),
                            reason: TurnEndReason::Completed,
                        },
                        SessionEvent::TurnStarted {
                            run_id: run_id.clone(),
                            turn_id: next_turn.clone(),
                        },
                        SessionEvent::StepStarted {
                            turn_id: next_turn.clone(),
                            step_id: next_step.clone(),
                        },
                        SessionEvent::InputAttached {
                            input_id: input.input_id,
                            step_id: next_step.clone(),
                            items: user_items(input.input),
                        },
                    ],
                    events,
                )
                .await?;
                turn_id = next_turn;
                step_id = next_step;
                continue;
            }

            self.commit_now(
                vec![
                    SessionEvent::StepTerminated {
                        step_id,
                        outcome: StepOutcome::Completed,
                        error: None,
                    },
                    SessionEvent::TurnTerminated {
                        turn_id,
                        reason: TurnEndReason::Completed,
                    },
                    SessionEvent::RunTerminated {
                        run_id,
                        outcome: RunOutcome::Completed,
                        error: None,
                    },
                ],
                events,
            )
            .await?;
            return Ok(summary);
        }
    }

    async fn request_model(
        &mut self,
        step_id: &StepId,
        channels: &mut RunChannels,
        events: &EventSink,
    ) -> Result<Response, AgentError> {
        let request_id = RequestId::random();
        let tools = self.tool_schemas();
        let instructions = (!self.instructions.is_empty()).then(|| self.instructions.clone());
        let reasoning_effort = self
            .model
            .reasoning_effort
            .as_ref()
            .map(|effort| format!("{effort:?}").to_lowercase());
        let reason = self.machine.expected_request_reason(
            &self.model.model,
            instructions.as_deref(),
            &tools,
            reasoning_effort.as_deref(),
            self.model.max_output_tokens,
            &self.session_config,
        );
        let context = self.machine.context();
        let mut builder = CreateResponseArgs::default();
        builder
            .model(self.model.model.clone())
            .input(context)
            .tools(tools.clone())
            .store(false);
        if let Some(instructions) = instructions.clone() {
            builder.instructions(instructions);
        }
        // Request construction is pure and happens before the durable "started" intent. Once that
        // intent is committed, every exit path below can explicitly close this request.
        let mut request = builder.build()?;
        request.reasoning = self.reasoning();
        request.max_output_tokens = self.model.max_output_tokens;
        self.commit_now(
            vec![
                SessionEvent::RequestSnapshot {
                    request_id: request_id.clone(),
                    step_id: step_id.clone(),
                    reason,
                    model: self.model.model.clone(),
                    instructions: instructions.clone(),
                    tools: tools.clone(),
                    reasoning_effort,
                    max_output_tokens: self.model.max_output_tokens,
                    session_config: self.session_config.clone(),
                },
                SessionEvent::ModelRequestStarted {
                    request_id: request_id.clone(),
                },
            ],
            events,
        )
        .await?;

        let client = self.model.client.clone();
        let responses = client.responses();
        let stream_request = responses.create_stream(request);
        tokio::pin!(stream_request);
        let stream = loop {
            tokio::select! {
                _ = channels.cancel.cancelled() => {
                    self.fail_request(&request_id, "request cancelled before dispatch", events).await?;
                    return Err(AgentError::Aborted);
                }
                approval = channels.approvals.recv() => {
                    reject_inactive_approval(approval);
                }
                command = channels.commands.recv() => {
                    if let Some(command) = command {
                        self.submit_command(command, events).await?;
                    }
                }
                result = &mut stream_request => break result,
            }
        };
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                self.fail_request(&request_id, &error.to_string(), events)
                    .await?;
                return Err(error.into());
            }
        };

        let mut buffered = Vec::new();
        let mut item_calls = HashMap::<String, CallId>::new();
        let mut interval = tokio::time::interval(STREAM_COMMIT_INTERVAL);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            let streamed = tokio::select! {
                _ = channels.cancel.cancelled() => {
                    self.flush_chunks(&request_id, &mut buffered, events).await?;
                    self.fail_request(&request_id, "request cancelled while streaming", events).await?;
                    return Err(AgentError::Aborted);
                }
                _ = interval.tick() => {
                    self.flush_chunks(&request_id, &mut buffered, events).await?;
                    continue;
                }
                command = channels.commands.recv() => {
                    self.flush_chunks(&request_id, &mut buffered, events).await?;
                    if let Some(command) = command {
                        self.submit_command(command, events).await?;
                    }
                    continue;
                }
                approval = channels.approvals.recv() => {
                    reject_inactive_approval(approval);
                    continue;
                }
                event = stream.next() => event,
            };
            let Some(streamed) = streamed else {
                self.flush_chunks(&request_id, &mut buffered, events)
                    .await?;
                self.fail_request(
                    &request_id,
                    "response stream ended before completion",
                    events,
                )
                .await?;
                return Err(AgentError::MissingResponse);
            };
            match streamed {
                Ok(ResponseStreamEvent::ResponseOutputTextDelta(delta)) => {
                    buffered.push(ObservedEvent::new(
                        self.clock.now(),
                        SessionEvent::AssistantChunk {
                            request_id: request_id.clone(),
                            chunk: AssistantChunk::OutputTextDelta { delta: delta.delta },
                        },
                    ))
                }
                Ok(ResponseStreamEvent::ResponseReasoningTextDelta(delta)) => {
                    buffered.push(ObservedEvent::new(
                        self.clock.now(),
                        SessionEvent::AssistantChunk {
                            request_id: request_id.clone(),
                            chunk: AssistantChunk::ReasoningTextDelta { delta: delta.delta },
                        },
                    ))
                }
                Ok(ResponseStreamEvent::ResponseOutputItemAdded(added)) => {
                    if let OutputItem::FunctionCall(call) = added.item {
                        let call_id = CallId::from_raw(call.call_id);
                        if let Some(item_id) = call.id {
                            item_calls.insert(item_id, call_id.clone());
                        }
                        buffered.push(ObservedEvent::new(
                            self.clock.now(),
                            SessionEvent::AssistantChunk {
                                request_id: request_id.clone(),
                                chunk: AssistantChunk::ToolCallDelta {
                                    call_id,
                                    name: Some(call.name),
                                    arguments_delta: String::new(),
                                },
                            },
                        ));
                    }
                }
                Ok(ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(delta)) => {
                    let call_id = item_calls
                        .get(&delta.item_id)
                        .cloned()
                        .unwrap_or_else(|| CallId::from_raw(delta.item_id));
                    buffered.push(ObservedEvent::new(
                        self.clock.now(),
                        SessionEvent::AssistantChunk {
                            request_id: request_id.clone(),
                            chunk: AssistantChunk::ToolCallDelta {
                                call_id,
                                name: None,
                                arguments_delta: delta.delta,
                            },
                        },
                    ));
                }
                Ok(ResponseStreamEvent::ResponseCompleted(completed)) => {
                    let observed_at = self.clock.now();
                    self.flush_chunks(&request_id, &mut buffered, events)
                        .await?;
                    let response = completed.response;
                    if response.output.is_empty() {
                        let error = "model response completed without output items";
                        self.fail_request_observed(&request_id, error, observed_at, events)
                            .await?;
                        return Err(AgentError::ModelResponse(error.into()));
                    }
                    if let Err(error) = self
                        .complete_assistant(&request_id, &response, observed_at.clone(), events)
                        .await
                    {
                        // A provider-shape validation failure leaves the request open because the
                        // assistant transaction was never committed. Close it explicitly before
                        // the run-level terminal transaction.
                        if matches!(error, AgentError::Machine(_)) {
                            self.fail_request_observed(
                                &request_id,
                                &error.to_string(),
                                observed_at,
                                events,
                            )
                            .await?;
                        }
                        return Err(error);
                    }
                    return Ok(response);
                }
                Ok(ResponseStreamEvent::ResponseFailed(failed)) => {
                    let observed_at = self.clock.now();
                    self.flush_chunks(&request_id, &mut buffered, events)
                        .await?;
                    let error = format!("{:?}", failed.response.error);
                    self.fail_request_observed(&request_id, &error, observed_at, events)
                        .await?;
                    return Err(AgentError::ModelResponse(error));
                }
                Ok(ResponseStreamEvent::ResponseIncomplete(incomplete)) => {
                    let observed_at = self.clock.now();
                    self.flush_chunks(&request_id, &mut buffered, events)
                        .await?;
                    let error = format!("incomplete: {:?}", incomplete.response.incomplete_details);
                    self.fail_request_observed(&request_id, &error, observed_at, events)
                        .await?;
                    return Err(AgentError::ModelResponse(error));
                }
                Ok(ResponseStreamEvent::ResponseError(error)) => {
                    let observed_at = self.clock.now();
                    self.flush_chunks(&request_id, &mut buffered, events)
                        .await?;
                    let error = format!("{error:?}");
                    self.fail_request_observed(&request_id, &error, observed_at, events)
                        .await?;
                    return Err(AgentError::ModelResponse(error));
                }
                Ok(_) => {}
                Err(error) => {
                    let observed_at = self.clock.now();
                    self.flush_chunks(&request_id, &mut buffered, events)
                        .await?;
                    self.fail_request_observed(
                        &request_id,
                        &error.to_string(),
                        observed_at,
                        events,
                    )
                    .await?;
                    return Err(error.into());
                }
            }
        }
    }

    async fn complete_assistant(
        &mut self,
        request_id: &RequestId,
        response: &Response,
        observed_at: EventTime,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let items = response
            .output
            .iter()
            .cloned()
            .map(InputItem::from)
            .collect::<Vec<_>>();
        let mut completed = Vec::with_capacity(items.len().saturating_add(1));
        completed.push(SessionEvent::AssistantCompleted {
            request_id: request_id.clone(),
            items,
            response: ResponseInfo {
                id: response.id.clone(),
                model: response.model.clone(),
                usage: response
                    .usage
                    .as_ref()
                    .map(crate::session_event::TokenUsage::from_provider),
            },
        });
        completed.extend(response.output.iter().filter_map(|item| match item {
            OutputItem::FunctionCall(call) => Some(SessionEvent::ToolCallRequested {
                request_id: request_id.clone(),
                call_id: CallId::from_raw(call.call_id.clone()),
                parent_call_id: None,
            }),
            _ => None,
        }));
        self.commit_observed(
            completed
                .into_iter()
                .map(|event| ObservedEvent::new(observed_at.clone(), event))
                .collect(),
            events,
        )
        .await?;
        Ok(())
    }

    async fn execute_tools(
        &mut self,
        calls: &[FunctionToolCall],
        channels: &mut RunChannels,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let mut tasks = JoinSet::new();
        let mut outcomes = vec![None::<ToolCompletion>; calls.len()];
        let mut resolved_tools = Vec::with_capacity(calls.len());
        let mut decisions = Vec::with_capacity(calls.len());
        // Only calls which actually requested an interactive decision are approval capabilities.
        // Keep resolved interactive calls in this map until the approval phase closes so a retry
        // of the same decision can be acknowledged idempotently after an acknowledgement loss.
        let mut interactive_approval_indexes = HashMap::new();
        let mut automatic_authorizations = Vec::with_capacity(calls.len());
        let mut approval_requests = Vec::new();
        for (index, call) in calls.iter().enumerate() {
            let tool = self
                .tools
                .iter()
                .find(|tool| tool.name() == call.name)
                .cloned();
            let decision = if tool.is_none() {
                outcomes[index] = Some(ToolCompletion {
                    result: ToolResult::error(format!("Tool not found: {}", call.name)),
                    status: ToolResultStatus::NotFound,
                });
                Some((ToolAuthorizationDecision::Unavailable, self.clock.now()))
            } else if self.session_config.allow_all_tools
                || tool.as_ref().is_some_and(|tool| !tool.requires_approval())
            {
                Some((ToolAuthorizationDecision::NotRequired, self.clock.now()))
            } else {
                interactive_approval_indexes.insert(call.call_id.clone(), index);
                approval_requests.push(call.clone());
                None
            };
            if let Some((decision, observed_at)) = &decision {
                automatic_authorizations.push(ObservedEvent::new(
                    observed_at.clone(),
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: CallId::from_raw(call.call_id.clone()),
                        decision: *decision,
                    },
                ));
            }
            resolved_tools.push(tool);
            decisions.push(decision);
        }

        // Authorization is an independently observed fact, not part of dispatch. Persist all
        // automatic decisions before exposing approval prompts, then persist every interactive
        // decision as it arrives. A crash while another call is still awaiting approval therefore
        // cannot erase an earlier decision or its waiting time.
        if !automatic_authorizations.is_empty() {
            self.commit_observed(automatic_authorizations, events)
                .await?;
        }
        for call in approval_requests {
            publish(events, AgentEvent::ApprovalRequired(call));
        }

        while decisions.iter().any(Option::is_none) {
            tokio::select! {
                _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                command = channels.commands.recv() => {
                    if let Some(command) = command {
                        self.submit_command(command, events).await?;
                    }
                }
                approval = channels.approvals.recv() => {
                    let Some(ApprovalCommand {
                        call_id,
                        allow,
                        acknowledgement,
                    }) = approval else {
                        return Err(AgentError::Aborted);
                    };
                    let Some(index) = interactive_approval_indexes.get(&call_id).copied() else {
                        let _ = acknowledgement.send(Err(format!(
                            "tool call {call_id} did not request interactive authorization"
                        )));
                        continue;
                    };
                    let requested_decision = requested_authorization(allow);
                    if let Some((persisted_decision, _)) = decisions[index] {
                        let result = if persisted_decision == requested_decision {
                            Ok(())
                        } else {
                            Err(format!(
                                "tool call {call_id} authorization is already resolved as {persisted_decision:?}"
                            ))
                        };
                        let _ = acknowledgement.send(result);
                        continue;
                    }
                    let decision = if requested_decision == ToolAuthorizationDecision::Allowed {
                        requested_decision
                    } else {
                        outcomes[index] = Some(ToolCompletion {
                            result: ToolResult::error("Tool call denied by user"),
                            status: ToolResultStatus::Denied,
                        });
                        requested_decision
                    };
                    let observed_at = self.clock.now();
                    let committed = self
                        .commit_observed(
                            vec![ObservedEvent::new(
                                observed_at.clone(),
                                SessionEvent::ToolAuthorizationResolved {
                                    call_id: CallId::from_raw(call_id),
                                    decision,
                                },
                            )],
                            events,
                        )
                        .await;
                    match committed {
                        Ok(_) => {
                            decisions[index] = Some((decision, observed_at));
                            let _ = acknowledgement.send(Ok(()));
                        }
                        Err(error) => {
                            let _ = acknowledgement.send(Err(error.to_string()));
                            return Err(error);
                        }
                    }
                }
            }
        }

        let dispatch_time = self.clock.now();
        let mut dispatch = Vec::with_capacity(calls.len());
        for (call, resolved) in calls.iter().zip(&decisions) {
            let (decision, _) = resolved
                .as_ref()
                .expect("all tool authorization decisions are resolved");
            let call_id = CallId::from_raw(call.call_id.clone());
            if decision.permits_execution() {
                dispatch.push(ObservedEvent::new(
                    dispatch_time.clone(),
                    SessionEvent::ToolDispatchIntended { call_id },
                ));
            }
        }
        // No tool task exists until the complete dispatch intent is durable. A crash before this
        // receipt has no tool side effects; a crash after it is conservatively recovered as
        // unknown side effects even if the task's actual start observation was not committed yet.
        if !dispatch.is_empty() {
            self.commit_observed(dispatch, events).await?;
        }

        let (task_events, mut task_event_receiver) = mpsc::unbounded_channel();
        for (index, ((call, tool), decision)) in calls
            .iter()
            .zip(resolved_tools)
            .zip(decisions.iter())
            .enumerate()
        {
            let (decision, _) = decision
                .as_ref()
                .expect("all tool authorization decisions are resolved");
            if !decision.permits_execution() {
                continue;
            }
            let tool = tool.expect("permitted tool must be registered");
            let call = call.clone();
            let env = self.env.clone();
            let clock = self.clock.clone();
            let task_events = task_events.clone();
            tasks.spawn(async move {
                let _ = task_events.send(ToolTaskEvent::Started {
                    index,
                    time: clock.now(),
                });
                let result = tool.execute(&call, &env).await;
                let _ = task_events.send(ToolTaskEvent::Finished {
                    index,
                    time: clock.now(),
                    result,
                });
            });
        }
        drop(task_events);

        let mut next_attachment = 0_usize;
        self.attach_ready_results(calls, &mut outcomes, &mut next_attachment, events)
            .await?;
        while next_attachment != calls.len() {
            tokio::select! {
                _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                command = channels.commands.recv() => {
                    if let Some(command) = command {
                        self.submit_command(command, events).await?;
                    }
                }
                approval = channels.approvals.recv() => {
                    acknowledge_persisted_approval(
                        approval,
                        &interactive_approval_indexes,
                        &decisions,
                    );
                }
                task_event = task_event_receiver.recv() => {
                    let Some(task_event) = task_event else {
                        return Err(AgentError::Task("tool execution event stream ended early".into()));
                    };
                    match task_event {
                        ToolTaskEvent::Started { index, time } => {
                            self.commit_observed(
                                vec![ObservedEvent::new(
                                    time,
                                    SessionEvent::ToolExecutionStarted {
                                        call_id: CallId::from_raw(calls[index].call_id.clone()),
                                    },
                                )],
                                events,
                            )
                            .await?;
                        }
                        ToolTaskEvent::Finished { index, time, result } => {
                            let status = if result.is_error {
                                ToolResultStatus::Error
                            } else {
                                ToolResultStatus::Success
                            };
                            self.commit_observed(
                                vec![ObservedEvent::new(
                                    time,
                                    SessionEvent::ToolExecutionFinished {
                                        call_id: CallId::from_raw(calls[index].call_id.clone()),
                                        outcome: if result.is_error {
                                            ToolExecutionOutcome::Error
                                        } else {
                                            ToolExecutionOutcome::Success
                                        },
                                    },
                                )],
                                events,
                            )
                            .await?;
                            outcomes[index] = Some(ToolCompletion { result, status });
                            self.attach_ready_results(
                                calls,
                                &mut outcomes,
                                &mut next_attachment,
                                events,
                            )
                            .await?;
                        }
                    }
                }
            }
        }
        while let Some(task) = tasks.join_next().await {
            task.map_err(|error| AgentError::Task(error.to_string()))?;
        }
        self.attach_ready_results(calls, &mut outcomes, &mut next_attachment, events)
            .await?;
        if next_attachment != calls.len() {
            return Err(AgentError::Task(
                "tool results did not become attachable in call order".into(),
            ));
        }
        Ok(())
    }

    async fn attach_ready_results(
        &mut self,
        calls: &[FunctionToolCall],
        outcomes: &mut [Option<ToolCompletion>],
        next: &mut usize,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let mut attached = Vec::new();
        while *next < calls.len() {
            let Some(completion) = outcomes[*next].take() else {
                break;
            };
            let call = &calls[*next];
            attached.push(SessionEvent::ToolResultAttached {
                call_id: CallId::from_raw(call.call_id.clone()),
                status: completion.status,
                item: function_output(call, completion.result.output),
            });
            *next += 1;
        }
        if !attached.is_empty() {
            self.commit_now(attached, events).await?;
        }
        Ok(())
    }

    async fn flush_chunks(
        &mut self,
        request_id: &RequestId,
        buffered: &mut Vec<ObservedEvent>,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        if buffered.is_empty() {
            return Ok(());
        }
        let batch = std::mem::take(buffered);
        match self.commit_observed(batch, events).await {
            Ok(_) => Ok(()),
            Err(error) => {
                // A rejected/pre-commit chunk batch must not strand ModelRequestStarted. If the
                // store is still available, persist the explicit failure and let the caller write
                // the atomic step/turn/run terminal batch.
                let _ = self
                    .fail_request(
                        request_id,
                        &format!("assistant chunk commit failed: {error}"),
                        events,
                    )
                    .await;
                Err(error)
            }
        }
    }

    async fn fail_request(
        &mut self,
        request_id: &RequestId,
        error: &str,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        self.fail_request_observed(request_id, error, self.clock.now(), events)
            .await
    }

    async fn fail_request_observed(
        &mut self,
        request_id: &RequestId,
        error: &str,
        observed_at: EventTime,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        self.commit_observed(
            vec![ObservedEvent::new(
                observed_at,
                SessionEvent::ModelRequestFailed {
                    request_id: request_id.clone(),
                    error: if error.trim().is_empty() {
                        "model request failed".into()
                    } else {
                        error.to_owned()
                    },
                },
            )],
            events,
        )
        .await?;
        Ok(())
    }

    async fn drain_inputs(
        &mut self,
        channels: &mut RunChannels,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        while let Ok(command) = channels.commands.try_recv() {
            self.submit_command(command, events).await?;
        }
        Ok(())
    }

    async fn submit_command(
        &mut self,
        command: InputCommand,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let InputCommand {
            input_id,
            input,
            origin,
            acknowledgement,
        } = command;
        let committed = self
            .commit_now(
                vec![SessionEvent::InputSubmitted {
                    input_id,
                    input,
                    origin,
                }],
                events,
            )
            .await;
        match committed {
            Ok(_) => {
                let _ = acknowledgement.send(Ok(()));
                Ok(())
            }
            Err(error) => {
                let _ = acknowledgement.send(Err(error.to_string()));
                Err(error)
            }
        }
    }

    fn pending_input(&self, origin: InputOrigin) -> Option<crate::session_machine::PendingInput> {
        self.machine
            .pending_inputs()
            .into_iter()
            .find(|input| input.origin == origin)
    }

    fn reasoning(&self) -> Option<Reasoning> {
        self.model.reasoning_effort.clone().map(|effort| Reasoning {
            effort: Some(effort),
            summary: None,
        })
    }

    async fn commit_now(
        &mut self,
        events: Vec<SessionEvent>,
        sink: &EventSink,
    ) -> Result<CommitReceipt, AgentError> {
        let observed = events
            .into_iter()
            .map(|event| ObservedEvent::new(self.clock.now(), event))
            .collect();
        self.commit_observed(observed, sink).await
    }

    async fn commit_observed(
        &mut self,
        events: Vec<ObservedEvent>,
        sink: &EventSink,
    ) -> Result<CommitReceipt, AgentError> {
        let tx_id = TxId::random();
        let drafts = events
            .into_iter()
            .map(|observed| EventDraft {
                tx_id: tx_id.clone(),
                time: observed.time,
                event: observed.event,
            })
            .collect();
        let planned = self.machine.plan_batch(drafts)?;
        self.commit_planned(planned, sink).await
    }

    async fn commit_planned(
        &mut self,
        planned: PlannedBatch,
        sink: &EventSink,
    ) -> Result<CommitReceipt, AgentError> {
        let append = AppendTx::from_planned(self.info.id.clone(), self.revision, &planned);
        let store = self.store.clone();
        let writer = self.writer.clone().ok_or_else(|| {
            AgentError::Task("session mutation attempted without writer capability".into())
        })?;
        let attempted = append.clone();
        let result = tokio::task::spawn_blocking(move || store.append(&attempted, &writer))
            .await
            .map_err(|error| AgentError::Task(error.to_string()))?;
        let receipt = match result {
            Ok(receipt) => receipt,
            Err(SessionStoreError::OutcomeUnknown { .. }) => {
                let store = self.store.clone();
                let id = self.info.id.clone();
                let tx_id = append.tx_id.clone();
                tokio::task::spawn_blocking(move || store.resolve(&id, &tx_id))
                    .await
                    .map_err(|error| AgentError::Task(error.to_string()))??
                    .ok_or(SessionStoreError::OutcomeUnknown {
                        tx_id: append.tx_id.clone(),
                    })?
            }
            Err(error) => return Err(error.into()),
        };
        if receipt.events.as_slice() != planned.events()
            || receipt.base_revision != self.revision
            || receipt.revision != self.revision.saturating_add(1)
        {
            return Err(AgentError::Task(
                "session store receipt did not match the planned transaction".into(),
            ));
        }
        self.machine.apply_batch(planned)?;
        self.revision = receipt.revision;
        self.info.updated_at = millis_to_seconds(receipt.committed_at_ms);
        publish(sink, AgentEvent::SessionCommitted(receipt.clone()));
        Ok(receipt)
    }

    async fn acquire_or_clone_writer(&self) -> Result<SessionWriterPermit, AgentError> {
        if let Some(writer) = &self.writer {
            return Ok(writer.clone());
        }
        let store = self.store.clone();
        let id = self.info.id.clone();
        tokio::task::spawn_blocking(move || store.acquire_writer(&id))
            .await
            .map_err(|error| AgentError::Task(error.to_string()))?
            .map_err(Into::into)
    }

    async fn acquire_writer_and_reload(&mut self, sink: &EventSink) -> Result<(), AgentError> {
        let writer = self.acquire_or_clone_writer().await?;
        let store = self.store.clone();
        let id = self.info.id.clone();
        let loaded = tokio::task::spawn_blocking(move || store.load(&id))
            .await
            .map_err(|error| AgentError::Task(error.to_string()))??;
        if loaded.metadata.archived_at_ms.is_some() {
            return Err(SessionError::Invalid("archived session cannot run".into()).into());
        }
        if loaded.metadata.project_id != self.info.project_id {
            return Err(SessionError::Invalid(format!(
                "session moved from project {} to {}",
                self.info.project_id, loaded.metadata.project_id
            ))
            .into());
        }
        if loaded.metadata.revision < self.revision {
            return Err(SessionStoreError::Corrupt(format!(
                "session revision moved backwards from {} to {}",
                self.revision, loaded.metadata.revision
            ))
            .into());
        }
        if loaded.metadata.config != self.session_config {
            return Err(SessionError::Invalid(
                "session configuration changed in another writer; reopen the session".into(),
            )
            .into());
        }
        let current_revision = self.revision;
        let mut catch_up = Vec::new();
        let mut events = Vec::new();
        for transaction in loaded.transactions {
            if transaction.revision > current_revision {
                // The receipt is published after the machine catches up, so only the usually-small
                // unseen suffix needs a second event copy. Historical events move directly into
                // replay instead of doubling the complete journal in memory on every run.
                events.extend(transaction.events.iter().cloned());
                catch_up.push(transaction);
            } else {
                events.extend(transaction.events);
            }
        }
        self.machine = SessionMachine::from_events(&events)?;
        self.revision = loaded.metadata.revision;
        self.info.title = loaded.metadata.title;
        self.info.created_at = millis_to_seconds(loaded.metadata.created_at_ms);
        self.info.updated_at = millis_to_seconds(loaded.metadata.updated_at_ms);
        self.writer = Some(writer);
        // A runtime may have stayed idle while another process committed. Publish those already
        // durable receipts in revision order before recovery or any new effect so every UI
        // document catches up through the same committed-event route and never sees a sequence gap.
        for receipt in catch_up {
            publish(sink, AgentEvent::SessionCommitted(receipt));
        }
        Ok(())
    }

    async fn recover_interrupted(&mut self, events: &EventSink) -> Result<(), AgentError> {
        let Some(planned) = self
            .machine
            .plan_recovery(TxId::random(), self.clock.now())?
        else {
            return Ok(());
        };
        self.commit_planned(planned, events).await?;
        Ok(())
    }

    async fn terminate_after_error(
        &mut self,
        aborted: bool,
        error: &AgentError,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let outcome = if aborted {
            RunOutcome::Aborted
        } else {
            RunOutcome::Failed
        };
        let step_outcome = if aborted {
            StepOutcome::Aborted
        } else {
            StepOutcome::Failed
        };
        let turn_reason = if aborted {
            TurnEndReason::Aborted
        } else if matches!(error, AgentError::MaxTurns(_)) {
            TurnEndReason::MaxTurns
        } else {
            TurnEndReason::Failed
        };
        let time = self.clock.now();
        let Some(recovery) = self.machine.plan_recovery(TxId::random(), time.clone())? else {
            return Ok(());
        };
        let message = error.to_string();
        let terminal = recovery
            .into_events()
            .into_iter()
            .map(|recorded| {
                let event = match recorded.event {
                    SessionEvent::ModelRequestFailed { request_id, .. } => {
                        SessionEvent::ModelRequestFailed {
                            request_id,
                            error: message.clone(),
                        }
                    }
                    SessionEvent::CompactionFinished {
                        compaction_id,
                        summary,
                        response,
                        ..
                    } => SessionEvent::CompactionFinished {
                        compaction_id,
                        outcome: step_outcome,
                        summary,
                        response,
                    },
                    SessionEvent::StepTerminated { step_id, .. } => SessionEvent::StepTerminated {
                        step_id,
                        outcome: step_outcome,
                        error: Some(message.clone()),
                    },
                    SessionEvent::TurnTerminated { turn_id, .. } => SessionEvent::TurnTerminated {
                        turn_id,
                        reason: turn_reason,
                    },
                    SessionEvent::RunTerminated { run_id, .. } => SessionEvent::RunTerminated {
                        run_id,
                        outcome,
                        error: Some(message.clone()),
                    },
                    event => event,
                };
                ObservedEvent::new(recorded.time, event)
            })
            .collect();
        // Tool/compaction/request closure and step/turn/run termination are one all-or-nothing
        // transaction. There is no partially terminal durable state to recover from later.
        self.commit_observed(terminal, events).await?;
        Ok(())
    }

    async fn set_initial_title_best_effort(&mut self, input: &str) {
        if self.info.title != "Untitled session" {
            return;
        }
        let title = input.split_whitespace().collect::<Vec<_>>().join(" ");
        let title = title.chars().take(48).collect::<String>();
        if title.is_empty() {
            return;
        }
        let _ = self.rename_session(&title).await;
    }

    async fn compact_once(
        &mut self,
        force: bool,
        custom_instructions: Option<&str>,
        channels: &mut RunChannels,
        events: &EventSink,
    ) -> Result<(), AgentError> {
        let Some(config) = self.compaction else {
            return if force {
                Err(AgentError::NothingToCompact)
            } else {
                Ok(())
            };
        };
        let tools = self.tool_schemas();
        let tokens_before = context_tokens(self.machine.state(), &self.instructions, &tools);
        if !force && !config.needs_compaction(tokens_before) {
            return Ok(());
        }
        let Some(prepared) = prepare_compaction(
            self.machine.state(),
            config.keep_recent_tokens,
            custom_instructions,
        ) else {
            return if force {
                Err(AgentError::NothingToCompact)
            } else {
                Ok(())
            };
        };
        let run_id = self.machine.active_run().cloned().ok_or_else(|| {
            AgentError::Task("automatic compaction requires an active run".into())
        })?;
        let compaction_id = CompactionId::random();
        self.commit_now(
            vec![SessionEvent::CompactionStarted {
                compaction_id: compaction_id.clone(),
                run_id,
                tokens_before,
                first_kept_id: prepared.first_kept_id,
            }],
            events,
        )
        .await?;
        publish(events, AgentEvent::CompactionStarted { tokens_before });

        let mut builder = CreateResponseArgs::default();
        builder
            .model(self.model.model.clone())
            .instructions(SUMMARY_INSTRUCTIONS)
            .input(prepared.prompt)
            .store(false);
        let mut request = builder.build()?;
        request.reasoning = self.reasoning();
        request.max_output_tokens = self.model.max_output_tokens;
        let client = self.model.client.clone();
        let responses = client.responses();
        let response = responses.create(request);
        tokio::pin!(response);
        let result = loop {
            tokio::select! {
                _ = channels.cancel.cancelled() => break Err(AgentError::Aborted),
                command = channels.commands.recv() => {
                    if let Some(command) = command {
                        self.submit_command(command, events).await?;
                    }
                }
                approval = channels.approvals.recv() => {
                    reject_inactive_approval(approval);
                }
                response = &mut response => break response.map_err(AgentError::from),
            }
        };
        // Capture provider settlement before SQLite work so compaction timing excludes commit
        // latency and remains comparable with model request timing.
        let observed_at = self.clock.now();
        let response = match result {
            Ok(response) => response,
            Err(error) => {
                self.commit_observed(
                    vec![ObservedEvent::new(
                        observed_at,
                        SessionEvent::CompactionFinished {
                            compaction_id,
                            outcome: if matches!(error, AgentError::Aborted) {
                                StepOutcome::Aborted
                            } else {
                                StepOutcome::Failed
                            },
                            summary: None,
                            response: None,
                        },
                    )],
                    events,
                )
                .await?;
                return Err(error);
            }
        };
        let summary = response.output_text().unwrap_or_default();
        if summary.trim().is_empty() {
            self.commit_observed(
                vec![ObservedEvent::new(
                    observed_at,
                    SessionEvent::CompactionFinished {
                        compaction_id,
                        outcome: StepOutcome::Failed,
                        summary: None,
                        response: Some(response_info(&response)),
                    },
                )],
                events,
            )
            .await?;
            return Err(AgentError::ModelResponse(
                "compaction returned an empty summary".into(),
            ));
        }
        self.commit_observed(
            vec![ObservedEvent::new(
                observed_at,
                SessionEvent::CompactionFinished {
                    compaction_id,
                    outcome: StepOutcome::Completed,
                    summary: Some(summary.clone()),
                    response: Some(response_info(&response)),
                },
            )],
            events,
        )
        .await?;
        publish(
            events,
            AgentEvent::CompactionFinished {
                tokens_before,
                first_kept_id: prepared.first_kept_id,
                summary,
            },
        );
        Ok(())
    }

    async fn run_manual_compaction(
        mut self,
        instructions: Option<&str>,
        mut channels: RunChannels,
        events: EventSink,
    ) -> Result<Self, Box<(Self, AgentError)>> {
        if let Err(error) = self.recover_interrupted(&events).await {
            return Err(Box::new((self, error)));
        }
        let Some(config) = self.compaction else {
            return Err(Box::new((self, AgentError::NothingToCompact)));
        };
        let tools = self.tool_schemas();
        let tokens = context_tokens(self.machine.state(), &self.instructions, &tools);
        if prepare_compaction(
            self.machine.state(),
            config.keep_recent_tokens,
            instructions,
        )
        .is_none()
        {
            return Err(Box::new((self, AgentError::NothingToCompact)));
        }
        let run_id = RunId::random();
        if let Err(error) = self
            .commit_now(
                vec![SessionEvent::RunStarted {
                    run_id: run_id.clone(),
                }],
                &events,
            )
            .await
        {
            return Err(Box::new((self, error)));
        }
        if let Err(error) = self
            .compact_once(true, instructions, &mut channels, &events)
            .await
        {
            return Err(Box::new((self, error)));
        }
        if let Err(error) = self
            .commit_now(
                vec![SessionEvent::RunTerminated {
                    run_id,
                    outcome: RunOutcome::Completed,
                    error: None,
                }],
                &events,
            )
            .await
        {
            return Err(Box::new((self, error)));
        }
        publish(
            &events,
            AgentEvent::RunFinished(RunSummary {
                output: format!("Compacted {tokens} tokens"),
                response_id: String::new(),
                usage: None,
            }),
        );
        Ok(self)
    }
}

#[derive(Clone)]
struct EventClock {
    inner: Arc<EventClockInner>,
}

struct EventClockInner {
    id: String,
    started: Instant,
}

impl EventClock {
    fn new() -> Self {
        Self {
            inner: Arc::new(EventClockInner {
                id: uuid::Uuid::new_v4().to_string(),
                started: Instant::now(),
            }),
        }
    }

    fn now(&self) -> EventTime {
        observed_event_time(
            &self.inner.id,
            self.inner.started.elapsed(),
            system_time_ms(),
        )
    }
}

fn observed_event_time(clock_id: &str, elapsed: Duration, wall_time_ms: i64) -> EventTime {
    EventTime {
        wall_time_ms,
        clock_id: clock_id.to_owned(),
        monotonic_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
    }
}

struct ObservedEvent {
    time: EventTime,
    event: SessionEvent,
}

impl ObservedEvent {
    fn new(time: EventTime, event: SessionEvent) -> Self {
        Self { time, event }
    }
}

enum ToolTaskEvent {
    Started {
        index: usize,
        time: EventTime,
    },
    Finished {
        index: usize,
        time: EventTime,
        result: ToolResult,
    },
}

#[derive(Clone)]
struct ToolCompletion {
    result: ToolResult,
    status: ToolResultStatus,
}

struct InputCommand {
    input_id: InputId,
    input: String,
    origin: InputOrigin,
    acknowledgement: oneshot::Sender<Result<(), String>>,
}

struct ApprovalCommand {
    call_id: String,
    allow: bool,
    acknowledgement: oneshot::Sender<Result<(), String>>,
}

fn requested_authorization(allow: bool) -> ToolAuthorizationDecision {
    if allow {
        ToolAuthorizationDecision::Allowed
    } else {
        ToolAuthorizationDecision::Denied
    }
}

fn acknowledge_persisted_approval(
    approval: Option<ApprovalCommand>,
    interactive_indexes: &HashMap<String, usize>,
    decisions: &[Option<(ToolAuthorizationDecision, EventTime)>],
) {
    let Some(ApprovalCommand {
        call_id,
        allow,
        acknowledgement,
    }) = approval
    else {
        return;
    };
    let result = interactive_indexes
        .get(&call_id)
        .and_then(|index| decisions.get(*index))
        .and_then(Option::as_ref)
        .map_or_else(
            || Err(format!("tool call {call_id} is not awaiting authorization")),
            |(persisted, _)| {
                let requested = requested_authorization(allow);
                if *persisted == requested {
                    Ok(())
                } else {
                    Err(format!(
                        "tool call {call_id} authorization is already resolved as {persisted:?}"
                    ))
                }
            },
        );
    let _ = acknowledgement.send(result);
}

fn reject_inactive_approval(approval: Option<ApprovalCommand>) {
    let Some(ApprovalCommand {
        call_id,
        acknowledgement,
        ..
    }) = approval
    else {
        return;
    };
    let _ = acknowledgement.send(Err(format!(
        "tool call {call_id} is not awaiting authorization"
    )));
}

struct RunChannels {
    commands: mpsc::UnboundedReceiver<InputCommand>,
    approvals: mpsc::UnboundedReceiver<ApprovalCommand>,
    cancel: CancellationToken,
}

#[derive(Clone)]
pub struct RunControl {
    commands: mpsc::UnboundedSender<InputCommand>,
    approvals: mpsc::UnboundedSender<ApprovalCommand>,
    cancel: CancellationToken,
}

impl RunControl {
    pub async fn steer(&self, message: impl Into<String>) -> Result<(), AgentError> {
        self.submit(message.into(), InputOrigin::Steer).await
    }

    pub async fn queue(&self, message: impl Into<String>) -> Result<(), AgentError> {
        self.submit(message.into(), InputOrigin::Queue).await
    }

    async fn submit(&self, input: String, origin: InputOrigin) -> Result<(), AgentError> {
        if input.trim().is_empty() {
            return Err(AgentError::EmptyInput);
        }
        let (acknowledgement, accepted) = oneshot::channel();
        self.commands
            .send(InputCommand {
                input_id: InputId::random(),
                input,
                origin,
                acknowledgement,
            })
            .map_err(|error| AgentError::Task(error.to_string()))?;
        accepted
            .await
            .map_err(|_| AgentError::Task("run settled before input admission completed".into()))?
            .map_err(AgentError::Task)
    }

    pub async fn approve(&self, call_id: impl Into<String>, allow: bool) -> Result<(), AgentError> {
        let (acknowledgement, accepted) = oneshot::channel();
        self.approvals
            .send(ApprovalCommand {
                call_id: call_id.into(),
                allow,
                acknowledgement,
            })
            .map_err(|error| AgentError::Task(error.to_string()))?;
        accepted
            .await
            .map_err(|_| {
                AgentError::Task("run settled before tool authorization completed".into())
            })?
            .map_err(AgentError::Task)
    }

    pub fn abort(&self) {
        self.cancel.cancel();
    }
}

pub struct ActiveAgent {
    control: RunControl,
    events: mpsc::UnboundedReceiver<AgentEvent>,
    task: JoinHandle<Agent>,
}

impl ActiveAgent {
    pub fn control(&self) -> RunControl {
        self.control.clone()
    }

    pub async fn next_event(&mut self) -> Option<AgentEvent> {
        self.events.recv().await
    }

    pub async fn finish(mut self) -> Result<Agent, AgentError> {
        loop {
            tokio::select! {
                result = &mut self.task => {
                    return result.map_err(|error| AgentError::Task(error.to_string()));
                }
                event = self.events.recv() => {
                    if event.is_none() {
                        return (&mut self.task).await.map_err(|error| AgentError::Task(error.to_string()));
                    }
                }
            }
        }
    }
}

impl Drop for ActiveAgent {
    fn drop(&mut self) {
        // Dropping the capability must not detach an unobservable background
        // run whose closed command channels would otherwise remain readable.
        self.control.abort();
    }
}

fn user_items(message: String) -> Vec<InputItem> {
    vec![InputItem::from(EasyInputMessage::from(message))]
}

fn function_output(call: &FunctionToolCall, output: String) -> InputItem {
    InputItem::from(Item::from(FunctionCallOutputItemParam {
        call_id: call.call_id.clone(),
        output: output.into(),
        id: None,
        status: None,
    }))
}

fn response_info(response: &Response) -> ResponseInfo {
    ResponseInfo {
        id: response.id.clone(),
        model: response.model.clone(),
        usage: response
            .usage
            .as_ref()
            .map(crate::session_event::TokenUsage::from_provider),
    }
}

fn publish(sink: &EventSink, event: AgentEvent) {
    let _ = sink.send(event);
}

fn system_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn millis_to_seconds(millis: i64) -> u64 {
    u64::try_from(millis.max(0)).unwrap_or_default() / 1_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_store::AppendFailpoint;
    use async_openai::types::responses::FunctionTool;
    use futures_util::future::BoxFuture;
    use serde_json::json;
    use std::fs;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::oneshot;
    use tokio::time::timeout;

    fn test_agent() -> Agent {
        let mut agent = Agent::new(
            Model::new("test", "key", "http://127.0.0.1", "test-model", 128_000),
            "test instructions",
            Session::memory(),
            ".",
        );
        agent.writer = Some(agent.store.acquire_writer(&agent.info.id).unwrap());
        agent
    }

    #[test]
    fn wall_clock_jumps_do_not_change_monotonic_event_durations() {
        let before = observed_event_time("clock", Duration::from_millis(10), 10_000);
        let wall_moved_back = observed_event_time("clock", Duration::from_millis(20), 5_000);
        let wall_moved_forward = observed_event_time("clock", Duration::from_millis(35), 500_000);

        assert!(wall_moved_back.wall_time_ms < before.wall_time_ms);
        assert!(wall_moved_forward.wall_time_ms > before.wall_time_ms);
        assert_eq!(wall_moved_back.duration_since(&before), Some(10_000_000));
        assert_eq!(
            wall_moved_forward.duration_since(&wall_moved_back),
            Some(15_000_000)
        );
    }

    struct DelayTool;

    impl AgentTool for DelayTool {
        fn name(&self) -> &str {
            "delay"
        }

        fn schema(&self) -> Tool {
            Tool::Function(FunctionTool {
                name: "delay".into(),
                description: None,
                parameters: Some(json!({"type": "object"})),
                strict: Some(false),
                defer_loading: None,
            })
        }

        fn requires_approval(&self) -> bool {
            false
        }

        fn execute<'a>(
            &'a self,
            call: &'a FunctionToolCall,
            _env: &'a Env,
        ) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let delay =
                    serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap()["delay"]
                        .as_u64()
                        .unwrap();
                tokio::time::sleep(Duration::from_millis(delay)).await;
                ToolResult::ok(call.call_id.clone())
            })
        }
    }

    struct ApprovalDelayTool;

    impl AgentTool for ApprovalDelayTool {
        fn name(&self) -> &str {
            "approval_delay"
        }

        fn schema(&self) -> Tool {
            Tool::Function(FunctionTool {
                name: "approval_delay".into(),
                description: None,
                parameters: Some(json!({"type": "object"})),
                strict: Some(false),
                defer_loading: None,
            })
        }

        fn requires_approval(&self) -> bool {
            true
        }

        fn execute<'a>(
            &'a self,
            call: &'a FunctionToolCall,
            _env: &'a Env,
        ) -> BoxFuture<'a, ToolResult> {
            Box::pin(async move {
                let delay =
                    serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap()["delay"]
                        .as_u64()
                        .unwrap();
                tokio::time::sleep(Duration::from_millis(delay)).await;
                ToolResult::ok(call.call_id.clone())
            })
        }
    }

    fn tool_call(call_id: &str, item_id: &str, delay: u64) -> FunctionToolCall {
        named_tool_call(call_id, item_id, "delay", delay)
    }

    fn named_tool_call(call_id: &str, item_id: &str, name: &str, delay: u64) -> FunctionToolCall {
        FunctionToolCall {
            arguments: json!({"delay": delay}).to_string(),
            call_id: call_id.into(),
            namespace: None,
            name: name.into(),
            id: Some(item_id.into()),
            status: None,
        }
    }

    fn temp_directory(label: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("kcastle-agent-v2-{label}-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    async fn read_http_request(socket: &mut TcpStream) {
        let mut request = Vec::new();
        let (body_start, content_length) = loop {
            let mut chunk = [0; 4096];
            let bytes = socket.read(&mut chunk).await.unwrap();
            assert_ne!(bytes, 0);
            request.extend_from_slice(&chunk[..bytes]);
            let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then_some(value.trim())
                })
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap();
            break (header_end + 4, content_length);
        };
        while request.len() < body_start + content_length {
            let mut chunk = [0; 4096];
            let bytes = socket.read(&mut chunk).await.unwrap();
            assert_ne!(bytes, 0);
            request.extend_from_slice(&chunk[..bytes]);
        }
    }

    async fn write_text_stream_response(socket: &mut TcpStream, text: &str, response_id: &str) {
        let body = format!(
            "data: {{\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"{text}\"}}\n\ndata: {{\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{{\"created_at\":0,\"id\":\"{response_id}\",\"model\":\"test-model\",\"object\":\"response\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"annotations\":[],\"text\":\"{text}\"}}],\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\"}}],\"status\":\"completed\"}}}}\n\n"
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    }

    async fn text_stream_model(text: &'static str) -> (Model, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let (body_start, content_length) = loop {
                let mut chunk = [0; 4096];
                let bytes = socket.read(&mut chunk).await.unwrap();
                assert_ne!(bytes, 0);
                request.extend_from_slice(&chunk[..bytes]);
                let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then_some(value.trim())
                    })
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap();
                break (header_end + 4, content_length);
            };
            while request.len() < body_start + content_length {
                let mut chunk = [0; 4096];
                let bytes = socket.read(&mut chunk).await.unwrap();
                assert_ne!(bytes, 0);
                request.extend_from_slice(&chunk[..bytes]);
            }
            let body = format!(
                "data: {{\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"{text}\"}}\n\ndata: {{\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{{\"created_at\":0,\"id\":\"resp_1\",\"model\":\"test-model\",\"object\":\"response\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"annotations\":[],\"text\":\"{text}\"}}],\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\"}}],\"status\":\"completed\"}}}}\n\n"
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        (
            Model::new(
                "test",
                "key",
                format!("http://{address}"),
                "test-model",
                128_000,
            ),
            server,
        )
    }

    async fn gated_two_request_text_stream_model(
        text: &'static str,
    ) -> (
        Model,
        oneshot::Receiver<()>,
        oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (first_request_tx, first_request_rx) = oneshot::channel();
        let (release_first_tx, release_first_rx) = oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            read_http_request(&mut first).await;
            first_request_tx.send(()).unwrap();
            release_first_rx.await.unwrap();
            write_text_stream_response(&mut first, text, "resp_1").await;

            let (mut second, _) = listener.accept().await.unwrap();
            read_http_request(&mut second).await;
            write_text_stream_response(&mut second, text, "resp_2").await;
        });
        (
            Model::new(
                "test",
                "key",
                format!("http://{address}"),
                "test-model",
                128_000,
            ),
            first_request_rx,
            release_first_tx,
            server,
        )
    }

    async fn prepare_tool_request(
        agent: &mut Agent,
        calls: &[FunctionToolCall],
        events: &EventSink,
    ) {
        let run_id = RunId::random();
        let turn_id = TurnId::random();
        let step_id = StepId::random();
        let input_id = InputId::random();
        agent
            .commit_now(
                vec![
                    SessionEvent::InputSubmitted {
                        input_id: input_id.clone(),
                        input: "run tools".into(),
                        origin: InputOrigin::Initial,
                    },
                    SessionEvent::RunStarted {
                        run_id: run_id.clone(),
                    },
                    SessionEvent::TurnStarted {
                        run_id,
                        turn_id: turn_id.clone(),
                    },
                    SessionEvent::StepStarted {
                        turn_id,
                        step_id: step_id.clone(),
                    },
                    SessionEvent::InputAttached {
                        input_id,
                        step_id: step_id.clone(),
                        items: user_items("run tools".into()),
                    },
                ],
                events,
            )
            .await
            .unwrap();
        let request_id = RequestId::random();
        let tools = agent.tool_schemas();
        agent
            .commit_now(
                vec![
                    SessionEvent::RequestSnapshot {
                        request_id: request_id.clone(),
                        step_id,
                        reason: crate::RequestHeaderReason::Initial,
                        model: "test-model".into(),
                        instructions: Some("test instructions".into()),
                        tools,
                        reasoning_effort: None,
                        max_output_tokens: None,
                        session_config: SessionConfig::default(),
                    },
                    SessionEvent::ModelRequestStarted {
                        request_id: request_id.clone(),
                    },
                ],
                events,
            )
            .await
            .unwrap();
        let items = calls
            .iter()
            .cloned()
            .map(OutputItem::FunctionCall)
            .map(InputItem::from)
            .collect();
        let mut completed = vec![SessionEvent::AssistantCompleted {
            request_id: request_id.clone(),
            items,
            response: ResponseInfo {
                id: "response".into(),
                model: "test-model".into(),
                usage: None,
            },
        }];
        completed.extend(calls.iter().map(|call| SessionEvent::ToolCallRequested {
            request_id: request_id.clone(),
            call_id: CallId::from_raw(call.call_id.clone()),
            parent_call_id: None,
        }));
        agent.commit_now(completed, events).await.unwrap();
    }

    #[tokio::test]
    async fn machine_does_not_advance_when_commit_fails_before_sqlite_commit() {
        let mut agent = test_agent();
        agent
            .store
            .inject_failpoint(AppendFailpoint::BeforeCommitOnce);
        let (events, _) = mpsc::unbounded_channel();
        let before_seq = agent.machine.next_seq();
        let before_revision = agent.revision;
        let result = agent
            .commit_now(
                vec![SessionEvent::InputSubmitted {
                    input_id: InputId::random(),
                    input: "hello".into(),
                    origin: InputOrigin::Initial,
                }],
                &events,
            )
            .await;
        assert!(matches!(
            result,
            Err(AgentError::Store(SessionStoreError::InjectedBeforeCommit))
        ));
        assert_eq!(agent.machine.next_seq(), before_seq);
        assert_eq!(agent.revision, before_revision);
    }

    #[tokio::test]
    async fn ambiguous_commit_is_resolved_and_applied_exactly_once() {
        let mut agent = test_agent();
        agent
            .store
            .inject_failpoint(AppendFailpoint::AfterCommitBeforeReceiptOnce);
        let (events, mut received) = mpsc::unbounded_channel();
        let receipt = agent
            .commit_now(
                vec![SessionEvent::InputSubmitted {
                    input_id: InputId::random(),
                    input: "hello".into(),
                    origin: InputOrigin::Initial,
                }],
                &events,
            )
            .await
            .unwrap();
        assert_eq!(agent.revision, 1);
        assert_eq!(agent.machine.next_seq(), 1);
        assert!(matches!(
            received.recv().await,
            Some(AgentEvent::SessionCommitted(committed)) if committed == receipt
        ));
        assert_eq!(
            agent.store.load(&agent.info.id).unwrap().transactions.len(),
            1
        );
    }

    #[tokio::test]
    async fn failed_run_closes_request_step_turn_and_run_in_one_transaction() {
        let mut agent = test_agent();
        let (events, mut received) = mpsc::unbounded_channel();
        let run_id = RunId::random();
        let turn_id = TurnId::random();
        let step_id = StepId::random();
        let input_id = InputId::random();
        let tools = agent.tool_schemas();
        agent
            .commit_now(
                vec![
                    SessionEvent::InputSubmitted {
                        input_id: input_id.clone(),
                        input: "hello".into(),
                        origin: InputOrigin::Initial,
                    },
                    SessionEvent::RunStarted {
                        run_id: run_id.clone(),
                    },
                    SessionEvent::TurnStarted {
                        run_id,
                        turn_id: turn_id.clone(),
                    },
                    SessionEvent::StepStarted {
                        turn_id,
                        step_id: step_id.clone(),
                    },
                    SessionEvent::InputAttached {
                        input_id,
                        step_id: step_id.clone(),
                        items: user_items("hello".into()),
                    },
                ],
                &events,
            )
            .await
            .unwrap();
        agent
            .commit_now(
                vec![
                    SessionEvent::RequestSnapshot {
                        request_id: "request".into(),
                        step_id,
                        reason: crate::RequestHeaderReason::Initial,
                        model: "test-model".into(),
                        instructions: Some("test instructions".into()),
                        tools,
                        reasoning_effort: None,
                        max_output_tokens: None,
                        session_config: SessionConfig::default(),
                    },
                    SessionEvent::ModelRequestStarted {
                        request_id: "request".into(),
                    },
                ],
                &events,
            )
            .await
            .unwrap();
        while received.try_recv().is_ok() {}

        agent
            .terminate_after_error(
                false,
                &AgentError::ModelResponse("provider failed".into()),
                &events,
            )
            .await
            .unwrap();
        let AgentEvent::SessionCommitted(receipt) = received.recv().await.unwrap() else {
            panic!("terminal transaction must be published");
        };
        assert!(matches!(
            receipt.events.as_slice(),
            [
                crate::RecordedEvent {
                    event: SessionEvent::ModelRequestFailed { .. },
                    ..
                },
                crate::RecordedEvent {
                    event: SessionEvent::StepTerminated {
                        outcome: StepOutcome::Failed,
                        ..
                    },
                    ..
                },
                crate::RecordedEvent {
                    event: SessionEvent::TurnTerminated {
                        reason: TurnEndReason::Failed,
                        ..
                    },
                    ..
                },
                crate::RecordedEvent {
                    event: SessionEvent::RunTerminated {
                        outcome: RunOutcome::Failed,
                        ..
                    },
                    ..
                }
            ]
        ));
        assert!(agent.machine.active_run().is_none());
    }

    #[tokio::test]
    async fn run_persists_each_effect_intent_and_a_complete_terminal_history() {
        let (model, server) = text_stream_model("hello").await;
        let active = Agent::new(model, "test instructions", Session::memory(), ".").start("hi");
        let agent = active.finish().await.unwrap();
        server.await.unwrap();

        assert!(format!("{:?}", agent.transcript()).contains("hello"));
        assert!(agent.machine.active_run().is_none());
        let loaded = agent.store.load(&agent.info.id).unwrap();
        let request_transaction = loaded
            .transactions
            .iter()
            .find(|transaction| {
                transaction
                    .events
                    .iter()
                    .any(|event| matches!(event.event, SessionEvent::RequestSnapshot { .. }))
            })
            .unwrap();
        assert!(
            request_transaction
                .events
                .iter()
                .any(|event| { matches!(event.event, SessionEvent::ModelRequestStarted { .. }) })
        );
        assert!(loaded.events().any(|event| {
            matches!(
                event.event,
                SessionEvent::RunTerminated {
                    outcome: RunOutcome::Completed,
                    ..
                }
            )
        }));
    }

    #[tokio::test]
    async fn tool_finishes_follow_observation_order_but_results_attach_in_call_order() {
        let mut agent = test_agent();
        agent.tools = vec![Arc::new(DelayTool)];
        let (events, _) = mpsc::unbounded_channel();
        let run_id = RunId::random();
        let turn_id = TurnId::random();
        let step_id = StepId::random();
        let input_id = InputId::random();
        agent
            .commit_now(
                vec![
                    SessionEvent::InputSubmitted {
                        input_id: input_id.clone(),
                        input: "run tools".into(),
                        origin: InputOrigin::Initial,
                    },
                    SessionEvent::RunStarted {
                        run_id: run_id.clone(),
                    },
                    SessionEvent::TurnStarted {
                        run_id,
                        turn_id: turn_id.clone(),
                    },
                    SessionEvent::StepStarted {
                        turn_id,
                        step_id: step_id.clone(),
                    },
                    SessionEvent::InputAttached {
                        input_id,
                        step_id: step_id.clone(),
                        items: user_items("run tools".into()),
                    },
                ],
                &events,
            )
            .await
            .unwrap();
        let request_id = RequestId::random();
        let tools = agent.tool_schemas();
        agent
            .commit_now(
                vec![
                    SessionEvent::RequestSnapshot {
                        request_id: request_id.clone(),
                        step_id,
                        reason: crate::RequestHeaderReason::Initial,
                        model: "test-model".into(),
                        instructions: Some("test instructions".into()),
                        tools,
                        reasoning_effort: None,
                        max_output_tokens: None,
                        session_config: SessionConfig::default(),
                    },
                    SessionEvent::ModelRequestStarted {
                        request_id: request_id.clone(),
                    },
                ],
                &events,
            )
            .await
            .unwrap();
        let calls = vec![
            tool_call("slow", "item-slow", 35),
            tool_call("fast", "item-fast", 1),
        ];
        let items = calls
            .iter()
            .cloned()
            .map(OutputItem::FunctionCall)
            .map(InputItem::from)
            .collect();
        agent
            .commit_now(
                vec![
                    SessionEvent::AssistantCompleted {
                        request_id: request_id.clone(),
                        items,
                        response: ResponseInfo {
                            id: "response".into(),
                            model: "test-model".into(),
                            usage: None,
                        },
                    },
                    SessionEvent::ToolCallRequested {
                        request_id: request_id.clone(),
                        call_id: "slow".into(),
                        parent_call_id: None,
                    },
                    SessionEvent::ToolCallRequested {
                        request_id,
                        call_id: "fast".into(),
                        parent_call_id: None,
                    },
                ],
                &events,
            )
            .await
            .unwrap();
        let (_command_tx, command_rx) = mpsc::unbounded_channel();
        let (_approval_tx, approval_rx) = mpsc::unbounded_channel();
        let mut channels = RunChannels {
            commands: command_rx,
            approvals: approval_rx,
            cancel: CancellationToken::new(),
        };
        agent
            .execute_tools(&calls, &mut channels, &events)
            .await
            .unwrap();

        let loaded = agent.store.load(&agent.info.id).unwrap();
        let finished = loaded
            .events()
            .filter_map(|event| match &event.event {
                SessionEvent::ToolExecutionFinished { call_id, .. } => {
                    Some(call_id.as_str().to_owned())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let attached = loaded
            .events()
            .filter_map(|event| match &event.event {
                SessionEvent::ToolResultAttached { call_id, .. } => {
                    Some(call_id.as_str().to_owned())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(finished, ["fast", "slow"]);
        assert_eq!(attached, ["slow", "fast"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tool_authorizations_keep_individual_observation_times_and_dispatch_after_all() {
        let mut agent = test_agent();
        agent.tools = vec![Arc::new(ApprovalDelayTool), Arc::new(DelayTool)];
        let calls = vec![
            named_tool_call("approved-slow", "item-approved-slow", "approval_delay", 35),
            named_tool_call("approved-fast", "item-approved-fast", "approval_delay", 1),
            named_tool_call("not-required", "item-not-required", "delay", 1),
            named_tool_call("not-found", "item-not-found", "missing", 0),
        ];
        let (events, mut received) = mpsc::unbounded_channel();
        prepare_tool_request(&mut agent, &calls, &events).await;
        while received.try_recv().is_ok() {}

        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        let mut channels = RunChannels {
            commands: command_rx,
            approvals: approval_rx,
            cancel: CancellationToken::new(),
        };
        let execution_calls = calls.clone();
        let execution = tokio::spawn(async move {
            agent
                .execute_tools(&execution_calls, &mut channels, &events)
                .await
                .unwrap();
            agent
        });

        let mut requested = Vec::new();
        while requested.len() < 2 {
            let event = timeout(Duration::from_secs(2), received.recv())
                .await
                .expect("approval request must be published")
                .expect("event stream must stay open");
            if let AgentEvent::ApprovalRequired(call) = event {
                requested.push(call.call_id);
            }
        }
        assert_eq!(requested, ["approved-slow", "approved-fast"]);

        tokio::time::sleep(Duration::from_millis(12)).await;
        let (slow_acknowledgement, slow_accepted) = oneshot::channel();
        approval_tx
            .send(ApprovalCommand {
                call_id: "approved-slow".into(),
                allow: true,
                acknowledgement: slow_acknowledgement,
            })
            .unwrap();
        slow_accepted.await.unwrap().unwrap();
        tokio::time::sleep(Duration::from_millis(24)).await;
        let (fast_acknowledgement, fast_accepted) = oneshot::channel();
        approval_tx
            .send(ApprovalCommand {
                call_id: "approved-fast".into(),
                allow: true,
                acknowledgement: fast_acknowledgement,
            })
            .unwrap();
        fast_accepted.await.unwrap().unwrap();
        let agent = timeout(Duration::from_secs(3), execution)
            .await
            .expect("tool execution must settle")
            .unwrap();
        drop(command_tx);
        drop(approval_tx);

        let loaded = agent.store.load(&agent.info.id).unwrap();
        let mut authorizations = HashMap::new();
        let mut starts = HashMap::new();
        let mut attached = Vec::new();
        for recorded in loaded.events() {
            match &recorded.event {
                SessionEvent::ToolAuthorizationResolved { call_id, decision } => {
                    authorizations.insert(
                        call_id.as_str().to_owned(),
                        (*decision, recorded.time.clone()),
                    );
                }
                SessionEvent::ToolExecutionStarted { call_id } => {
                    starts.insert(call_id.as_str().to_owned(), recorded.time.clone());
                }
                SessionEvent::ToolResultAttached { call_id, .. } => {
                    attached.push(call_id.as_str().to_owned());
                }
                _ => {}
            }
        }

        assert_eq!(authorizations.len(), calls.len());
        let (slow_decision, slow_authorized) = &authorizations["approved-slow"];
        let (fast_decision, fast_authorized) = &authorizations["approved-fast"];
        let (automatic_decision, automatic_authorized) = &authorizations["not-required"];
        let (missing_decision, missing_authorized) = &authorizations["not-found"];
        assert_eq!(*slow_decision, ToolAuthorizationDecision::Allowed);
        assert_eq!(*fast_decision, ToolAuthorizationDecision::Allowed);
        assert_eq!(*automatic_decision, ToolAuthorizationDecision::NotRequired);
        assert_eq!(*missing_decision, ToolAuthorizationDecision::Unavailable);
        assert_eq!(slow_authorized.clock_id, fast_authorized.clock_id);
        assert_eq!(slow_authorized.clock_id, automatic_authorized.clock_id);
        assert_eq!(slow_authorized.clock_id, missing_authorized.clock_id);
        assert!(slow_authorized.monotonic_ns < fast_authorized.monotonic_ns);
        assert!(slow_authorized.wall_time_ms < fast_authorized.wall_time_ms);
        assert!(automatic_authorized.monotonic_ns <= slow_authorized.monotonic_ns);
        assert!(missing_authorized.monotonic_ns <= slow_authorized.monotonic_ns);

        let last_authorization = authorizations
            .values()
            .map(|(_, time)| time.monotonic_ns)
            .max()
            .unwrap();
        assert_eq!(starts.len(), 3);
        assert!(!starts.contains_key("not-found"));
        for started in starts.values() {
            assert_eq!(started.clock_id, slow_authorized.clock_id);
            assert!(started.monotonic_ns >= last_authorization);
        }
        assert_eq!(
            attached,
            [
                "approved-slow",
                "approved-fast",
                "not-required",
                "not-found"
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn approvals_are_scoped_to_interactive_calls_and_retries_must_match() {
        let mut agent = test_agent();
        agent.tools = vec![Arc::new(ApprovalDelayTool), Arc::new(DelayTool)];
        let calls = vec![
            named_tool_call("interactive-a", "item-a", "approval_delay", 1),
            named_tool_call("interactive-b", "item-b", "approval_delay", 1),
            named_tool_call("automatic", "item-automatic", "delay", 1),
            named_tool_call("unavailable", "item-unavailable", "missing", 0),
        ];
        let (events, mut received) = mpsc::unbounded_channel();
        prepare_tool_request(&mut agent, &calls, &events).await;
        while received.try_recv().is_ok() {}

        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        let cancellation = CancellationToken::new();
        let control = RunControl {
            commands: command_tx,
            approvals: approval_tx,
            cancel: cancellation.clone(),
        };
        let mut channels = RunChannels {
            commands: command_rx,
            approvals: approval_rx,
            cancel: cancellation,
        };
        let execution_calls = calls.clone();
        let execution_events = events.clone();
        let execution = tokio::spawn(async move {
            agent
                .execute_tools(&execution_calls, &mut channels, &execution_events)
                .await
                .unwrap();
            agent
        });

        let mut requested = Vec::new();
        while requested.len() < 2 {
            let event = timeout(Duration::from_secs(2), received.recv())
                .await
                .expect("interactive approval requests must arrive")
                .expect("event stream must remain open");
            if let AgentEvent::ApprovalRequired(call) = event {
                requested.push(call.call_id);
            }
        }
        assert_eq!(requested, ["interactive-a", "interactive-b"]);

        control.approve("interactive-a", true).await.unwrap();
        control
            .approve("interactive-a", true)
            .await
            .expect("an identical retry is idempotent while the approval phase is open");
        let error = control.approve("interactive-a", false).await.unwrap_err();
        assert!(error.to_string().contains("already resolved as Allowed"));

        for call_id in ["automatic", "unavailable", "unknown"] {
            let error = control.approve(call_id, true).await.unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("did not request interactive authorization"),
                "unexpected error for {call_id}: {error}"
            );
        }
        control.approve("interactive-b", false).await.unwrap();

        let agent = timeout(Duration::from_secs(3), execution)
            .await
            .expect("tool execution must settle")
            .unwrap();
        let authorizations = agent
            .store
            .load(&agent.info.id)
            .unwrap()
            .events()
            .filter_map(|event| match &event.event {
                SessionEvent::ToolAuthorizationResolved { call_id, decision }
                    if call_id == &CallId::from("interactive-a") =>
                {
                    Some(*decision)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(authorizations, [ToolAuthorizationDecision::Allowed]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn approval_retry_is_acknowledged_while_the_only_approved_tool_is_running() {
        let mut agent = test_agent();
        agent.tools = vec![Arc::new(ApprovalDelayTool)];
        let calls = vec![named_tool_call(
            "interactive",
            "item-interactive",
            "approval_delay",
            1_000,
        )];
        let (events, mut received) = mpsc::unbounded_channel();
        prepare_tool_request(&mut agent, &calls, &events).await;
        while received.try_recv().is_ok() {}

        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        let cancellation = CancellationToken::new();
        let control = RunControl {
            commands: command_tx,
            approvals: approval_tx,
            cancel: cancellation.clone(),
        };
        let mut channels = RunChannels {
            commands: command_rx,
            approvals: approval_rx,
            cancel: cancellation,
        };
        let execution_calls = calls.clone();
        let execution_events = events.clone();
        let execution = tokio::spawn(async move {
            agent
                .execute_tools(&execution_calls, &mut channels, &execution_events)
                .await
                .unwrap();
            agent
        });

        loop {
            if matches!(
                timeout(Duration::from_secs(2), received.recv())
                    .await
                    .expect("approval request must arrive"),
                Some(AgentEvent::ApprovalRequired(_))
            ) {
                break;
            }
        }
        control.approve("interactive", true).await.unwrap();
        loop {
            let event = timeout(Duration::from_secs(2), received.recv())
                .await
                .expect("dispatch receipt must arrive")
                .expect("event stream must remain open");
            if matches!(
                event,
                AgentEvent::SessionCommitted(ref receipt)
                    if receipt.events.iter().any(|recorded| matches!(
                        &recorded.event,
                        SessionEvent::ToolDispatchIntended { call_id }
                            if call_id == &CallId::from("interactive")
                    ))
            ) {
                break;
            }
        }

        timeout(
            Duration::from_millis(100),
            control.approve("interactive", true),
        )
        .await
        .expect("an acknowledgement retry must not wait for the tool")
        .expect("the persisted decision makes an identical retry idempotent");
        let conflict = timeout(
            Duration::from_millis(100),
            control.approve("interactive", false),
        )
        .await
        .expect("a conflicting retry must not wait for the tool")
        .unwrap_err();
        assert!(conflict.to_string().contains("already resolved as Allowed"));

        timeout(Duration::from_secs(3), execution)
            .await
            .expect("tool execution must settle")
            .unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn approved_tool_is_durable_when_another_approval_is_aborted() {
        let mut agent = test_agent();
        agent.tools = vec![Arc::new(ApprovalDelayTool)];
        let calls = vec![
            named_tool_call("approved", "item-approved", "approval_delay", 1),
            named_tool_call("pending", "item-pending", "approval_delay", 1),
        ];
        let (events, mut received) = mpsc::unbounded_channel();
        prepare_tool_request(&mut agent, &calls, &events).await;
        while received.try_recv().is_ok() {}

        let (_command_tx, command_rx) = mpsc::unbounded_channel();
        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        let cancellation = CancellationToken::new();
        let mut channels = RunChannels {
            commands: command_rx,
            approvals: approval_rx,
            cancel: cancellation.clone(),
        };
        let execution_calls = calls.clone();
        let execution_events = events.clone();
        let execution = tokio::spawn(async move {
            let result = agent
                .execute_tools(&execution_calls, &mut channels, &execution_events)
                .await;
            (agent, result)
        });

        let mut requested = Vec::new();
        while requested.len() < 2 {
            let event = timeout(Duration::from_secs(2), received.recv())
                .await
                .expect("approval requests must arrive")
                .expect("event stream must remain open");
            if let AgentEvent::ApprovalRequired(call) = event {
                requested.push(call.call_id);
            }
        }
        assert_eq!(requested, ["approved", "pending"]);

        let (acknowledgement, accepted) = oneshot::channel();
        approval_tx
            .send(ApprovalCommand {
                call_id: "approved".into(),
                allow: true,
                acknowledgement,
            })
            .unwrap();
        accepted.await.unwrap().unwrap();
        cancellation.cancel();

        let (mut agent, result) = execution.await.unwrap();
        assert!(matches!(result, Err(AgentError::Aborted)));
        agent
            .terminate_after_error(true, &AgentError::Aborted, &events)
            .await
            .unwrap();

        let loaded = agent.store.load(&agent.info.id).unwrap();
        let mut authorizations = HashMap::new();
        let mut statuses = HashMap::new();
        let mut starts = Vec::new();
        for recorded in loaded.events() {
            match &recorded.event {
                SessionEvent::ToolAuthorizationResolved { call_id, decision } => {
                    authorizations.insert(call_id.as_str().to_owned(), *decision);
                }
                SessionEvent::ToolExecutionStarted { call_id } => {
                    starts.push(call_id.as_str().to_owned());
                }
                SessionEvent::ToolResultAttached {
                    call_id, status, ..
                } => {
                    statuses.insert(call_id.as_str().to_owned(), *status);
                }
                _ => {}
            }
        }
        assert_eq!(
            authorizations.get("approved"),
            Some(&ToolAuthorizationDecision::Allowed)
        );
        assert_eq!(
            authorizations.get("pending"),
            Some(&ToolAuthorizationDecision::Aborted)
        );
        assert!(starts.is_empty());
        assert_eq!(
            statuses.get("approved"),
            Some(&ToolResultStatus::AbortedBeforeDispatch)
        );
        assert_eq!(
            statuses.get("pending"),
            Some(&ToolResultStatus::AbortedBeforeDispatch)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn active_disk_writer_excludes_a_second_agent_without_blocking_readers() {
        let directory = temp_directory("writer-lifecycle");
        let session_a = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let info = session_a.info().clone();
        let session_b = Session::open_in_project(&info.path, "project-a")
            .await
            .unwrap();
        let (model, first_request, release_first, server) =
            gated_two_request_text_stream_model("done").await;
        let agent_a = Agent::new(model.clone(), "test instructions", session_a, ".");
        let agent_b = Agent::new(model, "test instructions", session_b, ".");

        let active_a = agent_a.start("first");
        timeout(Duration::from_secs(3), first_request)
            .await
            .expect("first model request must reach the local server")
            .expect("local server must signal the first request");

        let mut active_b = agent_b.start("contender");
        let failure = timeout(Duration::from_secs(2), active_b.next_event())
            .await
            .expect("contending agent must fail promptly")
            .expect("contending agent must publish its failure");
        assert!(
            matches!(failure, AgentEvent::RunFailed(message) if message.contains("active writer"))
        );
        let mut agent_b = active_b.finish().await.unwrap();

        let error = agent_b
            .rename_session("must not acquire while A is active")
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            AgentError::Store(SessionStoreError::WriterBusy { .. })
        ));

        let snapshot = Session::inspect(&info.path).unwrap();
        assert!(snapshot.recovery_needed());
        assert!(
            snapshot
                .events()
                .iter()
                .any(|event| matches!(event.event, SessionEvent::ModelRequestStarted { .. }))
        );
        assert!(!snapshot.events().iter().any(|event| matches!(
            event.event,
            SessionEvent::ModelRequestFailed { .. } | SessionEvent::RunTerminated { .. }
        )));
        let export = directory.join("while-active.jsonl");
        snapshot.export_jsonl(&export).unwrap();
        assert!(fs::metadata(&export).unwrap().len() > 0);
        drop(snapshot);

        release_first.send(()).unwrap();
        let agent_a = timeout(Duration::from_secs(3), active_a.finish())
            .await
            .expect("first agent must finish")
            .unwrap();
        let first_agent_revision = agent_a.revision;
        let first_agent_next_seq = agent_a.machine.next_seq();

        agent_b
            .rename_session("writer reacquired after A finished")
            .await
            .unwrap();
        let mut active_b = agent_b.start("second");
        let mut receipts = Vec::new();
        loop {
            let event = timeout(Duration::from_secs(3), active_b.next_event())
                .await
                .expect("second agent event stream must make progress")
                .expect("second agent event stream must remain open through completion");
            match event {
                AgentEvent::SessionCommitted(receipt) => receipts.push(receipt),
                AgentEvent::RunFinished(_) => break,
                AgentEvent::RunFailed(error) => panic!("second agent unexpectedly failed: {error}"),
                _ => {}
            }
        }
        let agent_b = timeout(Duration::from_secs(3), active_b.finish())
            .await
            .expect("second agent must reload and run after A releases the writer")
            .unwrap();
        server.await.unwrap();

        assert!(receipts.len() > first_agent_revision as usize);
        let mut expected_revision = 1_u64;
        let mut expected_seq = 0_u64;
        for receipt in &receipts {
            assert_eq!(receipt.base_revision, expected_revision - 1);
            assert_eq!(receipt.revision, expected_revision);
            for event in &receipt.events {
                assert_eq!(event.seq, expected_seq);
                expected_seq += 1;
            }
            expected_revision += 1;
        }
        assert_eq!(
            receipts[first_agent_revision as usize - 1].revision,
            first_agent_revision
        );
        assert_eq!(
            receipts[first_agent_revision as usize - 1]
                .events
                .last()
                .unwrap()
                .seq
                + 1,
            first_agent_next_seq
        );
        assert_eq!(
            receipts[first_agent_revision as usize].base_revision,
            first_agent_revision
        );
        assert_eq!(expected_revision, agent_b.revision + 1);
        assert_eq!(expected_seq, agent_b.machine.next_seq());

        let final_snapshot = Session::inspect(&info.path).unwrap();
        let inputs = final_snapshot
            .events()
            .iter()
            .filter_map(|event| match &event.event {
                SessionEvent::InputSubmitted { input, .. } => Some(input.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(inputs, ["first", "second"]);
        assert_eq!(
            final_snapshot
                .events()
                .iter()
                .filter(|event| matches!(
                    event.event,
                    SessionEvent::RunTerminated {
                        outcome: RunOutcome::Completed,
                        ..
                    }
                ))
                .count(),
            2
        );
        assert!(!final_snapshot.events().iter().any(|event| matches!(
            event.event,
            SessionEvent::RunTerminated {
                outcome: RunOutcome::Failed | RunOutcome::Aborted,
                ..
            }
        )));

        drop(final_snapshot);
        drop(agent_a);
        drop(agent_b);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_is_durably_acknowledged_while_stream_dispatch_waits_for_headers() {
        let session = Session::memory();
        let (model, first_request, release_first, server) =
            gated_two_request_text_stream_model("done").await;
        let agent = Agent::new(model, "test instructions", session, ".");
        let active = agent.start("first");
        let control = active.control();

        timeout(Duration::from_secs(3), first_request)
            .await
            .expect("model request must reach the local server")
            .expect("local server must signal the request before sending headers");
        timeout(Duration::from_millis(100), control.queue("queued"))
            .await
            .expect("queue admission must not wait for response headers")
            .expect("queue admission must commit durably");

        release_first.send(()).unwrap();
        let agent = timeout(Duration::from_secs(3), active.finish())
            .await
            .expect("both model requests must settle")
            .unwrap();
        server.await.unwrap();
        let loaded = agent.store.load(&agent.info.id).unwrap();
        let inputs = loaded
            .events()
            .filter_map(|event| match &event.event {
                SessionEvent::InputSubmitted { input, .. } => Some(input.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(inputs, ["first", "queued"]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_agent_rejects_external_config_drift_without_writing_journal() {
        let directory = temp_directory("config-drift");
        let session_a = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let info = session_a.info().clone();
        let session_b = Session::open_in_project(&info.path, "project-a")
            .await
            .unwrap();
        let model = Model::new("test", "key", "http://127.0.0.1:1", "test-model", 128_000);
        let mut agent_a = Agent::new(model.clone(), "test instructions", session_a, ".");
        let agent_b = Agent::new(model, "test instructions", session_b, ".");
        let changed = SessionConfig {
            model_id: Some("externally-selected-model".into()),
            reasoning_effort: Some("high".into()),
            allow_all_tools: true,
        };
        agent_a.persist_session_config(&changed).await.unwrap();
        let before = Session::inspect(&info.path).unwrap();
        assert_eq!(before.config(), &changed);
        assert!(before.events().is_empty());
        drop(before);

        let mut active_b = agent_b.start("must not be committed");
        let failure = timeout(Duration::from_secs(2), active_b.next_event())
            .await
            .expect("stale agent must fail promptly")
            .expect("stale agent must publish its failure");
        assert!(matches!(
            failure,
            AgentEvent::RunFailed(message)
                if message.contains("configuration changed") && message.contains("reopen")
        ));
        let agent_b = active_b.finish().await.unwrap();

        let after = Session::inspect(&info.path).unwrap();
        assert_eq!(after.config(), &changed);
        assert!(after.events().is_empty());

        drop(after);
        drop(agent_a);
        drop(agent_b);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn run_control_rejects_empty_inputs_before_admission() {
        let (commands, _) = mpsc::unbounded_channel();
        let (approvals, _) = mpsc::unbounded_channel();
        let control = RunControl {
            commands,
            approvals,
            cancel: CancellationToken::new(),
        };
        assert!(matches!(
            control.queue("  ").await,
            Err(AgentError::EmptyInput)
        ));
        assert!(matches!(
            control.steer("\n").await,
            Err(AgentError::EmptyInput)
        ));
    }

    #[tokio::test]
    async fn run_control_reports_error_when_settlement_drops_an_unacknowledged_input() {
        let (commands, mut received) = mpsc::unbounded_channel();
        let (approvals, _) = mpsc::unbounded_channel();
        let control = RunControl {
            commands,
            approvals,
            cancel: CancellationToken::new(),
        };

        let late = tokio::spawn(async move { control.queue("late input").await });
        let command = received.recv().await.expect("input must reach the owner");
        drop(command);
        drop(received);

        let error = late.await.unwrap().unwrap_err();
        assert!(error.to_string().contains("settled before input admission"));
    }
}
