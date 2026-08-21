use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
pub use async_openai::types::responses::ReasoningEffort;
use async_openai::types::responses::{
    CreateResponseArgs, FunctionCallOutputItemParam, FunctionToolCall, InputItem, Item, OutputItem,
    Reasoning, ResponseStreamEvent, ResponseUsage, Tool,
};
use futures_util::StreamExt;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::compaction::{
    CompactionConfig, SUMMARY_INSTRUCTIONS, context_tokens, prepare_compaction,
};
use crate::session::{
    EventClock, InputMode, Session, SessionConfig, SessionError, SessionInfo, StateCommit,
};
use crate::session_event::{
    AssistantChunk, RecordedEvent, RequestHeaderReason, SessionEvent, StepOutcome, SurfaceOp,
    ToolExecutionOutcome, ToolResultStatus, TurnEndReason, UserMessageMode,
};
use crate::state::{ResponseMetadata, State, TranscriptItem};
use crate::tool::{AgentTool, Env, ShellTool, ToolResult};

const DEFAULT_MAX_TURNS: usize = 100;

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
    #[error("state error: {0}")]
    State(String),
    #[error(transparent)]
    OpenAI(#[from] OpenAIError),
    #[error(transparent)]
    Session(#[from] SessionError),
    #[error("agent task failed: {0}")]
    Task(String),
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub output: String,
    pub response_id: String,
    pub usage: Option<ResponseUsage>,
}

#[derive(Debug, Clone)]
pub enum AgentEvent {
    SessionEvent(RecordedEvent),
    RunStarted(String),
    ModelStarted(usize),
    ReasoningDelta(String),
    TextDelta(String),
    ApprovalRequired(FunctionToolCall),
    ToolStarted(FunctionToolCall),
    ToolFinished {
        call: FunctionToolCall,
        result: ToolResult,
    },
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
    InputAdmitted {
        id: String,
        input: String,
        mode: InputMode,
    },
}

pub struct Agent {
    model: Model,
    instructions: String,
    state: State,
    commit: Box<dyn StateCommit>,
    env: Env,
    tools: Vec<Arc<dyn AgentTool>>,
    compaction: Option<CompactionConfig>,
    max_turns: usize,
    pending_steer: VecDeque<PendingInput>,
    pending_queue: VecDeque<PendingInput>,
    next_turn: u32,
    has_request_header: bool,
    last_request_fingerprint: Option<String>,
    active_turn: Option<u32>,
    active_step: Option<u32>,
    active_compaction: Option<ActiveCompaction>,
    session_config: SessionConfig,
    open_tools: HashMap<String, (u32, u32, u64)>,
    surface_entries: Vec<(u64, u64, bool)>,
    event_clock: EventClock,
}

#[derive(Clone, Debug)]
struct ActiveCompaction {
    id: String,
    start_seq: u64,
    tokens_before: usize,
    first_kept_id: u64,
}

impl Agent {
    pub fn new(
        model: Model,
        instructions: impl Into<String>,
        session: Session,
        cwd: impl Into<PathBuf>,
    ) -> Self {
        let context_window = model.context_window();
        let pending = session.pending_inputs();
        let next_turn = session
            .events()
            .iter()
            .filter_map(|event| event_turn(&event.event))
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let has_request_header = session
            .events()
            .iter()
            .any(|event| matches!(event.event, SessionEvent::RequestHeader { .. }));
        let last_request_fingerprint = last_request_fingerprint(session.events());
        let session_config = session.config().clone();
        let open_tools = open_tools_from_events(session.events());
        let surface_entries = surface_entries_from_events(session.events());
        let (active_turn, active_step) = active_lifecycle_from_events(session.events());
        let active_compaction = active_compaction_from_events(session.events());
        let (state, mut commit) = session.into_parts();
        commit.release_writer();
        let pending_steer = pending
            .iter()
            .filter(|(_, _, mode)| *mode == InputMode::Steer)
            .map(|(id, input, _)| PendingInput {
                id: id.clone(),
                input: input.clone(),
            })
            .collect();
        let pending_queue = pending
            .into_iter()
            .filter(|(_, _, mode)| *mode == InputMode::Queue)
            .map(|(id, input, _)| PendingInput { id, input })
            .collect();
        Self {
            model,
            instructions: instructions.into(),
            state,
            commit,
            env: Env { cwd: cwd.into() },
            tools: vec![Arc::new(ShellTool)],
            compaction: Some(CompactionConfig::new(context_window)),
            max_turns: DEFAULT_MAX_TURNS,
            pending_steer,
            pending_queue,
            next_turn,
            has_request_header,
            last_request_fingerprint,
            active_turn,
            active_step,
            active_compaction,
            session_config,
            open_tools,
            surface_entries,
            event_clock: EventClock::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        model: Model,
        instructions: impl Into<String>,
        state: State,
        commit: Box<dyn StateCommit>,
        cwd: impl Into<PathBuf>,
        tools: Vec<Arc<dyn AgentTool>>,
        compaction: Option<CompactionConfig>,
        max_turns: usize,
    ) -> Result<Self, AgentError> {
        if max_turns == 0 {
            return Err(AgentError::MaxTurns(0));
        }
        Ok(Self {
            model,
            instructions: instructions.into(),
            state,
            commit,
            env: Env { cwd: cwd.into() },
            tools,
            compaction,
            max_turns,
            pending_steer: VecDeque::new(),
            pending_queue: VecDeque::new(),
            next_turn: 1,
            has_request_header: false,
            last_request_fingerprint: None,
            active_turn: None,
            active_step: None,
            active_compaction: None,
            session_config: SessionConfig::default(),
            open_tools: HashMap::new(),
            surface_entries: Vec::new(),
            event_clock: EventClock::new(),
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn session_info(&self) -> &SessionInfo {
        self.commit.info()
    }

    pub fn transcript(&self) -> Vec<TranscriptItem> {
        self.state.transcript()
    }

    pub fn latest_usage(&self) -> Option<&ResponseUsage> {
        self.state
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

    fn reasoning(&self) -> Option<Reasoning> {
        self.model.reasoning_effort.clone().map(|effort| Reasoning {
            effort: Some(effort),
            summary: None,
        })
    }

    pub fn set_session(&mut self, session: Session) {
        self.next_turn = session
            .events()
            .iter()
            .filter_map(|event| event_turn(&event.event))
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        self.has_request_header = session
            .events()
            .iter()
            .any(|event| matches!(event.event, SessionEvent::RequestHeader { .. }));
        self.last_request_fingerprint = last_request_fingerprint(session.events());
        self.session_config = session.config().clone();
        self.open_tools = open_tools_from_events(session.events());
        self.surface_entries = surface_entries_from_events(session.events());
        self.event_clock = EventClock::new();
        (self.active_turn, self.active_step) = active_lifecycle_from_events(session.events());
        self.active_compaction = active_compaction_from_events(session.events());
        let pending = session.pending_inputs();
        self.pending_steer = pending
            .iter()
            .filter(|(_, _, mode)| *mode == InputMode::Steer)
            .map(|(id, input, _)| PendingInput {
                id: id.clone(),
                input: input.clone(),
            })
            .collect();
        self.pending_queue = pending
            .into_iter()
            .filter(|(_, _, mode)| *mode == InputMode::Queue)
            .map(|(id, input, _)| PendingInput { id, input })
            .collect();
        (self.state, self.commit) = session.into_parts();
        self.commit.release_writer();
    }

    pub fn set_cwd(&mut self, cwd: impl Into<PathBuf>) {
        self.env.cwd = cwd.into();
    }

    pub async fn rename_session(&mut self, title: &str) -> Result<(), AgentError> {
        let result = async {
            self.commit.prepare(&self.state).await?;
            self.commit.rename(title).await?;
            Ok(())
        }
        .await;
        self.commit.release_writer();
        result
    }

    pub async fn persist_session_config(
        &mut self,
        config: &SessionConfig,
    ) -> Result<(), AgentError> {
        let result = async {
            self.commit.prepare(&self.state).await?;
            self.commit.set_config(config).await?;
            Ok(())
        }
        .await;
        self.commit.release_writer();
        if result.is_ok() {
            self.session_config = config.clone();
        }
        result
    }

    pub fn release_session_writer(&mut self) {
        self.commit.release_writer();
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
        self.spawn(move |mut agent, channels, events| async move {
            if let Err(error) = agent.commit.prepare(&agent.state).await {
                return Err((agent, error.into()));
            }
            if let Err(error) = agent.close_interrupted_compaction(&events).await {
                return Err((agent, error));
            }
            if let Err(error) = agent.close_unresolved_tools(&events).await {
                return Err((agent, error));
            }
            if agent.active_turn.is_some() {
                agent
                    .close_active_lifecycle(
                        &events,
                        StepOutcome::Aborted,
                        TurnEndReason::Aborted,
                        Some("Recovered an interrupted run".into()),
                    )
                    .await;
            }
            match agent
                .compact_once(true, instructions.as_deref(), &channels.cancel, &events)
                .await
            {
                Ok(()) => Ok(agent),
                Err(error) => Err((agent, error)),
            }
        })
    }

    fn spawn<F, Fut>(self, operation: F) -> ActiveAgent
    where
        F: FnOnce(Agent, RunChannels, mpsc::Sender<AgentEvent>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<Agent, (Agent, AgentError)>> + Send + 'static,
    {
        let (steer_tx, steer_rx) = mpsc::unbounded_channel();
        let (queue_tx, queue_rx) = mpsc::unbounded_channel();
        let (approval_tx, approval_rx) = mpsc::unbounded_channel();
        let (events_tx, events_rx) = mpsc::channel(256);
        let cancel = CancellationToken::new();
        let control = RunControl {
            steer: steer_tx,
            queue: queue_tx,
            approvals: approval_tx,
            cancel: cancel.clone(),
        };
        let channels = RunChannels {
            steer: steer_rx,
            queue: queue_rx,
            approvals: approval_rx,
            cancel,
        };
        let task = tokio::spawn(async move {
            let mut agent = match operation(self, channels, events_tx.clone()).await {
                Ok(agent) => agent,
                Err((mut agent, error)) => {
                    if matches!(error, AgentError::Aborted) {
                        match agent.close_unresolved_tools(&events_tx).await {
                            Ok(()) => {
                                agent
                                    .close_active_lifecycle(
                                        &events_tx,
                                        StepOutcome::Aborted,
                                        TurnEndReason::Aborted,
                                        None,
                                    )
                                    .await;
                                let _ = agent.emit(&events_tx, AgentEvent::RunAborted).await;
                            }
                            Err(cleanup_error) => {
                                let _ = agent
                                    .emit(
                                        &events_tx,
                                        AgentEvent::RunFailed(format!(
                                            "abort cleanup failed: {cleanup_error}"
                                        )),
                                    )
                                    .await;
                            }
                        }
                    } else {
                        let _ = agent.close_unresolved_tools(&events_tx).await;
                        let message = error.to_string();
                        let reason = if matches!(error, AgentError::MaxTurns(_)) {
                            TurnEndReason::MaxTurns
                        } else {
                            TurnEndReason::Failed
                        };
                        agent
                            .close_active_lifecycle(
                                &events_tx,
                                StepOutcome::Failed,
                                reason,
                                Some(message.clone()),
                            )
                            .await;
                        let _ = agent.emit(&events_tx, AgentEvent::RunFailed(message)).await;
                    }
                    agent
                }
            };
            agent.release_session_writer();
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
        events: mpsc::Sender<AgentEvent>,
    ) -> Result<Self, (Self, AgentError)> {
        if input.trim().is_empty() {
            return Err((self, AgentError::EmptyInput));
        }
        if let Err(error) = self.commit.prepare(&self.state).await {
            return Err((self, error.into()));
        }
        if let Err(error) = self.close_interrupted_compaction(&events).await {
            return Err((self, error));
        }
        if let Err(error) = self.close_unresolved_tools(&events).await {
            return Err((self, error));
        }
        if self.active_turn.is_some() {
            self.close_active_lifecycle(
                &events,
                StepOutcome::Aborted,
                TurnEndReason::Aborted,
                Some("Recovered an interrupted run".into()),
            )
            .await;
        }
        if let Err(error) = self
            .emit(&events, AgentEvent::RunStarted(input.clone()))
            .await
        {
            return Err((self, AgentError::Task(error.to_string())));
        }
        let result = self.run_loop(input, &mut channels, &events).await;
        match result {
            Ok(summary) => {
                if let Err(error) = self.emit(&events, AgentEvent::RunFinished(summary)).await {
                    return Err((self, error));
                }
                Ok(self)
            }
            Err(error) => Err((self, error)),
        }
    }

    async fn emit(
        &mut self,
        events: &mpsc::Sender<AgentEvent>,
        event: AgentEvent,
    ) -> Result<(), AgentError> {
        events
            .send(event)
            .await
            .map_err(|error| AgentError::Task(error.to_string()))
    }

    async fn record_event(
        &mut self,
        events: &mpsc::Sender<AgentEvent>,
        event: SessionEvent,
        source_event_seqs: Vec<u64>,
        surface_op: Option<SurfaceOp>,
    ) -> Result<RecordedEvent, AgentError> {
        let time = self.event_clock.now();
        self.record_event_at(events, time, event, source_event_seqs, surface_op)
            .await
    }

    async fn record_event_at(
        &mut self,
        events: &mpsc::Sender<AgentEvent>,
        time: crate::EventTime,
        event: SessionEvent,
        source_event_seqs: Vec<u64>,
        surface_op: Option<SurfaceOp>,
    ) -> Result<RecordedEvent, AgentError> {
        let recorded = self
            .commit
            .event_at(time, event, source_event_seqs, surface_op)
            .await?;
        events
            .send(AgentEvent::SessionEvent(recorded.clone()))
            .await
            .map_err(|error| AgentError::Task(error.to_string()))?;
        Ok(recorded)
    }

    async fn run_loop(
        &mut self,
        input: String,
        channels: &mut RunChannels,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<RunSummary, AgentError> {
        let mut turn = self.next_turn;
        self.next_turn = self.next_turn.saturating_add(1);
        let mut step = 0_u32;
        let mut model_requests = 0_usize;
        let mut pending_user = Some((input, UserMessageMode::Initial, None));
        let mut first_request = true;

        self.active_turn = Some(turn);
        self.record_event(events, SessionEvent::TurnStart { turn }, vec![], None)
            .await?;

        loop {
            if model_requests >= self.max_turns {
                return Err(AgentError::MaxTurns(self.max_turns));
            }
            step = step.saturating_add(1);
            model_requests += 1;
            self.active_step = Some(step);
            self.record_event(events, SessionEvent::StepStart { turn, step }, vec![], None)
                .await?;

            if let Some((message, mode, input_id)) = pending_user.take() {
                self.append_user(message, turn, step, mode, input_id, events)
                    .await?;
            }

            self.compact_once(false, None, &channels.cancel, events)
                .await?;
            events
                .send(AgentEvent::ModelStarted(model_requests))
                .await
                .map_err(|error| AgentError::Task(error.to_string()))?;

            let tool_schemas = self.tool_schemas();
            if first_request {
                let reasoning_effort = self
                    .model
                    .reasoning_effort
                    .as_ref()
                    .map(|effort| format!("{effort:?}").to_lowercase());
                let fingerprint = request_fingerprint(
                    &self.model.model,
                    &self.instructions,
                    &tool_schemas,
                    reasoning_effort.as_deref(),
                    self.model.max_output_tokens,
                    &self.session_config,
                );
                let reason = match (&self.last_request_fingerprint, self.has_request_header) {
                    (_, false) => RequestHeaderReason::Initial,
                    (Some(previous), true) if previous != &fingerprint => {
                        RequestHeaderReason::Change
                    }
                    _ => RequestHeaderReason::Resume,
                };
                self.record_event(
                    events,
                    SessionEvent::RequestHeader {
                        turn,
                        step,
                        reason,
                        model: self.model.model.clone(),
                        instructions: self.instructions.clone(),
                        tools: tool_schemas.clone(),
                        reasoning_effort,
                        max_output_tokens: self.model.max_output_tokens,
                        session_config: self.session_config.clone(),
                    },
                    vec![],
                    None,
                )
                .await?;
                self.has_request_header = true;
                self.last_request_fingerprint = Some(fingerprint);
                first_request = false;
            }

            let mut request = CreateResponseArgs::default()
                .model(self.model.model.clone())
                .instructions(self.instructions.clone())
                .input(self.state.context())
                .tools(tool_schemas)
                .store(false)
                .build()?;
            request.reasoning = self.reasoning();
            request.max_output_tokens = self.model.max_output_tokens;
            self.record_event(
                events,
                SessionEvent::ModelRequestStart { turn, step },
                vec![],
                None,
            )
            .await?;
            let responses = self.model.client.responses();
            let mut stream = tokio::select! {
                _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                result = responses.create_stream(request) => result?,
            };
            let mut completed = None;
            let mut chunk_seqs = Vec::new();
            loop {
                let event = tokio::select! {
                    _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                    message = channels.steer.recv() => {
                        if let Some(message) = message {
                            self.admit_input(&message, InputMode::Steer, events).await?;
                        }
                        continue;
                    }
                    message = channels.queue.recv() => {
                        if let Some(message) = message {
                            self.admit_input(&message, InputMode::Queue, events).await?;
                        }
                        continue;
                    }
                    event = stream.next() => event,
                };
                let Some(event) = event else {
                    break;
                };
                match event? {
                    ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
                        let recorded = self
                            .record_event(
                                events,
                                SessionEvent::AssistantChunk {
                                    turn,
                                    step,
                                    chunk: AssistantChunk::OutputTextDelta {
                                        delta: delta.delta.clone(),
                                    },
                                },
                                vec![],
                                None,
                            )
                            .await?;
                        chunk_seqs.push(recorded.seq);
                        self.emit(events, AgentEvent::TextDelta(delta.delta))
                            .await?;
                    }
                    ResponseStreamEvent::ResponseReasoningTextDelta(delta) => {
                        let recorded = self
                            .record_event(
                                events,
                                SessionEvent::AssistantChunk {
                                    turn,
                                    step,
                                    chunk: AssistantChunk::ReasoningTextDelta {
                                        delta: delta.delta.clone(),
                                    },
                                },
                                vec![],
                                None,
                            )
                            .await?;
                        chunk_seqs.push(recorded.seq);
                        self.emit(events, AgentEvent::ReasoningDelta(delta.delta))
                            .await?;
                    }
                    ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(delta) => {
                        let recorded = self
                            .record_event(
                                events,
                                SessionEvent::AssistantChunk {
                                    turn,
                                    step,
                                    chunk: AssistantChunk::ToolCallArgumentsDelta {
                                        call_id: delta.item_id,
                                        name: None,
                                        delta: delta.delta,
                                    },
                                },
                                vec![],
                                None,
                            )
                            .await?;
                        chunk_seqs.push(recorded.seq);
                    }
                    ResponseStreamEvent::ResponseCompleted(event) => {
                        completed = Some(event.response);
                    }
                    ResponseStreamEvent::ResponseFailed(event) => {
                        return Err(AgentError::ModelResponse(format!(
                            "{:?}",
                            event.response.error
                        )));
                    }
                    ResponseStreamEvent::ResponseIncomplete(event) => {
                        return Err(AgentError::ModelResponse(format!(
                            "incomplete: {:?}",
                            event.response.incomplete_details
                        )));
                    }
                    ResponseStreamEvent::ResponseError(event) => {
                        return Err(AgentError::ModelResponse(format!("{event:?}")));
                    }
                    _ => {}
                }
            }

            let response = completed.ok_or(AgentError::MissingResponse)?;
            let calls = response
                .output
                .iter()
                .filter_map(|item| match item {
                    OutputItem::FunctionCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let response_items = response
                .output
                .iter()
                .cloned()
                .map(InputItem::from)
                .collect::<Vec<_>>();
            let response_metadata = ResponseMetadata {
                id: response.id.clone(),
                model: response.model.clone(),
                usage: response.usage.clone(),
            };
            self.append_assistant(
                response_items,
                response_metadata,
                turn,
                step,
                chunk_seqs,
                events,
            )
            .await?;

            self.execute_tools(&calls, turn, step, channels, events)
                .await?;

            while let Ok(message) = channels.steer.try_recv() {
                self.admit_input(&message, InputMode::Steer, events).await?;
            }
            while let Ok(message) = channels.queue.try_recv() {
                self.admit_input(&message, InputMode::Queue, events).await?;
            }

            self.record_event(
                events,
                SessionEvent::StepEnd {
                    turn,
                    step,
                    outcome: StepOutcome::Completed,
                    error: None,
                },
                vec![],
                None,
            )
            .await?;
            self.active_step = None;

            if let Some(message) = self.pending_steer.pop_front() {
                self.record_event(
                    events,
                    SessionEvent::InputConsumed {
                        id: message.id.clone(),
                    },
                    vec![],
                    None,
                )
                .await?;
                pending_user = Some((message.input, UserMessageMode::Steer, Some(message.id)));
                continue;
            }
            if !calls.is_empty() {
                continue;
            }

            self.record_event(
                events,
                SessionEvent::TurnEnd {
                    turn,
                    reason: TurnEndReason::Completed,
                },
                vec![],
                None,
            )
            .await?;
            self.active_turn = None;

            let summary = RunSummary {
                output: response.output_text().unwrap_or_default(),
                response_id: response.id,
                usage: response.usage,
            };
            if let Some(message) = self.pending_queue.pop_front() {
                turn = self.next_turn;
                self.next_turn = self.next_turn.saturating_add(1);
                step = 0;
                self.active_turn = Some(turn);
                self.record_event(events, SessionEvent::TurnStart { turn }, vec![], None)
                    .await?;
                self.record_event(
                    events,
                    SessionEvent::InputConsumed {
                        id: message.id.clone(),
                    },
                    vec![],
                    None,
                )
                .await?;
                pending_user = Some((message.input, UserMessageMode::Queue, Some(message.id)));
                continue;
            }
            return Ok(summary);
        }
    }

    async fn admit_input(
        &mut self,
        message: &PendingInput,
        mode: InputMode,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        self.record_event(
            events,
            SessionEvent::InputAdmitted {
                id: message.id.clone(),
                input: message.input.clone(),
                mode,
            },
            vec![],
            None,
        )
        .await?;
        self.emit(
            events,
            AgentEvent::InputAdmitted {
                id: message.id.clone(),
                input: message.input.clone(),
                mode,
            },
        )
        .await?;
        match mode {
            InputMode::Steer => self.pending_steer.push_back(PendingInput {
                id: message.id.clone(),
                input: message.input.clone(),
            }),
            InputMode::Queue => self.pending_queue.push_back(PendingInput {
                id: message.id.clone(),
                input: message.input.clone(),
            }),
        }
        Ok(())
    }

    async fn execute_tools(
        &mut self,
        calls: &[FunctionToolCall],
        turn: u32,
        step: u32,
        channels: &mut RunChannels,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let mut call_seqs = Vec::with_capacity(calls.len());
        for call in calls {
            let recorded = self
                .record_event(
                    events,
                    SessionEvent::ToolCall {
                        turn,
                        step,
                        call_id: call.call_id.clone(),
                        parent_call_id: None,
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    },
                    vec![],
                    None,
                )
                .await?;
            call_seqs.push(recorded.seq);
            self.open_tools
                .insert(call.call_id.clone(), (turn, step, recorded.seq));
            events
                .send(AgentEvent::ToolStarted(call.clone()))
                .await
                .map_err(|error| AgentError::Task(error.to_string()))?;
        }

        let mut allowed = Vec::with_capacity(calls.len());
        for call in calls {
            let Some(tool) = self.tools.iter().find(|tool| tool.name() == call.name) else {
                allowed.push(None);
                continue;
            };
            if !tool.requires_approval() {
                allowed.push(Some(true));
                continue;
            }
            events
                .send(AgentEvent::ApprovalRequired(call.clone()))
                .await
                .map_err(|error| AgentError::Task(error.to_string()))?;
            let decision = loop {
                let approval = tokio::select! {
                    _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                    message = channels.steer.recv() => {
                        if let Some(message) = message {
                            self.admit_input(&message, InputMode::Steer, events).await?;
                        }
                        continue;
                    }
                    message = channels.queue.recv() => {
                        if let Some(message) = message {
                            self.admit_input(&message, InputMode::Queue, events).await?;
                        }
                        continue;
                    }
                    approval = channels.approvals.recv() => approval,
                };
                let Some((call_id, allow)) = approval else {
                    return Err(AgentError::Aborted);
                };
                if call_id == call.call_id {
                    break allow;
                }
            };
            allowed.push(Some(decision));
        }

        let mut execution_seqs = vec![Vec::<u64>::new(); calls.len()];
        let mut outcomes = vec![None::<(ToolResult, ToolResultStatus)>; calls.len()];
        let (task_events, mut task_event_rx) = mpsc::unbounded_channel();
        let mut tasks = tokio::task::JoinSet::new();
        for (index, (call, allow)) in calls.iter().zip(allowed).enumerate() {
            let Some(allow) = allow else {
                outcomes[index] = Some((
                    ToolResult::error(format!("Tool not found: {}", call.name)),
                    ToolResultStatus::NotFound,
                ));
                continue;
            };
            if !allow {
                outcomes[index] = Some((
                    ToolResult::error("Tool call denied by user"),
                    ToolResultStatus::Denied,
                ));
                continue;
            }
            let tool = self
                .tools
                .iter()
                .find(|tool| tool.name() == call.name)
                .cloned()
                .expect("approved tool must remain registered");
            let call = call.clone();
            let env = self.env.clone();
            let clock = self.event_clock.clone();
            let task_events = task_events.clone();
            tasks.spawn(async move {
                let started = clock.now();
                if task_events
                    .send(ToolTaskEvent::Started {
                        index,
                        time: started,
                    })
                    .is_err()
                {
                    return;
                }
                let result = tool.execute(&call, &env).await;
                let finished = clock.now();
                let _ = task_events.send(ToolTaskEvent::Finished {
                    index,
                    time: finished,
                    result,
                });
            });
        }
        drop(task_events);

        let mut next_commit = 0_usize;
        while next_commit < calls.len() {
            while let Some((result, status)) = outcomes[next_commit].take() {
                self.append_tool_result(
                    &calls[next_commit],
                    result.clone(),
                    status,
                    turn,
                    step,
                    call_seqs[next_commit],
                    &execution_seqs[next_commit],
                    events,
                )
                .await?;
                events
                    .send(AgentEvent::ToolFinished {
                        call: calls[next_commit].clone(),
                        result,
                    })
                    .await
                    .map_err(|error| AgentError::Task(error.to_string()))?;
                next_commit += 1;
                if next_commit == calls.len() {
                    break;
                }
            }
            if next_commit == calls.len() {
                break;
            }
            let completed = tokio::select! {
                _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                message = channels.steer.recv() => {
                    if let Some(message) = message {
                        self.admit_input(&message, InputMode::Steer, events).await?;
                    }
                    continue;
                }
                message = channels.queue.recv() => {
                    if let Some(message) = message {
                        self.admit_input(&message, InputMode::Queue, events).await?;
                    }
                    continue;
                }
                outcome = task_event_rx.recv() => outcome,
            };
            let Some(completed) = completed else {
                return Err(AgentError::Task("tool execution stream ended early".into()));
            };
            match completed {
                ToolTaskEvent::Started { index, time } => {
                    let started = self
                        .record_event_at(
                            events,
                            time,
                            SessionEvent::ToolExecutionStart {
                                call_id: calls[index].call_id.clone(),
                            },
                            vec![call_seqs[index]],
                            None,
                        )
                        .await?;
                    execution_seqs[index].push(started.seq);
                }
                ToolTaskEvent::Finished {
                    index,
                    time,
                    result,
                } => {
                    let finished = self
                        .record_event_at(
                            events,
                            time,
                            SessionEvent::ToolExecutionFinish {
                                call_id: calls[index].call_id.clone(),
                                outcome: if result.is_error {
                                    ToolExecutionOutcome::Error
                                } else {
                                    ToolExecutionOutcome::Success
                                },
                            },
                            execution_seqs[index].clone(),
                            None,
                        )
                        .await?;
                    execution_seqs[index].push(finished.seq);
                    let status = if result.is_error {
                        ToolResultStatus::Error
                    } else {
                        ToolResultStatus::Success
                    };
                    outcomes[index] = Some((result, status));
                }
            }
        }
        while let Some(result) = tasks.join_next().await {
            result.map_err(|error| AgentError::Task(error.to_string()))?;
        }
        Ok(())
    }

    async fn append_user(
        &mut self,
        message: String,
        turn: u32,
        step: u32,
        mode: UserMessageMode,
        input_id: Option<String>,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        if message.trim().is_empty() {
            return Err(AgentError::EmptyInput);
        }
        self.commit.set_initial_title(&message).await?;
        let items = vec![InputItem::from(
            async_openai::types::responses::EasyInputMessage::from(message),
        )];
        let entry = self
            .state
            .append_items(items.clone(), None)
            .map_err(AgentError::State)?;
        let recorded = match self
            .record_event(
                events,
                SessionEvent::UserMessage {
                    turn,
                    step,
                    input_id,
                    mode,
                    items,
                },
                vec![],
                Some(SurfaceOp::Append),
            )
            .await
        {
            Ok(recorded) => recorded,
            Err(error) => {
                self.state.rollback(entry.id()).map_err(AgentError::State)?;
                return Err(error);
            }
        };
        self.surface_entries.push((entry.id(), recorded.seq, false));
        Ok(())
    }

    async fn append_assistant(
        &mut self,
        items: Vec<InputItem>,
        response: ResponseMetadata,
        turn: u32,
        step: u32,
        chunk_seqs: Vec<u64>,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let entry = self
            .state
            .append_items(items.clone(), Some(response.clone()))
            .map_err(AgentError::State)?;
        let recorded = match self
            .record_event(
                events,
                SessionEvent::AssistantMessage {
                    turn,
                    step,
                    items,
                    response,
                },
                chunk_seqs,
                Some(SurfaceOp::Append),
            )
            .await
        {
            Ok(recorded) => recorded,
            Err(error) => {
                self.state.rollback(entry.id()).map_err(AgentError::State)?;
                return Err(error);
            }
        };
        self.surface_entries.push((entry.id(), recorded.seq, false));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_tool_result(
        &mut self,
        call: &FunctionToolCall,
        result: ToolResult,
        status: ToolResultStatus,
        turn: u32,
        step: u32,
        call_seq: u64,
        execution_seqs: &[u64],
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let item = function_output(call, result.output.clone());
        let entry = self
            .state
            .append_items(vec![item.clone()], None)
            .map_err(AgentError::State)?;
        let mut sources = vec![call_seq];
        sources.extend_from_slice(execution_seqs);
        let recorded = match self
            .record_event(
                events,
                SessionEvent::ToolResult {
                    turn,
                    step,
                    call_id: call.call_id.clone(),
                    output: result.output,
                    status,
                    item,
                },
                sources,
                Some(SurfaceOp::Append),
            )
            .await
        {
            Ok(recorded) => recorded,
            Err(error) => {
                self.state.rollback(entry.id()).map_err(AgentError::State)?;
                return Err(error);
            }
        };
        self.surface_entries.push((entry.id(), recorded.seq, false));
        self.open_tools.remove(&call.call_id);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn append_compaction(
        &mut self,
        compaction_id: String,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        response: ResponseMetadata,
        start_seq: u64,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let entry = self
            .state
            .append_compaction(
                summary.clone(),
                first_kept_id,
                tokens_before,
                Some(response.clone()),
            )
            .map_err(AgentError::State)?;
        let replaced_event_seqs = self
            .surface_entries
            .iter()
            .filter_map(|(entry_id, seq, compaction)| {
                (*entry_id < first_kept_id || *compaction).then_some(*seq)
            })
            .collect();
        let recorded = match self
            .record_event(
                events,
                SessionEvent::CompactionEnd {
                    compaction_id,
                    summary,
                    first_kept_id,
                    tokens_before,
                    response: Some(response),
                    outcome: StepOutcome::Completed,
                },
                vec![start_seq],
                Some(SurfaceOp::Replace {
                    replaced_event_seqs,
                }),
            )
            .await
        {
            Ok(recorded) => recorded,
            Err(error) => {
                self.state.rollback(entry.id()).map_err(AgentError::State)?;
                return Err(error);
            }
        };
        self.surface_entries.push((entry.id(), recorded.seq, true));
        self.active_compaction = None;
        Ok(())
    }

    async fn compact_once(
        &mut self,
        force: bool,
        custom_instructions: Option<&str>,
        cancel: &CancellationToken,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let Some(config) = self.compaction else {
            return if force {
                Err(AgentError::NothingToCompact)
            } else {
                Ok(())
            };
        };
        let tool_schemas = self.tool_schemas();
        let tokens_before = context_tokens(&self.state, &self.instructions, &tool_schemas);
        if !force && !config.needs_compaction(tokens_before) {
            return Ok(());
        }
        let Some(prepared) =
            prepare_compaction(&self.state, config.keep_recent_tokens, custom_instructions)
        else {
            return if force {
                Err(AgentError::NothingToCompact)
            } else {
                Ok(())
            };
        };
        let compaction_id = Uuid::new_v4().to_string();
        let started = self
            .record_event(
                events,
                SessionEvent::CompactionStart {
                    compaction_id: compaction_id.clone(),
                    tokens_before,
                    first_kept_id: prepared.first_kept_id,
                },
                vec![],
                None,
            )
            .await?;
        self.active_compaction = Some(ActiveCompaction {
            id: compaction_id.clone(),
            start_seq: started.seq,
            tokens_before,
            first_kept_id: prepared.first_kept_id,
        });
        events
            .send(AgentEvent::CompactionStarted { tokens_before })
            .await
            .map_err(|error| AgentError::Task(error.to_string()))?;
        let mut request = CreateResponseArgs::default()
            .model(self.model.model.clone())
            .instructions(SUMMARY_INSTRUCTIONS)
            .input(prepared.prompt)
            .store(false)
            .build()?;
        request.reasoning = self.reasoning();
        request.max_output_tokens = self.model.max_output_tokens;
        let responses = self.model.client.responses();
        let response = tokio::select! {
            _ = cancel.cancelled() => Err(AgentError::Aborted),
            response = responses.create(request) => response.map_err(AgentError::from),
        };
        let response = match response {
            Ok(response) => response,
            Err(error) => {
                let outcome = if matches!(error, AgentError::Aborted) {
                    StepOutcome::Aborted
                } else {
                    StepOutcome::Failed
                };
                self.record_event(
                    events,
                    SessionEvent::CompactionEnd {
                        compaction_id,
                        summary: String::new(),
                        first_kept_id: prepared.first_kept_id,
                        tokens_before,
                        response: None,
                        outcome,
                    },
                    vec![started.seq],
                    None,
                )
                .await?;
                self.active_compaction = None;
                return Err(error);
            }
        };
        let summary = response.output_text().unwrap_or_default();
        if summary.trim().is_empty() {
            self.record_event(
                events,
                SessionEvent::CompactionEnd {
                    compaction_id,
                    summary: String::new(),
                    first_kept_id: prepared.first_kept_id,
                    tokens_before,
                    response: Some(ResponseMetadata {
                        id: response.id,
                        model: response.model,
                        usage: response.usage,
                    }),
                    outcome: StepOutcome::Failed,
                },
                vec![started.seq],
                None,
            )
            .await?;
            self.active_compaction = None;
            return Err(AgentError::ModelResponse(
                "compaction returned an empty summary".into(),
            ));
        }
        self.append_compaction(
            compaction_id,
            summary.clone(),
            prepared.first_kept_id,
            tokens_before,
            ResponseMetadata {
                id: response.id,
                model: response.model,
                usage: response.usage,
            },
            started.seq,
            events,
        )
        .await?;
        events
            .send(AgentEvent::CompactionFinished {
                tokens_before,
                first_kept_id: prepared.first_kept_id,
                summary,
            })
            .await
            .map_err(|error| AgentError::Task(error.to_string()))?;
        Ok(())
    }

    async fn close_interrupted_compaction(
        &mut self,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let Some(compaction) = self.active_compaction.clone() else {
            return Ok(());
        };
        self.record_event(
            events,
            SessionEvent::CompactionEnd {
                compaction_id: compaction.id,
                summary: String::new(),
                first_kept_id: compaction.first_kept_id,
                tokens_before: compaction.tokens_before,
                response: None,
                outcome: StepOutcome::Aborted,
            },
            vec![compaction.start_seq],
            None,
        )
        .await?;
        self.active_compaction = None;
        Ok(())
    }

    async fn close_unresolved_tools(
        &mut self,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<(), AgentError> {
        let unresolved = self.state.unresolved_tool_call_ids();
        for call_id in unresolved {
            let Some((turn, step, call_seq)) = self.open_tools.get(&call_id).copied() else {
                return Err(AgentError::State(format!(
                    "unresolved tool {call_id} has no durable tool/call event"
                )));
            };
            let output = "Tool execution was cancelled; its side effects are unknown. Do not retry automatically."
                .to_owned();
            let item = function_output_by_id(call_id.clone(), output.clone());
            let entry = self
                .state
                .append_items(vec![item.clone()], None)
                .map_err(AgentError::State)?;
            let recorded = match self
                .record_event(
                    events,
                    SessionEvent::ToolResult {
                        turn,
                        step,
                        call_id: call_id.clone(),
                        output,
                        status: ToolResultStatus::UnknownSideEffects,
                        item,
                    },
                    vec![call_seq],
                    Some(SurfaceOp::Append),
                )
                .await
            {
                Ok(recorded) => recorded,
                Err(error) => {
                    self.state.rollback(entry.id()).map_err(AgentError::State)?;
                    return Err(error);
                }
            };
            self.surface_entries.push((entry.id(), recorded.seq, false));
            self.open_tools.remove(&call_id);
        }
        Ok(())
    }

    async fn close_active_lifecycle(
        &mut self,
        events: &mpsc::Sender<AgentEvent>,
        outcome: StepOutcome,
        reason: TurnEndReason,
        error: Option<String>,
    ) {
        if let (Some(turn), Some(step)) = (self.active_turn, self.active_step) {
            let _ = self
                .record_event(
                    events,
                    SessionEvent::StepEnd {
                        turn,
                        step,
                        outcome,
                        error,
                    },
                    vec![],
                    None,
                )
                .await;
            self.active_step = None;
        }
        if let Some(turn) = self.active_turn {
            let _ = self
                .record_event(events, SessionEvent::TurnEnd { turn, reason }, vec![], None)
                .await;
            self.active_turn = None;
        }
    }
}

fn event_turn(event: &SessionEvent) -> Option<u32> {
    match event {
        SessionEvent::TurnStart { turn }
        | SessionEvent::TurnEnd { turn, .. }
        | SessionEvent::StepStart { turn, .. }
        | SessionEvent::StepEnd { turn, .. }
        | SessionEvent::UserMessage { turn, .. }
        | SessionEvent::RequestHeader { turn, .. }
        | SessionEvent::ModelRequestStart { turn, .. }
        | SessionEvent::AssistantChunk { turn, .. }
        | SessionEvent::AssistantMessage { turn, .. }
        | SessionEvent::ToolCall { turn, .. }
        | SessionEvent::ToolResult { turn, .. } => Some(*turn),
        _ => None,
    }
}

fn active_compaction_from_events(events: &[RecordedEvent]) -> Option<ActiveCompaction> {
    let mut active = None;
    for recorded in events {
        match &recorded.event {
            SessionEvent::CompactionStart {
                compaction_id,
                tokens_before,
                first_kept_id,
            } => {
                active = Some(ActiveCompaction {
                    id: compaction_id.clone(),
                    start_seq: recorded.seq,
                    tokens_before: *tokens_before,
                    first_kept_id: *first_kept_id,
                });
            }
            SessionEvent::CompactionEnd { .. } => active = None,
            _ => {}
        }
    }
    active
}

fn open_tools_from_events(events: &[RecordedEvent]) -> HashMap<String, (u32, u32, u64)> {
    let mut open = HashMap::new();
    for recorded in events {
        match &recorded.event {
            SessionEvent::ToolCall {
                turn,
                step,
                call_id,
                ..
            } => {
                open.insert(call_id.clone(), (*turn, *step, recorded.seq));
            }
            SessionEvent::ToolResult { call_id, .. } => {
                open.remove(call_id);
            }
            _ => {}
        }
    }
    open
}

fn active_lifecycle_from_events(events: &[RecordedEvent]) -> (Option<u32>, Option<u32>) {
    let mut turn = None;
    let mut step = None;
    for recorded in events {
        match recorded.event {
            SessionEvent::TurnStart { turn: value } => turn = Some(value),
            SessionEvent::TurnEnd { .. } => {
                turn = None;
                step = None;
            }
            SessionEvent::StepStart { step: value, .. } => step = Some(value),
            SessionEvent::StepEnd { .. } => step = None,
            _ => {}
        }
    }
    (turn, step)
}

fn surface_entries_from_events(events: &[RecordedEvent]) -> Vec<(u64, u64, bool)> {
    let mut next_entry_id = 1_u64;
    events
        .iter()
        .filter_map(|recorded| {
            let compaction = matches!(
                recorded.event,
                SessionEvent::CompactionEnd {
                    outcome: StepOutcome::Completed,
                    ..
                }
            );
            let is_surface = compaction
                || matches!(
                    recorded.event,
                    SessionEvent::UserMessage { .. }
                        | SessionEvent::AssistantMessage { .. }
                        | SessionEvent::ToolResult { .. }
                );
            is_surface.then(|| {
                let value = (next_entry_id, recorded.seq, compaction);
                next_entry_id = next_entry_id.saturating_add(1);
                value
            })
        })
        .collect()
}

fn last_request_fingerprint(events: &[RecordedEvent]) -> Option<String> {
    events.iter().rev().find_map(|recorded| {
        let SessionEvent::RequestHeader {
            model,
            instructions,
            tools,
            reasoning_effort,
            max_output_tokens,
            session_config,
            ..
        } = &recorded.event
        else {
            return None;
        };
        Some(request_fingerprint(
            model,
            instructions,
            tools,
            reasoning_effort.as_deref(),
            *max_output_tokens,
            session_config,
        ))
    })
}

fn request_fingerprint(
    model: &str,
    instructions: &str,
    tools: &[Tool],
    reasoning_effort: Option<&str>,
    max_output_tokens: Option<u32>,
    session_config: &SessionConfig,
) -> String {
    serde_json::to_string(&(
        model,
        instructions,
        tools,
        reasoning_effort,
        max_output_tokens,
        session_config,
    ))
    .unwrap_or_default()
}

struct RunChannels {
    steer: mpsc::UnboundedReceiver<PendingInput>,
    queue: mpsc::UnboundedReceiver<PendingInput>,
    approvals: mpsc::UnboundedReceiver<(String, bool)>,
    cancel: CancellationToken,
}

enum ToolTaskEvent {
    Started {
        index: usize,
        time: crate::EventTime,
    },
    Finished {
        index: usize,
        time: crate::EventTime,
        result: ToolResult,
    },
}

struct PendingInput {
    id: String,
    input: String,
}

#[derive(Clone)]
pub struct RunControl {
    steer: mpsc::UnboundedSender<PendingInput>,
    queue: mpsc::UnboundedSender<PendingInput>,
    approvals: mpsc::UnboundedSender<(String, bool)>,
    cancel: CancellationToken,
}

impl RunControl {
    pub fn steer(&self, message: impl Into<String>) -> Result<(), AgentError> {
        self.steer
            .send(PendingInput {
                id: Uuid::new_v4().to_string(),
                input: message.into(),
            })
            .map_err(|error| AgentError::Task(error.to_string()))
    }

    pub fn queue(&self, message: impl Into<String>) -> Result<(), AgentError> {
        self.queue
            .send(PendingInput {
                id: Uuid::new_v4().to_string(),
                input: message.into(),
            })
            .map_err(|error| AgentError::Task(error.to_string()))
    }

    pub fn approve(&self, call_id: impl Into<String>, allow: bool) -> Result<(), AgentError> {
        self.approvals
            .send((call_id.into(), allow))
            .map_err(|error| AgentError::Task(error.to_string()))
    }

    pub fn abort(&self) {
        self.cancel.cancel();
    }
}

pub struct ActiveAgent {
    control: RunControl,
    events: mpsc::Receiver<AgentEvent>,
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
                        return self.task.await.map_err(|error| AgentError::Task(error.to_string()));
                    }
                }
            }
        }
    }
}

fn function_output(call: &FunctionToolCall, output: String) -> InputItem {
    function_output_by_id(call.call_id.clone(), output)
}

fn function_output_by_id(call_id: String, output: String) -> InputItem {
    InputItem::from(Item::from(FunctionCallOutputItemParam {
        call_id,
        output: output.into(),
        id: None,
        status: None,
    }))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};
    use futures_util::future::BoxFuture;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Barrier;

    use super::{ActiveAgent, Agent, AgentEvent, Model, ReasoningEffort, RunControl};
    use crate::{
        AgentTool, Env, InputMode, Session, SessionConfig, SessionError, SessionEvent, SessionInfo,
        State, StateCommit, ToolResult,
    };

    async fn barrier_model(
        output: &'static str,
        barrier: Arc<Barrier>,
    ) -> (Model, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let (body_start, content_length) = loop {
                let mut chunk = [0; 4096];
                let bytes = socket.read(&mut chunk).await.unwrap();
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
                request.extend_from_slice(&chunk[..bytes]);
            }
            barrier.wait().await;
            let body = format!(
                "data: {{\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"{output}\"}}\n\ndata: {{\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{{\"created_at\":0,\"id\":\"resp_1\",\"model\":\"test-model\",\"object\":\"response\",\"output\":[{{\"type\":\"message\",\"content\":[{{\"type\":\"output_text\",\"annotations\":[],\"text\":\"{output}\"}}],\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\"}}],\"status\":\"completed\"}}}}\n\n"
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
                "test-key",
                format!("http://{address}"),
                "test-model",
                128_000,
            ),
            server,
        )
    }

    #[tokio::test]
    async fn sessions_run_concurrently_within_and_across_projects() {
        let directory = std::env::temp_dir().join(format!(
            "kcastle-concurrent-sessions-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first_dir = directory.join("project-a");
        let second_dir = directory.join("project-b");
        let first = Session::create_in_project(&first_dir, "project-a")
            .await
            .unwrap();
        let second = Session::create_in_project(&first_dir, "project-a")
            .await
            .unwrap();
        let third = Session::create_in_project(&second_dir, "project-b")
            .await
            .unwrap();
        let barrier = Arc::new(Barrier::new(4));
        let (first_model, first_server) = barrier_model("first", barrier.clone()).await;
        let (second_model, second_server) = barrier_model("second", barrier.clone()).await;
        let (third_model, third_server) = barrier_model("third", barrier.clone()).await;

        let first_active = Agent::new(first_model, "test", first, ".").start("one");
        let second_active = Agent::new(second_model, "test", second, ".").start("two");
        let third_active = Agent::new(third_model, "test", third, ".").start("three");
        tokio::time::timeout(std::time::Duration::from_secs(2), barrier.wait())
            .await
            .expect("all independent sessions should reach the provider concurrently");
        let (first, second, third) = tokio::join!(
            first_active.finish(),
            second_active.finish(),
            third_active.finish()
        );
        let first = first.unwrap();
        let second = second.unwrap();
        let third = third.unwrap();
        assert_eq!(first.session_info().project_id, "project-a");
        assert_eq!(second.session_info().project_id, "project-a");
        assert_eq!(third.session_info().project_id, "project-b");
        assert_ne!(first.session_info().id, second.session_info().id);
        assert!(format!("{:?}", first.transcript()).contains("first"));
        assert!(format!("{:?}", second.transcript()).contains("second"));
        assert!(format!("{:?}", third.transcript()).contains("third"));
        let first_info = first.session_info().clone();
        let second_info = second.session_info().clone();
        let third_info = third.session_info().clone();
        Session::delete(&first_info).expect("an idle agent must release its writer lease");
        Session::delete(&second_info).expect("sessions in the same project have separate leases");
        Session::delete(&third_info).expect("projects do not share writer leases");
        first_server.await.unwrap();
        second_server.await.unwrap();
        third_server.await.unwrap();
        drop((first, second, third));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn durable_queue_is_recovered_into_the_owning_agent_only() {
        let directory = std::env::temp_dir().join(format!(
            "kcastle-recovered-queue-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(
                SessionEvent::InputAdmitted {
                    id: "queued-1".into(),
                    input: "continue later".into(),
                    mode: InputMode::Queue,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        drop(commit);

        let recovered = Session::open_readonly_in_project(&path, "project-a").unwrap();
        let agent = Agent::new(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            recovered,
            ".",
        );
        assert!(agent.pending_steer.is_empty());
        assert_eq!(agent.pending_queue.len(), 1);
        assert_eq!(agent.pending_queue[0].input, "continue later");
        drop(agent);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn interrupted_compaction_is_closed_before_the_next_compaction() {
        let directory = std::env::temp_dir().join(format!(
            "kcastle-recovered-compaction-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        let interrupted = commit
            .event(
                SessionEvent::CompactionStart {
                    compaction_id: "interrupted".into(),
                    tokens_before: 1_000,
                    first_kept_id: 7,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        drop(commit);

        let session = Session::open_readonly(&path).unwrap();
        let mut agent = Agent::new(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            session,
            ".",
        );
        assert_eq!(
            agent
                .active_compaction
                .as_ref()
                .map(|active| active.id.as_str()),
            Some("interrupted")
        );
        agent.commit.prepare(&agent.state).await.unwrap();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(8);
        agent
            .close_interrupted_compaction(&events_tx)
            .await
            .unwrap();
        assert!(agent.active_compaction.is_none());
        assert!(matches!(
            events_rx.recv().await,
            Some(AgentEvent::SessionEvent(crate::RecordedEvent {
                event: SessionEvent::CompactionEnd {
                    outcome: crate::StepOutcome::Aborted,
                    ..
                },
                source_event_seqs,
                ..
            })) if source_event_seqs == vec![interrupted.seq]
        ));

        let next = agent
            .record_event(
                &events_tx,
                SessionEvent::CompactionStart {
                    compaction_id: "next".into(),
                    tokens_before: 800,
                    first_kept_id: 9,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        agent
            .record_event(
                &events_tx,
                SessionEvent::CompactionEnd {
                    compaction_id: "next".into(),
                    summary: String::new(),
                    first_kept_id: 9,
                    tokens_before: 800,
                    response: None,
                    outcome: crate::StepOutcome::Aborted,
                },
                vec![next.seq],
                None,
            )
            .await
            .unwrap();
        drop(agent);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(snapshot.events().len(), 4);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn streams_a_response_into_state() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let body = Arc::new(concat!(
            "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\n\n",
            "data: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"created_at\":0,\"id\":\"resp_1\",\"model\":\"test-model\",\"object\":\"response\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"annotations\":[],\"text\":\"hello\"}],\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\"}],\"status\":\"completed\"}}\n\n"
        ));
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let (body_start, content_length) = loop {
                let mut chunk = [0; 4096];
                let bytes = socket.read(&mut chunk).await.unwrap();
                assert_ne!(bytes, 0, "request ended before its headers");
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
                assert_ne!(bytes, 0, "request ended before its body");
                request.extend_from_slice(&chunk[..bytes]);
            }
            let request = String::from_utf8_lossy(&request);
            assert!(request.contains("\"reasoning\":{\"effort\":\"high\"}"));
            assert!(request.contains("\"max_output_tokens\":256000"));
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let model = Model::new(
            "test",
            "test-key",
            format!("http://{address}"),
            "test-model",
            128_000,
        )
        .with_reasoning(
            &[ReasoningEffort::Low, ReasoningEffort::High],
            ReasoningEffort::High,
        )
        .with_max_output_tokens(Some(256_000));
        let agent = Agent::new(model, "test", Session::memory(), ".");
        let mut active = agent.start("hello");
        let mut text = String::new();
        let mut finished = false;
        while let Some(event) = active.next_event().await {
            match event {
                AgentEvent::TextDelta(delta) => text.push_str(&delta),
                AgentEvent::RunFinished(summary) => {
                    assert_eq!(summary.output, "hello");
                    finished = true;
                }
                AgentEvent::RunFailed(error) => panic!("run failed: {error}"),
                _ => {}
            }
        }
        let agent = active.finish().await.unwrap();
        server.await.unwrap();
        assert_eq!(text, "hello");
        assert!(finished);
        assert_eq!(agent.state.entries().len(), 2);
    }

    #[derive(Debug)]
    struct DelayedTool;

    impl AgentTool for DelayedTool {
        fn name(&self) -> &str {
            "delay"
        }

        fn schema(&self) -> Tool {
            Tool::Function(FunctionTool {
                name: "delay".into(),
                description: None,
                parameters: None,
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
                let delay = if call.call_id == "slow" { 40 } else { 1 };
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                ToolResult::ok(call.call_id.clone())
            })
        }
    }

    struct DelayedStartCommit {
        inner: Box<dyn StateCommit>,
    }

    impl StateCommit for DelayedStartCommit {
        fn info(&self) -> &SessionInfo {
            self.inner.info()
        }

        fn prepare<'a>(&'a mut self, state: &'a State) -> BoxFuture<'a, Result<(), SessionError>> {
            self.inner.prepare(state)
        }

        fn event<'a>(
            &'a mut self,
            event: SessionEvent,
            source_event_seqs: Vec<u64>,
            surface_op: Option<crate::SurfaceOp>,
        ) -> BoxFuture<'a, Result<crate::RecordedEvent, SessionError>> {
            self.inner.event(event, source_event_seqs, surface_op)
        }

        fn event_at<'a>(
            &'a mut self,
            time: crate::EventTime,
            event: SessionEvent,
            source_event_seqs: Vec<u64>,
            surface_op: Option<crate::SurfaceOp>,
        ) -> BoxFuture<'a, Result<crate::RecordedEvent, SessionError>> {
            Box::pin(async move {
                if matches!(&event, SessionEvent::ToolExecutionStart { .. }) {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
                self.inner
                    .event_at(time, event, source_event_seqs, surface_op)
                    .await
            })
        }

        fn set_config<'a>(
            &'a mut self,
            config: &'a SessionConfig,
        ) -> BoxFuture<'a, Result<(), SessionError>> {
            self.inner.set_config(config)
        }

        fn set_initial_title<'a>(
            &'a mut self,
            message: &'a str,
        ) -> BoxFuture<'a, Result<(), SessionError>> {
            self.inner.set_initial_title(message)
        }

        fn rename<'a>(&'a mut self, title: &'a str) -> BoxFuture<'a, Result<(), SessionError>> {
            self.inner.rename(title)
        }

        fn release_writer(&mut self) {
            self.inner.release_writer();
        }
    }

    #[tokio::test]
    async fn parallel_tool_finish_order_is_actual_but_results_remain_model_order() {
        let model = Model::new("test", "key", "http://localhost", "model", 10_000);
        let mut agent = Agent::new(model, "test", Session::memory(), ".");
        agent.tools = vec![Arc::new(DelayedTool)];
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        agent
            .record_event(
                &events_tx,
                SessionEvent::TurnStart { turn: 1 },
                vec![],
                None,
            )
            .await
            .unwrap();
        agent
            .record_event(
                &events_tx,
                SessionEvent::StepStart { turn: 1, step: 1 },
                vec![],
                None,
            )
            .await
            .unwrap();
        let (_steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_approval_tx, approvals_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut channels = super::RunChannels {
            steer: steer_rx,
            queue: queue_rx,
            approvals: approvals_rx,
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        let calls = [
            FunctionToolCall {
                arguments: "{}".into(),
                call_id: "slow".into(),
                namespace: None,
                name: "delay".into(),
                id: None,
                status: None,
            },
            FunctionToolCall {
                arguments: "{}".into(),
                call_id: "fast".into(),
                namespace: None,
                name: "delay".into(),
                id: None,
                status: None,
            },
        ];
        agent
            .execute_tools(&calls, 1, 1, &mut channels, &events_tx)
            .await
            .unwrap();

        let mut finishes = Vec::new();
        let mut results = Vec::new();
        while let Ok(event) = events_rx.try_recv() {
            let AgentEvent::SessionEvent(recorded) = event else {
                continue;
            };
            match recorded.event {
                SessionEvent::ToolExecutionFinish { call_id, .. } => finishes.push(call_id),
                SessionEvent::ToolResult { call_id, .. } => results.push(call_id),
                _ => {}
            }
        }
        assert_eq!(finishes, ["fast", "slow"]);
        assert_eq!(results, ["slow", "fast"]);
    }

    #[tokio::test]
    async fn tool_execution_duration_excludes_commit_backpressure() {
        let (state, commit) = Session::memory().into_parts();
        let mut agent = Agent::from_parts(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            state,
            Box::new(DelayedStartCommit { inner: commit }),
            ".",
            vec![Arc::new(DelayedTool)],
            None,
            3,
        )
        .unwrap();
        let (events_tx, mut events_rx) = tokio::sync::mpsc::channel(64);
        agent
            .record_event(
                &events_tx,
                SessionEvent::TurnStart { turn: 1 },
                vec![],
                None,
            )
            .await
            .unwrap();
        agent
            .record_event(
                &events_tx,
                SessionEvent::StepStart { turn: 1, step: 1 },
                vec![],
                None,
            )
            .await
            .unwrap();
        let (_steer_tx, steer_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_queue_tx, queue_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_approval_tx, approvals_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut channels = super::RunChannels {
            steer: steer_rx,
            queue: queue_rx,
            approvals: approvals_rx,
            cancel: tokio_util::sync::CancellationToken::new(),
        };
        let calls = [FunctionToolCall {
            arguments: "{}".into(),
            call_id: "fast".into(),
            namespace: None,
            name: "delay".into(),
            id: None,
            status: None,
        }];
        agent
            .execute_tools(&calls, 1, 1, &mut channels, &events_tx)
            .await
            .unwrap();

        let mut started = None;
        let mut finished = None;
        while let Ok(event) = events_rx.try_recv() {
            let AgentEvent::SessionEvent(recorded) = event else {
                continue;
            };
            match recorded.event {
                SessionEvent::ToolExecutionStart { .. } => started = Some(recorded.time),
                SessionEvent::ToolExecutionFinish { .. } => finished = Some(recorded.time),
                _ => {}
            }
        }
        let duration = finished.unwrap().duration_since(&started.unwrap()).unwrap();
        assert!(
            duration < 25_000_000,
            "commit delay leaked into execution duration: {duration} ns"
        );
    }

    #[tokio::test]
    async fn finish_drains_buffered_events() {
        let (events_tx, events_rx) = tokio::sync::mpsc::channel(1);
        let (steer, _) = tokio::sync::mpsc::unbounded_channel();
        let (queue, _) = tokio::sync::mpsc::unbounded_channel();
        let (approvals, _) = tokio::sync::mpsc::unbounded_channel();
        let agent = Agent::new(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            Session::memory(),
            ".",
        );
        let task = tokio::spawn(async move {
            for index in 0..3 {
                events_tx
                    .send(AgentEvent::TextDelta(index.to_string()))
                    .await
                    .unwrap();
            }
            agent
        });
        let active = ActiveAgent {
            control: RunControl {
                steer,
                queue,
                approvals,
                cancel: tokio_util::sync::CancellationToken::new(),
            },
            events: events_rx,
            task,
        };

        tokio::time::timeout(std::time::Duration::from_secs(1), active.finish())
            .await
            .expect("finish should drain pending events")
            .unwrap();
    }

    struct EchoTool;

    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn schema(&self) -> Tool {
            Tool::Function(FunctionTool {
                name: "echo".into(),
                description: Some("Echo text".into()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": false
                })),
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
            Box::pin(async move { ToolResult::ok(call.arguments.clone()) })
        }
    }

    #[test]
    fn accepts_injected_tools_and_commit_port() {
        let (state, commit) = Session::memory().into_parts();
        let agent = Agent::from_parts(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            state,
            commit,
            ".",
            vec![Arc::new(EchoTool)],
            None,
            3,
        )
        .unwrap();
        assert!(matches!(
            agent.tool_schemas().as_slice(),
            [Tool::Function(tool)] if tool.name == "echo"
        ));
    }

    struct FailingCommit {
        info: SessionInfo,
    }

    impl StateCommit for FailingCommit {
        fn info(&self) -> &SessionInfo {
            &self.info
        }

        fn prepare<'a>(&'a mut self, _state: &'a State) -> BoxFuture<'a, Result<(), SessionError>> {
            Box::pin(async { Ok(()) })
        }

        fn event<'a>(
            &'a mut self,
            _event: SessionEvent,
            _source_event_seqs: Vec<u64>,
            _surface_op: Option<crate::SurfaceOp>,
        ) -> BoxFuture<'a, Result<crate::RecordedEvent, SessionError>> {
            Box::pin(async { Err(SessionError::Invalid("commit failed".into())) })
        }

        fn set_config<'a>(
            &'a mut self,
            _config: &'a SessionConfig,
        ) -> BoxFuture<'a, Result<(), SessionError>> {
            Box::pin(async { Ok(()) })
        }

        fn set_initial_title<'a>(
            &'a mut self,
            _message: &'a str,
        ) -> BoxFuture<'a, Result<(), SessionError>> {
            Box::pin(async { Ok(()) })
        }

        fn rename<'a>(&'a mut self, _title: &'a str) -> BoxFuture<'a, Result<(), SessionError>> {
            Box::pin(async { Ok(()) })
        }

        fn release_writer(&mut self) {}
    }

    #[tokio::test]
    async fn failed_commit_rolls_back_state() {
        let agent = Agent::from_parts(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            State::default(),
            Box::new(FailingCommit {
                info: SessionInfo::legacy(PathBuf::new(), "test", 0),
            }),
            ".",
            Vec::new(),
            None,
            3,
        )
        .unwrap();
        let mut active = agent.start("hello");
        let mut failed = false;
        while let Some(event) = active.next_event().await {
            if matches!(event, AgentEvent::RunFailed(_)) {
                failed = true;
            }
        }
        let agent = active.finish().await.unwrap();
        assert!(failed);
        assert!(agent.state.entries().is_empty());
    }
}
