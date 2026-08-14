use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;

use async_openai::Client;
use async_openai::config::OpenAIConfig;
use async_openai::error::OpenAIError;
use async_openai::types::responses::{
    CreateResponseArgs, FunctionCallOutputItemParam, FunctionToolCall, InputItem, Item, OutputItem,
    ResponseStreamEvent, ResponseUsage, Tool,
};
use futures_util::StreamExt;
use futures_util::future::join_all;
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::compaction::{
    CompactionConfig, SUMMARY_INSTRUCTIONS, context_tokens, prepare_compaction,
};
use crate::session::{Session, SessionError, SessionInfo, StateCommit};
use crate::state::{ResponseMetadata, State, StateEntry, TranscriptItem};
use crate::tool::{AgentTool, Env, ShellTool, ToolResult};

const DEFAULT_MAX_TURNS: usize = 100;

#[derive(Clone)]
pub struct Model {
    name: String,
    client: Client<OpenAIConfig>,
    model: String,
    context_window: usize,
}

impl Model {
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        api_base: impl Into<String>,
        model: impl Into<String>,
        context_window: usize,
    ) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(api_base);
        Self {
            name: name.into(),
            client: Client::with_config(config),
            model: model.into(),
            context_window,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn context_window(&self) -> usize {
        self.context_window
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
    RunStarted(String),
    ModelStarted(usize),
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
}

impl Agent {
    pub fn new(
        model: Model,
        instructions: impl Into<String>,
        session: Session,
        cwd: impl Into<PathBuf>,
    ) -> Self {
        let context_window = model.context_window();
        let (state, commit) = session.into_parts();
        Self {
            model,
            instructions: instructions.into(),
            state,
            commit,
            env: Env { cwd: cwd.into() },
            tools: vec![Arc::new(ShellTool)],
            compaction: Some(CompactionConfig::new(context_window)),
            max_turns: DEFAULT_MAX_TURNS,
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

    pub fn set_session(&mut self, session: Session) {
        (self.state, self.commit) = session.into_parts();
    }

    fn tool_schemas(&self) -> Vec<Tool> {
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
            match operation(self, channels, events_tx.clone()).await {
                Ok(agent) => agent,
                Err((mut agent, error)) => {
                    if matches!(error, AgentError::Aborted) {
                        match agent.close_unresolved_tools().await {
                            Ok(()) => {
                                let _ = events_tx.send(AgentEvent::RunAborted).await;
                            }
                            Err(cleanup_error) => {
                                let _ = events_tx
                                    .send(AgentEvent::RunFailed(format!(
                                        "abort cleanup failed: {cleanup_error}"
                                    )))
                                    .await;
                            }
                        }
                    } else {
                        let _ = events_tx
                            .send(AgentEvent::RunFailed(error.to_string()))
                            .await;
                    }
                    agent
                }
            }
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
        if let Err(error) = events.send(AgentEvent::RunStarted(input.clone())).await {
            return Err((self, AgentError::Task(error.to_string())));
        }
        if let Err(error) = self.append_user(input).await {
            return Err((self, error));
        }

        let result = self.run_loop(&mut channels, &events).await;
        match result {
            Ok(summary) => {
                let _ = events.send(AgentEvent::RunFinished(summary)).await;
                Ok(self)
            }
            Err(error) => Err((self, error)),
        }
    }

    async fn run_loop(
        &mut self,
        channels: &mut RunChannels,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<RunSummary, AgentError> {
        let mut turn = 0;
        let response = loop {
            let settled_response = loop {
                if turn >= self.max_turns {
                    return Err(AgentError::MaxTurns(self.max_turns));
                }
                self.compact_once(false, None, &channels.cancel, events)
                    .await?;
                turn += 1;
                events
                    .send(AgentEvent::ModelStarted(turn))
                    .await
                    .map_err(|error| AgentError::Task(error.to_string()))?;

                let tool_schemas = self.tool_schemas();
                let request = CreateResponseArgs::default()
                    .model(self.model.model.clone())
                    .instructions(self.instructions.clone())
                    .input(self.state.context())
                    .tools(tool_schemas)
                    .store(false)
                    .build()?;
                let responses = self.model.client.responses();
                let mut stream = tokio::select! {
                    _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                    result = responses.create_stream(request) => result?,
                };
                let mut completed = None;
                loop {
                    let event = tokio::select! {
                        _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                        event = stream.next() => event,
                    };
                    let Some(event) = event else {
                        break;
                    };
                    match event? {
                        ResponseStreamEvent::ResponseOutputTextDelta(delta) => {
                            events
                                .send(AgentEvent::TextDelta(delta.delta))
                                .await
                                .map_err(|error| AgentError::Task(error.to_string()))?;
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
                    .collect();
                self.append_items(
                    response_items,
                    Some(ResponseMetadata {
                        id: response.id.clone(),
                        model: response.model.clone(),
                        usage: response.usage.clone(),
                    }),
                )
                .await?;

                let outcomes = self.execute_tools(&calls, channels, events).await?;
                for (call, result) in calls.iter().zip(outcomes) {
                    events
                        .send(AgentEvent::ToolFinished {
                            call: call.clone(),
                            result: result.clone(),
                        })
                        .await
                        .map_err(|error| AgentError::Task(error.to_string()))?;
                    self.append_items(vec![function_output(call, result.output)], None)
                        .await?;
                }

                if let Ok(message) = channels.steer.try_recv() {
                    self.append_user(message).await?;
                    continue;
                }
                if !calls.is_empty() {
                    continue;
                }
                break response;
            };

            if let Ok(message) = channels.queue.try_recv() {
                self.append_user(message).await?;
                continue;
            }
            break settled_response;
        };

        Ok(RunSummary {
            output: response.output_text().unwrap_or_default(),
            response_id: response.id,
            usage: response.usage,
        })
    }

    async fn execute_tools(
        &self,
        calls: &[FunctionToolCall],
        channels: &mut RunChannels,
        events: &mpsc::Sender<AgentEvent>,
    ) -> Result<Vec<ToolResult>, AgentError> {
        let mut allowed = Vec::with_capacity(calls.len());
        for call in calls {
            let Some(tool) = self.tools.iter().find(|tool| tool.name() == call.name) else {
                allowed.push(false);
                continue;
            };
            if !tool.requires_approval() {
                allowed.push(true);
                continue;
            }
            events
                .send(AgentEvent::ApprovalRequired(call.clone()))
                .await
                .map_err(|error| AgentError::Task(error.to_string()))?;
            let decision = loop {
                let approval = tokio::select! {
                    _ = channels.cancel.cancelled() => return Err(AgentError::Aborted),
                    approval = channels.approvals.recv() => approval,
                };
                let Some((call_id, allow)) = approval else {
                    return Err(AgentError::Aborted);
                };
                if call_id == call.call_id {
                    break allow;
                }
            };
            allowed.push(decision);
        }

        for call in calls {
            events
                .send(AgentEvent::ToolStarted(call.clone()))
                .await
                .map_err(|error| AgentError::Task(error.to_string()))?;
        }
        let futures = calls.iter().zip(allowed).map(|(call, allow)| async move {
            match self.tools.iter().find(|tool| tool.name() == call.name) {
                None => ToolResult::error(format!("Tool not found: {}", call.name)),
                Some(_) if !allow => ToolResult::error("Tool call denied by user"),
                Some(tool) => tool.execute(call, &self.env).await,
            }
        });
        tokio::select! {
            _ = channels.cancel.cancelled() => Err(AgentError::Aborted),
            outcomes = join_all(futures) => Ok(outcomes),
        }
    }

    async fn append_user(&mut self, message: String) -> Result<(), AgentError> {
        if message.trim().is_empty() {
            return Err(AgentError::EmptyInput);
        }
        self.commit.set_initial_title(&message).await?;
        self.append_items(
            vec![InputItem::from(
                async_openai::types::responses::EasyInputMessage::from(message),
            )],
            None,
        )
        .await?;
        Ok(())
    }

    async fn append_items(
        &mut self,
        items: Vec<InputItem>,
        response: Option<ResponseMetadata>,
    ) -> Result<StateEntry, AgentError> {
        let entry = self
            .state
            .append_items(items, response)
            .map_err(AgentError::State)?;
        if let Err(error) = self.commit.commit(&entry).await {
            self.state.rollback(entry.id()).map_err(AgentError::State)?;
            return Err(error.into());
        }
        Ok(entry)
    }

    async fn append_compaction(
        &mut self,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        response: ResponseMetadata,
    ) -> Result<StateEntry, AgentError> {
        let entry = self
            .state
            .append_compaction(summary, first_kept_id, tokens_before, Some(response))
            .map_err(AgentError::State)?;
        if let Err(error) = self.commit.commit(&entry).await {
            self.state.rollback(entry.id()).map_err(AgentError::State)?;
            return Err(error.into());
        }
        Ok(entry)
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
        events
            .send(AgentEvent::CompactionStarted { tokens_before })
            .await
            .map_err(|error| AgentError::Task(error.to_string()))?;
        let request = CreateResponseArgs::default()
            .model(self.model.model.clone())
            .instructions(SUMMARY_INSTRUCTIONS)
            .input(prepared.prompt)
            .store(false)
            .build()?;
        let responses = self.model.client.responses();
        let response = tokio::select! {
            _ = cancel.cancelled() => return Err(AgentError::Aborted),
            response = responses.create(request) => response?,
        };
        let summary = response.output_text().unwrap_or_default();
        if summary.trim().is_empty() {
            return Err(AgentError::ModelResponse(
                "compaction returned an empty summary".into(),
            ));
        }
        self.append_compaction(
            summary.clone(),
            prepared.first_kept_id,
            tokens_before,
            ResponseMetadata {
                id: response.id,
                model: response.model,
                usage: response.usage,
            },
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

    async fn close_unresolved_tools(&mut self) -> Result<(), AgentError> {
        let items = self
            .state
            .unresolved_tool_call_ids()
            .into_iter()
            .map(|call_id| {
                function_output_by_id(
                    call_id,
                    "Tool execution was cancelled; its side effects are unknown. Do not retry automatically."
                        .into(),
                )
            })
            .collect::<Vec<_>>();
        if !items.is_empty() {
            self.append_items(items, None).await?;
        }
        Ok(())
    }
}

struct RunChannels {
    steer: mpsc::UnboundedReceiver<String>,
    queue: mpsc::UnboundedReceiver<String>,
    approvals: mpsc::UnboundedReceiver<(String, bool)>,
    cancel: CancellationToken,
}

#[derive(Clone)]
pub struct RunControl {
    steer: mpsc::UnboundedSender<String>,
    queue: mpsc::UnboundedSender<String>,
    approvals: mpsc::UnboundedSender<(String, bool)>,
    cancel: CancellationToken,
}

impl RunControl {
    pub fn steer(&self, message: impl Into<String>) -> Result<(), AgentError> {
        self.steer
            .send(message.into())
            .map_err(|error| AgentError::Task(error.to_string()))
    }

    pub fn queue(&self, message: impl Into<String>) -> Result<(), AgentError> {
        self.queue
            .send(message.into())
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

    use super::{ActiveAgent, Agent, AgentEvent, Model, RunControl};
    use crate::{
        AgentTool, Env, Session, SessionError, SessionInfo, State, StateCommit, StateEntry,
        ToolResult,
    };

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
            let mut request = vec![0; 16 * 1024];
            let _ = socket.read(&mut request).await.unwrap();
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
        );
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

        fn set_initial_title<'a>(
            &'a mut self,
            _message: &'a str,
        ) -> BoxFuture<'a, Result<(), SessionError>> {
            Box::pin(async { Ok(()) })
        }

        fn commit<'a>(
            &'a mut self,
            _entry: &'a StateEntry,
        ) -> BoxFuture<'a, Result<(), SessionError>> {
            Box::pin(async { Err(SessionError::Invalid("commit failed".into())) })
        }
    }

    #[tokio::test]
    async fn failed_commit_rolls_back_state() {
        let agent = Agent::from_parts(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            State::default(),
            Box::new(FailingCommit {
                info: SessionInfo {
                    path: PathBuf::new(),
                    title: "test".into(),
                    created_at: 0,
                },
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
