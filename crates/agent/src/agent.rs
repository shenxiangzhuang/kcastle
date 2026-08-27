use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use async_openai::types::responses::Tool;

use crate::agent_loop::{self, ActiveAgent, AgentError};
use crate::context::compaction::CompactionConfig;
use crate::model::{Model, ReasoningEffort};
use crate::session::event::EventTime;
use crate::session::machine::SessionMachine;
use crate::session::store::{MetadataUpdate, SessionStore, SessionWriterPermit};
use crate::session::{Session, SessionConfig, SessionInfo, SessionParts};
use crate::tools::{AgentTool, Env, ShellTool};

const DEFAULT_MAX_TURNS: usize = 100;

#[derive(Clone)]
pub(crate) struct EventClock {
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

    pub(crate) fn now(&self) -> EventTime {
        EventTime {
            wall_time_ms: system_time_ms(),
            clock_id: self.inner.id.clone(),
            monotonic_ns: u64::try_from(self.inner.started.elapsed().as_nanos())
                .unwrap_or(u64::MAX),
        }
    }
}

/// Idle owner of one replayable session and the dependencies used by its next operation.
pub struct Agent {
    pub(crate) model: Model,
    pub(crate) instructions: String,
    pub(crate) machine: SessionMachine,
    pub(crate) store: SessionStore,
    pub(crate) revision: u64,
    pub(crate) writer: Option<SessionWriterPermit>,
    pub(crate) info: SessionInfo,
    pub(crate) session_config: SessionConfig,
    pub(crate) env: Env,
    pub(crate) tools: Vec<Arc<dyn AgentTool>>,
    pub(crate) compaction: Option<CompactionConfig>,
    pub(crate) max_turns: usize,
    pub(crate) clock: EventClock,
}

impl Agent {
    #[allow(clippy::expect_used, reason = "DEFAULT_MAX_TURNS is nonzero")]
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
        agent_loop::start(self, input.into())
    }

    pub fn start_compaction(self, instructions: Option<String>) -> ActiveAgent {
        agent_loop::start_compaction(self, instructions)
    }

    pub(crate) async fn acquire_or_clone_writer(&self) -> Result<SessionWriterPermit, AgentError> {
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
}

fn millis_to_seconds(millis: i64) -> u64 {
    u64::try_from(millis.max(0)).unwrap_or(0) / 1_000
}

fn system_time_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}
