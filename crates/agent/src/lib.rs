#![cfg_attr(
    not(test),
    deny(
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::unwrap_used
    )
)]

mod agent;
mod agent_loop;
mod context;
mod model;
mod session;
mod tools;

pub use async_openai::types::responses::{EasyInputMessage, InputItem};

pub use agent::Agent;
pub use agent_loop::{ActiveAgent, AgentError, AgentEvent, RunControl, RunFailure, RunSummary};
pub use context::compaction::CompactionConfig;
pub use model::{Model, ReasoningEffort};
pub use session::event::{
    AssistantChunk, CallId, CompactionId, EventTime, InputId, InputOrigin, RecordedEvent,
    RequestHeaderReason, RequestId, ResponseInfo, RunId, RunOutcome, SESSION_FORMAT_VERSION,
    SessionEvent, StepId, StepOutcome, TokenUsage, ToolAuthorizationDecision, ToolExecutionOutcome,
    ToolResultStatus, TurnEndReason, TurnId, TxId,
};
pub use session::machine::{PendingInput, SessionMachineError};
pub use session::store::{
    CommitReceipt, SESSION_DATABASE_FILE, SessionErrorClass, SessionStoreError,
};
pub use session::{
    DEFAULT_PROJECT_ID, Session, SessionCatalog, SessionConfig, SessionError, SessionId,
    SessionInfo, SessionModelConfig, SessionSnapshot, validate_events,
};
pub use tools::{AgentTool, Env, ShellTool, ToolResult};
