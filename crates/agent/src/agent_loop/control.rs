use async_openai::error::OpenAIError;
use async_openai::types::responses::{FunctionToolCall, ResponseUsage};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::session::SessionError;
use crate::session::event::{InputId, InputOrigin};
use crate::session::machine::SessionMachineError;
use crate::session::store::{CommitReceipt, SessionStoreError};

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunFailure {
    message: String,
    retryable: bool,
}

impl RunFailure {
    pub fn new(message: impl Into<String>, retryable: bool) -> Self {
        Self {
            message: message.into(),
            retryable,
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn retryable(&self) -> bool {
        self.retryable
    }

    pub(crate) fn from_error(error: &AgentError) -> Self {
        use crate::session::store::SessionErrorClass;

        let retryable = match error {
            AgentError::MissingResponse | AgentError::ModelResponse(_) | AgentError::Task(_) => {
                true
            }
            AgentError::OpenAI(error) => openai_error_is_retryable(error),
            AgentError::Session(error) => error.classification() == SessionErrorClass::Transient,
            AgentError::Store(error) => error.classification() == SessionErrorClass::Transient,
            AgentError::EmptyInput
            | AgentError::MaxTurns(_)
            | AgentError::NothingToCompact
            | AgentError::Aborted
            | AgentError::Machine(_) => false,
        };
        Self::new(error.to_string(), retryable)
    }
}

fn openai_error_is_retryable(error: &OpenAIError) -> bool {
    match error {
        OpenAIError::Reqwest(error) => {
            error.is_connect()
                || error.is_timeout()
                || error.status().is_some_and(|status| {
                    status.as_u16() == 408 || status.as_u16() == 429 || status.is_server_error()
                })
        }
        OpenAIError::ApiError(error) => {
            error.status_code.as_u16() == 408
                || error.status_code.as_u16() == 429
                || error.status_code.is_server_error()
        }
        OpenAIError::StreamError(_) => true,
        _ => false,
    }
}

impl From<String> for RunFailure {
    fn from(message: String) -> Self {
        Self::new(message, false)
    }
}

impl From<&str> for RunFailure {
    fn from(message: &str) -> Self {
        Self::new(message, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_failure_marks_only_recoverable_agent_errors_as_retryable() {
        assert!(RunFailure::from_error(&AgentError::MissingResponse).retryable());
        assert!(!RunFailure::from_error(&AgentError::EmptyInput).retryable());
        assert!(
            !RunFailure::from_error(&AgentError::OpenAI(OpenAIError::InvalidArgument(
                "bad model".into()
            )))
            .retryable()
        );
    }
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub output: String,
    pub response_id: String,
    pub usage: Option<ResponseUsage>,
}

/// Durable content is published exclusively through `SessionCommitted`; the other variants are
/// the minimal transient control and completion protocol required by a host application.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    SessionCommitted(CommitReceipt),
    ApprovalRequired(FunctionToolCall),
    RunFinished(RunSummary),
    RunAborted,
    RunFailed(RunFailure),
}

pub(super) struct InputCommand {
    pub(super) input_id: InputId,
    pub(super) input: String,
    pub(super) origin: InputOrigin,
    pub(super) acknowledgement: oneshot::Sender<Result<(), String>>,
}

pub(super) struct ApprovalCommand {
    pub(super) call_id: String,
    pub(super) allow: bool,
    pub(super) acknowledgement: oneshot::Sender<Result<(), String>>,
}

pub(super) struct RunChannels {
    pub(super) commands: mpsc::UnboundedReceiver<InputCommand>,
    pub(super) approvals: mpsc::UnboundedReceiver<ApprovalCommand>,
    pub(super) cancel: CancellationToken,
}

#[derive(Clone)]
pub struct RunControl {
    pub(super) commands: mpsc::UnboundedSender<InputCommand>,
    pub(super) approvals: mpsc::UnboundedSender<ApprovalCommand>,
    pub(super) cancel: CancellationToken,
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
    pub(super) control: RunControl,
    pub(super) events: mpsc::UnboundedReceiver<AgentEvent>,
    pub(super) task: JoinHandle<Agent>,
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
        self.control.abort();
    }
}
