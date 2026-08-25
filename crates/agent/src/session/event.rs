use std::fmt;

use async_openai::types::responses::{
    InputItem, InputTokenDetails, OutputTokenDetails, ResponseUsage, Tool,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::context::ResponseMetadata;
use crate::session::SessionConfig;

pub const SESSION_FORMAT_VERSION: u32 = 3;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            pub fn random() -> Self {
                Self(Uuid::new_v4().to_string())
            }

            pub fn from_raw(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.0.fmt(formatter)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }
    };
}

string_id!(TxId);
string_id!(InputId);
string_id!(RunId);
string_id!(TurnId);
string_id!(StepId);
string_id!(RequestId);
string_id!(CallId);
string_id!(CompactionId);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventTime {
    pub wall_time_ms: i64,
    pub clock_id: String,
    pub monotonic_ns: u64,
}

impl EventTime {
    pub fn duration_since(&self, earlier: &Self) -> Option<u64> {
        (self.clock_id == earlier.clock_id)
            .then(|| self.monotonic_ns.checked_sub(earlier.monotonic_ns))
            .flatten()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventDraft {
    pub tx_id: TxId,
    pub time: EventTime,
    #[serde(flatten)]
    pub event: SessionEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedEvent {
    pub seq: u64,
    pub tx_id: TxId,
    pub time: EventTime,
    #[serde(flatten)]
    pub event: SessionEvent,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub uncached_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_write_input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_output_tokens: u64,
}

impl TokenUsage {
    pub fn input_tokens(self) -> u64 {
        self.uncached_input_tokens
            .saturating_add(self.cache_read_input_tokens)
            .saturating_add(self.cache_write_input_tokens)
    }

    pub fn total_output_tokens(self) -> u64 {
        self.output_tokens
            .saturating_add(self.reasoning_output_tokens)
    }

    pub fn from_provider(usage: &ResponseUsage) -> Self {
        let cached = u64::from(usage.input_tokens_details.cached_tokens);
        let reasoning = u64::from(usage.output_tokens_details.reasoning_tokens);
        Self {
            uncached_input_tokens: u64::from(usage.input_tokens).saturating_sub(cached),
            cache_read_input_tokens: cached,
            cache_write_input_tokens: 0,
            output_tokens: u64::from(usage.output_tokens).saturating_sub(reasoning),
            reasoning_output_tokens: reasoning,
        }
    }

    pub fn to_provider(self) -> ResponseUsage {
        let input_tokens = u32::try_from(self.input_tokens()).unwrap_or(u32::MAX);
        let output_tokens = u32::try_from(self.total_output_tokens()).unwrap_or(u32::MAX);
        ResponseUsage {
            input_tokens,
            input_tokens_details: InputTokenDetails {
                cached_tokens: u32::try_from(self.cache_read_input_tokens).unwrap_or(u32::MAX),
            },
            output_tokens,
            output_tokens_details: OutputTokenDetails {
                reasoning_tokens: u32::try_from(self.reasoning_output_tokens).unwrap_or(u32::MAX),
            },
            total_tokens: input_tokens.saturating_add(output_tokens),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResponseInfo {
    pub id: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl ResponseInfo {
    pub(crate) fn to_provider(&self) -> ResponseMetadata {
        ResponseMetadata {
            id: self.id.clone(),
            model: self.model.clone(),
            usage: self.usage.map(TokenUsage::to_provider),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    RunStarted {
        run_id: RunId,
    },
    RunTerminated {
        run_id: RunId,
        outcome: RunOutcome,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    TurnStarted {
        run_id: RunId,
        turn_id: TurnId,
    },
    TurnTerminated {
        turn_id: TurnId,
        reason: TurnEndReason,
    },
    StepStarted {
        turn_id: TurnId,
        step_id: StepId,
    },
    StepTerminated {
        step_id: StepId,
        outcome: StepOutcome,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    InputSubmitted {
        input_id: InputId,
        input: String,
        origin: InputOrigin,
    },
    InputAttached {
        input_id: InputId,
        step_id: StepId,
        items: Vec<InputItem>,
    },
    RequestSnapshot {
        request_id: RequestId,
        step_id: StepId,
        reason: RequestHeaderReason,
        model: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        instructions: Option<String>,
        tools: Vec<Tool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_output_tokens: Option<u32>,
        session_config: SessionConfig,
    },
    ModelRequestStarted {
        request_id: RequestId,
    },
    ModelRequestFailed {
        request_id: RequestId,
        error: String,
    },
    AssistantChunk {
        request_id: RequestId,
        chunk: AssistantChunk,
    },
    AssistantCompleted {
        request_id: RequestId,
        items: Vec<InputItem>,
        response: ResponseInfo,
    },
    ToolCallRequested {
        request_id: RequestId,
        call_id: CallId,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_call_id: Option<CallId>,
    },
    ToolAuthorizationResolved {
        call_id: CallId,
        decision: ToolAuthorizationDecision,
    },
    ToolDispatchIntended {
        call_id: CallId,
    },
    ToolExecutionStarted {
        call_id: CallId,
    },
    ToolExecutionFinished {
        call_id: CallId,
        outcome: ToolExecutionOutcome,
    },
    ToolResultAttached {
        call_id: CallId,
        status: ToolResultStatus,
        item: InputItem,
    },
    CompactionStarted {
        compaction_id: CompactionId,
        run_id: RunId,
        tokens_before: usize,
        first_kept_id: u64,
    },
    CompactionFinished {
        compaction_id: CompactionId,
        outcome: StepOutcome,
        #[serde(skip_serializing_if = "Option::is_none")]
        summary: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseInfo>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOutcome {
    Completed,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnEndReason {
    Completed,
    Failed,
    Aborted,
    MaxTurns,
    ToolConcluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepOutcome {
    Completed,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputOrigin {
    Initial,
    Steer,
    Queue,
    Context,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestHeaderReason {
    Initial,
    Resume,
    Change,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssistantChunk {
    OutputTextDelta {
        delta: String,
    },
    ReasoningTextDelta {
        delta: String,
    },
    ToolCallDelta {
        call_id: CallId,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        arguments_delta: String,
    },
    Usage {
        usage: TokenUsage,
    },
}

impl AssistantChunk {
    pub fn is_token_delta(&self) -> bool {
        match self {
            Self::OutputTextDelta { delta } | Self::ReasoningTextDelta { delta } => {
                !delta.is_empty()
            }
            Self::ToolCallDelta {
                name,
                arguments_delta,
                ..
            } => {
                !arguments_delta.is_empty() || name.as_deref().is_some_and(|name| !name.is_empty())
            }
            Self::Usage { .. } => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolAuthorizationDecision {
    NotRequired,
    Allowed,
    Denied,
    Unavailable,
    Aborted,
}

impl ToolAuthorizationDecision {
    pub fn permits_execution(self) -> bool {
        matches!(self, Self::NotRequired | Self::Allowed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionOutcome {
    Success,
    Error,
    UnknownSideEffects,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultStatus {
    Success,
    Error,
    Denied,
    NotFound,
    AbortedBeforeDispatch,
    UnknownSideEffects,
}
