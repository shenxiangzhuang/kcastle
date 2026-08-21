use async_openai::types::responses::{InputItem, ResponseUsage, Tool};
use serde::{Deserialize, Serialize};

use crate::session::{InputMode, SessionConfig};
use crate::state::ResponseMetadata;

pub const SESSION_FORMAT_VERSION: u32 = 1;

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
pub struct RecordedEvent {
    pub seq: u64,
    pub time: EventTime,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_event_seqs: Vec<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_op: Option<SurfaceOp>,
    #[serde(flatten)]
    pub event: SessionEvent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SurfaceOp {
    Append,
    Replace { replaced_event_seqs: Vec<u64> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    TurnStart {
        turn: u32,
    },
    TurnEnd {
        turn: u32,
        reason: TurnEndReason,
    },
    StepStart {
        turn: u32,
        step: u32,
    },
    StepEnd {
        turn: u32,
        step: u32,
        outcome: StepOutcome,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    InputAdmitted {
        id: String,
        input: String,
        mode: InputMode,
    },
    InputConsumed {
        id: String,
    },
    UserMessage {
        turn: u32,
        step: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        input_id: Option<String>,
        mode: UserMessageMode,
        items: Vec<InputItem>,
    },
    RequestHeader {
        turn: u32,
        step: u32,
        reason: RequestHeaderReason,
        model: String,
        instructions: String,
        tools: Vec<Tool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_effort: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_output_tokens: Option<u32>,
        session_config: SessionConfig,
    },
    ModelRequestStart {
        turn: u32,
        step: u32,
    },
    AssistantChunk {
        turn: u32,
        step: u32,
        chunk: AssistantChunk,
    },
    AssistantMessage {
        turn: u32,
        step: u32,
        items: Vec<InputItem>,
        response: ResponseMetadata,
    },
    ToolCall {
        turn: u32,
        step: u32,
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parent_call_id: Option<String>,
        name: String,
        arguments: String,
    },
    ToolExecutionStart {
        call_id: String,
    },
    ToolExecutionFinish {
        call_id: String,
        outcome: ToolExecutionOutcome,
    },
    ToolResult {
        turn: u32,
        step: u32,
        call_id: String,
        output: String,
        status: ToolResultStatus,
        item: InputItem,
    },
    CompactionStart {
        compaction_id: String,
        tokens_before: usize,
        first_kept_id: u64,
    },
    CompactionEnd {
        compaction_id: String,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<ResponseMetadata>,
        outcome: StepOutcome,
    },
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
pub enum UserMessageMode {
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
    ToolCallArgumentsDelta {
        call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        delta: String,
    },
    Usage {
        usage: ResponseUsage,
    },
}

impl AssistantChunk {
    pub fn is_non_empty_token(&self) -> bool {
        match self {
            Self::OutputTextDelta { delta }
            | Self::ReasoningTextDelta { delta }
            | Self::ToolCallArgumentsDelta { delta, .. } => !delta.is_empty(),
            Self::Usage { .. } => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionOutcome {
    Success,
    Error,
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
