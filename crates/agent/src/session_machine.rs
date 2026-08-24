use std::sync::Arc;

use async_openai::types::responses::{
    EasyInputContent, FunctionCallOutput, FunctionCallOutputItemParam, InputItem, InputParam, Item,
    Role, Tool,
};
use im::{HashMap, HashSet, OrdMap};
use thiserror::Error;

use crate::session::SessionConfig;
use crate::session_event::{
    AssistantChunk, CallId, CompactionId, EventDraft, EventTime, InputId, InputOrigin,
    RecordedEvent, RequestHeaderReason, RequestId, ResponseInfo, RunId, RunOutcome, SessionEvent,
    StepId, StepOutcome, ToolAuthorizationDecision, ToolExecutionOutcome, ToolResultStatus,
    TurnEndReason, TurnId, TxId,
};
use crate::state::State;
#[cfg(test)]
use crate::state::StateEntry;

/// Increment when an existing serialized event sequence can no longer be interpreted with the
/// same validity and state-transition semantics.
pub(crate) const SESSION_MACHINE_SEMANTICS_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingInput {
    pub input_id: InputId,
    pub input: String,
    pub origin: InputOrigin,
}

#[derive(Debug)]
pub struct PlannedBatch {
    tx_id: TxId,
    expected_seq: u64,
    events: Vec<RecordedEvent>,
    base_identity: Arc<()>,
    candidate: Box<SessionMachine>,
}

impl PlannedBatch {
    pub fn tx_id(&self) -> &TxId {
        &self.tx_id
    }

    pub fn expected_seq(&self) -> u64 {
        self.expected_seq
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn into_events(self) -> Vec<RecordedEvent> {
        self.events
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SessionMachineError {
    #[error("event sequence is {found}, expected {expected}")]
    Sequence { expected: u64, found: u64 },
    #[error("event batch must not be empty")]
    EmptyBatch,
    #[error("event batch mixes transaction {expected} with {found}")]
    MixedTransaction { expected: TxId, found: TxId },
    #[error("transaction {0} was already committed")]
    ReusedTransaction(TxId),
    #[error("invalid session event: {0}")]
    Invalid(String),
    #[error("could not project event into model state: {0}")]
    State(String),
}

#[derive(Debug, Clone)]
struct InputRecord {
    input: String,
    origin: InputOrigin,
    attached_step: Option<StepId>,
}

#[derive(Debug, Clone)]
struct RunRecord {
    terminal: Option<RunOutcome>,
    turn_count: u64,
    failed_turns: u64,
    aborted_turns: u64,
}

#[derive(Debug, Clone)]
struct TurnRecord {
    run_id: RunId,
    terminal: Option<TurnEndReason>,
    step_count: u64,
    failed_steps: u64,
    aborted_steps: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepPhase {
    AwaitingRequest,
    RequestActive,
    RequestFailed,
    AwaitingTools,
    AssistantCompleted,
    Terminal,
}

#[derive(Debug, Clone)]
struct StepRecord {
    turn_id: TurnId,
    terminal: Option<StepOutcome>,
    phase: StepPhase,
    current_request: Option<RequestId>,
    first_token: Option<EventTime>,
}

#[derive(Debug, Clone, PartialEq)]
struct RequestConfig {
    model: String,
    instructions: Option<String>,
    tools: Vec<Tool>,
    reasoning_effort: Option<String>,
    max_output_tokens: Option<u32>,
    session_config: SessionConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestTerminal {
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
struct RequestRecord {
    step_id: StepId,
    started: Option<EventTime>,
    first_token: Option<EventTime>,
    terminal: Option<RequestTerminal>,
    declared_tool_calls: Vec<CallId>,
}

#[derive(Debug, Clone)]
struct ToolRecord {
    request_id: RequestId,
    step_id: StepId,
    authorization: Option<ToolAuthorizationDecision>,
    dispatch_intended: Option<EventTime>,
    execution_started: Option<EventTime>,
    execution_finished: Option<ToolExecutionOutcome>,
    result_attached: bool,
}

#[derive(Debug, Clone)]
struct CompactionRecord {
    run_id: RunId,
    tokens_before: usize,
    first_kept_id: u64,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct SessionMachine {
    next_seq: u64,
    state: State,
    inputs: HashMap<InputId, InputRecord>,
    next_input_ordinal: u64,
    pending_input_ordinals: HashMap<InputId, u64>,
    pending_inputs_order: OrdMap<u64, InputId>,
    runs: HashMap<RunId, RunRecord>,
    turns: HashMap<TurnId, TurnRecord>,
    steps: HashMap<StepId, StepRecord>,
    requests: HashMap<RequestId, RequestRecord>,
    tools: HashMap<CallId, ToolRecord>,
    next_tool_ordinal: u64,
    open_tool_ordinals: HashMap<CallId, u64>,
    open_tools: OrdMap<u64, CallId>,
    pending_tool_registration: Option<(RequestId, usize)>,
    compactions: HashMap<CompactionId, CompactionRecord>,
    active_run: Option<RunId>,
    active_turn: Option<TurnId>,
    active_step: Option<StepId>,
    active_compaction: Option<CompactionId>,
    last_request_config: Option<Arc<RequestConfig>>,
    current_tx: Option<TxId>,
    seen_txs: HashSet<TxId>,
    // Clones share this token until a planned transition is applied. It prevents applying a
    // validated candidate to a machine that has since advanced or diverged at the same sequence.
    identity: Arc<()>,
}

impl Default for SessionMachine {
    fn default() -> Self {
        Self {
            next_seq: 0,
            state: State::default(),
            inputs: HashMap::new(),
            next_input_ordinal: 0,
            pending_input_ordinals: HashMap::new(),
            pending_inputs_order: OrdMap::new(),
            runs: HashMap::new(),
            turns: HashMap::new(),
            steps: HashMap::new(),
            requests: HashMap::new(),
            tools: HashMap::new(),
            next_tool_ordinal: 0,
            open_tool_ordinals: HashMap::new(),
            open_tools: OrdMap::new(),
            pending_tool_registration: None,
            compactions: HashMap::new(),
            active_run: None,
            active_turn: None,
            active_step: None,
            active_compaction: None,
            last_request_config: None,
            current_tx: None,
            seen_txs: HashSet::new(),
            identity: Arc::new(()),
        }
    }
}

impl SessionMachine {
    pub fn from_events(events: &[RecordedEvent]) -> Result<Self, SessionMachineError> {
        let mut machine = Self::default();
        let mut transaction = None;
        for event in events {
            if transaction
                .as_ref()
                .is_some_and(|tx_id| tx_id != &event.tx_id)
            {
                machine.validate_transaction_boundary()?;
            }
            machine.apply_in_place(event)?;
            transaction = Some(event.tx_id.clone());
        }
        machine.validate_transaction_boundary()?;
        Ok(machine)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn context(&self) -> InputParam {
        self.state.context()
    }

    pub fn pending_inputs(&self) -> Vec<PendingInput> {
        self.pending_inputs_order
            .values()
            .filter_map(|input_id| {
                let record = self.inputs.get(input_id)?;
                debug_assert!(record.attached_step.is_none());
                Some(PendingInput {
                    input_id: input_id.clone(),
                    input: record.input.clone(),
                    origin: record.origin,
                })
            })
            .collect()
    }

    pub fn active_run(&self) -> Option<&RunId> {
        self.active_run.as_ref()
    }

    pub fn active_turn(&self) -> Option<&TurnId> {
        self.active_turn.as_ref()
    }

    pub fn active_step(&self) -> Option<&StepId> {
        self.active_step.as_ref()
    }

    /// The request currently owned by the active step, if it has not reached a terminal event.
    /// Runtimes use this to include `ModelRequestFailed` in the same terminal transaction as the
    /// enclosing step, turn, and run.
    pub fn active_request(&self) -> Option<&RequestId> {
        self.active_step
            .as_ref()
            .and_then(|step_id| self.steps.get(step_id))
            .and_then(|step| step.current_request.as_ref())
    }

    pub fn active_compaction(&self) -> Option<&CompactionId> {
        self.active_compaction.as_ref()
    }

    pub fn step_first_token(&self, step_id: &StepId) -> Option<&EventTime> {
        self.steps
            .get(step_id)
            .and_then(|step| step.first_token.as_ref())
    }

    pub fn request_first_token(&self, request_id: &RequestId) -> Option<&EventTime> {
        self.requests
            .get(request_id)
            .and_then(|request| request.first_token.as_ref())
    }

    pub fn unresolved_tool_calls(&self) -> Vec<CallId> {
        self.open_tools.values().cloned().collect()
    }

    /// Returns the only valid header reason for the next model request.
    ///
    /// This comparison deliberately covers every request parameter persisted by
    /// [`SessionEvent::RequestSnapshot`]. Callers should ask the machine instead
    /// of maintaining a second request fingerprint.
    pub fn expected_request_reason(
        &self,
        model: &str,
        instructions: Option<&str>,
        tools: &[Tool],
        reasoning_effort: Option<&str>,
        max_output_tokens: Option<u32>,
        session_config: &SessionConfig,
    ) -> RequestHeaderReason {
        let Some(previous) = self.last_request_config.as_ref() else {
            return RequestHeaderReason::Initial;
        };
        if previous.model == model
            && previous.instructions.as_deref() == instructions
            && previous.tools == tools
            && previous.reasoning_effort.as_deref() == reasoning_effort
            && previous.max_output_tokens == max_output_tokens
            && previous.session_config == *session_config
        {
            RequestHeaderReason::Resume
        } else {
            RequestHeaderReason::Change
        }
    }

    pub fn plan_batch(&self, drafts: Vec<EventDraft>) -> Result<PlannedBatch, SessionMachineError> {
        let Some(first) = drafts.first() else {
            return Err(SessionMachineError::EmptyBatch);
        };
        ensure_non_empty_id("transaction", &first.tx_id)?;
        let tx_id = first.tx_id.clone();
        if self.seen_txs.contains(&tx_id) {
            return Err(SessionMachineError::ReusedTransaction(tx_id));
        }
        for draft in &drafts {
            if draft.tx_id != tx_id {
                return Err(SessionMachineError::MixedTransaction {
                    expected: tx_id,
                    found: draft.tx_id.clone(),
                });
            }
        }

        let expected_seq = self.next_seq;
        let events = drafts
            .into_iter()
            .enumerate()
            .map(|(offset, draft)| RecordedEvent {
                seq: expected_seq.saturating_add(offset as u64),
                tx_id: draft.tx_id,
                time: draft.time,
                event: draft.event,
            })
            .collect::<Vec<_>>();
        let base_identity = self.identity.clone();
        let mut candidate = self.clone();
        for event in &events {
            candidate.apply_in_place(event)?;
        }
        candidate.validate_transaction_boundary()?;
        candidate.identity = Arc::new(());
        Ok(PlannedBatch {
            tx_id,
            expected_seq,
            events,
            base_identity,
            candidate: Box::new(candidate),
        })
    }

    pub fn apply_batch(&mut self, batch: PlannedBatch) -> Result<(), SessionMachineError> {
        if batch.events.is_empty() {
            return Err(SessionMachineError::EmptyBatch);
        }
        if batch.expected_seq != self.next_seq {
            return Err(SessionMachineError::Sequence {
                expected: self.next_seq,
                found: batch.expected_seq,
            });
        }
        if !Arc::ptr_eq(&self.identity, &batch.base_identity) {
            return invalid("planned transaction no longer matches the session machine");
        }
        *self = *batch.candidate;
        Ok(())
    }

    pub fn plan_recovery(
        &self,
        tx_id: TxId,
        time: EventTime,
    ) -> Result<Option<PlannedBatch>, SessionMachineError> {
        let mut events = Vec::new();
        for call_id in self.unresolved_tool_calls() {
            let tool = self
                .tools
                .get(&call_id)
                .expect("unresolved call came from the tool index");
            let decision = tool.authorization;
            if decision.is_none() {
                events.push(SessionEvent::ToolAuthorizationResolved {
                    call_id: call_id.clone(),
                    decision: ToolAuthorizationDecision::Aborted,
                });
            }
            let status = match decision {
                Some(ToolAuthorizationDecision::Denied) => ToolResultStatus::Denied,
                Some(ToolAuthorizationDecision::Unavailable) => ToolResultStatus::NotFound,
                Some(ToolAuthorizationDecision::Aborted) | None => {
                    ToolResultStatus::AbortedBeforeDispatch
                }
                Some(ToolAuthorizationDecision::Allowed)
                | Some(ToolAuthorizationDecision::NotRequired) => match tool.execution_finished {
                    Some(ToolExecutionOutcome::Success) => ToolResultStatus::Success,
                    Some(ToolExecutionOutcome::Error) => ToolResultStatus::Error,
                    Some(ToolExecutionOutcome::UnknownSideEffects) => {
                        ToolResultStatus::UnknownSideEffects
                    }
                    None if tool.dispatch_intended.is_some() => {
                        events.push(SessionEvent::ToolExecutionFinished {
                            call_id: call_id.clone(),
                            outcome: ToolExecutionOutcome::UnknownSideEffects,
                        });
                        ToolResultStatus::UnknownSideEffects
                    }
                    None => ToolResultStatus::AbortedBeforeDispatch,
                },
            };
            let output = recovery_tool_output(status);
            events.push(SessionEvent::ToolResultAttached {
                call_id: call_id.clone(),
                status,
                item: tool_output_item(&call_id, output),
            });
        }

        if let Some(compaction_id) = self.active_compaction.clone() {
            events.push(SessionEvent::CompactionFinished {
                compaction_id,
                outcome: StepOutcome::Aborted,
                summary: None,
                response: None,
            });
        }
        if let Some(request_id) = self.active_request_id().cloned() {
            events.push(SessionEvent::ModelRequestFailed {
                request_id,
                error: "model request was interrupted before completion".into(),
            });
        }
        if let Some(step_id) = self.active_step.clone() {
            events.push(SessionEvent::StepTerminated {
                step_id,
                outcome: StepOutcome::Aborted,
                error: Some("session recovered after an interrupted step".into()),
            });
        }
        if let Some(turn_id) = self.active_turn.clone() {
            events.push(SessionEvent::TurnTerminated {
                turn_id,
                reason: TurnEndReason::Aborted,
            });
        }
        if let Some(run_id) = self.active_run.clone() {
            events.push(SessionEvent::RunTerminated {
                run_id,
                outcome: RunOutcome::Aborted,
                error: Some("session recovered after an interrupted run".into()),
            });
        }
        if events.is_empty() {
            return Ok(None);
        }
        let drafts = events
            .into_iter()
            .map(|event| EventDraft {
                tx_id: tx_id.clone(),
                time: time.clone(),
                event,
            })
            .collect();
        self.plan_batch(drafts).map(Some)
    }

    fn apply_in_place(&mut self, recorded: &RecordedEvent) -> Result<(), SessionMachineError> {
        if recorded.seq != self.next_seq {
            return Err(SessionMachineError::Sequence {
                expected: self.next_seq,
                found: recorded.seq,
            });
        }
        ensure_non_empty_id("transaction", &recorded.tx_id)?;
        if self.current_tx.as_ref() != Some(&recorded.tx_id)
            && self.seen_txs.contains(&recorded.tx_id)
        {
            return Err(SessionMachineError::ReusedTransaction(
                recorded.tx_id.clone(),
            ));
        }

        self.apply_event(recorded)?;
        if self.current_tx.as_ref() != Some(&recorded.tx_id) {
            self.current_tx = Some(recorded.tx_id.clone());
            self.seen_txs.insert(recorded.tx_id.clone());
        }
        self.next_seq = self.next_seq.saturating_add(1);
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn apply_event(&mut self, recorded: &RecordedEvent) -> Result<(), SessionMachineError> {
        match &recorded.event {
            SessionEvent::RunStarted { run_id } => {
                ensure_non_empty_id("run", run_id)?;
                if self.active_run.is_some() || self.runs.contains_key(run_id) {
                    return invalid(format!("run {run_id} cannot start"));
                }
                self.runs.insert(
                    run_id.clone(),
                    RunRecord {
                        terminal: None,
                        turn_count: 0,
                        failed_turns: 0,
                        aborted_turns: 0,
                    },
                );
                self.active_run = Some(run_id.clone());
            }
            SessionEvent::RunTerminated {
                run_id,
                outcome,
                error,
            } => {
                if self.active_run.as_ref() != Some(run_id)
                    || self.active_turn.is_some()
                    || self.active_step.is_some()
                    || self.active_compaction.is_some()
                {
                    return invalid(format!("run {run_id} cannot terminate"));
                }
                let run = self
                    .runs
                    .get_mut(run_id)
                    .ok_or_else(|| invalid_error(format!("unknown run {run_id}")))?;
                if run.terminal.is_some() {
                    return invalid(format!("run {run_id} terminated twice"));
                }
                if !terminal_error_matches(*outcome == RunOutcome::Completed, error.as_deref()) {
                    return invalid(format!(
                        "run {run_id} outcome {outcome:?} has an inconsistent error"
                    ));
                }
                let outcome_matches_children = match outcome {
                    RunOutcome::Completed => run.failed_turns == 0 && run.aborted_turns == 0,
                    RunOutcome::Failed => {
                        run.turn_count == 0 || (run.failed_turns > 0 && run.aborted_turns == 0)
                    }
                    RunOutcome::Aborted => {
                        run.turn_count == 0 || (run.aborted_turns > 0 && run.failed_turns == 0)
                    }
                };
                if !outcome_matches_children {
                    return invalid(format!(
                        "run {run_id} outcome {outcome:?} contradicts its turns"
                    ));
                }
                run.terminal = Some(*outcome);
                self.active_run = None;
            }
            SessionEvent::TurnStarted { run_id, turn_id } => {
                ensure_non_empty_id("turn", turn_id)?;
                if self.active_run.as_ref() != Some(run_id)
                    || self.active_turn.is_some()
                    || self.active_compaction.is_some()
                    || self.turns.contains_key(turn_id)
                {
                    return invalid(format!("turn {turn_id} cannot start in run {run_id}"));
                }
                let run = self
                    .runs
                    .get_mut(run_id)
                    .ok_or_else(|| invalid_error(format!("unknown run {run_id}")))?;
                if run.failed_turns > 0 || run.aborted_turns > 0 {
                    return invalid(format!(
                        "run {run_id} cannot continue after a non-completed turn"
                    ));
                }
                run.turn_count = run.turn_count.saturating_add(1);
                self.turns.insert(
                    turn_id.clone(),
                    TurnRecord {
                        run_id: run_id.clone(),
                        terminal: None,
                        step_count: 0,
                        failed_steps: 0,
                        aborted_steps: 0,
                    },
                );
                self.active_turn = Some(turn_id.clone());
            }
            SessionEvent::TurnTerminated { turn_id, reason } => {
                if self.active_turn.as_ref() != Some(turn_id)
                    || self.active_step.is_some()
                    || self.active_compaction.is_some()
                {
                    return invalid(format!("turn {turn_id} cannot terminate"));
                }
                let turn = self
                    .turns
                    .get_mut(turn_id)
                    .ok_or_else(|| invalid_error(format!("unknown turn {turn_id}")))?;
                if turn.terminal.is_some() {
                    return invalid(format!("turn {turn_id} terminated twice"));
                }
                if self.active_run.as_ref() != Some(&turn.run_id) {
                    return invalid(format!("turn {turn_id} is outside its run"));
                }
                let reason_matches_steps = match reason {
                    TurnEndReason::Completed | TurnEndReason::ToolConcluded => {
                        turn.failed_steps == 0 && turn.aborted_steps == 0
                    }
                    TurnEndReason::Failed | TurnEndReason::MaxTurns => {
                        turn.failed_steps > 0 && turn.aborted_steps == 0
                    }
                    TurnEndReason::Aborted => turn.aborted_steps > 0 && turn.failed_steps == 0,
                };
                if !reason_matches_steps {
                    return invalid(format!(
                        "turn {turn_id} reason {reason:?} contradicts its steps"
                    ));
                }
                let run_id = turn.run_id.clone();
                turn.terminal = Some(*reason);
                let run = self
                    .runs
                    .get_mut(&run_id)
                    .ok_or_else(|| invalid_error(format!("unknown run {run_id}")))?;
                match reason {
                    TurnEndReason::Failed | TurnEndReason::MaxTurns => {
                        run.failed_turns = run.failed_turns.saturating_add(1);
                    }
                    TurnEndReason::Aborted => {
                        run.aborted_turns = run.aborted_turns.saturating_add(1);
                    }
                    TurnEndReason::Completed | TurnEndReason::ToolConcluded => {}
                }
                self.active_turn = None;
            }
            SessionEvent::StepStarted { turn_id, step_id } => {
                ensure_non_empty_id("step", step_id)?;
                if self.active_turn.as_ref() != Some(turn_id)
                    || self.active_step.is_some()
                    || self.steps.contains_key(step_id)
                {
                    return invalid(format!("step {step_id} cannot start in turn {turn_id}"));
                }
                let turn = self
                    .turns
                    .get_mut(turn_id)
                    .ok_or_else(|| invalid_error(format!("unknown turn {turn_id}")))?;
                if turn.failed_steps > 0 || turn.aborted_steps > 0 {
                    return invalid(format!(
                        "turn {turn_id} cannot continue after a non-completed step"
                    ));
                }
                turn.step_count = turn.step_count.saturating_add(1);
                self.steps.insert(
                    step_id.clone(),
                    StepRecord {
                        turn_id: turn_id.clone(),
                        terminal: None,
                        phase: StepPhase::AwaitingRequest,
                        current_request: None,
                        first_token: None,
                    },
                );
                self.active_step = Some(step_id.clone());
            }
            SessionEvent::StepTerminated {
                step_id,
                outcome,
                error,
            } => {
                if self.active_step.as_ref() != Some(step_id) || self.active_compaction.is_some() {
                    return invalid(format!("step {step_id} cannot terminate"));
                }
                if !self.open_tools.is_empty() {
                    return invalid(format!("step {step_id} has unresolved tools"));
                }
                if self.pending_tool_registration.is_some() {
                    return invalid(format!("step {step_id} has unregistered tool calls"));
                }
                let step = self
                    .steps
                    .get_mut(step_id)
                    .ok_or_else(|| invalid_error(format!("unknown step {step_id}")))?;
                if step.terminal.is_some() {
                    return invalid(format!("step {step_id} terminated twice"));
                }
                if !terminal_error_matches(*outcome == StepOutcome::Completed, error.as_deref()) {
                    return invalid(format!(
                        "step {step_id} outcome {outcome:?} has an inconsistent error"
                    ));
                }
                if self.active_turn.as_ref() != Some(&step.turn_id) {
                    return invalid(format!("step {step_id} is outside its turn"));
                }
                if step.current_request.is_some() {
                    return invalid(format!("step {step_id} has an open model request"));
                }
                if *outcome == StepOutcome::Completed && step.phase != StepPhase::AssistantCompleted
                {
                    return invalid(format!(
                        "completed step {step_id} requires a completed assistant"
                    ));
                }
                let turn_id = step.turn_id.clone();
                step.terminal = Some(*outcome);
                step.phase = StepPhase::Terminal;
                let turn = self
                    .turns
                    .get_mut(&turn_id)
                    .ok_or_else(|| invalid_error(format!("unknown turn {turn_id}")))?;
                match outcome {
                    StepOutcome::Failed => {
                        turn.failed_steps = turn.failed_steps.saturating_add(1);
                    }
                    StepOutcome::Aborted => {
                        turn.aborted_steps = turn.aborted_steps.saturating_add(1);
                    }
                    StepOutcome::Completed => {}
                }
                self.active_step = None;
            }
            SessionEvent::InputSubmitted {
                input_id,
                input,
                origin,
            } => {
                ensure_non_empty_id("input", input_id)?;
                if input.trim().is_empty() || self.inputs.contains_key(input_id) {
                    return invalid(format!("input {input_id} cannot be submitted"));
                }
                let ordinal = self.next_input_ordinal;
                self.inputs.insert(
                    input_id.clone(),
                    InputRecord {
                        input: input.clone(),
                        origin: *origin,
                        attached_step: None,
                    },
                );
                self.next_input_ordinal = self.next_input_ordinal.saturating_add(1);
                self.pending_input_ordinals
                    .insert(input_id.clone(), ordinal);
                self.pending_inputs_order.insert(ordinal, input_id.clone());
            }
            SessionEvent::InputAttached {
                input_id,
                step_id,
                items,
            } => {
                require_active_step(self.active_step.as_ref(), step_id)?;
                if self.active_compaction.is_some()
                    || self
                        .steps
                        .get(step_id)
                        .is_none_or(|step| step.phase != StepPhase::AwaitingRequest)
                {
                    return invalid(format!(
                        "input {input_id} cannot attach in the current step phase"
                    ));
                }
                let input = self
                    .inputs
                    .get(input_id)
                    .ok_or_else(|| invalid_error(format!("unknown input {input_id}")))?;
                if input.attached_step.is_some() {
                    return invalid(format!("input {input_id} was attached twice"));
                }
                validate_attached_input(input_id, input, items)?;
                self.state
                    .append_items(items.clone(), None)
                    .map_err(SessionMachineError::State)?;
                self.inputs
                    .get_mut(input_id)
                    .expect("validated input is indexed")
                    .attached_step = Some(step_id.clone());
                let ordinal = self
                    .pending_input_ordinals
                    .remove(input_id)
                    .expect("unattached input has an insertion ordinal");
                self.pending_inputs_order.remove(&ordinal);
            }
            SessionEvent::RequestSnapshot {
                request_id,
                step_id,
                reason,
                model,
                instructions,
                tools,
                reasoning_effort,
                max_output_tokens,
                session_config,
            } => {
                ensure_non_empty_id("request", request_id)?;
                require_active_step(self.active_step.as_ref(), step_id)?;
                let step = self.steps.get(step_id).expect("active step is indexed");
                if self.active_compaction.is_some()
                    || !self.open_tools.is_empty()
                    || model.trim().is_empty()
                    || instructions.as_deref().is_some_and(str::is_empty)
                    || reasoning_effort.as_deref().is_some_and(str::is_empty)
                    || self.requests.contains_key(request_id)
                    || step.phase != StepPhase::AwaitingRequest
                    || step.current_request.is_some()
                {
                    return invalid(format!("request {request_id} has an invalid snapshot"));
                }
                let expected_reason = self.expected_request_reason(
                    model,
                    instructions.as_deref(),
                    tools,
                    reasoning_effort.as_deref(),
                    *max_output_tokens,
                    session_config,
                );
                let config = RequestConfig {
                    model: model.clone(),
                    instructions: instructions.clone(),
                    tools: tools.clone(),
                    reasoning_effort: reasoning_effort.clone(),
                    max_output_tokens: *max_output_tokens,
                    session_config: session_config.clone(),
                };
                if *reason != expected_reason {
                    return invalid(format!(
                        "request {request_id} reason is {reason:?}, expected {expected_reason:?}"
                    ));
                }
                self.requests.insert(
                    request_id.clone(),
                    RequestRecord {
                        step_id: step_id.clone(),
                        started: None,
                        first_token: None,
                        terminal: None,
                        declared_tool_calls: Vec::new(),
                    },
                );
                let step = self.steps.get_mut(step_id).expect("active step is indexed");
                step.current_request = Some(request_id.clone());
                step.phase = StepPhase::RequestActive;
                self.last_request_config = Some(Arc::new(config));
            }
            SessionEvent::ModelRequestStarted { request_id } => {
                let request = self.open_request_mut(request_id)?;
                if request.started.is_some() {
                    return invalid(format!("request {request_id} started twice"));
                }
                request.started = Some(recorded.time.clone());
            }
            SessionEvent::ModelRequestFailed { request_id, error } => {
                if error.trim().is_empty() {
                    return invalid(format!("request {request_id} failure has no error"));
                }
                let step_id = {
                    let request = self.open_request_mut(request_id)?;
                    if request.started.is_none() {
                        return invalid(format!("request {request_id} was not started"));
                    }
                    request.terminal = Some(RequestTerminal::Failed);
                    request.step_id.clone()
                };
                self.clear_current_request(&step_id, request_id)?;
                self.steps
                    .get_mut(&step_id)
                    .expect("request step is indexed")
                    .phase = StepPhase::RequestFailed;
            }
            SessionEvent::AssistantChunk { request_id, chunk } => {
                if let AssistantChunk::ToolCallDelta { call_id, name, .. } = chunk {
                    ensure_non_empty_id("streaming tool call", call_id)?;
                    if name.as_deref() == Some("") {
                        return invalid(format!("streaming tool call {call_id} has an empty name"));
                    }
                }
                let first_token = chunk.is_token_delta();
                let step_id = {
                    let request = self.started_request_mut(request_id)?;
                    if first_token && request.first_token.is_none() {
                        request.first_token = Some(recorded.time.clone());
                    }
                    request.step_id.clone()
                };
                if first_token {
                    let step = self
                        .steps
                        .get_mut(&step_id)
                        .expect("request step is indexed");
                    if step.first_token.is_none() {
                        step.first_token = Some(recorded.time.clone());
                    }
                }
            }
            SessionEvent::AssistantCompleted {
                request_id,
                items,
                response,
            } => {
                if items.is_empty() || response.id.is_empty() || response.model.is_empty() {
                    return invalid(format!(
                        "request {request_id} has an invalid assistant result"
                    ));
                }
                let declared_tool_calls = declared_tool_calls(items)?;
                let has_tool_calls = !declared_tool_calls.is_empty();
                let step_id = {
                    let request = self.started_request_mut(request_id)?;
                    request.terminal = Some(RequestTerminal::Completed);
                    request.declared_tool_calls = declared_tool_calls.clone();
                    request.step_id.clone()
                };
                if has_tool_calls {
                    self.pending_tool_registration = Some((request_id.clone(), 0));
                }
                self.state
                    .append_items(items.clone(), Some(response.to_provider()))
                    .map_err(SessionMachineError::State)?;
                self.clear_current_request(&step_id, request_id)?;
                self.steps
                    .get_mut(&step_id)
                    .expect("request step is indexed")
                    .phase = if has_tool_calls {
                    StepPhase::AwaitingTools
                } else {
                    StepPhase::AssistantCompleted
                };
            }
            SessionEvent::ToolCallRequested {
                request_id,
                call_id,
                parent_call_id,
            } => {
                ensure_non_empty_id("call", call_id)?;
                if self.tools.contains_key(call_id) {
                    return invalid(format!("tool call {call_id} is invalid"));
                }
                let request = self
                    .requests
                    .get(request_id)
                    .ok_or_else(|| invalid_error(format!("unknown request {request_id}")))?;
                if request.terminal != Some(RequestTerminal::Completed)
                    || self.active_step.as_ref() != Some(&request.step_id)
                {
                    return invalid(format!(
                        "tool call {call_id} was not declared by the completed response"
                    ));
                }
                let Some((pending_request, next_index)) = &self.pending_tool_registration else {
                    return invalid(format!(
                        "tool call {call_id} was not registered in response declaration order"
                    ));
                };
                if pending_request != request_id
                    || request.declared_tool_calls.get(*next_index) != Some(call_id)
                {
                    return invalid(format!(
                        "tool call {call_id} was not declared next in response declaration order"
                    ));
                }
                if let Some(parent_call_id) = parent_call_id {
                    let parent = self.tools.get(parent_call_id).ok_or_else(|| {
                        invalid_error(format!("unknown parent tool call {parent_call_id}"))
                    })?;
                    if parent_call_id == call_id || parent.step_id != request.step_id {
                        return invalid(format!("tool call {call_id} has an invalid parent"));
                    }
                    if parent.request_id != *request_id {
                        return invalid(format!(
                            "tool call {call_id} belongs to a different request than its parent"
                        ));
                    }
                }
                let request_step_id = request.step_id.clone();
                let declared_call_count = request.declared_tool_calls.len();
                let ordinal = self.next_tool_ordinal;
                self.tools.insert(
                    call_id.clone(),
                    ToolRecord {
                        request_id: request_id.clone(),
                        step_id: request_step_id,
                        authorization: None,
                        dispatch_intended: None,
                        execution_started: None,
                        execution_finished: None,
                        result_attached: false,
                    },
                );
                self.next_tool_ordinal = self.next_tool_ordinal.saturating_add(1);
                self.open_tool_ordinals.insert(call_id.clone(), ordinal);
                self.open_tools.insert(ordinal, call_id.clone());
                let pending = self
                    .pending_tool_registration
                    .as_mut()
                    .expect("tool call registration is tracked until complete");
                pending.1 += 1;
                if pending.1 == declared_call_count {
                    self.pending_tool_registration = None;
                }
            }
            SessionEvent::ToolAuthorizationResolved { call_id, decision } => {
                let tool = self.open_tool_mut(call_id)?;
                if tool.authorization.is_some() {
                    return invalid(format!("tool call {call_id} was authorized twice"));
                }
                tool.authorization = Some(*decision);
            }
            SessionEvent::ToolDispatchIntended { call_id } => {
                let tool = self.open_tool_mut(call_id)?;
                if !tool
                    .authorization
                    .is_some_and(ToolAuthorizationDecision::permits_execution)
                    || tool.dispatch_intended.is_some()
                    || tool.execution_started.is_some()
                {
                    return invalid(format!("tool call {call_id} cannot be dispatched"));
                }
                tool.dispatch_intended = Some(recorded.time.clone());
            }
            SessionEvent::ToolExecutionStarted { call_id } => {
                let tool = self.open_tool_mut(call_id)?;
                if tool.dispatch_intended.is_none()
                    || tool.execution_started.is_some()
                    || recorded
                        .time
                        .duration_since(
                            tool.dispatch_intended
                                .as_ref()
                                .expect("dispatch presence was checked"),
                        )
                        .is_none()
                {
                    return invalid(format!("tool call {call_id} cannot begin execution"));
                }
                tool.execution_started = Some(recorded.time.clone());
            }
            SessionEvent::ToolExecutionFinished { call_id, outcome } => {
                let tool = self.open_tool_mut(call_id)?;
                let observed_in_order = match outcome {
                    ToolExecutionOutcome::UnknownSideEffects => tool.dispatch_intended.is_some(),
                    ToolExecutionOutcome::Success | ToolExecutionOutcome::Error => tool
                        .execution_started
                        .as_ref()
                        .is_some_and(|started| recorded.time.duration_since(started).is_some()),
                };
                if tool.execution_finished.is_some() || !observed_in_order {
                    return invalid(format!("tool call {call_id} cannot finish execution"));
                }
                tool.execution_finished = Some(*outcome);
            }
            SessionEvent::ToolResultAttached {
                call_id,
                status,
                item,
            } => {
                if self.open_tools.values().next() != Some(call_id) {
                    return invalid(format!(
                        "tool result for {call_id} is not next in declaration order"
                    ));
                }
                validate_tool_output_item(call_id, item)?;
                let step_id = {
                    let tool = self.open_tool_mut(call_id)?;
                    validate_tool_result(tool, *status, call_id)?;
                    tool.step_id.clone()
                };
                self.state
                    .append_items(vec![item.clone()], None)
                    .map_err(SessionMachineError::State)?;
                self.tools
                    .get_mut(call_id)
                    .expect("validated tool is indexed")
                    .result_attached = true;
                let ordinal = self
                    .open_tool_ordinals
                    .remove(call_id)
                    .expect("open tool has an insertion ordinal");
                self.open_tools.remove(&ordinal);
                if self.open_tools.is_empty() {
                    self.steps
                        .get_mut(&step_id)
                        .expect("tool step is indexed")
                        .phase = StepPhase::AssistantCompleted;
                }
            }
            SessionEvent::CompactionStarted {
                compaction_id,
                run_id,
                tokens_before,
                first_kept_id,
            } => {
                ensure_non_empty_id("compaction", compaction_id)?;
                let active_step_is_ready = self.active_step.as_ref().is_none_or(|step_id| {
                    self.steps
                        .get(step_id)
                        .is_some_and(|step| step.phase == StepPhase::AwaitingRequest)
                });
                if self.active_run.as_ref() != Some(run_id)
                    || self.active_compaction.is_some()
                    || self.compactions.contains_key(compaction_id)
                    || self.active_request_id().is_some()
                    || !self.open_tools.is_empty()
                    || !active_step_is_ready
                    || !self.state.has_active_items_id(*first_kept_id)
                {
                    return invalid(format!("compaction {compaction_id} cannot start"));
                }
                self.compactions.insert(
                    compaction_id.clone(),
                    CompactionRecord {
                        run_id: run_id.clone(),
                        tokens_before: *tokens_before,
                        first_kept_id: *first_kept_id,
                        finished: false,
                    },
                );
                self.active_compaction = Some(compaction_id.clone());
            }
            SessionEvent::CompactionFinished {
                compaction_id,
                outcome,
                summary,
                response,
            } => {
                if self.active_compaction.as_ref() != Some(compaction_id) {
                    return invalid(format!("compaction {compaction_id} cannot finish"));
                }
                let compaction = self
                    .compactions
                    .get_mut(compaction_id)
                    .ok_or_else(|| invalid_error(format!("unknown compaction {compaction_id}")))?;
                if compaction.finished {
                    return invalid(format!("compaction {compaction_id} finished twice"));
                }
                if self.active_run.as_ref() != Some(&compaction.run_id) {
                    return invalid(format!("compaction {compaction_id} is outside its run"));
                }
                if *outcome == StepOutcome::Completed {
                    let summary = summary
                        .as_deref()
                        .filter(|summary| !summary.is_empty())
                        .ok_or_else(|| {
                            invalid_error(format!(
                                "completed compaction {compaction_id} has no summary"
                            ))
                        })?;
                    self.state
                        .append_compaction(
                            summary.to_owned(),
                            compaction.first_kept_id,
                            compaction.tokens_before,
                            response.as_ref().map(ResponseInfo::to_provider),
                        )
                        .map_err(SessionMachineError::State)?;
                }
                compaction.finished = true;
                self.active_compaction = None;
            }
        }
        Ok(())
    }

    fn open_request_mut(
        &mut self,
        request_id: &RequestId,
    ) -> Result<&mut RequestRecord, SessionMachineError> {
        let request = self
            .requests
            .get_mut(request_id)
            .ok_or_else(|| invalid_error(format!("unknown request {request_id}")))?;
        if request.terminal.is_some()
            || self.active_step.as_ref() != Some(&request.step_id)
            || self.steps.get(&request.step_id).is_none_or(|step| {
                step.phase != StepPhase::RequestActive
                    || step.current_request.as_ref() != Some(request_id)
            })
        {
            return invalid(format!("request {request_id} is not open"));
        }
        Ok(request)
    }

    fn started_request_mut(
        &mut self,
        request_id: &RequestId,
    ) -> Result<&mut RequestRecord, SessionMachineError> {
        let request = self.open_request_mut(request_id)?;
        if request.started.is_none() {
            return invalid(format!("request {request_id} was not started"));
        }
        Ok(request)
    }

    fn clear_current_request(
        &mut self,
        step_id: &StepId,
        request_id: &RequestId,
    ) -> Result<(), SessionMachineError> {
        let step = self
            .steps
            .get_mut(step_id)
            .ok_or_else(|| invalid_error(format!("unknown step {step_id}")))?;
        if step.current_request.as_ref() != Some(request_id) {
            return invalid(format!("request {request_id} is not current"));
        }
        step.current_request = None;
        Ok(())
    }

    fn open_tool_mut(&mut self, call_id: &CallId) -> Result<&mut ToolRecord, SessionMachineError> {
        let tool = self
            .tools
            .get_mut(call_id)
            .ok_or_else(|| invalid_error(format!("unknown tool call {call_id}")))?;
        if tool.result_attached || self.active_step.as_ref() != Some(&tool.step_id) {
            return invalid(format!("tool call {call_id} is not open"));
        }
        Ok(tool)
    }

    fn active_request_id(&self) -> Option<&RequestId> {
        self.active_step
            .as_ref()
            .and_then(|step_id| self.steps.get(step_id))
            .and_then(|step| step.current_request.as_ref())
    }

    fn validate_transaction_boundary(&self) -> Result<(), SessionMachineError> {
        if let Some((request_id, _)) = &self.pending_tool_registration {
            return invalid(format!(
                "request {request_id} did not register every tool call in its response transaction"
            ));
        }
        if self.active_turn.is_some() && self.active_step.is_none() {
            return invalid("an active turn must own an active step at a transaction boundary");
        }
        if let Some(run_id) = self.active_run.as_ref()
            && self.active_turn.is_none()
        {
            let run = self.runs.get(run_id).expect("active run is indexed");
            if run.turn_count > 0 {
                return invalid(
                    "a run with completed turn work must terminate or start its next turn atomically",
                );
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn semantic_snapshot(&self) -> MachineSnapshot {
        let mut inputs = self
            .inputs
            .iter()
            .map(|(id, record)| (id.clone(), record.attached_step.clone()))
            .collect::<Vec<_>>();
        inputs.sort_by(|left, right| left.0.cmp(&right.0));
        let mut first_tokens = self
            .steps
            .iter()
            .map(|(id, step)| (id.clone(), step.first_token.clone()))
            .collect::<Vec<_>>();
        first_tokens.sort_by(|left, right| left.0.cmp(&right.0));
        let mut runs = self
            .runs
            .iter()
            .map(|(id, run)| (id.clone(), run.terminal))
            .collect::<Vec<_>>();
        runs.sort_by(|left, right| left.0.cmp(&right.0));
        let mut turns = self
            .turns
            .iter()
            .map(|(id, turn)| (id.clone(), turn.terminal))
            .collect::<Vec<_>>();
        turns.sort_by(|left, right| left.0.cmp(&right.0));
        let mut steps = self
            .steps
            .iter()
            .map(|(id, step)| {
                (
                    id.clone(),
                    step.phase,
                    step.terminal,
                    step.current_request.clone(),
                )
            })
            .collect::<Vec<_>>();
        steps.sort_by(|left, right| left.0.cmp(&right.0));
        let mut tools = self
            .tools
            .iter()
            .map(|(id, tool)| ToolSnapshot {
                call_id: id.clone(),
                authorization: tool.authorization,
                dispatch_intended: tool.dispatch_intended.is_some(),
                execution_started: tool.execution_started.is_some(),
                execution_finished: tool.execution_finished,
                result_attached: tool.result_attached,
            })
            .collect::<Vec<_>>();
        tools.sort_by(|left, right| left.call_id.cmp(&right.call_id));
        MachineSnapshot {
            next_seq: self.next_seq,
            entries: self.state.entries().iter().cloned().collect(),
            pending: self.pending_inputs(),
            active_run: self.active_run.clone(),
            active_turn: self.active_turn.clone(),
            active_step: self.active_step.clone(),
            active_compaction: self.active_compaction.clone(),
            inputs,
            first_tokens,
            unresolved_tools: self.unresolved_tool_calls(),
            runs,
            turns,
            steps,
            tools,
            last_request_config: self
                .last_request_config
                .as_ref()
                .map(|config| config.as_ref().clone()),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct MachineSnapshot {
    next_seq: u64,
    entries: Vec<StateEntry>,
    pending: Vec<PendingInput>,
    active_run: Option<RunId>,
    active_turn: Option<TurnId>,
    active_step: Option<StepId>,
    active_compaction: Option<CompactionId>,
    inputs: Vec<(InputId, Option<StepId>)>,
    first_tokens: Vec<(StepId, Option<EventTime>)>,
    unresolved_tools: Vec<CallId>,
    runs: Vec<(RunId, Option<RunOutcome>)>,
    turns: Vec<(TurnId, Option<TurnEndReason>)>,
    steps: Vec<(StepId, StepPhase, Option<StepOutcome>, Option<RequestId>)>,
    tools: Vec<ToolSnapshot>,
    last_request_config: Option<RequestConfig>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct ToolSnapshot {
    call_id: CallId,
    authorization: Option<ToolAuthorizationDecision>,
    dispatch_intended: bool,
    execution_started: bool,
    execution_finished: Option<ToolExecutionOutcome>,
    result_attached: bool,
}

fn ensure_non_empty_id(
    kind: &str,
    id: &(impl AsRef<str> + std::fmt::Display),
) -> Result<(), SessionMachineError> {
    if id.as_ref().is_empty() {
        return invalid(format!("{kind} id must not be empty"));
    }
    Ok(())
}

fn require_active_step(
    active: Option<&StepId>,
    expected: &StepId,
) -> Result<(), SessionMachineError> {
    if active != Some(expected) {
        return invalid(format!("step {expected} is not active"));
    }
    Ok(())
}

fn terminal_error_matches(completed: bool, error: Option<&str>) -> bool {
    if completed {
        error.is_none()
    } else {
        error.is_some_and(|error| !error.trim().is_empty())
    }
}

fn validate_attached_input(
    input_id: &InputId,
    input: &InputRecord,
    items: &[InputItem],
) -> Result<(), SessionMachineError> {
    let matches_submitted_input = matches!(
        items,
        [InputItem::EasyMessage(message)]
            if message.role == Role::User
                && matches!(
                    &message.content,
                    EasyInputContent::Text(text) if text == &input.input
                )
    );
    if matches_submitted_input {
        Ok(())
    } else {
        invalid(format!(
            "input {input_id} attachment does not match its submitted {:?} text",
            input.origin
        ))
    }
}

fn validate_tool_result(
    tool: &ToolRecord,
    status: ToolResultStatus,
    call_id: &CallId,
) -> Result<(), SessionMachineError> {
    let valid = match status {
        ToolResultStatus::Success => tool.execution_finished == Some(ToolExecutionOutcome::Success),
        ToolResultStatus::Error => tool.execution_finished == Some(ToolExecutionOutcome::Error),
        ToolResultStatus::Denied => {
            tool.authorization == Some(ToolAuthorizationDecision::Denied)
                && tool.dispatch_intended.is_none()
                && tool.execution_started.is_none()
        }
        ToolResultStatus::NotFound => {
            tool.authorization == Some(ToolAuthorizationDecision::Unavailable)
                && tool.dispatch_intended.is_none()
                && tool.execution_started.is_none()
        }
        ToolResultStatus::AbortedBeforeDispatch => {
            tool.dispatch_intended.is_none()
                && tool.execution_started.is_none()
                && tool.authorization.is_some_and(|decision| {
                    decision.permits_execution() || decision == ToolAuthorizationDecision::Aborted
                })
        }
        ToolResultStatus::UnknownSideEffects => {
            tool.execution_finished == Some(ToolExecutionOutcome::UnknownSideEffects)
        }
    };
    if !valid {
        return invalid(format!(
            "tool result {status:?} contradicts lifecycle for {call_id}"
        ));
    }
    Ok(())
}

fn declared_tool_calls(items: &[InputItem]) -> Result<Vec<CallId>, SessionMachineError> {
    let mut calls = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        let InputItem::Item(Item::FunctionCall(call)) = item else {
            continue;
        };
        let call_id = CallId::from_raw(call.call_id.clone());
        ensure_non_empty_id("response tool call", &call_id)?;
        if call.name.trim().is_empty() || seen.contains(&call_id) {
            return invalid(format!("response tool call {call_id} is invalid"));
        }
        seen.insert(call_id.clone());
        calls.push(call_id);
    }
    Ok(calls)
}

fn validate_tool_output_item(
    call_id: &CallId,
    item: &InputItem,
) -> Result<(), SessionMachineError> {
    let matches = matches!(
        item,
        InputItem::Item(Item::FunctionCallOutput(output)) if output.call_id == call_id.as_str()
    );
    if !matches {
        return invalid(format!(
            "tool result item does not belong to call {call_id}"
        ));
    }
    Ok(())
}

fn tool_output_item(call_id: &CallId, output: &str) -> InputItem {
    InputItem::from(Item::from(FunctionCallOutputItemParam {
        call_id: call_id.to_string(),
        output: FunctionCallOutput::Text(output.to_owned()),
        id: None,
        status: None,
    }))
}

fn recovery_tool_output(status: ToolResultStatus) -> &'static str {
    match status {
        ToolResultStatus::Denied => "Tool call was denied before execution.",
        ToolResultStatus::NotFound => "Tool was unavailable and was not executed.",
        ToolResultStatus::AbortedBeforeDispatch => {
            "Tool call was interrupted before execution and had no side effects."
        }
        ToolResultStatus::UnknownSideEffects => {
            "Tool execution was interrupted; its side effects are unknown. Do not retry automatically."
        }
        ToolResultStatus::Success | ToolResultStatus::Error => {
            "Tool result was not recorded before recovery."
        }
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, SessionMachineError> {
    Err(invalid_error(message))
}

fn invalid_error(message: impl Into<String>) -> SessionMachineError {
    SessionMachineError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use async_openai::types::responses::{EasyInputMessage, FunctionToolCall, InputItem};
    use proptest::prelude::*;

    use super::*;

    fn time(milliseconds: u64) -> EventTime {
        EventTime {
            wall_time_ms: 1_000 + milliseconds as i64,
            clock_id: "machine-test".into(),
            monotonic_ns: milliseconds.saturating_mul(1_000_000),
        }
    }

    fn draft(tx: &str, milliseconds: u64, event: SessionEvent) -> EventDraft {
        EventDraft {
            tx_id: tx.into(),
            time: time(milliseconds),
            event,
        }
    }

    fn commit(
        machine: &mut SessionMachine,
        all_events: &mut Vec<RecordedEvent>,
        drafts: Vec<EventDraft>,
    ) {
        let batch = machine.plan_batch(drafts).unwrap();
        all_events.extend(batch.events.iter().cloned());
        machine.apply_batch(batch).unwrap();
    }

    fn input_pair(tx: &str, index: usize, offset: u64) -> Vec<EventDraft> {
        let input_id = InputId::from_raw(format!("input-{index}"));
        let input = format!("message-{index}");
        vec![
            draft(
                tx,
                offset,
                SessionEvent::InputSubmitted {
                    input_id: input_id.clone(),
                    input: input.clone(),
                    origin: InputOrigin::Queue,
                },
            ),
            draft(
                tx,
                offset,
                SessionEvent::InputAttached {
                    input_id,
                    step_id: "step".into(),
                    items: vec![InputItem::from(EasyInputMessage::from(input))],
                },
            ),
        ]
    }

    fn lifecycle(tx: &str) -> Vec<EventDraft> {
        vec![
            draft(
                tx,
                0,
                SessionEvent::RunStarted {
                    run_id: "run".into(),
                },
            ),
            draft(
                tx,
                1,
                SessionEvent::TurnStarted {
                    run_id: "run".into(),
                    turn_id: "turn".into(),
                },
            ),
            draft(
                tx,
                2,
                SessionEvent::StepStarted {
                    turn_id: "turn".into(),
                    step_id: "step".into(),
                },
            ),
        ]
    }

    fn request_events(tx: &str, token: Option<AssistantChunk>) -> Vec<EventDraft> {
        let mut events = vec![
            draft(
                tx,
                10,
                SessionEvent::RequestSnapshot {
                    request_id: "request".into(),
                    step_id: "step".into(),
                    reason: RequestHeaderReason::Initial,
                    model: "test-model".into(),
                    instructions: None,
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: None,
                    session_config: SessionConfig::default(),
                },
            ),
            draft(
                tx,
                11,
                SessionEvent::ModelRequestStarted {
                    request_id: "request".into(),
                },
            ),
        ];
        if let Some(chunk) = token {
            events.push(draft(
                tx,
                12,
                SessionEvent::AssistantChunk {
                    request_id: "request".into(),
                    chunk,
                },
            ));
        }
        events.push(draft(
            tx,
            13,
            SessionEvent::AssistantCompleted {
                request_id: "request".into(),
                items: vec![InputItem::from(EasyInputMessage::from("done"))],
                response: ResponseInfo {
                    id: "response".into(),
                    model: "test-model".into(),
                    usage: None,
                },
            },
        ));
        events
    }

    fn request_with_tool_events(tx: &str) -> Vec<EventDraft> {
        let mut events = request_events(tx, None);
        let SessionEvent::AssistantCompleted { items, .. } = &mut events
            .last_mut()
            .expect("request events include the assistant completion")
            .event
        else {
            unreachable!("last request event is the assistant completion");
        };
        items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
            arguments: "{}".into(),
            call_id: "call".into(),
            namespace: None,
            name: "shell".into(),
            id: Some("function-item".into()),
            status: None,
        })));
        events.push(draft(
            tx,
            14,
            SessionEvent::ToolCallRequested {
                request_id: "request".into(),
                call_id: "call".into(),
                parent_call_id: None,
            },
        ));
        events
    }

    fn request_with_two_tool_events(tx: &str) -> Vec<EventDraft> {
        let mut events = request_events(tx, None);
        let SessionEvent::AssistantCompleted { items, .. } = &mut events
            .last_mut()
            .expect("request events include the assistant completion")
            .event
        else {
            unreachable!("last request event is the assistant completion");
        };
        for call_id in ["call-z", "call-a"] {
            items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                arguments: "{}".into(),
                call_id: call_id.into(),
                namespace: None,
                name: "shell".into(),
                id: None,
                status: None,
            })));
        }
        for call_id in ["call-z", "call-a"] {
            events.push(draft(
                tx,
                14,
                SessionEvent::ToolCallRequested {
                    request_id: "request".into(),
                    call_id: call_id.into(),
                    parent_call_id: None,
                },
            ));
        }
        events
    }

    proptest! {
        #[test]
        fn live_application_equals_full_replay(inputs in prop::collection::vec("[a-z]{1,16}", 0..20)) {
            let mut live = SessionMachine::default();
            let mut events = Vec::new();
            commit(&mut live, &mut events, lifecycle("tx-lifecycle"));
            for (index, input) in inputs.iter().enumerate() {
                let input_id = InputId::from_raw(format!("input-{index}"));
                let tx = format!("tx-input-{index}");
                commit(
                    &mut live,
                    &mut events,
                    vec![
                        draft(
                            &tx,
                            3 + index as u64,
                            SessionEvent::InputSubmitted {
                                input_id: input_id.clone(),
                                input: input.clone(),
                                origin: InputOrigin::Queue,
                            },
                        ),
                        draft(
                            &tx,
                            3 + index as u64,
                            SessionEvent::InputAttached {
                                input_id,
                                step_id: "step".into(),
                                items: vec![InputItem::from(EasyInputMessage::from(input.clone()))],
                            },
                        ),
                    ],
                );
            }
            commit(&mut live, &mut events, request_events("tx-request", None));
            commit(
                &mut live,
                &mut events,
                vec![
                    draft(
                        "tx-terminal",
                        20,
                        SessionEvent::StepTerminated {
                            step_id: "step".into(),
                            outcome: StepOutcome::Completed,
                            error: None,
                        },
                    ),
                    draft(
                        "tx-terminal",
                        21,
                        SessionEvent::TurnTerminated {
                            turn_id: "turn".into(),
                            reason: TurnEndReason::Completed,
                        },
                    ),
                    draft(
                        "tx-terminal",
                        22,
                        SessionEvent::RunTerminated {
                            run_id: "run".into(),
                            outcome: RunOutcome::Completed,
                            error: None,
                        },
                    ),
                ],
            );

            let replayed = SessionMachine::from_events(&events).unwrap();
            prop_assert_eq!(live.semantic_snapshot(), replayed.semantic_snapshot());
        }
    }

    #[test]
    fn every_incremental_prefix_equals_full_replay() {
        let mut live = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut live, &mut events, lifecycle("tx-lifecycle"));

        for index in 0..128 {
            let tx = format!("tx-prefix-{index}");
            commit(
                &mut live,
                &mut events,
                input_pair(&tx, index, index as u64 + 10),
            );
            let replayed = SessionMachine::from_events(&events).unwrap();
            assert_eq!(live.semantic_snapshot(), replayed.semantic_snapshot());
        }
    }

    #[test]
    fn planned_batch_is_bound_to_the_exact_machine() {
        let machine = SessionMachine::default();
        let first = machine.plan_batch(lifecycle("tx-first")).unwrap();
        let stale = machine.plan_batch(lifecycle("tx-stale")).unwrap();

        let mut live = machine;
        live.apply_batch(first).unwrap();
        assert!(live.apply_batch(stale).is_err());
    }

    #[test]
    fn planning_cost_does_not_grow_linearly_with_history() {
        const TRANSACTIONS: usize = 10_000;
        const QUARTER: usize = TRANSACTIONS / 4;

        let mut machine = SessionMachine::default();
        let lifecycle = machine.plan_batch(lifecycle("tx-lifecycle")).unwrap();
        machine.apply_batch(lifecycle).unwrap();
        let mut middle = Duration::ZERO;
        let mut late = Duration::ZERO;

        for transaction in 0..TRANSACTIONS {
            let tx = format!("tx-perf-{transaction}");
            let first_input = transaction * 2;
            let mut drafts = input_pair(&tx, first_input, transaction as u64 + 10);
            drafts.extend(input_pair(&tx, first_input + 1, transaction as u64 + 10));
            let started = Instant::now();
            let batch = machine.plan_batch(drafts).unwrap();
            machine.apply_batch(batch).unwrap();
            let elapsed = started.elapsed();
            if (QUARTER..QUARTER * 2).contains(&transaction) {
                middle += elapsed;
            } else if (QUARTER * 3..TRANSACTIONS).contains(&transaction) {
                late += elapsed;
            }
        }

        // A deep clone per plan makes the final quarter roughly 2.3x the second quarter for this
        // monotonically growing history. Persistent collections should stay close to logarithmic;
        // a 2x ceiling leaves ample room for shared-CI scheduling noise while catching O(T²).
        eprintln!("10k transaction planning: middle-quarter={middle:?}, late-quarter={late:?}");
        assert!(
            late <= middle.saturating_mul(2),
            "planning regressed toward linear-per-history cost: middle={middle:?}, late={late:?}"
        );
        assert_eq!(machine.next_seq(), 3 + TRANSACTIONS as u64 * 4);
    }

    #[test]
    fn input_can_only_be_attached_once() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-input",
                    3,
                    SessionEvent::InputSubmitted {
                        input_id: "input".into(),
                        input: "hello".into(),
                        origin: InputOrigin::Queue,
                    },
                ),
                draft(
                    "tx-input",
                    4,
                    SessionEvent::InputAttached {
                        input_id: "input".into(),
                        step_id: "step".into(),
                        items: vec![InputItem::from(EasyInputMessage::from("hello"))],
                    },
                ),
            ],
        );
        let error = machine
            .plan_batch(vec![draft(
                "tx-duplicate",
                5,
                SessionEvent::InputAttached {
                    input_id: "input".into(),
                    step_id: "step".into(),
                    items: vec![InputItem::from(EasyInputMessage::from("hello"))],
                },
            )])
            .unwrap_err();
        assert!(error.to_string().contains("attached twice"));
        assert!(machine.pending_inputs().is_empty());
    }

    #[test]
    fn input_attachment_is_the_submitted_user_text_not_arbitrary_model_state() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        let original = machine.semantic_snapshot();

        let injected_call = InputItem::Item(Item::FunctionCall(FunctionToolCall {
            arguments: "{}".into(),
            call_id: "orphan-call".into(),
            namespace: None,
            name: "shell".into(),
            id: None,
            status: None,
        }));
        let error = machine
            .plan_batch(vec![
                draft(
                    "tx-injected-input",
                    3,
                    SessionEvent::InputSubmitted {
                        input_id: "input-injected".into(),
                        input: "hello".into(),
                        origin: InputOrigin::Queue,
                    },
                ),
                draft(
                    "tx-injected-input",
                    4,
                    SessionEvent::InputAttached {
                        input_id: "input-injected".into(),
                        step_id: "step".into(),
                        items: vec![injected_call],
                    },
                ),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("does not match"));
        assert_eq!(machine.semantic_snapshot(), original);

        let error = machine
            .plan_batch(vec![
                draft(
                    "tx-mismatched-input",
                    3,
                    SessionEvent::InputSubmitted {
                        input_id: "input-mismatched".into(),
                        input: "hello".into(),
                        origin: InputOrigin::Steer,
                    },
                ),
                draft(
                    "tx-mismatched-input",
                    4,
                    SessionEvent::InputAttached {
                        input_id: "input-mismatched".into(),
                        step_id: "step".into(),
                        items: vec![InputItem::from(EasyInputMessage::from("different"))],
                    },
                ),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("does not match"));
        assert_eq!(machine.semantic_snapshot(), original);
    }

    #[test]
    fn a_step_cannot_start_another_request_after_its_assistant_completed() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_with_tool_events("tx-request"),
        );

        let error = machine
            .plan_batch(vec![draft(
                "tx-second-request",
                15,
                SessionEvent::RequestSnapshot {
                    request_id: "request-2".into(),
                    step_id: "step".into(),
                    reason: RequestHeaderReason::Resume,
                    model: "test-model".into(),
                    instructions: None,
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: None,
                    session_config: SessionConfig::default(),
                },
            )])
            .unwrap_err();
        assert!(error.to_string().contains("invalid snapshot"));
        assert_eq!(
            machine.unresolved_tool_calls(),
            [CallId::from("call")],
            "the rejected plan must not disturb the original response"
        );
    }

    #[test]
    fn assistant_completion_and_tool_request_can_share_a_transaction() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));

        let response = request_with_tool_events("tx-response");
        let batch = machine.plan_batch(response).unwrap();
        assert!(matches!(
            (&batch.events[2].event, &batch.events[3].event),
            (
                SessionEvent::AssistantCompleted { .. },
                SessionEvent::ToolCallRequested { .. }
            )
        ));
        machine.apply_batch(batch).unwrap();

        let close_step = draft(
            "tx-close-too-early",
            15,
            SessionEvent::StepTerminated {
                step_id: "step".into(),
                outcome: StepOutcome::Completed,
                error: None,
            },
        );
        assert!(machine.plan_batch(vec![close_step]).is_err());

        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-tool-result",
                    16,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call".into(),
                        decision: ToolAuthorizationDecision::Denied,
                    },
                ),
                draft(
                    "tx-tool-result",
                    17,
                    SessionEvent::ToolResultAttached {
                        call_id: "call".into(),
                        status: ToolResultStatus::Denied,
                        item: tool_output_item(&CallId::from("call"), "denied"),
                    },
                ),
                draft(
                    "tx-tool-result",
                    18,
                    SessionEvent::StepTerminated {
                        step_id: "step".into(),
                        outcome: StepOutcome::Completed,
                        error: None,
                    },
                ),
                draft(
                    "tx-tool-result",
                    19,
                    SessionEvent::TurnTerminated {
                        turn_id: "turn".into(),
                        reason: TurnEndReason::Completed,
                    },
                ),
                draft(
                    "tx-tool-result",
                    20,
                    SessionEvent::RunTerminated {
                        run_id: "run".into(),
                        outcome: RunOutcome::Completed,
                        error: None,
                    },
                ),
            ],
        );
        assert!(machine.unresolved_tool_calls().is_empty());
        assert!(machine.active_step().is_none());
        assert!(machine.active_run().is_none());
    }

    #[test]
    fn tool_registration_must_match_the_canonical_response_transaction() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));

        let mut missing = request_with_tool_events("tx-missing");
        missing.pop();
        let mut mismatched = request_with_tool_events("tx-mismatched");
        let SessionEvent::ToolCallRequested { call_id, .. } = &mut mismatched
            .last_mut()
            .expect("request has tool registration")
            .event
        else {
            unreachable!("last request event registers the tool call");
        };
        *call_id = "different-call".into();

        for (drafts, expected_error) in [
            (missing, "did not register every tool call"),
            (mismatched, "was not declared"),
        ] {
            let error = machine.plan_batch(drafts).unwrap_err();
            assert!(error.to_string().contains(expected_error));
            assert_eq!(machine.next_seq(), 3, "failed plans must be atomic");
        }
    }

    #[test]
    fn tool_registration_and_result_attachment_preserve_response_order() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));

        let mut reversed_registration = request_with_two_tool_events("tx-reversed");
        reversed_registration.swap(3, 4);
        let error = machine.plan_batch(reversed_registration).unwrap_err();
        assert!(error.to_string().contains("declaration order"));
        assert_eq!(machine.next_seq(), 3);

        commit(
            &mut machine,
            &mut events,
            request_with_two_tool_events("tx-response"),
        );
        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-authorizations",
                    15,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call-z".into(),
                        decision: ToolAuthorizationDecision::Denied,
                    },
                ),
                draft(
                    "tx-authorizations",
                    16,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call-a".into(),
                        decision: ToolAuthorizationDecision::Denied,
                    },
                ),
            ],
        );

        let error = machine
            .plan_batch(vec![draft(
                "tx-out-of-order-result",
                17,
                SessionEvent::ToolResultAttached {
                    call_id: "call-a".into(),
                    status: ToolResultStatus::Denied,
                    item: tool_output_item(&CallId::from("call-a"), "denied"),
                },
            )])
            .unwrap_err();
        assert!(error.to_string().contains("declaration order"));
        assert_eq!(
            machine.unresolved_tool_calls(),
            [CallId::from("call-z"), CallId::from("call-a")]
        );

        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-ordered-results",
                    18,
                    SessionEvent::ToolResultAttached {
                        call_id: "call-z".into(),
                        status: ToolResultStatus::Denied,
                        item: tool_output_item(&CallId::from("call-z"), "denied"),
                    },
                ),
                draft(
                    "tx-ordered-results",
                    19,
                    SessionEvent::ToolResultAttached {
                        call_id: "call-a".into(),
                        status: ToolResultStatus::Denied,
                        item: tool_output_item(&CallId::from("call-a"), "denied"),
                    },
                ),
            ],
        );
        assert!(machine.unresolved_tool_calls().is_empty());
    }

    #[test]
    fn unknown_side_effects_require_a_durable_execution_terminal_milestone() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_with_tool_events("tx-request"),
        );
        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-dispatch",
                    15,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call".into(),
                        decision: ToolAuthorizationDecision::Allowed,
                    },
                ),
                draft(
                    "tx-dispatch",
                    16,
                    SessionEvent::ToolDispatchIntended {
                        call_id: "call".into(),
                    },
                ),
            ],
        );

        let error = machine
            .plan_batch(vec![draft(
                "tx-premature-result",
                17,
                SessionEvent::ToolResultAttached {
                    call_id: "call".into(),
                    status: ToolResultStatus::UnknownSideEffects,
                    item: tool_output_item(&CallId::from("call"), "unknown"),
                },
            )])
            .unwrap_err();
        assert!(error.to_string().contains("contradicts lifecycle"));

        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-unknown-terminal",
                    18,
                    SessionEvent::ToolExecutionFinished {
                        call_id: "call".into(),
                        outcome: ToolExecutionOutcome::UnknownSideEffects,
                    },
                ),
                draft(
                    "tx-unknown-terminal",
                    19,
                    SessionEvent::ToolResultAttached {
                        call_id: "call".into(),
                        status: ToolResultStatus::UnknownSideEffects,
                        item: tool_output_item(&CallId::from("call"), "unknown"),
                    },
                ),
            ],
        );
        assert!(machine.unresolved_tool_calls().is_empty());
    }

    #[test]
    fn completed_step_rejects_unresolved_tools() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_with_tool_events("tx-request"),
        );
        commit(
            &mut machine,
            &mut events,
            vec![draft(
                "tx-tool",
                15,
                SessionEvent::ToolAuthorizationResolved {
                    call_id: "call".into(),
                    decision: ToolAuthorizationDecision::NotRequired,
                },
            )],
        );
        let terminal = vec![
            draft(
                "tx-terminal",
                21,
                SessionEvent::StepTerminated {
                    step_id: "step".into(),
                    outcome: StepOutcome::Completed,
                    error: None,
                },
            ),
            draft(
                "tx-terminal",
                22,
                SessionEvent::TurnTerminated {
                    turn_id: "turn".into(),
                    reason: TurnEndReason::Completed,
                },
            ),
            draft(
                "tx-terminal",
                23,
                SessionEvent::RunTerminated {
                    run_id: "run".into(),
                    outcome: RunOutcome::Completed,
                    error: None,
                },
            ),
        ];
        assert!(machine.plan_batch(terminal.clone()).is_err());

        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-result",
                    17,
                    SessionEvent::ToolDispatchIntended {
                        call_id: "call".into(),
                    },
                ),
                draft(
                    "tx-result",
                    18,
                    SessionEvent::ToolExecutionStarted {
                        call_id: "call".into(),
                    },
                ),
                draft(
                    "tx-result",
                    19,
                    SessionEvent::ToolExecutionFinished {
                        call_id: "call".into(),
                        outcome: ToolExecutionOutcome::Success,
                    },
                ),
                draft(
                    "tx-result",
                    20,
                    SessionEvent::ToolResultAttached {
                        call_id: "call".into(),
                        status: ToolResultStatus::Success,
                        item: tool_output_item(&CallId::from("call"), "ok"),
                    },
                ),
            ],
        );
        let terminal = machine.plan_batch(terminal).unwrap();
        machine.apply_batch(terminal).unwrap();
        assert!(machine.unresolved_tool_calls().is_empty());
    }

    #[test]
    fn tool_name_only_chunk_sets_first_token() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_events(
                "tx-request",
                Some(AssistantChunk::ToolCallDelta {
                    call_id: "stream-call".into(),
                    name: Some("shell".into()),
                    arguments_delta: String::new(),
                }),
            ),
        );
        assert_eq!(
            machine
                .step_first_token(&StepId::from("step"))
                .map(|time| time.monotonic_ns),
            Some(12_000_000)
        );
    }

    #[test]
    fn compaction_is_mutually_exclusive_with_step_work() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(&mut machine, &mut events, input_pair("tx-input", 0, 3));
        commit(
            &mut machine,
            &mut events,
            vec![draft(
                "tx-compaction-start",
                4,
                SessionEvent::CompactionStarted {
                    compaction_id: "compaction".into(),
                    run_id: "run".into(),
                    tokens_before: 100,
                    first_kept_id: 1,
                },
            )],
        );

        let error = machine
            .plan_batch(vec![draft(
                "tx-request-during-compaction",
                5,
                SessionEvent::RequestSnapshot {
                    request_id: "request".into(),
                    step_id: "step".into(),
                    reason: RequestHeaderReason::Initial,
                    model: "test-model".into(),
                    instructions: None,
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: None,
                    session_config: SessionConfig::default(),
                },
            )])
            .unwrap_err();
        assert!(error.to_string().contains("invalid snapshot"));

        commit(
            &mut machine,
            &mut events,
            vec![draft(
                "tx-pending-during-compaction",
                6,
                SessionEvent::InputSubmitted {
                    input_id: "pending".into(),
                    input: "queued while compacting".into(),
                    origin: InputOrigin::Queue,
                },
            )],
        );
        let error = machine
            .plan_batch(vec![draft(
                "tx-attach-during-compaction",
                7,
                SessionEvent::InputAttached {
                    input_id: "pending".into(),
                    step_id: "step".into(),
                    items: vec![InputItem::from(EasyInputMessage::from(
                        "queued while compacting",
                    ))],
                },
            )])
            .unwrap_err();
        assert!(error.to_string().contains("current step phase"));
        assert_eq!(machine.pending_inputs().len(), 1);

        commit(
            &mut machine,
            &mut events,
            vec![draft(
                "tx-compaction-finish",
                8,
                SessionEvent::CompactionFinished {
                    compaction_id: "compaction".into(),
                    outcome: StepOutcome::Aborted,
                    summary: None,
                    response: None,
                },
            )],
        );
        assert!(
            machine
                .plan_batch(request_events("tx-request", None))
                .is_ok()
        );
    }

    #[test]
    fn lifecycle_transactions_cannot_leave_a_completed_parent_open() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_events("tx-request", None),
        );
        let step_terminal = draft(
            "tx-incomplete-terminal",
            20,
            SessionEvent::StepTerminated {
                step_id: "step".into(),
                outcome: StepOutcome::Completed,
                error: None,
            },
        );
        let turn_terminal = draft(
            "tx-incomplete-terminal",
            21,
            SessionEvent::TurnTerminated {
                turn_id: "turn".into(),
                reason: TurnEndReason::Completed,
            },
        );

        let error = machine.plan_batch(vec![step_terminal.clone()]).unwrap_err();
        assert!(error.to_string().contains("active turn"));
        let error = machine
            .plan_batch(vec![step_terminal.clone(), turn_terminal])
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("terminate or start its next turn")
        );

        let next_step = draft(
            "tx-incomplete-terminal",
            22,
            SessionEvent::StepStarted {
                turn_id: "turn".into(),
                step_id: "step-2".into(),
            },
        );
        assert!(
            machine.plan_batch(vec![step_terminal, next_step]).is_ok(),
            "the same boundary must accept the runtime's atomic step continuation"
        );
    }

    #[test]
    fn parent_terminal_outcomes_must_match_their_children() {
        let machine = SessionMachine::from_events(
            &SessionMachine::default()
                .plan_batch(lifecycle("tx-lifecycle"))
                .unwrap()
                .events,
        )
        .unwrap();
        let terminal = |turn_reason, run_outcome| {
            vec![
                draft(
                    "tx-terminal",
                    20,
                    SessionEvent::StepTerminated {
                        step_id: "step".into(),
                        outcome: StepOutcome::Failed,
                        error: Some("failed".into()),
                    },
                ),
                draft(
                    "tx-terminal",
                    21,
                    SessionEvent::TurnTerminated {
                        turn_id: "turn".into(),
                        reason: turn_reason,
                    },
                ),
                draft(
                    "tx-terminal",
                    22,
                    SessionEvent::RunTerminated {
                        run_id: "run".into(),
                        outcome: run_outcome,
                        error: (run_outcome != RunOutcome::Completed).then(|| "failed".into()),
                    },
                ),
            ]
        };

        let error = machine
            .plan_batch(vec![
                draft(
                    "tx-continue-failed-step",
                    20,
                    SessionEvent::StepTerminated {
                        step_id: "step".into(),
                        outcome: StepOutcome::Failed,
                        error: Some("failed".into()),
                    },
                ),
                draft(
                    "tx-continue-failed-step",
                    21,
                    SessionEvent::StepStarted {
                        turn_id: "turn".into(),
                        step_id: "step-2".into(),
                    },
                ),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("cannot continue"));

        let mut missing_error = terminal(TurnEndReason::Failed, RunOutcome::Failed);
        let SessionEvent::StepTerminated { error, .. } = &mut missing_error[0].event else {
            unreachable!();
        };
        *error = None;
        let error = machine.plan_batch(missing_error).unwrap_err();
        assert!(error.to_string().contains("inconsistent error"));

        let error = machine
            .plan_batch(vec![
                draft(
                    "tx-continue-failed-turn",
                    20,
                    SessionEvent::StepTerminated {
                        step_id: "step".into(),
                        outcome: StepOutcome::Failed,
                        error: Some("failed".into()),
                    },
                ),
                draft(
                    "tx-continue-failed-turn",
                    21,
                    SessionEvent::TurnTerminated {
                        turn_id: "turn".into(),
                        reason: TurnEndReason::Failed,
                    },
                ),
                draft(
                    "tx-continue-failed-turn",
                    22,
                    SessionEvent::TurnStarted {
                        run_id: "run".into(),
                        turn_id: "turn-2".into(),
                    },
                ),
                draft(
                    "tx-continue-failed-turn",
                    23,
                    SessionEvent::StepStarted {
                        turn_id: "turn-2".into(),
                        step_id: "step-2".into(),
                    },
                ),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("cannot continue"));

        let error = machine
            .plan_batch(terminal(TurnEndReason::Completed, RunOutcome::Completed))
            .unwrap_err();
        assert!(error.to_string().contains("contradicts its steps"));
        let error = machine
            .plan_batch(terminal(TurnEndReason::Failed, RunOutcome::Completed))
            .unwrap_err();
        assert!(error.to_string().contains("contradicts its turns"));
        assert!(
            machine
                .plan_batch(terminal(TurnEndReason::Failed, RunOutcome::Failed))
                .is_ok()
        );
    }

    #[test]
    fn recovery_is_one_batch_and_idempotent_after_apply() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_with_tool_events("tx-request"),
        );
        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-tool",
                    15,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call".into(),
                        decision: ToolAuthorizationDecision::Allowed,
                    },
                ),
                draft(
                    "tx-tool",
                    16,
                    SessionEvent::ToolDispatchIntended {
                        call_id: "call".into(),
                    },
                ),
                draft(
                    "tx-tool",
                    17,
                    SessionEvent::ToolExecutionStarted {
                        call_id: "call".into(),
                    },
                ),
            ],
        );

        let recovery_time = EventTime {
            wall_time_ms: 2_000,
            clock_id: "restarted-process".into(),
            monotonic_ns: 0,
        };
        let recovery = machine
            .plan_recovery("tx-recovery".into(), recovery_time)
            .unwrap()
            .unwrap();
        assert!(recovery.events.len() >= 5);
        assert!(
            recovery
                .events
                .iter()
                .all(|event| event.tx_id == TxId::from("tx-recovery"))
        );
        machine.apply_batch(recovery).unwrap();
        assert!(
            machine
                .plan_recovery("tx-recovery-2".into(), time(31))
                .unwrap()
                .is_none()
        );
        assert!(machine.unresolved_tool_calls().is_empty());
        assert!(machine.active_run().is_none());
    }

    #[test]
    fn recovery_preserves_a_durable_tool_outcome_across_the_finish_result_gap() {
        for (outcome, expected_status) in [
            (ToolExecutionOutcome::Success, ToolResultStatus::Success),
            (ToolExecutionOutcome::Error, ToolResultStatus::Error),
        ] {
            let mut machine = SessionMachine::default();
            let mut events = Vec::new();
            commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
            commit(
                &mut machine,
                &mut events,
                request_with_tool_events("tx-request"),
            );
            commit(
                &mut machine,
                &mut events,
                vec![
                    draft(
                        "tx-tool",
                        15,
                        SessionEvent::ToolAuthorizationResolved {
                            call_id: "call".into(),
                            decision: ToolAuthorizationDecision::Allowed,
                        },
                    ),
                    draft(
                        "tx-tool",
                        16,
                        SessionEvent::ToolDispatchIntended {
                            call_id: "call".into(),
                        },
                    ),
                    draft(
                        "tx-tool",
                        17,
                        SessionEvent::ToolExecutionStarted {
                            call_id: "call".into(),
                        },
                    ),
                    draft(
                        "tx-tool",
                        18,
                        SessionEvent::ToolExecutionFinished {
                            call_id: "call".into(),
                            outcome,
                        },
                    ),
                ],
            );

            let recovery = machine
                .plan_recovery("tx-recovery".into(), time(30))
                .unwrap()
                .unwrap();
            assert!(
                !recovery
                    .events
                    .iter()
                    .any(|event| matches!(event.event, SessionEvent::ToolExecutionFinished { .. })),
                "a recorded {outcome:?} outcome must not be downgraded during recovery"
            );
            assert!(recovery.events.iter().any(|event| matches!(
                &event.event,
                SessionEvent::ToolResultAttached {
                    call_id,
                    status,
                    ..
                } if call_id == &CallId::from("call") && status == &expected_status
            )));

            machine.apply_batch(recovery).unwrap();
            assert!(machine.unresolved_tool_calls().is_empty());
        }
    }

    #[test]
    fn recovery_distinguishes_finished_and_in_flight_parallel_tools() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        commit(
            &mut machine,
            &mut events,
            request_with_two_tool_events("tx-request"),
        );
        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-dispatch",
                    15,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call-z".into(),
                        decision: ToolAuthorizationDecision::Allowed,
                    },
                ),
                draft(
                    "tx-dispatch",
                    15,
                    SessionEvent::ToolAuthorizationResolved {
                        call_id: "call-a".into(),
                        decision: ToolAuthorizationDecision::Allowed,
                    },
                ),
                draft(
                    "tx-dispatch",
                    16,
                    SessionEvent::ToolDispatchIntended {
                        call_id: "call-z".into(),
                    },
                ),
                draft(
                    "tx-dispatch",
                    16,
                    SessionEvent::ToolDispatchIntended {
                        call_id: "call-a".into(),
                    },
                ),
            ],
        );
        commit(
            &mut machine,
            &mut events,
            vec![
                draft(
                    "tx-start-z",
                    17,
                    SessionEvent::ToolExecutionStarted {
                        call_id: "call-z".into(),
                    },
                ),
                draft(
                    "tx-start-z",
                    18,
                    SessionEvent::ToolExecutionFinished {
                        call_id: "call-z".into(),
                        outcome: ToolExecutionOutcome::Success,
                    },
                ),
            ],
        );
        commit(
            &mut machine,
            &mut events,
            vec![draft(
                "tx-start-a",
                17,
                SessionEvent::ToolExecutionStarted {
                    call_id: "call-a".into(),
                },
            )],
        );

        let recovery = machine
            .plan_recovery("tx-recovery".into(), time(30))
            .unwrap()
            .unwrap();
        let synthesized_finishes = recovery
            .events
            .iter()
            .filter_map(|event| match &event.event {
                SessionEvent::ToolExecutionFinished { call_id, outcome } => {
                    Some((call_id.clone(), *outcome))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            synthesized_finishes,
            [(
                CallId::from("call-a"),
                ToolExecutionOutcome::UnknownSideEffects
            )]
        );
        let statuses = recovery
            .events
            .iter()
            .filter_map(|event| match &event.event {
                SessionEvent::ToolResultAttached {
                    call_id, status, ..
                } => Some((call_id.clone(), *status)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            statuses,
            [
                (CallId::from("call-z"), ToolResultStatus::Success),
                (CallId::from("call-a"), ToolResultStatus::UnknownSideEffects),
            ]
        );

        machine.apply_batch(recovery).unwrap();
        assert!(machine.unresolved_tool_calls().is_empty());
    }

    #[test]
    fn recovery_preserves_tool_call_insertion_order() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));

        let mut response = request_events("tx-response", None);
        {
            let SessionEvent::AssistantCompleted { items, .. } =
                &mut response.last_mut().expect("request has completion").event
            else {
                unreachable!();
            };
            for call_id in ["call-z", "call-a"] {
                items.push(InputItem::Item(Item::FunctionCall(FunctionToolCall {
                    arguments: "{}".into(),
                    call_id: call_id.into(),
                    namespace: None,
                    name: "shell".into(),
                    id: None,
                    status: None,
                })));
            }
        }
        for call_id in ["call-z", "call-a"] {
            response.push(draft(
                "tx-response",
                14,
                SessionEvent::ToolCallRequested {
                    request_id: "request".into(),
                    call_id: call_id.into(),
                    parent_call_id: None,
                },
            ));
        }
        commit(&mut machine, &mut events, response);

        assert_eq!(
            machine.unresolved_tool_calls(),
            [CallId::from("call-z"), CallId::from("call-a")]
        );
        let recovery = machine
            .plan_recovery("tx-recovery".into(), time(30))
            .unwrap()
            .unwrap();
        let recovered_order = recovery
            .events
            .iter()
            .filter_map(|event| match &event.event {
                SessionEvent::ToolAuthorizationResolved { call_id, .. } => Some(call_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            recovered_order,
            [CallId::from("call-z"), CallId::from("call-a")]
        );
    }

    #[test]
    fn recovery_explicitly_fails_an_interrupted_model_request() {
        let mut machine = SessionMachine::default();
        let mut events = Vec::new();
        commit(&mut machine, &mut events, lifecycle("tx-lifecycle"));
        let mut request = request_events("tx-request", None);
        request.truncate(2);
        commit(&mut machine, &mut events, request);

        let recovery = machine
            .plan_recovery("tx-recovery".into(), time(30))
            .unwrap()
            .unwrap();
        assert!(matches!(
            (&recovery.events[0].event, &recovery.events[1].event),
            (
                SessionEvent::ModelRequestFailed { request_id, error },
                SessionEvent::StepTerminated { .. }
            ) if request_id == &RequestId::from("request") && error.contains("interrupted")
        ));

        machine.apply_batch(recovery).unwrap();
        assert!(
            machine
                .plan_recovery("tx-recovery-2".into(), time(31))
                .unwrap()
                .is_none()
        );
    }
}
