use std::collections::{HashMap, HashSet};
use std::fmt;

use kcastle_agent::{
    AssistantChunk, CallId, CompactionId, EventTime, InputId, InputOrigin, RecordedEvent,
    RequestHeaderReason, RequestId, ResponseInfo, RunId, RunOutcome, SessionEvent, StepId,
    StepOutcome, TokenUsage, ToolAuthorizationDecision, ToolExecutionOutcome, ToolResultStatus,
    TurnEndReason, TurnId,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SessionDocument {
    cursor: EventCursor,
    events: Vec<RecordedEvent>,
    graph: ExecutionGraph,
    conversation_order: Vec<ConversationItemId>,
    trajectory_order: Vec<TrajectoryItemId>,
    display_ordinals: DisplayOrdinals,
    stats: SessionStats,
    revisions: ProjectionRevisions,
}

/// Stable display labels assigned when an entity first enters trajectory order.
///
/// Keeping this index beside the canonical order makes ordinal lookup O(1) for
/// every projection. Streaming text receipts never walk historical trajectory
/// rows, while conversation and trajectory necessarily share the same labels.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DisplayOrdinals {
    turns: HashMap<TurnId, u32>,
    steps: HashMap<(TurnId, StepId), u32>,
    next_turn: u32,
    next_steps: HashMap<TurnId, u32>,
    #[cfg(test)]
    observations: usize,
}

impl DisplayOrdinals {
    fn observe(&mut self, turn_id: Option<TurnId>, step_id: Option<StepId>) {
        #[cfg(test)]
        {
            self.observations = self.observations.saturating_add(1);
        }
        let Some(turn_id) = turn_id else {
            return;
        };
        if !self.turns.contains_key(&turn_id) {
            let ordinal = self.next_turn.max(1);
            self.turns.insert(turn_id.clone(), ordinal);
            self.next_turn = ordinal.saturating_add(1);
        }
        let Some(step_id) = step_id else {
            return;
        };
        let key = (turn_id.clone(), step_id);
        if !self.steps.contains_key(&key) {
            let next = self.next_steps.entry(turn_id).or_insert(1);
            self.steps.insert(key, *next);
            *next = next.saturating_add(1);
        }
    }

    pub(crate) fn turn(&self, turn_id: &TurnId) -> Option<u32> {
        self.turns.get(turn_id).copied()
    }

    pub(crate) fn step(&self, turn_id: &TurnId, step_id: &StepId) -> Option<u32> {
        self.steps.get(&(turn_id.clone(), step_id.clone())).copied()
    }

    #[cfg(test)]
    pub(crate) fn observations(&self) -> usize {
        self.observations
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EventCursor {
    pub(crate) next_seq: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ProjectionRevisions {
    pub(crate) document: u64,
    pub(crate) conversation: u64,
    pub(crate) trajectory: u64,
    pub(crate) geometry: u64,
    pub(crate) stats: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExecutionGraph {
    runs: HashMap<RunId, RunNode>,
    turns: HashMap<TurnId, TurnNode>,
    steps: HashMap<StepId, StepNode>,
    inputs: HashMap<InputId, InputNode>,
    requests: HashMap<RequestId, ResponseNode>,
    tools: HashMap<CallId, ToolNode>,
    compactions: HashMap<CompactionId, CompactionNode>,
    prompts: HashMap<u64, PromptNode>,
    failures: HashMap<StepId, RequestFailureNode>,
    last_prompt: Option<PromptSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RunNode {
    id: RunId,
    started: EventTime,
    completed: Option<EventTime>,
    outcome: Option<RunOutcome>,
    error: Option<String>,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TurnNode {
    id: TurnId,
    run_id: RunId,
    started: EventTime,
    completed: Option<EventTime>,
    reason: Option<TurnEndReason>,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StepNode {
    id: StepId,
    turn_id: TurnId,
    started: EventTime,
    completed: Option<EventTime>,
    outcome: Option<StepOutcome>,
    error: Option<String>,
    request_ids: Vec<RequestId>,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputState {
    Submitted,
    Attached,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct InputNode {
    id: InputId,
    text: String,
    origin: InputOrigin,
    step_id: Option<StepId>,
    state: InputState,
    attached_payload: String,
    timing: TimingMetrics,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResponseChannel {
    Reasoning,
    Output,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResponseSegment {
    ordinal: u32,
    channel: ResponseChannel,
    text: String,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolCallMetadata {
    name: String,
    arguments: String,
    first_observed: Option<EventTime>,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResponseNode {
    request_id: RequestId,
    step_id: StepId,
    model: String,
    segments: Vec<ResponseSegment>,
    active_segment: Option<usize>,
    combined_text: String,
    tool_payload: String,
    tool_calls: HashMap<CallId, ToolCallMetadata>,
    tool_call_order: Vec<CallId>,
    completed_items_json: String,
    response: Option<ResponseInfo>,
    status: ItemStatus,
    visible: bool,
    timing: TimingMetrics,
    usage: Option<TokenUsage>,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToolNode {
    call_id: CallId,
    request_id: RequestId,
    parent_call_id: Option<CallId>,
    name: String,
    arguments: String,
    output: String,
    authorization: Option<ToolAuthorizationDecision>,
    execution_outcome: Option<ToolExecutionOutcome>,
    item_json: String,
    status: ItemStatus,
    timing: TimingMetrics,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompactionNode {
    id: CompactionId,
    run_id: RunId,
    tokens_before: usize,
    first_kept_id: u64,
    summary: String,
    response: Option<ResponseInfo>,
    outcome: Option<StepOutcome>,
    status: ItemStatus,
    timing: TimingMetrics,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PromptSnapshot {
    instructions: String,
    tools_json: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PromptChangeKind {
    Initial,
    System,
    Tools,
    SystemAndTools,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PromptNode {
    step_id: StepId,
    kind: PromptChangeKind,
    summary: String,
    instructions: String,
    timing: TimingMetrics,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestFailureNode {
    step_id: StepId,
    request_id: Option<RequestId>,
    error: String,
    timing: TimingMetrics,
    source_seqs: Vec<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TimingMetrics {
    pub(crate) started: Option<EventTimeRef>,
    pub(crate) first_token: Option<EventTimeRef>,
    pub(crate) requested: Option<EventTimeRef>,
    pub(crate) authorization_resolved: Option<EventTimeRef>,
    pub(crate) dispatch_intended: Option<EventTimeRef>,
    pub(crate) execution_started: Option<EventTimeRef>,
    pub(crate) execution_finished: Option<EventTimeRef>,
    pub(crate) completed: Option<EventTimeRef>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EventTimeRef(EventTime);

impl EventTimeRef {
    pub(crate) fn wall_time_ms(&self) -> i64 {
        self.0.wall_time_ms
    }

    pub(crate) fn monotonic_ns(&self) -> u64 {
        self.0.monotonic_ns
    }

    pub(crate) fn clock_id(&self) -> &str {
        &self.0.clock_id
    }
}

impl From<&EventTime> for EventTimeRef {
    fn from(value: &EventTime) -> Self {
        Self(value.clone())
    }
}

impl TimingMetrics {
    pub(crate) fn duration_ns(&self) -> Option<u64> {
        duration_between(self.completed.as_ref(), self.started.as_ref())
    }

    pub(crate) fn ttft_ns(&self) -> Option<u64> {
        duration_between(self.first_token.as_ref(), self.started.as_ref())
    }

    pub(crate) fn decode_ns(&self) -> Option<u64> {
        duration_between(self.completed.as_ref(), self.first_token.as_ref())
    }

    pub(crate) fn execution_ns(&self) -> Option<u64> {
        duration_between(
            self.execution_finished.as_ref(),
            self.execution_started.as_ref(),
        )
    }

    pub(crate) fn request_registration_ns(&self) -> Option<u64> {
        duration_between(self.requested.as_ref(), self.started.as_ref())
    }

    pub(crate) fn authorization_ns(&self) -> Option<u64> {
        duration_between(
            self.authorization_resolved.as_ref(),
            self.requested.as_ref(),
        )
    }

    pub(crate) fn dispatch_ns(&self) -> Option<u64> {
        duration_between(
            self.dispatch_intended.as_ref(),
            self.authorization_resolved.as_ref(),
        )
    }

    pub(crate) fn runner_start_ns(&self) -> Option<u64> {
        duration_between(
            self.execution_started.as_ref(),
            self.dispatch_intended.as_ref(),
        )
    }

    pub(crate) fn pre_execution_ns(&self) -> Option<u64> {
        duration_between(self.execution_started.as_ref(), self.started.as_ref())
    }

    pub(crate) fn post_execution_ns(&self) -> Option<u64> {
        duration_between(self.completed.as_ref(), self.execution_finished.as_ref())
    }
}

fn duration_between(later: Option<&EventTimeRef>, earlier: Option<&EventTimeRef>) -> Option<u64> {
    later?.0.duration_since(&earlier?.0)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SessionStats {
    pub(crate) turns: usize,
    pub(crate) steps: usize,
    pub(crate) llm_ns: u64,
    pub(crate) tool_ns: u64,
    pub(crate) ttft_ns: u64,
    pub(crate) ttft_samples: usize,
    pub(crate) decode_ns: u64,
    pub(crate) decode_tokens: u64,
    pub(crate) uncached_input_tokens: u64,
    pub(crate) cache_read_input_tokens: u64,
    pub(crate) cache_write_input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) reasoning_output_tokens: u64,
}

impl SessionStats {
    pub(crate) fn input_tokens(self) -> u64 {
        self.uncached_input_tokens
            .saturating_add(self.cache_read_input_tokens)
            .saturating_add(self.cache_write_input_tokens)
    }

    pub(crate) fn total_output_tokens(self) -> u64 {
        self.output_tokens
            .saturating_add(self.reasoning_output_tokens)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct StatsContribution {
    llm_ns: u64,
    tool_ns: u64,
    ttft_ns: u64,
    ttft_samples: usize,
    decode_ns: u64,
    decode_tokens: u64,
    usage: TokenUsage,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ConversationItemId {
    Input(InputId),
    ResponseSegment { request_id: RequestId, ordinal: u32 },
    Tool(CallId),
    Compaction(CompactionId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TrajectoryItemId {
    PromptChange(u64),
    Input(InputId),
    Assistant(RequestId),
    Tool(CallId),
    Compaction(CompactionId),
    RequestFailure(StepId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConversationRole {
    User,
    Steering,
    Context,
    Reasoning,
    Assistant,
    Tool,
    Notice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryLane {
    Input,
    Model,
    Tools,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrajectoryKind {
    System,
    User,
    Steering,
    Context,
    Assistant,
    Tool,
    Compaction,
    RequestFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ItemStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Aborted,
    Denied,
    NotExecuted,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConversationItemView<'a> {
    pub(crate) id: &'a ConversationItemId,
    pub(crate) role: ConversationRole,
    pub(crate) title: Option<&'a str>,
    pub(crate) text: &'a str,
    pub(crate) payload: Option<&'a str>,
    pub(crate) status: ItemStatus,
    pub(crate) turn_id: Option<&'a TurnId>,
    pub(crate) step_id: Option<&'a StepId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TrajectoryItemView<'a> {
    pub(crate) id: &'a TrajectoryItemId,
    pub(crate) kind: TrajectoryKind,
    pub(crate) lane: TrajectoryLane,
    pub(crate) title: &'a str,
    pub(crate) text: &'a str,
    pub(crate) payload: Option<&'a str>,
    pub(crate) status: ItemStatus,
    pub(crate) timing: &'a TimingMetrics,
    pub(crate) usage: Option<TokenUsage>,
    pub(crate) turn_id: Option<&'a TurnId>,
    pub(crate) step_id: Option<&'a StepId>,
    pub(crate) source_seqs: &'a [u64],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DetailsView<'a> {
    pub(crate) id: &'a TrajectoryItemId,
    pub(crate) kind: TrajectoryKind,
    pub(crate) title: &'a str,
    pub(crate) text: &'a str,
    pub(crate) payload: Option<&'a str>,
    pub(crate) status: ItemStatus,
    pub(crate) timing: &'a TimingMetrics,
    pub(crate) usage: Option<TokenUsage>,
    pub(crate) source_seqs: &'a [u64],
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ProjectionDelta {
    pub(crate) first_seq: Option<u64>,
    pub(crate) last_seq: Option<u64>,
    pub(crate) changed_conversation: Vec<ConversationItemId>,
    pub(crate) changed_trajectory: Vec<TrajectoryItemId>,
    pub(crate) conversation_order_changed: bool,
    pub(crate) trajectory_order_changed: bool,
    pub(crate) geometry_changed: bool,
    pub(crate) stats_changed: bool,
    pub(crate) revisions: ProjectionRevisions,
}

#[derive(Clone, Debug)]
pub(crate) struct PlannedBatch {
    expected_next_seq: u64,
    events: Vec<RecordedEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProjectionError {
    Sequence { expected: u64, actual: u64 },
    StalePlan { expected: u64, actual: u64 },
    Duplicate { kind: &'static str, id: String },
    Missing { kind: &'static str, id: String },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sequence { expected, actual } => {
                write!(formatter, "expected event seq {expected}, got {actual}")
            }
            Self::StalePlan { expected, actual } => write!(
                formatter,
                "planned batch starts at seq {expected}, document is at {actual}"
            ),
            Self::Duplicate { kind, id } => write!(formatter, "duplicate {kind} {id}"),
            Self::Missing { kind, id } => write!(formatter, "missing {kind} {id}"),
        }
    }
}

impl std::error::Error for ProjectionError {}

#[derive(Default)]
struct EventChanges {
    conversation: Vec<ConversationItemId>,
    trajectory: Vec<TrajectoryItemId>,
    ordinal_context: Vec<TrajectoryItemId>,
    conversation_order: bool,
    trajectory_order: bool,
    geometry: bool,
    stats: bool,
}

impl SessionDocument {
    pub(crate) fn from_events(events: Vec<RecordedEvent>) -> Result<Self, ProjectionError> {
        let mut document = Self::default();
        document.apply_batch(events)?;
        Ok(document)
    }

    #[cfg(test)]
    pub(crate) fn cursor(&self) -> EventCursor {
        self.cursor
    }

    #[cfg(test)]
    pub(crate) fn graph(&self) -> &ExecutionGraph {
        &self.graph
    }

    pub(crate) fn stats(&self) -> SessionStats {
        self.stats
    }

    pub(crate) fn revisions(&self) -> ProjectionRevisions {
        self.revisions
    }

    pub(crate) fn display_ordinals(&self) -> &DisplayOrdinals {
        &self.display_ordinals
    }

    pub(crate) fn plan_batch(
        &self,
        events: Vec<RecordedEvent>,
    ) -> Result<PlannedBatch, ProjectionError> {
        let mut overlay = PlanOverlay::default();
        let mut expected = self.cursor.next_seq;
        for recorded in &events {
            if recorded.seq != expected {
                return Err(ProjectionError::Sequence {
                    expected,
                    actual: recorded.seq,
                });
            }
            self.validate_event(&recorded.event, &mut overlay)?;
            expected = expected.saturating_add(1);
        }
        Ok(PlannedBatch {
            expected_next_seq: self.cursor.next_seq,
            events,
        })
    }

    pub(crate) fn apply_planned(
        &mut self,
        plan: PlannedBatch,
    ) -> Result<ProjectionDelta, ProjectionError> {
        if self.cursor.next_seq != plan.expected_next_seq {
            return Err(ProjectionError::StalePlan {
                expected: plan.expected_next_seq,
                actual: self.cursor.next_seq,
            });
        }
        let mut delta = ProjectionDelta::default();
        let mut changed_conversation = HashSet::new();
        let mut changed_trajectory = HashSet::new();
        for recorded in plan.events {
            delta.first_seq.get_or_insert(recorded.seq);
            delta.last_seq = Some(recorded.seq);
            self.apply_recorded(
                &recorded,
                &mut delta,
                &mut changed_conversation,
                &mut changed_trajectory,
            );
            self.events.push(recorded);
            self.cursor.next_seq = self.cursor.next_seq.saturating_add(1);
            self.revisions.document = self.revisions.document.saturating_add(1);
        }
        delta.revisions = self.revisions;
        Ok(delta)
    }

    pub(crate) fn apply_batch(
        &mut self,
        events: Vec<RecordedEvent>,
    ) -> Result<ProjectionDelta, ProjectionError> {
        let plan = self.plan_batch(events)?;
        self.apply_planned(plan)
    }

    pub(crate) fn conversation(&self) -> Vec<ConversationItemView<'_>> {
        self.conversation_order
            .iter()
            .filter_map(|id| self.conversation_item(id))
            .collect()
    }

    pub(crate) fn conversation_ids(&self) -> &[ConversationItemId] {
        &self.conversation_order
    }

    pub(crate) fn conversation_by_id<'a>(
        &'a self,
        id: &'a ConversationItemId,
    ) -> Option<ConversationItemView<'a>> {
        self.conversation_item(id)
    }

    pub(crate) fn trajectory(&self) -> Vec<TrajectoryItemView<'_>> {
        self.trajectory_order
            .iter()
            .filter_map(|id| self.trajectory_item(id))
            .collect()
    }

    pub(crate) fn trajectory_ids(&self) -> &[TrajectoryItemId] {
        &self.trajectory_order
    }

    pub(crate) fn trajectory_by_id<'a>(
        &'a self,
        id: &'a TrajectoryItemId,
    ) -> Option<TrajectoryItemView<'a>> {
        self.trajectory_item(id)
    }

    pub(crate) fn details<'a>(&'a self, id: &'a TrajectoryItemId) -> Option<DetailsView<'a>> {
        let item = self.trajectory_item(id)?;
        Some(DetailsView {
            id: item.id,
            kind: item.kind,
            title: item.title,
            text: item.text,
            payload: item.payload,
            status: item.status,
            timing: item.timing,
            usage: item.usage,
            source_seqs: item.source_seqs,
        })
    }

    pub(crate) fn details_raw(&self, id: &TrajectoryItemId) -> Option<String> {
        let source_seqs = self.details(id)?.source_seqs;
        let mut raw = Vec::with_capacity(source_seqs.len());
        for seq in source_seqs {
            let event = usize::try_from(*seq)
                .ok()
                .and_then(|index| self.events.get(index))
                .filter(|event| event.seq == *seq)
                .or_else(|| self.events.iter().find(|event| event.seq == *seq))?;
            raw.push(serde_json::to_string_pretty(event).ok()?);
        }
        Some(raw.join("\n"))
    }

    fn apply_recorded(
        &mut self,
        recorded: &RecordedEvent,
        delta: &mut ProjectionDelta,
        changed_conversation: &mut HashSet<ConversationItemId>,
        changed_trajectory: &mut HashSet<TrajectoryItemId>,
    ) {
        let stats_before = self.stats;
        let mut changes = EventChanges::default();
        self.apply_event(recorded, &mut changes);
        let mut ordinal_ids = changes.ordinal_context.clone();
        if changes.trajectory_order {
            ordinal_ids.extend(changes.trajectory.iter().cloned());
        }
        if !ordinal_ids.is_empty() {
            // Every order insertion also reports its stable ID as changed. An
            // already-known turn/step is an O(1) no-op; a newly visible entity
            // assigns at most one turn and one step ordinal.
            let contexts = ordinal_ids
                .iter()
                .filter_map(|id| {
                    self.trajectory_item(id)
                        .map(|item| (item.turn_id.cloned(), item.step_id.cloned()))
                })
                .collect::<Vec<_>>();
            for (turn_id, step_id) in contexts {
                self.display_ordinals.observe(turn_id, step_id);
            }
        }
        changes.stats |= self.stats != stats_before;

        if !changes.conversation.is_empty() || changes.conversation_order {
            self.revisions.conversation = self.revisions.conversation.saturating_add(1);
        }
        if !changes.trajectory.is_empty() || changes.trajectory_order {
            self.revisions.trajectory = self.revisions.trajectory.saturating_add(1);
        }
        if changes.geometry {
            self.revisions.geometry = self.revisions.geometry.saturating_add(1);
        }
        if changes.stats {
            self.revisions.stats = self.revisions.stats.saturating_add(1);
        }

        for id in changes.conversation {
            if changed_conversation.insert(id.clone()) {
                delta.changed_conversation.push(id);
            }
        }
        for id in changes.trajectory {
            if changed_trajectory.insert(id.clone()) {
                delta.changed_trajectory.push(id);
            }
        }
        delta.conversation_order_changed |= changes.conversation_order;
        delta.trajectory_order_changed |= changes.trajectory_order;
        delta.geometry_changed |= changes.geometry;
        delta.stats_changed |= changes.stats;
    }

    fn apply_event(&mut self, recorded: &RecordedEvent, changes: &mut EventChanges) {
        let seq = recorded.seq;
        let time = &recorded.time;
        match &recorded.event {
            SessionEvent::RunStarted { run_id } => {
                self.graph.runs.insert(
                    run_id.clone(),
                    RunNode {
                        id: run_id.clone(),
                        started: time.clone(),
                        completed: None,
                        outcome: None,
                        error: None,
                        source_seqs: vec![seq],
                    },
                );
            }
            SessionEvent::RunTerminated {
                run_id,
                outcome,
                error,
            } => {
                let run = self
                    .graph
                    .runs
                    .get_mut(run_id)
                    .expect("run termination was validated");
                run.completed.get_or_insert_with(|| time.clone());
                run.outcome.get_or_insert(*outcome);
                if run.error.is_none() {
                    run.error.clone_from(error);
                }
                run.source_seqs.push(seq);
            }
            SessionEvent::TurnStarted { run_id, turn_id } => {
                self.graph.turns.insert(
                    turn_id.clone(),
                    TurnNode {
                        id: turn_id.clone(),
                        run_id: run_id.clone(),
                        started: time.clone(),
                        completed: None,
                        reason: None,
                        source_seqs: vec![seq],
                    },
                );
            }
            SessionEvent::TurnTerminated { turn_id, reason } => {
                let turn = self
                    .graph
                    .turns
                    .get_mut(turn_id)
                    .expect("turn termination was validated");
                if turn.completed.is_none() {
                    turn.completed = Some(time.clone());
                    turn.reason = Some(*reason);
                    self.stats.turns = self.stats.turns.saturating_add(1);
                }
                turn.source_seqs.push(seq);
            }
            SessionEvent::StepStarted { turn_id, step_id } => {
                self.graph.steps.insert(
                    step_id.clone(),
                    StepNode {
                        id: step_id.clone(),
                        turn_id: turn_id.clone(),
                        started: time.clone(),
                        completed: None,
                        outcome: None,
                        error: None,
                        request_ids: Vec::new(),
                        source_seqs: vec![seq],
                    },
                );
            }
            SessionEvent::StepTerminated {
                step_id,
                outcome,
                error,
            } => {
                let request_ids = {
                    let step = self
                        .graph
                        .steps
                        .get_mut(step_id)
                        .expect("step termination was validated");
                    if step.completed.is_none() {
                        step.completed = Some(time.clone());
                        step.outcome = Some(*outcome);
                        step.error.clone_from(error);
                        self.stats.steps = self.stats.steps.saturating_add(1);
                    }
                    step.source_seqs.push(seq);
                    step.request_ids.clone()
                };

                let terminal_status = status_from_step_outcome(*outcome);
                let mut has_visible_response = false;
                for request_id in request_ids {
                    let response = self
                        .graph
                        .requests
                        .get_mut(&request_id)
                        .expect("step request is indexed");
                    has_visible_response |= response.visible;
                    if response.visible && response.status != ItemStatus::Completed {
                        response.status = terminal_status;
                        response.source_seqs.push(seq);
                        changes
                            .trajectory
                            .push(TrajectoryItemId::Assistant(request_id.clone()));
                        for segment in &response.segments {
                            changes
                                .conversation
                                .push(ConversationItemId::ResponseSegment {
                                    request_id: request_id.clone(),
                                    ordinal: segment.ordinal,
                                });
                        }
                    }
                }
                if !has_visible_response && !matches!(outcome, StepOutcome::Completed) {
                    let failure_id = TrajectoryItemId::RequestFailure(step_id.clone());
                    if let Some(failure) = self.graph.failures.get_mut(step_id) {
                        failure.source_seqs.push(seq);
                        changes.trajectory.push(failure_id);
                    } else {
                        self.graph.failures.insert(
                            step_id.clone(),
                            RequestFailureNode {
                                step_id: step_id.clone(),
                                request_id: None,
                                error: error
                                    .clone()
                                    .unwrap_or_else(|| "model request did not complete".to_owned()),
                                timing: TimingMetrics {
                                    started: self
                                        .graph
                                        .steps
                                        .get(step_id)
                                        .map(|step| EventTimeRef::from(&step.started)),
                                    completed: Some(EventTimeRef::from(time)),
                                    ..TimingMetrics::default()
                                },
                                source_seqs: vec![seq],
                            },
                        );
                        self.trajectory_order.push(failure_id.clone());
                        changes.trajectory.push(failure_id);
                        changes.trajectory_order = true;
                        changes.geometry = true;
                    }
                }
            }
            SessionEvent::InputSubmitted {
                input_id,
                input,
                origin,
            } => {
                let timing = point_timing(time);
                self.graph.inputs.insert(
                    input_id.clone(),
                    InputNode {
                        id: input_id.clone(),
                        text: input.clone(),
                        origin: *origin,
                        step_id: None,
                        state: InputState::Submitted,
                        attached_payload: String::new(),
                        timing,
                        source_seqs: vec![seq],
                    },
                );
                let conversation_id = ConversationItemId::Input(input_id.clone());
                let trajectory_id = TrajectoryItemId::Input(input_id.clone());
                self.conversation_order.push(conversation_id.clone());
                self.trajectory_order.push(trajectory_id.clone());
                changes.conversation.push(conversation_id);
                changes.trajectory.push(trajectory_id);
                changes.conversation_order = true;
                changes.trajectory_order = true;
                changes.geometry = true;
            }
            SessionEvent::InputAttached {
                input_id,
                step_id,
                items,
            } => {
                let input = self
                    .graph
                    .inputs
                    .get_mut(input_id)
                    .expect("input attachment was validated");
                input.step_id = Some(step_id.clone());
                input.state = InputState::Attached;
                input.attached_payload = serde_json::to_string(items).unwrap_or_default();
                input.source_seqs.push(seq);
                changes
                    .conversation
                    .push(ConversationItemId::Input(input_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Input(input_id.clone()));
                changes
                    .ordinal_context
                    .push(TrajectoryItemId::Input(input_id.clone()));
            }
            SessionEvent::RequestSnapshot {
                request_id,
                step_id,
                reason,
                model,
                instructions,
                tools,
                ..
            } => {
                self.graph.requests.insert(
                    request_id.clone(),
                    ResponseNode {
                        request_id: request_id.clone(),
                        step_id: step_id.clone(),
                        model: model.clone(),
                        segments: Vec::new(),
                        active_segment: None,
                        combined_text: String::new(),
                        tool_payload: String::new(),
                        tool_calls: HashMap::new(),
                        tool_call_order: Vec::new(),
                        completed_items_json: String::new(),
                        response: None,
                        status: ItemStatus::Pending,
                        visible: false,
                        timing: TimingMetrics::default(),
                        usage: None,
                        source_seqs: vec![seq],
                    },
                );
                self.graph
                    .steps
                    .get_mut(step_id)
                    .expect("request step was validated")
                    .request_ids
                    .push(request_id.clone());

                let current_prompt = PromptSnapshot {
                    instructions: instructions.clone().unwrap_or_default(),
                    tools_json: serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_owned()),
                };
                let prompt_kind =
                    prompt_change_kind(self.graph.last_prompt.as_ref(), &current_prompt, *reason);
                self.graph.last_prompt = Some(current_prompt.clone());
                if let Some(kind) = prompt_kind {
                    let id = TrajectoryItemId::PromptChange(seq);
                    self.graph.prompts.insert(
                        seq,
                        PromptNode {
                            step_id: step_id.clone(),
                            kind,
                            summary: format!("{model} · {} tools", tools.len()),
                            instructions: current_prompt.instructions,
                            timing: point_timing(time),
                            source_seqs: vec![seq],
                        },
                    );
                    if kind == PromptChangeKind::Initial {
                        self.trajectory_order.insert(0, id.clone());
                    } else {
                        self.trajectory_order.push(id.clone());
                    }
                    changes.trajectory.push(id);
                    changes.trajectory_order = true;
                    changes.geometry = true;
                }
            }
            SessionEvent::ModelRequestStarted { request_id } => {
                let response = self
                    .graph
                    .requests
                    .get_mut(request_id)
                    .expect("model request start was validated");
                response.status = ItemStatus::Running;
                // The step may begin before automatic compaction and request
                // construction. LLM duration and TTFT start only when the
                // provider request itself starts, otherwise that preparation
                // time is double-counted as both compaction and model time.
                response.timing.started = Some(EventTimeRef::from(time));
                response.source_seqs.push(seq);
            }
            SessionEvent::ModelRequestFailed { request_id, error } => {
                let before = response_contribution(
                    self.graph
                        .requests
                        .get(request_id)
                        .expect("model request failure was validated"),
                );
                let (step_id, was_visible, segment_ids, timing) = {
                    let response = self
                        .graph
                        .requests
                        .get_mut(request_id)
                        .expect("model request failure was validated");
                    response.status = ItemStatus::Failed;
                    response.source_seqs.push(seq);
                    (
                        response.step_id.clone(),
                        response.visible,
                        response
                            .segments
                            .iter()
                            .map(|segment| segment.ordinal)
                            .collect::<Vec<_>>(),
                        TimingMetrics {
                            started: response.timing.started.clone(),
                            completed: Some(EventTimeRef::from(time)),
                            ..TimingMetrics::default()
                        },
                    )
                };
                let after = response_contribution(
                    self.graph
                        .requests
                        .get(request_id)
                        .expect("model request failure was validated"),
                );
                self.stats.replace_contribution(before, after);

                if was_visible {
                    changes
                        .trajectory
                        .push(TrajectoryItemId::Assistant(request_id.clone()));
                    for ordinal in segment_ids {
                        changes
                            .conversation
                            .push(ConversationItemId::ResponseSegment {
                                request_id: request_id.clone(),
                                ordinal,
                            });
                    }
                } else if !self.graph.failures.contains_key(&step_id) {
                    let id = TrajectoryItemId::RequestFailure(step_id.clone());
                    self.graph.failures.insert(
                        step_id.clone(),
                        RequestFailureNode {
                            step_id,
                            request_id: Some(request_id.clone()),
                            error: error.clone(),
                            timing,
                            source_seqs: vec![seq],
                        },
                    );
                    self.trajectory_order.push(id.clone());
                    changes.trajectory.push(id);
                    changes.trajectory_order = true;
                    changes.geometry = true;
                }
            }
            SessionEvent::AssistantChunk { request_id, chunk } => {
                let before = response_contribution(
                    self.graph
                        .requests
                        .get(request_id)
                        .expect("assistant chunk was validated"),
                );
                let mut conversation_change = None;
                let (became_visible, first_token_changed, payload_segment_ids) = {
                    let response = self
                        .graph
                        .requests
                        .get_mut(request_id)
                        .expect("assistant chunk was validated");
                    let was_visible = response.visible;
                    let had_first_token = response.timing.first_token.is_some();
                    response.source_seqs.push(seq);
                    response.status = ItemStatus::Running;
                    if chunk.is_token_delta() && !had_first_token {
                        response.timing.first_token = Some(EventTimeRef::from(time));
                    }
                    match chunk {
                        AssistantChunk::OutputTextDelta { delta } => {
                            let (ordinal, created) =
                                append_response_text(response, ResponseChannel::Output, delta, seq);
                            conversation_change = Some((ordinal, created));
                            response.visible |= !delta.is_empty();
                        }
                        AssistantChunk::ReasoningTextDelta { delta } => {
                            let (ordinal, created) = append_response_text(
                                response,
                                ResponseChannel::Reasoning,
                                delta,
                                seq,
                            );
                            conversation_change = Some((ordinal, created));
                            response.visible |= !delta.is_empty();
                        }
                        AssistantChunk::ToolCallDelta {
                            call_id,
                            name,
                            arguments_delta,
                        } => {
                            if !response.tool_calls.contains_key(call_id) {
                                response.tool_call_order.push(call_id.clone());
                            }
                            let call =
                                response
                                    .tool_calls
                                    .entry(call_id.clone())
                                    .or_insert_with(|| ToolCallMetadata {
                                        name: String::new(),
                                        arguments: String::new(),
                                        first_observed: Some(time.clone()),
                                        source_seqs: Vec::new(),
                                    });
                            call.first_observed.get_or_insert_with(|| time.clone());
                            call.source_seqs.push(seq);
                            if let Some(name) = name.as_deref().filter(|name| !name.is_empty()) {
                                call.name = name.to_owned();
                            }
                            call.arguments.push_str(arguments_delta);
                            response.tool_payload = format_tool_payload(
                                &response.tool_calls,
                                &response.tool_call_order,
                            );
                            response.visible |= chunk.is_token_delta();
                        }
                        AssistantChunk::Usage { usage } => response.usage = Some(*usage),
                    }
                    let payload_segment_ids =
                        if matches!(chunk, AssistantChunk::ToolCallDelta { .. }) {
                            response
                                .segments
                                .iter()
                                .map(|segment| segment.ordinal)
                                .collect()
                        } else {
                            Vec::new()
                        };
                    (
                        !was_visible && response.visible,
                        !had_first_token && response.timing.first_token.is_some(),
                        payload_segment_ids,
                    )
                };
                let after = response_contribution(
                    self.graph
                        .requests
                        .get(request_id)
                        .expect("assistant chunk was validated"),
                );
                self.stats.replace_contribution(before, after);

                if let Some((ordinal, created)) = conversation_change {
                    let id = ConversationItemId::ResponseSegment {
                        request_id: request_id.clone(),
                        ordinal,
                    };
                    if created {
                        self.conversation_order.push(id.clone());
                        changes.conversation_order = true;
                    }
                    changes.conversation.push(id);
                }
                for ordinal in payload_segment_ids {
                    changes
                        .conversation
                        .push(ConversationItemId::ResponseSegment {
                            request_id: request_id.clone(),
                            ordinal,
                        });
                }
                if became_visible {
                    let id = TrajectoryItemId::Assistant(request_id.clone());
                    self.trajectory_order.push(id.clone());
                    changes.trajectory_order = true;
                    changes.trajectory.push(id);
                } else if self
                    .graph
                    .requests
                    .get(request_id)
                    .is_some_and(|response| response.visible)
                {
                    changes
                        .trajectory
                        .push(TrajectoryItemId::Assistant(request_id.clone()));
                }
                changes.geometry |= became_visible || first_token_changed;
            }
            SessionEvent::AssistantCompleted {
                request_id,
                items,
                response: response_info,
            } => {
                let before = response_contribution(
                    self.graph
                        .requests
                        .get(request_id)
                        .expect("assistant completion was validated"),
                );
                let old_segment_ids = self
                    .graph
                    .requests
                    .get(request_id)
                    .expect("assistant completion was validated")
                    .segments
                    .iter()
                    .map(|segment| segment.ordinal)
                    .collect::<Vec<_>>();
                let (became_visible, segment_ids) = {
                    let response = self
                        .graph
                        .requests
                        .get_mut(request_id)
                        .expect("assistant completion was validated");
                    let was_visible = response.visible;
                    response.source_seqs.push(seq);
                    response.completed_items_json =
                        serde_json::to_string(items).unwrap_or_default();
                    response.tool_call_order = tool_call_order_from_items(items);
                    response.tool_calls = merge_final_tool_calls(
                        std::mem::take(&mut response.tool_calls),
                        tool_calls_from_items(items),
                    );
                    for call in response.tool_calls.values_mut() {
                        call.source_seqs.push(seq);
                    }
                    response.tool_payload =
                        format_tool_payload(&response.tool_calls, &response.tool_call_order);
                    response.response = Some(response_info.clone());
                    response.usage = response_info.usage.or(response.usage);
                    response.status = ItemStatus::Completed;
                    response.timing.completed = Some(EventTimeRef::from(time));
                    reconcile_completed_segments(response, items, seq);
                    response.visible = true;
                    (
                        !was_visible,
                        response
                            .segments
                            .iter()
                            .map(|segment| segment.ordinal)
                            .collect::<Vec<_>>(),
                    )
                };
                let after = response_contribution(
                    self.graph
                        .requests
                        .get(request_id)
                        .expect("assistant completion was validated"),
                );
                self.stats.replace_contribution(before, after);

                if old_segment_ids != segment_ids {
                    let insertion = self
                        .conversation_order
                        .iter()
                        .position(|id| {
                            matches!(
                                id,
                                ConversationItemId::ResponseSegment {
                                    request_id: candidate,
                                    ..
                                } if candidate == request_id
                            )
                        })
                        .unwrap_or(self.conversation_order.len());
                    self.conversation_order.retain(|id| {
                        !matches!(
                            id,
                            ConversationItemId::ResponseSegment {
                                request_id: candidate,
                                ..
                            } if candidate == request_id
                        )
                    });
                    for (offset, ordinal) in segment_ids.iter().copied().enumerate() {
                        self.conversation_order.insert(
                            (insertion + offset).min(self.conversation_order.len()),
                            ConversationItemId::ResponseSegment {
                                request_id: request_id.clone(),
                                ordinal,
                            },
                        );
                    }
                    changes.conversation_order = true;
                }
                for ordinal in segment_ids {
                    changes
                        .conversation
                        .push(ConversationItemId::ResponseSegment {
                            request_id: request_id.clone(),
                            ordinal,
                        });
                }
                let trajectory_id = TrajectoryItemId::Assistant(request_id.clone());
                if became_visible {
                    self.trajectory_order.push(trajectory_id.clone());
                    changes.trajectory_order = true;
                }
                changes.trajectory.push(trajectory_id);
                changes.geometry = true;
            }
            SessionEvent::ToolCallRequested {
                request_id,
                call_id,
                parent_call_id,
            } => {
                let (became_visible, placeholder, metadata) = {
                    let response = self
                        .graph
                        .requests
                        .get_mut(request_id)
                        .expect("tool call request was validated");
                    let was_visible = response.visible;
                    response.visible = true;
                    response.active_segment = None;
                    response.source_seqs.push(seq);
                    let metadata = response
                        .tool_calls
                        .get(call_id)
                        .cloned()
                        .unwrap_or_else(|| ToolCallMetadata {
                            name: call_id.to_string(),
                            arguments: String::new(),
                            first_observed: None,
                            source_seqs: Vec::new(),
                        });
                    let placeholder = if response.segments.is_empty() {
                        let (ordinal, created) =
                            append_response_text(response, ResponseChannel::Output, "", seq);
                        Some((ordinal, created))
                    } else {
                        None
                    };
                    (!was_visible, placeholder, metadata)
                };
                if let Some((ordinal, created)) = placeholder {
                    let id = ConversationItemId::ResponseSegment {
                        request_id: request_id.clone(),
                        ordinal,
                    };
                    if created {
                        self.conversation_order.push(id.clone());
                        changes.conversation_order = true;
                    }
                    changes.conversation.push(id);
                }
                if became_visible {
                    let id = TrajectoryItemId::Assistant(request_id.clone());
                    self.trajectory_order.push(id.clone());
                    changes.trajectory.push(id);
                    changes.trajectory_order = true;
                } else {
                    changes
                        .trajectory
                        .push(TrajectoryItemId::Assistant(request_id.clone()));
                }

                let ToolCallMetadata {
                    name,
                    arguments,
                    first_observed,
                    mut source_seqs,
                } = metadata;
                source_seqs.push(seq);
                self.graph.tools.insert(
                    call_id.clone(),
                    ToolNode {
                        call_id: call_id.clone(),
                        request_id: request_id.clone(),
                        parent_call_id: parent_call_id.clone(),
                        name,
                        arguments,
                        output: String::new(),
                        authorization: None,
                        execution_outcome: None,
                        item_json: String::new(),
                        status: ItemStatus::Running,
                        timing: TimingMetrics {
                            started: Some(
                                first_observed
                                    .as_ref()
                                    .map_or_else(|| EventTimeRef::from(time), EventTimeRef::from),
                            ),
                            requested: Some(EventTimeRef::from(time)),
                            ..TimingMetrics::default()
                        },
                        source_seqs,
                    },
                );
                let conversation_id = ConversationItemId::Tool(call_id.clone());
                let trajectory_id = TrajectoryItemId::Tool(call_id.clone());
                self.conversation_order.push(conversation_id.clone());
                self.trajectory_order.push(trajectory_id.clone());
                changes.conversation.push(conversation_id);
                changes.trajectory.push(trajectory_id);
                changes.conversation_order = true;
                changes.trajectory_order = true;
                changes.geometry = true;
            }
            SessionEvent::ToolAuthorizationResolved { call_id, decision } => {
                let tool = self
                    .graph
                    .tools
                    .get_mut(call_id)
                    .expect("tool authorization was validated");
                tool.authorization = Some(*decision);
                tool.timing.authorization_resolved = Some(EventTimeRef::from(time));
                tool.status = match decision {
                    ToolAuthorizationDecision::NotRequired | ToolAuthorizationDecision::Allowed => {
                        ItemStatus::Running
                    }
                    ToolAuthorizationDecision::Denied => ItemStatus::Denied,
                    ToolAuthorizationDecision::Unavailable => ItemStatus::NotExecuted,
                    ToolAuthorizationDecision::Aborted => ItemStatus::Aborted,
                };
                tool.source_seqs.push(seq);
                changes
                    .conversation
                    .push(ConversationItemId::Tool(call_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Tool(call_id.clone()));
            }
            SessionEvent::ToolDispatchIntended { call_id } => {
                let tool = self
                    .graph
                    .tools
                    .get_mut(call_id)
                    .expect("tool dispatch was validated");
                tool.timing.dispatch_intended = Some(EventTimeRef::from(time));
                tool.status = ItemStatus::Running;
                tool.source_seqs.push(seq);
                changes
                    .conversation
                    .push(ConversationItemId::Tool(call_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Tool(call_id.clone()));
            }
            SessionEvent::ToolExecutionStarted { call_id } => {
                let tool = self
                    .graph
                    .tools
                    .get_mut(call_id)
                    .expect("tool execution start was validated");
                if tool.timing.execution_started.is_none() {
                    tool.timing.execution_started = Some(EventTimeRef::from(time));
                    changes.geometry = true;
                }
                tool.status = ItemStatus::Running;
                tool.source_seqs.push(seq);
                changes
                    .conversation
                    .push(ConversationItemId::Tool(call_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Tool(call_id.clone()));
            }
            SessionEvent::ToolExecutionFinished { call_id, outcome } => {
                let tool = self
                    .graph
                    .tools
                    .get_mut(call_id)
                    .expect("tool execution finish was validated");
                tool.timing.execution_finished = Some(EventTimeRef::from(time));
                tool.execution_outcome = Some(*outcome);
                tool.status = status_from_tool_execution(*outcome);
                tool.source_seqs.push(seq);
                changes
                    .conversation
                    .push(ConversationItemId::Tool(call_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Tool(call_id.clone()));
                changes.geometry = true;
            }
            SessionEvent::ToolResultAttached {
                call_id,
                status,
                item,
            } => {
                let before = tool_contribution(
                    self.graph
                        .tools
                        .get(call_id)
                        .expect("tool result was validated"),
                );
                let tool = self
                    .graph
                    .tools
                    .get_mut(call_id)
                    .expect("tool result was validated");
                tool.output = tool_result_preview(item);
                tool.item_json = serde_json::to_string(item).unwrap_or_default();
                tool.status = status_from_tool_result(*status);
                tool.timing.completed = Some(EventTimeRef::from(time));
                tool.source_seqs.push(seq);
                let after = tool_contribution(tool);
                self.stats.replace_contribution(before, after);
                changes
                    .conversation
                    .push(ConversationItemId::Tool(call_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Tool(call_id.clone()));
                changes.geometry = true;
            }
            SessionEvent::CompactionStarted {
                compaction_id,
                run_id,
                tokens_before,
                first_kept_id,
            } => {
                self.graph.compactions.insert(
                    compaction_id.clone(),
                    CompactionNode {
                        id: compaction_id.clone(),
                        run_id: run_id.clone(),
                        tokens_before: *tokens_before,
                        first_kept_id: *first_kept_id,
                        summary: String::new(),
                        response: None,
                        outcome: None,
                        status: ItemStatus::Running,
                        timing: TimingMetrics {
                            started: Some(EventTimeRef::from(time)),
                            ..TimingMetrics::default()
                        },
                        source_seqs: vec![seq],
                    },
                );
                let conversation_id = ConversationItemId::Compaction(compaction_id.clone());
                let trajectory_id = TrajectoryItemId::Compaction(compaction_id.clone());
                self.conversation_order.push(conversation_id.clone());
                self.trajectory_order.push(trajectory_id.clone());
                changes.conversation.push(conversation_id);
                changes.trajectory.push(trajectory_id);
                changes.conversation_order = true;
                changes.trajectory_order = true;
                changes.geometry = true;
            }
            SessionEvent::CompactionFinished {
                compaction_id,
                outcome,
                summary,
                response,
            } => {
                let compaction = self
                    .graph
                    .compactions
                    .get_mut(compaction_id)
                    .expect("compaction finish was validated");
                compaction.outcome = Some(*outcome);
                compaction.summary = summary.clone().unwrap_or_default();
                compaction.response.clone_from(response);
                compaction.status = status_from_step_outcome(*outcome);
                compaction.timing.completed = Some(EventTimeRef::from(time));
                compaction.source_seqs.push(seq);
                changes
                    .conversation
                    .push(ConversationItemId::Compaction(compaction_id.clone()));
                changes
                    .trajectory
                    .push(TrajectoryItemId::Compaction(compaction_id.clone()));
                changes.geometry = true;
            }
        }
    }
}

impl SessionDocument {
    fn conversation_item<'a>(
        &'a self,
        id: &'a ConversationItemId,
    ) -> Option<ConversationItemView<'a>> {
        match id {
            ConversationItemId::Input(input_id) => {
                let input = self.graph.inputs.get(input_id)?;
                let step_id = input.step_id.as_ref();
                Some(ConversationItemView {
                    id,
                    role: conversation_role(input.origin),
                    title: None,
                    text: &input.text,
                    payload: nonempty(&input.attached_payload),
                    status: if input.state == InputState::Attached {
                        ItemStatus::Completed
                    } else {
                        ItemStatus::Pending
                    },
                    turn_id: step_id.and_then(|step_id| self.turn_for_step(step_id)),
                    step_id,
                })
            }
            ConversationItemId::ResponseSegment {
                request_id,
                ordinal,
            } => {
                let response = self.graph.requests.get(request_id)?;
                let segment = response
                    .segments
                    .iter()
                    .find(|segment| segment.ordinal == *ordinal)?;
                let text = if segment.text.is_empty() && !response.tool_payload.is_empty() {
                    "(tool call only)"
                } else {
                    &segment.text
                };
                Some(ConversationItemView {
                    id,
                    role: match segment.channel {
                        ResponseChannel::Reasoning => ConversationRole::Reasoning,
                        ResponseChannel::Output => ConversationRole::Assistant,
                    },
                    title: None,
                    text,
                    payload: nonempty(&response.tool_payload),
                    status: response.status,
                    turn_id: self.turn_for_step(&response.step_id),
                    step_id: Some(&response.step_id),
                })
            }
            ConversationItemId::Tool(call_id) => {
                let tool = self.graph.tools.get(call_id)?;
                let response = self.graph.requests.get(&tool.request_id)?;
                Some(ConversationItemView {
                    id,
                    role: ConversationRole::Tool,
                    title: Some(&tool.name),
                    text: &tool.output,
                    payload: nonempty(&tool.arguments),
                    status: tool.status,
                    turn_id: self.turn_for_step(&response.step_id),
                    step_id: Some(&response.step_id),
                })
            }
            ConversationItemId::Compaction(compaction_id) => {
                let compaction = self.graph.compactions.get(compaction_id)?;
                Some(ConversationItemView {
                    id,
                    role: ConversationRole::Notice,
                    title: Some("Compaction"),
                    text: if compaction.summary.is_empty() {
                        "Compacting context…"
                    } else {
                        &compaction.summary
                    },
                    payload: None,
                    status: compaction.status,
                    turn_id: None,
                    step_id: None,
                })
            }
        }
    }

    fn trajectory_item<'a>(&'a self, id: &'a TrajectoryItemId) -> Option<TrajectoryItemView<'a>> {
        match id {
            TrajectoryItemId::PromptChange(source_seq) => {
                let prompt = self.graph.prompts.get(source_seq)?;
                let (turn_id, step_id) = if prompt.kind == PromptChangeKind::Initial {
                    (None, None)
                } else {
                    (self.turn_for_step(&prompt.step_id), Some(&prompt.step_id))
                };
                Some(TrajectoryItemView {
                    id,
                    kind: TrajectoryKind::System,
                    lane: TrajectoryLane::Input,
                    title: prompt_title(prompt.kind),
                    text: &prompt.summary,
                    payload: nonempty(&prompt.instructions),
                    status: ItemStatus::Completed,
                    timing: &prompt.timing,
                    usage: None,
                    turn_id,
                    step_id,
                    source_seqs: &prompt.source_seqs,
                })
            }
            TrajectoryItemId::Input(input_id) => {
                let input = self.graph.inputs.get(input_id)?;
                let step_id = input.step_id.as_ref();
                Some(TrajectoryItemView {
                    id,
                    kind: trajectory_input_kind(input.origin),
                    lane: TrajectoryLane::Input,
                    title: input_title(input.origin),
                    text: &input.text,
                    payload: nonempty(&input.attached_payload),
                    status: if input.state == InputState::Attached {
                        ItemStatus::Completed
                    } else {
                        ItemStatus::Pending
                    },
                    timing: &input.timing,
                    usage: None,
                    turn_id: step_id.and_then(|step_id| self.turn_for_step(step_id)),
                    step_id,
                    source_seqs: &input.source_seqs,
                })
            }
            TrajectoryItemId::Assistant(request_id) => {
                let response = self.graph.requests.get(request_id)?;
                Some(TrajectoryItemView {
                    id,
                    kind: TrajectoryKind::Assistant,
                    lane: TrajectoryLane::Model,
                    title: "Assistant",
                    text: if response.combined_text.is_empty() && !response.tool_payload.is_empty()
                    {
                        "(tool call only)"
                    } else {
                        &response.combined_text
                    },
                    payload: nonempty(&response.tool_payload),
                    status: response.status,
                    timing: &response.timing,
                    usage: response.usage,
                    turn_id: self.turn_for_step(&response.step_id),
                    step_id: Some(&response.step_id),
                    source_seqs: &response.source_seqs,
                })
            }
            TrajectoryItemId::Tool(call_id) => {
                let tool = self.graph.tools.get(call_id)?;
                let response = self.graph.requests.get(&tool.request_id)?;
                Some(TrajectoryItemView {
                    id,
                    kind: TrajectoryKind::Tool,
                    lane: TrajectoryLane::Tools,
                    title: &tool.name,
                    text: &tool.output,
                    payload: nonempty(&tool.arguments),
                    status: tool.status,
                    timing: &tool.timing,
                    usage: None,
                    turn_id: self.turn_for_step(&response.step_id),
                    step_id: Some(&response.step_id),
                    source_seqs: &tool.source_seqs,
                })
            }
            TrajectoryItemId::Compaction(compaction_id) => {
                let compaction = self.graph.compactions.get(compaction_id)?;
                Some(TrajectoryItemView {
                    id,
                    kind: TrajectoryKind::Compaction,
                    lane: TrajectoryLane::Model,
                    title: "Compaction",
                    text: if compaction.summary.is_empty() {
                        "Compacting context…"
                    } else {
                        &compaction.summary
                    },
                    payload: None,
                    status: compaction.status,
                    timing: &compaction.timing,
                    usage: compaction
                        .response
                        .as_ref()
                        .and_then(|response| response.usage),
                    turn_id: None,
                    step_id: None,
                    source_seqs: &compaction.source_seqs,
                })
            }
            TrajectoryItemId::RequestFailure(step_id) => {
                let failure = self.graph.failures.get(step_id)?;
                Some(TrajectoryItemView {
                    id,
                    kind: TrajectoryKind::RequestFailure,
                    lane: TrajectoryLane::Model,
                    title: "Request failed",
                    text: &failure.error,
                    payload: None,
                    status: ItemStatus::Failed,
                    timing: &failure.timing,
                    usage: None,
                    turn_id: self.turn_for_step(&failure.step_id),
                    step_id: Some(&failure.step_id),
                    source_seqs: &failure.source_seqs,
                })
            }
        }
    }

    fn turn_for_step(&self, step_id: &StepId) -> Option<&TurnId> {
        self.graph.steps.get(step_id).map(|step| &step.turn_id)
    }

    #[cfg(test)]
    fn stats_from_full_scan(&self) -> SessionStats {
        let mut stats = SessionStats {
            turns: self
                .graph
                .turns
                .values()
                .filter(|turn| turn.completed.is_some())
                .count(),
            steps: self
                .graph
                .steps
                .values()
                .filter(|step| step.completed.is_some())
                .count(),
            ..SessionStats::default()
        };
        for response in self.graph.requests.values() {
            stats.add_contribution(response_contribution(response));
        }
        for tool in self.graph.tools.values() {
            stats.add_contribution(tool_contribution(tool));
        }
        stats
    }
}

impl SessionStats {
    fn add_contribution(&mut self, contribution: StatsContribution) {
        self.llm_ns = self.llm_ns.saturating_add(contribution.llm_ns);
        self.tool_ns = self.tool_ns.saturating_add(contribution.tool_ns);
        self.ttft_ns = self.ttft_ns.saturating_add(contribution.ttft_ns);
        self.ttft_samples = self.ttft_samples.saturating_add(contribution.ttft_samples);
        self.decode_ns = self.decode_ns.saturating_add(contribution.decode_ns);
        self.decode_tokens = self
            .decode_tokens
            .saturating_add(contribution.decode_tokens);
        self.uncached_input_tokens = self
            .uncached_input_tokens
            .saturating_add(contribution.usage.uncached_input_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .saturating_add(contribution.usage.cache_read_input_tokens);
        self.cache_write_input_tokens = self
            .cache_write_input_tokens
            .saturating_add(contribution.usage.cache_write_input_tokens);
        self.output_tokens = self
            .output_tokens
            .saturating_add(contribution.usage.output_tokens);
        self.reasoning_output_tokens = self
            .reasoning_output_tokens
            .saturating_add(contribution.usage.reasoning_output_tokens);
    }

    fn subtract_contribution(&mut self, contribution: StatsContribution) {
        self.llm_ns = self.llm_ns.saturating_sub(contribution.llm_ns);
        self.tool_ns = self.tool_ns.saturating_sub(contribution.tool_ns);
        self.ttft_ns = self.ttft_ns.saturating_sub(contribution.ttft_ns);
        self.ttft_samples = self.ttft_samples.saturating_sub(contribution.ttft_samples);
        self.decode_ns = self.decode_ns.saturating_sub(contribution.decode_ns);
        self.decode_tokens = self
            .decode_tokens
            .saturating_sub(contribution.decode_tokens);
        self.uncached_input_tokens = self
            .uncached_input_tokens
            .saturating_sub(contribution.usage.uncached_input_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .saturating_sub(contribution.usage.cache_read_input_tokens);
        self.cache_write_input_tokens = self
            .cache_write_input_tokens
            .saturating_sub(contribution.usage.cache_write_input_tokens);
        self.output_tokens = self
            .output_tokens
            .saturating_sub(contribution.usage.output_tokens);
        self.reasoning_output_tokens = self
            .reasoning_output_tokens
            .saturating_sub(contribution.usage.reasoning_output_tokens);
    }

    fn replace_contribution(&mut self, before: StatsContribution, after: StatsContribution) {
        self.subtract_contribution(before);
        self.add_contribution(after);
    }
}

fn response_contribution(response: &ResponseNode) -> StatsContribution {
    let usage = response.usage.unwrap_or_default();
    if response.status != ItemStatus::Completed {
        return StatsContribution {
            usage,
            ..StatsContribution::default()
        };
    }
    let ttft = response.timing.ttft_ns();
    StatsContribution {
        llm_ns: response.timing.duration_ns().unwrap_or_default(),
        ttft_ns: ttft.unwrap_or_default(),
        ttft_samples: usize::from(ttft.is_some()),
        decode_ns: response.timing.decode_ns().unwrap_or_default(),
        decode_tokens: usage.total_output_tokens(),
        usage,
        ..StatsContribution::default()
    }
}

fn tool_contribution(tool: &ToolNode) -> StatsContribution {
    StatsContribution {
        tool_ns: tool.timing.duration_ns().unwrap_or_default(),
        ..StatsContribution::default()
    }
}

fn append_response_text(
    response: &mut ResponseNode,
    channel: ResponseChannel,
    delta: &str,
    source_seq: u64,
) -> (u32, bool) {
    let active = response.active_segment.filter(|index| {
        response
            .segments
            .get(*index)
            .is_some_and(|segment| segment.channel == channel)
    });
    let (index, created) = if let Some(index) = active {
        (index, false)
    } else {
        let ordinal = u32::try_from(response.segments.len()).unwrap_or(u32::MAX);
        response.segments.push(ResponseSegment {
            ordinal,
            channel,
            text: String::new(),
            source_seqs: Vec::new(),
        });
        let index = response.segments.len() - 1;
        response.active_segment = Some(index);
        (index, true)
    };
    let segment = &mut response.segments[index];
    segment.text.push_str(delta);
    segment.source_seqs.push(source_seq);
    response.combined_text.push_str(delta);
    (segment.ordinal, created)
}

fn prompt_change_kind(
    previous: Option<&PromptSnapshot>,
    current: &PromptSnapshot,
    reason: RequestHeaderReason,
) -> Option<PromptChangeKind> {
    let Some(previous) = previous else {
        return (reason == RequestHeaderReason::Initial).then_some(PromptChangeKind::Initial);
    };
    let system_changed = previous.instructions != current.instructions;
    let tools_changed = previous.tools_json != current.tools_json;
    match (system_changed, tools_changed) {
        (false, false) => None,
        (true, false) => Some(PromptChangeKind::System),
        (false, true) => Some(PromptChangeKind::Tools),
        (true, true) => Some(PromptChangeKind::SystemAndTools),
    }
}

fn input_items_text(items: &[kcastle_agent::InputItem]) -> String {
    let mut parts = Vec::new();
    for item in items {
        if let Ok(value) = serde_json::to_value(item) {
            collect_semantic_text(&value, &mut parts);
        }
    }
    parts.join("\n")
}

/// Streaming deltas are observations, while `AssistantCompleted.items` is the canonical provider
/// payload. Reconcile the visible segments at completion so a dropped, duplicated, or corrected
/// delta can never become the durable UI truth.
fn reconcile_completed_segments(
    response: &mut ResponseNode,
    items: &[kcastle_agent::InputItem],
    source_seq: u64,
) {
    let mut canonical_segments: Vec<(ResponseChannel, String)> = Vec::new();
    for item in items {
        let Ok(value) = serde_json::to_value(item) else {
            continue;
        };
        let channel = match value.get("type").and_then(serde_json::Value::as_str) {
            Some("reasoning") => Some(ResponseChannel::Reasoning),
            Some("message") => Some(ResponseChannel::Output),
            _ => None,
        };
        let Some(channel) = channel else {
            continue;
        };
        let mut parts = Vec::new();
        collect_semantic_text(&value, &mut parts);
        let text = parts.join("\n");
        if text.is_empty() {
            continue;
        }
        if let Some((last_channel, last_text)) = canonical_segments.last_mut()
            && *last_channel == channel
        {
            if !last_text.is_empty() {
                last_text.push('\n');
            }
            last_text.push_str(&text);
        } else {
            canonical_segments.push((channel, text));
        }
    }
    // Completed provider items are the canonical ordering authority. Only fall back to streamed
    // observations when a provider omits textual items entirely.
    let resolved_segments = if canonical_segments.is_empty() {
        if response.segments.is_empty() {
            vec![(ResponseChannel::Output, input_items_text(items))]
        } else {
            response
                .segments
                .iter()
                .map(|segment| (segment.channel, segment.text.clone()))
                .collect()
        }
    } else {
        canonical_segments
    };

    response.segments = resolved_segments
        .into_iter()
        .filter(|(_, text)| !text.is_empty())
        .enumerate()
        .map(|(ordinal, (channel, text))| ResponseSegment {
            ordinal: u32::try_from(ordinal).unwrap_or(u32::MAX),
            channel,
            text,
            source_seqs: vec![source_seq],
        })
        .collect();
    response.combined_text = response
        .segments
        .iter()
        .map(|segment| segment.text.as_str())
        .collect::<String>();
    response.active_segment = None;
}

fn tool_calls_from_items(items: &[kcastle_agent::InputItem]) -> HashMap<CallId, ToolCallMetadata> {
    let mut calls = HashMap::new();
    for item in items {
        let Ok(value) = serde_json::to_value(item) else {
            continue;
        };
        let serde_json::Value::Object(object) = value else {
            continue;
        };
        if object.get("type").and_then(serde_json::Value::as_str) != Some("function_call") {
            continue;
        }
        let Some(call_id) = object
            .get("call_id")
            .and_then(serde_json::Value::as_str)
            .filter(|call_id| !call_id.is_empty())
        else {
            continue;
        };
        calls.insert(
            CallId::from_raw(call_id),
            ToolCallMetadata {
                name: object
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                arguments: object
                    .get("arguments")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                first_observed: None,
                source_seqs: Vec::new(),
            },
        );
    }
    calls
}

fn tool_call_order_from_items(items: &[kcastle_agent::InputItem]) -> Vec<CallId> {
    items
        .iter()
        .filter_map(|item| serde_json::to_value(item).ok())
        .filter_map(|value| match value {
            serde_json::Value::Object(object)
                if object.get("type").and_then(serde_json::Value::as_str)
                    == Some("function_call") =>
            {
                object
                    .get("call_id")
                    .and_then(serde_json::Value::as_str)
                    .filter(|call_id| !call_id.is_empty())
                    .map(CallId::from_raw)
            }
            _ => None,
        })
        .collect()
}

fn merge_final_tool_calls(
    streamed: HashMap<CallId, ToolCallMetadata>,
    mut completed: HashMap<CallId, ToolCallMetadata>,
) -> HashMap<CallId, ToolCallMetadata> {
    for (call_id, call) in &mut completed {
        if let Some(streamed) = streamed.get(call_id) {
            call.first_observed.clone_from(&streamed.first_observed);
            call.source_seqs.clone_from(&streamed.source_seqs);
        }
    }
    completed
}

fn format_tool_payload(calls: &HashMap<CallId, ToolCallMetadata>, order: &[CallId]) -> String {
    order
        .iter()
        .filter_map(|call_id| calls.get(call_id))
        .map(
            |call| match (call.name.is_empty(), call.arguments.is_empty()) {
                (false, false) => format!("{} {}", call.name, call.arguments),
                (false, true) => call.name.clone(),
                (true, false) => call.arguments.clone(),
                (true, true) => String::new(),
            },
        )
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_result_preview(item: &kcastle_agent::InputItem) -> String {
    let Ok(value) = serde_json::to_value(item) else {
        return String::new();
    };
    if let serde_json::Value::Object(object) = &value
        && object.get("type").and_then(serde_json::Value::as_str) == Some("function_call_output")
        && let Some(output) = object.get("output")
    {
        if let Some(output) = output.as_str() {
            return output.to_owned();
        }
        let mut parts = Vec::new();
        collect_semantic_text(output, &mut parts);
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }
    let mut parts = Vec::new();
    collect_semantic_text(&value, &mut parts);
    parts.join("\n")
}

fn collect_semantic_text(value: &serde_json::Value, output: &mut Vec<String>) {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_semantic_text(value, output);
            }
        }
        serde_json::Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(serde_json::Value::as_str) {
                output.push(text.to_owned());
            }
            for key in ["content", "summary"] {
                if let Some(value) = object.get(key) {
                    if let Some(text) = value.as_str() {
                        output.push(text.to_owned());
                    } else {
                        collect_semantic_text(value, output);
                    }
                }
            }
        }
        _ => {}
    }
}

fn point_timing(time: &EventTime) -> TimingMetrics {
    TimingMetrics {
        started: Some(EventTimeRef::from(time)),
        completed: Some(EventTimeRef::from(time)),
        ..TimingMetrics::default()
    }
}

fn conversation_role(origin: InputOrigin) -> ConversationRole {
    match origin {
        InputOrigin::Initial | InputOrigin::Queue => ConversationRole::User,
        InputOrigin::Steer => ConversationRole::Steering,
        InputOrigin::Context => ConversationRole::Context,
    }
}

fn trajectory_input_kind(origin: InputOrigin) -> TrajectoryKind {
    match origin {
        InputOrigin::Initial | InputOrigin::Queue => TrajectoryKind::User,
        InputOrigin::Steer => TrajectoryKind::Steering,
        InputOrigin::Context => TrajectoryKind::Context,
    }
}

fn input_title(origin: InputOrigin) -> &'static str {
    match origin {
        InputOrigin::Initial | InputOrigin::Queue => "User",
        InputOrigin::Steer => "Steering",
        InputOrigin::Context => "Context",
    }
}

fn prompt_title(kind: PromptChangeKind) -> &'static str {
    match kind {
        PromptChangeKind::Initial => "Initial System Prompt",
        PromptChangeKind::System => "System Prompt Updated",
        PromptChangeKind::Tools => "Tools Updated",
        PromptChangeKind::SystemAndTools => "System Prompt and Tools Updated",
    }
}

fn status_from_step_outcome(outcome: StepOutcome) -> ItemStatus {
    match outcome {
        StepOutcome::Completed => ItemStatus::Completed,
        StepOutcome::Failed => ItemStatus::Failed,
        StepOutcome::Aborted => ItemStatus::Aborted,
    }
}

fn status_from_tool_execution(outcome: ToolExecutionOutcome) -> ItemStatus {
    match outcome {
        ToolExecutionOutcome::Success => ItemStatus::Running,
        ToolExecutionOutcome::Error => ItemStatus::Failed,
        ToolExecutionOutcome::UnknownSideEffects => ItemStatus::Unknown,
    }
}

fn status_from_tool_result(status: ToolResultStatus) -> ItemStatus {
    match status {
        ToolResultStatus::Success => ItemStatus::Completed,
        ToolResultStatus::Error => ItemStatus::Failed,
        ToolResultStatus::Denied => ItemStatus::Denied,
        ToolResultStatus::NotFound | ToolResultStatus::AbortedBeforeDispatch => {
            ItemStatus::NotExecuted
        }
        ToolResultStatus::UnknownSideEffects => ItemStatus::Unknown,
    }
}

fn nonempty(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

#[derive(Default)]
struct PlanOverlay {
    runs: HashSet<RunId>,
    turns: HashSet<TurnId>,
    steps: HashSet<StepId>,
    inputs: HashSet<InputId>,
    requests: HashSet<RequestId>,
    tools: HashSet<CallId>,
    compactions: HashSet<CompactionId>,
}

impl SessionDocument {
    fn validate_event(
        &self,
        event: &SessionEvent,
        overlay: &mut PlanOverlay,
    ) -> Result<(), ProjectionError> {
        match event {
            SessionEvent::RunStarted { run_id } => {
                require_new(
                    !self.graph.runs.contains_key(run_id) && overlay.runs.insert(run_id.clone()),
                    "run",
                    run_id,
                )?;
            }
            SessionEvent::RunTerminated { run_id, .. } => {
                require_present(
                    self.graph.runs.contains_key(run_id) || overlay.runs.contains(run_id),
                    "run",
                    run_id,
                )?;
            }
            SessionEvent::TurnStarted { run_id, turn_id } => {
                require_present(
                    self.graph.runs.contains_key(run_id) || overlay.runs.contains(run_id),
                    "run",
                    run_id,
                )?;
                require_new(
                    !self.graph.turns.contains_key(turn_id)
                        && overlay.turns.insert(turn_id.clone()),
                    "turn",
                    turn_id,
                )?;
            }
            SessionEvent::TurnTerminated { turn_id, .. } => {
                require_present(
                    self.graph.turns.contains_key(turn_id) || overlay.turns.contains(turn_id),
                    "turn",
                    turn_id,
                )?;
            }
            SessionEvent::StepStarted { turn_id, step_id } => {
                require_present(
                    self.graph.turns.contains_key(turn_id) || overlay.turns.contains(turn_id),
                    "turn",
                    turn_id,
                )?;
                require_new(
                    !self.graph.steps.contains_key(step_id)
                        && overlay.steps.insert(step_id.clone()),
                    "step",
                    step_id,
                )?;
            }
            SessionEvent::StepTerminated { step_id, .. } => {
                self.require_step(step_id, overlay)?;
            }
            SessionEvent::InputSubmitted { input_id, .. } => {
                require_new(
                    !self.graph.inputs.contains_key(input_id)
                        && overlay.inputs.insert(input_id.clone()),
                    "input",
                    input_id,
                )?;
            }
            SessionEvent::InputAttached {
                input_id, step_id, ..
            } => {
                require_present(
                    self.graph.inputs.contains_key(input_id) || overlay.inputs.contains(input_id),
                    "input",
                    input_id,
                )?;
                self.require_step(step_id, overlay)?;
            }
            SessionEvent::RequestSnapshot {
                request_id,
                step_id,
                ..
            } => {
                self.require_step(step_id, overlay)?;
                require_new(
                    !self.graph.requests.contains_key(request_id)
                        && overlay.requests.insert(request_id.clone()),
                    "request",
                    request_id,
                )?;
            }
            SessionEvent::ModelRequestStarted { request_id }
            | SessionEvent::ModelRequestFailed { request_id, .. }
            | SessionEvent::AssistantChunk { request_id, .. }
            | SessionEvent::AssistantCompleted { request_id, .. } => {
                self.require_request(request_id, overlay)?;
            }
            SessionEvent::ToolCallRequested {
                request_id,
                call_id,
                ..
            } => {
                self.require_request(request_id, overlay)?;
                require_new(
                    !self.graph.tools.contains_key(call_id)
                        && overlay.tools.insert(call_id.clone()),
                    "tool call",
                    call_id,
                )?;
            }
            SessionEvent::ToolAuthorizationResolved { call_id, .. }
            | SessionEvent::ToolDispatchIntended { call_id }
            | SessionEvent::ToolExecutionStarted { call_id }
            | SessionEvent::ToolExecutionFinished { call_id, .. }
            | SessionEvent::ToolResultAttached { call_id, .. } => {
                require_present(
                    self.graph.tools.contains_key(call_id) || overlay.tools.contains(call_id),
                    "tool call",
                    call_id,
                )?;
            }
            SessionEvent::CompactionStarted {
                compaction_id,
                run_id,
                ..
            } => {
                require_present(
                    self.graph.runs.contains_key(run_id) || overlay.runs.contains(run_id),
                    "run",
                    run_id,
                )?;
                require_new(
                    !self.graph.compactions.contains_key(compaction_id)
                        && overlay.compactions.insert(compaction_id.clone()),
                    "compaction",
                    compaction_id,
                )?;
            }
            SessionEvent::CompactionFinished { compaction_id, .. } => {
                require_present(
                    self.graph.compactions.contains_key(compaction_id)
                        || overlay.compactions.contains(compaction_id),
                    "compaction",
                    compaction_id,
                )?;
            }
        }
        Ok(())
    }

    fn require_step(&self, step_id: &StepId, overlay: &PlanOverlay) -> Result<(), ProjectionError> {
        require_present(
            self.graph.steps.contains_key(step_id) || overlay.steps.contains(step_id),
            "step",
            step_id,
        )
    }

    fn require_request(
        &self,
        request_id: &RequestId,
        overlay: &PlanOverlay,
    ) -> Result<(), ProjectionError> {
        require_present(
            self.graph.requests.contains_key(request_id) || overlay.requests.contains(request_id),
            "request",
            request_id,
        )
    }
}

fn require_new(
    condition: bool,
    kind: &'static str,
    id: &impl fmt::Display,
) -> Result<(), ProjectionError> {
    condition
        .then_some(())
        .ok_or_else(|| ProjectionError::Duplicate {
            kind,
            id: id.to_string(),
        })
}

fn require_present(
    condition: bool,
    kind: &'static str,
    id: &impl fmt::Display,
) -> Result<(), ProjectionError> {
    condition
        .then_some(())
        .ok_or_else(|| ProjectionError::Missing {
            kind,
            id: id.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use kcastle_agent::{
        EasyInputMessage, InputItem, SessionConfig, ToolExecutionOutcome, TurnEndReason, TxId,
    };
    use proptest::prelude::*;

    use super::*;

    fn id<T: From<&'static str>>(value: &'static str) -> T {
        value.into()
    }

    fn usage() -> TokenUsage {
        TokenUsage {
            uncached_input_tokens: 20,
            cache_read_input_tokens: 80,
            cache_write_input_tokens: 3,
            output_tokens: 15,
            reasoning_output_tokens: 5,
        }
    }

    fn response(id: &str, usage: TokenUsage) -> ResponseInfo {
        ResponseInfo {
            id: id.to_owned(),
            model: "deepseek-v4".to_owned(),
            usage: Some(usage),
        }
    }

    fn recorded(seq: u64, event: SessionEvent) -> RecordedEvent {
        RecordedEvent {
            seq,
            tx_id: TxId::from_raw(format!("tx-{seq}")),
            time: EventTime {
                wall_time_ms: i64::try_from(seq).unwrap_or(i64::MAX),
                clock_id: "fixture".to_owned(),
                monotonic_ns: seq.saturating_mul(1_000_000),
            },
            event,
        }
    }

    fn fixture() -> Vec<RecordedEvent> {
        let run_id = id::<RunId>("run-1");
        let turn_1 = id::<TurnId>("turn-1");
        let step_1 = id::<StepId>("step-1");
        let input_1 = id::<InputId>("input-1");
        let request_1 = id::<RequestId>("request-1");
        let call_1 = id::<CallId>("call-1");
        let compaction = id::<CompactionId>("compaction-1");
        let turn_2 = id::<TurnId>("turn-2");
        let step_2 = id::<StepId>("step-2");
        let input_2 = id::<InputId>("input-2");
        let request_2 = id::<RequestId>("request-2");
        let input_item = |text: &str| InputItem::from(EasyInputMessage::from(text));
        let function_call_item = || {
            serde_json::from_value::<InputItem>(serde_json::json!({
                "type": "function_call",
                "arguments": "{\"command\":\"true\"}",
                "call_id": "call-1",
                "name": "shell"
            }))
            .unwrap()
        };
        let tool_output_item = || {
            serde_json::from_value::<InputItem>(serde_json::json!({
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "ok"
            }))
            .unwrap()
        };

        let mut events = vec![
            recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: run_id.clone(),
                },
            ),
            recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id: run_id.clone(),
                    turn_id: turn_1.clone(),
                },
            ),
            recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id: turn_1.clone(),
                    step_id: step_1.clone(),
                },
            ),
            recorded(
                3,
                SessionEvent::InputSubmitted {
                    input_id: input_1.clone(),
                    input: "evaluate this".to_owned(),
                    origin: InputOrigin::Initial,
                },
            ),
            recorded(
                4,
                SessionEvent::InputAttached {
                    input_id: input_1,
                    step_id: step_1.clone(),
                    items: vec![input_item("evaluate this")],
                },
            ),
            recorded(
                5,
                SessionEvent::RequestSnapshot {
                    request_id: request_1.clone(),
                    step_id: step_1.clone(),
                    reason: RequestHeaderReason::Initial,
                    model: "deepseek-v4".to_owned(),
                    instructions: Some("be precise".to_owned()),
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: Some(4_096),
                    session_config: SessionConfig::default(),
                },
            ),
            recorded(
                6,
                SessionEvent::ModelRequestStarted {
                    request_id: request_1.clone(),
                },
            ),
            recorded(
                7,
                SessionEvent::AssistantChunk {
                    request_id: request_1.clone(),
                    chunk: AssistantChunk::ToolCallDelta {
                        call_id: call_1.clone(),
                        name: Some("streamed-shell".to_owned()),
                        arguments_delta: "{\"command\":\"streamed\"}".to_owned(),
                    },
                },
            ),
            recorded(
                8,
                SessionEvent::AssistantChunk {
                    request_id: request_1.clone(),
                    chunk: AssistantChunk::Usage { usage: usage() },
                },
            ),
            recorded(
                9,
                SessionEvent::AssistantCompleted {
                    request_id: request_1.clone(),
                    items: vec![input_item("hello"), function_call_item()],
                    response: response("response-1", usage()),
                },
            ),
            recorded(
                10,
                SessionEvent::ToolCallRequested {
                    request_id: request_1,
                    call_id: call_1.clone(),
                    parent_call_id: None,
                },
            ),
            recorded(
                11,
                SessionEvent::ToolAuthorizationResolved {
                    call_id: call_1.clone(),
                    decision: ToolAuthorizationDecision::NotRequired,
                },
            ),
            recorded(
                12,
                SessionEvent::ToolDispatchIntended {
                    call_id: call_1.clone(),
                },
            ),
            recorded(
                13,
                SessionEvent::ToolExecutionStarted {
                    call_id: call_1.clone(),
                },
            ),
            recorded(
                14,
                SessionEvent::ToolExecutionFinished {
                    call_id: call_1.clone(),
                    outcome: ToolExecutionOutcome::Success,
                },
            ),
            recorded(
                15,
                SessionEvent::ToolResultAttached {
                    call_id: call_1,
                    status: ToolResultStatus::Success,
                    item: tool_output_item(),
                },
            ),
            recorded(
                16,
                SessionEvent::StepTerminated {
                    step_id: step_1,
                    outcome: StepOutcome::Completed,
                    error: None,
                },
            ),
            recorded(
                17,
                SessionEvent::TurnTerminated {
                    turn_id: turn_1,
                    reason: TurnEndReason::Completed,
                },
            ),
            recorded(
                18,
                SessionEvent::CompactionStarted {
                    compaction_id: compaction.clone(),
                    run_id: run_id.clone(),
                    tokens_before: 123_456,
                    first_kept_id: 1,
                },
            ),
            recorded(
                19,
                SessionEvent::CompactionFinished {
                    compaction_id: compaction,
                    outcome: StepOutcome::Completed,
                    summary: Some("compacted".to_owned()),
                    response: Some(response(
                        "compaction-response",
                        TokenUsage {
                            uncached_input_tokens: 999_999,
                            cache_read_input_tokens: 999_999,
                            cache_write_input_tokens: 999_999,
                            output_tokens: 999_999,
                            reasoning_output_tokens: 999_999,
                        },
                    )),
                },
            ),
            recorded(
                20,
                SessionEvent::TurnStarted {
                    run_id: run_id.clone(),
                    turn_id: turn_2.clone(),
                },
            ),
            recorded(
                21,
                SessionEvent::StepStarted {
                    turn_id: turn_2.clone(),
                    step_id: step_2.clone(),
                },
            ),
            recorded(
                22,
                SessionEvent::InputSubmitted {
                    input_id: input_2.clone(),
                    input: "continue".to_owned(),
                    origin: InputOrigin::Queue,
                },
            ),
            recorded(
                23,
                SessionEvent::InputAttached {
                    input_id: input_2,
                    step_id: step_2.clone(),
                    items: vec![input_item("continue")],
                },
            ),
            recorded(
                24,
                SessionEvent::RequestSnapshot {
                    request_id: request_2.clone(),
                    step_id: step_2.clone(),
                    reason: RequestHeaderReason::Resume,
                    model: "deepseek-v4".to_owned(),
                    instructions: Some("be precise".to_owned()),
                    tools: Vec::new(),
                    reasoning_effort: None,
                    max_output_tokens: Some(4_096),
                    session_config: SessionConfig::default(),
                },
            ),
            recorded(
                25,
                SessionEvent::ModelRequestStarted {
                    request_id: request_2.clone(),
                },
            ),
            recorded(
                26,
                SessionEvent::AssistantChunk {
                    request_id: request_2.clone(),
                    chunk: AssistantChunk::OutputTextDelta {
                        delta: "partial".to_owned(),
                    },
                },
            ),
            recorded(
                27,
                SessionEvent::ModelRequestFailed {
                    request_id: request_2,
                    error: "connection closed".to_owned(),
                },
            ),
            recorded(
                28,
                SessionEvent::StepTerminated {
                    step_id: step_2,
                    outcome: StepOutcome::Failed,
                    error: Some("connection closed".to_owned()),
                },
            ),
            recorded(
                29,
                SessionEvent::TurnTerminated {
                    turn_id: turn_2,
                    reason: TurnEndReason::Failed,
                },
            ),
            recorded(
                30,
                SessionEvent::RunTerminated {
                    run_id,
                    outcome: RunOutcome::Failed,
                    error: Some("connection closed".to_owned()),
                },
            ),
        ];
        for event in &mut events {
            let transaction = match event.seq {
                0..=2 => "tx-lifecycle-1",
                3..=4 => "tx-input-1",
                5..=10 => "tx-response-1",
                // A completed turn and the next turn boundary are one durable
                // transition. Compaction is part of that transition, so a
                // crash cannot leave a live run between turns.
                11..=21 => "tx-next-turn-1",
                22..=23 => "tx-input-2",
                24..=27 => "tx-response-2",
                28..=30 => "tx-terminal-2",
                _ => unreachable!("fixture sequence is bounded"),
            };
            event.tx_id = TxId::from(transaction);
        }
        events
    }

    #[test]
    fn golden_projection_preserves_dsh_semantics() {
        let events = fixture();
        kcastle_agent::SessionMachine::from_events(&events)
            .expect("golden input must itself be a committed v2 log");
        let document = SessionDocument::from_events(events).unwrap();
        let trajectory = document
            .trajectory()
            .into_iter()
            .map(|item| {
                format!(
                    "{:?}|{:?}|{}|{}|{:?}|{:?}",
                    item.kind,
                    item.lane,
                    item.title,
                    item.text,
                    item.status,
                    item.timing.duration_ns()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            trajectory,
            "System|Input|Initial System Prompt|deepseek-v4 · 0 tools|Completed|Some(0)\n\
             User|Input|User|evaluate this|Completed|Some(0)\n\
             Assistant|Model|Assistant|hello|Completed|Some(3000000)\n\
             Tool|Tools|shell|ok|Completed|Some(8000000)\n\
             Compaction|Model|Compaction|compacted|Completed|Some(1000000)\n\
             User|Input|User|continue|Completed|Some(0)\n\
             Assistant|Model|Assistant|partial|Failed|None"
        );

        let conversation = document
            .conversation()
            .into_iter()
            .map(|item| format!("{:?}|{}|{:?}", item.role, item.text, item.status))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            conversation,
            "User|evaluate this|Completed\n\
             Assistant|hello|Completed\n\
             Tool|ok|Completed\n\
             Notice|compacted|Completed\n\
             User|continue|Completed\n\
             Assistant|partial|Failed"
        );
    }

    #[test]
    fn completion_replaces_stream_draft_and_tool_delta_refreshes_segment_payload() {
        let mut events = fixture();
        events[7].event = SessionEvent::AssistantChunk {
            request_id: RequestId::from("request-1"),
            chunk: AssistantChunk::OutputTextDelta {
                delta: "stream draft".to_owned(),
            },
        };
        events[8].event = SessionEvent::AssistantChunk {
            request_id: RequestId::from("request-1"),
            chunk: AssistantChunk::ToolCallDelta {
                call_id: CallId::from("call-1"),
                name: Some("streamed-shell".to_owned()),
                arguments_delta: "{\"command\":\"streamed\"}".to_owned(),
            },
        };

        let mut document = SessionDocument::default();
        document.apply_batch(events[..9].to_vec()).unwrap();
        let streamed = document
            .conversation()
            .into_iter()
            .find(|item| item.role == ConversationRole::Assistant)
            .expect("streamed assistant segment is visible");
        assert_eq!(streamed.text, "stream draft");
        assert_eq!(
            streamed.payload,
            Some("streamed-shell {\"command\":\"streamed\"}")
        );

        document.apply_batch(vec![events[9].clone()]).unwrap();
        let completed = document
            .conversation()
            .into_iter()
            .find(|item| item.role == ConversationRole::Assistant)
            .expect("completed assistant segment is visible");
        assert_eq!(completed.text, "hello");
        assert_eq!(completed.payload, Some("shell {\"command\":\"true\"}"));
    }

    #[test]
    fn completion_uses_provider_item_order_instead_of_stream_order() {
        let request_id = RequestId::from("request-1");
        let mut events = fixture();
        events[7].event = SessionEvent::AssistantChunk {
            request_id: request_id.clone(),
            chunk: AssistantChunk::OutputTextDelta {
                delta: "streamed output first".to_owned(),
            },
        };
        events[8].event = SessionEvent::AssistantChunk {
            request_id: request_id.clone(),
            chunk: AssistantChunk::ReasoningTextDelta {
                delta: "streamed reasoning second".to_owned(),
            },
        };
        let reasoning = serde_json::from_value::<InputItem>(serde_json::json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "canonical reasoning"}]
        }))
        .unwrap();
        let output = serde_json::from_value::<InputItem>(serde_json::json!({
            "type": "message",
            "id": "msg-canonical",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "canonical output", "annotations": []}]
        }))
        .unwrap();
        events[9].event = SessionEvent::AssistantCompleted {
            request_id,
            items: vec![reasoning, output],
            response: response("response-1", usage()),
        };

        let document = SessionDocument::from_events(events).unwrap();
        let assistant = document
            .conversation()
            .into_iter()
            .filter(|item| {
                matches!(item.id, ConversationItemId::ResponseSegment { request_id, .. } if request_id.as_str() == "request-1")
            })
            .map(|item| (item.role, item.text.to_owned()))
            .collect::<Vec<_>>();
        assert_eq!(
            assistant,
            vec![
                (
                    ConversationRole::Reasoning,
                    "canonical reasoning".to_owned()
                ),
                (ConversationRole::Assistant, "canonical output".to_owned()),
            ]
        );
    }

    #[test]
    fn completed_tool_payload_preserves_provider_declaration_order() {
        let call = |call_id: &str, name: &str| {
            serde_json::from_value::<InputItem>(serde_json::json!({
                "type": "function_call",
                "arguments": "{}",
                "call_id": call_id,
                "name": name
            }))
            .unwrap()
        };
        let items = vec![call("call-z", "first"), call("call-a", "second")];
        let calls = tool_calls_from_items(&items);
        let order = tool_call_order_from_items(&items);
        assert_eq!(format_tool_payload(&calls, &order), "first {}\nsecond {}");
    }

    #[test]
    fn incremental_stats_equal_a_full_scan_and_exclude_compaction_usage() {
        let mut document = SessionDocument::default();
        for event in fixture() {
            document.apply_batch(vec![event]).unwrap();
            assert_eq!(document.stats(), document.stats_from_full_scan());
        }
        assert_eq!(
            document.stats(),
            SessionStats {
                turns: 2,
                steps: 2,
                llm_ns: 3_000_000,
                tool_ns: 8_000_000,
                ttft_ns: 1_000_000,
                ttft_samples: 1,
                decode_ns: 2_000_000,
                decode_tokens: 20,
                uncached_input_tokens: 20,
                cache_read_input_tokens: 80,
                cache_write_input_tokens: 3,
                output_tokens: 15,
                reasoning_output_tokens: 5,
            }
        );
        assert_eq!(document.stats().input_tokens(), 103);
        assert_eq!(document.stats().total_output_tokens(), 20);
        let tool = document
            .trajectory()
            .into_iter()
            .find(|item| item.kind == TrajectoryKind::Tool)
            .unwrap();
        assert_eq!(tool.title, "shell");
        assert_eq!(tool.payload, Some("{\"command\":\"true\"}"));
        assert_eq!(
            tool.timing.started.as_ref().map(EventTimeRef::wall_time_ms),
            Some(7)
        );
        assert_eq!(
            tool.timing
                .requested
                .as_ref()
                .map(EventTimeRef::wall_time_ms),
            Some(10)
        );
        assert_eq!(
            tool.timing
                .authorization_resolved
                .as_ref()
                .map(EventTimeRef::wall_time_ms),
            Some(11)
        );
        assert_eq!(tool.timing.duration_ns(), Some(8_000_000));
        assert_eq!(tool.timing.execution_ns(), Some(1_000_000));
        assert_eq!(tool.timing.request_registration_ns(), Some(3_000_000));
        assert_eq!(tool.timing.authorization_ns(), Some(1_000_000));
        assert_eq!(tool.timing.dispatch_ns(), Some(1_000_000));
        assert_eq!(tool.timing.runner_start_ns(), Some(1_000_000));
        assert_eq!(tool.timing.pre_execution_ns(), Some(6_000_000));
        assert_eq!(tool.timing.post_execution_ns(), Some(1_000_000));
    }

    #[test]
    fn pre_request_compaction_gap_is_excluded_from_llm_duration_and_ttft() {
        let mut events = fixture();
        // Automatic compaction runs after StepStarted and before the request
        // snapshot. Model timings must therefore be invariant to this gap.
        for event in events.iter_mut().filter(|event| event.seq >= 5) {
            event.time.wall_time_ms = event.time.wall_time_ms.saturating_add(10_000);
            event.time.monotonic_ns = event.time.monotonic_ns.saturating_add(10_000_000_000);
        }

        let document = SessionDocument::from_events(events).unwrap();
        let assistant = document
            .trajectory()
            .into_iter()
            .find(|item| {
                item.kind == TrajectoryKind::Assistant && item.status == ItemStatus::Completed
            })
            .unwrap();

        assert_eq!(assistant.timing.duration_ns(), Some(3_000_000));
        assert_eq!(assistant.timing.ttft_ns(), Some(1_000_000));
        assert_eq!(
            assistant
                .timing
                .started
                .as_ref()
                .map(EventTimeRef::wall_time_ms),
            Some(10_006)
        );
        assert_eq!(document.stats().llm_ns, 3_000_000);
        assert_eq!(document.stats().ttft_ns, 1_000_000);
    }

    #[test]
    fn initial_system_is_semantically_first_and_unchanged_resume_is_omitted() {
        let document = SessionDocument::from_events(fixture()).unwrap();
        let trajectory = document.trajectory();
        assert_eq!(
            trajectory.first().map(|item| item.kind),
            Some(TrajectoryKind::System)
        );
        assert_eq!(
            trajectory
                .iter()
                .filter(|item| item.kind == TrajectoryKind::System)
                .count(),
            1
        );
        assert_eq!(trajectory[0].title, "Initial System Prompt");
        assert_eq!(trajectory[0].turn_id, None);
        assert_eq!(trajectory[0].step_id, None);
    }

    #[test]
    fn partial_assistant_is_visible_but_has_no_completed_timing() {
        let document = SessionDocument::from_events(fixture()).unwrap();
        let partial = document
            .trajectory()
            .into_iter()
            .find(|item| item.text == "partial")
            .unwrap();
        assert_eq!(partial.status, ItemStatus::Failed);
        assert_eq!(partial.timing.duration_ns(), None);
        assert_eq!(partial.timing.ttft_ns(), Some(1_000_000));
    }

    #[test]
    fn raw_details_are_derived_from_the_same_stable_entity() {
        let document = SessionDocument::from_events(fixture()).unwrap();
        let id = TrajectoryItemId::Assistant(RequestId::from("request-1"));
        let details = document.details(&id).unwrap();
        assert_eq!(details.text, "hello");
        let raw = document.details_raw(&id).unwrap();
        assert!(raw.contains("request_snapshot"));
        assert!(raw.contains("assistant_chunk"));
        assert!(raw.contains("assistant_completed"));
    }

    #[test]
    fn invalid_plan_is_atomic_and_a_plan_cannot_be_applied_stale() {
        let mut document = SessionDocument::default();
        let before = document.clone();
        let error = document
            .apply_batch(vec![recorded(
                0,
                SessionEvent::TurnStarted {
                    run_id: RunId::from("missing"),
                    turn_id: TurnId::from("turn"),
                },
            )])
            .unwrap_err();
        assert!(matches!(
            error,
            ProjectionError::Missing { kind: "run", .. }
        ));
        assert_eq!(document, before);

        let stale = document
            .plan_batch(vec![recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: RunId::from("planned"),
                },
            )])
            .unwrap();
        document
            .apply_batch(vec![recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: RunId::from("other"),
                },
            )])
            .unwrap();
        assert!(matches!(
            document.apply_planned(stale),
            Err(ProjectionError::StalePlan { .. })
        ));
    }

    #[test]
    fn live_single_event_projection_equals_replay_projection() {
        let events = fixture();
        let replayed = SessionDocument::from_events(events.clone()).unwrap();
        let mut live = SessionDocument::default();
        for event in events {
            live.apply_batch(vec![event]).unwrap();
        }
        assert_eq!(live, replayed);
        assert_eq!(live.conversation(), replayed.conversation());
        assert_eq!(live.trajectory(), replayed.trajectory());
        assert_eq!(live.stats(), replayed.stats());
    }

    #[test]
    fn rejected_batch_does_not_leak_partial_selectors() {
        let mut document = SessionDocument::default();
        let rejected = vec![
            recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: RunId::from("would-have-been-visible"),
                },
            ),
            recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id: RunId::from("missing"),
                    turn_id: TurnId::from("invalid-turn"),
                },
            ),
        ];
        assert!(document.apply_batch(rejected).is_err());
        assert_eq!(document.cursor(), EventCursor::default());
        assert!(document.conversation().is_empty());
        assert!(document.trajectory().is_empty());
        assert_eq!(document.stats(), SessionStats::default());
        assert_eq!(document.graph(), &ExecutionGraph::default());
    }

    #[test]
    fn projection_delta_separates_content_stats_and_geometry_invalidation() {
        let events = fixture();
        let mut document = SessionDocument::default();
        document.apply_batch(events[..7].to_vec()).unwrap();

        let first_token = document.apply_batch(vec![events[7].clone()]).unwrap();
        assert!(!first_token.conversation_order_changed);
        assert!(first_token.trajectory_order_changed);
        assert!(first_token.geometry_changed);
        assert!(!first_token.stats_changed);

        let usage = document.apply_batch(vec![events[8].clone()]).unwrap();
        assert!(!usage.conversation_order_changed);
        assert!(!usage.trajectory_order_changed);
        assert!(!usage.geometry_changed);
        assert!(usage.stats_changed);
        assert_eq!(
            usage.changed_trajectory,
            vec![TrajectoryItemId::Assistant(RequestId::from("request-1"))]
        );

        let completed = document.apply_batch(vec![events[9].clone()]).unwrap();
        assert!(completed.geometry_changed);
        assert!(completed.stats_changed);
        assert!(!completed.trajectory_order_changed);
    }

    proptest! {
        #[test]
        fn arbitrary_batching_equals_full_replay(boundaries in prop::collection::vec(any::<bool>(), 29)) {
            let events = fixture();
            let expected = SessionDocument::from_events(events.clone()).unwrap();
            let mut actual = SessionDocument::default();
            let mut batch_start = 0;
            for (index, boundary) in boundaries.into_iter().enumerate() {
                if boundary {
                    actual.apply_batch(events[batch_start..=index].to_vec()).unwrap();
                    batch_start = index + 1;
                }
            }
            actual.apply_batch(events[batch_start..].to_vec()).unwrap();
            prop_assert_eq!(actual, expected);
        }
    }
}
