use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use gpui_kit::{Context, Window};
use kcastle_agent::{
    Agent, AgentEvent, CommitReceipt, Model, ReasoningEffort, RunControl, RunFailure, Session,
    SessionConfig, SessionInfo,
};

use crate::agent_config::{ConfiguredModel, initial_session_title};
use crate::domain::session_document::SessionDocument;
use crate::domain::{ApprovalState, RunId, SessionView};
use crate::settings::EnterBehavior;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SessionRuntimeStatus {
    Idle,
    Creating,
    Configuring,
    Running,
    Settling,
    Failed(RunFailure),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeOperation {
    StartRun,
    SubmitDuringRun,
    Configure,
    Rename,
    ChangePermissionDuringRun,
}

enum ModelUpdate {
    Keep,
    Replace(Box<Model>),
}

impl ModelUpdate {
    fn apply(self, agent: &mut Agent) {
        if let Self::Replace(model) = self {
            agent.set_model(*model);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RunTerminal {
    Finished,
    Aborted,
    Failed(RunFailure),
}

/// The only authority for runtime lifecycle transitions.
///
/// In particular, a terminal event only moves a run to `Settling`. The runtime cannot become idle
/// until `ActiveAgent::finish` returns ownership of the agent.
#[derive(Clone, Debug)]
struct RuntimeLifecycle {
    status: SessionRuntimeStatus,
    terminal: Option<RunTerminal>,
}

impl Default for RuntimeLifecycle {
    fn default() -> Self {
        Self {
            status: SessionRuntimeStatus::Idle,
            terminal: None,
        }
    }
}

impl RuntimeLifecycle {
    fn status(&self) -> &SessionRuntimeStatus {
        &self.status
    }

    fn allows(&self, operation: RuntimeOperation) -> bool {
        use RuntimeOperation::{
            ChangePermissionDuringRun, Configure, Rename, StartRun, SubmitDuringRun,
        };
        match &self.status {
            SessionRuntimeStatus::Idle | SessionRuntimeStatus::Failed(_) => {
                matches!(operation, StartRun | Configure | Rename)
            }
            SessionRuntimeStatus::Running => {
                matches!(operation, SubmitDuringRun | ChangePermissionDuringRun)
            }
            SessionRuntimeStatus::Creating
            | SessionRuntimeStatus::Configuring
            | SessionRuntimeStatus::Settling => false,
        }
    }

    fn is_active(&self) -> bool {
        matches!(
            self.status,
            SessionRuntimeStatus::Creating
                | SessionRuntimeStatus::Configuring
                | SessionRuntimeStatus::Running
                | SessionRuntimeStatus::Settling
        )
    }

    fn begin_creating(&mut self) -> bool {
        if !self.allows(RuntimeOperation::StartRun) {
            return false;
        }
        self.terminal = None;
        self.status = SessionRuntimeStatus::Creating;
        true
    }

    fn begin_running(&mut self) -> bool {
        if !matches!(
            self.status,
            SessionRuntimeStatus::Idle
                | SessionRuntimeStatus::Failed(_)
                | SessionRuntimeStatus::Creating
        ) {
            return false;
        }
        self.terminal = None;
        self.status = SessionRuntimeStatus::Running;
        true
    }

    fn begin_configuring(&mut self) -> bool {
        if !self.allows(RuntimeOperation::Configure) {
            return false;
        }
        self.status = SessionRuntimeStatus::Configuring;
        true
    }

    fn begin_settlement_config(&mut self) {
        debug_assert!(matches!(self.status, SessionRuntimeStatus::Settling));
        self.status = SessionRuntimeStatus::Configuring;
    }

    fn complete_config(&mut self, status: SessionRuntimeStatus) {
        debug_assert!(matches!(self.status, SessionRuntimeStatus::Configuring));
        debug_assert!(matches!(
            status,
            SessionRuntimeStatus::Idle | SessionRuntimeStatus::Failed(_)
        ));
        self.terminal = None;
        self.status = status;
    }

    fn fail(&mut self, failure: impl Into<RunFailure>) {
        let failure = failure.into();
        match self.status {
            SessionRuntimeStatus::Running | SessionRuntimeStatus::Settling => {
                self.observe_terminal(RunTerminal::Failed(failure));
            }
            _ => self.status = SessionRuntimeStatus::Failed(failure),
        }
    }

    /// Returns true only for the first successful completion event, so completion counters cannot
    /// be advanced by duplicate or stale terminal notifications.
    fn observe_terminal(&mut self, terminal: RunTerminal) -> bool {
        match self.status {
            SessionRuntimeStatus::Running => {
                let completed = matches!(terminal, RunTerminal::Finished);
                self.terminal = Some(terminal);
                self.status = SessionRuntimeStatus::Settling;
                completed
            }
            SessionRuntimeStatus::Settling => {
                if matches!(terminal, RunTerminal::Failed(_)) {
                    self.terminal = Some(terminal);
                }
                false
            }
            _ => false,
        }
    }

    fn status_after_join(&self) -> SessionRuntimeStatus {
        debug_assert!(matches!(self.status, SessionRuntimeStatus::Settling));
        match &self.terminal {
            Some(RunTerminal::Finished | RunTerminal::Aborted) => SessionRuntimeStatus::Idle,
            Some(RunTerminal::Failed(error)) => SessionRuntimeStatus::Failed(error.clone()),
            None => SessionRuntimeStatus::Failed("run ended without a terminal event".into()),
        }
    }

    fn stream_closed(&mut self) {
        if matches!(self.status, SessionRuntimeStatus::Running) {
            self.status = SessionRuntimeStatus::Settling;
        }
    }

    fn complete_join(&mut self, status: SessionRuntimeStatus) {
        debug_assert!(matches!(self.status, SessionRuntimeStatus::Settling));
        self.terminal = None;
        self.status = status;
    }

    fn join_failed(&mut self, error: impl Into<RunFailure>) {
        self.terminal = None;
        self.status = SessionRuntimeStatus::Failed(error.into());
    }

    fn complete_local_configuration(&mut self) {
        debug_assert!(matches!(
            self.status,
            SessionRuntimeStatus::Idle | SessionRuntimeStatus::Failed(_)
        ));
        self.terminal = None;
        self.status = SessionRuntimeStatus::Idle;
    }
}

#[derive(Default)]
struct ApprovalQueue(VecDeque<ApprovalState>);

impl ApprovalQueue {
    fn front(&self) -> Option<&ApprovalState> {
        self.0.front()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn push(&mut self, approval: ApprovalState) {
        if self
            .0
            .iter()
            .any(|existing| existing.call_id == approval.call_id)
        {
            return;
        }
        self.0.push_back(approval);
    }

    fn is_front(&self, call_id: &str) -> bool {
        self.0
            .front()
            .is_some_and(|approval| approval.call_id == call_id)
    }

    fn pop_front(&mut self) -> Option<ApprovalState> {
        self.0.pop_front()
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRuntimeSnapshot {
    pub(crate) session: SessionInfo,
    pub(crate) view: Arc<SessionView>,
    pub(crate) status: SessionRuntimeStatus,
    pub(crate) approval: Option<ApprovalState>,
    pub(crate) started_at: Option<Instant>,
    pub(crate) allow_all_tools: bool,
    pub(crate) config: SessionConfig,
    pub(crate) active_run: Option<RunId>,
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRuntimeObservation {
    pub(crate) session: SessionInfo,
    pub(crate) status: SessionRuntimeStatus,
    pub(crate) approval_needed: bool,
    pub(crate) completed_runs: u64,
    pub(crate) transcript_updates: u64,
    pub(crate) durable_revision: u64,
    pub(crate) metadata_generation: u64,
}

/// Per-session execution boundary owned by GPUI.
///
/// Durable content has exactly one route: committed events -> `SessionDocument` -> `SessionView`.
/// Transient events may change status and approvals, but never fabricate visible session items.
pub(crate) struct SessionRuntime {
    session: SessionInfo,
    project_id: String,
    sessions_dir: PathBuf,
    agent: Option<Agent>,
    control: Option<RunControl>,
    active_run: Option<RunId>,
    next_run: RunId,
    completed_runs: u64,
    transcript_updates: u64,
    durable_revision: u64,
    metadata_generation: u64,
    document: SessionDocument,
    view: Arc<SessionView>,
    approvals: ApprovalQueue,
    lifecycle: RuntimeLifecycle,
    started_at: Option<Instant>,
    allow_all_tools: bool,
    config: SessionConfig,
    tool_schemas: HashMap<String, String>,
}

impl SessionRuntime {
    pub(crate) fn new(
        agent: Agent,
        project_id: String,
        sessions_dir: PathBuf,
        document: SessionDocument,
        config: SessionConfig,
    ) -> Self {
        let session = agent.session_info().clone();
        let durable_revision = agent.session_revision();
        let tool_schemas = agent
            .tool_schemas()
            .into_iter()
            .filter_map(|schema| {
                let value = serde_json::to_value(schema).ok()?;
                let function = value.get("function").unwrap_or(&value);
                let name = function.get("name")?.as_str()?.to_owned();
                let display = serde_json::to_string_pretty(function).ok()?;
                Some((name, display))
            })
            .collect::<HashMap<_, _>>();
        let view = Arc::new(SessionView::from_document(
            &document,
            &session.title,
            &tool_schemas,
            None,
        ));
        Self {
            session,
            project_id,
            sessions_dir,
            agent: Some(agent),
            control: None,
            active_run: None,
            next_run: RunId::default(),
            completed_runs: 0,
            transcript_updates: 0,
            durable_revision,
            metadata_generation: 0,
            document,
            view,
            approvals: ApprovalQueue::default(),
            lifecycle: RuntimeLifecycle::default(),
            started_at: None,
            allow_all_tools: config.allow_all_tools,
            config,
            tool_schemas,
        }
    }

    pub(crate) fn snapshot(&self) -> SessionRuntimeSnapshot {
        SessionRuntimeSnapshot {
            session: self.session.clone(),
            view: Arc::clone(&self.view),
            status: self.lifecycle.status().clone(),
            approval: self.approvals.front().cloned(),
            started_at: self.started_at,
            allow_all_tools: self.allow_all_tools,
            config: self.config.clone(),
            active_run: self.active_run,
        }
    }

    pub(crate) fn observation(&self) -> SessionRuntimeObservation {
        SessionRuntimeObservation {
            session: self.session.clone(),
            status: self.lifecycle.status().clone(),
            approval_needed: !self.approvals.is_empty(),
            completed_runs: self.completed_runs,
            transcript_updates: self.transcript_updates,
            durable_revision: self.durable_revision,
            metadata_generation: self.metadata_generation,
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.lifecycle.is_active()
    }

    /// Returns true only when this inactive runtime and its idle Agent were loaded from exactly
    /// the same durable snapshot as `session`. Metadata and configuration are not journal events,
    /// so revision equality alone is insufficient for safe cache reuse.
    pub(crate) fn matches_loaded_session(&self, session: &Session) -> bool {
        !self.is_active()
            && self.durable_revision == session.revision()
            && self.session == *session.info()
            && self.agent.as_ref().is_some_and(|agent| {
                agent.session_revision() == session.revision()
                    && agent.session_config() == session.config()
            })
    }

    #[cfg(test)]
    pub(crate) fn mark_failed_for_test(&mut self, message: impl Into<String>) {
        debug_assert!(!self.lifecycle.is_active());
        self.lifecycle.fail(message.into());
    }

    pub(crate) fn set_allow_all_tools(&mut self, allow: bool, cx: &mut Context<Self>) -> bool {
        if allow == self.allow_all_tools && allow == self.config.allow_all_tools {
            return true;
        }
        if self
            .lifecycle
            .allows(RuntimeOperation::ChangePermissionDuringRun)
        {
            self.allow_all_tools = allow;
            if allow {
                let approvals = self
                    .approvals
                    .0
                    .drain(..)
                    .map(|approval| approval.call_id)
                    .collect::<Vec<_>>();
                let Some(control) = self.control.clone() else {
                    self.fail_runtime("run control is unavailable");
                    return false;
                };
                cx.spawn(async move |this, cx| {
                    for call_id in approvals {
                        if let Err(error) = control.approve(call_id).await {
                            let _ = this.update(cx, |runtime, cx| {
                                runtime.fail_runtime(error.to_string());
                                cx.notify();
                            });
                            break;
                        }
                    }
                })
                .detach();
            }
            cx.notify();
            return true;
        }
        if !self.lifecycle.allows(RuntimeOperation::Configure) {
            return false;
        }
        let mut config = self.config.clone();
        config.allow_all_tools = allow;
        self.apply_config(config, ModelUpdate::Keep, cx)
    }

    pub(crate) fn select_model(
        &mut self,
        configured: &ConfiguredModel,
        cx: &mut Context<Self>,
    ) -> bool {
        let mut config = self.config.clone();
        config.model = configured.session_model_config();
        self.apply_config(
            config,
            ModelUpdate::Replace(Box::new(configured.model.clone())),
            cx,
        )
    }

    pub(crate) fn refresh_model(
        &mut self,
        configured: &ConfiguredModel,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.lifecycle.allows(RuntimeOperation::Configure)
            || self.config.model.model_id.as_deref() != Some(&configured.id)
        {
            return false;
        }
        let Some(agent) = self.agent.as_mut() else {
            return false;
        };
        agent.set_model(configured.model.clone());
        self.lifecycle.complete_local_configuration();
        cx.notify();
        true
    }

    pub(crate) fn set_reasoning_effort(
        &mut self,
        effort: ReasoningEffort,
        cx: &mut Context<Self>,
    ) -> bool {
        if self.agent.is_none() {
            return false;
        }
        let mut config = self.config.clone();
        config.model.reasoning_effort = Some(effort);
        self.apply_config(config, ModelUpdate::Keep, cx)
    }

    fn apply_config(
        &mut self,
        config: SessionConfig,
        model_update: ModelUpdate,
        cx: &mut Context<Self>,
    ) -> bool {
        if !self.lifecycle.allows(RuntimeOperation::Configure) || config == self.config {
            return false;
        }
        if self.session.path.as_os_str().is_empty() {
            if let Some(agent) = &mut self.agent {
                model_update.apply(agent);
            }
            self.allow_all_tools = config.allow_all_tools;
            self.config = config;
            self.lifecycle.complete_local_configuration();
            cx.notify();
            return true;
        }
        let Some(mut agent) = self.agent.take() else {
            return false;
        };
        let previous_allow_all_tools = self.allow_all_tools;
        self.allow_all_tools = config.allow_all_tools;
        if !self.lifecycle.begin_configuring() {
            self.agent = Some(agent);
            self.allow_all_tools = previous_allow_all_tools;
            return false;
        }
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = agent.persist_session_config(&config).await;
            let _ = this.update(cx, |runtime, cx| {
                match result {
                    Ok(()) => {
                        runtime.complete_persisted_config(config, model_update, &mut agent);
                    }
                    Err(error) => {
                        if runtime.allow_all_tools == config.allow_all_tools {
                            runtime.allow_all_tools = previous_allow_all_tools;
                        }
                        runtime
                            .lifecycle
                            .complete_config(SessionRuntimeStatus::Failed(
                                error.to_string().into(),
                            ));
                    }
                }
                runtime.agent = Some(agent);
                cx.notify();
            });
        })
        .detach();
        true
    }

    fn complete_persisted_config(
        &mut self,
        config: SessionConfig,
        model_update: ModelUpdate,
        agent: &mut Agent,
    ) {
        model_update.apply(agent);
        self.config = config;
        self.sync_agent_metadata(agent);
        self.metadata_generation = self.metadata_generation.saturating_add(1);
        self.lifecycle.complete_config(SessionRuntimeStatus::Idle);
    }

    pub(crate) fn rename(&mut self, title: String, cx: &mut Context<Self>) -> bool {
        if !self.lifecycle.allows(RuntimeOperation::Rename)
            || self.session.path.as_os_str().is_empty()
        {
            return false;
        }
        let Some(mut agent) = self.agent.take() else {
            return false;
        };
        if !self.lifecycle.begin_configuring() {
            self.agent = Some(agent);
            return false;
        }
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = agent.rename_session(&title).await;
            let _ = this.update(cx, |runtime, cx| {
                match result {
                    Ok(()) => {
                        runtime.session = agent.session_info().clone();
                        runtime.refresh_view();
                        runtime.metadata_generation = runtime.metadata_generation.saturating_add(1);
                        runtime
                            .lifecycle
                            .complete_config(SessionRuntimeStatus::Idle);
                    }
                    Err(error) => runtime
                        .lifecycle
                        .complete_config(SessionRuntimeStatus::Failed(error.to_string().into())),
                }
                runtime.agent = Some(agent);
                cx.notify();
            });
        })
        .detach();
        true
    }

    pub(crate) fn submit(
        &mut self,
        input: String,
        behavior: EnterBehavior,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if input.trim().is_empty() {
            return;
        }
        if let Some(control) = &self.control {
            if !self.lifecycle.allows(RuntimeOperation::SubmitDuringRun) {
                return;
            }
            let control = control.clone();
            cx.spawn_in(window, async move |this, cx| {
                let result = match behavior {
                    EnterBehavior::Steer => control.steer(input).await,
                    EnterBehavior::Queue => control.queue(input).await,
                };
                if let Err(error) = result {
                    let _ = cx.update(|_, app| {
                        this.update(app, |runtime, cx| {
                            runtime.fail_runtime(error.to_string());
                            cx.notify();
                        })
                    });
                }
            })
            .detach();
            cx.notify();
            return;
        }
        if !self.lifecycle.allows(RuntimeOperation::StartRun) {
            return;
        }
        if self.session.path.as_os_str().is_empty() {
            self.create_and_start(input, window, cx);
        } else {
            self.start(input, window, cx);
        }
    }

    fn create_and_start(&mut self, input: String, window: &mut Window, cx: &mut Context<Self>) {
        if !self.lifecycle.begin_creating() {
            return;
        }
        let sessions_dir = self.sessions_dir.clone();
        let project_id = self.project_id.clone();
        let runtime_config = self.config.clone();
        let session_id = self.session.id.clone();
        cx.notify();
        cx.spawn_in(window, async move |this, cx| {
            let title = initial_session_title(&input).unwrap_or_else(|| "Untitled session".into());
            let result = Session::create_named_in_project_with_id(
                sessions_dir,
                project_id,
                runtime_config,
                session_id,
                title,
            )
            .await;
            let _ = cx.update(|window, app| {
                this.update(app, |runtime, cx| match result {
                    Ok(session) => {
                        runtime.session = session.info().clone();
                        if let Some(agent) = &mut runtime.agent {
                            agent.set_session(session);
                        }
                        runtime.start(input, window, cx);
                    }
                    Err(error) => {
                        runtime.lifecycle.fail(error.to_string());
                        cx.notify();
                    }
                })
            });
        })
        .detach();
    }

    fn start(&mut self, input: String, window: &mut Window, cx: &mut Context<Self>) {
        let Some(agent) = self.agent.take() else {
            self.lifecycle.fail("Agent is unavailable");
            cx.notify();
            return;
        };
        if !self.lifecycle.begin_running() {
            self.agent = Some(agent);
            return;
        }
        let run = self.next_run.next();
        self.next_run = run;
        let mut active = agent.start(input);
        self.control = Some(active.control());
        self.active_run = Some(run);
        self.started_at = Some(Instant::now());
        cx.notify();

        cx.spawn_in(window, async move |this, cx| {
            while let Some(event) = active.next_event().await {
                let _ = cx.update(|_, app| {
                    this.update(app, |runtime, cx| {
                        if runtime.active_run == Some(run) {
                            runtime.apply_event(event);
                            cx.notify();
                        }
                    })
                });
            }
            let _ = cx.update(|_, app| {
                this.update(app, |runtime, cx| {
                    if runtime.active_run == Some(run) {
                        runtime.lifecycle.stream_closed();
                        cx.notify();
                    }
                })
            });
            let result = active.finish().await;
            let _ = cx.update(|_, app| {
                this.update(app, |runtime, cx| {
                    if runtime.active_run != Some(run) {
                        return;
                    }
                    runtime.active_run = None;
                    runtime.control = None;
                    runtime.approvals.clear();
                    runtime.started_at = None;
                    match result {
                        Ok(agent) => {
                            runtime.settle_agent_after_run(agent, cx);
                        }
                        Err(error) => {
                            runtime.allow_all_tools = runtime.config.allow_all_tools;
                            runtime.lifecycle.join_failed(error.to_string());
                        }
                    }
                    cx.notify();
                })
            });
        })
        .detach();
    }

    fn settle_agent_after_run(&mut self, mut agent: Agent, cx: &mut Context<Self>) {
        self.sync_agent_metadata(&agent);
        let settled_status = self.lifecycle.status_after_join();
        let Some(config) = deferred_allow_all_config(&self.config, self.allow_all_tools) else {
            self.agent = Some(agent);
            self.lifecycle.complete_join(settled_status);
            return;
        };

        self.lifecycle.begin_settlement_config();
        cx.spawn(async move |this, cx| {
            let result = agent.persist_session_config(&config).await;
            let _ = this.update(cx, |runtime, cx| {
                match result {
                    Ok(()) => {
                        resolve_deferred_allow_all(
                            &mut runtime.config,
                            &mut runtime.allow_all_tools,
                            config,
                            true,
                        );
                        runtime.sync_agent_metadata(&agent);
                        runtime.lifecycle.complete_config(settled_status);
                    }
                    Err(error) => {
                        resolve_deferred_allow_all(
                            &mut runtime.config,
                            &mut runtime.allow_all_tools,
                            config,
                            false,
                        );
                        runtime.lifecycle.complete_config(settlement_config_failure(
                            &settled_status,
                            &error.to_string(),
                        ));
                    }
                }
                runtime.agent = Some(agent);
                cx.notify();
            });
        })
        .detach();
    }

    pub(crate) fn abort(&mut self, cx: &mut Context<Self>) {
        if let Some(control) = &self.control {
            control.abort();
            cx.notify();
        }
    }

    pub(crate) fn decide(&mut self, call_id: String, allow: bool, cx: &mut Context<Self>) {
        if !self.lifecycle.allows(RuntimeOperation::SubmitDuringRun)
            || !self.approvals.is_front(&call_id)
        {
            return;
        }
        let Some(control) = self.control.clone() else {
            self.fail_runtime("run control is unavailable");
            cx.notify();
            return;
        };
        let Some(approval) = self.approvals.pop_front() else {
            self.fail_runtime("approval queue changed before the decision was applied");
            cx.notify();
            return;
        };
        cx.spawn(async move |this, cx| {
            let result = if allow {
                control.approve(call_id).await
            } else {
                control.deny(call_id).await
            };
            if let Err(error) = result {
                let _ = this.update(cx, |runtime, cx| {
                    if runtime.lifecycle.allows(RuntimeOperation::SubmitDuringRun) {
                        runtime.approvals.0.push_front(approval);
                    }
                    runtime.fail_runtime(error.to_string());
                    cx.notify();
                });
            }
        })
        .detach();
        cx.notify();
    }

    fn apply_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::SessionCommitted(receipt) => self.apply_receipt(receipt),
            AgentEvent::ApprovalRequired(call) => {
                if !self.lifecycle.allows(RuntimeOperation::SubmitDuringRun) {
                    return;
                }
                if self.allow_all_tools {
                    if let Some(control) = self.control.clone() {
                        tokio::spawn(async move {
                            let _ = control.approve(call.call_id).await;
                        });
                    } else {
                        self.fail_runtime("run control is unavailable");
                    }
                } else {
                    self.approvals.push(ApprovalState {
                        call_id: call.call_id,
                        name: call.name,
                        arguments: call.arguments,
                    });
                }
            }
            AgentEvent::RunFinished(_) => {
                self.approvals.clear();
                if self.lifecycle.observe_terminal(RunTerminal::Finished) {
                    self.completed_runs = self.completed_runs.saturating_add(1);
                }
            }
            AgentEvent::RunAborted => {
                self.approvals.clear();
                self.lifecycle.observe_terminal(RunTerminal::Aborted);
            }
            AgentEvent::RunFailed(error) => {
                self.approvals.clear();
                self.lifecycle.observe_terminal(RunTerminal::Failed(error));
            }
        }
    }

    fn apply_receipt(&mut self, receipt: CommitReceipt) {
        let previous_revision = self.document.revisions().conversation;
        let committed_revision = receipt.revision;
        let committed_at_ms = receipt.committed_at_ms;
        match self.document.apply_batch(receipt.events) {
            Ok(delta) => {
                if self.document.revisions().conversation != previous_revision {
                    self.transcript_updates = self.transcript_updates.saturating_add(1);
                }
                let next_view = Arc::new(SessionView::after_delta(
                    &self.document,
                    &delta,
                    &self.session.title,
                    &self.tool_schemas,
                    &self.view,
                ));
                self.view = next_view;
                self.durable_revision = committed_revision;
                if committed_at_ms >= 0 {
                    let updated_at = millis_to_seconds(committed_at_ms);
                    self.session.updated_at = self.session.updated_at.max(updated_at);
                }
            }
            Err(error) => {
                self.fail_runtime(format!("committed session projection failed: {error}"));
            }
        }
    }

    fn fail_runtime(&mut self, error: impl Into<RunFailure>) {
        self.lifecycle.fail(error);
        if matches!(self.lifecycle.status(), SessionRuntimeStatus::Settling) {
            self.approvals.clear();
            if let Some(control) = &self.control {
                control.abort();
            }
        }
    }

    fn refresh_view(&mut self) {
        self.view = Arc::new(SessionView::from_document(
            &self.document,
            &self.session.title,
            &self.tool_schemas,
            Some(&self.view),
        ));
    }

    fn sync_agent_metadata(&mut self, agent: &Agent) {
        let title_changed = self.session.title != agent.session_info().title;
        self.session = agent.session_info().clone();
        self.durable_revision = agent.session_revision();
        if title_changed {
            self.refresh_view();
        }
    }
}

impl Drop for SessionRuntime {
    fn drop(&mut self) {
        if let Some(control) = &self.control {
            control.abort();
        }
    }
}

fn deferred_allow_all_config(
    durable: &SessionConfig,
    effective_allow_all_tools: bool,
) -> Option<SessionConfig> {
    if durable.allow_all_tools == effective_allow_all_tools {
        return None;
    }
    let mut requested = durable.clone();
    requested.allow_all_tools = effective_allow_all_tools;
    Some(requested)
}

fn resolve_deferred_allow_all(
    durable: &mut SessionConfig,
    effective_allow_all_tools: &mut bool,
    requested: SessionConfig,
    persisted: bool,
) {
    if persisted {
        *durable = requested;
    }
    *effective_allow_all_tools = durable.allow_all_tools;
}

fn settlement_config_failure(
    settled_status: &SessionRuntimeStatus,
    persistence_error: &str,
) -> SessionRuntimeStatus {
    let message = match settled_status {
        SessionRuntimeStatus::Failed(run_error) => {
            format!(
                "{}; could not save permission setting: {persistence_error}",
                run_error.message()
            )
        }
        _ => format!("could not save permission setting: {persistence_error}"),
    };
    SessionRuntimeStatus::Failed(RunFailure::new(message, false))
}

fn millis_to_seconds(millis: i64) -> u64 {
    u64::try_from(millis.max(0)).unwrap_or_default() / 1_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::AppContext;

    use crate::settings::ProviderModel;

    fn lifecycle(status: SessionRuntimeStatus) -> RuntimeLifecycle {
        RuntimeLifecycle {
            status,
            terminal: None,
        }
    }

    fn runtime() -> SessionRuntime {
        let config = SessionConfig::default();
        let agent = Agent::new(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            Session::memory(),
            ".",
        );
        SessionRuntime::new(
            agent,
            "default".into(),
            PathBuf::from("sessions"),
            SessionDocument::default(),
            config,
        )
    }

    #[gpui::test]
    fn refreshing_a_model_does_not_change_session_config(cx: &mut gpui::TestAppContext) {
        let runtime = cx.new(|_| runtime());
        let configured = ConfiguredModel::new(
            "provider",
            ProviderModel::new("model", "Model", 20_000, None),
            Model::new("refreshed", "new-key", "http://localhost", "model", 20_000),
        );
        cx.update_entity(&runtime, |runtime, _| {
            runtime.config.model.model_id = Some(configured.id.clone());
        });
        let before = cx.read_entity(&runtime, |runtime, _| runtime.config.clone());

        assert!(cx.update_entity(&runtime, |runtime, cx| {
            runtime.refresh_model(&configured, cx)
        }));

        cx.read_entity(&runtime, |runtime, _| {
            assert_eq!(runtime.config, before);
            assert_eq!(runtime.agent.as_ref().unwrap().model().name(), "refreshed");
        });
    }

    fn approval(call_id: &str) -> AgentEvent {
        AgentEvent::ApprovalRequired(
            serde_json::from_value(serde_json::json!({
                "arguments": "{}",
                "call_id": call_id,
                "name": "shell"
            }))
            .expect("approval fixture must deserialize"),
        )
    }

    #[test]
    fn lifecycle_gates_operations_by_state() {
        use RuntimeOperation::{
            ChangePermissionDuringRun, Configure, Rename, StartRun, SubmitDuringRun,
        };

        for status in [
            SessionRuntimeStatus::Idle,
            SessionRuntimeStatus::Failed("retryable".into()),
        ] {
            let lifecycle = lifecycle(status);
            assert!(lifecycle.allows(StartRun));
            assert!(lifecycle.allows(Configure));
            assert!(lifecycle.allows(Rename));
            assert!(!lifecycle.allows(SubmitDuringRun));
            assert!(!lifecycle.allows(ChangePermissionDuringRun));
        }

        let running = lifecycle(SessionRuntimeStatus::Running);
        assert!(running.allows(SubmitDuringRun));
        assert!(running.allows(ChangePermissionDuringRun));
        assert!(!running.allows(StartRun));
        assert!(!running.allows(Configure));
        assert!(!running.allows(Rename));

        for status in [
            SessionRuntimeStatus::Creating,
            SessionRuntimeStatus::Configuring,
            SessionRuntimeStatus::Settling,
        ] {
            let lifecycle = lifecycle(status);
            for operation in [
                StartRun,
                SubmitDuringRun,
                Configure,
                Rename,
                ChangePermissionDuringRun,
            ] {
                assert!(!lifecycle.allows(operation));
            }
        }
    }

    #[test]
    fn terminal_event_cannot_make_the_runtime_idle_before_join() {
        let mut lifecycle = RuntimeLifecycle::default();
        assert!(lifecycle.begin_running());
        assert!(lifecycle.observe_terminal(RunTerminal::Finished));

        assert_eq!(lifecycle.status(), &SessionRuntimeStatus::Settling);
        assert!(lifecycle.is_active());
        assert_eq!(lifecycle.status_after_join(), SessionRuntimeStatus::Idle);
        assert_eq!(lifecycle.status(), &SessionRuntimeStatus::Settling);

        lifecycle.complete_join(SessionRuntimeStatus::Idle);
        assert_eq!(lifecycle.status(), &SessionRuntimeStatus::Idle);
    }

    #[test]
    fn failed_terminal_event_stays_active_until_join_then_surfaces_error() {
        let mut lifecycle = RuntimeLifecycle::default();
        assert!(lifecycle.begin_running());
        assert!(!lifecycle.observe_terminal(RunTerminal::Failed("provider failed".into())));

        assert_eq!(lifecycle.status(), &SessionRuntimeStatus::Settling);
        assert_eq!(
            lifecycle.status_after_join(),
            SessionRuntimeStatus::Failed("provider failed".into())
        );
    }

    #[test]
    fn approvals_are_fifo_deduplicated_and_all_terminal_events_clear_them() {
        for terminal in [
            AgentEvent::RunFinished(kcastle_agent::RunSummary {
                output: "done".into(),
                response_id: "response".into(),
                usage: None,
            }),
            AgentEvent::RunAborted,
            AgentEvent::RunFailed("failed".into()),
        ] {
            let mut runtime = runtime();
            assert!(runtime.lifecycle.begin_running());
            runtime.apply_event(approval("call-1"));
            runtime.apply_event(approval("call-2"));
            runtime.apply_event(approval("call-1"));

            assert_eq!(runtime.approvals.0.len(), 2);
            assert_eq!(runtime.approvals.front().unwrap().call_id, "call-1");
            assert_eq!(runtime.approvals.0[1].call_id, "call-2");

            runtime.apply_event(terminal);
            assert!(runtime.approvals.is_empty());
            assert_eq!(runtime.lifecycle.status(), &SessionRuntimeStatus::Settling);
        }
    }

    #[test]
    fn duplicate_completion_does_not_double_count_a_run() {
        let mut runtime = runtime();
        assert!(runtime.lifecycle.begin_running());
        let completion = AgentEvent::RunFinished(kcastle_agent::RunSummary {
            output: "done".into(),
            response_id: "response".into(),
            usage: None,
        });

        runtime.apply_event(completion.clone());
        runtime.apply_event(completion);

        assert_eq!(runtime.completed_runs, 1);
        assert_eq!(runtime.lifecycle.status(), &SessionRuntimeStatus::Settling);
    }

    #[tokio::test]
    async fn settled_agent_metadata_refreshes_the_materialized_title() {
        let mut runtime = runtime();
        assert_eq!(runtime.view.conversation.title, "New chat");
        let mut agent = runtime.agent.take().expect("idle runtime owns its agent");
        agent.rename_session("First automatic title").await.unwrap();

        runtime.sync_agent_metadata(&agent);
        runtime.agent = Some(agent);

        assert_eq!(runtime.session.title, "First automatic title");
        assert_eq!(runtime.view.conversation.title, "First automatic title");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persisted_model_selection_updates_config_and_agent_together() {
        let session_id = kcastle_agent::SessionId::new();
        let root = std::env::temp_dir().join(format!(
            "kcastle-runtime-config-{}-{session_id}",
            std::process::id()
        ));
        let sessions_dir = root.join("sessions");
        let durable = SessionConfig::default();
        let session = Session::create_in_project_with_id(
            &sessions_dir,
            "project",
            durable.clone(),
            session_id,
        )
        .await
        .unwrap();
        let path = session.info().path.clone();
        let agent = Agent::new(
            Model::new("test", "key", "http://localhost", "model", 10_000),
            "test",
            session,
            &root,
        );
        let mut runtime = SessionRuntime::new(
            agent,
            "project".into(),
            sessions_dir,
            SessionDocument::default(),
            durable,
        );
        // Make a stale metadata mirror deterministic even when create and update share one second.
        runtime.session.updated_at = 0;
        let initial_generation = runtime.metadata_generation;
        let mut agent = runtime.agent.take().unwrap();
        assert!(runtime.lifecycle.begin_configuring());
        let mut requested = SessionConfig {
            allow_all_tools: true,
            ..SessionConfig::default()
        };
        requested.model.model_id = Some("provider/new-model".into());
        let selected_model = Model::new("selected", "key", "http://localhost", "new-model", 20_000);

        agent.persist_session_config(&requested).await.unwrap();
        runtime.complete_persisted_config(
            requested.clone(),
            ModelUpdate::Replace(Box::new(selected_model)),
            &mut agent,
        );
        assert_eq!(agent.model().model(), "new-model");
        runtime.agent = Some(agent);

        let loaded = Session::open_in_project(&path, "project").await.unwrap();
        assert_eq!(loaded.config(), &requested);
        assert_eq!(runtime.config, requested);
        assert_eq!(runtime.session, *loaded.info());
        assert_eq!(runtime.metadata_generation, initial_generation + 1);
        assert!(runtime.matches_loaded_session(&loaded));
        assert_eq!(runtime.lifecycle.status(), &SessionRuntimeStatus::Idle);

        drop(loaded);
        drop(runtime);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn committed_timestamp_is_converted_from_milliseconds_to_session_seconds() {
        let mut runtime = runtime();
        let expected_seconds = runtime.session.updated_at.saturating_add(60);
        let committed_at_ms = i64::try_from(expected_seconds.saturating_mul(1_000)).unwrap();
        let tx_id = kcastle_agent::TxId::from_raw("timestamp-units");
        runtime.apply_receipt(CommitReceipt {
            session_id: runtime.session.id.clone(),
            tx_id: tx_id.clone(),
            base_revision: 0,
            revision: 1,
            request_digest: "timestamp-units".into(),
            committed_at_ms,
            events: vec![kcastle_agent::RecordedEvent {
                seq: 0,
                tx_id,
                time: kcastle_agent::EventTime {
                    wall_time_ms: committed_at_ms,
                    clock_id: "test-clock".into(),
                    monotonic_ns: 0,
                },
                event: kcastle_agent::SessionEvent::InputSubmitted {
                    input_id: kcastle_agent::InputId::from_raw("timestamp-input"),
                    input: "timestamp".into(),
                    origin: kcastle_agent::InputOrigin::Queue,
                },
            }],
        });

        assert_eq!(runtime.session.updated_at, expected_seconds);
    }

    #[test]
    fn deferred_allow_all_persistence_commits_or_rolls_back_atomically() {
        let durable = SessionConfig::default();
        let requested = deferred_allow_all_config(&durable, true).unwrap();
        assert!(requested.allow_all_tools);

        let mut committed = durable.clone();
        let mut committed_effective = true;
        resolve_deferred_allow_all(
            &mut committed,
            &mut committed_effective,
            requested.clone(),
            true,
        );
        assert!(committed.allow_all_tools);
        assert!(committed_effective);
        assert!(deferred_allow_all_config(&committed, committed_effective).is_none());

        let mut rolled_back = durable;
        let mut rolled_back_effective = true;
        resolve_deferred_allow_all(
            &mut rolled_back,
            &mut rolled_back_effective,
            requested,
            false,
        );
        assert!(!rolled_back.allow_all_tools);
        assert!(!rolled_back_effective);
        assert!(deferred_allow_all_config(&rolled_back, rolled_back_effective).is_none());
    }

    #[test]
    fn deferred_config_failure_preserves_the_run_failure_context() {
        assert_eq!(
            settlement_config_failure(
                &SessionRuntimeStatus::Failed("provider failed".into()),
                "disk full",
            ),
            SessionRuntimeStatus::Failed(
                "provider failed; could not save permission setting: disk full".into()
            )
        );
    }
}
