use std::collections::HashSet;
use std::fs::{self, File as StdFile, OpenOptions as StdOpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use async_openai::types::responses::{FunctionCallOutputItemParam, InputItem, Item};
use fs2::FileExt;
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::state::{ResponseMetadata, State, StateEntry};

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("session I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid session: {0}")]
    Invalid(String),
    #[error("session serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("session catalog contains {0} unreadable entries")]
    Catalog(usize),
    #[error("session is busy in another runtime: {0}")]
    Busy(PathBuf),
    #[error("session changed since it was opened: {0}")]
    Stale(PathBuf),
}

pub const DEFAULT_PROJECT_ID: &str = "default";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub allow_all_tools: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn from_legacy_path(path: &Path) -> Self {
        Self(
            path.file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("legacy-session")
                .to_owned(),
        )
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub id: SessionId,
    pub project_id: String,
    pub path: PathBuf,
    pub title: String,
    pub created_at: u64,
}

impl SessionInfo {
    pub fn legacy(path: PathBuf, title: impl Into<String>, created_at: u64) -> Self {
        Self {
            id: SessionId::from_legacy_path(&path),
            project_id: DEFAULT_PROJECT_ID.into(),
            path,
            title: title.into(),
            created_at,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionIssue {
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionCatalog {
    pub sessions: Vec<SessionInfo>,
    pub issues: Vec<SessionIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    pub backup_path: PathBuf,
    pub discarded_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    info: SessionInfo,
    state: State,
    events: Vec<SessionEvent>,
    config: SessionConfig,
    recovery_needed: bool,
}

impl SessionSnapshot {
    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn events(&self) -> &[SessionEvent] {
        &self.events
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn pending_inputs(&self) -> Vec<(String, String, InputMode)> {
        let mut pending = Vec::new();
        for event in &self.events {
            match event {
                SessionEvent::InputAdmitted { id, input, mode } => {
                    pending.push((id.clone(), input.clone(), *mode))
                }
                SessionEvent::InputConsumed { id } => {
                    pending.retain(|(candidate, _, _)| candidate != id);
                }
                _ => {}
            }
        }
        pending
    }

    pub fn recovery_needed(&self) -> bool {
        self.recovery_needed
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionEvent {
    RunStarted {
        input: String,
    },
    ReasoningDelta {
        delta: String,
    },
    TextDelta {
        delta: String,
    },
    ResponseCommitted,
    RunFinished,
    RunAborted,
    RunFailed {
        message: String,
    },
    InputAdmitted {
        id: String,
        input: String,
        mode: InputMode,
    },
    InputConsumed {
        id: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    Steer,
    Queue,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum Record {
    Session {
        title: String,
        created_at: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        session_id: Option<SessionId>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        project_id: Option<String>,
        #[serde(default)]
        config: SessionConfig,
    },
    Title {
        title: String,
    },
    Entry {
        entry: StateEntry,
    },
    Event {
        event: SessionEvent,
    },
    Project {
        project_id: String,
    },
    Config {
        config: SessionConfig,
    },
}

#[derive(Debug)]
pub struct Session {
    info: SessionInfo,
    state: State,
    events: Vec<SessionEvent>,
    config: SessionConfig,
    file: Option<File>,
    recovery: Option<RecoveryReport>,
    writer_lock: Option<WriterLease>,
    needs_project_binding: bool,
    recovery_needed: bool,
}

pub trait StateCommit: Send + Sync {
    fn info(&self) -> &SessionInfo;

    fn prepare<'a>(&'a mut self, state: &'a State) -> BoxFuture<'a, Result<(), SessionError>>;

    fn event<'a>(&'a mut self, event: &'a SessionEvent) -> BoxFuture<'a, Result<(), SessionError>>;

    fn set_config<'a>(
        &'a mut self,
        config: &'a SessionConfig,
    ) -> BoxFuture<'a, Result<(), SessionError>>;

    fn set_initial_title<'a>(
        &'a mut self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<(), SessionError>>;

    fn rename<'a>(&'a mut self, title: &'a str) -> BoxFuture<'a, Result<(), SessionError>>;

    fn commit<'a>(&'a mut self, entry: &'a StateEntry) -> BoxFuture<'a, Result<(), SessionError>>;

    fn release_writer(&mut self);
}

struct SessionCommit {
    info: SessionInfo,
    file: Option<File>,
    _writer_lock: Option<WriterLease>,
    path: Option<PathBuf>,
    expected_events: Vec<SessionEvent>,
    config: SessionConfig,
    needs_project_binding: bool,
}

impl Session {
    pub fn memory() -> Self {
        Self {
            info: SessionInfo {
                id: SessionId::new(),
                project_id: DEFAULT_PROJECT_ID.into(),
                path: PathBuf::new(),
                title: "Untitled session".into(),
                created_at: now_secs(),
            },
            state: State::default(),
            events: Vec::new(),
            config: SessionConfig::default(),
            file: None,
            recovery: None,
            writer_lock: None,
            needs_project_binding: false,
            recovery_needed: false,
        }
    }

    pub async fn create(directory: impl AsRef<Path>) -> Result<Self, SessionError> {
        Self::create_in_project(directory, DEFAULT_PROJECT_ID).await
    }

    pub async fn create_in_project(
        directory: impl AsRef<Path>,
        project_id: impl Into<String>,
    ) -> Result<Self, SessionError> {
        Self::create_in_project_with_config(directory, project_id, SessionConfig::default()).await
    }

    pub async fn create_in_project_with_config(
        directory: impl AsRef<Path>,
        project_id: impl Into<String>,
        config: SessionConfig,
    ) -> Result<Self, SessionError> {
        Self::create_in_project_with_id(directory, project_id, config, SessionId::new()).await
    }

    pub async fn create_in_project_with_id(
        directory: impl AsRef<Path>,
        project_id: impl Into<String>,
        config: SessionConfig,
        id: SessionId,
    ) -> Result<Self, SessionError> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| SessionError::Invalid(error.to_string()))?;
        let path = directory.join(format!("{}-{}.jsonl", now.as_secs(), id.as_str()));
        let writer_lock = Some(acquire_writer_lock(&path)?);
        let info = SessionInfo {
            id,
            project_id: project_id.into(),
            path: path.clone(),
            title: "Untitled session".into(),
            created_at: now.as_secs(),
        };
        let mut file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
            .await?;
        write_record(
            &mut file,
            &Record::Session {
                title: info.title.clone(),
                created_at: info.created_at,
                session_id: Some(info.id.clone()),
                project_id: Some(info.project_id.clone()),
                config: config.clone(),
            },
        )
        .await?;
        Ok(Self {
            info,
            state: State::default(),
            events: Vec::new(),
            config,
            file: Some(file),
            recovery: None,
            writer_lock,
            needs_project_binding: false,
            recovery_needed: false,
        })
    }

    /// Loads a session projection without opening it for writes or repairing it.
    pub fn inspect(path: impl AsRef<Path>) -> Result<SessionSnapshot, SessionError> {
        let parsed = read_session(path.as_ref())?;
        let events = parsed.events.clone();
        let state = State::restore(parsed.entries).map_err(SessionError::Invalid)?;
        Ok(SessionSnapshot {
            info: parsed.info,
            state,
            events,
            config: parsed.config,
            recovery_needed: parsed.torn_tail.is_some() || parsed.append_newline,
        })
    }

    /// Opens a session for browsing. The returned commit port acquires the writer lease lazily
    /// when an Agent actually starts mutating the session.
    pub fn open_readonly(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        Self::open_readonly_inner(path.as_ref(), None)
    }

    pub fn open_readonly_in_project(
        path: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<Self, SessionError> {
        Self::open_readonly_inner(path.as_ref(), Some(project_id))
    }

    fn open_readonly_inner(path: &Path, project_id: Option<&str>) -> Result<Self, SessionError> {
        let path = path.to_path_buf();
        let mut parsed = read_session(&path)?;
        let needs_project_binding = match project_id {
            Some(expected) if parsed.project_explicit && parsed.info.project_id != expected => {
                return Err(SessionError::Invalid(format!(
                    "session belongs to project {} instead of {expected}",
                    parsed.info.project_id
                )));
            }
            Some(expected) if !parsed.project_explicit => {
                parsed.info.project_id = expected.to_owned();
                true
            }
            _ => false,
        };
        let recovery_needed = parsed.torn_tail.is_some() || parsed.append_newline;
        let events = parsed.events.clone();
        let state = State::restore(parsed.entries).map_err(SessionError::Invalid)?;
        Ok(Self {
            info: parsed.info,
            state,
            events,
            config: parsed.config,
            file: None,
            recovery: None,
            writer_lock: None,
            needs_project_binding,
            recovery_needed,
        })
    }

    pub async fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();
        let writer_lock = Some(acquire_writer_lock(&path)?);
        let parsed = read_session(&path)?;
        let events = parsed.events.clone();
        let recovery = if let Some(torn_tail) = parsed.torn_tail.as_ref() {
            Some(repair_torn_tail(&path, parsed.valid_end, torn_tail)?)
        } else {
            None
        };
        if parsed.append_newline {
            StdOpenOptions::new()
                .append(true)
                .open(&path)?
                .write_all(b"\n")?;
        }
        let state = State::restore(parsed.entries).map_err(SessionError::Invalid)?;
        let file = OpenOptions::new().append(true).open(&path).await?;
        let mut session = Self {
            info: parsed.info,
            state,
            events,
            config: parsed.config,
            file: Some(file),
            recovery,
            writer_lock,
            needs_project_binding: false,
            recovery_needed: false,
        };
        let unresolved = session.state.unresolved_tool_call_ids();
        if !unresolved.is_empty() {
            let items = unresolved
                .into_iter()
                .map(|call_id| {
                    InputItem::from(Item::from(FunctionCallOutputItemParam {
                        call_id,
                        output: "Tool execution was interrupted; its side effects are unknown. Do not retry automatically."
                            .into(),
                        id: None,
                        status: None,
                    }))
                })
                .collect();
            session.append_items(items, None).await?;
        }
        Ok(session)
    }

    pub fn list(directory: impl AsRef<Path>) -> Result<Vec<SessionInfo>, SessionError> {
        let catalog = Self::catalog(directory)?;
        if !catalog.issues.is_empty() {
            return Err(SessionError::Catalog(catalog.issues.len()));
        }
        Ok(catalog.sessions)
    }

    pub fn catalog(directory: impl AsRef<Path>) -> Result<SessionCatalog, SessionError> {
        Self::catalog_inner(directory.as_ref(), None)
    }

    pub fn catalog_in_project(
        directory: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<SessionCatalog, SessionError> {
        Self::catalog_inner(directory.as_ref(), Some(project_id))
    }

    fn catalog_inner(
        directory: &Path,
        project_id: Option<&str>,
    ) -> Result<SessionCatalog, SessionError> {
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                return Ok(SessionCatalog::default());
            }
            Err(error) => return Err(error.into()),
        };
        let mut catalog = SessionCatalog::default();
        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    catalog.issues.push(SessionIssue {
                        path: directory.to_path_buf(),
                        message: error.to_string(),
                    });
                    continue;
                }
            };
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "jsonl") {
                continue;
            }
            match read_session_info(&path) {
                Ok((mut info, explicit)) => {
                    if let Some(expected) = project_id {
                        if explicit && info.project_id != expected {
                            catalog.issues.push(SessionIssue {
                                path,
                                message: format!(
                                    "session belongs to project {} instead of {expected}",
                                    info.project_id
                                ),
                            });
                            continue;
                        }
                        if !explicit {
                            info.project_id = expected.to_owned();
                        }
                    }
                    catalog.sessions.push(info)
                }
                Err(error) => catalog.issues.push(SessionIssue {
                    path,
                    message: error.to_string(),
                }),
            }
        }
        let sessions = &mut catalog.sessions;
        sessions.sort_by_key(|session| std::cmp::Reverse(session.created_at));
        Ok(catalog)
    }

    pub fn delete(session: &SessionInfo) -> Result<(), SessionError> {
        let writer_lock = acquire_writer_lock(&session.path)?;
        fs::remove_file(&session.path)?;
        drop(writer_lock);
        let _ = fs::remove_file(lock_path(&session.path));
        Ok(())
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn events(&self) -> &[SessionEvent] {
        &self.events
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub(crate) fn pending_inputs(&self) -> Vec<(String, String, InputMode)> {
        pending_inputs_from_events(&self.events)
    }

    pub fn take_recovery_report(&mut self) -> Option<RecoveryReport> {
        self.recovery.take()
    }

    pub fn recovery_needed(&self) -> bool {
        self.recovery_needed
    }

    pub fn into_parts(self) -> (State, Box<dyn StateCommit>) {
        let expected_events = self.events;
        (
            self.state,
            Box::new(SessionCommit {
                path: (!self.info.path.as_os_str().is_empty()).then(|| self.info.path.clone()),
                info: self.info,
                file: self.file,
                _writer_lock: self.writer_lock,
                expected_events,
                config: self.config,
                needs_project_binding: self.needs_project_binding,
            }),
        )
    }

    pub async fn set_initial_title(&mut self, message: &str) -> Result<(), SessionError> {
        if self.info.title != "Untitled session" {
            return Ok(());
        }
        let Some(title) = initial_title(message) else {
            return Ok(());
        };
        self.write(&Record::Title {
            title: title.clone(),
        })
        .await?;
        self.info.title = title;
        Ok(())
    }

    pub async fn rename(&mut self, title: &str) -> Result<(), SessionError> {
        let title = normalized_title(title)?;
        self.write(&Record::Title {
            title: title.clone(),
        })
        .await?;
        self.info.title = title;
        Ok(())
    }

    pub async fn append_items(
        &mut self,
        items: Vec<InputItem>,
        response: Option<ResponseMetadata>,
    ) -> Result<StateEntry, SessionError> {
        let entry = self
            .state
            .append_items(items, response)
            .map_err(SessionError::Invalid)?;
        if let Err(error) = self
            .write(&Record::Entry {
                entry: entry.clone(),
            })
            .await
        {
            self.state
                .rollback(entry.id())
                .map_err(SessionError::Invalid)?;
            return Err(error);
        }
        Ok(entry)
    }

    async fn write(&mut self, record: &Record) -> Result<(), SessionError> {
        if let Some(file) = &mut self.file {
            write_record(file, record).await?;
        }
        Ok(())
    }
}

fn pending_inputs_from_events(events: &[SessionEvent]) -> Vec<(String, String, InputMode)> {
    let mut pending = Vec::new();
    for event in events {
        match event {
            SessionEvent::InputAdmitted { id, input, mode } => {
                pending.push((id.clone(), input.clone(), *mode))
            }
            SessionEvent::InputConsumed { id } => {
                pending.retain(|(candidate, _, _)| candidate != id);
            }
            _ => {}
        }
    }
    pending
}

fn lock_path(path: &Path) -> PathBuf {
    path.with_extension("jsonl.lock")
}

#[derive(Debug)]
struct WriterLease {
    file: StdFile,
    key: PathBuf,
}

impl Drop for WriterLease {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
        writer_leases()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.key);
    }
}

fn writer_leases() -> &'static Mutex<HashSet<PathBuf>> {
    static LEASES: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    LEASES.get_or_init(|| Mutex::new(HashSet::new()))
}

fn writer_lease_key(path: &Path) -> PathBuf {
    path.parent()
        .and_then(|parent| fs::canonicalize(parent).ok())
        .and_then(|parent| path.file_name().map(|name| parent.join(name)))
        .unwrap_or_else(|| path.to_path_buf())
}

fn acquire_writer_lock(path: &Path) -> Result<WriterLease, SessionError> {
    let lock_path = lock_path(path);
    let key = writer_lease_key(&lock_path);
    if !writer_leases()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key.clone())
    {
        return Err(SessionError::Busy(path.to_path_buf()));
    }

    let result = (|| {
        let file = StdOpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        file.try_lock_exclusive()
            .map_err(|error| match error.kind() {
                ErrorKind::WouldBlock => SessionError::Busy(path.to_path_buf()),
                _ => SessionError::Io(error),
            })?;
        Ok(WriterLease {
            file,
            key: key.clone(),
        })
    })();
    if result.is_err() {
        writer_leases()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&key);
    }
    result
}

struct ParsedSession {
    info: SessionInfo,
    entries: Vec<StateEntry>,
    events: Vec<SessionEvent>,
    config: SessionConfig,
    project_explicit: bool,
    valid_end: usize,
    torn_tail: Option<Vec<u8>>,
    append_newline: bool,
}

impl StateCommit for SessionCommit {
    fn info(&self) -> &SessionInfo {
        &self.info
    }

    fn prepare<'a>(&'a mut self, state: &'a State) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            if self.file.is_some() {
                return Ok(());
            }
            let Some(path) = self.path.clone() else {
                return Ok(());
            };
            let writer_lock = acquire_writer_lock(&path)?;
            let parsed = read_session(&path)?;
            if parsed.entries != state.entries() {
                return Err(SessionError::Stale(path));
            }
            if parsed.events != self.expected_events {
                return Err(SessionError::Stale(path));
            }
            if parsed.config != self.config {
                return Err(SessionError::Stale(path));
            }
            if parsed.info.id != self.info.id
                || parsed.info.title != self.info.title
                || parsed.info.created_at != self.info.created_at
            {
                return Err(SessionError::Stale(path));
            }
            if parsed.project_explicit {
                if parsed.info.project_id != self.info.project_id {
                    return Err(SessionError::Stale(path));
                }
            } else if !self.needs_project_binding && parsed.info.project_id != self.info.project_id
            {
                return Err(SessionError::Stale(path));
            }
            if let Some(torn_tail) = parsed.torn_tail.as_ref() {
                let _ = repair_torn_tail(&path, parsed.valid_end, torn_tail)?;
            } else if parsed.append_newline {
                StdOpenOptions::new()
                    .append(true)
                    .open(&path)?
                    .write_all(b"\n")?;
            }
            self.file = Some(OpenOptions::new().append(true).open(&path).await?);
            self._writer_lock = Some(writer_lock);
            if self.needs_project_binding {
                write_record(
                    self.file.as_mut().expect("writer was opened"),
                    &Record::Project {
                        project_id: self.info.project_id.clone(),
                    },
                )
                .await?;
                self.needs_project_binding = false;
            }
            Ok(())
        })
    }

    fn event<'a>(&'a mut self, event: &'a SessionEvent) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            if let Some(file) = &mut self.file {
                write_record(
                    file,
                    &Record::Event {
                        event: event.clone(),
                    },
                )
                .await?;
            }
            self.expected_events.push(event.clone());
            Ok(())
        })
    }

    fn set_config<'a>(
        &'a mut self,
        config: &'a SessionConfig,
    ) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            if self.config == *config {
                return Ok(());
            }
            if let Some(file) = &mut self.file {
                write_record(
                    file,
                    &Record::Config {
                        config: config.clone(),
                    },
                )
                .await?;
            }
            self.config = config.clone();
            Ok(())
        })
    }

    fn set_initial_title<'a>(
        &'a mut self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            if self.info.title != "Untitled session" {
                return Ok(());
            }
            let Some(title) = initial_title(message) else {
                return Ok(());
            };
            if let Some(file) = &mut self.file {
                write_record(
                    file,
                    &Record::Title {
                        title: title.clone(),
                    },
                )
                .await?;
            }
            self.info.title = title;
            Ok(())
        })
    }

    fn rename<'a>(&'a mut self, title: &'a str) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            let title = normalized_title(title)?;
            if let Some(file) = &mut self.file {
                write_record(
                    file,
                    &Record::Title {
                        title: title.clone(),
                    },
                )
                .await?;
            }
            self.info.title = title;
            Ok(())
        })
    }

    fn commit<'a>(&'a mut self, entry: &'a StateEntry) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            if let Some(file) = &mut self.file {
                write_record(
                    file,
                    &Record::Entry {
                        entry: entry.clone(),
                    },
                )
                .await?;
            }
            Ok(())
        })
    }

    fn release_writer(&mut self) {
        self.file = None;
        self._writer_lock = None;
    }
}

async fn write_record(file: &mut File, record: &Record) -> Result<(), SessionError> {
    let mut encoded = serde_json::to_vec(record)?;
    encoded.push(b'\n');
    file.write_all(&encoded).await?;
    file.flush().await?;
    Ok(())
}

fn read_session(path: &Path) -> Result<ParsedSession, SessionError> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() {
        return Err(SessionError::Invalid("empty file".into()));
    }
    let mut title = None;
    let mut created_at = None;
    let mut session_id = None;
    let mut project_id = None;
    let mut project_explicit = false;
    let mut entries = Vec::new();
    let mut events = Vec::new();
    let mut config = SessionConfig::default();
    let mut valid_end = 0;
    let mut torn_tail = None;

    for chunk in bytes.split_inclusive(|byte| *byte == b'\n') {
        let line = chunk.strip_suffix(b"\n").unwrap_or(chunk);
        if line.is_empty() {
            valid_end += chunk.len();
            continue;
        }
        let record = match serde_json::from_slice::<Record>(line) {
            Ok(record) => record,
            Err(error)
                if error.classify() != serde_json::error::Category::Data
                    && valid_end + chunk.len() == bytes.len()
                    && !chunk.ends_with(b"\n") =>
            {
                torn_tail = Some(chunk.to_vec());
                break;
            }
            Err(error) => return Err(SessionError::Json(error)),
        };
        match record {
            Record::Session {
                title: value,
                created_at: value_created_at,
                session_id: value_session_id,
                project_id: value_project_id,
                config: value_config,
            } if title.is_none() => {
                title = Some(value);
                created_at = Some(value_created_at);
                session_id = value_session_id;
                project_id = value_project_id;
                project_explicit = project_id.is_some();
                config = value_config;
            }
            Record::Session { .. } => {
                return Err(SessionError::Invalid("duplicate session header".into()));
            }
            Record::Title { title: value } => title = Some(value),
            Record::Entry { entry } => entries.push(entry),
            Record::Event { event } => events.push(event),
            Record::Project { project_id: value } => {
                project_id = Some(value);
                project_explicit = true;
            }
            Record::Config { config: value } => config = value,
        }
        valid_end += chunk.len();
    }

    let title = title.ok_or_else(|| SessionError::Invalid("missing session header".into()))?;
    let created_at =
        created_at.ok_or_else(|| SessionError::Invalid("missing creation time".into()))?;
    Ok(ParsedSession {
        info: SessionInfo {
            id: session_id.unwrap_or_else(|| SessionId::from_legacy_path(path)),
            project_id: project_id.unwrap_or_else(|| DEFAULT_PROJECT_ID.into()),
            path: path.to_path_buf(),
            title,
            created_at,
        },
        entries,
        events,
        config,
        project_explicit,
        valid_end,
        append_newline: torn_tail.is_none() && !bytes.ends_with(b"\n") && valid_end == bytes.len(),
        torn_tail,
    })
}

fn repair_torn_tail(
    path: &Path,
    valid_end: usize,
    torn_tail: &[u8],
) -> Result<RecoveryReport, SessionError> {
    let backup_path =
        path.with_extension(format!("jsonl.recovery-{}-{}", now_secs(), Uuid::new_v4()));
    let mut backup = StdOpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&backup_path)?;
    backup.write_all(torn_tail)?;
    backup.sync_all()?;
    let file = StdOpenOptions::new().write(true).open(path)?;
    file.set_len(valid_end as u64)?;
    file.sync_all()?;
    Ok(RecoveryReport {
        backup_path,
        discarded_bytes: torn_tail.len(),
    })
}

#[derive(Deserialize)]
struct HeaderRecord {
    record: String,
    title: Option<String>,
    created_at: Option<u64>,
    session_id: Option<SessionId>,
    project_id: Option<String>,
}

fn read_session_info(path: &Path) -> Result<(SessionInfo, bool), SessionError> {
    let file = StdOpenOptions::new().read(true).open(path)?;
    let mut title = None;
    let mut created_at = None;
    let mut session_id = None;
    let mut project_id = None;
    let mut project_explicit = false;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let record: HeaderRecord = match serde_json::from_str(&line) {
            Ok(record) => record,
            Err(_) if title.is_some() => break,
            Err(error) => return Err(error.into()),
        };
        match record.record.as_str() {
            "session" => {
                title = record.title;
                created_at = record.created_at;
                session_id = record.session_id;
                project_id = record.project_id;
                project_explicit = project_id.is_some();
            }
            "title" => {
                if let Some(value) = record.title {
                    title = Some(value);
                }
            }
            "project" => {
                if let Some(value) = record.project_id {
                    project_id = Some(value);
                    project_explicit = true;
                }
            }
            "entry" => {}
            _ => {}
        }
    }
    Ok((
        SessionInfo {
            id: session_id.unwrap_or_else(|| SessionId::from_legacy_path(path)),
            project_id: project_id.unwrap_or_else(|| DEFAULT_PROJECT_ID.into()),
            path: path.to_path_buf(),
            title: title.ok_or_else(|| SessionError::Invalid("missing session header".into()))?,
            created_at: created_at
                .ok_or_else(|| SessionError::Invalid("missing creation time".into()))?,
        },
        project_explicit,
    ))
}

fn initial_title(message: &str) -> Option<String> {
    let title = message.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = title.chars().take(80).collect::<String>();
    (!title.is_empty()).then_some(title)
}

fn normalized_title(title: &str) -> Result<String, SessionError> {
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = title.chars().take(80).collect::<String>();
    if title.is_empty() {
        Err(SessionError::Invalid(
            "session title cannot be empty".into(),
        ))
    } else {
        Ok(title)
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;

    use async_openai::types::responses::{EasyInputMessage, FunctionToolCall, InputItem, Item};

    use super::Session;

    #[tokio::test]
    async fn round_trips_and_repairs_a_torn_tail() {
        let directory = test_directory("tail");
        let mut session = Session::create(&directory).await.unwrap();
        session
            .set_initial_title("hello native session")
            .await
            .unwrap();
        session
            .append_items(vec![InputItem::from(EasyInputMessage::from("hello"))], None)
            .await
            .unwrap();
        let path = session.info().path.clone();
        drop(session);

        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(br#"{"record":"entry""#)
            .unwrap();
        let damaged = std::fs::read(&path).unwrap();
        let snapshot = Session::inspect(&path).unwrap();
        assert!(snapshot.recovery_needed());
        assert_eq!(snapshot.info().title, "hello native session");
        assert_eq!(snapshot.state().entries().len(), 1);
        assert_eq!(std::fs::read(&path).unwrap(), damaged);

        let mut session = Session::open(&path).await.unwrap();
        assert_eq!(session.info().title, "hello native session");
        assert_eq!(session.state().entries().len(), 1);
        let recovery = session.take_recovery_report().unwrap();
        assert!(recovery.backup_path.exists());
        assert_eq!(recovery.discarded_bytes, br#"{"record":"entry""#.len());
        drop(session);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn rename_is_persisted_and_rejects_empty_titles() {
        let directory = test_directory("rename");
        let mut session = Session::create(&directory).await.unwrap();
        session
            .append_items(vec![InputItem::from(EasyInputMessage::from("hello"))], None)
            .await
            .unwrap();
        session.rename("  A   clearer title  ").await.unwrap();
        assert_eq!(session.info().title, "A clearer title");
        assert!(session.rename("   ").await.is_err());
        assert_eq!(
            Session::list(&directory).unwrap()[0].title,
            "A clearer title"
        );
        let path = session.info().path.clone();
        drop(session);

        let session = Session::open(path).await.unwrap();
        assert_eq!(session.info().title, "A clearer title");
        drop(session);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn identity_and_project_binding_round_trip() {
        let directory = test_directory("identity");
        let session = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let expected = session.info().clone();
        assert_eq!(expected.project_id, "project-a");
        assert!(!expected.id.as_str().is_empty());
        drop(session);

        let snapshot = Session::inspect(&expected.path).unwrap();
        assert_eq!(snapshot.info(), &expected);
        assert_eq!(Session::catalog(&directory).unwrap().sessions, [expected]);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn draft_identity_is_preserved_when_it_becomes_persistent() {
        let directory = test_directory("draft-identity");
        let draft = Session::memory();
        let id = draft.info().id.clone();
        let session = Session::create_in_project_with_id(
            &directory,
            "project-a",
            super::SessionConfig::default(),
            id.clone(),
        )
        .await
        .unwrap();
        assert_eq!(session.info().id, id);
        assert!(session.info().path.to_string_lossy().contains(id.as_str()));
        drop(session);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn legacy_session_is_bound_to_its_project_on_first_write() {
        let directory = test_directory("legacy-project");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("legacy.jsonl");
        std::fs::write(
            &path,
            b"{\"record\":\"session\",\"title\":\"Legacy\",\"created_at\":1}\n",
        )
        .unwrap();

        let catalog = Session::catalog_in_project(&directory, "project-a").unwrap();
        assert_eq!(catalog.sessions[0].project_id, "project-a");
        let session = Session::open_readonly_in_project(&path, "project-a").unwrap();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        drop(commit);

        assert_eq!(
            Session::inspect(&path).unwrap().info().project_id,
            "project-a"
        );
        assert!(Session::open_readonly_in_project(&path, "project-b").is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn session_configuration_and_pending_inputs_round_trip() {
        let directory = test_directory("config-inbox");
        let config = super::SessionConfig {
            model_id: Some("provider/model".into()),
            reasoning_effort: Some("high".into()),
            allow_all_tools: true,
        };
        let session =
            Session::create_in_project_with_config(&directory, "project-a", config.clone())
                .await
                .unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(&super::SessionEvent::InputAdmitted {
                id: "queued-1".into(),
                input: "continue".into(),
                mode: super::InputMode::Queue,
            })
            .await
            .unwrap();
        drop(commit);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(snapshot.config(), &config);
        assert_eq!(
            snapshot.pending_inputs(),
            [(
                "queued-1".into(),
                "continue".into(),
                super::InputMode::Queue
            )]
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn writer_lease_is_scoped_to_one_session() {
        let directory = test_directory("writer-lease");
        let first = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let first_info = first.info().clone();
        let second = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let second_info = second.info().clone();

        assert!(matches!(
            Session::open(&first_info.path).await,
            Err(super::SessionError::Busy(path)) if path == first_info.path
        ));
        assert!(matches!(
            Session::delete(&first_info),
            Err(super::SessionError::Busy(path)) if path == first_info.path
        ));
        assert_eq!(
            Session::inspect(&first_info.path).unwrap().info(),
            &first_info
        );

        drop(first);
        let reopened = Session::open(&first_info.path).await.unwrap();
        assert_eq!(reopened.info(), &first_info);
        assert_eq!(second.info(), &second_info);

        drop(reopened);
        drop(second);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn readonly_session_browses_while_busy_and_acquires_lazily() {
        let directory = test_directory("readonly-busy");
        let writer = Session::create(&directory).await.unwrap();
        let path = writer.info().path.clone();
        let readonly = Session::open_readonly(&path).unwrap();
        let (state, mut commit) = readonly.into_parts();

        assert!(matches!(
            commit.prepare(&state).await,
            Err(super::SessionError::Busy(busy)) if busy == path
        ));
        drop(writer);
        commit.prepare(&state).await.unwrap();

        drop(commit);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn lazy_writer_rejects_a_stale_readonly_projection() {
        let directory = test_directory("readonly-stale");
        let writer = Session::create(&directory).await.unwrap();
        let path = writer.info().path.clone();
        drop(writer);
        let readonly = Session::open_readonly(&path).unwrap();
        let (state, mut commit) = readonly.into_parts();

        let mut concurrent = Session::open(&path).await.unwrap();
        concurrent
            .append_items(vec![InputItem::from(EasyInputMessage::from("new"))], None)
            .await
            .unwrap();
        drop(concurrent);

        assert!(matches!(
            commit.prepare(&state).await,
            Err(super::SessionError::Stale(stale)) if stale == path
        ));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn lazy_writer_rejects_event_only_changes() {
        let directory = test_directory("readonly-event-stale");
        let writer = Session::create(&directory).await.unwrap();
        let path = writer.info().path.clone();
        drop(writer);
        let readonly = Session::open_readonly(&path).unwrap();
        let (state, mut stale_commit) = readonly.into_parts();

        let concurrent = Session::open(&path).await.unwrap();
        let (concurrent_state, mut concurrent_commit) = concurrent.into_parts();
        concurrent_commit.prepare(&concurrent_state).await.unwrap();
        concurrent_commit
            .event(&super::SessionEvent::RunStarted {
                input: "new".into(),
            })
            .await
            .unwrap();
        drop(concurrent_commit);

        assert!(matches!(
            stale_commit.prepare(&state).await,
            Err(super::SessionError::Stale(stale)) if stale == path
        ));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn visible_stream_events_are_durable_before_a_terminal_record() {
        let directory = test_directory("durable-stream");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(&super::SessionEvent::RunStarted {
                input: "hello".into(),
            })
            .await
            .unwrap();
        commit
            .event(&super::SessionEvent::ReasoningDelta {
                delta: "thinking".into(),
            })
            .await
            .unwrap();
        commit
            .event(&super::SessionEvent::TextDelta {
                delta: "partial answer".into(),
            })
            .await
            .unwrap();
        drop(commit);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(
            snapshot.events(),
            [
                super::SessionEvent::RunStarted {
                    input: "hello".into()
                },
                super::SessionEvent::ReasoningDelta {
                    delta: "thinking".into()
                },
                super::SessionEvent::TextDelta {
                    delta: "partial answer".into()
                },
            ]
        );

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn open_closes_interrupted_tool_calls() {
        let directory = test_directory("tool-recovery");
        let mut session = Session::create(&directory).await.unwrap();
        let call = FunctionToolCall {
            arguments: r#"{"command":"touch maybe"}"#.into(),
            call_id: "call_1".into(),
            namespace: None,
            name: "shell".into(),
            id: None,
            status: None,
        };
        session
            .append_items(vec![InputItem::from(Item::from(call))], None)
            .await
            .unwrap();
        let path = session.info().path.clone();
        drop(session);

        let session = Session::open(&path).await.unwrap();
        assert!(session.state().unresolved_tool_call_ids().is_empty());
        assert!(format!("{:?}", session.state().context()).contains("side effects are unknown"));
        drop(session);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn list_uses_headers_even_when_history_is_damaged() {
        let directory = test_directory("header-list");
        let mut session = Session::create(&directory).await.unwrap();
        session.set_initial_title("listed session").await.unwrap();
        session
            .append_items(vec![InputItem::from(EasyInputMessage::from("hello"))], None)
            .await
            .unwrap();
        let path = session.info().path.clone();
        let expected = session.info().clone();
        drop(session);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"not-json\n")
            .unwrap();

        assert_eq!(Session::list(&directory).unwrap(), [expected]);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn catalog_reports_bad_headers_without_hiding_valid_sessions() {
        let directory = test_directory("catalog-errors");
        let session = Session::create(&directory).await.unwrap();
        let expected = session.info().clone();
        drop(session);
        let damaged = directory.join("damaged.jsonl");
        std::fs::write(&damaged, b"not-json\n").unwrap();

        let catalog = Session::catalog(&directory).unwrap();
        assert_eq!(catalog.sessions, [expected]);
        assert_eq!(catalog.issues.len(), 1);
        assert_eq!(catalog.issues[0].path, damaged);
        assert!(matches!(
            Session::list(&directory),
            Err(super::SessionError::Catalog(1))
        ));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn delete_removes_a_saved_session() {
        let directory = test_directory("delete");
        let session = Session::create(&directory).await.unwrap();
        let info = session.info().clone();
        drop(session);

        Session::delete(&info).unwrap();

        assert!(!info.path.exists());
        assert!(Session::list(&directory).unwrap().is_empty());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn open_rejects_semantically_invalid_final_record() {
        let directory = test_directory("semantic-tail");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        drop(session);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(br#"{"record":"entry","entry":{"type":"items","id":2}}"#)
            .unwrap();
        let damaged = std::fs::read(&path).unwrap();

        assert!(Session::open(&path).await.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), damaged);
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn test_directory(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "kcastle-session-{label}-{}-{nonce}",
            std::process::id()
        ))
    }
}
