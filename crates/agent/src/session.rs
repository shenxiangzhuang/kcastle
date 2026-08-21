use std::collections::{HashMap, HashSet};
use std::fs::{self, File as StdFile, OpenOptions as StdOpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::session_event::{
    AssistantChunk, EventTime, RecordedEvent, SESSION_FORMAT_VERSION, SessionEvent, SurfaceOp,
};
use crate::state::State;

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
    #[error("unsupported session format {found}; expected {expected}")]
    UnsupportedFormat { found: u32, expected: u32 },
}

pub const DEFAULT_PROJECT_ID: &str = "default";
pub const ARCHIVE_DIRECTORY: &str = "archive";

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
    pub search: HashMap<PathBuf, SessionSearchData>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionSearchData {
    pub values: Arc<[String]>,
    pub searchable: Arc<str>,
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
    events: Vec<RecordedEvent>,
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

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn pending_inputs(&self) -> Vec<(String, String, InputMode)> {
        let mut pending = Vec::new();
        for recorded in &self.events {
            match &recorded.event {
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
        format_version: Option<u32>,
        title: String,
        created_at_ms: u64,
        session_id: SessionId,
        project_id: String,
        #[serde(default)]
        config: SessionConfig,
    },
    Title {
        title: String,
    },
    Event {
        #[serde(flatten)]
        event: Box<RecordedEvent>,
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
    events: Vec<RecordedEvent>,
    config: SessionConfig,
    file: Option<File>,
    recovery: Option<RecoveryReport>,
    writer_lock: Option<WriterLease>,
    needs_project_binding: bool,
    recovery_needed: bool,
    observed_stamp: Option<CatalogFileStamp>,
}

pub trait StateCommit: Send + Sync {
    fn info(&self) -> &SessionInfo;

    fn prepare<'a>(&'a mut self, state: &'a State) -> BoxFuture<'a, Result<(), SessionError>>;

    fn event<'a>(
        &'a mut self,
        event: SessionEvent,
        source_event_seqs: Vec<u64>,
        surface_op: Option<SurfaceOp>,
    ) -> BoxFuture<'a, Result<RecordedEvent, SessionError>>;

    fn event_at<'a>(
        &'a mut self,
        time: EventTime,
        event: SessionEvent,
        source_event_seqs: Vec<u64>,
        surface_op: Option<SurfaceOp>,
    ) -> BoxFuture<'a, Result<RecordedEvent, SessionError>> {
        let _ = time;
        self.event(event, source_event_seqs, surface_op)
    }

    fn flush_events(&mut self) -> BoxFuture<'_, Result<(), SessionError>> {
        Box::pin(async { Ok(()) })
    }

    fn set_config<'a>(
        &'a mut self,
        config: &'a SessionConfig,
    ) -> BoxFuture<'a, Result<(), SessionError>>;

    fn set_initial_title<'a>(
        &'a mut self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<(), SessionError>>;

    fn rename<'a>(&'a mut self, title: &'a str) -> BoxFuture<'a, Result<(), SessionError>>;

    fn release_writer(&mut self);
}

struct SessionCommit {
    info: SessionInfo,
    file: Option<File>,
    _writer_lock: Option<WriterLease>,
    path: Option<PathBuf>,
    next_seq: u64,
    expected_event_digest: EventDigest,
    validator: EventValidator,
    config: SessionConfig,
    needs_project_binding: bool,
    clock: EventClock,
    known_stamp: Option<CatalogFileStamp>,
    recovery_needed: bool,
    pending_event_bytes: Vec<u8>,
    last_event_flush: Instant,
}

const EVENT_FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);
const EVENT_FLUSH_BYTES: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EventDigest {
    left: u64,
    right: u64,
}

impl Default for EventDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl EventDigest {
    const LEFT_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const RIGHT_OFFSET: u64 = 0x8422_2325_cbf2_9ce4;

    fn new() -> Self {
        Self {
            left: Self::LEFT_OFFSET,
            right: Self::RIGHT_OFFSET,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
            self.left ^= u64::from(*byte);
            self.left = self.left.wrapping_mul(0x0000_0100_0000_01b3);
            self.right ^= u64::from(*byte);
            self.right = self.right.wrapping_mul(0x9e37_79b1_85eb_ca87);
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EventClock {
    id: String,
    origin: Instant,
}

impl EventClock {
    pub(crate) fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            origin: Instant::now(),
        }
    }

    pub(crate) fn now(&self) -> EventTime {
        EventTime {
            wall_time_ms: now_millis(),
            clock_id: self.id.clone(),
            monotonic_ns: u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX),
        }
    }
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
            observed_stamp: None,
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
                format_version: Some(SESSION_FORMAT_VERSION),
                title: info.title.clone(),
                created_at_ms: now.as_millis().try_into().unwrap_or(u64::MAX),
                session_id: info.id.clone(),
                project_id: info.project_id.clone(),
                config: config.clone(),
            },
        )
        .await?;
        let observed_stamp = Some(catalog_file_stamp(&path)?);
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
            observed_stamp,
        })
    }

    /// Loads a session projection without opening it for writes or repairing it.
    pub fn inspect(path: impl AsRef<Path>) -> Result<SessionSnapshot, SessionError> {
        let parsed = read_session(path.as_ref())?;
        let events = parsed.events.clone();
        let state = state_from_events(&events)?;
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
        let observed_stamp = Some(parsed.stamp.clone());
        let events = parsed.events.clone();
        let state = state_from_events(&events)?;
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
            observed_stamp,
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
        let state = state_from_events(&events)?;
        let file = OpenOptions::new().append(true).open(&path).await?;
        let observed_stamp = Some(catalog_file_stamp(&path)?);
        let session = Self {
            info: parsed.info,
            state,
            events,
            config: parsed.config,
            file: Some(file),
            recovery,
            writer_lock,
            needs_project_binding: false,
            recovery_needed: false,
            observed_stamp,
        };
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
            match catalog_entry(&path) {
                Ok(CachedCatalogValue::Valid {
                    mut info,
                    project_explicit,
                    search,
                }) => {
                    if let Some(expected) = project_id {
                        if project_explicit && info.project_id != expected {
                            catalog.issues.push(SessionIssue {
                                path,
                                message: format!(
                                    "session belongs to project {} instead of {expected}",
                                    info.project_id
                                ),
                            });
                            continue;
                        }
                        if !project_explicit {
                            info.project_id = expected.to_owned();
                        }
                    }
                    catalog.search.insert(path, search);
                    catalog.sessions.push(info)
                }
                Ok(CachedCatalogValue::Invalid(message)) => {
                    catalog.issues.push(SessionIssue { path, message })
                }
                Err(error) => catalog.issues.push(SessionIssue {
                    path,
                    message: error.to_string(),
                }),
            }
        }
        prune_catalog_cache(directory, &catalog.sessions, &catalog.issues);
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

    pub fn archive(session: &SessionInfo) -> Result<SessionInfo, SessionError> {
        let directory = session
            .path
            .parent()
            .ok_or_else(|| SessionError::Invalid("session has no parent directory".into()))?;
        if directory
            .file_name()
            .is_some_and(|name| name == ARCHIVE_DIRECTORY)
        {
            return Err(SessionError::Invalid("session is already archived".into()));
        }
        relocate(session, &directory.join(ARCHIVE_DIRECTORY))
    }

    pub fn restore(session: &SessionInfo) -> Result<SessionInfo, SessionError> {
        let archive = session
            .path
            .parent()
            .filter(|directory| {
                directory
                    .file_name()
                    .is_some_and(|name| name == ARCHIVE_DIRECTORY)
            })
            .ok_or_else(|| SessionError::Invalid("session is not archived".into()))?;
        let directory = archive
            .parent()
            .ok_or_else(|| SessionError::Invalid("archive has no parent directory".into()))?;
        relocate(session, directory)
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn events(&self) -> &[RecordedEvent] {
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
        let next_seq = self.events.len() as u64;
        let expected_event_digest =
            event_digest(&self.events).expect("validated session events must remain serializable");
        let validator = EventValidator::from_events(&self.events)
            .expect("session events were validated when the session was opened");
        let path = (!self.info.path.as_os_str().is_empty()).then(|| self.info.path.clone());
        let known_stamp = self.observed_stamp;
        (
            self.state,
            Box::new(SessionCommit {
                path,
                info: self.info,
                file: self.file,
                _writer_lock: self.writer_lock,
                next_seq,
                expected_event_digest,
                validator,
                config: self.config,
                needs_project_binding: self.needs_project_binding,
                clock: EventClock::new(),
                known_stamp,
                recovery_needed: self.recovery_needed,
                pending_event_bytes: Vec::new(),
                last_event_flush: Instant::now(),
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

    async fn write(&mut self, record: &Record) -> Result<(), SessionError> {
        if let Some(file) = &mut self.file {
            write_record(file, record).await?;
            self.observed_stamp = Some(catalog_file_stamp(&self.info.path)?);
        }
        Ok(())
    }
}

fn relocate(session: &SessionInfo, directory: &Path) -> Result<SessionInfo, SessionError> {
    let file_name = session
        .path
        .file_name()
        .ok_or_else(|| SessionError::Invalid("session has no file name".into()))?;
    fs::create_dir_all(directory)?;
    let target = directory.join(file_name);
    if target.try_exists()? {
        return Err(std::io::Error::new(
            ErrorKind::AlreadyExists,
            format!("session already exists: {}", target.display()),
        )
        .into());
    }
    let writer_lock = acquire_writer_lock(&session.path)?;
    fs::rename(&session.path, &target)?;
    drop(writer_lock);
    let _ = fs::remove_file(lock_path(&session.path));
    let mut relocated = session.clone();
    relocated.path = target;
    Ok(relocated)
}

fn state_from_events(events: &[RecordedEvent]) -> Result<State, SessionError> {
    let mut state = State::default();
    for recorded in events {
        match &recorded.event {
            SessionEvent::UserMessage { items, .. } => {
                state
                    .append_items(items.clone(), None)
                    .map_err(SessionError::Invalid)?;
            }
            SessionEvent::AssistantMessage {
                items, response, ..
            } => {
                state
                    .append_items(items.clone(), Some(response.clone()))
                    .map_err(SessionError::Invalid)?;
            }
            SessionEvent::ToolResult { item, .. } => {
                state
                    .append_items(vec![item.clone()], None)
                    .map_err(SessionError::Invalid)?;
            }
            SessionEvent::CompactionEnd {
                summary,
                first_kept_id,
                tokens_before,
                response,
                outcome: crate::session_event::StepOutcome::Completed,
                ..
            } => {
                state
                    .append_compaction(
                        summary.clone(),
                        *first_kept_id,
                        *tokens_before,
                        response.clone(),
                    )
                    .map_err(SessionError::Invalid)?;
            }
            _ => {}
        }
    }
    Ok(state)
}

fn pending_inputs_from_events(events: &[RecordedEvent]) -> Vec<(String, String, InputMode)> {
    let mut pending = Vec::new();
    for recorded in events {
        match &recorded.event {
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
    events: Vec<RecordedEvent>,
    config: SessionConfig,
    stamp: CatalogFileStamp,
    project_explicit: bool,
    valid_end: usize,
    torn_tail: Option<Vec<u8>>,
    append_newline: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CatalogFileStamp {
    len: u64,
    modified: Option<SystemTime>,
}

#[derive(Clone, Debug)]
enum CachedCatalogValue {
    Valid {
        info: SessionInfo,
        project_explicit: bool,
        search: SessionSearchData,
    },
    Invalid(String),
}

#[derive(Clone, Debug)]
struct CachedCatalogEntry {
    stamp: CatalogFileStamp,
    value: CachedCatalogValue,
    last_used: u64,
    weight: usize,
}

#[derive(Default)]
struct CatalogCache {
    entries: HashMap<PathBuf, CachedCatalogEntry>,
    tick: u64,
    weight: usize,
}

impl CatalogCache {
    const MAX_ENTRIES: usize = 256;
    const MAX_WEIGHT: usize = 64 * 1024 * 1024;

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    fn get(&mut self, path: &Path, stamp: &CatalogFileStamp) -> Option<CachedCatalogValue> {
        let tick = self.next_tick();
        let cached = self
            .entries
            .get_mut(path)
            .filter(|entry| entry.stamp == *stamp)?;
        cached.last_used = tick;
        Some(cached.value.clone())
    }

    fn insert(&mut self, path: PathBuf, stamp: CatalogFileStamp, value: CachedCatalogValue) {
        let weight = catalog_value_weight(&value);
        let last_used = self.next_tick();
        if let Some(previous) = self.entries.remove(&path) {
            self.weight = self.weight.saturating_sub(previous.weight);
        }
        self.weight = self.weight.saturating_add(weight);
        self.entries.insert(
            path,
            CachedCatalogEntry {
                stamp,
                value,
                last_used,
                weight,
            },
        );
        while (self.entries.len() > Self::MAX_ENTRIES || self.weight > Self::MAX_WEIGHT)
            && self.entries.len() > 1
        {
            let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(path, _)| path.clone())
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&oldest) {
                self.weight = self.weight.saturating_sub(removed.weight);
            }
        }
    }

    fn prune_directory(&mut self, directory: &Path, present: &HashSet<&Path>) {
        self.entries
            .retain(|path, _| path.parent() != Some(directory) || present.contains(path.as_path()));
        self.weight = self.entries.values().map(|entry| entry.weight).sum();
    }
}

fn catalog_cache() -> &'static Mutex<CatalogCache> {
    static CACHE: OnceLock<Mutex<CatalogCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(CatalogCache::default()))
}

fn catalog_value_weight(value: &CachedCatalogValue) -> usize {
    match value {
        CachedCatalogValue::Valid { info, search, .. } => {
            info.title.len()
                + info.project_id.len()
                + search.searchable.len()
                + search.values.iter().map(String::capacity).sum::<usize>()
        }
        CachedCatalogValue::Invalid(message) => message.len(),
    }
}

fn catalog_file_stamp(path: &Path) -> Result<CatalogFileStamp, SessionError> {
    let metadata = fs::metadata(path)?;
    Ok(catalog_stamp_from_metadata(&metadata))
}

fn catalog_stamp_from_metadata(metadata: &fs::Metadata) -> CatalogFileStamp {
    CatalogFileStamp {
        len: metadata.len(),
        modified: metadata.modified().ok(),
    }
}

fn read_stable_session_file(path: &Path) -> Result<(Vec<u8>, CatalogFileStamp), SessionError> {
    let mut file = StdOpenOptions::new().read(true).open(path)?;
    let before = catalog_stamp_from_metadata(&file.metadata()?);
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let after = catalog_stamp_from_metadata(&file.metadata()?);
    if before != after || after.len != bytes.len() as u64 {
        return Err(SessionError::Stale(path.to_path_buf()));
    }
    Ok((bytes, after))
}

fn catalog_entry(path: &Path) -> Result<CachedCatalogValue, SessionError> {
    let stamp = catalog_file_stamp(path)?;
    if let Some(cached) = catalog_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path, &stamp)
    {
        return Ok(cached);
    }

    #[cfg(test)]
    record_catalog_parse(path);
    let value = match read_session_with_search(path) {
        Ok((parsed, values)) => {
            let searchable = values.join("\n").to_lowercase().into();
            CachedCatalogValue::Valid {
                info: parsed.info,
                project_explicit: parsed.project_explicit,
                search: SessionSearchData {
                    values: values.into(),
                    searchable,
                },
            }
        }
        Err(SessionError::Io(error)) => return Err(SessionError::Io(error)),
        Err(error) => CachedCatalogValue::Invalid(error.to_string()),
    };
    if catalog_file_stamp(path).ok().as_ref() == Some(&stamp) {
        catalog_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(path.to_path_buf(), stamp, value.clone());
    }
    Ok(value)
}

fn prune_catalog_cache(directory: &Path, sessions: &[SessionInfo], issues: &[SessionIssue]) {
    let present = sessions
        .iter()
        .map(|session| session.path.as_path())
        .chain(issues.iter().map(|issue| issue.path.as_path()))
        .collect::<HashSet<_>>();
    catalog_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .prune_directory(directory, &present);
}

#[cfg(test)]
fn catalog_parse_counts() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static COUNTS: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
fn record_catalog_parse(path: &Path) {
    let mut counts = catalog_parse_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(path.to_path_buf()).or_default() += 1;
}

#[cfg(test)]
fn catalog_parse_count(path: &Path) -> usize {
    catalog_parse_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path)
        .copied()
        .unwrap_or_default()
}

#[cfg(test)]
fn session_parse_counts() -> &'static Mutex<HashMap<PathBuf, usize>> {
    static COUNTS: OnceLock<Mutex<HashMap<PathBuf, usize>>> = OnceLock::new();
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(test)]
fn record_session_parse(path: &Path) {
    let mut counts = session_parse_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(path.to_path_buf()).or_default() += 1;
}

#[cfg(test)]
fn session_parse_count(path: &Path) -> usize {
    session_parse_counts()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path)
        .copied()
        .unwrap_or_default()
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
            let current_stamp = catalog_file_stamp(&path)?;
            if self.recovery_needed || self.known_stamp.as_ref() != Some(&current_stamp) {
                let parsed = read_session(&path)?;
                let parsed_state = state_from_events(&parsed.events)?;
                if parsed_state.entries() != state.entries()
                    || parsed.events.len() as u64 != self.next_seq
                    || event_digest(&parsed.events)? != self.expected_event_digest
                    || parsed.config != self.config
                    || parsed.info.id != self.info.id
                    || parsed.info.title != self.info.title
                    || parsed.info.created_at != self.info.created_at
                {
                    return Err(SessionError::Stale(path));
                }
                if parsed.project_explicit {
                    if parsed.info.project_id != self.info.project_id {
                        return Err(SessionError::Stale(path));
                    }
                } else if !self.needs_project_binding
                    && parsed.info.project_id != self.info.project_id
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
                self.recovery_needed = false;
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
            self.known_stamp = Some(catalog_file_stamp(&path)?);
            self.last_event_flush = Instant::now();
            Ok(())
        })
    }

    fn event<'a>(
        &'a mut self,
        event: SessionEvent,
        source_event_seqs: Vec<u64>,
        surface_op: Option<SurfaceOp>,
    ) -> BoxFuture<'a, Result<RecordedEvent, SessionError>> {
        let time = self.clock.now();
        self.event_at(time, event, source_event_seqs, surface_op)
    }

    fn event_at<'a>(
        &'a mut self,
        time: EventTime,
        event: SessionEvent,
        source_event_seqs: Vec<u64>,
        surface_op: Option<SurfaceOp>,
    ) -> BoxFuture<'a, Result<RecordedEvent, SessionError>> {
        Box::pin(async move {
            let event = RecordedEvent {
                seq: self.next_seq,
                time,
                source_event_seqs,
                surface_op,
                event,
            };
            self.validator.check(&event)?;
            if self.file.is_some() {
                let mut encoded = serde_json::to_vec(&Record::Event {
                    event: Box::new(event.clone()),
                })?;
                encoded.push(b'\n');
                self.pending_event_bytes.extend_from_slice(&encoded);
                let chunk = matches!(event.event, SessionEvent::AssistantChunk { .. });
                if !chunk
                    || self.pending_event_bytes.len() >= EVENT_FLUSH_BYTES
                    || self.last_event_flush.elapsed() >= EVENT_FLUSH_INTERVAL
                {
                    self.flush_pending_events().await?;
                }
            }
            self.validator.apply(&event);
            update_event_digest(&mut self.expected_event_digest, &event)?;
            self.next_seq = self.next_seq.saturating_add(1);
            Ok(event)
        })
    }

    fn flush_events(&mut self) -> BoxFuture<'_, Result<(), SessionError>> {
        Box::pin(self.flush_pending_events())
    }

    fn set_config<'a>(
        &'a mut self,
        config: &'a SessionConfig,
    ) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            if self.config == *config {
                return Ok(());
            }
            self.write_control_record(&Record::Config {
                config: config.clone(),
            })
            .await?;
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
            self.write_control_record(&Record::Title {
                title: title.clone(),
            })
            .await?;
            self.info.title = title;
            Ok(())
        })
    }

    fn rename<'a>(&'a mut self, title: &'a str) -> BoxFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            let title = normalized_title(title)?;
            self.write_control_record(&Record::Title {
                title: title.clone(),
            })
            .await?;
            self.info.title = title;
            Ok(())
        })
    }

    fn release_writer(&mut self) {
        // Normal run termination writes a structural event, which flushes every buffered chunk.
        // A failed write can leave bytes here; dropping them avoids turning the original I/O error
        // into a debug-only panic while releasing the lease.
        self.pending_event_bytes.clear();
        self.file = None;
        self._writer_lock = None;
    }
}

impl SessionCommit {
    async fn flush_pending_events(&mut self) -> Result<(), SessionError> {
        if self.pending_event_bytes.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::take(&mut self.pending_event_bytes);
        let result = async {
            let file = self
                .file
                .as_mut()
                .expect("pending session events require an open writer");
            file.write_all(&bytes).await?;
            file.flush().await?;
            Ok::<(), SessionError>(())
        }
        .await;
        if let Err(error) = result {
            self.pending_event_bytes = bytes;
            return Err(error);
        }
        self.last_event_flush = Instant::now();
        self.refresh_known_stamp()?;
        Ok(())
    }

    async fn write_control_record(&mut self, record: &Record) -> Result<(), SessionError> {
        self.flush_pending_events().await?;
        if let Some(file) = &mut self.file {
            write_record(file, record).await?;
            self.refresh_known_stamp()?;
        }
        Ok(())
    }

    fn refresh_known_stamp(&mut self) -> Result<(), SessionError> {
        if let Some(path) = self.path.as_deref() {
            self.known_stamp = Some(catalog_file_stamp(path)?);
        }
        Ok(())
    }
}

async fn write_record(file: &mut File, record: &Record) -> Result<(), SessionError> {
    let mut encoded = serde_json::to_vec(record)?;
    encoded.push(b'\n');
    file.write_all(&encoded).await?;
    file.flush().await?;
    Ok(())
}

fn event_digest(events: &[RecordedEvent]) -> Result<EventDigest, SessionError> {
    let mut digest = EventDigest::new();
    for event in events {
        update_event_digest(&mut digest, event)?;
    }
    Ok(digest)
}

fn update_event_digest(
    digest: &mut EventDigest,
    event: &RecordedEvent,
) -> Result<(), SessionError> {
    digest.update(&serde_json::to_vec(event)?);
    Ok(())
}

fn read_session(path: &Path) -> Result<ParsedSession, SessionError> {
    read_session_inner(path, false).map(|(parsed, _)| parsed)
}

fn read_session_with_search(path: &Path) -> Result<(ParsedSession, Vec<String>), SessionError> {
    read_session_inner(path, true)
}

fn read_session_inner(
    path: &Path,
    collect_search: bool,
) -> Result<(ParsedSession, Vec<String>), SessionError> {
    #[cfg(test)]
    record_session_parse(path);
    let (bytes, stamp) = read_stable_session_file(path)?;
    if bytes.is_empty() {
        return Err(SessionError::Invalid("empty file".into()));
    }
    let mut title = None;
    let mut created_at_ms = None;
    let mut session_id = None;
    let mut project_id = None;
    let mut project_explicit = false;
    let mut events = Vec::new();
    let mut config = SessionConfig::default();
    let mut valid_end = 0;
    let mut torn_tail = None;
    let mut search = collect_search.then(SessionSearchProjection::default);

    for chunk in bytes.split_inclusive(|byte| *byte == b'\n') {
        let line = chunk.strip_suffix(b"\n").unwrap_or(chunk);
        if line.is_empty() {
            valid_end += chunk.len();
            continue;
        }
        let torn_candidate = valid_end + chunk.len() == bytes.len() && !chunk.ends_with(b"\n");
        let record = if title.is_none() {
            let value = match serde_json::from_slice::<serde_json::Value>(line) {
                Ok(value) => value,
                Err(error)
                    if error.classify() != serde_json::error::Category::Data && torn_candidate =>
                {
                    torn_tail = Some(chunk.to_vec());
                    break;
                }
                Err(error) => return Err(SessionError::Json(error)),
            };
            validate_format_probe(&value)?;
            serde_json::from_value::<Record>(value)?
        } else {
            match serde_json::from_slice::<Record>(line) {
                Ok(record) => record,
                Err(error)
                    if error.classify() != serde_json::error::Category::Data && torn_candidate =>
                {
                    torn_tail = Some(chunk.to_vec());
                    break;
                }
                Err(error) => return Err(SessionError::Json(error)),
            }
        };
        if let Some(search) = &mut search {
            search.record(&record);
        }
        match record {
            Record::Session {
                format_version,
                title: value,
                created_at_ms: value_created_at_ms,
                session_id: value_session_id,
                project_id: value_project_id,
                config: value_config,
            } if title.is_none() => {
                let format_version = format_version.unwrap_or(0);
                if format_version != SESSION_FORMAT_VERSION {
                    return Err(SessionError::UnsupportedFormat {
                        found: format_version,
                        expected: SESSION_FORMAT_VERSION,
                    });
                }
                title = Some(value);
                created_at_ms = Some(value_created_at_ms);
                session_id = Some(value_session_id);
                project_id = Some(value_project_id);
                project_explicit = true;
                config = value_config;
            }
            Record::Session { .. } => {
                return Err(SessionError::Invalid("duplicate session header".into()));
            }
            Record::Title { title: value } => title = Some(value),
            Record::Event { event } => {
                let event = *event;
                let expected = events.len() as u64;
                if event.seq != expected {
                    return Err(SessionError::Invalid(format!(
                        "event seq {} is not contiguous; expected {expected}",
                        event.seq
                    )));
                }
                events.push(event)
            }
            Record::Project { project_id: value } => {
                project_id = Some(value);
                project_explicit = true;
            }
            Record::Config { config: value } => config = value,
        }
        valid_end += chunk.len();
    }

    let title = title.ok_or_else(|| SessionError::Invalid("missing session header".into()))?;
    let created_at_ms =
        created_at_ms.ok_or_else(|| SessionError::Invalid("missing creation time".into()))?;
    validate_events(&events)?;
    Ok((
        ParsedSession {
            info: SessionInfo {
                id: session_id.unwrap_or_else(|| SessionId::from_legacy_path(path)),
                project_id: project_id.unwrap_or_else(|| DEFAULT_PROJECT_ID.into()),
                path: path.to_path_buf(),
                title,
                created_at: created_at_ms / 1_000,
            },
            events,
            config,
            stamp,
            project_explicit,
            valid_end,
            append_newline: torn_tail.is_none()
                && !bytes.ends_with(b"\n")
                && valid_end == bytes.len(),
            torn_tail,
        },
        search
            .map(SessionSearchProjection::finish)
            .unwrap_or_default(),
    ))
}

fn validate_format_probe(probe: &serde_json::Value) -> Result<(), SessionError> {
    if probe.get("record").and_then(serde_json::Value::as_str) != Some("session") {
        return Ok(());
    }
    let found = probe
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    if found == SESSION_FORMAT_VERSION {
        Ok(())
    } else {
        Err(SessionError::UnsupportedFormat {
            found,
            expected: SESSION_FORMAT_VERSION,
        })
    }
}

#[derive(Debug, Default)]
struct PendingSearchValue {
    first_seq: u64,
    value: String,
}

#[derive(Debug, Default)]
struct SessionSearchProjection {
    title: Option<String>,
    values: Vec<String>,
    assistant_chunks: HashMap<(u32, u32), PendingSearchValue>,
    pending_inputs: HashMap<String, PendingSearchValue>,
}

impl SessionSearchProjection {
    fn record(&mut self, record: &Record) {
        match record {
            Record::Session { title, .. } | Record::Title { title } => {
                self.title = Some(title.clone())
            }
            Record::Event { event } => match &event.event {
                SessionEvent::InputAdmitted { id, input, .. } => {
                    self.pending_inputs.insert(
                        id.clone(),
                        PendingSearchValue {
                            first_seq: event.seq,
                            value: input.clone(),
                        },
                    );
                }
                SessionEvent::InputConsumed { id } => {
                    self.pending_inputs.remove(id);
                }
                SessionEvent::UserMessage { items, .. }
                | SessionEvent::AssistantMessage { items, .. } => {
                    if let SessionEvent::AssistantMessage { turn, step, .. } = &event.event {
                        self.assistant_chunks.remove(&(*turn, *step));
                    }
                    collect_serialized_search_values(items, &mut self.values);
                }
                SessionEvent::AssistantChunk { turn, step, chunk } => {
                    let pending =
                        self.assistant_chunks
                            .entry((*turn, *step))
                            .or_insert_with(|| PendingSearchValue {
                                first_seq: event.seq,
                                value: String::new(),
                            });
                    match chunk {
                        AssistantChunk::OutputTextDelta { delta }
                        | AssistantChunk::ReasoningTextDelta { delta }
                        | AssistantChunk::ToolCallArgumentsDelta { delta, .. } => {
                            pending.value.push_str(delta)
                        }
                        AssistantChunk::Usage { .. } => {}
                    }
                }
                SessionEvent::ToolCall {
                    name, arguments, ..
                } => {
                    push_search_value(&mut self.values, name);
                    push_search_value(&mut self.values, arguments);
                }
                SessionEvent::ToolResult { output, .. } => {
                    push_search_value(&mut self.values, output)
                }
                SessionEvent::CompactionEnd { summary, .. } => {
                    push_search_value(&mut self.values, summary)
                }
                SessionEvent::StepEnd {
                    turn,
                    step,
                    outcome,
                    error,
                } => {
                    if *outcome == crate::session_event::StepOutcome::Completed {
                        self.assistant_chunks.remove(&(*turn, *step));
                    }
                    if let Some(error) = error {
                        push_search_value(&mut self.values, error);
                    }
                }
                _ => {}
            },
            Record::Project { .. } | Record::Config { .. } => {}
        }
    }

    fn finish(mut self) -> Vec<String> {
        if let Some(title) = self.title.take() {
            push_search_value(&mut self.values, &title);
        }
        let mut pending = self
            .assistant_chunks
            .into_values()
            .chain(self.pending_inputs.into_values())
            .filter(|pending| !pending.value.trim().is_empty())
            .collect::<Vec<_>>();
        pending.sort_by_key(|pending| pending.first_seq);
        self.values
            .extend(pending.into_iter().map(|pending| pending.value));
        self.values
    }
}

fn collect_serialized_search_values<T: Serialize>(value: &T, output: &mut Vec<String>) {
    if let Ok(value) = serde_json::to_value(value) {
        collect_session_search_values(&value, output);
    }
}

fn push_search_value(output: &mut Vec<String>, value: &str) {
    if !value.trim().is_empty() {
        output.push(value.to_owned());
    }
}

fn collect_session_search_values(value: &serde_json::Value, output: &mut Vec<String>) {
    match value {
        serde_json::Value::String(value) => {
            if !value.trim().is_empty() {
                output.push(value.clone());
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_session_search_values(value, output);
            }
        }
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                if !matches!(
                    key.as_str(),
                    "id" | "call_id" | "created_at" | "model" | "type" | "kind" | "role" | "status"
                ) {
                    collect_session_search_values(value, output);
                }
            }
        }
        _ => {}
    }
}

#[derive(Debug, Clone)]
struct ToolLifecycle {
    turn: u32,
    step: u32,
    execution_started: Option<EventTime>,
    execution_finished: bool,
    result: bool,
}

#[derive(Debug, Default)]
struct EventValidator {
    next_seq: u64,
    active_turn: Option<u32>,
    active_step: Option<(u32, u32)>,
    turns: HashSet<u32>,
    steps: HashSet<(u32, u32)>,
    tools: HashMap<String, ToolLifecycle>,
    active_compaction: Option<String>,
    compactions: HashSet<String>,
    inputs: HashSet<String>,
    completed_inputs: HashSet<String>,
    finalized_assistants: HashSet<(u32, u32)>,
    model_requests: HashSet<(u32, u32)>,
}

impl EventValidator {
    fn from_events(events: &[RecordedEvent]) -> Result<Self, SessionError> {
        let mut validator = Self::default();
        for recorded in events {
            validator.push(recorded)?;
        }
        Ok(validator)
    }

    fn push(&mut self, recorded: &RecordedEvent) -> Result<(), SessionError> {
        self.check(recorded)?;
        self.apply(recorded);
        Ok(())
    }

    fn check(&self, recorded: &RecordedEvent) -> Result<(), SessionError> {
        if recorded.seq != self.next_seq {
            return invalid_event(
                recorded,
                format!("sequence is {}, expected {}", recorded.seq, self.next_seq),
            );
        }
        let mut sources = HashSet::new();
        for source in &recorded.source_event_seqs {
            if *source >= recorded.seq || !sources.insert(*source) {
                return invalid_event(recorded, format!("invalid source event reference {source}"));
            }
        }
        if let Some(SurfaceOp::Replace {
            replaced_event_seqs,
        }) = &recorded.surface_op
        {
            let mut replaced = HashSet::new();
            for source in replaced_event_seqs {
                if *source >= recorded.seq || !replaced.insert(*source) {
                    return invalid_event(
                        recorded,
                        format!("invalid replaced event reference {source}"),
                    );
                }
            }
        }
        match &recorded.event {
            SessionEvent::TurnStart { turn } => {
                if self.active_turn.is_some() || self.turns.contains(turn) {
                    return invalid_event(
                        recorded,
                        format!("turn {turn} is already active or used"),
                    );
                }
            }
            SessionEvent::TurnEnd { turn, .. } => {
                if self.active_turn != Some(*turn) || self.active_step.is_some() {
                    return invalid_event(recorded, format!("turn {turn} cannot end here"));
                }
                if self
                    .tools
                    .values()
                    .any(|tool| tool.turn == *turn && !tool.result)
                {
                    return invalid_event(recorded, format!("turn {turn} has unresolved tools"));
                }
            }
            SessionEvent::StepStart { turn, step } => {
                if self.active_turn != Some(*turn)
                    || self.active_step.is_some()
                    || self.steps.contains(&(*turn, *step))
                {
                    return invalid_event(
                        recorded,
                        format!("step {turn}.{step} is already active or used"),
                    );
                }
            }
            SessionEvent::StepEnd { turn, step, .. } => {
                if self.active_step != Some((*turn, *step)) {
                    return invalid_event(recorded, format!("step {turn}.{step} cannot end here"));
                }
                if self
                    .tools
                    .values()
                    .any(|tool| tool.turn == *turn && tool.step == *step && !tool.result)
                {
                    return invalid_event(
                        recorded,
                        format!("step {turn}.{step} has unresolved tools"),
                    );
                }
            }
            SessionEvent::InputAdmitted { id, .. } => {
                if self.inputs.contains(id) {
                    return invalid_event(recorded, format!("input {id} was admitted twice"));
                }
            }
            SessionEvent::InputConsumed { id } => {
                if !self.inputs.contains(id) || self.completed_inputs.contains(id) {
                    return invalid_event(recorded, format!("input {id} cannot be consumed"));
                }
            }
            SessionEvent::UserMessage { turn, step, .. }
            | SessionEvent::RequestHeader { turn, step, .. }
            | SessionEvent::AssistantChunk { turn, step, .. } => {
                require_active_step(recorded, self.active_step, *turn, *step)?;
            }
            SessionEvent::ModelRequestStart { turn, step } => {
                require_active_step(recorded, self.active_step, *turn, *step)?;
                if self.model_requests.contains(&(*turn, *step)) {
                    return invalid_event(recorded, "model request was started twice");
                }
            }
            SessionEvent::AssistantMessage { turn, step, .. } => {
                require_active_step(recorded, self.active_step, *turn, *step)?;
                if self.finalized_assistants.contains(&(*turn, *step)) {
                    return invalid_event(recorded, "assistant message was finalized twice");
                }
            }
            SessionEvent::ToolCall {
                turn,
                step,
                call_id,
                ..
            } => {
                require_active_step(recorded, self.active_step, *turn, *step)?;
                if self.tools.contains_key(call_id) {
                    return invalid_event(recorded, format!("tool call {call_id} is duplicated"));
                }
            }
            SessionEvent::ToolExecutionStart { call_id } => {
                let Some(tool) = self.tools.get(call_id) else {
                    return invalid_event(recorded, format!("unknown tool call {call_id}"));
                };
                if tool.execution_started.is_some() || tool.result {
                    return invalid_event(recorded, format!("tool call {call_id} cannot start"));
                }
            }
            SessionEvent::ToolExecutionFinish { call_id, .. } => {
                let Some(tool) = self.tools.get(call_id) else {
                    return invalid_event(recorded, format!("unknown tool call {call_id}"));
                };
                let Some(started) = tool.execution_started.as_ref() else {
                    return invalid_event(recorded, format!("tool call {call_id} cannot finish"));
                };
                if tool.execution_finished
                    || tool.result
                    || recorded.time.duration_since(started).is_none()
                {
                    return invalid_event(recorded, format!("tool call {call_id} cannot finish"));
                }
            }
            SessionEvent::ToolResult {
                turn,
                step,
                call_id,
                status,
                ..
            } => {
                let Some(tool) = self.tools.get(call_id) else {
                    return invalid_event(recorded, format!("unknown tool call {call_id}"));
                };
                if (tool.turn, tool.step) != (*turn, *step) || tool.result {
                    return invalid_event(
                        recorded,
                        format!("tool result {call_id} is out of scope"),
                    );
                }
                let executed = matches!(
                    status,
                    crate::session_event::ToolResultStatus::Success
                        | crate::session_event::ToolResultStatus::Error
                );
                if executed && !tool.execution_finished {
                    return invalid_event(
                        recorded,
                        format!("executed tool call {call_id} has no execution finish"),
                    );
                }
            }
            SessionEvent::CompactionStart { compaction_id, .. } => {
                if self.active_compaction.is_some() || self.compactions.contains(compaction_id) {
                    return invalid_event(
                        recorded,
                        format!("compaction {compaction_id} is already active or used"),
                    );
                }
            }
            SessionEvent::CompactionEnd { compaction_id, .. } => {
                if self.active_compaction.as_deref() != Some(compaction_id) {
                    return invalid_event(
                        recorded,
                        format!("compaction {compaction_id} cannot end"),
                    );
                }
            }
        }
        Ok(())
    }

    fn apply(&mut self, recorded: &RecordedEvent) {
        self.next_seq = self.next_seq.saturating_add(1);
        match &recorded.event {
            SessionEvent::TurnStart { turn } => {
                self.turns.insert(*turn);
                self.active_turn = Some(*turn);
            }
            SessionEvent::TurnEnd { .. } => {
                self.active_turn = None;
            }
            SessionEvent::StepStart { turn, step } => {
                self.steps.insert((*turn, *step));
                self.active_step = Some((*turn, *step));
            }
            SessionEvent::StepEnd { .. } => {
                self.active_step = None;
            }
            SessionEvent::InputAdmitted { id, .. } => {
                self.inputs.insert(id.clone());
            }
            SessionEvent::InputConsumed { id } => {
                self.completed_inputs.insert(id.clone());
            }
            SessionEvent::ModelRequestStart { turn, step } => {
                self.model_requests.insert((*turn, *step));
            }
            SessionEvent::AssistantMessage { turn, step, .. } => {
                self.finalized_assistants.insert((*turn, *step));
            }
            SessionEvent::ToolCall {
                turn,
                step,
                call_id,
                ..
            } => {
                self.tools.insert(
                    call_id.clone(),
                    ToolLifecycle {
                        turn: *turn,
                        step: *step,
                        execution_started: None,
                        execution_finished: false,
                        result: false,
                    },
                );
            }
            SessionEvent::ToolExecutionStart { call_id } => {
                self.tools
                    .get_mut(call_id)
                    .expect("tool start was validated")
                    .execution_started = Some(recorded.time.clone());
            }
            SessionEvent::ToolExecutionFinish { call_id, .. } => {
                self.tools
                    .get_mut(call_id)
                    .expect("tool finish was validated")
                    .execution_finished = true;
            }
            SessionEvent::ToolResult { call_id, .. } => {
                self.tools
                    .get_mut(call_id)
                    .expect("tool result was validated")
                    .result = true;
            }
            SessionEvent::CompactionStart { compaction_id, .. } => {
                self.compactions.insert(compaction_id.clone());
                self.active_compaction = Some(compaction_id.clone());
            }
            SessionEvent::CompactionEnd { .. } => {
                self.active_compaction = None;
            }
            SessionEvent::UserMessage { .. }
            | SessionEvent::RequestHeader { .. }
            | SessionEvent::AssistantChunk { .. } => {}
        }
    }
}

fn validate_events(events: &[RecordedEvent]) -> Result<(), SessionError> {
    EventValidator::from_events(events).map(|_| ())
}

fn require_active_step(
    recorded: &RecordedEvent,
    active_step: Option<(u32, u32)>,
    turn: u32,
    step: u32,
) -> Result<(), SessionError> {
    if active_step == Some((turn, step)) {
        Ok(())
    } else {
        invalid_event(
            recorded,
            format!("event is outside active step {turn}.{step}"),
        )
    }
}

fn invalid_event<T>(
    recorded: &RecordedEvent,
    message: impl Into<String>,
) -> Result<T, SessionError> {
    Err(SessionError::Invalid(format!(
        "event {} ({:?}): {}",
        recorded.seq,
        recorded.event,
        message.into()
    )))
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

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

#[cfg(any())]
mod legacy_tests {
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
    async fn archived_session_leaves_the_catalog_and_can_be_restored() {
        let directory = test_directory("archive");
        let session = Session::create(&directory).await.unwrap();
        let info = session.info().clone();
        drop(session);

        let archived = Session::archive(&info).unwrap();
        assert_eq!(
            archived.path.parent(),
            Some(directory.join("archive").as_path())
        );
        assert!(!info.path.exists());
        assert_eq!(Session::list(&directory).unwrap(), []);
        assert_eq!(
            Session::list(directory.join("archive")).unwrap(),
            std::slice::from_ref(&archived)
        );

        let restored = Session::restore(&archived).unwrap();
        assert_eq!(restored, info);
        assert_eq!(Session::list(&directory).unwrap(), [info]);
        assert!(Session::list(directory.join("archive")).unwrap().is_empty());
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

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;

    use async_openai::types::responses::{EasyInputMessage, InputItem};

    use super::{Session, SessionError, catalog_parse_count, session_parse_count};
    use crate::session_event::{
        AssistantChunk, EventTime, SESSION_FORMAT_VERSION, SessionEvent, SurfaceOp, TurnEndReason,
        UserMessageMode,
    };

    #[tokio::test]
    async fn v1_round_trips_events_and_rebuilds_state() {
        let directory = test_directory("v1-round-trip");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        commit
            .event(SessionEvent::StepStart { turn: 1, step: 1 }, vec![], None)
            .await
            .unwrap();
        commit
            .event(
                SessionEvent::UserMessage {
                    turn: 1,
                    step: 1,
                    input_id: None,
                    mode: UserMessageMode::Initial,
                    items: vec![InputItem::from(EasyInputMessage::from("hello"))],
                },
                vec![],
                Some(SurfaceOp::Append),
            )
            .await
            .unwrap();
        commit
            .event(
                SessionEvent::StepEnd {
                    turn: 1,
                    step: 1,
                    outcome: crate::session_event::StepOutcome::Completed,
                    error: None,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        commit
            .event(
                SessionEvent::TurnEnd {
                    turn: 1,
                    reason: TurnEndReason::Completed,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        drop(commit);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(snapshot.events().len(), 5);
        assert!(
            snapshot
                .events()
                .iter()
                .enumerate()
                .all(|(index, event)| event.seq == index as u64)
        );
        assert_eq!(snapshot.state().entries().len(), 1);
        assert!(format!("{:?}", snapshot.state().context()).contains("hello"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_legacy_session_format_explicitly() {
        let directory = test_directory("reject-v0");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("legacy.jsonl");
        std::fs::write(
            &path,
            b"{\"record\":\"session\",\"title\":\"Legacy\",\"created_at\":1}\n",
        )
        .unwrap();
        assert!(matches!(
            Session::inspect(&path),
            Err(SessionError::UnsupportedFormat {
                found: 0,
                expected: SESSION_FORMAT_VERSION,
            })
        ));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn rejects_non_contiguous_event_sequences() {
        let directory = test_directory("bad-seq");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        let event = commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        drop(commit);
        let mut invalid = event;
        invalid.seq = 9;
        let line = serde_json::json!({
            "record": "event",
            "seq": invalid.seq,
            "time": invalid.time,
            "type": "turn_end",
            "turn": 1,
            "reason": "completed"
        });
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(format!("{line}\n").as_bytes())
            .unwrap();
        assert!(matches!(
            Session::inspect(&path),
            Err(SessionError::Invalid(_))
        ));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn rejects_invalid_sources_before_writing() {
        let session = Session::memory();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        assert!(matches!(
            commit
                .event(SessionEvent::TurnStart { turn: 1 }, vec![0], None)
                .await,
            Err(SessionError::Invalid(_))
        ));
        let recorded = commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        assert_eq!(recorded.seq, 0);
    }

    #[tokio::test]
    async fn enforces_tool_execution_lifecycle() {
        let session = Session::memory();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        commit
            .event(SessionEvent::StepStart { turn: 1, step: 1 }, vec![], None)
            .await
            .unwrap();
        commit
            .event(
                SessionEvent::ToolCall {
                    turn: 1,
                    step: 1,
                    call_id: "call-1".into(),
                    parent_call_id: None,
                    name: "shell".into(),
                    arguments: "{}".into(),
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        assert!(matches!(
            commit
                .event(
                    SessionEvent::ToolExecutionFinish {
                        call_id: "call-1".into(),
                        outcome: crate::ToolExecutionOutcome::Success,
                    },
                    vec![],
                    None,
                )
                .await,
            Err(SessionError::Invalid(_))
        ));
        assert!(matches!(
            commit
                .event(
                    SessionEvent::StepEnd {
                        turn: 1,
                        step: 1,
                        outcome: crate::StepOutcome::Completed,
                        error: None,
                    },
                    vec![],
                    None,
                )
                .await,
            Err(SessionError::Invalid(_))
        ));
        let started = commit
            .event(
                SessionEvent::ToolExecutionStart {
                    call_id: "call-1".into(),
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        assert_eq!(started.seq, 3);
    }

    #[tokio::test]
    async fn reports_and_repairs_a_torn_v1_tail() {
        let directory = test_directory("tail");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        drop(session);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(br#"{\"record\":\"event\""#)
            .unwrap();
        assert!(Session::inspect(&path).unwrap().recovery_needed());
        let mut reopened = Session::open(&path).await.unwrap();
        assert!(reopened.take_recovery_report().is_some());
        drop(reopened);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn catalog_omits_sessions_with_invalid_event_lifecycles() {
        let directory = test_directory("catalog-invalid-events");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        drop(session);
        let invalid = serde_json::json!({
            "record": "event",
            "seq": 0,
            "time": {
                "wall_time_ms": 1,
                "clock_id": "catalog-invalid-events",
                "monotonic_ns": 0
            },
            "type": "step_start",
            "turn": 1,
            "step": 1
        });
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(format!("{invalid}\n").as_bytes())
            .unwrap();

        let catalog = Session::catalog(&directory).unwrap();
        assert!(catalog.sessions.is_empty());
        assert_eq!(catalog.issues.len(), 1);
        assert_eq!(catalog.issues[0].path, path);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn catalog_reuses_validated_files_and_exposes_the_same_parse_for_search() {
        let directory = test_directory("catalog-cache");
        let mut session = Session::create(&directory).await.unwrap();
        session.rename("search needle").await.unwrap();
        let path = session.info().path.clone();
        drop(session);

        let before = catalog_parse_count(&path);
        let first = Session::catalog(&directory).unwrap();
        assert_eq!(catalog_parse_count(&path), before + 1);
        assert!(
            first.search[&path]
                .values
                .iter()
                .any(|value| value == "search needle")
        );
        let second = Session::catalog(&directory).unwrap();
        assert_eq!(catalog_parse_count(&path), before + 1);
        assert_eq!(first, second);

        let mut reopened = Session::open(&path).await.unwrap();
        reopened.rename("updated needle").await.unwrap();
        drop(reopened);
        let updated = Session::catalog(&directory).unwrap();
        assert_eq!(catalog_parse_count(&path), before + 2);
        assert!(
            updated.search[&path]
                .values
                .iter()
                .any(|value| value == "updated needle")
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn unchanged_readonly_commit_prepares_without_reparsing_the_log() {
        let directory = test_directory("prepare-fast-path");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        drop(session);

        let before = session_parse_count(&path);
        let session = Session::open_readonly(&path).unwrap();
        assert_eq!(session_parse_count(&path), before + 1);
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        assert_eq!(session_parse_count(&path), before + 1);
        commit.release_writer();
        commit.prepare(&state).await.unwrap();
        assert_eq!(session_parse_count(&path), before + 1);

        drop(commit);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn readonly_commit_repairs_a_torn_tail_before_appending() {
        let directory = test_directory("readonly-repair-tail");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        drop(session);
        OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(br#"{"record":"event""#)
            .unwrap();

        let readonly = Session::open_readonly(&path).unwrap();
        assert!(readonly.recovery_needed());
        let (state, mut commit) = readonly.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        drop(commit);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(snapshot.events().len(), 1);
        assert!(matches!(
            snapshot.events()[0].event,
            SessionEvent::TurnStart { turn: 1 }
        ));
        assert!(
            std::fs::read_dir(&directory)
                .unwrap()
                .filter_map(Result::ok)
                .any(|entry| entry.file_name().to_string_lossy().contains(".recovery-"))
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn readonly_commit_restores_a_missing_newline_before_appending() {
        let directory = test_directory("readonly-repair-newline");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        drop(session);
        let mut bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        std::fs::write(&path, bytes).unwrap();

        let readonly = Session::open_readonly(&path).unwrap();
        assert!(readonly.recovery_needed());
        let (state, mut commit) = readonly.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        drop(commit);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(snapshot.events().len(), 1);
        assert!(!snapshot.recovery_needed());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn assistant_chunks_are_written_as_one_batch_before_a_structural_event() {
        let directory = test_directory("chunk-batch");
        let session = Session::create(&directory).await.unwrap();
        let path = session.info().path.clone();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        for event in [
            SessionEvent::TurnStart { turn: 1 },
            SessionEvent::StepStart { turn: 1, step: 1 },
            SessionEvent::ModelRequestStart { turn: 1, step: 1 },
        ] {
            commit.event(event, vec![], None).await.unwrap();
        }
        for delta in ["hello ", "world"] {
            commit
                .event(
                    SessionEvent::AssistantChunk {
                        turn: 1,
                        step: 1,
                        chunk: AssistantChunk::OutputTextDelta {
                            delta: delta.into(),
                        },
                    },
                    vec![],
                    None,
                )
                .await
                .unwrap();
        }

        assert_eq!(Session::inspect(&path).unwrap().events().len(), 3);
        commit
            .event(
                SessionEvent::StepEnd {
                    turn: 1,
                    step: 1,
                    outcome: crate::StepOutcome::Aborted,
                    error: None,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        commit
            .event(
                SessionEvent::TurnEnd {
                    turn: 1,
                    reason: TurnEndReason::Aborted,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        drop(commit);

        let snapshot = Session::inspect(&path).unwrap();
        assert_eq!(snapshot.events().len(), 7);
        let catalog = Session::catalog(&directory).unwrap();
        assert_eq!(
            catalog.search[&path]
                .values
                .iter()
                .filter(|value| value.contains("hello"))
                .collect::<Vec<_>>(),
            ["hello world"]
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn concurrent_tool_timestamps_may_finish_outside_delivery_order() {
        let session = Session::memory();
        let (state, mut commit) = session.into_parts();
        commit.prepare(&state).await.unwrap();
        commit
            .event(SessionEvent::TurnStart { turn: 1 }, vec![], None)
            .await
            .unwrap();
        commit
            .event(SessionEvent::StepStart { turn: 1, step: 1 }, vec![], None)
            .await
            .unwrap();
        for call_id in ["first", "second"] {
            commit
                .event(
                    SessionEvent::ToolCall {
                        turn: 1,
                        step: 1,
                        call_id: call_id.into(),
                        parent_call_id: None,
                        name: "shell".into(),
                        arguments: "{}".into(),
                    },
                    vec![],
                    None,
                )
                .await
                .unwrap();
        }
        let time = |monotonic_ns| EventTime {
            wall_time_ms: monotonic_ns as i64,
            clock_id: "parallel-tools".into(),
            monotonic_ns,
        };
        commit
            .event_at(
                time(100),
                SessionEvent::ToolExecutionStart {
                    call_id: "first".into(),
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        commit
            .event_at(
                time(200),
                SessionEvent::ToolExecutionStart {
                    call_id: "second".into(),
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        commit
            .event_at(
                time(250),
                SessionEvent::ToolExecutionFinish {
                    call_id: "second".into(),
                    outcome: crate::ToolExecutionOutcome::Success,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
        commit
            .event_at(
                time(150),
                SessionEvent::ToolExecutionFinish {
                    call_id: "first".into(),
                    outcome: crate::ToolExecutionOutcome::Success,
                },
                vec![],
                None,
            )
            .await
            .unwrap();
    }

    #[test]
    fn event_time_uses_monotonic_duration_only_within_a_clock() {
        let first = crate::EventTime {
            wall_time_ms: 100,
            clock_id: "clock".into(),
            monotonic_ns: 10,
        };
        let second = crate::EventTime {
            wall_time_ms: 50,
            clock_id: "clock".into(),
            monotonic_ns: 25,
        };
        assert_eq!(second.duration_since(&first), Some(15));
        let other = crate::EventTime {
            clock_id: "other".into(),
            ..second
        };
        assert_eq!(other.duration_since(&first), None);
    }

    fn test_directory(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "kcastle-session-v1-{label}-{}-{nonce}",
            std::process::id()
        ))
    }
}
