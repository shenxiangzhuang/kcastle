use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) mod event;
pub(crate) mod machine;
pub(crate) mod store;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use crate::session::event::RecordedEvent;
use crate::session::machine::{PendingInput, SessionMachine, SessionMachineError};
use crate::session::store::{
    ArchiveFilter, CreateStoredSession, LoadedSession, MetadataUpdate, SESSION_DATABASE_FILE,
    SessionErrorClass, SessionStore, SessionStoreError, StoredSessionMetadata, classify_io_error,
};
pub fn validate_events(events: &[RecordedEvent]) -> Result<(), SessionMachineError> {
    SessionMachine::from_events(events).map(|_| ())
}

pub const DEFAULT_PROJECT_ID: &str = "default";
const ARCHIVE_DIRECTORY: &str = "archive";
const SESSION_LOCATOR_EXTENSION: &str = "session-v2";

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("session store failed: {0}")]
    Store(#[from] SessionStoreError),
    #[error("session history is invalid: {0}")]
    Machine(#[from] SessionMachineError),
    #[error("session I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid session locator: {0}")]
    Invalid(String),
    #[error("unsupported session format {found}; expected {expected}")]
    UnsupportedFormat { found: u32, expected: u32 },
}

impl SessionError {
    pub fn classification(&self) -> SessionErrorClass {
        match self {
            Self::Store(error) => error.classification(),
            Self::Machine(_) | Self::Invalid(_) | Self::UnsupportedFormat { .. } => {
                SessionErrorClass::DeterministicInvalid
            }
            Self::Io(error) => classify_io_error(error),
        }
    }

    pub fn is_deterministic_invalid(&self) -> bool {
        self.classification() == SessionErrorClass::DeterministicInvalid
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub allow_all_tools: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_raw(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn is_storage_safe(&self) -> bool {
        !self.0.is_empty()
            && self.0.len() <= 128
            && self
                .0
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub id: SessionId,
    pub project_id: String,
    /// An opaque logical locator. Session v2 rows live in the project database; this path is not
    /// an individual data file and callers must pass it back to `Session` APIs rather than open it.
    pub path: PathBuf,
    pub title: String,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionCatalog {
    pub sessions: Vec<SessionInfo>,
    /// Raw canonical values extracted atomically with the journal revision. Consumers own search
    /// normalization, snippets, and presentation summaries.
    pub search_values: HashMap<PathBuf, Arc<[String]>>,
}

#[derive(Debug, Clone)]
pub struct SessionSnapshot {
    info: SessionInfo,
    machine: SessionMachine,
    events: Vec<RecordedEvent>,
    config: SessionConfig,
}

impl SessionSnapshot {
    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    pub fn pending_inputs(&self) -> Vec<PendingInput> {
        self.machine.pending_inputs()
    }

    pub fn recovery_needed(&self) -> bool {
        self.machine.active_run().is_some()
            || self.machine.active_turn().is_some()
            || self.machine.active_step().is_some()
            || self.machine.active_compaction().is_some()
            || !self.machine.unresolved_tool_calls().is_empty()
    }

    pub fn export_jsonl(&self, destination: impl AsRef<Path>) -> Result<(), SessionError> {
        let (directory, id) = parse_locator(&self.info.path)?;
        SessionStore::open_database_readonly(directory.join(SESSION_DATABASE_FILE))?
            .export_jsonl_to_path(&id, destination)
            .map_err(Into::into)
    }
}

pub struct Session {
    info: SessionInfo,
    machine: SessionMachine,
    events: Vec<RecordedEvent>,
    config: SessionConfig,
    revision: u64,
    store: SessionStore,
}

pub(crate) struct SessionParts {
    pub(crate) info: SessionInfo,
    pub(crate) machine: SessionMachine,
    pub(crate) revision: u64,
    pub(crate) store: SessionStore,
    pub(crate) config: SessionConfig,
}

impl Session {
    pub fn memory() -> Self {
        let store = SessionStore::open_in_memory().expect("in-memory SQLite must open");
        let id = SessionId::new();
        let metadata = store
            .create_session(CreateStoredSession {
                id: id.clone(),
                project_id: DEFAULT_PROJECT_ID.into(),
                title: "Untitled session".into(),
                config: SessionConfig::default(),
                created_at_ms: now_millis(),
            })
            .expect("in-memory session must be created");
        Self::from_loaded(
            store,
            LoadedSession {
                metadata,
                transactions: Vec::new(),
            },
            PathBuf::new(),
        )
        .expect("empty in-memory session must replay")
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
        Self::create_named_in_project_with_id(directory, project_id, config, id, "Untitled session")
            .await
    }

    pub async fn create_named_in_project_with_id(
        directory: impl AsRef<Path>,
        project_id: impl Into<String>,
        config: SessionConfig,
        id: SessionId,
        title: impl Into<String>,
    ) -> Result<Self, SessionError> {
        let directory = canonical_session_directory(directory.as_ref());
        let project_id = project_id.into();
        let title = title.into();
        tokio::task::spawn_blocking(move || {
            Self::create_inner(directory, project_id, config, id, title)
        })
        .await
        .map_err(|error| SessionError::Invalid(format!("session task failed: {error}")))?
    }

    fn create_inner(
        directory: PathBuf,
        project_id: String,
        config: SessionConfig,
        id: SessionId,
        title: String,
    ) -> Result<Self, SessionError> {
        let store = SessionStore::open_project(&directory)?;
        let metadata = store.create_session(CreateStoredSession {
            id: id.clone(),
            project_id,
            title,
            config,
            created_at_ms: now_millis(),
        })?;
        let path = locator(&directory, &id, false);
        Self::from_loaded(
            store,
            LoadedSession {
                metadata,
                transactions: Vec::new(),
            },
            path,
        )
    }

    pub fn inspect(path: impl AsRef<Path>) -> Result<SessionSnapshot, SessionError> {
        Self::open_readonly(path)
    }

    /// Opens a capability-limited snapshot. It cannot be passed to [`crate::Agent`] or mutated.
    pub fn open_readonly(path: impl AsRef<Path>) -> Result<SessionSnapshot, SessionError> {
        Self::open_snapshot_inner(path.as_ref(), None)
    }

    pub fn open_readonly_in_project(
        path: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<SessionSnapshot, SessionError> {
        Self::open_snapshot_inner(path.as_ref(), Some(project_id))
    }

    pub async fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();
        tokio::task::spawn_blocking(move || Self::open_inner(&path, None))
            .await
            .map_err(|error| SessionError::Invalid(format!("session task failed: {error}")))?
    }

    pub async fn open_in_project(
        path: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();
        let project_id = project_id.to_owned();
        tokio::task::spawn_blocking(move || Self::open_inner(&path, Some(&project_id)))
            .await
            .map_err(|error| SessionError::Invalid(format!("session task failed: {error}")))?
    }

    pub fn open_writable_in_project(
        path: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<Self, SessionError> {
        Self::open_inner(path.as_ref(), Some(project_id))
    }

    fn open_snapshot_inner(
        path: &Path,
        project_id: Option<&str>,
    ) -> Result<SessionSnapshot, SessionError> {
        let (directory, id) = parse_locator(path)?;
        let database = directory.join(SESSION_DATABASE_FILE);
        if !database.is_file() {
            return Err(
                std::io::Error::new(ErrorKind::NotFound, "session database not found").into(),
            );
        }
        let store = SessionStore::open_database_readonly(database)?;
        let loaded = store.load(&id)?;
        if let Some(expected) = project_id
            && loaded.metadata.project_id != expected
        {
            return Err(SessionError::Invalid(format!(
                "session belongs to project {} instead of {expected}",
                loaded.metadata.project_id
            )));
        }
        let LoadedSession {
            metadata,
            transactions,
        } = loaded;
        let archived = metadata.archived_at_ms.is_some();
        let path = locator(&directory, &id, archived);
        let events = transactions
            .into_iter()
            .flat_map(|transaction| transaction.events)
            .collect::<Vec<_>>();
        let machine = SessionMachine::from_events(&events)?;
        let config = metadata.config.clone();
        let info = info_from_metadata(path, &metadata);
        Ok(SessionSnapshot {
            info,
            machine,
            events,
            config,
        })
    }

    fn open_inner(path: &Path, project_id: Option<&str>) -> Result<Self, SessionError> {
        let (directory, id) = parse_locator(path)?;
        let database = directory.join(SESSION_DATABASE_FILE);
        if !database.is_file() {
            return Err(
                std::io::Error::new(ErrorKind::NotFound, "session database not found").into(),
            );
        }
        let store = SessionStore::open_database(database)?;
        let loaded = store.load(&id)?;
        if let Some(expected) = project_id
            && loaded.metadata.project_id != expected
        {
            return Err(SessionError::Invalid(format!(
                "session belongs to project {} instead of {expected}",
                loaded.metadata.project_id
            )));
        }
        let archived = loaded.metadata.archived_at_ms.is_some();
        let path = locator(&directory, &id, archived);
        Self::from_loaded(store, loaded, path)
    }

    fn from_loaded(
        store: SessionStore,
        loaded: LoadedSession,
        path: PathBuf,
    ) -> Result<Self, SessionError> {
        let LoadedSession {
            metadata,
            transactions,
        } = loaded;
        let events = transactions
            .into_iter()
            .flat_map(|transaction| transaction.events)
            .collect::<Vec<_>>();
        let machine = SessionMachine::from_events(&events)?;
        let config = metadata.config.clone();
        let info = info_from_metadata(path, &metadata);
        Ok(Self {
            info,
            machine,
            events,
            config,
            revision: metadata.revision,
            store,
        })
    }

    pub fn list(directory: impl AsRef<Path>) -> Result<Vec<SessionInfo>, SessionError> {
        Ok(Self::catalog(directory)?.sessions)
    }

    pub fn catalog(directory: impl AsRef<Path>) -> Result<SessionCatalog, SessionError> {
        Self::catalog_in_project(directory, DEFAULT_PROJECT_ID)
    }

    pub fn catalog_in_project(
        directory: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<SessionCatalog, SessionError> {
        let requested = directory.as_ref();
        let archived = requested
            .file_name()
            .is_some_and(|name| name == ARCHIVE_DIRECTORY);
        let directory = canonical_session_directory(requested);
        let database = directory.join(SESSION_DATABASE_FILE);
        if !database.is_file() {
            return Ok(SessionCatalog::default());
        }
        let store = SessionStore::open_database_readonly(database)?;
        let filter = if archived {
            ArchiveFilter::Archived
        } else {
            ArchiveFilter::Active
        };
        let mut catalog = SessionCatalog::default();
        for entry in store.catalog(project_id, filter)? {
            let metadata = entry.metadata;
            let path = locator(&directory, &metadata.id, archived);
            catalog
                .search_values
                .insert(path.clone(), entry.search_values.into());
            catalog.sessions.push(info_from_metadata(path, &metadata));
        }
        Ok(catalog)
    }

    pub fn archived_catalog_in_project(
        directory: impl AsRef<Path>,
        project_id: &str,
    ) -> Result<SessionCatalog, SessionError> {
        Self::catalog_in_project(directory.as_ref().join(ARCHIVE_DIRECTORY), project_id)
    }

    pub fn delete(session: &SessionInfo) -> Result<(), SessionError> {
        let (directory, id) = validated_locator(session)?;
        let store = SessionStore::open_project(directory)?;
        let permit = store.acquire_writer(&id)?;
        store.delete(&id, &permit)?;
        Ok(())
    }

    pub fn archive(session: &SessionInfo) -> Result<SessionInfo, SessionError> {
        let (directory, id) = validated_locator(session)?;
        let store = SessionStore::open_project(&directory)?;
        let permit = store.acquire_writer(&id)?;
        let metadata = store.archive(&id, &permit)?;
        Ok(info_from_metadata(
            locator(&directory, &id, true),
            &metadata,
        ))
    }

    pub fn restore(session: &SessionInfo) -> Result<SessionInfo, SessionError> {
        let (directory, id) = validated_locator(session)?;
        let store = SessionStore::open_project(&directory)?;
        let permit = store.acquire_writer(&id)?;
        let metadata = store.restore(&id, &permit)?;
        Ok(info_from_metadata(
            locator(&directory, &id, false),
            &metadata,
        ))
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    /// Moves the canonical history out for a UI projection before this session is transferred to
    /// an [`crate::Agent`]. The agent owns the already-replayed machine and does not retain a
    /// second event copy.
    pub fn take_events(&mut self) -> Vec<RecordedEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Revision of the canonical store snapshot replayed into this session.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn recovery_needed(&self) -> bool {
        self.machine.active_run().is_some()
            || self.machine.active_turn().is_some()
            || self.machine.active_step().is_some()
            || self.machine.active_compaction().is_some()
            || !self.machine.unresolved_tool_calls().is_empty()
    }

    pub(crate) fn into_parts(self) -> SessionParts {
        SessionParts {
            info: self.info,
            machine: self.machine,
            revision: self.revision,
            store: self.store,
            config: self.config,
        }
    }

    pub async fn rename(&mut self, title: &str) -> Result<(), SessionError> {
        let permit = self.store.acquire_writer(&self.info.id)?;
        let metadata = self.store.update_metadata(
            &self.info.id,
            MetadataUpdate {
                title: Some(title.to_owned()),
                ..MetadataUpdate::default()
            },
            &permit,
        )?;
        self.info.title = metadata.title;
        self.info.updated_at = millis_to_seconds(metadata.updated_at_ms);
        Ok(())
    }

    pub fn export_jsonl(&self, destination: impl AsRef<Path>) -> Result<(), SessionError> {
        self.store
            .export_jsonl_to_path(&self.info.id, destination)
            .map_err(Into::into)
    }
}

fn canonical_session_directory(directory: &Path) -> PathBuf {
    if directory
        .file_name()
        .is_some_and(|name| name == ARCHIVE_DIRECTORY)
    {
        directory.parent().unwrap_or(directory).to_path_buf()
    } else {
        directory.to_path_buf()
    }
}

fn validated_locator(session: &SessionInfo) -> Result<(PathBuf, SessionId), SessionError> {
    let (directory, id) = parse_locator(&session.path)?;
    if id != session.id {
        return Err(SessionError::Invalid(format!(
            "session locator identifies {id}, but metadata identifies {}",
            session.id
        )));
    }
    let metadata = SessionStore::open_database_readonly(directory.join(SESSION_DATABASE_FILE))?
        .metadata(&id)?;
    if metadata.project_id != session.project_id {
        return Err(SessionError::Invalid(format!(
            "session belongs to project {} instead of {}",
            metadata.project_id, session.project_id
        )));
    }
    Ok((directory, id))
}

fn locator(directory: &Path, id: &SessionId, archived: bool) -> PathBuf {
    let directory = if archived {
        directory.join(ARCHIVE_DIRECTORY)
    } else {
        directory.to_path_buf()
    };
    directory.join(format!("{}.{}", id.as_str(), SESSION_LOCATOR_EXTENSION))
}

fn parse_locator(path: &Path) -> Result<(PathBuf, SessionId), SessionError> {
    if path.extension().and_then(|value| value.to_str()) != Some(SESSION_LOCATOR_EXTENSION) {
        return Err(SessionError::Invalid(format!(
            "{} is not a Session v2 locator",
            path.display()
        )));
    }
    let raw = path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| SessionError::Invalid(format!("{} has no session ID", path.display())))?;
    let parent = path.parent().ok_or_else(|| {
        SessionError::Invalid(format!("{} has no project directory", path.display()))
    })?;
    let id = SessionId::from_raw(raw);
    if !id.is_storage_safe() {
        return Err(SessionError::Invalid(format!(
            "{} contains an invalid session ID",
            path.display()
        )));
    }
    Ok((canonical_session_directory(parent), id))
}

fn info_from_metadata(path: PathBuf, metadata: &StoredSessionMetadata) -> SessionInfo {
    SessionInfo {
        id: metadata.id.clone(),
        project_id: metadata.project_id.clone(),
        path,
        title: metadata.title.clone(),
        created_at: millis_to_seconds(metadata.created_at_ms),
        updated_at: millis_to_seconds(metadata.updated_at_ms),
    }
}

fn millis_to_seconds(millis: i64) -> u64 {
    u64::try_from(millis.max(0)).unwrap_or_default() / 1_000
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InputOrigin;
    use std::fs;

    fn temp_directory(label: &str) -> PathBuf {
        let directory =
            std::env::temp_dir().join(format!("kcastle-session-v2-{label}-{}", Uuid::new_v4()));
        fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[tokio::test]
    async fn project_database_returns_logical_session_locators() {
        let directory = temp_directory("catalog");
        let session = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        assert!(!session.info.path.exists());
        assert!(directory.join(SESSION_DATABASE_FILE).is_file());
        let catalog = Session::catalog_in_project(&directory, "project-a").unwrap();
        assert_eq!(catalog.sessions, vec![session.info.clone()]);
        // A live Session intentionally owns the project's SQLite connection. Windows does not
        // allow its database or WAL files to be removed until that capability is released.
        drop(session);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn caller_supplies_initial_session_title() {
        let directory = temp_directory("named-session");
        let session = Session::create_named_in_project_with_id(
            &directory,
            "project-a",
            SessionConfig::default(),
            SessionId::from_raw("named-session"),
            "Desktop-owned title",
        )
        .await
        .unwrap();

        assert_eq!(session.info().title, "Desktop-owned title");

        drop(session);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn archive_and_restore_update_catalog_without_moving_files() {
        let directory = temp_directory("archive");
        let session = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        let archived = Session::archive(session.info()).unwrap();
        assert!(
            Session::catalog_in_project(&directory, "project-a")
                .unwrap()
                .sessions
                .is_empty()
        );
        assert_eq!(
            Session::catalog_in_project(directory.join(ARCHIVE_DIRECTORY), "project-a")
                .unwrap()
                .sessions,
            vec![archived.clone()]
        );
        let restored = Session::restore(&archived).unwrap();
        assert_eq!(restored.id, session.info.id);
        drop(session);
        fs::remove_dir_all(directory).unwrap();
    }

    #[tokio::test]
    async fn rename_refreshes_the_public_updated_at_from_the_metadata_receipt() {
        let directory = temp_directory("rename-metadata");
        let mut session = Session::create_in_project(&directory, "project-a")
            .await
            .unwrap();
        session.info.updated_at = 0;

        session.rename("Renamed session").await.unwrap();

        let metadata = session.store.metadata(&session.info.id).unwrap();
        assert!(metadata.updated_at_ms > 0);
        assert_eq!(
            session.info.updated_at,
            millis_to_seconds(metadata.updated_at_ms)
        );
        drop(session);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn old_jsonl_is_not_a_session_v2_locator() {
        let error = match Session::open_readonly("old.jsonl") {
            Ok(_) => panic!("old JSONL must not open as Session v2"),
            Err(error) => error,
        };
        assert!(matches!(error, SessionError::Invalid(_)));
    }

    #[test]
    fn session_error_classification_preserves_transient_io_and_invalid_history() {
        let transient = SessionError::Io(std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "temporary storage timeout",
        ));
        assert_eq!(transient.classification(), SessionErrorClass::Transient);

        let invalid = SessionError::Store(SessionStoreError::Corrupt("bad journal".into()));
        assert!(invalid.is_deterministic_invalid());
    }

    #[test]
    fn a_non_database_catalog_is_deterministically_invalid() {
        let directory = temp_directory("corrupt-catalog");
        fs::write(
            directory.join(SESSION_DATABASE_FILE),
            b"not a sqlite database",
        )
        .unwrap();

        let error = Session::catalog_in_project(&directory, "project-a").unwrap_err();
        assert_eq!(
            error.classification(),
            SessionErrorClass::DeterministicInvalid
        );

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn pending_inputs_are_replayed_by_the_machine() {
        let session = Session::memory();
        assert!(session.machine.pending_inputs().is_empty());
        assert!(!session.recovery_needed());
        let _ = InputOrigin::Queue;
    }
}
