use std::collections::HashMap;
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::AtomicU8;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::session::event::{
    AssistantChunk, EventTime, RecordedEvent, SESSION_FORMAT_VERSION, SessionEvent, TxId,
};
use crate::session::machine::{PlannedBatch, SESSION_MACHINE_SEMANTICS_VERSION};
use crate::session::{SessionConfig, SessionId};
use rusqlite::{Connection, OpenFlags, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SESSION_DATABASE_FILE: &str = "sessions.sqlite3";
pub const JSONL_STORE_FORMAT_VERSION: u32 = 3;
// This SQLite store is new in session v2. Earlier development-only revisions never shipped, so
// the first public on-disk schema starts at version 1.
const DATABASE_SCHEMA_VERSION: u32 = 1;
const CATALOG_EXTRACTOR_VERSION: i64 = 1;
// A catalog row is safe to present only when both its serialized event representation and the
// state-machine semantics that interpret those events match this binary. Keep the two inputs
// explicit so either compatibility boundary must deliberately advance the persisted contract.
const CATALOG_LOADABILITY_VERSION: i64 =
    ((SESSION_FORMAT_VERSION as i64) << 32) | SESSION_MACHINE_SEMANTICS_VERSION as i64;

#[cfg(test)]
const FAILPOINT_NONE: u8 = 0;
#[cfg(test)]
const FAILPOINT_BEFORE_COMMIT: u8 = 1;
#[cfg(test)]
const FAILPOINT_AFTER_COMMIT: u8 = 2;
#[cfg(test)]
const FAILPOINT_PAUSE_BEFORE_COMMIT: u8 = 3;
#[cfg(test)]
const FAILPOINT_PAUSE_RESOLVE_AFTER_REVISION: u8 = 4;
#[cfg(test)]
const FAILPOINT_RESOLVE_PAUSED: u8 = 5;

#[derive(Debug, Error)]
pub enum SessionStoreError {
    #[error("session store I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("session store SQLite operation failed: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("session store serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("session store lock was poisoned")]
    LockPoisoned,
    #[error("session {0:?} was not found")]
    SessionNotFound(SessionId),
    #[error("transaction {tx_id} was not found for session {session_id:?}")]
    TransactionNotFound {
        session_id: SessionId,
        tx_id: TransactionId,
    },
    #[error(
        "session {session_id:?} is at revision {current_revision}, not expected revision {expected_revision}"
    )]
    RevisionConflict {
        session_id: SessionId,
        expected_revision: u64,
        current_revision: u64,
    },
    #[error("event sequence is {found}, expected {expected}")]
    EventSequenceConflict { expected: u64, found: u64 },
    #[error("transaction ID {tx_id} was reused with different content")]
    TransactionConflict { tx_id: TransactionId },
    #[error("transaction {tx_id} may have committed; resolve it by retrying the same AppendTx")]
    OutcomeUnknown { tx_id: TransactionId },
    #[error("an injected failure occurred before commit")]
    InjectedBeforeCommit,
    #[error("transaction must contain at least one event")]
    EmptyTransaction,
    #[error("invalid session store value: {0}")]
    Invalid(String),
    #[error("session store history is corrupt: {0}")]
    Corrupt(String),
    #[error("session {session_id:?} already has an active writer")]
    WriterBusy { session_id: SessionId },
    #[error("writer permit does not authorize session {session_id:?} in this database")]
    InvalidWriterPermit { session_id: SessionId },
    #[error("a readonly session store cannot acquire a writer permit")]
    ReadonlyStore,
    #[error("numeric value is outside SQLite's signed 64-bit range: {0}")]
    NumericOverflow(u64),
    #[error("unsupported session database schema {found}; expected {expected}")]
    UnsupportedSchemaVersion { found: i64, expected: u32 },
}

/// Whether a failed session operation invalidates persisted session data or may succeed later.
///
/// Desktop catalog caches use this distinction to remove deterministic bad data without making a
/// temporary lock or storage outage look like the user deleted their sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionErrorClass {
    DeterministicInvalid,
    Transient,
    Operational,
}

impl SessionStoreError {
    pub fn classification(&self) -> SessionErrorClass {
        match self {
            Self::Io(error) => classify_io_error(error),
            Self::Sqlite(error) => classify_sqlite_error(error),
            Self::Json(_)
            | Self::SessionNotFound(_)
            | Self::TransactionNotFound { .. }
            | Self::EventSequenceConflict { .. }
            | Self::EmptyTransaction
            | Self::Invalid(_)
            | Self::Corrupt(_)
            | Self::NumericOverflow(_)
            | Self::UnsupportedSchemaVersion { .. } => SessionErrorClass::DeterministicInvalid,
            Self::RevisionConflict { .. }
            | Self::OutcomeUnknown { .. }
            | Self::WriterBusy { .. } => SessionErrorClass::Transient,
            Self::LockPoisoned
            | Self::TransactionConflict { .. }
            | Self::InjectedBeforeCommit
            | Self::InvalidWriterPermit { .. }
            | Self::ReadonlyStore => SessionErrorClass::Operational,
        }
    }

    pub fn is_deterministic_invalid(&self) -> bool {
        self.classification() == SessionErrorClass::DeterministicInvalid
    }
}

pub(crate) fn classify_io_error(error: &io::Error) -> SessionErrorClass {
    match error.kind() {
        io::ErrorKind::NotFound | io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => {
            SessionErrorClass::DeterministicInvalid
        }
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted => {
            SessionErrorClass::Transient
        }
        _ => SessionErrorClass::Operational,
    }
}

fn classify_sqlite_error(error: &rusqlite::Error) -> SessionErrorClass {
    if let Some(code) = error.sqlite_error_code() {
        return match code {
            rusqlite::ffi::ErrorCode::DatabaseCorrupt | rusqlite::ffi::ErrorCode::NotADatabase => {
                SessionErrorClass::DeterministicInvalid
            }
            rusqlite::ffi::ErrorCode::DatabaseBusy
            | rusqlite::ffi::ErrorCode::DatabaseLocked
            | rusqlite::ffi::ErrorCode::OperationInterrupted
            | rusqlite::ffi::ErrorCode::SystemIoFailure
            | rusqlite::ffi::ErrorCode::CannotOpen
            | rusqlite::ffi::ErrorCode::FileLockingProtocolFailed
            | rusqlite::ffi::ErrorCode::SchemaChanged => SessionErrorClass::Transient,
            _ => SessionErrorClass::Operational,
        };
    }

    match error {
        rusqlite::Error::FromSqlConversionFailure(..)
        | rusqlite::Error::IntegralValueOutOfRange(..)
        | rusqlite::Error::Utf8Error(..)
        | rusqlite::Error::InvalidColumnType(..)
        | rusqlite::Error::QueryReturnedNoRows => SessionErrorClass::DeterministicInvalid,
        _ => SessionErrorClass::Operational,
    }
}

pub type TransactionId = TxId;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredSessionMetadata {
    pub id: SessionId,
    pub project_id: String,
    pub title: String,
    pub config: SessionConfig,
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
    pub archived_at_ms: Option<i64>,
    pub revision: u64,
}

/// One catalog row and its already-extracted search values.
///
/// This is intentionally separate from [`LoadedSession`]: listing sessions must never deserialize
/// or replay the journal. The projection is advanced atomically with every journal commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredCatalogEntry {
    pub metadata: StoredSessionMetadata,
    pub search_values: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CreateStoredSession {
    pub id: SessionId,
    pub project_id: String,
    pub title: String,
    pub config: SessionConfig,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Default)]
pub struct MetadataUpdate {
    pub project_id: Option<String>,
    pub title: Option<String>,
    pub config: Option<SessionConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArchiveFilter {
    Active,
    Archived,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendFailpoint {
    BeforeCommitOnce,
    AfterCommitBeforeReceiptOnce,
    PauseBeforeCommit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendTx {
    pub session_id: SessionId,
    pub tx_id: TransactionId,
    pub expected_revision: u64,
    pub events: Vec<RecordedEvent>,
}

impl AppendTx {
    pub fn from_planned(
        session_id: SessionId,
        expected_revision: u64,
        batch: &PlannedBatch,
    ) -> Self {
        Self {
            session_id,
            tx_id: batch.tx_id().clone(),
            expected_revision,
            events: batch.events().to_vec(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitReceipt {
    pub session_id: SessionId,
    pub tx_id: TransactionId,
    pub base_revision: u64,
    pub revision: u64,
    pub request_digest: String,
    pub committed_at_ms: i64,
    pub events: Vec<RecordedEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadedSession {
    pub metadata: StoredSessionMetadata,
    pub transactions: Vec<CommitReceipt>,
}

impl LoadedSession {
    #[cfg(test)]
    pub fn events(&self) -> impl Iterator<Item = &RecordedEvent> {
        self.transactions
            .iter()
            .flat_map(|transaction| transaction.events.iter())
    }
}

#[derive(Clone)]
pub struct SessionStore {
    inner: Arc<SessionStoreInner>,
}

struct SessionStoreInner {
    connection: Mutex<Connection>,
    database_identity: DatabaseIdentity,
    writable: bool,
    #[cfg(test)]
    failpoint: AtomicU8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DatabaseIdentity {
    Disk(PathBuf),
    Memory(u64),
}

/// Unforgeable authority to mutate one session in one database.
///
/// Clones share one guard. The last clone releases both the process-local
/// reservation and, for disk stores, the operating system advisory lock.
#[derive(Clone)]
pub(crate) struct SessionWriterPermit {
    inner: Arc<SessionWriterGuard>,
}

struct SessionWriterGuard {
    permit_id: u64,
    key: WriterRegistryKey,
    file: Option<File>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct WriterRegistryKey {
    database: DatabaseIdentity,
    session_id: SessionId,
}

struct WriterRegistryEntry {
    permit_id: u64,
    guard: Weak<SessionWriterGuard>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DatabaseAccess {
    Writable,
    Readonly,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DatabaseRegistryKey {
    path: PathBuf,
    access: DatabaseAccess,
}

static DATABASE_REGISTRY: OnceLock<Mutex<HashMap<DatabaseRegistryKey, Weak<SessionStoreInner>>>> =
    OnceLock::new();
static WRITER_REGISTRY: OnceLock<Mutex<HashMap<WriterRegistryKey, WriterRegistryEntry>>> =
    OnceLock::new();
static NEXT_MEMORY_DATABASE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_WRITER_PERMIT_ID: AtomicU64 = AtomicU64::new(1);

impl Drop for SessionWriterGuard {
    fn drop(&mut self) {
        if let Some(file) = &self.file {
            let _ = File::unlock(file);
        }
        let Some(registry) = WRITER_REGISTRY.get() else {
            return;
        };
        let Ok(mut registry) = registry.lock() else {
            return;
        };
        if registry
            .get(&self.key)
            .is_some_and(|entry| entry.permit_id == self.permit_id)
        {
            registry.remove(&self.key);
        }
    }
}

impl SessionStore {
    pub fn open_project(directory: impl AsRef<Path>) -> Result<Self, SessionStoreError> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        Self::open_database(directory.join(SESSION_DATABASE_FILE))
    }

    pub fn open_database(path: impl AsRef<Path>) -> Result<Self, SessionStoreError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let path = normalized_database_path(&path, false)?;
        Self::open_registered_database(path, DatabaseAccess::Writable)
    }

    pub fn open_database_readonly(path: impl AsRef<Path>) -> Result<Self, SessionStoreError> {
        let path = normalized_database_path(path.as_ref(), true)?;
        Self::open_registered_database(path, DatabaseAccess::Readonly)
    }

    pub fn open_in_memory() -> Result<Self, SessionStoreError> {
        Self::from_writable_connection(Connection::open_in_memory()?, None, false)
    }

    fn open_registered_database(
        path: PathBuf,
        access: DatabaseAccess,
    ) -> Result<Self, SessionStoreError> {
        let key = DatabaseRegistryKey {
            path: path.clone(),
            access,
        };
        let registry = DATABASE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry
            .lock()
            .map_err(|_| SessionStoreError::LockPoisoned)?;
        if let Some(inner) = registry.get(&key).and_then(Weak::upgrade) {
            return Ok(Self { inner });
        }

        let store = match access {
            DatabaseAccess::Writable => {
                let connection = Connection::open(&path)?;
                Self::from_writable_connection(connection, Some(path), true)?
            }
            DatabaseAccess::Readonly => {
                let connection = Connection::open_with_flags(
                    &path,
                    OpenFlags::SQLITE_OPEN_READ_ONLY
                        | OpenFlags::SQLITE_OPEN_URI
                        | OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )?;
                Self::from_readonly_connection(connection, path)?
            }
        };
        registry.insert(key, Arc::downgrade(&store.inner));
        Ok(store)
    }

    fn from_writable_connection(
        mut connection: Connection,
        database_path: Option<PathBuf>,
        disk: bool,
    ) -> Result<Self, SessionStoreError> {
        let database_path = database_path
            .map(|path| normalized_database_path(&path, true))
            .transpose()?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        initialize_or_validate_schema(&mut connection)?;
        if disk {
            connection.pragma_update(None, "journal_mode", "WAL")?;
            connection.pragma_update(None, "synchronous", "FULL")?;
        }
        let database_identity = database_path.as_ref().map_or_else(
            || DatabaseIdentity::Memory(NEXT_MEMORY_DATABASE_ID.fetch_add(1, Ordering::Relaxed)),
            |path| DatabaseIdentity::Disk(path.clone()),
        );
        Ok(Self {
            inner: Arc::new(SessionStoreInner {
                connection: Mutex::new(connection),
                database_identity,
                writable: true,
                #[cfg(test)]
                failpoint: AtomicU8::new(FAILPOINT_NONE),
            }),
        })
    }

    fn from_readonly_connection(
        connection: Connection,
        database_path: PathBuf,
    ) -> Result<Self, SessionStoreError> {
        connection.busy_timeout(Duration::from_secs(5))?;
        require_schema_version(&connection)?;
        Ok(Self {
            inner: Arc::new(SessionStoreInner {
                connection: Mutex::new(connection),
                database_identity: DatabaseIdentity::Disk(database_path.clone()),
                writable: false,
                #[cfg(test)]
                failpoint: AtomicU8::new(FAILPOINT_NONE),
            }),
        })
    }

    #[cfg(test)]
    fn database_path(&self) -> Option<&Path> {
        match &self.inner.database_identity {
            DatabaseIdentity::Disk(path) => Some(path),
            DatabaseIdentity::Memory(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn inject_failpoint(&self, failpoint: AppendFailpoint) {
        let value = match failpoint {
            AppendFailpoint::BeforeCommitOnce => FAILPOINT_BEFORE_COMMIT,
            AppendFailpoint::AfterCommitBeforeReceiptOnce => FAILPOINT_AFTER_COMMIT,
            AppendFailpoint::PauseBeforeCommit => FAILPOINT_PAUSE_BEFORE_COMMIT,
        };
        self.inner.failpoint.store(value, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn pause_resolve_after_revision(&self) {
        self.inner
            .failpoint
            .store(FAILPOINT_PAUSE_RESOLVE_AFTER_REVISION, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn resolve_pause_reached(&self) -> bool {
        self.inner.failpoint.load(Ordering::SeqCst) == FAILPOINT_RESOLVE_PAUSED
    }

    #[cfg(test)]
    fn resume_resolve(&self) {
        let _ = self.inner.failpoint.compare_exchange(
            FAILPOINT_RESOLVE_PAUSED,
            FAILPOINT_NONE,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    pub fn create_session(
        &self,
        request: CreateStoredSession,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        validate_session_id(&request.id)?;
        validate_nonempty("project ID", &request.project_id)?;
        let title = normalize_title(&request.title)?;
        let config = serde_json::to_vec(&request.config)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute(
            "INSERT INTO sessions (
                id, project_id, title, config_json, created_at_ms, updated_at_ms,
                archived_at_ms, revision, next_event_seq
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?5, NULL, 0, 0)",
            params![
                request.id.as_str(),
                request.project_id,
                title,
                config,
                request.created_at_ms
            ],
        )?;
        let session_key = positive_sql_i64(transaction.last_insert_rowid(), "session key")?;
        transaction.execute(
            "INSERT INTO session_catalog_projection (
                session_key, indexed_revision, extractor_version, loadability_version, valid
             ) VALUES (?1, 0, ?2, ?3, 1)",
            params![
                session_key,
                CATALOG_EXTRACTOR_VERSION,
                CATALOG_LOADABILITY_VERSION
            ],
        )?;
        transaction.commit()?;
        self.metadata_with_connection(&connection, &request.id)
    }

    pub fn metadata(
        &self,
        session_id: &SessionId,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        let connection = self.connection()?;
        self.metadata_with_connection(&connection, session_id)
    }

    /// Acquires the exclusive mutation capability for one session.
    ///
    /// Disk stores use an OS advisory lock, so process exit releases ownership
    /// even if no application cleanup runs. The lockfile is a permanent inode;
    /// its existence never means that it is locked and it is never unlinked.
    pub(crate) fn acquire_writer(
        &self,
        session_id: &SessionId,
    ) -> Result<SessionWriterPermit, SessionStoreError> {
        if !self.inner.writable {
            return Err(SessionStoreError::ReadonlyStore);
        }
        {
            let connection = self.connection()?;
            ensure_session_exists(&connection, session_id)?;
        }

        let key = WriterRegistryKey {
            database: self.inner.database_identity.clone(),
            session_id: session_id.clone(),
        };
        let registry = WRITER_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry
            .lock()
            .map_err(|_| SessionStoreError::LockPoisoned)?;
        if registry
            .get(&key)
            .and_then(|entry| entry.guard.upgrade())
            .is_some()
        {
            return Err(SessionStoreError::WriterBusy {
                session_id: session_id.clone(),
            });
        }

        let file = match &self.inner.database_identity {
            DatabaseIdentity::Disk(database_path) => {
                let parent = database_path.parent().ok_or_else(|| {
                    SessionStoreError::Invalid(format!(
                        "database path has no parent: {}",
                        database_path.display()
                    ))
                })?;
                let lock_directory = parent.join(".session-writers");
                fs::create_dir_all(&lock_directory)?;
                let lock_path = lock_directory.join(writer_lock_file_name(session_id)?);
                let file = OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(lock_path)?;
                match file.try_lock() {
                    Ok(()) => Some(file),
                    Err(TryLockError::WouldBlock) => {
                        return Err(SessionStoreError::WriterBusy {
                            session_id: session_id.clone(),
                        });
                    }
                    Err(TryLockError::Error(error)) => return Err(error.into()),
                }
            }
            DatabaseIdentity::Memory(_) => None,
        };

        let permit_id = NEXT_WRITER_PERMIT_ID.fetch_add(1, Ordering::Relaxed);
        let inner = Arc::new(SessionWriterGuard {
            permit_id,
            key: key.clone(),
            file,
        });
        registry.insert(
            key,
            WriterRegistryEntry {
                permit_id,
                guard: Arc::downgrade(&inner),
            },
        );
        drop(registry);
        let permit = SessionWriterPermit { inner };
        let existence = {
            let connection = self.connection()?;
            ensure_session_exists(&connection, session_id)
        };
        if let Err(error) = existence {
            drop(permit);
            return Err(error);
        }
        Ok(permit)
    }

    pub(crate) fn update_metadata(
        &self,
        session_id: &SessionId,
        update: MetadataUpdate,
        permit: &SessionWriterPermit,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        self.validate_writer(permit, session_id)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let mut metadata = self.metadata_with_connection(&transaction, session_id)?;
        if let Some(project_id) = update.project_id {
            validate_nonempty("project ID", &project_id)?;
            metadata.project_id = project_id;
        }
        if let Some(title) = update.title {
            metadata.title = normalize_title(&title)?;
        }
        if let Some(config) = update.config {
            metadata.config = config;
        }
        metadata.updated_at_ms = now_millis();
        transaction.execute(
            "UPDATE sessions
             SET project_id = ?2, title = ?3, config_json = ?4, updated_at_ms = ?5
             WHERE id = ?1",
            params![
                session_id.as_str(),
                metadata.project_id,
                metadata.title,
                serde_json::to_vec(&metadata.config)?,
                metadata.updated_at_ms
            ],
        )?;
        transaction.commit()?;
        Ok(metadata)
    }

    pub(crate) fn archive(
        &self,
        session_id: &SessionId,
        permit: &SessionWriterPermit,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        self.set_archived(session_id, Some(now_millis()), permit)
    }

    pub(crate) fn restore(
        &self,
        session_id: &SessionId,
        permit: &SessionWriterPermit,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        self.set_archived(session_id, None, permit)
    }

    fn set_archived(
        &self,
        session_id: &SessionId,
        archived_at_ms: Option<i64>,
        permit: &SessionWriterPermit,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        self.validate_writer(permit, session_id)?;
        let connection = self.connection()?;
        let changed = connection.execute(
            "UPDATE sessions SET archived_at_ms = ?2, updated_at_ms = ?3 WHERE id = ?1",
            params![session_id.as_str(), archived_at_ms, now_millis()],
        )?;
        if changed == 0 {
            return Err(SessionStoreError::SessionNotFound(session_id.clone()));
        }
        self.metadata_with_connection(&connection, session_id)
    }

    pub(crate) fn delete(
        &self,
        session_id: &SessionId,
        permit: &SessionWriterPermit,
    ) -> Result<(), SessionStoreError> {
        self.validate_writer(permit, session_id)?;
        let connection = self.connection()?;
        let changed = connection.execute(
            "DELETE FROM sessions WHERE id = ?1",
            params![session_id.as_str()],
        )?;
        if changed == 0 {
            return Err(SessionStoreError::SessionNotFound(session_id.clone()));
        }
        Ok(())
    }

    pub fn catalog(
        &self,
        project_id: &str,
        archive_filter: ArchiveFilter,
    ) -> Result<Vec<StoredCatalogEntry>, SessionStoreError> {
        let mut connection = self.connection()?;
        let snapshot = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let suffix = match archive_filter {
            ArchiveFilter::Active => "AND s.archived_at_ms IS NULL",
            ArchiveFilter::Archived => "AND s.archived_at_ms IS NOT NULL",
        };
        let sql = format!(
            "SELECT s.id, s.project_id, s.title, s.config_json, s.created_at_ms, s.updated_at_ms,
                    s.archived_at_ms, s.revision, f.value
             FROM sessions AS s
             JOIN session_catalog_projection AS p
               ON p.session_key = s.session_key
              AND p.indexed_revision = s.revision
              AND p.extractor_version = ?2
              AND p.loadability_version = ?3
              AND p.valid = 1
             LEFT JOIN session_search_fragments AS f
               ON f.session_key = s.session_key AND typeof(f.value) = 'text'
             WHERE s.project_id = ?1 {suffix}
               AND typeof(s.id) = 'text'
               AND typeof(s.project_id) = 'text'
               AND typeof(s.title) = 'text'
               AND typeof(s.config_json) = 'blob'
               AND typeof(s.created_at_ms) = 'integer'
               AND typeof(s.updated_at_ms) = 'integer'
               AND (s.archived_at_ms IS NULL OR typeof(s.archived_at_ms) = 'integer')
               AND typeof(s.revision) = 'integer'
             ORDER BY s.created_at_ms DESC, s.id ASC, f.event_seq ASC, f.ordinal ASC"
        );
        let mut statement = snapshot.prepare(&sql)?;
        let rows = statement.query_map(
            params![
                project_id,
                CATALOG_EXTRACTOR_VERSION,
                CATALOG_LOADABILITY_VERSION
            ],
            |row| {
                Ok((
                    raw_metadata_from_row(row)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            },
        )?;
        let mut sessions = Vec::new();
        let mut current: Option<(RawSessionMetadata, Vec<String>)> = None;
        for row in rows {
            let (metadata, value) = row?;
            if current
                .as_ref()
                .is_some_and(|(active, _)| active.id != metadata.id)
            {
                let (metadata, search_values) = current.take().expect("checked as present");
                if let Ok(metadata) = metadata.try_into() {
                    sessions.push(StoredCatalogEntry {
                        metadata,
                        search_values,
                    });
                }
            }
            let (_, search_values) = current.get_or_insert_with(|| (metadata, Vec::new()));
            if let Some(value) = value {
                search_values.push(value);
            }
        }
        if let Some((metadata, search_values)) = current
            && let Ok(metadata) = metadata.try_into()
        {
            sessions.push(StoredCatalogEntry {
                metadata,
                search_values,
            });
        }
        drop(statement);
        snapshot.commit()?;
        Ok(sessions)
    }

    pub(crate) fn append(
        &self,
        request: &AppendTx,
        permit: &SessionWriterPermit,
    ) -> Result<CommitReceipt, SessionStoreError> {
        self.validate_writer(permit, &request.session_id)?;
        if request.events.is_empty() {
            return Err(SessionStoreError::EmptyTransaction);
        }
        if request.tx_id.is_empty() {
            return Err(SessionStoreError::Invalid(
                "transaction ID must not be empty".into(),
            ));
        }
        if request
            .events
            .iter()
            .any(|event| event.tx_id != request.tx_id)
        {
            return Err(SessionStoreError::Invalid(
                "every event draft must carry the enclosing transaction ID".into(),
            ));
        }
        let clock_id = &request.events[0].time.clock_id;
        validate_nonempty("clock ID", clock_id)?;
        if request
            .events
            .iter()
            .any(|event| event.time.clock_id != *clock_id)
        {
            return Err(SessionStoreError::Invalid(
                "events in one transaction must share one clock ID".into(),
            ));
        }
        let request_digest = request_digest(request.expected_revision, &request.events)?;
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;

        let current = transaction
            .query_row(
                "SELECT session_key, revision, next_event_seq FROM sessions WHERE id = ?1",
                params![request.session_id.as_str()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| SessionStoreError::SessionNotFound(request.session_id.clone()))?;
        let session_key = positive_sql_i64(current.0, "session key")?;
        let current_revision = from_sql_u64(current.1, "session revision")?;
        let first_event_seq = from_sql_u64(current.2, "next event sequence")?;

        if let Some(existing_request) = transaction
            .query_row(
                "SELECT base_revision, request_digest
                 FROM journal_transactions
                 WHERE session_key = ?1 AND tx_id = ?2",
                params![session_key, request.tx_id.as_str()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?
        {
            let existing_base_revision =
                from_sql_u64(existing_request.0, "transaction base revision")?;
            if existing_base_revision != request.expected_revision
                || existing_request.1 != request_digest
            {
                return Err(SessionStoreError::TransactionConflict {
                    tx_id: request.tx_id.clone(),
                });
            }
            let receipt = load_receipt_from_connection(
                &transaction,
                &request.session_id,
                session_key,
                &request.tx_id,
            )?;
            validate_retry_matches(request, &receipt)?;
            ensure_catalog_projection_current(&transaction, session_key, current_revision)?;
            return Ok(receipt);
        }

        if current_revision != request.expected_revision {
            return Err(SessionStoreError::RevisionConflict {
                session_id: request.session_id.clone(),
                expected_revision: request.expected_revision,
                current_revision,
            });
        }
        if let Some((expected, found)) =
            request
                .events
                .iter()
                .enumerate()
                .find_map(|(index, event)| {
                    let expected = first_event_seq + index as u64;
                    (event.seq != expected).then_some((expected, event.seq))
                })
        {
            return Err(SessionStoreError::EventSequenceConflict { expected, found });
        }
        let revision = current_revision
            .checked_add(1)
            .ok_or(SessionStoreError::NumericOverflow(u64::MAX))?;
        let next_event_seq = first_event_seq
            .checked_add(request.events.len() as u64)
            .ok_or(SessionStoreError::NumericOverflow(u64::MAX))?;
        let committed_at_ms = now_millis();

        let changed = transaction.execute(
            "UPDATE sessions
             SET revision = ?2, next_event_seq = ?3, updated_at_ms = ?4
             WHERE session_key = ?1 AND revision = ?5",
            params![
                session_key,
                to_sql_i64(revision)?,
                to_sql_i64(next_event_seq)?,
                committed_at_ms,
                to_sql_i64(current_revision)?
            ],
        )?;
        if changed != 1 {
            return Err(SessionStoreError::RevisionConflict {
                session_id: request.session_id.clone(),
                expected_revision: request.expected_revision,
                current_revision,
            });
        }

        transaction.execute(
            "INSERT INTO journal_transactions (
                session_key, revision, tx_id, base_revision, first_event_seq, event_count,
                clock_id, request_digest, committed_at_ms
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session_key,
                to_sql_i64(revision)?,
                request.tx_id.as_str(),
                to_sql_i64(current_revision)?,
                to_sql_i64(first_event_seq)?,
                to_sql_i64(request.events.len() as u64)?,
                clock_id,
                request_digest,
                committed_at_ms
            ],
        )?;

        let mut recorded_events = Vec::with_capacity(request.events.len());
        {
            let mut insert_event = transaction.prepare_cached(
                "INSERT INTO journal_events (
                    session_key, seq, transaction_revision, ordinal, wall_time_ms,
                    monotonic_ns, event_json
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for (ordinal, recorded) in request.events.iter().enumerate() {
                let event_json = serde_json::to_vec(&recorded.event)?;
                insert_event.execute(params![
                    session_key,
                    to_sql_i64(recorded.seq)?,
                    to_sql_i64(revision)?,
                    to_sql_i64(ordinal as u64)?,
                    recorded.time.wall_time_ms,
                    to_sql_i64(recorded.time.monotonic_ns)?,
                    event_json
                ])?;
                recorded_events.push(recorded.clone());
            }
        }

        update_catalog_projection(
            &transaction,
            session_key,
            current_revision,
            revision,
            &request.events,
        )?;

        #[cfg(test)]
        if self.consume_failpoint(FAILPOINT_PAUSE_BEFORE_COMMIT) {
            let ready_path =
                std::env::var_os("KCASTLE_TEST_APPEND_PAUSE_READY").ok_or_else(|| {
                    SessionStoreError::Invalid("missing subprocess ready path".into())
                })?;
            File::create(ready_path)?.sync_all()?;
            loop {
                std::thread::park();
            }
        }

        #[cfg(test)]
        if self.consume_failpoint(FAILPOINT_BEFORE_COMMIT) {
            return Err(SessionStoreError::InjectedBeforeCommit);
        }
        transaction.commit()?;

        #[cfg(test)]
        if self.consume_failpoint(FAILPOINT_AFTER_COMMIT) {
            return Err(SessionStoreError::OutcomeUnknown {
                tx_id: request.tx_id.clone(),
            });
        }
        Ok(CommitReceipt {
            session_id: request.session_id.clone(),
            tx_id: request.tx_id.clone(),
            base_revision: current_revision,
            revision,
            request_digest,
            committed_at_ms,
            events: recorded_events,
        })
    }

    pub fn resolve(
        &self,
        session_id: &SessionId,
        tx_id: &TransactionId,
    ) -> Result<Option<CommitReceipt>, SessionStoreError> {
        let mut connection = self.connection()?;
        let snapshot = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let session_key = session_key_with_connection(&snapshot, session_id)?;
        match load_receipt_from_connection(&snapshot, session_id, session_key, tx_id) {
            Ok(receipt) => {
                let revision = snapshot.query_row(
                    "SELECT revision FROM sessions WHERE session_key = ?1",
                    params![session_key],
                    |row| row.get::<_, i64>(0),
                )?;
                #[cfg(test)]
                self.pause_resolve_if_requested();
                ensure_catalog_projection_current(
                    &snapshot,
                    session_key,
                    from_sql_u64(revision, "session revision")?,
                )?;
                snapshot.commit()?;
                Ok(Some(receipt))
            }
            Err(SessionStoreError::TransactionNotFound { .. }) => {
                snapshot.commit()?;
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    pub fn load(&self, session_id: &SessionId) -> Result<LoadedSession, SessionStoreError> {
        let mut connection = self.connection()?;
        let snapshot = connection.transaction_with_behavior(TransactionBehavior::Deferred)?;
        let metadata = self.metadata_with_connection(&snapshot, session_id)?;
        let (session_key, stored_next_event_seq) = snapshot
            .query_row(
                "SELECT session_key, next_event_seq FROM sessions WHERE id = ?1",
                params![session_id.as_str()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?
            .ok_or_else(|| SessionStoreError::SessionNotFound(session_id.clone()))?;
        let session_key = positive_sql_i64(session_key, "session key")?;
        let stored_next_event_seq = from_sql_u64(stored_next_event_seq, "next event sequence")?;
        ensure_catalog_projection_loadable(&snapshot, session_key)?;
        let mut statement = snapshot.prepare(
            "SELECT tx_id, base_revision, revision, clock_id, request_digest,
                    committed_at_ms, first_event_seq, event_count
             FROM journal_transactions
             WHERE session_key = ?1
             ORDER BY revision ASC",
        )?;
        let transaction_rows = statement.query_map(params![session_key], |row| {
            Ok(RawTransaction {
                tx_id: row.get(0)?,
                base_revision: row.get(1)?,
                revision: row.get(2)?,
                clock_id: row.get(3)?,
                request_digest: row.get(4)?,
                committed_at_ms: row.get(5)?,
                first_event_seq: row.get(6)?,
                event_count: row.get(7)?,
            })
        })?;
        let mut decoders = Vec::new();
        let mut revision_indexes = HashMap::new();
        for raw in transaction_rows {
            let decoder = TransactionDecoder::new(session_id, raw?)?;
            revision_indexes.insert(decoder.revision(), decoders.len());
            decoders.push(decoder);
        }
        drop(statement);

        let mut event_statement = snapshot.prepare(
            "SELECT transaction_revision, ordinal, seq, wall_time_ms, monotonic_ns, event_json
             FROM journal_events
             WHERE session_key = ?1
             ORDER BY seq ASC",
        )?;
        let event_rows =
            event_statement.query_map(params![session_key], raw_stored_event_from_row)?;
        for row in event_rows {
            let raw = row?;
            let revision = from_sql_u64(raw.transaction_revision, "event transaction revision")?;
            let Some(index) = revision_indexes.get(&revision).copied() else {
                return Err(SessionStoreError::Corrupt(format!(
                    "event references missing transaction revision {revision}"
                )));
            };
            decoders[index].push(raw)?;
        }
        drop(event_statement);

        let transactions = decoders
            .into_iter()
            .map(TransactionDecoder::finish)
            .collect::<Result<Vec<_>, _>>()?;

        validate_loaded_transactions(&transactions)?;
        if transactions
            .last()
            .map_or(0, |transaction| transaction.revision)
            != metadata.revision
        {
            return Err(SessionStoreError::Corrupt(format!(
                "session revision {} does not match journal tail",
                metadata.revision
            )));
        }
        let journal_next_event_seq = match transactions
            .last()
            .and_then(|transaction| transaction.events.last())
        {
            Some(event) => event.seq.checked_add(1).ok_or_else(|| {
                SessionStoreError::Corrupt("journal event sequence overflows u64".into())
            })?,
            None => 0,
        };
        if stored_next_event_seq != journal_next_event_seq {
            return Err(SessionStoreError::Corrupt(format!(
                "session next event sequence {stored_next_event_seq} does not match journal tail {journal_next_event_seq}"
            )));
        }
        snapshot.commit()?;
        Ok(LoadedSession {
            metadata,
            transactions,
        })
    }

    pub fn export_jsonl<W: Write>(
        &self,
        session_id: &SessionId,
        mut writer: W,
    ) -> Result<(), SessionStoreError> {
        let loaded = self.load(session_id)?;
        write_json_line(
            &mut writer,
            &JsonlExportRecord::Session {
                format_version: JSONL_STORE_FORMAT_VERSION,
                metadata: &loaded.metadata,
            },
        )?;
        for transaction in &loaded.transactions {
            write_json_line(&mut writer, &JsonlExportRecord::Transaction { transaction })?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn export_jsonl_to_path(
        &self,
        session_id: &SessionId,
        path: impl AsRef<Path>,
    ) -> Result<(), SessionStoreError> {
        let file = OpenOptions::new().create_new(true).write(true).open(path)?;
        self.export_jsonl(session_id, file)
    }

    fn metadata_with_connection(
        &self,
        connection: &Connection,
        session_id: &SessionId,
    ) -> Result<StoredSessionMetadata, SessionStoreError> {
        let raw = connection
            .query_row(
                "SELECT id, project_id, title, config_json, created_at_ms, updated_at_ms,
                        archived_at_ms, revision
                 FROM sessions WHERE id = ?1",
                params![session_id.as_str()],
                raw_metadata_from_row,
            )
            .optional()?
            .ok_or_else(|| SessionStoreError::SessionNotFound(session_id.clone()))?;
        raw.try_into()
    }

    fn connection(&self) -> Result<MutexGuard<'_, Connection>, SessionStoreError> {
        self.inner
            .connection
            .lock()
            .map_err(|_| SessionStoreError::LockPoisoned)
    }

    fn validate_writer(
        &self,
        permit: &SessionWriterPermit,
        session_id: &SessionId,
    ) -> Result<(), SessionStoreError> {
        if !self.inner.writable {
            return Err(SessionStoreError::ReadonlyStore);
        }
        let authorized = permit.inner.key.session_id == *session_id
            && permit.inner.key.database == self.inner.database_identity;
        if authorized {
            Ok(())
        } else {
            Err(SessionStoreError::InvalidWriterPermit {
                session_id: session_id.clone(),
            })
        }
    }

    #[cfg(test)]
    fn consume_failpoint(&self, failpoint: u8) -> bool {
        self.inner
            .failpoint
            .compare_exchange(
                failpoint,
                FAILPOINT_NONE,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
    }

    #[cfg(test)]
    fn pause_resolve_if_requested(&self) {
        if self
            .inner
            .failpoint
            .compare_exchange(
                FAILPOINT_PAUSE_RESOLVE_AFTER_REVISION,
                FAILPOINT_RESOLVE_PAUSED,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            while self.inner.failpoint.load(Ordering::SeqCst) == FAILPOINT_RESOLVE_PAUSED {
                std::thread::yield_now();
            }
        }
    }
}

fn ensure_catalog_projection_current(
    connection: &Connection,
    session_key: i64,
    revision: u64,
) -> Result<(), SessionStoreError> {
    let projection = connection
        .query_row(
            "SELECT indexed_revision, extractor_version, loadability_version, valid
             FROM session_catalog_projection
             WHERE session_key = ?1",
            params![session_key],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .optional()?;
    let Some((indexed_revision, extractor_version, loadability_version, valid)) = projection else {
        return Err(SessionStoreError::Corrupt(format!(
            "session {session_key} is missing its catalog projection"
        )));
    };
    let indexed_revision = from_sql_u64(indexed_revision, "catalog indexed revision")?;
    if indexed_revision != revision
        || extractor_version != CATALOG_EXTRACTOR_VERSION
        || loadability_version != CATALOG_LOADABILITY_VERSION
        || valid != 1
    {
        return Err(SessionStoreError::Corrupt(format!(
            "session {session_key} catalog projection is not current and loadable (journal revision {revision}, indexed revision {indexed_revision}, extractor {extractor_version}, loadability {loadability_version}, valid {valid})"
        )));
    }
    Ok(())
}

fn ensure_catalog_projection_loadable(
    connection: &Connection,
    session_key: i64,
) -> Result<(), SessionStoreError> {
    let loadability_version = connection
        .query_row(
            "SELECT loadability_version
             FROM session_catalog_projection
             WHERE session_key = ?1",
            params![session_key],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    let Some(loadability_version) = loadability_version else {
        return Err(SessionStoreError::Corrupt(format!(
            "session {session_key} is missing its catalog projection"
        )));
    };
    if loadability_version != CATALOG_LOADABILITY_VERSION {
        return Err(SessionStoreError::Corrupt(format!(
            "session {session_key} has incompatible loadability version {loadability_version}; expected {CATALOG_LOADABILITY_VERSION}"
        )));
    }
    Ok(())
}

fn update_catalog_projection(
    transaction: &rusqlite::Transaction<'_>,
    session_key: i64,
    current_revision: u64,
    revision: u64,
    events: &[RecordedEvent],
) -> Result<(), SessionStoreError> {
    ensure_catalog_projection_current(transaction, session_key, current_revision)?;

    let mut insert_fragment = transaction.prepare_cached(
        "INSERT INTO session_search_fragments (session_key, event_seq, ordinal, value)
         VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut insert_draft = transaction.prepare_cached(
        "INSERT INTO session_request_draft_fragments (
            session_key, request_id, event_seq, ordinal
         ) VALUES (?1, ?2, ?3, ?4)",
    )?;
    let mut delete_request_drafts = transaction.prepare_cached(
        "DELETE FROM session_search_fragments AS fragment
         WHERE fragment.session_key = ?1
           AND EXISTS (
               SELECT 1
               FROM session_request_draft_fragments AS draft
               WHERE draft.session_key = fragment.session_key
                 AND draft.event_seq = fragment.event_seq
                 AND draft.ordinal = fragment.ordinal
                 AND draft.request_id = ?2
           )",
    )?;

    for recorded in events {
        let event_seq = to_sql_i64(recorded.seq)?;
        let mut add_fragment = |ordinal: u64,
                                value: &str,
                                draft_request_id: Option<&str>|
         -> Result<(), SessionStoreError> {
            if value.trim().is_empty() {
                return Ok(());
            }
            let ordinal = to_sql_i64(ordinal)?;
            insert_fragment.execute(params![session_key, event_seq, ordinal, value])?;
            if let Some(request_id) = draft_request_id {
                insert_draft.execute(params![session_key, request_id, event_seq, ordinal])?;
            }
            Ok(())
        };

        match &recorded.event {
            SessionEvent::InputSubmitted { input, .. } => add_fragment(0, input, None)?,
            SessionEvent::AssistantChunk { request_id, chunk } => match chunk {
                AssistantChunk::OutputTextDelta { delta }
                | AssistantChunk::ReasoningTextDelta { delta } => {
                    add_fragment(0, delta, Some(request_id.as_str()))?;
                }
                AssistantChunk::ToolCallDelta {
                    name,
                    arguments_delta,
                    ..
                } => {
                    if let Some(name) = name {
                        add_fragment(0, name, Some(request_id.as_str()))?;
                    }
                    add_fragment(1, arguments_delta, Some(request_id.as_str()))?;
                }
                AssistantChunk::Usage { .. } => {}
            },
            SessionEvent::AssistantCompleted {
                request_id, items, ..
            } => {
                // The completed response is canonical. Draft deltas are only a transient search
                // surface and must not coexist with the final items.
                delete_request_drafts.execute(params![session_key, request_id.as_str()])?;
                let payload = serde_json::to_string(items)?;
                add_fragment(0, &payload, None)?;
            }
            SessionEvent::ToolResultAttached { item, .. } => {
                let payload = serde_json::to_string(item)?;
                add_fragment(0, &payload, None)?;
            }
            SessionEvent::CompactionFinished {
                summary: Some(summary),
                ..
            } => add_fragment(0, summary, None)?,
            SessionEvent::ModelRequestFailed { error, .. }
            | SessionEvent::RunTerminated {
                error: Some(error), ..
            }
            | SessionEvent::StepTerminated {
                error: Some(error), ..
            } => add_fragment(0, error, None)?,
            _ => {}
        }
    }
    drop(delete_request_drafts);
    drop(insert_draft);
    drop(insert_fragment);

    let changed = transaction.execute(
        "UPDATE session_catalog_projection
         SET indexed_revision = ?2
         WHERE session_key = ?1
           AND indexed_revision = ?3
           AND extractor_version = ?4
           AND loadability_version = ?5
           AND valid = 1",
        params![
            session_key,
            to_sql_i64(revision)?,
            to_sql_i64(current_revision)?,
            CATALOG_EXTRACTOR_VERSION,
            CATALOG_LOADABILITY_VERSION
        ],
    )?;
    if changed != 1 {
        return Err(SessionStoreError::Corrupt(format!(
            "session {session_key} catalog projection did not advance from revision {current_revision}"
        )));
    }
    Ok(())
}

#[derive(Serialize)]
struct RequestDigestMaterial<'a> {
    base_revision: u64,
    events: &'a [RecordedEvent],
}

#[derive(Debug)]
struct RawSessionMetadata {
    id: String,
    project_id: String,
    title: String,
    config_json: Vec<u8>,
    created_at_ms: i64,
    updated_at_ms: i64,
    archived_at_ms: Option<i64>,
    revision: i64,
}

impl TryFrom<RawSessionMetadata> for StoredSessionMetadata {
    type Error = SessionStoreError;

    fn try_from(raw: RawSessionMetadata) -> Result<Self, Self::Error> {
        Ok(Self {
            id: session_id_from_string(raw.id)?,
            project_id: raw.project_id,
            title: raw.title,
            config: serde_json::from_slice(&raw.config_json)?,
            created_at_ms: raw.created_at_ms,
            updated_at_ms: raw.updated_at_ms,
            archived_at_ms: raw.archived_at_ms,
            revision: from_sql_u64(raw.revision, "session revision")?,
        })
    }
}

fn raw_metadata_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawSessionMetadata> {
    Ok(RawSessionMetadata {
        id: row.get(0)?,
        project_id: row.get(1)?,
        title: row.get(2)?,
        config_json: row.get(3)?,
        created_at_ms: row.get(4)?,
        updated_at_ms: row.get(5)?,
        archived_at_ms: row.get(6)?,
        revision: row.get(7)?,
    })
}

fn session_id_from_string(value: String) -> Result<SessionId, SessionStoreError> {
    let id = SessionId::from_raw(value);
    validate_session_id(&id)?;
    Ok(id)
}

fn validate_session_id(session_id: &SessionId) -> Result<(), SessionStoreError> {
    if session_id.is_storage_safe() {
        Ok(())
    } else {
        Err(SessionStoreError::Invalid(format!(
            "session ID must be 1-128 ASCII letters, digits, '-' or '_': {:?}",
            session_id.as_str()
        )))
    }
}

fn session_key_with_connection(
    connection: &Connection,
    session_id: &SessionId,
) -> Result<i64, SessionStoreError> {
    let key = connection
        .query_row(
            "SELECT session_key FROM sessions WHERE id = ?1",
            params![session_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .ok_or_else(|| SessionStoreError::SessionNotFound(session_id.clone()))?;
    positive_sql_i64(key, "session key")
}

fn ensure_session_exists(
    connection: &Connection,
    session_id: &SessionId,
) -> Result<(), SessionStoreError> {
    let exists = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sessions WHERE id = ?1)",
        params![session_id.as_str()],
        |row| row.get::<_, bool>(0),
    )?;
    if exists {
        Ok(())
    } else {
        Err(SessionStoreError::SessionNotFound(session_id.clone()))
    }
}

fn writer_lock_file_name(session_id: &SessionId) -> Result<String, SessionStoreError> {
    validate_session_id(session_id)?;
    Ok(format!("{}.lock", session_id.as_str()))
}

#[derive(Debug)]
struct RawTransaction {
    tx_id: String,
    base_revision: i64,
    revision: i64,
    clock_id: String,
    request_digest: String,
    committed_at_ms: i64,
    first_event_seq: i64,
    event_count: i64,
}

struct RawStoredEvent {
    transaction_revision: i64,
    ordinal: i64,
    seq: i64,
    wall_time_ms: i64,
    monotonic_ns: i64,
    event_json: Vec<u8>,
}

/// Builds one receipt from untrusted SQLite rows.
///
/// Full-session loading and exact transaction resolution deliberately use different query plans,
/// but they must accept and reject the same stored transaction shapes. Keeping the row validation
/// here prevents those two trust boundaries from drifting apart.
struct TransactionDecoder {
    receipt: CommitReceipt,
    first_event_seq: u64,
    expected_event_count: usize,
    clock_id: String,
}

impl TransactionDecoder {
    fn new(session_id: &SessionId, raw: RawTransaction) -> Result<Self, SessionStoreError> {
        if raw.clock_id.trim().is_empty() {
            return Err(corrupt_transaction(&raw.tx_id, "has an empty clock ID"));
        }
        let decoder = Self {
            receipt: CommitReceipt {
                session_id: session_id.clone(),
                tx_id: TransactionId::from_raw(raw.tx_id),
                base_revision: from_sql_u64(raw.base_revision, "base revision")?,
                revision: from_sql_u64(raw.revision, "transaction revision")?,
                request_digest: raw.request_digest,
                committed_at_ms: raw.committed_at_ms,
                // The declaration is untrusted allocation input. Grow only from rows SQLite
                // actually returns and compare the exact count in `finish`.
                events: Vec::new(),
            },
            first_event_seq: from_sql_u64(raw.first_event_seq, "first event sequence")?,
            expected_event_count: usize_from_sql(raw.event_count, "event count")?,
            clock_id: raw.clock_id,
        };
        decoder.event_range()?;
        Ok(decoder)
    }

    fn revision(&self) -> u64 {
        self.receipt.revision
    }

    fn event_range(&self) -> Result<(u64, u64), SessionStoreError> {
        let event_span = self
            .expected_event_count
            .checked_sub(1)
            .ok_or_else(|| corrupt_transaction(&self.receipt.tx_id, "declares no events"))?;
        let event_span = u64::try_from(event_span)
            .map_err(|_| corrupt_transaction(&self.receipt.tx_id, "event count is too large"))?;
        let last_event_seq = self
            .first_event_seq
            .checked_add(event_span)
            .ok_or_else(|| corrupt_transaction(&self.receipt.tx_id, "event range overflows"))?;
        Ok((self.first_event_seq, last_event_seq))
    }

    fn push(&mut self, raw: RawStoredEvent) -> Result<(), SessionStoreError> {
        let event_revision = from_sql_u64(raw.transaction_revision, "event transaction revision")?;
        let ordinal = usize_from_sql(raw.ordinal, "event transaction ordinal")?;
        if event_revision != self.receipt.revision || ordinal != self.receipt.events.len() {
            return Err(corrupt_transaction(
                &self.receipt.tx_id,
                "event range contains the wrong revision or ordinal",
            ));
        }
        self.receipt.events.push(RecordedEvent {
            seq: from_sql_u64(raw.seq, "event sequence")?,
            tx_id: self.receipt.tx_id.clone(),
            time: EventTime {
                wall_time_ms: raw.wall_time_ms,
                clock_id: self.clock_id.clone(),
                monotonic_ns: from_sql_u64(raw.monotonic_ns, "event monotonic timestamp")?,
            },
            event: serde_json::from_slice(&raw.event_json)?,
        });
        Ok(())
    }

    fn finish(self) -> Result<CommitReceipt, SessionStoreError> {
        if self.receipt.events.len() != self.expected_event_count
            || self
                .receipt
                .events
                .iter()
                .enumerate()
                .any(|(index, event)| event.seq != self.first_event_seq + index as u64)
        {
            return Err(corrupt_transaction(
                &self.receipt.tx_id,
                "does not contain its declared event range",
            ));
        }
        if request_digest(self.receipt.base_revision, &self.receipt.events)?
            != self.receipt.request_digest
        {
            return Err(corrupt_transaction(
                &self.receipt.tx_id,
                "request digest does not match its canonical events",
            ));
        }
        Ok(self.receipt)
    }
}

fn raw_stored_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawStoredEvent> {
    Ok(RawStoredEvent {
        transaction_revision: row.get(0)?,
        ordinal: row.get(1)?,
        seq: row.get(2)?,
        wall_time_ms: row.get(3)?,
        monotonic_ns: row.get(4)?,
        event_json: row.get(5)?,
    })
}

fn corrupt_transaction(tx_id: &impl std::fmt::Display, detail: &str) -> SessionStoreError {
    SessionStoreError::Corrupt(format!("transaction {tx_id} {detail}"))
}

fn load_receipt_from_connection(
    connection: &Connection,
    session_id: &SessionId,
    session_key: i64,
    tx_id: &TransactionId,
) -> Result<CommitReceipt, SessionStoreError> {
    let raw = connection
        .query_row(
            "SELECT base_revision, revision, clock_id, request_digest,
                    committed_at_ms, first_event_seq, event_count
             FROM journal_transactions
             WHERE session_key = ?1 AND tx_id = ?2",
            params![session_key, tx_id.as_str()],
            |row| {
                Ok(RawTransaction {
                    tx_id: tx_id.as_str().to_owned(),
                    base_revision: row.get(0)?,
                    revision: row.get(1)?,
                    clock_id: row.get(2)?,
                    request_digest: row.get(3)?,
                    committed_at_ms: row.get(4)?,
                    first_event_seq: row.get(5)?,
                    event_count: row.get(6)?,
                })
            },
        )
        .optional()?
        .ok_or_else(|| SessionStoreError::TransactionNotFound {
            session_id: session_id.clone(),
            tx_id: tx_id.clone(),
        })?;
    let mut decoder = TransactionDecoder::new(session_id, raw)?;
    let (first_event_seq, last_event_seq) = decoder.event_range()?;
    // The declared range is the primary-key range for one committed transaction, so ambiguous
    // commit resolution stays O(transaction events), not O(session history). `resolve` verifies
    // the catalog projection first; journal tampering outside this range invalidates that marker.
    let mut statement = connection.prepare(
        "SELECT transaction_revision, ordinal, seq, wall_time_ms, monotonic_ns, event_json
         FROM journal_events
         WHERE session_key = ?1 AND seq BETWEEN ?2 AND ?3
         ORDER BY seq ASC",
    )?;
    let rows = statement.query_map(
        params![
            session_key,
            to_sql_i64(first_event_seq)?,
            to_sql_i64(last_event_seq)?
        ],
        raw_stored_event_from_row,
    )?;
    for row in rows {
        decoder.push(row?)?;
    }
    decoder.finish()
}

fn validate_retry_matches(
    request: &AppendTx,
    receipt: &CommitReceipt,
) -> Result<(), SessionStoreError> {
    if receipt.base_revision == request.expected_revision && receipt.events == request.events {
        Ok(())
    } else {
        Err(SessionStoreError::TransactionConflict {
            tx_id: request.tx_id.clone(),
        })
    }
}

fn validate_loaded_transactions(transactions: &[CommitReceipt]) -> Result<(), SessionStoreError> {
    let mut expected_seq = 0_u64;
    for (expected_revision, transaction) in (1_u64..).zip(transactions) {
        if transaction.revision != expected_revision
            || transaction.base_revision != expected_revision - 1
        {
            return Err(SessionStoreError::Corrupt(format!(
                "transaction revision {} is not contiguous",
                transaction.revision
            )));
        }
        if transaction.events.is_empty() {
            return Err(SessionStoreError::Corrupt(format!(
                "transaction {} is empty",
                transaction.tx_id
            )));
        }
        for event in &transaction.events {
            if event.seq != expected_seq {
                return Err(SessionStoreError::Corrupt(format!(
                    "event sequence {} is not contiguous; expected {expected_seq}",
                    event.seq
                )));
            }
            expected_seq += 1;
        }
    }
    Ok(())
}

fn validate_nonempty(label: &str, value: &str) -> Result<(), SessionStoreError> {
    if value.trim().is_empty() {
        Err(SessionStoreError::Invalid(format!(
            "{label} must not be empty"
        )))
    } else {
        Ok(())
    }
}

fn normalize_title(title: &str) -> Result<String, SessionStoreError> {
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = title.chars().take(80).collect::<String>();
    if title.is_empty() {
        Err(SessionStoreError::Invalid(
            "session title must not be empty".into(),
        ))
    } else {
        Ok(title)
    }
}

fn to_sql_i64(value: u64) -> Result<i64, SessionStoreError> {
    i64::try_from(value).map_err(|_| SessionStoreError::NumericOverflow(value))
}

fn positive_sql_i64(value: i64, label: &str) -> Result<i64, SessionStoreError> {
    if value > 0 {
        Ok(value)
    } else {
        Err(SessionStoreError::Corrupt(format!(
            "{label} is not positive: {value}"
        )))
    }
}

fn from_sql_u64(value: i64, label: &str) -> Result<u64, SessionStoreError> {
    u64::try_from(value)
        .map_err(|_| SessionStoreError::Corrupt(format!("{label} is negative: {value}")))
}

fn usize_from_sql(value: i64, label: &str) -> Result<usize, SessionStoreError> {
    usize::try_from(value)
        .map_err(|_| SessionStoreError::Corrupt(format!("{label} is invalid: {value}")))
}

fn request_digest(
    base_revision: u64,
    events: &[RecordedEvent],
) -> Result<String, SessionStoreError> {
    let mut digest = StableDigestWriter::new();
    serde_json::to_writer(
        &mut digest,
        &RequestDigestMaterial {
            base_revision,
            events,
        },
    )?;
    Ok(digest.finish())
}

struct StableDigestWriter {
    left: u64,
    right: u64,
    byte_count: u64,
}

impl StableDigestWriter {
    fn new() -> Self {
        Self {
            left: 0xcbf2_9ce4_8422_2325,
            right: 0x8422_2325_cbf2_9ce4,
            byte_count: 0,
        }
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.left ^= u64::from(*byte);
            self.left = self.left.wrapping_mul(0x0000_0100_0000_01b3);
            self.right ^= u64::from(*byte);
            self.right = self.right.wrapping_mul(0x9e37_79b1_85eb_ca87);
        }
    }

    fn finish(mut self) -> String {
        self.update(&self.byte_count.to_le_bytes());
        format!("{:016x}{:016x}", self.left, self.right)
    }
}

impl Write for StableDigestWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.update(buffer);
        self.byte_count = self.byte_count.saturating_add(buffer.len() as u64);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

fn normalized_database_path(
    path: &Path,
    require_existing: bool,
) -> Result<PathBuf, SessionStoreError> {
    if require_existing || path.exists() {
        return Ok(fs::canonicalize(path)?);
    }

    let file_name = path.file_name().ok_or_else(|| {
        SessionStoreError::Invalid(format!(
            "database path has no file name: {}",
            path.display()
        ))
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(fs::canonicalize(parent)?.join(file_name))
}

#[derive(Serialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum JsonlExportRecord<'a> {
    Session {
        format_version: u32,
        metadata: &'a StoredSessionMetadata,
    },
    Transaction {
        transaction: &'a CommitReceipt,
    },
}

fn write_json_line(
    writer: &mut impl Write,
    value: &impl Serialize,
) -> Result<(), SessionStoreError> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn initialize_or_validate_schema(connection: &mut Connection) -> Result<(), SessionStoreError> {
    let found = schema_version(connection)?;
    match found {
        version if version == i64::from(DATABASE_SCHEMA_VERSION) => Ok(()),
        0 => {
            let user_table_count = connection.query_row(
                "SELECT count(*)
                 FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get::<_, i64>(0),
            )?;
            if user_table_count != 0 {
                return Err(unsupported_schema_version(found));
            }
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            transaction.execute_batch(SCHEMA)?;
            transaction.pragma_update(None, "user_version", DATABASE_SCHEMA_VERSION)?;
            transaction.commit()?;
            Ok(())
        }
        _ => Err(unsupported_schema_version(found)),
    }
}

fn require_schema_version(connection: &Connection) -> Result<(), SessionStoreError> {
    let found = schema_version(connection)?;
    if found == i64::from(DATABASE_SCHEMA_VERSION) {
        Ok(())
    } else {
        Err(unsupported_schema_version(found))
    }
}

fn schema_version(connection: &Connection) -> Result<i64, SessionStoreError> {
    Ok(connection.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))?)
}

fn unsupported_schema_version(found: i64) -> SessionStoreError {
    SessionStoreError::UnsupportedSchemaVersion {
        found,
        expected: DATABASE_SCHEMA_VERSION,
    }
}

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS sessions (
    session_key     INTEGER PRIMARY KEY,
    id              TEXT NOT NULL UNIQUE,
    project_id      TEXT NOT NULL,
    title           TEXT NOT NULL,
    config_json     BLOB NOT NULL,
    created_at_ms   INTEGER NOT NULL,
    updated_at_ms   INTEGER NOT NULL,
    archived_at_ms  INTEGER,
    revision        INTEGER NOT NULL DEFAULT 0 CHECK (revision >= 0),
    next_event_seq  INTEGER NOT NULL DEFAULT 0 CHECK (next_event_seq >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS sessions_project_catalog
ON sessions(project_id, archived_at_ms, created_at_ms DESC, id);

CREATE TABLE IF NOT EXISTS session_catalog_projection (
    session_key        INTEGER PRIMARY KEY,
    indexed_revision   INTEGER NOT NULL DEFAULT 0 CHECK (indexed_revision >= 0),
    extractor_version  INTEGER NOT NULL CHECK (extractor_version > 0),
    loadability_version INTEGER NOT NULL CHECK (loadability_version > 0),
    valid               INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0, 1)),
    FOREIGN KEY (session_key) REFERENCES sessions(session_key) ON DELETE CASCADE
) STRICT;

-- Search remains incremental: one event contributes a small, ordered set of fragments. Appending
-- never rewrites or concatenates the preceding transcript.
CREATE TABLE IF NOT EXISTS session_search_fragments (
    session_key INTEGER NOT NULL,
    event_seq   INTEGER NOT NULL CHECK (event_seq >= 0),
    ordinal     INTEGER NOT NULL CHECK (ordinal >= 0),
    value       TEXT NOT NULL,
    PRIMARY KEY (session_key, event_seq, ordinal),
    FOREIGN KEY (session_key) REFERENCES sessions(session_key) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

-- Only streaming assistant fragments need a secondary lookup. Keeping that lookup in a separate
-- sparse table avoids taxing ordinary inputs while allowing AssistantCompleted to replace every
-- draft for one request in indexed time.
CREATE TABLE IF NOT EXISTS session_request_draft_fragments (
    session_key INTEGER NOT NULL,
    request_id  TEXT NOT NULL,
    event_seq   INTEGER NOT NULL CHECK (event_seq >= 0),
    ordinal     INTEGER NOT NULL CHECK (ordinal >= 0),
    PRIMARY KEY (session_key, request_id, event_seq, ordinal),
    FOREIGN KEY (session_key, event_seq, ordinal)
        REFERENCES session_search_fragments(session_key, event_seq, ordinal) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS journal_transactions (
    session_key      INTEGER NOT NULL,
    revision         INTEGER NOT NULL CHECK (revision > 0),
    tx_id            TEXT NOT NULL CHECK (length(tx_id) > 0),
    base_revision    INTEGER NOT NULL CHECK (base_revision >= 0),
    first_event_seq  INTEGER NOT NULL CHECK (first_event_seq >= 0),
    event_count      INTEGER NOT NULL CHECK (event_count > 0),
    clock_id         TEXT NOT NULL CHECK (length(clock_id) > 0),
    request_digest   TEXT NOT NULL,
    committed_at_ms  INTEGER NOT NULL,
    PRIMARY KEY (session_key, revision),
    UNIQUE (session_key, tx_id),
    FOREIGN KEY (session_key) REFERENCES sessions(session_key) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS journal_events (
    session_key          INTEGER NOT NULL,
    seq                  INTEGER NOT NULL CHECK (seq >= 0),
    transaction_revision INTEGER NOT NULL CHECK (transaction_revision > 0),
    ordinal              INTEGER NOT NULL CHECK (ordinal >= 0),
    wall_time_ms          INTEGER NOT NULL,
    monotonic_ns          INTEGER NOT NULL CHECK (monotonic_ns >= 0),
    event_json           BLOB NOT NULL,
    PRIMARY KEY (session_key, seq),
    FOREIGN KEY (session_key, transaction_revision)
        REFERENCES journal_transactions(session_key, revision) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

-- The catalog must stay projection-only, so journal mutations outside the canonical append
-- transaction invalidate the affected projection at write time. Canonical inserts happen after
-- `sessions.revision` advances but before `indexed_revision` advances, which distinguishes them
-- from post-commit tampering without scanning journal history during catalog reads.
CREATE TRIGGER IF NOT EXISTS journal_transactions_invalidate_catalog_after_insert
AFTER INSERT ON journal_transactions
BEGIN
    UPDATE session_catalog_projection
       SET valid = 0
     WHERE session_key = NEW.session_key
       AND indexed_revision = (
           SELECT revision FROM sessions WHERE session_key = NEW.session_key
       );
END;

CREATE TRIGGER IF NOT EXISTS journal_transactions_invalidate_catalog_after_update
AFTER UPDATE ON journal_transactions
BEGIN
    UPDATE session_catalog_projection
       SET valid = 0
     WHERE session_key IN (OLD.session_key, NEW.session_key);
END;

CREATE TRIGGER IF NOT EXISTS journal_transactions_invalidate_catalog_after_delete
AFTER DELETE ON journal_transactions
BEGIN
    UPDATE session_catalog_projection SET valid = 0 WHERE session_key = OLD.session_key;
END;

CREATE TRIGGER IF NOT EXISTS journal_events_invalidate_catalog_after_insert
AFTER INSERT ON journal_events
BEGIN
    UPDATE session_catalog_projection
       SET valid = 0
     WHERE session_key = NEW.session_key
       AND indexed_revision = (
           SELECT revision FROM sessions WHERE session_key = NEW.session_key
       );
END;

CREATE TRIGGER IF NOT EXISTS journal_events_invalidate_catalog_after_update
AFTER UPDATE ON journal_events
BEGIN
    UPDATE session_catalog_projection
       SET valid = 0
     WHERE session_key IN (OLD.session_key, NEW.session_key);
END;

CREATE TRIGGER IF NOT EXISTS journal_events_invalidate_catalog_after_delete
AFTER DELETE ON journal_events
BEGIN
    UPDATE session_catalog_projection SET valid = 0 WHERE session_key = OLD.session_key;
END;

-- `next_event_seq` is journal integrity metadata, but catalog reads intentionally do not scan the
-- journal tail. Canonical appends advance both revision and sequence in the same UPDATE. A
-- sequence-only mutation therefore identifies corruption and must make the row invisible before a
-- desktop runtime can be created for it. If revision is also changed, the indexed-revision join
-- already hides the row.
CREATE TRIGGER IF NOT EXISTS sessions_invalidate_catalog_after_next_event_seq_update
AFTER UPDATE OF next_event_seq ON sessions
WHEN NEW.next_event_seq != OLD.next_event_seq AND NEW.revision = OLD.revision
BEGIN
    UPDATE session_catalog_projection SET valid = 0 WHERE session_key = NEW.session_key;
END;
"#;

#[cfg(test)]
mod tests {
    use super::{
        AppendFailpoint, AppendTx, ArchiveFilter, CATALOG_EXTRACTOR_VERSION,
        CATALOG_LOADABILITY_VERSION, CreateStoredSession, DATABASE_SCHEMA_VERSION, MetadataUpdate,
        SESSION_DATABASE_FILE, SessionErrorClass, SessionStore, SessionStoreError, TransactionId,
        validate_retry_matches, writer_lock_file_name,
    };
    use crate::session::event::{
        AssistantChunk, EventDraft, EventTime, InputId, InputOrigin, RecordedEvent, RequestId,
        ResponseInfo, RunId, SessionEvent, StepId, TurnId, TxId,
    };
    use crate::session::machine::SessionMachine;
    use crate::session::{SessionConfig, SessionId};
    use async_openai::types::responses::{EasyInputMessage, InputItem};
    use rusqlite::{Connection, params};
    use std::fs::{File, OpenOptions, TryLockError};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    #[test]
    fn in_memory_store_obeys_append_contract() {
        run_append_contract(SessionStore::open_in_memory().unwrap());
    }

    #[test]
    fn disk_store_obeys_append_contract_and_uses_wal() {
        let directory = test_directory("disk-contract");
        let store = SessionStore::open_project(&directory).unwrap();
        assert_eq!(
            store.database_path(),
            Some(
                std::fs::canonicalize(&directory)
                    .unwrap()
                    .join("sessions.sqlite3")
                    .as_path()
            )
        );
        let mode = store
            .connection()
            .unwrap()
            .pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))
            .unwrap();
        assert_eq!(mode.to_lowercase(), "wal");
        run_append_contract(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn readonly_open_requires_current_schema_and_never_creates_a_database() {
        let directory = test_directory("readonly-schema-version");
        std::fs::create_dir_all(&directory).unwrap();
        let database_path = directory.join(SESSION_DATABASE_FILE);
        let writable = SessionStore::open_database(&database_path).unwrap();
        let session = create_session(&writable, "readonly");
        let version = writable
            .connection()
            .unwrap()
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(version, i64::from(DATABASE_SCHEMA_VERSION));
        drop(writable);

        let readonly = SessionStore::open_database_readonly(&database_path).unwrap();
        assert!(
            readonly
                .connection()
                .unwrap()
                .is_readonly(rusqlite::MAIN_DB)
                .unwrap()
        );
        assert_eq!(readonly.load(&session.id).unwrap().metadata.id, session.id);
        drop(readonly);

        let missing_path = directory.join("missing.sqlite3");
        assert!(SessionStore::open_database_readonly(&missing_path).is_err());
        assert!(!missing_path.exists());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn unsupported_or_unversioned_existing_databases_are_rejected() {
        let directory = test_directory("unsupported-schema-version");
        std::fs::create_dir_all(&directory).unwrap();

        let unsupported_path = directory.join("unsupported.sqlite3");
        let unsupported = Connection::open(&unsupported_path).unwrap();
        let unsupported_schema = i64::from(DATABASE_SCHEMA_VERSION) + 1;
        unsupported
            .pragma_update(None, "user_version", unsupported_schema)
            .unwrap();
        drop(unsupported);
        for error in [
            SessionStore::open_database(&unsupported_path)
                .err()
                .unwrap(),
            SessionStore::open_database_readonly(&unsupported_path)
                .err()
                .unwrap(),
        ] {
            assert!(matches!(
                error,
                SessionStoreError::UnsupportedSchemaVersion {
                    found,
                    expected: DATABASE_SCHEMA_VERSION
                } if found == unsupported_schema
            ));
        }

        let unversioned_path = directory.join("unversioned.sqlite3");
        let unversioned = Connection::open(&unversioned_path).unwrap();
        unversioned
            .execute("CREATE TABLE legacy_session (value TEXT)", [])
            .unwrap();
        drop(unversioned);
        assert!(matches!(
            SessionStore::open_database(&unversioned_path),
            Err(SessionStoreError::UnsupportedSchemaVersion {
                found: 0,
                expected: DATABASE_SCHEMA_VERSION
            })
        ));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn store_errors_distinguish_corrupt_data_from_temporary_storage_failures() {
        use rusqlite::ffi::{Error, ErrorCode};

        let sqlite = |code| {
            SessionStoreError::Sqlite(rusqlite::Error::SqliteFailure(
                Error {
                    code,
                    extended_code: 0,
                },
                None,
            ))
        };
        assert_eq!(
            sqlite(ErrorCode::DatabaseCorrupt).classification(),
            SessionErrorClass::DeterministicInvalid
        );
        assert_eq!(
            sqlite(ErrorCode::NotADatabase).classification(),
            SessionErrorClass::DeterministicInvalid
        );
        for error in [
            sqlite(ErrorCode::DatabaseBusy),
            sqlite(ErrorCode::DatabaseLocked),
            sqlite(ErrorCode::SystemIoFailure),
        ] {
            assert_eq!(error.classification(), SessionErrorClass::Transient);
        }
        assert_eq!(
            SessionStoreError::Io(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "retry",
            ))
            .classification(),
            SessionErrorClass::Transient
        );
        assert_eq!(
            SessionStoreError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad bytes",
            ))
            .classification(),
            SessionErrorClass::DeterministicInvalid
        );
    }

    #[test]
    fn transaction_schema_does_not_duplicate_event_payloads() {
        let store = SessionStore::open_in_memory().unwrap();
        let connection = store.connection().unwrap();
        let mut statement = connection
            .prepare("SELECT name FROM pragma_table_info('sessions') ORDER BY cid")
            .unwrap();
        let session_columns = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!session_columns.iter().any(|column| column == "owner_token"));
        assert!(
            !session_columns
                .iter()
                .any(|column| column == "lease_expires_at_ms")
        );

        let mut statement = connection
            .prepare("SELECT name FROM pragma_table_info('journal_transactions') ORDER BY cid")
            .unwrap();
        let columns = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(columns.iter().any(|column| column == "request_digest"));
        assert!(columns.iter().any(|column| column == "session_key"));
        assert!(columns.iter().any(|column| column == "clock_id"));
        assert!(!columns.iter().any(|column| column == "session_id"));
        assert!(!columns.iter().any(|column| column == "request_blob"));

        let projection_columns = connection
            .prepare(
                "SELECT name FROM pragma_table_info('session_catalog_projection') ORDER BY cid",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            projection_columns
                .iter()
                .any(|column| column == "loadability_version")
        );

        let mut statement = connection
            .prepare("SELECT name FROM pragma_table_info('journal_events') ORDER BY cid")
            .unwrap();
        let event_columns = statement
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(event_columns.iter().any(|column| column == "session_key"));
        assert!(event_columns.iter().any(|column| column == "wall_time_ms"));
        assert!(event_columns.iter().any(|column| column == "monotonic_ns"));
        for duplicated_column in ["session_id", "tx_id", "clock_id", "event_kind", "time_json"] {
            assert!(
                !event_columns
                    .iter()
                    .any(|column| column == duplicated_column)
            );
        }

        let secondary_index_count = connection
            .query_row(
                "SELECT count(*) FROM pragma_index_list('journal_events') WHERE origin != 'pk'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(secondary_index_count, 0);
        let resolve_query_plan = connection
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT transaction_revision, ordinal, seq
                 FROM journal_events
                 WHERE session_key = 1 AND seq BETWEEN 10 AND 20
                 ORDER BY seq ASC",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(3))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join("\n")
            .to_lowercase();
        assert!(resolve_query_plan.contains("search journal_events using primary key"));
        assert!(!resolve_query_plan.contains("temp b-tree"));
        let without_rowid = connection
            .query_row(
                "SELECT wr FROM pragma_table_list WHERE name = 'journal_events'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(without_rowid, 1);
    }

    #[test]
    fn disk_opens_reuse_one_inner_per_normalized_path_and_access_mode() {
        let directory = test_directory("connection-registry");
        let first_writer = SessionStore::open_project(&directory).unwrap();
        let second_writer =
            SessionStore::open_database(directory.join(".").join(SESSION_DATABASE_FILE)).unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &first_writer.inner,
            &second_writer.inner
        ));

        let first_reader =
            SessionStore::open_database_readonly(directory.join(SESSION_DATABASE_FILE)).unwrap();
        let second_reader =
            SessionStore::open_database_readonly(directory.join(".").join(SESSION_DATABASE_FILE))
                .unwrap();
        assert!(std::sync::Arc::ptr_eq(
            &first_reader.inner,
            &second_reader.inner
        ));
        assert!(!std::sync::Arc::ptr_eq(
            &first_writer.inner,
            &first_reader.inner
        ));

        drop(second_reader);
        drop(first_reader);
        drop(second_writer);
        drop(first_writer);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn writer_permit_is_process_and_os_exclusive_without_blocking_readers() {
        let directory = test_directory("writer-permit-exclusive");
        let first_store = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&first_store, "writer");

        // Bypass the connection registry. The process-local writer registry
        // must still arbitrate distinct handles for the same database.
        let second_store = SessionStore::from_writable_connection(
            Connection::open(directory.join(SESSION_DATABASE_FILE)).unwrap(),
            Some(first_store.database_path().unwrap().to_path_buf()),
            true,
        )
        .unwrap();
        assert!(!std::sync::Arc::ptr_eq(
            &first_store.inner,
            &second_store.inner
        ));

        let permit = first_store.acquire_writer(&session.id).unwrap();
        assert!(matches!(
            second_store.acquire_writer(&session.id),
            Err(SessionStoreError::WriterBusy { .. })
        ));

        // A separately opened descriptor observes the OS-level lock too.
        let lock_path = directory
            .join(".session-writers")
            .join(writer_lock_file_name(&session.id).unwrap());
        let lock_probe = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        assert!(matches!(
            lock_probe.try_lock(),
            Err(TryLockError::WouldBlock)
        ));

        let readonly =
            SessionStore::open_database_readonly(first_store.database_path().unwrap()).unwrap();
        assert_eq!(readonly.load(&session.id).unwrap().metadata.id, session.id);
        let mut export = Vec::new();
        readonly.export_jsonl(&session.id, &mut export).unwrap();
        assert!(!export.is_empty());

        drop(permit);
        let second_permit = second_store.acquire_writer(&session.id).unwrap();
        drop(second_permit);
        assert!(lock_path.exists(), "lockfile must remain after release");
        drop(readonly);
        drop(second_store);
        drop(first_store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "subprocess helper; invoked by hard_kill_rolls_back_and_releases_writer"]
    fn append_pause_subprocess() {
        let Some(database_path) = std::env::var_os("KCASTLE_TEST_DATABASE_PATH") else {
            return;
        };
        let Some(session_id) = std::env::var_os("KCASTLE_TEST_SESSION_ID") else {
            return;
        };
        let store = SessionStore::open_database(database_path).unwrap();
        let session_id = SessionId::from_raw(session_id.to_string_lossy());
        let permit = store.acquire_writer(&session_id).unwrap();
        store.inject_failpoint(AppendFailpoint::PauseBeforeCommit);
        let request = append_request(&session_id, "tx-hard-kill", 0, two_events());
        let _ = store.append(&request, &permit);
        panic!("pause failpoint unexpectedly returned");
    }

    #[test]
    fn hard_kill_rolls_back_and_releases_writer() {
        let directory = test_directory("hard-kill-atomicity");
        let store = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&store, "hard-kill");
        let database_path = store.database_path().unwrap().to_path_buf();
        let ready_path = directory.join("append-ready");
        let test_binary = std::env::current_exe().unwrap();
        let mut child = Command::new(test_binary)
            .args([
                "--ignored",
                "--exact",
                "session::store::tests::append_pause_subprocess",
                "--nocapture",
            ])
            .env("KCASTLE_TEST_DATABASE_PATH", &database_path)
            .env("KCASTLE_TEST_SESSION_ID", session.id.as_str())
            .env("KCASTLE_TEST_APPEND_PAUSE_READY", &ready_path)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !ready_path.exists() && std::time::Instant::now() < deadline {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let reached_precommit = ready_path.exists();
        let was_busy = matches!(
            store.acquire_writer(&session.id),
            Err(SessionStoreError::WriterBusy { .. })
        );
        if child.try_wait().unwrap().is_none() {
            child.kill().unwrap();
        }
        child.wait().unwrap();

        assert!(
            reached_precommit,
            "child never reached the pre-commit failpoint"
        );
        assert!(was_busy, "parent acquired a writer while the child held it");
        let loaded = store.load(&session.id).unwrap();
        assert_eq!(loaded.metadata.revision, 0);
        assert_eq!(loaded.events().count(), 0);

        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-hard-kill", 0, two_events());
        let receipt = store.append(&request, &permit).unwrap();
        assert_eq!(receipt.revision, 1);
        assert_eq!(receipt.events.len(), 2);
        drop(permit);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn different_sessions_can_hold_writer_permits_concurrently() {
        let store = SessionStore::open_in_memory().unwrap();
        let first = create_session(&store, "first");
        let second = create_session(&store, "second");
        let first_permit = store.acquire_writer(&first.id).unwrap();
        let second_permit = store.acquire_writer(&second.id).unwrap();
        let first_request = append_request(&first.id, "tx-first", 0, two_events());
        let second_request = append_request(&second.id, "tx-second", 0, two_events());
        store.append(&first_request, &first_permit).unwrap();
        store.append(&second_request, &second_permit).unwrap();
        assert_eq!(store.load(&first.id).unwrap().metadata.revision, 1);
        assert_eq!(store.load(&second.id).unwrap().metadata.revision, 1);
    }

    #[test]
    fn preexisting_unlocked_lockfile_can_be_acquired_and_drop_releases_it() {
        let directory = test_directory("preexisting-lockfile");
        let store = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&store, "preexisting");
        let lock_directory = directory.join(".session-writers");
        std::fs::create_dir_all(&lock_directory).unwrap();
        let lock_path = lock_directory.join(writer_lock_file_name(&session.id).unwrap());
        File::create(&lock_path).unwrap();

        let permit = store.acquire_writer(&session.id).unwrap();
        let permit_clone = permit.clone();
        assert!(matches!(
            store.acquire_writer(&session.id),
            Err(SessionStoreError::WriterBusy { .. })
        ));
        drop(permit);
        assert!(matches!(
            store.acquire_writer(&session.id),
            Err(SessionStoreError::WriterBusy { .. })
        ));
        drop(permit_clone);
        drop(store.acquire_writer(&session.id).unwrap());
        assert!(lock_path.exists());

        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn writer_permit_is_bound_to_both_session_and_database() {
        let store = SessionStore::open_in_memory().unwrap();
        let authorized = create_session(&store, "authorized");
        let other = create_session(&store, "other");
        let permit = store.acquire_writer(&authorized.id).unwrap();
        let wrong_session_request = append_request(&other.id, "tx-wrong-session", 0, two_events());
        assert!(matches!(
            store.append(&wrong_session_request, &permit),
            Err(SessionStoreError::InvalidWriterPermit { .. })
        ));
        assert!(matches!(
            store.update_metadata(
                &other.id,
                MetadataUpdate {
                    title: Some("forbidden".into()),
                    ..MetadataUpdate::default()
                },
                &permit,
            ),
            Err(SessionStoreError::InvalidWriterPermit { .. })
        ));
        assert!(matches!(
            store.archive(&other.id, &permit),
            Err(SessionStoreError::InvalidWriterPermit { .. })
        ));
        assert!(matches!(
            store.restore(&other.id, &permit),
            Err(SessionStoreError::InvalidWriterPermit { .. })
        ));
        assert!(matches!(
            store.delete(&other.id, &permit),
            Err(SessionStoreError::InvalidWriterPermit { .. })
        ));

        let other_store = SessionStore::open_in_memory().unwrap();
        other_store
            .create_session(CreateStoredSession {
                id: authorized.id.clone(),
                project_id: "project".into(),
                title: "same ID, other database".into(),
                config: SessionConfig::default(),
                created_at_ms: 1_000,
            })
            .unwrap();
        let wrong_database_request =
            append_request(&authorized.id, "tx-wrong-database", 0, two_events());
        assert!(matches!(
            other_store.append(&wrong_database_request, &permit),
            Err(SessionStoreError::InvalidWriterPermit { .. })
        ));
    }

    #[test]
    fn load_rejects_tampered_next_event_sequence() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "next-sequence");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-next-sequence", 0, two_events());
        store.append(&request, &permit).unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE sessions SET next_event_seq = 99 WHERE id = ?1",
                rusqlite::params![session.id.as_str()],
            )
            .unwrap();

        assert!(matches!(
            store.load(&session.id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("next event sequence 99")
                    && message.contains("journal tail 2")
        ));
        assert!(
            store
                .catalog("project", ArchiveFilter::Active)
                .unwrap()
                .is_empty(),
            "a session with corrupt journal-tail metadata must be absent from the catalog"
        );
    }

    #[test]
    fn oversized_declared_event_count_is_corrupt_and_hidden_from_catalog() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "oversized-event-count");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-oversized-event-count", 0, two_events());
        store.append(&request, &permit).unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE journal_transactions SET event_count = ?2
                 WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)",
                params![session.id.as_str(), i64::MAX],
            )
            .unwrap();

        assert!(matches!(
            store.load(&session.id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("declared event range")
        ));
        assert!(matches!(
            store.resolve(&session.id, &request.tx_id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("declared event range")
        ));
        assert!(
            store
                .catalog("project", ArchiveFilter::Active)
                .unwrap()
                .is_empty(),
            "a structurally corrupt session must be absent from the catalog"
        );
    }

    #[test]
    fn load_and_resolve_reject_events_outside_the_declared_transaction_range() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "extra-transaction-event");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-extra-transaction-event", 0, two_events());
        store.append(&request, &permit).unwrap();

        let extra_event = serde_json::to_vec(&SessionEvent::RunStarted {
            run_id: RunId::from_raw("extra-run"),
        })
        .unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "INSERT INTO journal_events (
                    session_key, seq, transaction_revision, ordinal,
                    wall_time_ms, monotonic_ns, event_json
                 ) VALUES (
                    (SELECT session_key FROM sessions WHERE id = ?1), 2, 1, 2, 3, 3, ?2
                 )",
                params![session.id.as_str(), extra_event],
            )
            .unwrap();

        assert!(matches!(
            store.load(&session.id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("declared event range")
        ));
        assert!(matches!(
            store.resolve(&session.id, &request.tx_id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("catalog projection")
        ));
    }

    #[test]
    fn load_observes_complete_snapshots_during_concurrent_appends() {
        const APPEND_COUNT: u64 = 200;

        let directory = test_directory("load-snapshot");
        let writer = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&writer, "snapshot");
        let permit = writer.acquire_writer(&session.id).unwrap();
        let reader = SessionStore::open_database_readonly(writer.database_path().unwrap()).unwrap();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer_done = done.clone();
        let writer_session_id = session.id.clone();
        let writer_thread = std::thread::spawn(move || {
            for revision in 0..APPEND_COUNT {
                let request = append_request(
                    &writer_session_id,
                    &format!("tx-snapshot-{revision}"),
                    revision,
                    vec![recorded(
                        revision,
                        SessionEvent::RunStarted {
                            run_id: RunId::from_raw(format!("run-snapshot-{revision}")),
                        },
                    )],
                );
                writer.append(&request, &permit).unwrap();
                if revision % 8 == 0 {
                    std::thread::yield_now();
                }
            }
            writer_done.store(true, std::sync::atomic::Ordering::Release);
        });

        let mut snapshots = 0;
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            let loaded = reader.load(&session.id).unwrap();
            assert_eq!(loaded.metadata.revision as usize, loaded.transactions.len());
            assert_eq!(loaded.metadata.revision as usize, loaded.events().count());
            snapshots += 1;
        }
        writer_thread.join().unwrap();
        let loaded = reader.load(&session.id).unwrap();
        assert_eq!(loaded.metadata.revision, APPEND_COUNT);
        assert_eq!(loaded.transactions.len(), APPEND_COUNT as usize);
        assert!(snapshots > 0);

        drop(reader);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn transaction_is_all_or_nothing_when_commit_fails() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "atomic");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-atomic", 0, two_events());
        store.inject_failpoint(AppendFailpoint::BeforeCommitOnce);
        assert!(matches!(
            store.append(&request, &permit),
            Err(SessionStoreError::InjectedBeforeCommit)
        ));
        let loaded = store.load(&session.id).unwrap();
        assert_eq!(loaded.metadata.revision, 0);
        assert_eq!(loaded.events().count(), 0);
        let receipt = store.append(&request, &permit).unwrap();
        assert_eq!(receipt.events.len(), 2);
        assert_eq!(receipt.revision, 1);
    }

    #[test]
    fn transaction_rejects_mixed_clock_domains() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "mixed-clocks");
        let permit = store.acquire_writer(&session.id).unwrap();
        let mut events = two_events();
        events[1].time.clock_id = "different-clock".into();
        let request = append_request(&session.id, "tx-mixed-clocks", 0, events);

        assert!(matches!(
            store.append(&request, &permit),
            Err(SessionStoreError::Invalid(message))
                if message.contains("must share one clock ID")
        ));
        assert_eq!(store.load(&session.id).unwrap().metadata.revision, 0);
    }

    #[test]
    fn committed_transaction_is_resolved_after_receipt_is_lost() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "ambiguous");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-ambiguous", 0, two_events());
        store.inject_failpoint(AppendFailpoint::AfterCommitBeforeReceiptOnce);
        assert!(matches!(
            store.append(&request, &permit),
            Err(SessionStoreError::OutcomeUnknown { tx_id }) if tx_id == request.tx_id
        ));
        let resolved = store.resolve(&session.id, &request.tx_id).unwrap().unwrap();
        assert_eq!(resolved.revision, 1);
        assert_eq!(resolved.events.len(), 2);
        assert_eq!(store.append(&request, &permit).unwrap(), resolved);
    }

    #[test]
    fn resolve_uses_one_snapshot_across_concurrent_catalog_advancement() {
        let directory = test_directory("resolve-snapshot");
        let store = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&store, "resolve-snapshot");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-resolve-snapshot", 0, two_events());
        let expected = store.append(&request, &permit).unwrap();

        store.pause_resolve_after_revision();
        let resolver = store.clone();
        let session_id = session.id.clone();
        let tx_id = request.tx_id.clone();
        let resolving = std::thread::spawn(move || resolver.resolve(&session_id, &tx_id));
        let deadline = Instant::now() + Duration::from_secs(5);
        while !store.resolve_pause_reached() {
            assert!(Instant::now() < deadline, "resolve did not reach its pause");
            std::thread::yield_now();
        }

        let mut external = Connection::open(store.database_path().unwrap()).unwrap();
        let advancement = external.transaction().unwrap();
        advancement
            .execute(
                "UPDATE sessions SET revision = revision + 1 WHERE id = ?1",
                params![session.id.as_str()],
            )
            .unwrap();
        advancement
            .execute(
                "UPDATE session_catalog_projection
                    SET indexed_revision = indexed_revision + 1
                  WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)",
                params![session.id.as_str()],
            )
            .unwrap();
        advancement.commit().unwrap();

        store.resume_resolve();
        let resolved = resolving.join().unwrap().unwrap().unwrap();
        assert_eq!(resolved, expected);

        drop(external);
        drop(permit);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn digest_gate_is_followed_by_exact_event_comparison() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "digest-collision");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-collision", 0, two_events());
        let receipt = store.append(&request, &permit).unwrap();
        let mut colliding_request = request;
        colliding_request.events[0].time.wall_time_ms += 1;

        // This helper runs after base revision and digest have already matched. Even if two
        // different requests ever collide at that gate, canonical events remain authoritative.
        assert!(matches!(
            validate_retry_matches(&colliding_request, &receipt),
            Err(SessionStoreError::TransactionConflict { .. })
        ));
    }

    #[test]
    fn replay_rejects_canonical_event_tampering() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "tampered-event");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-tampered-event", 0, two_events());
        store.append(&request, &permit).unwrap();

        let replacement = serde_json::to_vec(&SessionEvent::RunStarted {
            run_id: RunId::from_raw("tampered-run"),
        })
        .unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE journal_events
                 SET event_json = ?3
                 WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)
                   AND seq = ?2",
                rusqlite::params![session.id.as_str(), 0_i64, replacement],
            )
            .unwrap();

        assert!(matches!(
            store.load(&session.id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("request digest does not match its canonical events")
        ));
        assert!(matches!(
            store.resolve(&session.id, &request.tx_id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("request digest does not match its canonical events")
        ));
    }

    #[test]
    fn planned_batch_is_committed_without_resequencing() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "planned");
        let permit = store.acquire_writer(&session.id).unwrap();
        let mut machine = SessionMachine::default();
        let batch = machine
            .plan_batch(vec![EventDraft {
                tx_id: TxId::from_raw("tx-planned"),
                time: event_time(7),
                event: SessionEvent::RunStarted {
                    run_id: RunId::from_raw("run-planned"),
                },
            }])
            .unwrap();
        let request = AppendTx::from_planned(session.id, 0, &batch);
        let receipt = store.append(&request, &permit).unwrap();
        assert_eq!(receipt.events.as_slice(), batch.events());
        machine.apply_batch(batch).unwrap();
        assert_eq!(machine.next_seq(), 1);
    }

    #[test]
    fn metadata_catalog_archive_delete_and_jsonl_export_round_trip() {
        let directory = test_directory("metadata");
        let store = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&store, "metadata");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request = append_request(&session.id, "tx-export", 0, two_events());
        store.append(&request, &permit).unwrap();

        let config = SessionConfig {
            model_id: Some("provider/model".into()),
            reasoning_effort: Some("high".into()),
            allow_all_tools: true,
        };
        let updated = store
            .update_metadata(
                &session.id,
                MetadataUpdate {
                    title: Some("  renamed   session ".into()),
                    config: Some(config.clone()),
                    ..MetadataUpdate::default()
                },
                &permit,
            )
            .unwrap();
        assert_eq!(updated.title, "renamed session");
        assert_eq!(updated.config, config);
        let active = store.catalog("project", ArchiveFilter::Active).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].metadata.title, "renamed session");
        assert!(
            store
                .catalog("another-project", ArchiveFilter::Active)
                .unwrap()
                .is_empty()
        );
        store.archive(&session.id, &permit).unwrap();
        assert!(
            store
                .catalog("project", ArchiveFilter::Active)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            store
                .catalog("project", ArchiveFilter::Archived)
                .unwrap()
                .len(),
            1
        );
        store.restore(&session.id, &permit).unwrap();

        let mut export = Vec::new();
        store.export_jsonl(&session.id, &mut export).unwrap();
        let lines = String::from_utf8(export).unwrap();
        let records = lines
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0]["record"], "session");
        assert_eq!(records[1]["record"], "transaction");

        drop(permit);
        drop(store);
        let reopened = SessionStore::open_project(&directory).unwrap();
        let reopened_permit = reopened.acquire_writer(&session.id).unwrap();
        let loaded = reopened.load(&session.id).unwrap();
        assert_eq!(loaded.metadata.title, "renamed session");
        assert_eq!(loaded.events().count(), 2);
        reopened.delete(&session.id, &reopened_permit).unwrap();
        assert!(matches!(
            reopened.load(&session.id),
            Err(SessionStoreError::SessionNotFound(_))
        ));
        assert!(
            reopened
                .catalog("project", ArchiveFilter::Active)
                .unwrap()
                .is_empty()
        );
        let projection_rows = reopened
            .connection()
            .unwrap()
            .query_row(
                "SELECT count(*) FROM session_catalog_projection",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(projection_rows, 0, "delete must cascade into projections");
        drop(reopened);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn catalog_reads_projection_without_loading_journal_and_final_replaces_drafts() {
        let store = SessionStore::open_in_memory().unwrap();
        let session = create_session(&store, "projection");
        let permit = store.acquire_writer(&session.id).unwrap();
        let request_id = RequestId::from_raw("request-projection");
        let final_items = vec![InputItem::from(EasyInputMessage::from("canonical answer"))];
        let final_payload = serde_json::to_string(&final_items).unwrap();
        let draft_request = append_request(
            &session.id,
            "tx-projection-drafts",
            0,
            vec![
                recorded(
                    0,
                    SessionEvent::AssistantChunk {
                        request_id: request_id.clone(),
                        chunk: AssistantChunk::OutputTextDelta {
                            delta: "obsolete draft".into(),
                        },
                    },
                ),
                recorded(
                    1,
                    SessionEvent::AssistantChunk {
                        request_id: request_id.clone(),
                        chunk: AssistantChunk::ReasoningTextDelta {
                            delta: "obsolete reasoning".into(),
                        },
                    },
                ),
            ],
        );
        store.append(&draft_request, &permit).unwrap();
        assert_eq!(
            store.catalog("project", ArchiveFilter::Active).unwrap()[0].search_values,
            vec!["obsolete draft", "obsolete reasoning"]
        );
        let completion = append_request(
            &session.id,
            "tx-projection-completed",
            1,
            vec![recorded(
                2,
                SessionEvent::AssistantCompleted {
                    request_id,
                    items: final_items,
                    response: ResponseInfo {
                        id: "response-projection".into(),
                        model: "test-model".into(),
                        usage: None,
                    },
                },
            )],
        );
        store.append(&completion, &permit).unwrap();

        let catalog = store.catalog("project", ArchiveFilter::Active).unwrap();
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog[0].search_values, vec![final_payload.clone()]);
        assert!(!catalog[0].search_values.join(" ").contains("obsolete"));

        // Deliberately make replay impossible. The write-side integrity trigger invalidates the
        // projection, so catalog can hide the row without loading or deserializing journal events.
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE journal_events SET event_json = X'FF'
                 WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)",
                params![session.id.as_str()],
            )
            .unwrap();
        assert!(store.load(&session.id).is_err());
        assert!(
            store
                .catalog("project", ArchiveFilter::Active)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn projection_commit_and_retry_have_exactly_once_semantics() {
        let store = SessionStore::open_in_memory().unwrap();
        let before = create_session(&store, "before-failure");
        let before_permit = store.acquire_writer(&before.id).unwrap();
        let before_request = append_request(
            &before.id,
            "tx-before-projection",
            0,
            vec![recorded(
                0,
                SessionEvent::InputSubmitted {
                    input_id: InputId::from_raw("input-before"),
                    input: "before needle".into(),
                    origin: InputOrigin::Queue,
                },
            )],
        );
        store.inject_failpoint(AppendFailpoint::BeforeCommitOnce);
        assert!(matches!(
            store.append(&before_request, &before_permit),
            Err(SessionStoreError::InjectedBeforeCommit)
        ));
        let before_catalog = store.catalog("project", ArchiveFilter::Active).unwrap();
        let before_entry = before_catalog
            .iter()
            .find(|entry| entry.metadata.id == before.id)
            .unwrap();
        assert_eq!(before_entry.metadata.revision, 0);
        assert!(before_entry.search_values.is_empty());

        let after = create_session(&store, "after-failure");
        let after_permit = store.acquire_writer(&after.id).unwrap();
        let after_request = append_request(
            &after.id,
            "tx-after-projection",
            0,
            vec![recorded(
                0,
                SessionEvent::InputSubmitted {
                    input_id: InputId::from_raw("input-after"),
                    input: "after needle".into(),
                    origin: InputOrigin::Queue,
                },
            )],
        );
        store.inject_failpoint(AppendFailpoint::AfterCommitBeforeReceiptOnce);
        assert!(matches!(
            store.append(&after_request, &after_permit),
            Err(SessionStoreError::OutcomeUnknown { .. })
        ));
        let resolved = store.append(&after_request, &after_permit).unwrap();
        assert_eq!(resolved.revision, 1);
        assert_eq!(
            store
                .catalog("project", ArchiveFilter::Active)
                .unwrap()
                .into_iter()
                .find(|entry| entry.metadata.id == after.id)
                .unwrap()
                .search_values,
            vec!["after needle"]
        );
        let fragment_count = store
            .connection()
            .unwrap()
            .query_row(
                "SELECT count(*) FROM session_search_fragments
                 WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)",
                params![after.id.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(fragment_count, 1);
    }

    #[test]
    fn catalog_hides_only_sessions_with_invalid_or_stale_projections() {
        let store = SessionStore::open_in_memory().unwrap();
        let stale = create_session(&store, "stale-projection");
        let invalid = create_session(&store, "invalid-projection");
        let visible = create_session(&store, "visible-projection");
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE session_catalog_projection SET indexed_revision = 9
                 WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)",
                params![stale.id.as_str()],
            )
            .unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE session_catalog_projection SET valid = 0
                 WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?1)",
                params![invalid.id.as_str()],
            )
            .unwrap();

        let catalog = store.catalog("project", ArchiveFilter::Active).unwrap();
        assert_eq!(
            catalog
                .into_iter()
                .map(|entry| entry.metadata.id)
                .collect::<Vec<_>>(),
            vec![visible.id]
        );
    }

    #[test]
    fn catalog_and_load_reject_an_incompatible_loadability_projection_without_replay() {
        let store = SessionStore::open_in_memory().unwrap();
        let incompatible = create_session(&store, "incompatible-loadability");
        let visible = create_session(&store, "current-loadability");
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE session_catalog_projection
                    SET loadability_version = ?1
                  WHERE session_key = (SELECT session_key FROM sessions WHERE id = ?2)",
                params![CATALOG_LOADABILITY_VERSION + 1, incompatible.id.as_str()],
            )
            .unwrap();

        let catalog = store.catalog("project", ArchiveFilter::Active).unwrap();
        assert_eq!(
            catalog
                .into_iter()
                .map(|entry| entry.metadata.id)
                .collect::<Vec<_>>(),
            vec![visible.id]
        );
        assert!(matches!(
            store.load(&incompatible.id),
            Err(SessionStoreError::Corrupt(message))
                if message.contains("loadability")
        ));
    }

    #[test]
    fn invalid_session_ids_never_commit_and_catalog_skips_corrupt_rows_individually() {
        let store = SessionStore::open_in_memory().unwrap();
        let visible = create_session(&store, "visible");
        let corrupt_id = create_session(&store, "corrupt-id");
        let corrupt_config = create_session(&store, "corrupt-config");
        let before = store
            .connection()
            .unwrap()
            .query_row("SELECT count(*) FROM sessions", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();

        let invalid = store.create_session(CreateStoredSession {
            id: SessionId::from_raw("../not-a-locator"),
            project_id: "project".into(),
            title: "invalid".into(),
            config: SessionConfig::default(),
            created_at_ms: 1_000,
        });
        assert!(matches!(invalid, Err(SessionStoreError::Invalid(_))));
        assert_eq!(
            store
                .connection()
                .unwrap()
                .query_row("SELECT count(*) FROM sessions", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            before
        );

        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE sessions SET id = 'bad/id' WHERE id = ?1",
                params![corrupt_id.id.as_str()],
            )
            .unwrap();
        store
            .connection()
            .unwrap()
            .execute(
                "UPDATE sessions SET config_json = x'00' WHERE id = ?1",
                params![corrupt_config.id.as_str()],
            )
            .unwrap();

        let catalog = store.catalog("project", ArchiveFilter::Active).unwrap();
        assert_eq!(
            catalog
                .into_iter()
                .map(|entry| entry.metadata.id)
                .collect::<Vec<_>>(),
            [visible.id]
        );
    }

    #[test]
    fn catalog_snapshot_never_mixes_journal_and_projection_revisions() {
        const APPEND_COUNT: u64 = 100;

        let directory = test_directory("catalog-snapshot");
        let writer = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&writer, "catalog-snapshot");
        let permit = writer.acquire_writer(&session.id).unwrap();
        let reader = SessionStore::open_database_readonly(writer.database_path().unwrap()).unwrap();
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer_done = done.clone();
        let writer_session_id = session.id.clone();
        let writer_thread = std::thread::spawn(move || {
            for revision in 0..APPEND_COUNT {
                let request = append_request(
                    &writer_session_id,
                    &format!("tx-catalog-{revision}"),
                    revision,
                    vec![recorded(
                        revision,
                        SessionEvent::InputSubmitted {
                            input_id: InputId::from_raw(format!("input-{revision}")),
                            input: format!("value-{revision}"),
                            origin: InputOrigin::Queue,
                        },
                    )],
                );
                writer.append(&request, &permit).unwrap();
                if revision % 8 == 0 {
                    std::thread::yield_now();
                }
            }
            writer_done.store(true, std::sync::atomic::Ordering::Release);
        });

        let mut snapshots = 0;
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            let catalog = reader.catalog("project", ArchiveFilter::Active).unwrap();
            let entry = &catalog[0];
            assert_eq!(entry.metadata.revision as usize, entry.search_values.len());
            snapshots += 1;
        }
        writer_thread.join().unwrap();
        let final_catalog = reader.catalog("project", ArchiveFilter::Active).unwrap();
        assert_eq!(final_catalog[0].metadata.revision, APPEND_COUNT);
        assert_eq!(final_catalog[0].search_values.len(), APPEND_COUNT as usize);
        assert!(snapshots > 0);

        drop(reader);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn catalog_scales_by_projection_rows_for_ten_thousand_sessions_without_replay() {
        const SESSION_COUNT: usize = 10_000;
        let store = SessionStore::open_in_memory().unwrap();
        let config = serde_json::to_vec(&SessionConfig::default()).unwrap();
        {
            let mut connection = store.connection().unwrap();
            let transaction = connection.transaction().unwrap();
            let mut insert_session = transaction
                .prepare_cached(
                    "INSERT INTO sessions (
                        id, project_id, title, config_json, created_at_ms, updated_at_ms,
                        archived_at_ms, revision, next_event_seq
                     ) VALUES (?1, 'project', ?1, ?2, ?3, ?3, NULL, 0, 0)",
                )
                .unwrap();
            let mut insert_projection = transaction
                .prepare_cached(
                    "INSERT INTO session_catalog_projection (
                        session_key, indexed_revision, extractor_version,
                        loadability_version, valid
                     ) VALUES (?1, 0, ?2, ?3, 1)",
                )
                .unwrap();
            for index in 0..SESSION_COUNT {
                insert_session
                    .execute(params![format!("catalog-{index}"), &config, index as i64])
                    .unwrap();
                insert_projection
                    .execute(params![
                        transaction.last_insert_rowid(),
                        CATALOG_EXTRACTOR_VERSION,
                        CATALOG_LOADABILITY_VERSION
                    ])
                    .unwrap();
            }
            drop(insert_projection);
            drop(insert_session);
            transaction.commit().unwrap();
        }

        let catalog_started = std::time::Instant::now();
        let catalog = store.catalog("project", ArchiveFilter::Active).unwrap();
        let catalog_elapsed = catalog_started.elapsed();
        assert_eq!(catalog.len(), SESSION_COUNT);
        assert!(
            catalog
                .iter()
                .all(|entry| entry.metadata.revision == 0 && entry.search_values.is_empty())
        );
        assert!(
            catalog_elapsed < std::time::Duration::from_secs(3),
            "10k-session projection query took {catalog_elapsed:?}"
        );
        eprintln!("10k-session projection catalog query: {catalog_elapsed:?}");
        let query_plan = store
            .connection()
            .unwrap()
            .prepare(
                "EXPLAIN QUERY PLAN
                 SELECT s.id, f.value
                 FROM sessions AS s
                 JOIN session_catalog_projection AS p
                  ON p.session_key = s.session_key
                  AND p.indexed_revision = s.revision
                  AND p.extractor_version = ?1
                  AND p.loadability_version = ?2
                  AND p.valid = 1
                 LEFT JOIN session_search_fragments AS f ON f.session_key = s.session_key
                 WHERE s.project_id = 'project' AND s.archived_at_ms IS NULL",
            )
            .unwrap()
            .query_map(
                params![CATALOG_EXTRACTOR_VERSION, CATALOG_LOADABILITY_VERSION],
                |row| row.get::<_, String>(3),
            )
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join("\n")
            .to_lowercase();
        assert!(!query_plan.contains("journal"));
    }

    #[test]
    #[ignore = "manual 100k-event storage and replay benchmark"]
    fn benchmark_100k_short_input_events_reports_footprint_and_replay() {
        const EVENT_COUNT: u64 = 100_000;
        const FOOTPRINT_GATE_BYTES: u64 = 20 * 1024 * 1024;

        let directory = test_directory("benchmark-100k");
        let store = SessionStore::open_project(&directory).unwrap();
        let session = create_session(&store, "benchmark");
        let permit = store.acquire_writer(&session.id).unwrap();
        let events = (0..EVENT_COUNT)
            .map(|seq| {
                recorded(
                    seq,
                    SessionEvent::InputSubmitted {
                        input_id: InputId::from_raw(format!("input-{seq}")),
                        input: "x".into(),
                        origin: InputOrigin::Queue,
                    },
                )
            })
            .collect();
        let request = append_request(&session.id, "tx-benchmark-100k", 0, events);

        let append_started = std::time::Instant::now();
        let receipt = store.append(&request, &permit).unwrap();
        let append_elapsed = append_started.elapsed();
        assert_eq!(receipt.events.len(), EVENT_COUNT as usize);
        drop(receipt);
        drop(request);

        let checkpoint_busy = store
            .connection()
            .unwrap()
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(checkpoint_busy, 0);
        let database_path = directory.join(SESSION_DATABASE_FILE);
        let footprint_bytes = sqlite_footprint_bytes(&database_path);
        assert!(
            footprint_bytes <= FOOTPRINT_GATE_BYTES,
            "100k-event database footprint is {:.2} MiB, above the 20 MiB gate",
            footprint_bytes as f64 / (1024.0 * 1024.0)
        );

        let load_started = std::time::Instant::now();
        let loaded = store.load(&session.id).unwrap();
        let load_elapsed = load_started.elapsed();
        let replay_events = loaded.events().cloned().collect::<Vec<_>>();
        assert_eq!(replay_events.len(), EVENT_COUNT as usize);

        let replay_started = std::time::Instant::now();
        let machine = SessionMachine::from_events(&replay_events).unwrap();
        let replay_elapsed = replay_started.elapsed();
        assert_eq!(machine.next_seq(), EVENT_COUNT);

        eprintln!(
            "100k InputSubmitted: db={:.2} MiB, append={append_elapsed:?}, load={load_elapsed:?} ({:.0} events/s), machine replay={replay_elapsed:?} ({:.0} events/s)",
            footprint_bytes as f64 / (1024.0 * 1024.0),
            EVENT_COUNT as f64 / load_elapsed.as_secs_f64().max(f64::EPSILON),
            EVENT_COUNT as f64 / replay_elapsed.as_secs_f64().max(f64::EPSILON),
        );

        drop(machine);
        drop(replay_events);
        drop(loaded);
        drop(permit);
        drop(store);
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn run_append_contract(store: SessionStore) {
        let session = create_session(&store, "contract");
        let permit = store.acquire_writer(&session.id).unwrap();
        let first = append_request(&session.id, "tx-one", 0, two_events());
        let receipt = store.append(&first, &permit).unwrap();
        assert_eq!(receipt.base_revision, 0);
        assert_eq!(receipt.revision, 1);
        assert_eq!(
            receipt
                .events
                .iter()
                .map(|event| event.seq)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(store.append(&first, &permit).unwrap(), receipt);

        let changed = append_request(
            &session.id,
            "tx-one",
            0,
            vec![recorded(
                9,
                SessionEvent::RunStarted {
                    run_id: RunId::from_raw("different-run"),
                },
            )],
        );
        assert!(matches!(
            store.append(&changed, &permit),
            Err(SessionStoreError::TransactionConflict { .. })
        ));

        let stale = append_request(
            &session.id,
            "tx-stale",
            0,
            vec![recorded(
                2,
                SessionEvent::RunStarted {
                    run_id: RunId::from_raw("stale-run"),
                },
            )],
        );
        assert!(matches!(
            store.append(&stale, &permit),
            Err(SessionStoreError::RevisionConflict {
                expected_revision: 0,
                current_revision: 1,
                ..
            })
        ));

        let wrong_sequence = append_request(
            &session.id,
            "tx-wrong-sequence",
            1,
            vec![recorded(
                9,
                SessionEvent::StepStarted {
                    turn_id: TurnId::from_raw("turn-1"),
                    step_id: StepId::from_raw("wrong-step"),
                },
            )],
        );
        assert!(matches!(
            store.append(&wrong_sequence, &permit),
            Err(SessionStoreError::EventSequenceConflict {
                expected: 2,
                found: 9,
            })
        ));

        let second = append_request(
            &session.id,
            "tx-two",
            1,
            vec![recorded(
                2,
                SessionEvent::StepStarted {
                    turn_id: TurnId::from_raw("turn-1"),
                    step_id: StepId::from_raw("step-1"),
                },
            )],
        );
        let second_receipt = store.append(&second, &permit).unwrap();
        assert_eq!(second_receipt.revision, 2);
        assert_eq!(second_receipt.events[0].seq, 2);
        let loaded = store.load(&session.id).unwrap();
        assert_eq!(loaded.metadata.revision, 2);
        assert_eq!(loaded.transactions.len(), 2);
        assert_eq!(loaded.events().count(), 3);
    }

    fn create_session(store: &SessionStore, label: &str) -> super::StoredSessionMetadata {
        store
            .create_session(CreateStoredSession {
                id: SessionId::new(),
                project_id: "project".into(),
                title: format!("{label} session"),
                config: SessionConfig::default(),
                created_at_ms: 1_000,
            })
            .unwrap()
    }

    fn append_request(
        session_id: &SessionId,
        tx_id: &str,
        expected_revision: u64,
        events: Vec<RecordedEvent>,
    ) -> AppendTx {
        let tx_id = TransactionId::from_raw(tx_id);
        let events = events
            .into_iter()
            .map(|mut event| {
                event.tx_id = tx_id.clone();
                event
            })
            .collect();
        AppendTx {
            session_id: session_id.clone(),
            tx_id,
            expected_revision,
            events,
        }
    }

    fn two_events() -> Vec<RecordedEvent> {
        vec![
            recorded(
                0,
                SessionEvent::RunStarted {
                    run_id: RunId::from_raw("run-1"),
                },
            ),
            recorded(
                1,
                SessionEvent::TurnStarted {
                    run_id: RunId::from_raw("run-1"),
                    turn_id: TurnId::from_raw("turn-1"),
                },
            ),
        ]
    }

    fn recorded(monotonic_ns: u64, event: SessionEvent) -> RecordedEvent {
        RecordedEvent {
            seq: monotonic_ns,
            tx_id: TxId::from_raw("unassigned"),
            time: event_time(monotonic_ns),
            event,
        }
    }

    fn event_time(monotonic_ns: u64) -> EventTime {
        EventTime {
            wall_time_ms: monotonic_ns as i64,
            clock_id: "test-clock".into(),
            monotonic_ns,
        }
    }

    fn test_directory(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "kcastle-session-store-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    fn sqlite_footprint_bytes(database_path: &std::path::Path) -> u64 {
        ["", "-wal", "-shm"]
            .into_iter()
            .filter_map(|suffix| {
                let mut path = database_path.as_os_str().to_os_string();
                path.push(suffix);
                std::fs::metadata(std::path::PathBuf::from(path))
                    .ok()
                    .map(|metadata| metadata.len())
            })
            .sum()
    }
}
