use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rusqlite::{Connection, Transaction, TransactionBehavior};

const APP_DATABASE_FILE: &str = "app.sqlite3";
const APP_DATABASE_SCHEMA_VERSION: u32 = 1;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS legacy_migrations (
    name TEXT PRIMARY KEY,
    completed_at_ms INTEGER NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS app_preferences (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    legacy_reasoning_effort TEXT,
    selected_model TEXT,
    allow_all_tools INTEGER NOT NULL CHECK (allow_all_tools IN (0, 1)),
    appearance TEXT NOT NULL CHECK (appearance IN ('system', 'light', 'dark')),
    enter_behavior TEXT NOT NULL CHECK (enter_behavior IN ('steer', 'queue')),
    reduce_motion INTEGER NOT NULL CHECK (reduce_motion IN (0, 1)),
    trajectory_actual_duration INTEGER NOT NULL CHECK (trajectory_actual_duration IN (0, 1))
) STRICT;

CREATE TABLE IF NOT EXISTS model_reasoning_efforts (
    model_id TEXT PRIMARY KEY,
    effort TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS providers (
    provider_id TEXT PRIMARY KEY CHECK (length(provider_id) > 0),
    ordinal INTEGER NOT NULL UNIQUE CHECK (ordinal >= 0),
    display_name TEXT NOT NULL,
    api_base TEXT NOT NULL,
    api_key TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS provider_models (
    provider_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    model_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    context_window INTEGER NOT NULL CHECK (context_window > 0),
    max_output_tokens INTEGER CHECK (max_output_tokens > 0),
    PRIMARY KEY (provider_id, model_id),
    UNIQUE (provider_id, ordinal),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY
        CHECK (length(project_id) BETWEEN 1 AND 128)
        CHECK (project_id NOT GLOB '*[^A-Za-z0-9_-]*'),
    ordinal INTEGER NOT NULL UNIQUE CHECK (ordinal >= 0),
    name TEXT NOT NULL,
    workspace_path_json BLOB NOT NULL,
    archived INTEGER NOT NULL CHECK (archived IN (0, 1))
) STRICT;
"#;

#[derive(Clone)]
pub(crate) struct AppStore {
    root: PathBuf,
    connection: Arc<Mutex<Connection>>,
}

impl AppStore {
    pub(crate) fn open(root: PathBuf) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(root.join("projects"))?;
        let database = root.join(APP_DATABASE_FILE);
        let mut connection = Connection::open(&database)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.pragma_update(None, "foreign_keys", "ON")?;
        initialize_or_validate_schema(&mut connection)?;
        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        restrict_database_permissions(&database)?;
        Ok(Self {
            root,
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    pub(crate) fn project_root(&self, project_id: &str) -> PathBuf {
        self.root.join("projects").join(project_id)
    }

    pub(crate) fn sessions_dir(&self, project_id: &str) -> PathBuf {
        self.project_root(project_id).join("sessions")
    }

    pub(crate) fn connection(&self) -> Result<MutexGuard<'_, Connection>, Box<dyn Error>> {
        self.connection
            .lock()
            .map_err(|_| "app database lock was poisoned".into())
    }

    pub(crate) fn write<T>(
        &self,
        operation: impl FnOnce(&Transaction<'_>) -> Result<T, Box<dyn Error>>,
    ) -> Result<T, Box<dyn Error>> {
        let mut connection = self.connection()?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value = operation(&transaction)?;
        transaction.commit()?;
        Ok(value)
    }

    pub(crate) fn migration_complete(&self, name: &str) -> Result<bool, Box<dyn Error>> {
        let connection = self.connection()?;
        Ok(connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM legacy_migrations WHERE name = ?1)",
            [name],
            |row| row.get(0),
        )?)
    }

    pub(crate) fn mark_migration_complete(
        transaction: &Transaction<'_>,
        name: &str,
    ) -> Result<(), Box<dyn Error>> {
        transaction.execute(
            "INSERT OR IGNORE INTO legacy_migrations (name, completed_at_ms)
             VALUES (?1, unixepoch('subsec') * 1000)",
            [name],
        )?;
        Ok(())
    }

    pub(crate) fn backup_legacy_file(&self, name: &str) -> Result<(), Box<dyn Error>> {
        let source = self.root.join(name);
        if !source.is_file() {
            return Ok(());
        }
        let backup_directory = self.root.join("backups").join("pre-app-sqlite");
        fs::create_dir_all(&backup_directory)?;
        let mut destination = backup_directory.join(name);
        if destination.exists() {
            if fs::read(&source)? == fs::read(&destination)? {
                fs::remove_file(source)?;
                return Ok(());
            }
            let mut suffix = 1_u64;
            loop {
                let candidate = backup_directory.join(format!("{name}.{suffix}"));
                if !candidate.exists() {
                    destination = candidate;
                    break;
                }
                suffix = suffix
                    .checked_add(1)
                    .ok_or("legacy backup suffix overflowed")?;
            }
        }
        fs::rename(source, destination)?;
        Ok(())
    }
}

fn initialize_or_validate_schema(connection: &mut Connection) -> Result<(), Box<dyn Error>> {
    let found = connection.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))?;
    match found {
        version if version == i64::from(APP_DATABASE_SCHEMA_VERSION) => Ok(()),
        0 => {
            let user_table_count = connection.query_row(
                "SELECT count(*) FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get::<_, i64>(0),
            )?;
            if user_table_count != 0 {
                return Err("unversioned app database already contains tables".into());
            }
            let transaction =
                connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
            transaction.execute_batch(SCHEMA)?;
            transaction.pragma_update(None, "user_version", APP_DATABASE_SCHEMA_VERSION)?;
            transaction.commit()?;
            Ok(())
        }
        _ => Err(format!(
            "unsupported app database schema {found}; expected {APP_DATABASE_SCHEMA_VERSION}"
        )
        .into()),
    }
}

#[cfg(unix)]
fn restrict_database_permissions(database: &Path) -> Result<(), Box<dyn Error>> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(database, fs::Permissions::from_mode(0o600))?;
    for suffix in ["-wal", "-shm"] {
        let sidecar = PathBuf::from(format!("{}{suffix}", database.display()));
        if sidecar.exists() {
            fs::set_permissions(sidecar, fs::Permissions::from_mode(0o600))?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn restrict_database_permissions(_database: &Path) -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::project::ProjectStore;
    use crate::settings::SettingsStore;

    use super::*;

    #[test]
    fn app_database_uses_wal_and_the_supported_schema() {
        let root = test_root("schema");
        let store = AppStore::open(root.clone()).unwrap();
        let connection = store.connection().unwrap();
        let journal_mode = connection
            .pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))
            .unwrap();
        let schema_version = connection
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap();
        let integrity = connection
            .query_row("PRAGMA quick_check", [], |row| row.get::<_, String>(0))
            .unwrap();
        assert_eq!(journal_mode, "wal");
        assert_eq!(schema_version, i64::from(APP_DATABASE_SCHEMA_VERSION));
        assert_eq!(integrity, "ok");
        drop(connection);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn app_database_and_wal_sidecars_are_user_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = test_root("permissions");
        let store = AppStore::open(root.clone()).unwrap();
        let database = root.join(APP_DATABASE_FILE);
        assert_eq!(
            fs::metadata(&database).unwrap().permissions().mode() & 0o777,
            0o600
        );
        for suffix in ["-wal", "-shm"] {
            let sidecar = PathBuf::from(format!("{}{suffix}", database.display()));
            if sidecar.exists() {
                assert_eq!(
                    fs::metadata(sidecar).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unsupported_app_schema_is_rejected_without_rewriting_it() {
        let root = test_root("unsupported");
        fs::create_dir_all(&root).unwrap();
        let database = root.join(APP_DATABASE_FILE);
        let connection = Connection::open(&database).unwrap();
        connection.pragma_update(None, "user_version", 99).unwrap();
        drop(connection);

        let error = match AppStore::open(root.clone()) {
            Ok(_) => panic!("unsupported schema unexpectedly opened"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("unsupported app database schema 99")
        );
        let connection = Connection::open(database).unwrap();
        assert_eq!(
            connection
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            99
        );
        drop(connection);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn settings_and_projects_share_one_app_database_across_restarts() {
        let root = test_root("shared");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let store = AppStore::open(root.clone()).unwrap();
        let mut settings = SettingsStore::load(store.clone()).unwrap();
        settings.set_selected_model("provider/model").unwrap();
        let (projects, active) = ProjectStore::load(store, Some(workspace.clone())).unwrap();
        assert_eq!(
            projects.project(active).unwrap().path,
            workspace.canonicalize().unwrap()
        );
        drop(projects);
        drop(settings);

        let store = AppStore::open(root.clone()).unwrap();
        let settings = SettingsStore::load(store.clone()).unwrap();
        let (projects, _) = ProjectStore::load(store, None).unwrap();
        assert_eq!(settings.selected_model(), Some("provider/model"));
        assert!(
            projects
                .projects()
                .iter()
                .any(|project| project.name == "workspace")
        );
        assert!(root.join(APP_DATABASE_FILE).is_file());
        assert!(!root.join("settings.json").exists());
        assert!(!root.join("projects.json").exists());
        drop(projects);
        drop(settings);
        fs::remove_dir_all(root).unwrap();
    }

    fn test_root(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kcastle-app-store-{label}-{suffix}"))
    }
}
