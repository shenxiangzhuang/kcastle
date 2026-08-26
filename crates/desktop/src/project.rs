use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::app_store::AppStore;

const DEFAULT_PROJECT_ID: &str = "default";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ProjectId(String);

impl ProjectId {
    pub(crate) fn default_project() -> Self {
        Self(DEFAULT_PROJECT_ID.into())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Project {
    pub(crate) id: ProjectId,
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) sessions_dir: PathBuf,
    pub(crate) missing: bool,
}

impl Project {
    pub(crate) fn is_default(&self) -> bool {
        self.id.as_str() == DEFAULT_PROJECT_ID
    }
}

#[derive(Debug, Clone)]
struct ProjectRecord {
    id: ProjectId,
    name: String,
    path: PathBuf,
    archived: bool,
}

pub(crate) struct ProjectStore {
    store: AppStore,
    projects: Vec<Project>,
    archived: Vec<ProjectRecord>,
}

impl ProjectStore {
    pub(crate) fn load(
        source: impl ProjectAppStoreSource,
        initial_project: Option<PathBuf>,
    ) -> Result<(Self, usize), Box<dyn Error>> {
        let store = source.into_app_store()?;
        let records = load_project_records(&store)?;

        let mut projects = Vec::new();
        let mut archived = Vec::new();
        for record in records {
            if record.archived {
                archived.push(record);
            } else {
                projects.push(Project::from_record(record, &store));
            }
        }

        let default = Project::default_project(&store);
        fs::create_dir_all(&default.sessions_dir)?;
        if let Some(index) = projects.iter().position(Project::is_default) {
            projects[index] = default;
        } else {
            projects.insert(0, default);
        }

        let mut store = Self {
            store,
            projects,
            archived,
        };
        let active = match initial_project {
            Some(path) => store.add(path)?,
            None => store
                .projects
                .iter()
                .position(Project::is_default)
                .expect("default project must exist"),
        };
        Ok((store, active))
    }

    pub(crate) fn projects(&self) -> &[Project] {
        &self.projects
    }

    pub(crate) fn project(&self, index: usize) -> Option<&Project> {
        self.projects.get(index)
    }

    pub(crate) fn add(&mut self, path: PathBuf) -> Result<usize, Box<dyn Error>> {
        let path = path.canonicalize()?;
        if !path.is_dir() {
            return Err(format!("project is not a directory: {}", path.display()).into());
        }
        if let Some(index) = self
            .projects
            .iter()
            .position(|project| project.path == path)
        {
            return Ok(index);
        }
        if let Some(index) = self
            .archived
            .iter()
            .position(|project| project.path == path)
        {
            let mut record = self.archived.remove(index);
            record.archived = false;
            let project = Project::from_record(record, &self.store);
            fs::create_dir_all(&project.sessions_dir)?;
            self.projects.push(project);
            self.save()?;
            return Ok(self.projects.len() - 1);
        }

        let project = Project::new(path, &self.store);
        fs::create_dir_all(&project.sessions_dir)?;
        self.projects.push(project);
        self.save()?;
        Ok(self.projects.len() - 1)
    }

    pub(crate) fn remove(&mut self, index: usize) -> Result<(), Box<dyn Error>> {
        let Some(project) = self.projects.get(index) else {
            return Ok(());
        };
        if project.is_default() {
            return Err("the default project cannot be removed".into());
        }
        let project = self.projects.remove(index);
        let mut record = project.into_record();
        record.archived = true;
        self.archived.push(record);
        self.save()?;
        Ok(())
    }

    pub(crate) fn relocate(
        &mut self,
        index: usize,
        new_path: PathBuf,
    ) -> Result<(), Box<dyn Error>> {
        let new_path = new_path.canonicalize()?;
        if !new_path.is_dir() {
            return Err(format!("project is not a directory: {}", new_path.display()).into());
        }
        if self.projects.get(index).is_some_and(Project::is_default) {
            return Err("the default project cannot be relocated".into());
        }
        if self
            .projects
            .iter()
            .enumerate()
            .any(|(other, project)| other != index && project.path == new_path)
            || self.archived.iter().any(|project| project.path == new_path)
        {
            return Err(format!(
                "project directory is already registered: {}",
                new_path.display()
            )
            .into());
        }
        let project = self
            .projects
            .get_mut(index)
            .ok_or_else(|| format!("unknown project index: {index}"))?;
        project.name = display_name(&new_path);
        project.path = new_path;
        project.missing = false;
        self.save()
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        self.store.write(|transaction| {
            transaction.execute("DELETE FROM projects", [])?;
            let mut ordinal = 0_i64;
            for project in &self.projects {
                insert_project(
                    transaction,
                    project.id.as_str(),
                    ordinal,
                    &project.name,
                    &project.path,
                    false,
                )?;
                ordinal += 1;
            }
            for project in &self.archived {
                insert_project(
                    transaction,
                    project.id.as_str(),
                    ordinal,
                    &project.name,
                    &project.path,
                    true,
                )?;
                ordinal += 1;
            }
            Ok(())
        })
    }
}

impl Project {
    fn default_project(store: &AppStore) -> Self {
        let path = store.project_root(DEFAULT_PROJECT_ID);
        Self {
            id: ProjectId::default_project(),
            name: "Default".into(),
            path,
            sessions_dir: store.sessions_dir(DEFAULT_PROJECT_ID),
            missing: false,
        }
    }

    fn new(path: PathBuf, store: &AppStore) -> Self {
        let name = display_name(&path);
        let storage_key = storage_key(&name, &path);
        Self {
            id: ProjectId(storage_key.clone()),
            name,
            path,
            sessions_dir: store.sessions_dir(&storage_key),
            missing: false,
        }
    }

    fn from_record(record: ProjectRecord, store: &AppStore) -> Self {
        let missing = !record.path.is_dir();
        let sessions_dir = store.sessions_dir(record.id.as_str());
        Self {
            id: record.id,
            name: record.name,
            path: record.path,
            sessions_dir,
            missing,
        }
    }

    fn into_record(self) -> ProjectRecord {
        ProjectRecord {
            id: self.id,
            name: self.name,
            path: self.path,
            archived: false,
        }
    }
}

pub(crate) trait ProjectAppStoreSource {
    fn into_app_store(self) -> Result<AppStore, Box<dyn Error>>;
}

impl ProjectAppStoreSource for AppStore {
    fn into_app_store(self) -> Result<AppStore, Box<dyn Error>> {
        Ok(self)
    }
}

impl ProjectAppStoreSource for PathBuf {
    fn into_app_store(self) -> Result<AppStore, Box<dyn Error>> {
        AppStore::open(self)
    }
}

fn load_project_records(store: &AppStore) -> Result<Vec<ProjectRecord>, Box<dyn Error>> {
    let connection = store.connection()?;
    let mut statement = connection.prepare(
        "SELECT project_id, name, workspace_path_json, archived FROM projects ORDER BY ordinal",
    )?;
    let rows = statement.query_map([], |row| {
        let id = ProjectId(row.get(0)?);
        validate_project_id(id.as_str()).map_err(|error| {
            rusqlite::Error::FromSqlConversionFailure(
                0,
                rusqlite::types::Type::Text,
                Box::new(error),
            )
        })?;
        Ok(ProjectRecord {
            id,
            name: row.get(1)?,
            path: serde_json::from_slice(&row.get::<_, Vec<u8>>(2)?).map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    2,
                    rusqlite::types::Type::Blob,
                    Box::new(error),
                )
            })?,
            archived: row.get::<_, i64>(3)? != 0,
        })
    })?;
    Ok(rows.collect::<Result<Vec<_>, _>>()?)
}

fn insert_project(
    transaction: &rusqlite::Transaction<'_>,
    project_id: &str,
    ordinal: i64,
    name: &str,
    path: &Path,
    archived: bool,
) -> Result<(), Box<dyn Error>> {
    validate_project_id(project_id)?;
    let workspace_path_json = serde_json::to_vec(path)?;
    transaction.execute(
        "INSERT INTO projects (project_id, ordinal, name, workspace_path_json, archived)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            project_id,
            ordinal,
            name,
            workspace_path_json,
            i64::from(archived),
        ],
    )?;
    Ok(())
}

fn validate_project_id(project_id: &str) -> Result<(), std::io::Error> {
    if !project_id.is_empty()
        && project_id.len() <= 128
        && project_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid project ID: {project_id:?}"),
        ))
    }
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("Workspace")
        .to_owned()
}

fn storage_key(name: &str, path: &Path) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in path.as_os_str().to_string_lossy().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let slug = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    format!("{}-{hash:016x}", slug.trim_matches('-'))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn default_project_is_explicit_and_uses_the_default_directory() {
        let root = test_root("default");
        let (store, active) = ProjectStore::load(root.clone(), None).unwrap();
        let project = store.project(active).unwrap();

        assert!(project.is_default());
        assert_eq!(project.name, "Default");
        assert_eq!(project.path, root.join("projects/default"));
        assert_eq!(project.sessions_dir, root.join("projects/default/sessions"));
        assert!(project.sessions_dir.is_dir());

        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn projects_have_isolated_stable_session_directories_and_persist() {
        let root = test_root("isolated");
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();

        let (mut store, first_index) =
            ProjectStore::load(root.clone(), Some(first.clone())).unwrap();
        let first_sessions = store.project(first_index).unwrap().sessions_dir.clone();
        let second_index = store.add(second.clone()).unwrap();
        assert_ne!(
            first_sessions,
            store.project(second_index).unwrap().sessions_dir
        );

        let relocated = root.join("relocated");
        fs::create_dir_all(&relocated).unwrap();
        let relocated = relocated.canonicalize().unwrap();
        store.relocate(first_index, relocated.clone()).unwrap();
        assert_eq!(store.project(first_index).unwrap().path, relocated);
        assert_eq!(
            store.project(first_index).unwrap().sessions_dir,
            first_sessions
        );

        let (reloaded, active) = ProjectStore::load(root.clone(), Some(second.clone())).unwrap();
        assert_eq!(reloaded.projects().len(), 3);
        assert_eq!(reloaded.project(active).unwrap().name, "second");

        drop(reloaded);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_and_removed_projects_are_retained_without_becoming_default() {
        let root = test_root("missing");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let workspace = workspace.canonicalize().unwrap();
        let (store, _) = ProjectStore::load(root.clone(), Some(workspace.clone())).unwrap();
        let sessions_dir = store
            .projects()
            .iter()
            .find(|project| project.path == workspace)
            .unwrap()
            .sessions_dir
            .clone();
        drop(store);
        fs::remove_dir_all(&workspace).unwrap();

        let (mut reloaded, _) = ProjectStore::load(root.clone(), None).unwrap();
        let missing = reloaded
            .projects()
            .iter()
            .position(|project| project.sessions_dir == sessions_dir)
            .unwrap();
        assert!(reloaded.project(missing).unwrap().missing);
        reloaded.remove(missing).unwrap();

        drop(reloaded);
        let (reloaded, _) = ProjectStore::load(root.clone(), None).unwrap();
        assert!(
            reloaded
                .projects()
                .iter()
                .all(|project| project.sessions_dir != sessions_dir)
        );
        assert!(reloaded.projects().iter().any(Project::is_default));

        drop(reloaded);
        fs::remove_dir_all(root).unwrap();
    }

    fn test_root(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("kcastle-project-{label}-{suffix}"))
    }
}
