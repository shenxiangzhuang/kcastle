use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const REGISTRY_FILE: &str = "projects.json";
const REGISTRY_VERSION: u32 = 2;
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

#[derive(Debug, Default, Serialize, Deserialize)]
struct LegacyRegistry {
    projects: Vec<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistryV2 {
    version: u32,
    projects: Vec<ProjectRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectRecord {
    id: ProjectId,
    name: String,
    path: PathBuf,
    sessions_dir: PathBuf,
    #[serde(default)]
    archived: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StoredRegistry {
    V2(RegistryV2),
    Legacy(LegacyRegistry),
}

pub(crate) struct ProjectStore {
    root: PathBuf,
    projects: Vec<Project>,
    archived: Vec<ProjectRecord>,
}

impl ProjectStore {
    pub(crate) fn load(
        root: PathBuf,
        initial_project: Option<PathBuf>,
    ) -> Result<(Self, usize), Box<dyn Error>> {
        fs::create_dir_all(root.join("sessions"))?;
        let registry_path = root.join(REGISTRY_FILE);
        let stored = match fs::read(&registry_path) {
            Ok(bytes) => Some(serde_json::from_slice::<StoredRegistry>(&bytes)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };

        let (records, migrated) = match stored {
            Some(StoredRegistry::V2(registry)) => {
                if registry.version != REGISTRY_VERSION {
                    return Err(format!(
                        "unsupported project registry version: {}",
                        registry.version
                    )
                    .into());
                }
                (registry.projects, false)
            }
            Some(StoredRegistry::Legacy(registry)) => (
                registry
                    .projects
                    .into_iter()
                    .map(|path| ProjectRecord::from_legacy(path, &root))
                    .collect(),
                true,
            ),
            None => (Vec::new(), true),
        };

        let mut projects = Vec::new();
        let mut archived = Vec::new();
        for record in records {
            if record.archived {
                archived.push(record);
            } else {
                projects.push(Project::from_record(record));
            }
        }

        let default = Project::default_project(&root);
        fs::create_dir_all(&default.sessions_dir)?;
        if let Some(index) = projects.iter().position(Project::is_default) {
            projects[index] = default;
        } else {
            projects.insert(0, default);
        }

        let mut store = Self {
            root,
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
        if migrated {
            store.save()?;
        }
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
            let project = Project::from_record(record);
            fs::create_dir_all(&project.sessions_dir)?;
            self.projects.push(project);
            self.save()?;
            return Ok(self.projects.len() - 1);
        }

        let project = Project::new(path, &self.root);
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
        let mut projects = self
            .projects
            .iter()
            .cloned()
            .map(Project::into_record)
            .collect::<Vec<_>>();
        projects.extend(self.archived.iter().cloned());
        let registry = RegistryV2 {
            version: REGISTRY_VERSION,
            projects,
        };
        let path = self.root.join(REGISTRY_FILE);
        let temporary = self.root.join(format!("{REGISTRY_FILE}.tmp"));
        let mut file = File::create(&temporary)?;
        file.write_all(&serde_json::to_vec_pretty(&registry)?)?;
        file.sync_all()?;
        fs::rename(&temporary, &path)?;
        if let Ok(directory) = File::open(&self.root) {
            let _ = directory.sync_all();
        }
        Ok(())
    }
}

impl Project {
    fn default_project(root: &Path) -> Self {
        let path = root.join("sessions").join(DEFAULT_PROJECT_ID);
        Self {
            id: ProjectId::default_project(),
            name: display_name(&path),
            path: path.clone(),
            sessions_dir: path,
            missing: false,
        }
    }

    fn new(path: PathBuf, root: &Path) -> Self {
        let name = display_name(&path);
        let storage_key = legacy_storage_key(&name, &path);
        Self {
            id: ProjectId(storage_key.clone()),
            name,
            path,
            sessions_dir: root.join("sessions").join(storage_key),
            missing: false,
        }
    }

    fn from_record(record: ProjectRecord) -> Self {
        let missing = !record.path.is_dir();
        Self {
            id: record.id,
            name: record.name,
            path: record.path,
            sessions_dir: record.sessions_dir,
            missing,
        }
    }

    fn into_record(self) -> ProjectRecord {
        ProjectRecord {
            id: self.id,
            name: self.name,
            path: self.path,
            sessions_dir: self.sessions_dir,
            archived: false,
        }
    }
}

impl ProjectRecord {
    fn from_legacy(path: PathBuf, root: &Path) -> Self {
        let name = display_name(&path);
        let storage_key = legacy_storage_key(&name, &path);
        let legacy_sessions = root.join("projects").join(&storage_key).join("sessions");
        let sessions_dir = if legacy_sessions.exists() {
            legacy_sessions
        } else {
            root.join("sessions").join(&storage_key)
        };
        Self {
            id: ProjectId(storage_key),
            name,
            path,
            sessions_dir,
            archived: false,
        }
    }
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("Workspace")
        .to_owned()
}

fn legacy_storage_key(name: &str, path: &Path) -> String {
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
        assert_eq!(project.name, "default");
        assert_eq!(project.path, root.join("sessions/default"));
        assert_eq!(project.sessions_dir, root.join("sessions/default"));
        assert!(project.sessions_dir.is_dir());

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

        let (reloaded, _) = ProjectStore::load(root.clone(), None).unwrap();
        assert!(
            reloaded
                .projects()
                .iter()
                .all(|project| project.sessions_dir != sessions_dir)
        );
        assert!(reloaded.projects().iter().any(Project::is_default));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_registry_keeps_existing_session_storage() {
        let root = test_root("legacy");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let name = display_name(&workspace);
        let storage_key = legacy_storage_key(&name, &workspace);
        let legacy_sessions = root.join("projects").join(storage_key).join("sessions");
        fs::create_dir_all(&legacy_sessions).unwrap();
        fs::write(
            root.join(REGISTRY_FILE),
            serde_json::to_vec(&LegacyRegistry {
                projects: vec![workspace.clone()],
            })
            .unwrap(),
        )
        .unwrap();

        let (store, _) = ProjectStore::load(root.clone(), None).unwrap();
        let project = store
            .projects()
            .iter()
            .find(|project| project.path == workspace)
            .unwrap();
        assert_eq!(project.sessions_dir, legacy_sessions);

        let persisted: RegistryV2 =
            serde_json::from_slice(&fs::read(root.join(REGISTRY_FILE)).unwrap()).unwrap();
        assert_eq!(persisted.version, REGISTRY_VERSION);

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
