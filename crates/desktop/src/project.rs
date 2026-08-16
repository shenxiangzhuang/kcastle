use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const REGISTRY_FILE: &str = "projects.json";

#[derive(Debug, Clone)]
pub(crate) struct Project {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) sessions_dir: PathBuf,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Registry {
    projects: Vec<PathBuf>,
}

pub(crate) struct ProjectStore {
    root: PathBuf,
    projects: Vec<Project>,
}

impl ProjectStore {
    pub(crate) fn load(
        root: PathBuf,
        initial_project: PathBuf,
    ) -> Result<(Self, usize), Box<dyn Error>> {
        fs::create_dir_all(root.join("projects"))?;
        let registry_path = root.join(REGISTRY_FILE);
        let registry = match fs::read(&registry_path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Registry::default(),
            Err(error) => return Err(error.into()),
        };
        let projects = registry
            .projects
            .into_iter()
            .filter(|path| path.is_dir())
            .map(|path| Project::new(path, &root))
            .collect();
        let mut store = Self { root, projects };
        let active = store.add(initial_project)?;
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
        let project = Project::new(path, &self.root);
        fs::create_dir_all(&project.sessions_dir)?;
        self.projects.push(project);
        self.save()?;
        Ok(self.projects.len() - 1)
    }

    pub(crate) fn remove(&mut self, index: usize) -> Result<(), Box<dyn Error>> {
        if index >= self.projects.len() {
            return Ok(());
        }
        self.projects.remove(index);
        self.save()?;
        Ok(())
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        let registry = Registry {
            projects: self
                .projects
                .iter()
                .map(|project| project.path.clone())
                .collect(),
        };
        let path = self.root.join(REGISTRY_FILE);
        let temporary = self.root.join(format!("{REGISTRY_FILE}.tmp"));
        fs::write(&temporary, serde_json::to_vec_pretty(&registry)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }
}

impl Project {
    fn new(path: PathBuf, root: &Path) -> Self {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("Workspace")
            .to_owned();
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
        let id = format!("{}-{hash:016x}", slug.trim_matches('-'));
        Self {
            name,
            path,
            sessions_dir: root.join("projects").join(id).join("sessions"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn projects_have_isolated_session_directories_and_persist() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("kcastle-project-test-{suffix}"));
        let first = root.join("first");
        let second = root.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();

        let (mut store, first_index) = ProjectStore::load(root.clone(), first.clone()).unwrap();
        let second_index = store.add(second.clone()).unwrap();
        assert_ne!(
            store.project(first_index).unwrap().sessions_dir,
            store.project(second_index).unwrap().sessions_dir
        );

        let (reloaded, active) = ProjectStore::load(root.clone(), second.clone()).unwrap();
        assert_eq!(reloaded.projects().len(), 2);
        assert_eq!(reloaded.project(active).unwrap().name, "second");

        let mut reloaded = reloaded;
        reloaded.remove(0).unwrap();
        let (reloaded, _) = ProjectStore::load(root.clone(), second).unwrap();
        assert_eq!(reloaded.projects().len(), 1);
        assert_eq!(reloaded.projects()[0].name, "second");
        fs::remove_dir_all(root).unwrap();
    }
}
