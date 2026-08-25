use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use kcastle_agent::{
    Session, SessionCatalog, SessionError, SessionErrorClass, SessionId, SessionInfo,
};

use crate::project::{ProjectId, ProjectStore};

#[derive(Clone, Debug)]
pub(crate) struct SessionSearchDocument {
    pub(crate) searchable: Arc<str>,
    pub(crate) summary: String,
    pub(crate) snippets: Arc<[String]>,
}

#[derive(Default)]
pub(crate) struct SessionCatalogCache {
    pub(crate) project_sessions: HashMap<PathBuf, Vec<SessionInfo>>,
    pub(crate) session_search_documents: HashMap<PathBuf, SessionSearchDocument>,
    pub(crate) session_catalog_indices: HashMap<(ProjectId, SessionId), usize>,
}

pub(crate) fn should_clear_catalog_after_error(error: &SessionError) -> bool {
    error.classification() == SessionErrorClass::DeterministicInvalid
}

pub(crate) fn load_session_catalog_cache(project_store: &ProjectStore) -> SessionCatalogCache {
    load_session_catalog_cache_with(project_store, |directory, project_id| {
        Session::catalog_in_project(directory, project_id)
    })
}

pub(crate) fn load_session_catalog_cache_with(
    project_store: &ProjectStore,
    mut load: impl FnMut(&Path, &str) -> Result<SessionCatalog, SessionError>,
) -> SessionCatalogCache {
    let project_count = project_store.projects().len();
    let mut cache = SessionCatalogCache {
        project_sessions: HashMap::with_capacity(project_count),
        session_search_documents: HashMap::new(),
        session_catalog_indices: HashMap::new(),
    };
    for project in project_store.projects() {
        cache
            .project_sessions
            .entry(project.sessions_dir.clone())
            .or_default();
        apply_project_catalog_result(
            &project.id,
            &project.sessions_dir,
            load(&project.sessions_dir, project.id.as_str()),
            &mut cache.project_sessions,
            &mut cache.session_search_documents,
            &mut cache.session_catalog_indices,
        );
    }
    cache
}

pub(crate) fn session_search_document(values: Arc<[String]>) -> SessionSearchDocument {
    let searchable = values.join("\n").to_lowercase().into();
    let summary = values
        .iter()
        .find(|value| value.chars().count() >= 4 && !value.starts_with("resp_"))
        .map(|value| truncate_chars(value, 88))
        .unwrap_or_default();
    SessionSearchDocument {
        searchable,
        summary,
        snippets: values,
    }
}

pub(crate) fn remove_project_catalog_members(
    project_id: &ProjectId,
    sessions_dir: &Path,
    project_sessions: &HashMap<PathBuf, Vec<SessionInfo>>,
    search_documents: &mut HashMap<PathBuf, SessionSearchDocument>,
    catalog_indices: &mut HashMap<(ProjectId, SessionId), usize>,
) {
    let Some(sessions) = project_sessions.get(sessions_dir) else {
        return;
    };
    for session in sessions {
        search_documents.remove(&session.path);
        catalog_indices.remove(&(project_id.clone(), session.id.clone()));
    }
}

pub(crate) fn clear_project_catalog_cache(
    project_id: &ProjectId,
    sessions_dir: &Path,
    project_sessions: &mut HashMap<PathBuf, Vec<SessionInfo>>,
    search_documents: &mut HashMap<PathBuf, SessionSearchDocument>,
    catalog_indices: &mut HashMap<(ProjectId, SessionId), usize>,
) {
    remove_project_catalog_members(
        project_id,
        sessions_dir,
        project_sessions,
        search_documents,
        catalog_indices,
    );
    project_sessions.insert(sessions_dir.to_owned(), Vec::new());
}

pub(crate) fn apply_project_catalog_result(
    project_id: &ProjectId,
    sessions_dir: &Path,
    result: Result<SessionCatalog, SessionError>,
    project_sessions: &mut HashMap<PathBuf, Vec<SessionInfo>>,
    search_documents: &mut HashMap<PathBuf, SessionSearchDocument>,
    catalog_indices: &mut HashMap<(ProjectId, SessionId), usize>,
) -> bool {
    match result {
        Ok(catalog) => {
            replace_project_catalog_cache(
                project_id,
                sessions_dir,
                catalog,
                project_sessions,
                search_documents,
                catalog_indices,
            );
            true
        }
        Err(error) if should_clear_catalog_after_error(&error) => {
            clear_project_catalog_cache(
                project_id,
                sessions_dir,
                project_sessions,
                search_documents,
                catalog_indices,
            );
            false
        }
        Err(_) => false,
    }
}

fn replace_project_catalog_cache(
    project_id: &ProjectId,
    sessions_dir: &Path,
    catalog: SessionCatalog,
    project_sessions: &mut HashMap<PathBuf, Vec<SessionInfo>>,
    search_documents: &mut HashMap<PathBuf, SessionSearchDocument>,
    catalog_indices: &mut HashMap<(ProjectId, SessionId), usize>,
) {
    remove_project_catalog_members(
        project_id,
        sessions_dir,
        project_sessions,
        search_documents,
        catalog_indices,
    );
    let SessionCatalog {
        sessions,
        search_values,
    } = catalog;
    search_documents.extend(
        search_values
            .into_iter()
            .map(|(path, values)| (path, session_search_document(values))),
    );
    project_sessions.insert(sessions_dir.to_owned(), sessions);
    if let Some(sessions) = project_sessions.get(sessions_dir) {
        catalog_indices.extend(
            sessions
                .iter()
                .enumerate()
                .map(|(index, session)| ((project_id.clone(), session.id.clone()), index)),
        );
    }
}

pub(crate) fn remove_session_catalog_entry(
    project_id: &ProjectId,
    sessions_dir: &Path,
    path: &Path,
    project_sessions: &mut HashMap<PathBuf, Vec<SessionInfo>>,
    search_documents: &mut HashMap<PathBuf, SessionSearchDocument>,
    catalog_indices: &mut HashMap<(ProjectId, SessionId), usize>,
) -> Option<SessionId> {
    search_documents.remove(path);
    let sessions = project_sessions.get_mut(sessions_dir)?;
    let index = sessions.iter().position(|session| session.path == path)?;
    let removed = sessions.remove(index);

    catalog_indices.remove(&(project_id.clone(), removed.id.clone()));
    catalog_indices.extend(
        sessions
            .iter()
            .enumerate()
            .map(|(index, session)| ((project_id.clone(), session.id.clone()), index)),
    );
    Some(removed.id)
}

pub(crate) fn load_project_archived_sessions(
    project_store: &ProjectStore,
) -> HashMap<PathBuf, Vec<SessionInfo>> {
    let mut sessions = HashMap::new();
    for project in project_store.projects() {
        match Session::archived_catalog_in_project(&project.sessions_dir, project.id.as_str()) {
            Ok(catalog) => {
                sessions.insert(project.sessions_dir.clone(), catalog.sessions);
            }
            Err(_) => {
                sessions.insert(project.sessions_dir.clone(), Vec::new());
            }
        }
    }
    sessions
}

pub(crate) fn truncate_chars(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let text = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{}…", text.trim_end())
    } else {
        text
    }
}

pub(crate) fn matching_search_snippet(values: &[String], query: &str) -> Option<String> {
    values
        .iter()
        .find(|value| value.to_lowercase().contains(query))
        .map(|value| truncate_chars(value, 88))
}
