use std::fs::{self, OpenOptions as StdOpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use async_openai::types::responses::{FunctionCallOutputItemParam, InputItem, Item};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;

use crate::state::{ResponseMetadata, State, StateEntry};

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("session I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid session: {0}")]
    Invalid(String),
    #[error("session serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionInfo {
    pub path: PathBuf,
    pub title: String,
    pub created_at: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
enum Record {
    Session { title: String, created_at: u64 },
    Title { title: String },
    Entry { entry: StateEntry },
}

#[derive(Debug)]
pub struct Session {
    info: SessionInfo,
    state: State,
    file: Option<File>,
}

pub trait StateCommit: Send + Sync {
    fn info(&self) -> &SessionInfo;

    fn set_initial_title<'a>(
        &'a mut self,
        message: &'a str,
    ) -> BoxFuture<'a, Result<(), SessionError>>;

    fn rename<'a>(&'a mut self, title: &'a str) -> BoxFuture<'a, Result<(), SessionError>>;

    fn commit<'a>(&'a mut self, entry: &'a StateEntry) -> BoxFuture<'a, Result<(), SessionError>>;
}

struct SessionCommit {
    info: SessionInfo,
    file: Option<File>,
}

impl Session {
    pub fn memory() -> Self {
        Self {
            info: SessionInfo {
                path: PathBuf::new(),
                title: "Untitled session".into(),
                created_at: now_secs(),
            },
            state: State::default(),
            file: None,
        }
    }

    pub async fn create(directory: impl AsRef<Path>) -> Result<Self, SessionError> {
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| SessionError::Invalid(error.to_string()))?;
        let path = directory.join(format!(
            "{}-{:09}-{}.jsonl",
            now.as_secs(),
            now.subsec_nanos(),
            std::process::id()
        ));
        let info = SessionInfo {
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
            },
        )
        .await?;
        Ok(Self {
            info,
            state: State::default(),
            file: Some(file),
        })
    }

    pub async fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();
        let (info, entries, append_newline) = read_session(&path)?;
        if append_newline {
            StdOpenOptions::new()
                .append(true)
                .open(&path)?
                .write_all(b"\n")?;
        }
        let state = State::restore(entries).map_err(SessionError::Invalid)?;
        let file = OpenOptions::new().append(true).open(&path).await?;
        let mut session = Self {
            info,
            state,
            file: Some(file),
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
        let directory = directory.as_ref();
        let entries = match fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error.into()),
        };
        let mut sessions = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "jsonl"))
            .filter_map(|entry| read_session_info(&entry.path()).ok())
            .collect::<Vec<_>>();
        sessions.sort_by_key(|session| std::cmp::Reverse(session.created_at));
        Ok(sessions)
    }

    pub fn delete(session: &SessionInfo) -> Result<(), SessionError> {
        fs::remove_file(&session.path)?;
        Ok(())
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn into_parts(self) -> (State, Box<dyn StateCommit>) {
        (
            self.state,
            Box::new(SessionCommit {
                info: self.info,
                file: self.file,
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

impl StateCommit for SessionCommit {
    fn info(&self) -> &SessionInfo {
        &self.info
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
}

async fn write_record(file: &mut File, record: &Record) -> Result<(), SessionError> {
    let mut encoded = serde_json::to_vec(record)?;
    encoded.push(b'\n');
    file.write_all(&encoded).await?;
    file.flush().await?;
    Ok(())
}

fn read_session(path: &Path) -> Result<(SessionInfo, Vec<StateEntry>, bool), SessionError> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() {
        return Err(SessionError::Invalid("empty file".into()));
    }
    let mut title = None;
    let mut created_at = None;
    let mut entries = Vec::new();
    let mut valid_end = 0;

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
                StdOpenOptions::new()
                    .write(true)
                    .open(path)?
                    .set_len(valid_end as u64)?;
                break;
            }
            Err(error) => return Err(SessionError::Json(error)),
        };
        match record {
            Record::Session {
                title: value,
                created_at: value_created_at,
            } if title.is_none() => {
                title = Some(value);
                created_at = Some(value_created_at);
            }
            Record::Session { .. } => {
                return Err(SessionError::Invalid("duplicate session header".into()));
            }
            Record::Title { title: value } => title = Some(value),
            Record::Entry { entry } => entries.push(entry),
        }
        valid_end += chunk.len();
    }

    let title = title.ok_or_else(|| SessionError::Invalid("missing session header".into()))?;
    let created_at =
        created_at.ok_or_else(|| SessionError::Invalid("missing creation time".into()))?;
    Ok((
        SessionInfo {
            path: path.to_path_buf(),
            title,
            created_at,
        },
        entries,
        !bytes.ends_with(b"\n") && valid_end == bytes.len(),
    ))
}

#[derive(Deserialize)]
struct HeaderRecord {
    record: String,
    title: Option<String>,
    created_at: Option<u64>,
}

fn read_session_info(path: &Path) -> Result<SessionInfo, SessionError> {
    let file = StdOpenOptions::new().read(true).open(path)?;
    let mut title = None;
    let mut created_at = None;
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
            }
            "title" => {
                if let Some(value) = record.title {
                    title = Some(value);
                }
            }
            "entry" => {}
            _ => {}
        }
    }
    Ok(SessionInfo {
        path: path.to_path_buf(),
        title: title.ok_or_else(|| SessionError::Invalid("missing session header".into()))?,
        created_at: created_at
            .ok_or_else(|| SessionError::Invalid("missing creation time".into()))?,
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
        let session = Session::open(&path).await.unwrap();
        assert_eq!(session.info().title, "hello native session");
        assert_eq!(session.state().entries().len(), 1);
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
