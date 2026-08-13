use std::fs::{self, OpenOptions as StdOpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use async_openai::types::responses::{InputItem, ResponseUsage};
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
        Ok(Self {
            info,
            state,
            file: Some(file),
        })
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
            .filter_map(|entry| read_session(&entry.path()).ok().map(|value| value.0))
            .collect::<Vec<_>>();
        sessions.sort_by_key(|session| std::cmp::Reverse(session.created_at));
        Ok(sessions)
    }

    pub fn info(&self) -> &SessionInfo {
        &self.info
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub async fn set_initial_title(&mut self, message: &str) -> Result<(), SessionError> {
        if self.info.title != "Untitled session" {
            return Ok(());
        }
        let title = message.split_whitespace().collect::<Vec<_>>().join(" ");
        let title = title.chars().take(80).collect::<String>();
        if title.is_empty() {
            return Ok(());
        }
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

    pub async fn append_compaction(
        &mut self,
        summary: String,
        first_kept_id: u64,
        tokens_before: usize,
        response_id: String,
        model: String,
        usage: Option<ResponseUsage>,
    ) -> Result<StateEntry, SessionError> {
        let entry = self
            .state
            .append_compaction(
                summary,
                first_kept_id,
                tokens_before,
                Some(ResponseMetadata {
                    id: response_id,
                    model,
                    usage,
                }),
            )
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
            Err(_error) if valid_end + chunk.len() == bytes.len() && !chunk.ends_with(b"\n") => {
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

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;

    use async_openai::types::responses::{EasyInputMessage, InputItem};

    use super::Session;

    #[tokio::test]
    async fn round_trips_and_repairs_a_torn_tail() {
        let directory = std::env::temp_dir().join(format!(
            "kcastle-session-test-{}-{}",
            std::process::id(),
            super::now_secs()
        ));
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
}
