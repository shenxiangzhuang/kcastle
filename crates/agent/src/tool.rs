use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool};
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;
use tokio::time::timeout;

const OUTPUT_LIMIT: usize = 100_000;

#[derive(Debug, Clone)]
pub struct Env {
    pub cwd: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResult {
    pub output: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: false,
        }
    }

    pub fn error(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            is_error: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ShellTool;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellArgs {
    command: String,
    #[serde(default = "default_timeout")]
    timeout: f64,
}

fn default_timeout() -> f64 {
    120.0
}

impl ShellTool {
    pub fn schema(self) -> Tool {
        Tool::Function(FunctionTool {
            name: "shell".into(),
            description: Some(
                "Run a shell command in the working directory and return stdout and stderr.".into(),
            ),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run in the working directory"
                    },
                    "timeout": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 600,
                        "default": 120
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            })),
            strict: Some(false),
            defer_loading: None,
        })
    }

    pub fn handles(self, call: &FunctionToolCall) -> bool {
        call.name == "shell"
    }

    pub async fn execute(self, call: &FunctionToolCall, env: &Env) -> ToolResult {
        let args = match serde_json::from_str::<ShellArgs>(&call.arguments) {
            Ok(args)
                if !args.command.trim().is_empty()
                    && args.timeout.is_finite()
                    && (0.1..=600.0).contains(&args.timeout) =>
            {
                args
            }
            Ok(_) => return ToolResult::error("Invalid arguments: command or timeout is invalid"),
            Err(error) => return ToolResult::error(format!("Invalid arguments: {error}")),
        };

        let mut command = platform_shell(&args.command);
        command
            .current_dir(&env.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => return ToolResult::error(format!("Failed to start command: {error}")),
        };
        let duration = Duration::from_secs_f64(args.timeout);
        let output = match timeout(duration, child.wait_with_output()).await {
            Ok(Ok(output)) => output,
            Ok(Err(error)) => {
                return ToolResult::error(format!("Failed to wait for command: {error}"));
            }
            Err(_) => {
                return ToolResult::error(format!(
                    "Command timed out after {} seconds",
                    args.timeout
                ));
            }
        };

        let mut body = Vec::with_capacity(output.stdout.len() + output.stderr.len() + 80);
        body.extend_from_slice(
            format!("exit_code={}\n", output.status.code().unwrap_or(-1)).as_bytes(),
        );
        body.extend_from_slice(&output.stdout);
        if !output.stderr.is_empty() {
            if !output.stdout.ends_with(b"\n") {
                body.push(b'\n');
            }
            body.extend_from_slice(b"stderr:\n");
            body.extend_from_slice(&output.stderr);
        }
        let tail = if body.len() > OUTPUT_LIMIT {
            &body[body.len() - OUTPUT_LIMIT..]
        } else {
            &body
        };
        ToolResult::ok(String::from_utf8_lossy(tail))
    }
}

#[cfg(unix)]
fn platform_shell(command: &str) -> Command {
    let mut process = Command::new("sh");
    process.arg("-lc").arg(command);
    process
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::PathBuf;

    use async_openai::types::responses::FunctionToolCall;

    use super::{Env, ShellTool};

    #[tokio::test]
    async fn shell_returns_process_output() {
        let call = FunctionToolCall {
            arguments: r#"{"command":"printf rust-native"}"#.into(),
            call_id: "call_1".into(),
            namespace: None,
            name: "shell".into(),
            id: None,
            status: None,
        };
        let result = ShellTool
            .execute(
                &call,
                &Env {
                    cwd: PathBuf::from("."),
                },
            )
            .await;
        assert!(!result.is_error);
        assert!(result.output.contains("exit_code=0"));
        assert!(result.output.contains("rust-native"));
    }
}

#[cfg(windows)]
fn platform_shell(command: &str) -> Command {
    let mut process = Command::new("cmd");
    process.arg("/C").arg(command);
    process
}
