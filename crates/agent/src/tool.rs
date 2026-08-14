use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use async_openai::types::responses::{FunctionTool, FunctionToolCall, Tool as ToolSchema};
use futures_util::future::BoxFuture;
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

pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn schema(&self) -> ToolSchema;
    fn requires_approval(&self) -> bool;
    fn execute<'a>(&'a self, call: &'a FunctionToolCall, env: &'a Env)
    -> BoxFuture<'a, ToolResult>;
}

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

impl AgentTool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::Function(FunctionTool {
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

    fn requires_approval(&self) -> bool {
        true
    }

    fn execute<'a>(
        &'a self,
        call: &'a FunctionToolCall,
        env: &'a Env,
    ) -> BoxFuture<'a, ToolResult> {
        Box::pin(async move {
            let args = match serde_json::from_str::<ShellArgs>(&call.arguments) {
                Ok(args)
                    if !args.command.trim().is_empty()
                        && args.timeout.is_finite()
                        && (0.1..=600.0).contains(&args.timeout) =>
                {
                    args
                }
                Ok(_) => {
                    return ToolResult::error("Invalid arguments: command or timeout is invalid");
                }
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
                Err(error) => {
                    return ToolResult::error(format!("Failed to start command: {error}"));
                }
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

            let result = format_output(
                output.status.code().unwrap_or(-1),
                &output.stdout,
                &output.stderr,
            );
            if output.status.success() {
                ToolResult::ok(result)
            } else {
                ToolResult::error(result)
            }
        })
    }
}

fn format_output(exit_code: i32, stdout: &[u8], stderr: &[u8]) -> String {
    let prefix = format!("exit_code={exit_code}\n");
    let mut output = Vec::with_capacity(stdout.len() + stderr.len() + 10);
    output.extend_from_slice(stdout);
    if !stderr.is_empty() {
        if !stdout.ends_with(b"\n") {
            output.push(b'\n');
        }
        output.extend_from_slice(b"stderr:\n");
        output.extend_from_slice(stderr);
    }
    let keep = OUTPUT_LIMIT.saturating_sub(prefix.len());
    let tail = &output[output.len().saturating_sub(keep)..];
    prefix + &String::from_utf8_lossy(tail)
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

    use super::{AgentTool, Env, OUTPUT_LIMIT, ShellTool, format_output};

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

    #[tokio::test]
    async fn shell_reports_nonzero_exit_as_error() {
        let call = FunctionToolCall {
            arguments: r#"{"command":"exit 7"}"#.into(),
            call_id: "failed".into(),
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
        assert!(result.is_error);
        assert!(result.output.starts_with("exit_code=7\n"));
    }

    #[tokio::test]
    async fn shell_timeout_stops_waiting_for_the_process() {
        let call = FunctionToolCall {
            arguments: r#"{"command":"sleep 5","timeout":0.05}"#.into(),
            call_id: "timeout".into(),
            namespace: None,
            name: "shell".into(),
            id: None,
            status: None,
        };
        let started = std::time::Instant::now();
        let result = ShellTool
            .execute(
                &call,
                &Env {
                    cwd: PathBuf::from("."),
                },
            )
            .await;
        assert!(result.is_error);
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn truncated_output_keeps_exit_code() {
        let output = format_output(7, &vec![b'x'; OUTPUT_LIMIT + 50], b"");
        assert!(output.starts_with("exit_code=7\n"));
        assert_eq!(output.len(), OUTPUT_LIMIT);
    }
}

#[cfg(windows)]
fn platform_shell(command: &str) -> Command {
    let mut process = Command::new("cmd");
    process.arg("/C").arg(command);
    process
}
