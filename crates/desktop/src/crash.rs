use std::backtrace::Backtrace;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::panic::{self, PanicHookInfo};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn install(data_root: &Path) {
    let log_path = data_root.join("crashes").join("panic.log");
    let previous = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if let Err(error) = append_report(&log_path, info) {
            eprintln!("failed to write panic report: {error}");
        }
        previous(info);
    }));
}

fn append_report(path: &Path, info: &PanicHookInfo<'_>) -> io::Result<()> {
    let mut file = open_log(path)?;
    let location = info
        .location()
        .map(|location| {
            format!(
                "{}:{}:{}",
                location.file(),
                location.line(),
                location.column()
            )
        })
        .unwrap_or_else(|| "unknown".into());
    let message = info.payload_as_str().unwrap_or("non-string panic payload");
    file.write_all(report(message, &location, &Backtrace::force_capture()).as_bytes())
}

fn open_log(path: &Path) -> io::Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options.open(path)?;
    restrict_log_permissions(path)?;
    Ok(file)
}

#[cfg(unix)]
fn restrict_log_permissions(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn restrict_log_permissions(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn report(message: &str, location: &str, backtrace: &Backtrace) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!(
        "\n--- kcastle panic ---\nunix_time={timestamp}\nversion={}\npid={}\nlocation={location}\nmessage={message}\nbacktrace:\n{backtrace}\n",
        env!("CARGO_PKG_VERSION"),
        std::process::id(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_report_contains_the_diagnostic_boundary() {
        let report = report("boom", "source.rs:7:9", &Backtrace::disabled());

        assert!(report.contains(&format!("version={}", env!("CARGO_PKG_VERSION"))));
        assert!(report.contains("location=source.rs:7:9"));
        assert!(report.contains("message=boom"));
    }

    #[cfg(unix)]
    #[test]
    fn panic_log_is_user_only_even_when_the_file_already_exists() {
        use std::os::unix::fs::PermissionsExt;

        let path = std::env::temp_dir().join(format!(
            "kcastle-panic-log-permissions-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::write(&path, b"old report").expect("create permissive fixture");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644))
            .expect("make fixture permissive");

        drop(open_log(&path).expect("open panic log"));

        assert_eq!(
            fs::metadata(&path)
                .expect("read panic log metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
        fs::remove_file(path).expect("remove panic log fixture");
    }
}
