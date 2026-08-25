use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_owned()];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).expect("source directory should be readable") {
            let path = entry.expect("source entry should be readable").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files
}

#[test]
fn pure_layers_do_not_depend_on_gpui() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for layer in ["domain", "layout", "application"] {
        for path in rust_files(&source.join(layer)) {
            let text = fs::read_to_string(&path).expect("source file should be readable");
            assert!(
                !text.contains("gpui::") && !text.contains("use gpui"),
                "pure layer imported GPUI: {}",
                path.display()
            );
        }
    }
}

#[test]
fn draw_phase_apis_are_confined_to_the_gpui_adapter() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let adapter = source.join("platform/gpui");
    let guard = source.join("architecture_tests.rs");
    for path in rust_files(&source) {
        if path.starts_with(&adapter) || path == guard {
            continue;
        }
        let text = fs::read_to_string(&path).expect("source file should be readable");
        for forbidden in [".on_next_frame(", ".layout_bounds("] {
            assert!(
                !text.contains(forbidden),
                "{forbidden} escaped the GPUI adapter: {}",
                path.display()
            );
        }
    }
}

#[test]
fn desktop_app_has_no_state_deref_escape_hatch() {
    let app = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("app.rs"),
    )
    .expect("app source should be readable");
    assert!(!app.contains("impl Deref for DesktopApp"));
    assert!(!app.contains("impl DerefMut for DesktopApp"));
}

#[test]
fn sidebar_rendering_does_not_list_sessions_from_disk() {
    let sidebar = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("sidebar.rs"),
    )
    .expect("sidebar source should be readable");
    assert!(
        !sidebar.contains("Session::list") && !sidebar.contains("std::fs::"),
        "sidebar rendering must consume cached session metadata"
    );
}

#[test]
fn agent_public_api_excludes_desktop_presentation_policy() {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("agent")
        .join("src")
        .join("lib.rs");
    let public_api = fs::read_to_string(source).expect("agent public API should be readable");
    for forbidden in [
        "TranscriptItem",
        "SessionSearchData",
        "SessionMachine,",
        "PlannedBatch",
        "ModelPreset",
        "PROVIDER_ID",
    ] {
        assert!(
            !public_api.contains(forbidden),
            "desktop presentation policy leaked into agent public API: {forbidden}"
        );
    }
}
