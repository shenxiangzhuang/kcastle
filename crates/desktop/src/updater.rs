#[cfg(not(test))]
use std::time::Duration;

use gpui_kit::Context;
#[cfg(not(test))]
use gpui_kit::Window;
#[cfg(not(test))]
use velopack::{UpdateCheck, UpdateManager, sources::HttpSource};

use crate::app::DesktopApp;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const UPDATE_TARGET: &str = "linux-x64";
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const UPDATE_TARGET: &str = "linux-arm64";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const UPDATE_TARGET: &str = "osx-arm64";
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const UPDATE_TARGET: &str = "osx-x64";
#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const UPDATE_TARGET: &str = "win-x64";

#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    all(
        target_os = "macos",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ),
    all(target_os = "windows", target_arch = "x86_64")
)))]
compile_error!("kcastle-desktop updater does not support this target");

const ACTIVE_SESSION_NOTICE: &str = "Stop active sessions before restarting to update";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AvailableUpdate {
    pub(crate) version: String,
}

#[cfg(not(test))]
fn update_manager() -> Result<UpdateManager, String> {
    UpdateManager::new(
        HttpSource::new(update_source(env!("CARGO_PKG_VERSION"))),
        None,
        None,
    )
    .map_err(|error| error.to_string())
}

fn release_channel(version: &str) -> &str {
    version
        .split_once('-')
        .and_then(|(_, prerelease)| prerelease.split('.').next())
        .unwrap_or("stable")
}

fn update_source(version: &str) -> String {
    format!(
        "https://updates.kcastle.mathewshen.me/{}/{}",
        release_channel(version),
        UPDATE_TARGET
    )
}

#[cfg(not(test))]
fn download_available_update() -> Result<Option<AvailableUpdate>, String> {
    let manager = update_manager()?;
    if let Some(update) = manager.get_update_pending_restart() {
        return Ok(Some(AvailableUpdate {
            version: update.Version,
        }));
    }

    let update = match manager
        .check_for_updates()
        .map_err(|error| error.to_string())?
    {
        UpdateCheck::UpdateAvailable(update) => update,
        UpdateCheck::RemoteIsEmpty | UpdateCheck::NoUpdateAvailable => return Ok(None),
    };
    let version = update.TargetFullRelease.Version.clone();
    manager
        .download_updates(&update, None)
        .map_err(|error| error.to_string())?;
    Ok(Some(AvailableUpdate { version }))
}

fn restart_block_reason(has_active_sessions: bool) -> Option<&'static str> {
    has_active_sessions.then_some(ACTIVE_SESSION_NOTICE)
}

impl DesktopApp {
    #[cfg(not(test))]
    pub(crate) fn check_for_updates(&self, window: &mut Window, cx: &mut Context<Self>) {
        let executor = cx.background_executor().clone();
        cx.spawn_in(window, async move |this, cx| {
            loop {
                let result = executor.spawn(async { download_available_update() }).await;
                let ready = result.as_ref().is_ok_and(Option::is_some);
                let _ = cx.update(|_, app| {
                    if let Some(this) = this.upgrade() {
                        this.update(app, |this, cx| {
                            if let Ok(update) = result {
                                this.available_update = update;
                                cx.notify();
                            }
                        });
                    }
                });
                if ready {
                    break;
                }
                executor.timer(Duration::from_secs(60 * 60)).await;
            }
        })
        .detach();
    }

    #[cfg(not(test))]
    pub(crate) fn restart_to_update(&mut self, cx: &mut Context<Self>) {
        let has_active_sessions = self.has_active_sessions(cx);
        if let Some(message) = restart_block_reason(has_active_sessions) {
            self.notice(message);
            cx.notify();
            return;
        }

        let result = (|| {
            let manager = update_manager()?;
            let update = manager
                .get_update_pending_restart()
                .ok_or_else(|| "The downloaded update is no longer available".to_owned())?;
            manager
                .wait_exit_then_apply_updates(&update, true, true, Vec::<String>::new())
                .map_err(|error| error.to_string())
        })();
        match result {
            Ok(()) => cx.quit(),
            Err(error) => {
                self.notice(format!("Could not install update: {error}"));
                cx.notify();
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn restart_to_update(&mut self, _cx: &mut Context<Self>) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_waits_for_active_sessions() {
        assert_eq!(restart_block_reason(false), None);
        assert_eq!(restart_block_reason(true), Some(ACTIVE_SESSION_NOTICE));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_updates_follow_the_running_architecture() {
        let target = match std::env::consts::ARCH {
            "aarch64" => "osx-arm64",
            "x86_64" => "osx-x64",
            arch => panic!("unsupported macOS architecture: {arch}"),
        };
        for (version, channel) in [
            ("0.2.0-alpha.28", "alpha"),
            ("0.2.0-alpha.29", "alpha"),
            ("0.2.0-beta.1", "beta"),
            ("0.2.0", "stable"),
        ] {
            assert_eq!(
                update_source(version),
                format!("https://updates.kcastle.mathewshen.me/{channel}/{target}")
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn universal_bridge_accepts_only_newer_native_releases() {
        use std::fs;
        use velopack::{
            UpdateCheck, UpdateManager, VelopackAsset, VelopackAssetFeed,
            locator::VelopackLocatorConfig, sources::FileSource,
        };

        let root = std::env::temp_dir().join(format!("kcastle-update-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let manifest = root.join("sq.version");
        let updater = root.join("UpdateMac");
        fs::write(&updater, "").unwrap();
        for target in ["osx-arm64", "osx-x64"] {
            for (installed, feed, available, expected) in [
                ("0.2.0-alpha.27", "osx-universal", "0.2.0-alpha.28", true),
                ("0.2.0-alpha.28", target, "0.2.0-alpha.28", false),
                ("0.2.0-alpha.28", target, "0.2.0-alpha.29", true),
                ("0.2.0-alpha.28", target, "0.2.0-alpha.30", true),
                ("0.2.0-alpha.29", target, "0.2.0-alpha.30", true),
            ] {
                fs::write(
                    &manifest,
                    format!(
                        "<package><metadata><id>Kcastle</id><version>{installed}</version>\
                         <channel>alpha</channel><mainExe>kcastle</mainExe></metadata></package>"
                    ),
                )
                .unwrap();
                let source = root.join(feed);
                fs::create_dir_all(&source).unwrap();
                let asset = VelopackAsset {
                    PackageId: "Kcastle".into(),
                    Version: available.into(),
                    Type: "Full".into(),
                    FileName: format!("Kcastle-{available}-alpha-full.nupkg"),
                    ..Default::default()
                };
                fs::write(
                    source.join("releases.alpha.json"),
                    serde_json::to_vec(&VelopackAssetFeed {
                        Assets: vec![asset],
                    })
                    .unwrap(),
                )
                .unwrap();
                let manager = UpdateManager::new(
                    FileSource::new(source),
                    None,
                    Some(VelopackLocatorConfig {
                        RootAppDir: root.clone(),
                        UpdateExePath: updater.clone(),
                        PackagesDir: root.join("packages"),
                        ManifestPath: manifest.clone(),
                        CurrentBinaryDir: root.clone(),
                        IsPortable: true,
                    }),
                )
                .unwrap();
                match manager.check_for_updates().unwrap() {
                    UpdateCheck::UpdateAvailable(update) => {
                        assert!(expected, "{installed} -> {available} via {feed}");
                        assert_eq!(update.TargetFullRelease.Version, available);
                        assert!(!update.IsDowngrade);
                        assert!(update.DeltasToTarget.is_empty());
                    }
                    UpdateCheck::NoUpdateAvailable => assert!(!expected),
                    UpdateCheck::RemoteIsEmpty => panic!("missing migration release"),
                }
            }
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn update_source_uses_the_public_r2_domain() {
        assert_eq!(release_channel("0.2.0-alpha.7"), "alpha");
        assert_eq!(release_channel("0.2.0-beta.1"), "beta");
        assert_eq!(release_channel("0.2.0"), "stable");
        assert!(
            update_source("0.2.0-alpha.7")
                .starts_with("https://updates.kcastle.mathewshen.me/alpha/")
        );
    }
}
