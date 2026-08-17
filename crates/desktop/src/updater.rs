#[cfg(not(test))]
use std::time::Duration;

use gpui::Context;
#[cfg(not(test))]
use gpui::Window;
use semver::Version;
use serde::Deserialize;

use crate::app::DesktopApp;

#[cfg(not(test))]
const RELEASES_API: &str =
    "https://api.github.com/repos/shenxiangzhuang/kcastle/releases?per_page=100";
const RELEASE_PAGE_PREFIX: &str = "https://github.com/shenxiangzhuang/kcastle/releases/";

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AvailableUpdate {
    pub(crate) version: String,
    pub(crate) release_url: String,
}

#[derive(Debug, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    draft: bool,
}

#[cfg(not(test))]
async fn fetch_available_update(current_version: &str) -> Result<Option<AvailableUpdate>, String> {
    let current = Version::parse(current_version)
        .map_err(|error| format!("invalid current version {current_version}: {error}"))?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .user_agent(format!("kcastle-desktop/{current_version}"))
        .build()
        .map_err(|error| error.to_string())?;
    let releases = client
        .get(RELEASES_API)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .await
        .map_err(|error| error.to_string())?
        .error_for_status()
        .map_err(|error| error.to_string())?
        .json::<Vec<GitHubRelease>>()
        .await
        .map_err(|error| error.to_string())?;

    Ok(select_available_update(&current, releases))
}

fn select_available_update(
    current: &Version,
    releases: impl IntoIterator<Item = GitHubRelease>,
) -> Option<AvailableUpdate> {
    releases
        .into_iter()
        .filter(|release| !release.draft && release.html_url.starts_with(RELEASE_PAGE_PREFIX))
        .filter_map(|release| {
            let version = Version::parse(
                release
                    .tag_name
                    .strip_prefix('v')
                    .unwrap_or(&release.tag_name),
            )
            .ok()?;
            (version > *current).then_some((version, release.html_url))
        })
        .max_by(|(left, _), (right, _)| left.cmp(right))
        .map(|(version, release_url)| AvailableUpdate {
            version: version.to_string(),
            release_url,
        })
}

impl DesktopApp {
    #[cfg(not(test))]
    pub(crate) fn check_for_updates(&self, window: &mut Window, cx: &mut Context<Self>) {
        cx.spawn_in(window, async move |this, cx| {
            let update = fetch_available_update(env!("CARGO_PKG_VERSION")).await;
            let _ = cx.update(|_, app| {
                if let Some(this) = this.upgrade() {
                    this.update(app, |this, cx| {
                        if let Ok(update) = update {
                            this.available_update = update;
                            cx.notify();
                        }
                    });
                }
            });
        })
        .detach();
    }

    pub(crate) fn open_available_update(&self, cx: &mut Context<Self>) {
        if let Some(update) = &self.available_update {
            cx.open_url(&update.release_url);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release(tag: &str) -> GitHubRelease {
        GitHubRelease {
            tag_name: tag.to_owned(),
            html_url: format!("{RELEASE_PAGE_PREFIX}tag/{tag}"),
            draft: false,
        }
    }

    #[test]
    fn selects_highest_release_newer_than_current() {
        let current = Version::parse("0.2.0-alpha.2").unwrap();
        let update = select_available_update(
            &current,
            [
                release("v0.1.0"),
                release("v0.2.0-alpha.3"),
                release("v0.2.0-beta.1"),
            ],
        )
        .unwrap();

        assert_eq!(update.version, "0.2.0-beta.1");
        assert!(update.release_url.ends_with("/tag/v0.2.0-beta.1"));
    }

    #[test]
    fn returns_none_when_no_newer_valid_release_exists() {
        let current = Version::parse("0.2.0").unwrap();
        let mut draft = release("v0.3.0");
        draft.draft = true;
        let mut untrusted = release("v0.4.0");
        untrusted.html_url = "https://example.com/download".to_owned();

        assert_eq!(
            select_available_update(
                &current,
                [release("not-semver"), release("v0.2.0"), draft, untrusted]
            ),
            None
        );
    }
}
