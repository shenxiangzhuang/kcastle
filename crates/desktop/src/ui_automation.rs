use std::fmt::Write as _;
use std::path::Path;

/// Stable identifiers exported to platform accessibility clients.
///
/// These identifiers are an automation contract. Keep them independent of visible copy and GPUI
/// element IDs so tests remain stable when labels or layout change.
pub(crate) mod ids {
    pub(crate) const APP_MAIN: &str = "kcastle.main";
    pub(crate) const SIDEBAR: &str = "kcastle.sidebar";
    pub(crate) const SIDEBAR_TOGGLE: &str = "kcastle.sidebar.toggle";
    pub(crate) const NEW_SESSION: &str = "kcastle.session.new";
    pub(crate) const SESSION_SEARCH_TOGGLE: &str = "kcastle.session.search.toggle";
    pub(crate) const SESSION_SEARCH_INPUT: &str = "kcastle.session.search.input";
    pub(crate) const WORKSPACE_LIST: &str = "kcastle.workspace.list";
    pub(crate) const WORKSPACE_ADD: &str = "kcastle.workspace.add";
    pub(crate) const SETTINGS_OPEN: &str = "kcastle.settings.open";

    pub(crate) const CONVERSATION_TABS: &str = "kcastle.conversation.tabs";
    pub(crate) const CHAT_TAB: &str = "kcastle.conversation.chat";
    pub(crate) const TRAJECTORY_TAB: &str = "kcastle.conversation.trajectory";
    pub(crate) const CHAT_PANEL: &str = "kcastle.chat";
    pub(crate) const TRANSCRIPT: &str = "kcastle.chat.transcript";
    pub(crate) const BACK_TO_BOTTOM: &str = "kcastle.chat.back-to-bottom";

    pub(crate) const COMPOSER: &str = "kcastle.composer";
    pub(crate) const COMPOSER_INPUT: &str = "kcastle.composer.input";
    pub(crate) const COMPOSER_COMMANDS: &str = "kcastle.composer.commands";
    pub(crate) const COMPOSER_PERMISSION: &str = "kcastle.composer.permission";
    pub(crate) const COMPOSER_MODEL: &str = "kcastle.composer.model";
    pub(crate) const COMPOSER_SEND: &str = "kcastle.composer.send";
    pub(crate) const COMPOSER_STOP: &str = "kcastle.composer.stop";
    pub(crate) const COMPOSER_MENU: &str = "kcastle.composer.menu";
    pub(crate) const COMPOSER_MENU_CLOSE: &str = "kcastle.composer.menu.close";

    pub(crate) const APPROVAL: &str = "kcastle.approval";
    pub(crate) const APPROVAL_DENY: &str = "kcastle.approval.deny";
    pub(crate) const APPROVAL_ALLOW: &str = "kcastle.approval.allow";

    pub(crate) const TRAJECTORY_PANEL: &str = "kcastle.trajectory";
    pub(crate) const TRAJECTORY_SEARCH_INPUT: &str = "kcastle.trajectory.search.input";

    pub(crate) const DIALOG: &str = "kcastle.dialog";
    pub(crate) const DIALOG_PRIMARY_INPUT: &str = "kcastle.dialog.input";
    pub(crate) const DIALOG_CLOSE: &str = "kcastle.dialog.close";

    #[cfg(test)]
    pub(crate) const CORE: &[&str] = &[
        APP_MAIN,
        SIDEBAR,
        SIDEBAR_TOGGLE,
        NEW_SESSION,
        SESSION_SEARCH_TOGGLE,
        SESSION_SEARCH_INPUT,
        WORKSPACE_LIST,
        WORKSPACE_ADD,
        SETTINGS_OPEN,
        CONVERSATION_TABS,
        CHAT_TAB,
        TRAJECTORY_TAB,
        CHAT_PANEL,
        TRANSCRIPT,
        BACK_TO_BOTTOM,
        COMPOSER,
        COMPOSER_INPUT,
        COMPOSER_COMMANDS,
        COMPOSER_PERMISSION,
        COMPOSER_MODEL,
        COMPOSER_SEND,
        COMPOSER_STOP,
        COMPOSER_MENU,
        COMPOSER_MENU_CLOSE,
        APPROVAL,
        APPROVAL_DENY,
        APPROVAL_ALLOW,
        TRAJECTORY_PANEL,
        TRAJECTORY_SEARCH_INPUT,
        DIALOG,
        DIALOG_PRIMARY_INPUT,
        DIALOG_CLOSE,
    ];
}

pub(crate) fn workspace(project_id: &str) -> String {
    dynamic_id("workspace", project_id)
}

#[cfg(test)]
pub(crate) fn workspace_action(action: &str, project_id: &str) -> String {
    dynamic_id(&format!("workspace.{action}"), project_id)
}

pub(crate) fn session(project_id: &str, path: &Path) -> String {
    let session_id = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown");
    dynamic_id(&format!("session.{}", segment(project_id)), session_id)
}

fn dynamic_id(kind: &str, value: &str) -> String {
    format!("kcastle.{kind}.{}", segment(value))
}

fn segment(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            encoded.push(char::from(byte));
        } else {
            write!(encoded, "_{byte:02X}").expect("writing to a String cannot fail");
        }
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::Path;

    use super::{ids, session, workspace, workspace_action};

    #[test]
    fn core_automation_ids_are_unique_and_namespaced() {
        let unique = ids::CORE.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), ids::CORE.len());
        assert!(ids::CORE.iter().all(|id| id.starts_with("kcastle.")));
    }

    #[test]
    fn dynamic_ids_are_stable_and_safe_for_platform_clients() {
        assert_eq!(workspace("repo/a"), "kcastle.workspace.repo_2Fa");
        assert_ne!(workspace("repo/a"), workspace("repo_a"));
        assert_eq!(
            workspace_action("new-session", "repo/a"),
            "kcastle.workspace.new-session.repo_2Fa"
        );
        assert_eq!(
            session("repo/a", Path::new("/sessions/session-42")),
            "kcastle.session.repo_2Fa.session-42"
        );
    }
}
