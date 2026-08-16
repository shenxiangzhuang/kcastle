use gpui::{Context, Window};

use crate::app::DesktopApp;
use crate::domain::Effect;

pub(crate) fn run_effects(
    app: &mut DesktopApp,
    effects: Vec<Effect>,
    window: &mut Window,
    cx: &mut Context<DesktopApp>,
) {
    for effect in effects {
        match effect {
            Effect::ApplyChatTail => app.request_chat_tail(),
            Effect::CreateSession { operation, input } => {
                app.create_session_for_run(operation, input, window, cx)
            }
            Effect::StartRun { run, input } => app.start_run(run, input, window, cx),
            Effect::OpenSession { operation, path } => {
                app.open_session_effect(operation, path, window, cx)
            }
            Effect::RenameSession { operation, title } => {
                app.rename_session_effect(operation, title, window, cx)
            }
        }
    }
}
