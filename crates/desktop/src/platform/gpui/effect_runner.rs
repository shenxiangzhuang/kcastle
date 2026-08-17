use gpui::{Context, Window};

use crate::app::DesktopApp;
use crate::domain::Effect;

pub(crate) fn run_effects(
    app: &mut DesktopApp,
    effects: Vec<Effect>,
    _window: &mut Window,
    _cx: &mut Context<DesktopApp>,
) {
    for effect in effects {
        match effect {
            Effect::ApplyChatTail => app.request_chat_tail(),
        }
    }
}
