use gpui::Window;

pub(crate) fn arm_next_frame(window: &mut Window, ready: tokio::sync::oneshot::Sender<()>) {
    window.on_next_frame(move |_, _| {
        let _ = ready.send(());
    });
    window.refresh();
}
