use gpui::Window;

pub(crate) struct NativeTitlebarController {
    #[cfg(target_os = "macos")]
    inner: Option<macos::MacTitlebarController>,
}

impl NativeTitlebarController {
    pub(crate) fn install(window: &Window) -> Self {
        Self {
            #[cfg(target_os = "macos")]
            inner: {
                #[cfg(test)]
                {
                    let _ = window;
                    None
                }
                #[cfg(not(test))]
                {
                    macos::MacTitlebarController::install(window)
                }
            },
        }
    }

    pub(crate) fn sync(&self, window: &Window) {
        #[cfg(target_os = "macos")]
        if let Some(inner) = &self.inner {
            inner.sync(window.is_fullscreen());
        }

        #[cfg(not(target_os = "macos"))]
        let _ = window;
    }
}

#[cfg(target_os = "macos")]
#[cfg_attr(test, allow(dead_code))]
mod macos {
    use std::ptr::NonNull;
    use std::sync::{Arc, Mutex, MutexGuard};

    use block2::RcBlock;
    use gpui::Window;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2_app_kit::{
        NSButton, NSView, NSWindow, NSWindowButton, NSWindowDidEnterFullScreenNotification,
        NSWindowDidExitFullScreenNotification, NSWindowWillEnterFullScreenNotification,
    };
    use objc2_foundation::{
        NSNotification, NSNotificationCenter, NSObjectProtocol, NSPoint, NSRect,
    };
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};

    const TRAFFIC_LIGHT_X: f64 = 16.0;
    const TRAFFIC_LIGHT_Y: f64 = 13.0;

    #[derive(Clone, Copy)]
    struct TrafficLightFrames {
        titlebar: NSRect,
        close: NSRect,
        minimize: NSRect,
        zoom: NSRect,
    }

    #[derive(Default)]
    struct TrafficLightState {
        original_frames: Option<TrafficLightFrames>,
    }

    struct TrafficLightButtons {
        close: Retained<NSButton>,
        minimize: Retained<NSButton>,
        zoom: Retained<NSButton>,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TrafficLightLayout {
        container_height: f64,
        titlebar_origin_y: f64,
        minimize_x: f64,
        zoom_x: f64,
    }

    pub(super) struct MacTitlebarController {
        center: Retained<NSNotificationCenter>,
        observers: Vec<Retained<ProtocolObject<dyn NSObjectProtocol>>>,
        state: Arc<Mutex<TrafficLightState>>,
        window: Retained<NSWindow>,
    }

    impl MacTitlebarController {
        pub(super) fn install(window: &Window) -> Option<Self> {
            let native_window = native_window(window)?;
            let state = Arc::new(Mutex::new(TrafficLightState::default()));
            let center = NSNotificationCenter::defaultCenter();
            let window_address = Retained::as_ptr(&native_window) as usize;
            let mut observers = Vec::with_capacity(3);
            // SAFETY: These AppKit notification names are immutable process-wide constants.
            let will_enter = unsafe { NSWindowWillEnterFullScreenNotification };
            // SAFETY: These AppKit notification names are immutable process-wide constants.
            let did_enter = unsafe { NSWindowDidEnterFullScreenNotification };
            // SAFETY: These AppKit notification names are immutable process-wide constants.
            let did_exit = unsafe { NSWindowDidExitFullScreenNotification };

            observers.push(observe(
                &center,
                &native_window,
                will_enter,
                window_address,
                Arc::clone(&state),
                FullscreenAction::Restore,
            ));
            observers.push(observe(
                &center,
                &native_window,
                did_enter,
                window_address,
                Arc::clone(&state),
                FullscreenAction::Restore,
            ));
            observers.push(observe(
                &center,
                &native_window,
                did_exit,
                window_address,
                Arc::clone(&state),
                FullscreenAction::ApplyWindowed,
            ));

            apply_windowed_titlebar(&native_window, &state, false);
            Some(Self {
                center,
                observers,
                state,
                window: native_window,
            })
        }

        pub(super) fn sync(&self, fullscreen: bool) {
            if fullscreen {
                restore_native_titlebar(&self.window, &self.state);
            } else {
                apply_windowed_titlebar(&self.window, &self.state, false);
            }
        }
    }

    impl Drop for MacTitlebarController {
        fn drop(&mut self) {
            for observer in &self.observers {
                // SAFETY: Each token was returned by this notification center and remains retained.
                let observer: &ProtocolObject<dyn NSObjectProtocol> = observer;
                let observer: &AnyObject = AsRef::<AnyObject>::as_ref(observer);
                unsafe { self.center.removeObserver(observer) };
            }
        }
    }

    #[derive(Clone, Copy)]
    enum FullscreenAction {
        Restore,
        ApplyWindowed,
    }

    fn observe(
        center: &NSNotificationCenter,
        window: &NSWindow,
        name: &objc2_foundation::NSNotificationName,
        window_address: usize,
        state: Arc<Mutex<TrafficLightState>>,
        action: FullscreenAction,
    ) -> Retained<ProtocolObject<dyn NSObjectProtocol>> {
        let block = RcBlock::new(move |_notification: NonNull<NSNotification>| {
            // SAFETY: The controller retains this NSWindow until all observers are removed.
            let window = unsafe { &*(window_address as *const NSWindow) };
            match action {
                FullscreenAction::Restore => {
                    window.setTitlebarAppearsTransparent(false);
                    restore_native_titlebar(window, &state);
                }
                FullscreenAction::ApplyWindowed => {
                    window.setTitlebarAppearsTransparent(true);
                    apply_windowed_titlebar(window, &state, true);
                }
            }
        });

        // SAFETY: The filter object and block remain valid for the observer registration. The
        // callback captures only a retained-window address plus synchronized value state.
        unsafe {
            center.addObserverForName_object_queue_usingBlock(
                Some(name),
                Some(window),
                None,
                &block,
            )
        }
    }

    fn native_window(window: &Window) -> Option<Retained<NSWindow>> {
        let handle = HasWindowHandle::window_handle(window).ok()?;
        let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
            return None;
        };
        // SAFETY: raw-window-handle documents AppKit's pointer as an NSView. GPUI owns the view
        // for at least as long as this controller, and `window()` returns a retained NSWindow.
        let view = unsafe { &*handle.ns_view.as_ptr().cast::<NSView>() };
        view.window()
    }

    fn traffic_light_buttons(window: &NSWindow) -> Option<TrafficLightButtons> {
        Some(TrafficLightButtons {
            close: window.standardWindowButton(NSWindowButton::CloseButton)?,
            minimize: window.standardWindowButton(NSWindowButton::MiniaturizeButton)?,
            zoom: window.standardWindowButton(NSWindowButton::ZoomButton)?,
        })
    }

    fn titlebar_container(close_button: &NSButton) -> Option<Retained<NSView>> {
        // SAFETY: The button is one of AppKit's standard window controls. Its two ancestor views
        // form AppKit's traffic-light group and titlebar container.
        unsafe { close_button.superview()?.superview() }
    }

    fn capture_native_frames(window: &NSWindow) -> Option<TrafficLightFrames> {
        let buttons = traffic_light_buttons(window)?;
        let titlebar = titlebar_container(&buttons.close)?;
        Some(TrafficLightFrames {
            titlebar: titlebar.frame(),
            close: buttons.close.frame(),
            minimize: buttons.minimize.frame(),
            zoom: buttons.zoom.frame(),
        })
    }

    fn restore_native_titlebar(window: &NSWindow, state: &Mutex<TrafficLightState>) {
        let Some(frames) = lock_state(state).original_frames else {
            return;
        };
        let Some(buttons) = traffic_light_buttons(window) else {
            return;
        };
        let Some(titlebar) = titlebar_container(&buttons.close) else {
            return;
        };

        buttons.close.setFrame(frames.close);
        buttons.minimize.setFrame(frames.minimize);
        buttons.zoom.setFrame(frames.zoom);
        titlebar.setFrame(frames.titlebar);
        update_tracking_areas(&titlebar, &buttons);
    }

    fn apply_windowed_titlebar(
        window: &NSWindow,
        state: &Mutex<TrafficLightState>,
        recapture: bool,
    ) {
        let Some(buttons) = traffic_light_buttons(window) else {
            return;
        };
        let Some(titlebar) = titlebar_container(&buttons.close) else {
            return;
        };
        {
            let mut state = lock_state(state);
            if recapture {
                state.original_frames = None;
            }
            if state.original_frames.is_none() {
                state.original_frames = capture_native_frames(window);
            }
        }

        let close_frame = buttons.close.frame();
        let minimize_frame = buttons.minimize.frame();
        let button_padding =
            minimize_frame.origin.x - close_frame.origin.x - close_frame.size.width;
        let layout = traffic_light_layout(
            window.frame().size.height,
            close_frame.size.width,
            close_frame.size.height,
            button_padding,
        );
        let mut titlebar_frame = titlebar.frame();
        titlebar_frame.size.height = layout.container_height;
        titlebar_frame.origin.y = layout.titlebar_origin_y;

        titlebar.setFrame(titlebar_frame);
        buttons
            .close
            .setFrameOrigin(NSPoint::new(TRAFFIC_LIGHT_X, TRAFFIC_LIGHT_Y));
        buttons
            .minimize
            .setFrameOrigin(NSPoint::new(layout.minimize_x, TRAFFIC_LIGHT_Y));
        buttons
            .zoom
            .setFrameOrigin(NSPoint::new(layout.zoom_x, TRAFFIC_LIGHT_Y));
        update_tracking_areas(&titlebar, &buttons);
    }

    fn traffic_light_layout(
        window_height: f64,
        button_width: f64,
        button_height: f64,
        button_padding: f64,
    ) -> TrafficLightLayout {
        let container_height = button_height + TRAFFIC_LIGHT_Y * 2.0;
        let minimize_x = TRAFFIC_LIGHT_X + button_width + button_padding;
        TrafficLightLayout {
            container_height,
            titlebar_origin_y: window_height - container_height,
            minimize_x,
            zoom_x: minimize_x + button_width + button_padding,
        }
    }

    fn update_tracking_areas(titlebar: &NSView, buttons: &TrafficLightButtons) {
        titlebar.updateTrackingAreas();
        buttons.close.updateTrackingAreas();
        buttons.minimize.updateTrackingAreas();
        buttons.zoom.updateTrackingAreas();
    }

    fn lock_state(state: &Mutex<TrafficLightState>) -> MutexGuard<'_, TrafficLightState> {
        state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn traffic_lights_share_the_fixed_forty_point_titlebar_axis() {
            let layout = traffic_light_layout(720.0, 14.0, 14.0, 10.0);
            assert_eq!(layout.container_height, 40.0);
            assert_eq!(layout.titlebar_origin_y, 680.0);
            assert_eq!(layout.minimize_x, 40.0);
            assert_eq!(layout.zoom_x, 64.0);
        }
    }
}
