//! A native clip view keeps WebKit's layout viewport unchanged while the transcript scrolls.
use std::ptr::NonNull;

use gpui_kit::Window;
use objc2::{MainThreadMarker, MainThreadOnly, rc::Retained};
use objc2_app_kit::NSView;
use objc2_foundation::{NSPoint, NSRect, NSSize};
use objc2_quartz_core::CACornerMask;
use raw_window_handle::{
    AppKitWindowHandle, HandleError, HasWindowHandle, RawWindowHandle, WindowHandle,
};

use super::Placement;

pub(super) struct ClipView(Retained<NSView>);
impl ClipView {
    pub(super) fn release_focus(&self) {
        let Some(window) = self.0.window() else {
            return;
        };
        let Some(responder) = window.firstResponder() else {
            return;
        };
        if let Ok(view) = responder.downcast::<NSView>()
            && view.isDescendantOf(&self.0)
            // The native hierarchy is inspected only on its owning UI thread.
            && let Some(parent) = unsafe { self.0.superview() }
        {
            window.makeFirstResponder(Some(&parent));
        }
    }

    pub(super) fn new(window: &Window) -> Result<Self, String> {
        let mtm = MainThreadMarker::new().ok_or("WebKit must be created on the UI thread")?;
        let handle = HasWindowHandle::window_handle(window).map_err(|error| error.to_string())?;
        let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
            return Err("HTML preview requires an AppKit window".into());
        };
        // GPUI owns the NSView for the window's lifetime; this retained child is detached on drop.
        let parent = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
        let view = NSView::initWithFrame(NSView::alloc(mtm), NSRect::ZERO);
        view.setWantsLayer(true);
        if let Some(layer) = view.layer() {
            layer.setMasksToBounds(true);
            layer.setMaskedCorners(
                CACornerMask::LayerMinXMinYCorner
                    | CACornerMask::LayerMaxXMinYCorner
                    | CACornerMask::LayerMinXMaxYCorner
                    | CACornerMask::LayerMaxXMaxYCorner,
            );
        }
        parent.addSubview(&view);
        Ok(Self(view))
    }

    pub(super) fn place(&self, placement: Placement) {
        let clip = placement.clip;
        // All native view access stays on the main thread; the retained child owns its attachment.
        if let Some(parent) = unsafe { self.0.superview() } {
            let y = if parent.isFlipped() {
                f64::from(clip.origin.y)
            } else {
                parent.bounds().size.height - f64::from(clip.bottom())
            };
            self.0.setFrame(NSRect::new(
                NSPoint::new(f64::from(clip.origin.x), y),
                NSSize::new(f64::from(clip.size.width), f64::from(clip.size.height)),
            ));
        }
        if let Some(layer) = self.0.layer() {
            let mut corners = CACornerMask::empty();
            if clip.bottom() == placement.full.bottom() {
                corners |= CACornerMask::LayerMinXMinYCorner | CACornerMask::LayerMaxXMinYCorner;
            }
            if clip.top() == placement.full.top() {
                corners |= CACornerMask::LayerMinXMaxYCorner | CACornerMask::LayerMaxXMaxYCorner;
            }
            layer.setMaskedCorners(corners);
            layer.setCornerRadius(12.0);
        }
    }
}

pub(super) fn snapshot(view: &wry::WebView) -> super::Snapshot {
    use objc2::AnyThread;
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage};
    use objc2_foundation::{NSDictionary, NSError, NSNumber};
    use objc2_web_kit::WKSnapshotConfiguration;
    use wry::WebViewExtMacOS;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let sender = std::cell::RefCell::new(Some(sender));
    let webview = view.webview();
    let configuration = unsafe { WKSnapshotConfiguration::new(webview.mtm()) };
    let bounds = webview.bounds();
    // Bound the transient bitmap allocation while preserving the full document's aspect ratio.
    let scale = (8_000_000.0 / (bounds.size.width * bounds.size.height).max(1.0))
        .sqrt()
        .min(1.0);
    let width = (bounds.size.width * scale).min(1600.0);
    unsafe {
        configuration.setRect(bounds);
        configuration.setSnapshotWidth(Some(&NSNumber::new_f64(width)));
        configuration.setAfterScreenUpdates(true);
    }
    let callback = block2::RcBlock::new(move |image: *mut NSImage, error: *mut NSError| {
        // WebKit owns these callback objects for this invocation; encode before returning.
        let result = unsafe {
            if let Some(error) = error.as_ref() {
                Err(error.localizedDescription().to_string())
            } else {
                image
                    .as_ref()
                    .and_then(NSImage::TIFFRepresentation)
                    .and_then(|data| {
                        NSBitmapImageRep::initWithData(NSBitmapImageRep::alloc(), &data)
                    })
                    .and_then(|bitmap| {
                        bitmap.representationUsingType_properties(
                            NSBitmapImageFileType::PNG,
                            &NSDictionary::new(),
                        )
                    })
                    .map(|data| data.to_vec())
                    .ok_or_else(|| "Could not encode the preview image".to_owned())
            }
        };
        if let Some(sender) = sender.borrow_mut().take() {
            let _ = sender.send(result);
        }
    });
    unsafe {
        webview.takeSnapshotWithConfiguration_completionHandler(Some(&configuration), &callback);
    }
    receiver
}
impl HasWindowHandle for ClipView {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let handle = AppKitWindowHandle::new(NonNull::from(&*self.0).cast());
        // The handle is borrowed from our retained native view and cannot outlive it.
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::AppKit(handle)) })
    }
}
impl Drop for ClipView {
    fn drop(&mut self) {
        self.0.removeFromSuperview();
    }
}

/// AccessKit owns GPUI's content-view children. Publish native WebKit roots alongside
/// that tree at the window boundary so browser controls remain reachable to assistive tools.
pub(super) struct Accessibility {
    window: Retained<objc2_app_kit::NSWindow>,
    children: Vec<usize>,
}
impl Accessibility {
    pub(super) fn update<'a>(
        owner: &mut Option<Self>,
        previews: impl Iterator<Item = &'a super::Preview>,
    ) {
        use objc2_app_kit::{NSAccessibility, NSWindowButton};
        use objc2_foundation::NSArray;
        use wry::WebViewExtMacOS;
        let browsers = previews
            .filter(|p| p.applied.is_some())
            .filter_map(|p| p.browser.as_ref())
            .collect::<Vec<_>>();
        let Some(window) = browsers
            .first()
            .map(|browser| browser.ns_window())
            .or_else(|| owner.as_ref().map(|owner| owner.window.clone()))
        else {
            return;
        };
        // Clearing NSWindow's explicit children does not restore AccessKit's tree.
        // Keep publishing the GPUI root when every native preview is hidden.
        let mut children = window
            .contentView()
            .and_then(|view| view.accessibilityChildren())
            .map_or_else(Vec::new, |children| children.to_vec());
        for browser in &browsers {
            let view = browser.webview();
            // WKWebView implements NSAccessibility; the parent window outlives this attachment.
            unsafe {
                view.setAccessibilityParent(Some(window.as_ref()));
            }
            children.push(view.into());
        }
        for button in [
            NSWindowButton::CloseButton,
            NSWindowButton::MiniaturizeButton,
            NSWindowButton::ZoomButton,
        ] {
            if let Some(button) = window.standardWindowButton(button) {
                children.push(button.into());
            }
        }
        let identity = children
            .iter()
            .map(|child| Retained::as_ptr(child) as usize)
            .collect::<Vec<_>>();
        if owner
            .as_ref()
            .is_some_and(|owner| owner.children == identity)
        {
            return;
        }
        // Every object is a native accessibility element or AccessKit's existing root.
        unsafe {
            window.setAccessibilityChildren(Some(&NSArray::from_retained_slice(&children)));
        }
        if let Some(owner) = owner {
            owner.children = identity;
        } else {
            *owner = Some(Self {
                window,
                children: identity,
            });
        }
    }
}
impl Drop for Accessibility {
    fn drop(&mut self) {
        use objc2_app_kit::NSAccessibility;
        // Restore AppKit's default children before the preview views are released.
        unsafe {
            self.window.setAccessibilityChildren(None);
        }
    }
}

/// GPUI registers its current cursor over its entire NSView, including WebKit children.
/// Suspend those legacy cursor rectangles while WebKit owns the pointer; WebKit's
/// tracking areas then control CSS cursors. Restore them outside the visible browser.
struct CursorState {
    window: Retained<objc2_app_kit::NSWindow>,
    regions: Vec<NSRect>,
    browser_owns: bool,
}
impl CursorState {
    fn sync(&mut self) {
        let point = self.window.mouseLocationOutsideOfEventStream();
        let inside =
            self.window.isKeyWindow() && self.regions.iter().any(|rect| contains(*rect, point));
        if inside != self.browser_owns {
            self.browser_owns = inside;
            if inside {
                self.window.disableCursorRects();
                objc2_app_kit::NSCursor::arrowCursor().set();
            } else {
                self.window.enableCursorRects();
                self.window.resetCursorRects();
            }
        }
    }
}
fn contains(rect: NSRect, point: NSPoint) -> bool {
    point.x >= rect.origin.x
        && point.y >= rect.origin.y
        && point.x < rect.origin.x + rect.size.width
        && point.y < rect.origin.y + rect.size.height
}
pub(super) struct CursorOwner {
    state: std::rc::Rc<std::cell::RefCell<CursorState>>,
    monitor: Retained<objc2::runtime::AnyObject>,
}
impl CursorOwner {
    pub(super) fn update<'a>(
        owner: &mut Option<Self>,
        previews: impl Iterator<Item = &'a super::Preview>,
    ) {
        use objc2_app_kit::{NSEvent, NSEventMask};
        let clips = previews
            .filter(|preview| preview.applied.is_some())
            .filter_map(|preview| preview.browser.as_ref().map(|browser| &browser.clip.0))
            .collect::<Vec<_>>();
        if owner.is_none() {
            let Some(window) = clips.first().and_then(|view| view.window()) else {
                return;
            };
            let state = std::rc::Rc::new(std::cell::RefCell::new(CursorState {
                window,
                regions: Vec::new(),
                browser_owns: false,
            }));
            let events = state.clone();
            let handler = block2::RcBlock::new(move |event: NonNull<NSEvent>| {
                // AppKit invokes this local monitor on the owning UI thread, before dispatch.
                let native_event = unsafe { event.as_ref() };
                let mut state = events.borrow_mut();
                if native_event.window(state.window.mtm()).as_deref() == Some(&state.window) {
                    state.sync();
                }
                event.as_ptr()
            });
            let mask = NSEventMask::MouseMoved
                | NSEventMask::MouseEntered
                | NSEventMask::MouseExited
                | NSEventMask::LeftMouseDragged
                | NSEventMask::RightMouseDragged
                | NSEventMask::OtherMouseDragged;
            let Some(monitor) =
                (unsafe { NSEvent::addLocalMonitorForEventsMatchingMask_handler(mask, &handler) })
            else {
                return;
            };
            *owner = Some(Self { state, monitor });
        }
        if let Some(owner) = owner {
            let mut state = owner.state.borrow_mut();
            state.regions = clips
                .iter()
                .map(|view| view.convertRect_toView(view.bounds(), None))
                .collect();
            state.sync();
        }
    }
}
impl Drop for CursorOwner {
    fn drop(&mut self) {
        // Remove the monitor before releasing its state/window; every disable has a matching enable.
        unsafe {
            objc2_app_kit::NSEvent::removeMonitor(&self.monitor);
        }
        let state = self.state.borrow();
        if state.browser_owns {
            state.window.enableCursorRects();
        }
    }
}
#[cfg(test)]
mod cursor_tests {
    use super::*;
    #[test]
    fn native_cursor_ownership_stops_at_the_visible_clip() {
        let clip = NSRect::new(NSPoint::new(20.0, 80.0), NSSize::new(400.0, 200.0));
        assert!(contains(clip, NSPoint::new(30.0, 90.0)));
        for point in [
            NSPoint::new(30.0, 79.0),
            NSPoint::new(420.0, 90.0),
            NSPoint::new(30.0, 280.0),
        ] {
            assert!(!contains(clip, point));
        }
        assert!(!contains(NSRect::ZERO, NSPoint::new(0.0, 0.0)));
    }
}
