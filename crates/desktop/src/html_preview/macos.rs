//! A native clip view keeps WebKit's layout viewport unchanged while the transcript scrolls.
use std::{cell::Cell, ptr::NonNull};

use gpui_kit::{Bounds, Pixels, Window, point, px};
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send, rc::Retained};
use objc2_app_kit::NSView;
use objc2_core_graphics::CGMutablePath;
use objc2_foundation::{NSPoint, NSRect, NSSize};
use objc2_quartz_core::{CACornerMask, CAShapeLayer, kCAFillRuleEvenOdd};
use raw_window_handle::{
    AppKitWindowHandle, HandleError, HasWindowHandle, RawWindowHandle, WindowHandle,
};

use super::Placement;

define_class!(
    // NSView has no additional subclass invariants. Its geometry is accessed on the UI thread.
    #[unsafe(super = NSView)]
    #[name = "KcastleHtmlClipView"]
    #[thread_kind = MainThreadOnly]
    #[ivars = Cell<Option<Bounds<Pixels>>>]
    struct PreviewClip;

    impl PreviewClip {
        #[unsafe(method_id(hitTest:))]
        fn hit_test(&self, point: NSPoint) -> Option<Retained<NSView>> {
            let parent = unsafe { self.superview() };
            if !self.covers(self.convertPoint_fromView(point, parent.as_deref())) {
                None
            } else {
                // Preserve AppKit's normal child hit testing outside the overlay's outline.
                unsafe { msg_send![super(self), hitTest: point] }
            }
        }
    }
);

impl PreviewClip {
    fn covers(&self, local: NSPoint) -> bool {
        let position = point(
            px(local.x as f32),
            px((self.bounds().size.height - local.y) as f32),
        );
        contains(self.bounds(), local)
            && !self
                .ivars()
                .get()
                .is_some_and(|hole| super::pill_contains(hole, position))
    }

    fn covers_window_point(&self, point: NSPoint) -> bool {
        self.covers(self.convertPoint_fromView(point, None))
    }
}

pub(super) struct ClipView(Retained<PreviewClip>);
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
        let view = PreviewClip::alloc(mtm).set_ivars(Cell::new(None));
        let view: Retained<PreviewClip> =
            unsafe { msg_send![super(view), initWithFrame: NSRect::ZERO] };
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
            let hole = placement
                .occlusion
                .map(|bounds| Bounds::new(bounds.origin - clip.origin, bounds.size));
            self.0.ivars().set(hole);
            if let Some(hole) = hole {
                let path = CGMutablePath::new();
                let mask = CAShapeLayer::new();
                let rect = NSRect::new(
                    NSPoint::new(
                        f64::from(hole.left()),
                        f64::from(clip.size.height - hole.bottom()),
                    ),
                    NSSize::new(f64::from(hole.size.width), f64::from(hole.size.height)),
                );
                let radius = rect.size.width.min(rect.size.height) / 2.0;
                // Subtract only the pill; surrounding HTML retains its pixels and input.
                unsafe {
                    CGMutablePath::add_rect(Some(&path), std::ptr::null(), self.0.bounds());
                    CGMutablePath::add_rounded_rect(
                        Some(&path),
                        std::ptr::null(),
                        rect,
                        radius,
                        radius,
                    );
                    mask.setFillRule(kCAFillRuleEvenOdd);
                    mask.setPath(Some(&path));
                    layer.setMask(Some(&mask));
                }
            } else {
                unsafe {
                    layer.setMask(None);
                }
            }
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
/// The same visible regions route wheels before AppKit/WebKit gesture latching.
struct CursorState {
    input_view: Retained<NSView>,
    window: Retained<objc2_app_kit::NSWindow>,
    regions: Vec<(Retained<PreviewClip>, Retained<objc2_web_kit::WKWebView>)>,
    browser_owns: bool,
}
impl CursorState {
    fn sync(&mut self) {
        let point = self.window.mouseLocationOutsideOfEventStream();
        let inside = self.window.isKeyWindow()
            && self
                .regions
                .iter()
                .any(|(clip, _)| clip.covers_window_point(point));
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
        use objc2_app_kit::{NSEvent, NSEventMask, NSEventModifierFlags, NSEventType};
        use objc2_foundation::NSString;
        use wry::WebViewExtMacOS;
        let clips = previews
            .filter(|preview| preview.applied.is_some())
            .filter_map(|preview| preview.browser.as_ref())
            .collect::<Vec<_>>();
        if owner.is_none() {
            // This is GPUI's raw-window-handle view, where ClipView::new attached us.
            // NSWindow.contentView is only its AppKit wrapper, not an input receiver.
            let Some(input_view) = clips
                .first()
                .and_then(|browser| unsafe { browser.clip.0.superview() })
            else {
                return;
            };
            let Some(window) = input_view.window() else {
                return;
            };
            let state = std::rc::Rc::new(std::cell::RefCell::new(CursorState {
                input_view,
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
                    if native_event.r#type() == NSEventType::ScrollWheel {
                        if native_event.modifierFlags().intersects(
                            NSEventModifierFlags::Control | NSEventModifierFlags::Command,
                        ) {
                            return event.as_ptr();
                        }
                        let point = native_event.locationInWindow();
                        // Hit-test every event, including momentum. AppKit/WebKit may latch a
                        // gesture to a child that has since moved with the virtual transcript.
                        if let Some((_, view)) = state
                            .regions
                            .iter()
                            .find(|(clip, _)| clip.covers_window_point(point))
                        {
                            let local = view.convertPoint_fromView(point, None);
                            let y = if view.isFlipped() {
                                local.y
                            } else {
                                view.bounds().size.height - local.y
                            };
                            let factor = if native_event.hasPreciseScrollingDeltas() {
                                1.0
                            } else {
                                20.0
                            };
                            let dx = -native_event.scrollingDeltaX() * factor;
                            let dy = -native_event.scrollingDeltaY() * factor;
                            if dx != 0.0 || dy != 0.0 {
                                let script = NSString::from_str(&format!(
                                    "window.previewWheel({},{},{dx},{dy})",
                                    local.x, y
                                ));
                                // The trusted host alone forwards this to its opaque iframe.
                                unsafe {
                                    view.evaluateJavaScript_completionHandler(&script, None);
                                }
                            }
                            // Never deliver the same wheel to WebKit's native scrolling path.
                            return std::ptr::null_mut();
                        }
                        // Outside a preview, bypass any old WebKit gesture target as well.
                        let input_view = state.input_view.clone();
                        drop(state);
                        input_view.scrollWheel(native_event);
                        return std::ptr::null_mut();
                    } else {
                        state.sync();
                    }
                }
                event.as_ptr()
            });
            let mask = NSEventMask::MouseMoved
                | NSEventMask::MouseEntered
                | NSEventMask::MouseExited
                | NSEventMask::LeftMouseDragged
                | NSEventMask::RightMouseDragged
                | NSEventMask::OtherMouseDragged
                | NSEventMask::ScrollWheel;
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
                .map(|browser| (browser.clip.0.clone(), browser.webview().into_super()))
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
