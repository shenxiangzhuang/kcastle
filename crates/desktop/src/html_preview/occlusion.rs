//! Native window regions leave the floating GPUI button visible and clickable.
use super::Placement;

#[cfg(target_os = "windows")]
pub(super) fn apply(view: &wry::WebView, placement: Placement) -> Result<(), wry::Error> {
    use windows::Win32::{
        Foundation::HWND,
        Graphics::Gdi::{
            CombineRgn, CreateRectRgn, CreateRoundRectRgn, DeleteObject, RGN_DIFF, SetWindowRgn,
        },
        UI::HiDpi::GetDpiForWindow,
    };
    use wry::WebViewExtWindows;
    // Wry gives each child WebView its own container HWND. Never shape GPUI's parent window.
    unsafe {
        let mut hwnd = HWND::default();
        view.controller()
            .ParentWindow(&mut hwnd)
            .map_err(std::io::Error::other)?;
        let scale = GetDpiForWindow(hwnd) as f32 / 96.0;
        let width = (f32::from(placement.clip.size.width) * scale).ceil() as i32;
        let height = (f32::from(placement.clip.size.height) * scale).ceil() as i32;
        let region = CreateRectRgn(0, 0, width, height);
        if region.is_invalid() {
            return Err(std::io::Error::last_os_error().into());
        }
        let result = (|| {
            if let Some(hole) = placement.occlusion {
                let origin = hole.origin - placement.clip.origin;
                let left = (f32::from(origin.x) * scale).floor() as i32;
                let top = (f32::from(origin.y) * scale).floor() as i32;
                let right = (f32::from(origin.x + hole.size.width) * scale).ceil() as i32;
                let bottom = (f32::from(origin.y + hole.size.height) * scale).ceil() as i32;
                let diameter = (right - left).min(bottom - top);
                let cutout = CreateRoundRectRgn(left, top, right, bottom, diameter, diameter);
                if cutout.is_invalid() {
                    return Err(std::io::Error::last_os_error());
                }
                let combined = CombineRgn(Some(region), Some(region), Some(cutout), RGN_DIFF);
                let _ = DeleteObject(cutout.into());
                if combined.0 == 0 {
                    return Err(std::io::Error::other("Could not shape HTML preview"));
                }
            }
            if SetWindowRgn(hwnd, Some(region), true) == 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        })();
        // Successful SetWindowRgn transfers ownership to Windows.
        if result.is_err() {
            let _ = DeleteObject(region.into());
        }
        result.map_err(Into::into)
    }
}

#[cfg(target_os = "linux")]
pub(super) fn apply(view: &wry::WebView, placement: Placement) -> Result<(), wry::Error> {
    use cairo::{RectangleInt, Region};
    use gtk::prelude::WidgetExt;
    use wry::WebViewExtUnix;
    // Wry's GTK toplevel wraps its own X11 child, not GPUI's X11 window.
    let window = view
        .webview()
        .toplevel()
        .and_then(|widget| widget.window())
        .ok_or_else(|| std::io::Error::other("Missing HTML preview window"))?;
    let region = Region::create_rectangle(&RectangleInt::new(
        0,
        0,
        f32::from(placement.clip.size.width).ceil() as i32,
        f32::from(placement.clip.size.height).ceil() as i32,
    ));
    if let Some(hole) = placement.occlusion {
        let left = f32::from(hole.left() - placement.clip.left());
        let top = f32::from(hole.top() - placement.clip.top());
        let width = f32::from(hole.size.width);
        let height = f32::from(hole.size.height);
        let radius = width.min(height) / 2.0;
        // X11 regions are integer rectangles; one scanline per pixel preserves the pill outline.
        for y in top.floor() as i32..(top + height).ceil() as i32 {
            let dy = ((y as f32 + 0.5) - (top + radius)).abs().min(radius);
            let inset = radius - (radius * radius - dy * dy).sqrt();
            let x = (left + inset).ceil() as i32;
            let right = (left + width - inset).floor() as i32;
            region
                .subtract_rectangle(&RectangleInt::new(x, y, (right - x).max(0), 1))
                .map_err(std::io::Error::other)?;
        }
    }
    region.status().map_err(std::io::Error::other)?;
    window.shape_combine_region(Some(&region), 0, 0);
    window.input_shape_combine_region(&region, 0, 0);
    Ok(())
}
