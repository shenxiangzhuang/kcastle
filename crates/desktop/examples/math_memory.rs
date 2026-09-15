//! Cold-process RaTeX heap measurement (excludes native/GPU allocations).
//! cargo run -p kcastle-desktop --example math_memory -- '\text{中文}'
//! An optional second argument writes the SVG for before/after comparison.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
struct Meter;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn add(n: usize) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            add(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Relaxed);
        unsafe { System.dealloc(p, l) };
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Relaxed);
            add(n);
        }
        q
    }
}
#[global_allocator]
static ALLOCATOR: Meter = Meter;
fn main() {
    let expr = std::env::args().nth(1).unwrap_or("x".into());
    let output = std::env::args().nth(2);
    let before = LIVE.load(Relaxed);
    let ast = ratex_parser::parse(&expr).unwrap();
    let layout = ratex_layout::layout(&ast, &Default::default());
    let list = ratex_layout::to_display_list(&layout);
    let svg = ratex_svg::render_to_svg(
        &list,
        &ratex_svg::SvgOptions {
            embed_glyphs: true,
            ..Default::default()
        },
    );
    assert!(svg.contains("<svg"));
    println!(
        "expression={expr:?} svg_bytes={} live_delta_MiB={:.3} peak_MiB={:.3}",
        svg.len(),
        (LIVE.load(Relaxed) - before) as f64 / 1048576.0,
        PEAK.load(Relaxed) as f64 / 1048576.0
    );
    if let Some(path) = output {
        std::fs::write(path, &svg).unwrap();
    }
    drop((ast, layout, list, svg));
    println!(
        "after_drop_delta_MiB={:.3}",
        (LIVE.load(Relaxed) - before) as f64 / 1048576.0
    );
}
