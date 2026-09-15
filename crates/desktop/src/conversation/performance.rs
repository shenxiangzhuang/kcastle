//! Headless GPUI presentation tests: no journal I/O, network, or private session fixtures.
use std::{
    path::PathBuf,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

use gpui_kit::{Context, Entity, TestAppContext, VisualTestContext, px, size};

use crate::{
    app::DesktopApp,
    domain::{Message, MessageId, Role, SessionView},
};

const RICH_TEXT: &str = "A **formatted** paragraph with $x^2$.\n\n\
```haskell\nquicksort [] = []\nquicksort (x:xs) = quicksort [a | a <- xs, a < x] ++ [x] ++ quicksort [a | a <- xs, a >= x]\n```\n\n\
| Step | Cost |\n| --- | --- |\n| Attention | $O(t^2 d)$ |\n| Total | $\\sum_t O(t^2) = O(T^3)$ |\n";

fn fixture(first_id: u64, count: usize, text: &str) -> Arc<SessionView> {
    let mut snapshot = SessionView::default();
    snapshot.conversation.messages = (0..count)
        .map(|id| {
            Arc::new(Message {
                key: MessageId(first_id + id as u64),
                revision: 0,
                role: Role::Assistant,
                text: format!("Message {id}\n\n{text}"),
                tool_call_id: None,
                title: None,
                payload: None,
                schema: None,
                pending: false,
                failed: false,
                started_at_ms: None,
                duration_ms: None,
                turn: 0,
                step: 0,
                request_id: None,
            })
        })
        .collect();
    Arc::new(snapshot)
}

fn setup(cx: &mut TestAppContext) -> (PathBuf, Entity<DesktopApp>, &mut VisualTestContext) {
    let root = std::env::temp_dir().join(format!(
        "kcastle-chat-perf-{}",
        kcastle_agent::SessionId::new()
    ));
    let (startup, _) = crate::desktop_startup(root.clone()).unwrap();
    cx.update(crate::init_ui);
    let (view, cx) = cx.add_window_view(|window, cx| {
        let app = DesktopApp::new(startup, window, cx);
        window.blur(cx);
        app
    });
    cx.simulate_resize(size(px(1180.0), px(720.0)));
    cx.run_until_parked();
    (root, view, cx)
}

// Exercise the publication/rendering boundary after session loading and projection.
fn publish(
    app: &mut DesktopApp,
    snapshot: &Arc<SessionView>,
    namespace: &str,
    cx: &mut Context<DesktopApp>,
) {
    app.core.session_view = snapshot.clone();
    app.core.transient_messages.clear();
    app.message_presentations.get_mut().activate(namespace);
    let chat = app.chat.get_mut();
    chat.activate(namespace.to_owned());
    chat.pending_anchor = Some(crate::layout::ScrollAnchor::Tail);
    chat.list.set_follow_mode(gpui_kit::FollowMode::Tail);
    cx.notify();
}

#[gpui_kit::test]
fn chat_switch_and_scroll_do_not_wait_for_markdown(cx: &mut TestAppContext) {
    let (root, view, cx) = setup(cx);
    let first = fixture(10000, 1000, RICH_TEXT);
    let second = fixture(20000, 1000, RICH_TEXT);
    let (resume, gate) = tokio::sync::oneshot::channel();
    view.update(cx, |app, cx| {
        app.chat.get_mut().worker_gate = Some(gate);
        publish(app, &first, "first", cx);
    });
    cx.run_until_parked();
    assert!(
        cx.debug_bounds("chat-plain:10999").is_some(),
        "the first frame must contain readable source while preparation is suspended"
    );
    view.read_with(cx, |app, _| {
        let chat = app.chat.borrow();
        assert_eq!(chat.worker_starts.load(Ordering::Relaxed), 1);
        assert_eq!(chat.prepared_chunks(), 0);
    });

    // Repeated switches must publish new text without waiting or starting extra workers.
    for (snapshot, namespace, selector) in [
        (&second, "second", "chat-plain:20999"),
        (&first, "first", "chat-plain:10999"),
        (&second, "second", "chat-plain:20999"),
    ] {
        view.update(cx, |app, cx| publish(app, snapshot, namespace, cx));
        cx.run_until_parked();
        assert!(cx.debug_bounds(selector).is_some());
        view.read_with(cx, |app, _| {
            assert!(Arc::ptr_eq(&app.core.session_view, snapshot));
            assert_eq!(app.chat.borrow().worker_starts.load(Ordering::Relaxed), 1);
            assert_eq!(app.chat.borrow().prepared_chunks(), 0);
        });
    }
    let viewport = view.read_with(cx, |app, _| app.chat.borrow().list.viewport_bounds());
    let before = view.read_with(cx, |app, _| app.chat.borrow().anchor());
    cx.simulate_event(gpui_kit::ScrollWheelEvent {
        position: viewport.center(),
        delta: gpui_kit::ScrollDelta::Pixels(gpui_kit::point(px(0.0), px(600.0))),
        ..Default::default()
    });
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        assert_ne!(
            app.chat.borrow().anchor(),
            before,
            "scroll input must still move the viewport"
        );
        assert_eq!(app.chat.borrow().worker_starts.load(Ordering::Relaxed), 1);
    });

    resume.send(()).unwrap();
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        let chat = app.chat.borrow();
        assert!(chat.prepared_chunks() > 0, "the current viewport must eventually render Markdown");
        assert!(chat.worker_starts.load(Ordering::Relaxed) < 30, "count all preparation starts, including evicted/cancelled work, not just retained results");
    });
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}

#[gpui_kit::test]
#[ignore = "release timing baseline; run just bench-chat"]
fn chat_presentation_benchmark(cx: &mut TestAppContext) {
    if cfg!(debug_assertions) {
        panic!("run this benchmark with --release");
    }
    let (root, view, cx) = setup(cx);
    let fixtures = [
        ("long_history", fixture(10000, 1000, RICH_TEXT)),
        (
            "huge_message",
            fixture(20000, 1, &"A **long** message paragraph.\n\n".repeat(20000)),
        ),
        ("code_and_math", fixture(30000, 1, &RICH_TEXT.repeat(40))),
    ];
    println!(
        "CHAT_BENCH os={} arch={} viewport=1180x720 samples=20 units=ms",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    for (name, snapshot) in fixtures {
        let mut first_frames = Vec::new();
        let mut settles = Vec::new();
        let bytes: usize = snapshot
            .conversation
            .messages
            .iter()
            .map(|m| m.text.len())
            .sum();
        for sample in 0..=20 {
            // Alternate namespaces to discard Chat presentations on every sample.
            let namespace = format!("{name}-{}", sample % 2);
            let starts = view.read_with(cx, |app, _| {
                app.chat.borrow().worker_starts.load(Ordering::Relaxed)
            });
            let start = Instant::now();
            // TestAppContext flushes dirty windows synchronously when update returns.
            view.update(cx, |app, cx| publish(app, &snapshot, &namespace, cx));
            let first_frame_ms = start.elapsed().as_secs_f64() * 1000.0;
            // No executor advancement above: this measures source publication + synchronous
            // headless drawing, separately from scheduler drain and progressive re-layout.
            view.read_with(cx, |app, _| {
                assert_eq!(app.chat.borrow().prepared_chunks(), 0)
            });
            let start = Instant::now();
            cx.run_until_parked();
            let settle_ms = start.elapsed().as_secs_f64() * 1000.0;
            let work = view.read_with(cx, |app, _| {
                let chat = app.chat.borrow();
                assert!(chat.prepared_chunks() > 0);
                chat.worker_starts.load(Ordering::Relaxed) - starts
            });
            assert!(
                work < 60,
                "offscreen preparation regressed: {name} started {work} workers"
            );
            println!(
                "CHAT_SAMPLE fixture={name} sample={sample} bytes={bytes} first_frame_ms={first_frame_ms:.3} settle_ms={settle_ms:.3} work={work}"
            );
            if sample > 0 {
                first_frames.push(first_frame_ms);
                settles.push(settle_ms);
            }
        }
        first_frames.sort_by(f64::total_cmp);
        settles.sort_by(f64::total_cmp);
        // Nearest-rank percentiles for the 20 warmed samples. Sample 0 is reported separately.
        println!(
            "CHAT_SUMMARY fixture={name} first_frame_p50_ms={:.3} first_frame_p95_ms={:.3} settle_p50_ms={:.3} settle_p95_ms={:.3}",
            first_frames[9], first_frames[18], settles[9], settles[18]
        );
    }
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}
