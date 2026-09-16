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
        assert_eq!(chat.unsettled_chunks(), 0, "all demanded blocks must finish without resize/input");
        assert!(chat.worker_starts.load(Ordering::Relaxed) < 30, "count all preparation starts, including evicted/cancelled work, not just retained results");
    });
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}

#[gpui_kit::test]
fn long_code_rows_share_one_preparation(cx: &mut TestAppContext) {
    let (root, view, cx) = setup(cx);
    let source = format!(
        "```rust\n/* open\n{}close */\nlet value = 42;\n```",
        "comment\n".repeat(100)
    );
    let snapshot = fixture(50000, 1, &source);
    view.update(cx, |app, cx| publish(app, &snapshot, "code", cx));
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        let chat = app.chat.borrow();
        assert!(
            chat.prepared_chunks() > 1,
            "multiple visible code slices must be rich"
        );
        assert!(
            chat.worker_starts.load(Ordering::Relaxed) == 2,
            "one index and one shared code parse for the tail viewport"
        );
    });
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}

#[gpui_kit::test]
fn revisiting_a_session_reuses_code_preparation(cx: &mut TestAppContext) {
    let (root, view, cx) = setup(cx);
    let first = fixture(
        60000,
        1,
        &format!("```rust\n{}\n```", "let value = 42;\n".repeat(100)),
    );
    let second = fixture(70000, 1, "A different **session**.");
    for (snapshot, namespace) in [(&first, "first"), (&second, "second")] {
        view.update(cx, |app, cx| publish(app, snapshot, namespace, cx));
        cx.run_until_parked();
    }
    let starts = view.read_with(cx, |app, _| {
        app.chat.borrow().worker_starts.load(Ordering::Relaxed)
    });
    view.update(cx, |app, cx| publish(app, &first, "first", cx));
    view.read_with(cx, |app, _| {
        assert!(
            app.chat.borrow().prepared_chunks() > 0,
            "a warm session must show prepared code in the first frame, before advancing workers"
        )
    });
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        assert!(app.chat.borrow().prepared_chunks() > 0);
        assert_eq!(
            app.chat.borrow().worker_starts.load(Ordering::Relaxed),
            starts,
            "returning to a recent session must reuse its index and highlighted code"
        );
    });
    view.update(cx, |app, cx| publish(app, &second, "second", cx));
    cx.run_until_parked();
    let bytes = view.read_with(cx, |app, _| app.chat.borrow().cache_bytes());
    view.read_with(cx, |_, cx| cx.background_executor().clone())
        .advance_clock(std::time::Duration::from_secs(6 * 60));
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        assert!(
            app.chat.borrow().cache_bytes() < bytes,
            "idle entries expire without input"
        );
        assert!(
            app.chat.borrow().prepared_chunks() > 0,
            "idle reading keeps visible text prepared"
        );
        assert_eq!(
            app.chat.borrow().worker_starts.load(Ordering::Relaxed),
            starts
        );
    });
    view.update(cx, |app, cx| publish(app, &first, "first", cx));
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        assert!(app.chat.borrow().worker_starts.load(Ordering::Relaxed) > starts)
    });
    let starts = view.read_with(cx, |app, _| {
        app.chat.borrow().worker_starts.load(Ordering::Relaxed)
    });
    view.update(cx, |app, cx| publish(app, &second, "second", cx));
    view.read_with(cx, |app, _| {
        assert!(app.chat.borrow().prepared_chunks() > 0)
    });
    cx.run_until_parked();
    view.read_with(cx, |app, _| {
        assert_eq!(
            app.chat.borrow().worker_starts.load(Ordering::Relaxed),
            starts,
            "a stationary reader keeps the semantic index needed for a quick round-trip"
        )
    });
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}

#[gpui_kit::test]
fn measured_neighbours_stay_prepared_across_frames(cx: &mut TestAppContext) {
    let (root, view, cx) = setup(cx);
    let snapshot = fixture(
        90000,
        50,
        "A **paragraph**.\n\nAnother paragraph.\n\n```rust\nlet value = 42;\n```",
    );
    view.update(cx, |app, cx| publish(app, &snapshot, "neighbours", cx));
    cx.run_until_parked();
    let starts = view.read_with(cx, |app, _| {
        app.chat.borrow().worker_starts.load(Ordering::Relaxed)
    });
    for _ in 0..3 {
        view.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert_eq!(app.chat.borrow().demanded_unprepared(), 0);
            assert_eq!(
                app.chat.borrow().worker_starts.load(Ordering::Relaxed),
                starts,
                "remeasured overscan must not cause repeated parsing on redraw"
            );
        });
    }
    let viewport = view.read_with(cx, |app, _| app.chat.borrow().list.viewport_bounds());
    for delta in [250.0, -250.0] {
        cx.simulate_event(gpui_kit::ScrollWheelEvent {
            position: viewport.center(),
            delta: gpui_kit::ScrollDelta::Pixels(gpui_kit::point(px(0.0), px(delta))),
            ..Default::default()
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert_eq!(app.chat.borrow().demanded_unprepared(), 0)
        });
    }
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}

#[gpui_kit::test]
fn oversized_sources_do_not_start_unbounded_preparation(cx: &mut TestAppContext) {
    let (root, view, cx) = setup(cx);
    for (lines, max_work) in [(20000, 1), (80000, 0)] {
        let snapshot = fixture(
            95000,
            1,
            &format!("```rust\n{}\n```", "let value = 42;\n".repeat(lines)),
        );
        let starts = view.read_with(cx, |app, _| {
            app.chat.borrow().worker_starts.load(Ordering::Relaxed)
        });
        view.update(cx, |app, cx| {
            publish(app, &snapshot, &format!("oversized-{lines}"), cx)
        });
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            let chat = app.chat.borrow();
            assert!(chat.worker_starts.load(Ordering::Relaxed) - starts <= max_work,
                "oversized code may build a bounded index, but must not start whole-block highlighting");
            assert!(chat.retained_chunks() < 30);
            assert!(chat.cache_bytes() <= 8 * 1024 * 1024);
            assert_eq!(chat.demanded_unprepared(), 0);
        });
        assert!(cx.debug_bounds("chat-plain:95000").is_some());
    }
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
        ("chinese_prose", fixture(40000, 1, &"中文正文与 English 混排，包含 **加粗内容** 和 `Ord + Clone`，检查段落布局、标点换行与复制。".repeat(60))),
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
            // Keep this cold-cache benchmark independent of cross-session reuse.
            view.update(cx, |app, _| {
                app.chat.get_mut().set_cache_budget(8 * 1024 * 1024)
            });
            let namespace = format!("{name}-{sample}");
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

#[gpui_kit::test]
#[ignore = "cache budget comparison; run just bench-chat-cache 8 (or 16, 32)"]
fn chat_cache_benchmark(cx: &mut TestAppContext) {
    if cfg!(debug_assertions) {
        panic!("use --release");
    }
    let mib: usize = std::env::var("KCASTLE_CHAT_CACHE_MIB")
        .unwrap_or_else(|_| "8".into())
        .parse()
        .unwrap();
    assert!([8, 16, 32].contains(&mib));
    let budget = mib * 1024 * 1024;
    let (root, view, cx) = setup(cx);
    view.update(cx, |app, _| app.chat.get_mut().set_cache_budget(budget));
    let code = format!(
        "```rust\n{}\n```",
        "let value: Option<usize> = Some(42); // highlighted code\n".repeat(400)
    );
    let sessions = (0..3)
        .map(|i| fixture(100000 + i * 1000, 100, &code))
        .collect::<Vec<_>>();
    let mut peak_bytes = 0;
    for pass in 0..2 {
        let starts = view.read_with(cx, |app, _| {
            app.chat.borrow().worker_starts.load(Ordering::Relaxed)
        });
        let mut times = Vec::new();
        for (session, snapshot) in sessions.iter().enumerate() {
            let start = Instant::now();
            view.update(cx, |app, cx| {
                publish(app, snapshot, &format!("budget-{session}"), cx)
            });
            cx.run_until_parked();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
            // Revisit the latest positions first, like returning to recently read code.
            let positions: Vec<_> = if pass == 0 {
                (0..12).collect()
            } else {
                (0..12).rev().collect()
            };
            for position in positions {
                let message = MessageId(100000 + session as u64 * 1000 + position * 3);
                let start = Instant::now();
                view.update(cx, |app, cx| {
                    let chat = app.chat.borrow();
                    let index = chat
                        .rows
                        .iter()
                        .position(|row| row.key.message == message)
                        .unwrap();
                    chat.list.set_follow_mode(gpui_kit::FollowMode::Normal);
                    chat.list.scroll_to(gpui_kit::ListOffset {
                        item_ix: index,
                        offset_in_item: px(0.0),
                    });
                    cx.notify();
                });
                cx.run_until_parked();
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                let bytes = view.read_with(cx, |app, _| app.chat.borrow().cache_bytes());
                peak_bytes = peak_bytes.max(bytes);
                assert!(bytes <= budget);
            }
        }
        times.sort_by(f64::total_cmp);
        let work = view.read_with(cx, |app, _| {
            app.chat.borrow().worker_starts.load(Ordering::Relaxed) - starts
        });
        println!(
            "CHAT_CACHE budget_mib={mib} pass={pass} operations={} settle_p50_ms={:.3} settle_p95_ms={:.3} work={work} peak_cache_bytes={peak_bytes}",
            times.len(),
            times[times.len() / 2],
            times[(times.len() * 95).div_ceil(100) - 1]
        );
    }
    view.read_with(cx, |_, cx| cx.background_executor().clone())
        .advance_clock(std::time::Duration::from_secs(6 * 60));
    cx.run_until_parked();
    println!(
        "CHAT_CACHE budget_mib={mib} after_idle_bytes={}",
        view.read_with(cx, |app, _| app.chat.borrow().cache_bytes())
    );
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}

#[gpui_kit::test]
fn expanding_reasoning_preserves_assistant_content(cx: &mut TestAppContext) {
    let (root, view, cx) = setup(cx);
    let mut snapshot = fixture(80000, 1, "First paragraph.\n\nLast paragraph $x$.");
    let mut reasoning = (*snapshot.conversation.messages[0]).clone();
    reasoning.key = MessageId(79999);
    reasoning.role = Role::Reasoning;
    reasoning.text = "Some reasoning".into();
    Arc::make_mut(&mut snapshot)
        .conversation
        .messages
        .push_front(Arc::new(reasoning));
    view.update(cx, |app, cx| publish(app, &snapshot, "expansion", cx));
    cx.run_until_parked();
    assert!(cx.debug_bounds("math:x").is_some());
    for expanded in [true, false] {
        view.update(cx, |app, cx| app.toggle_reasoning(0, cx));
        cx.run_until_parked();
        view.read_with(cx, |app, _| {
            assert_eq!(
                app.message_presentations
                    .borrow()
                    .expanded(MessageId(79999)),
                expanded
            );
        });
        assert!(
            cx.debug_bounds("math:x").is_some(),
            "toggling reasoning must not erase the assistant's final paragraph"
        );
    }
    drop(view);
    cx.update(|window, _| window.remove_window());
    cx.run_until_parked();
    std::fs::remove_dir_all(root).unwrap();
}
