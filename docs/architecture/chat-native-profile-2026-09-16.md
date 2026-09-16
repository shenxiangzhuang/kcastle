# Native chat rendering profile — 2026-09-16

This run exercises the real release application, macOS window, wheel input and Metal
renderer. It supplements the headless cache benchmark; it is not an old/new build
comparison or a frame-latency benchmark.

## Environment and workload

- Apple M4 Pro, 48 GiB RAM, macOS 26.5.2 (25F84).
- Base commit `59920d80ad399e706856143a0379c64d0aaf5edf`, with the current uncommitted
  shared preparation-cache changes, before the highlighter-reuse follow-up below.
  Production cache budget: 8 MiB.
- `cargo rustc -p kcastle-desktop --release --locked --bin kcastle-desktop -- -C strip=none -C debuginfo=1`.
  Release optimization and thin LTO remain enabled; symbols are retained for sampling.
  Profiled executable SHA-256:
  `3f9ce289ff9e01af2d53ac50b9321a06d8577922d767c04d6b59bba1c60cf1a3`.
- Independent bundle `dev.kcastle.desktop.native-profile` and independent
  `KCASTLE_DATA_DIR`; the installed application and real user sessions were untouched.
- Three synthetic sessions, each with 100 user/assistant exchanges. Each assistant
  reply contains Markdown and a 400-line Rust code block. The generated journal was
  checked with both `validate_events` and `Session::open_readonly_in_project`.
- 1180 × 720 captured window. No model/network requests. This is a deliberately
  code-heavy fixture, not a representative sample of all conversations.

## Native visual checks

Completed with actual wheel input and saved screenshots:

1. Open A at the tail, scroll up 100 pages and return 100 pages. Rust keywords,
   identifiers and literals are highlighted in the captured screens; the return
   reaches the original last reply.
2. Leave A in the middle of history, switch A → B → A. B displays its own code;
   A restores the same reading position around replies 93/94, with highlighted code.
3. Open C and scroll upward 350 pages, reaching reply 73. The final captured screen
   has contiguous code content and highlighting, with the composer still usable.
4. Leave the view unchanged for almost six minutes. The captured reading position
   and highlighted code are still intact afterward.

No crash, persistent blank code block, wrong-session content, or lost reading position
was observed in these completed checks. These screenshots are taken after automation
waits for state; they **cannot rule out transient plain-to-highlighted flashes or
missed display frames**. The fixture identifiers contain line numbers; there is no
claim here that the code widget has a separate line-number gutter.

Automation initially failed intermittently with `noWindowsAvailable` when its window
was not available for input. Those failed phases are excluded from comparisons. After
the user kept the test window in the foreground, the complete A/B/A/C sequence ran.

## Process measurements

`sample PID DURATION 1 -file ...` collected native stacks at a requested 1 ms interval.
A separate monitor read process CPU time and RSS using `ps` every approximately 250 ms.
Average CPU below is the change in cumulative process CPU time divided by elapsed
time **between samples inside each action interval**. 100% represents one CPU core.
Boundary work between the UI timestamps and the nearest samples can be missed.

| Completed workload | Sampled interval | CPU time | Average CPU | RSS start → peak/end |
| --- | ---: | ---: | ---: | ---: |
| A: first upward 100 pages (run 2) | 6.995 s | 4.00 s | 57.2% | 136.97 → 156.25 MiB |
| A: same path back, 100 pages (run 2) | 5.957 s | 1.84 s | 30.9% | 156.25 → 169.45 MiB |
| C: first upward 350 pages, after A/B/A | 27.547 s | 15.92 s | 57.8% | 210.36 → 219.84 MiB |

In this one paired run, return scrolling used about 54% fewer CPU seconds than first
scrolling. This includes warmed font/glyph/layout state as well as the preparation
cache; it does not isolate the cache's contribution. Different automation pacing and
the approximately 250 ms measurement boundaries also limit the comparison. UI action
duration is **not** input-to-display latency.

Before opening a session, RSS was about 94 MiB in run 2. Across the final multi-session
run, the observed RSS peak was about 221 MiB. The native sampler separately reported
a 315.2 MiB physical-footprint peak. RSS and physical footprint are different OS
metrics; neither is the 8 MiB cache counter. They also include session data, UI state,
fonts, graphics resources, and allocator retention. The 350-page phase's approximately
9.5 MiB RSS growth is an observation, not proof of a long-term memory ceiling.

A subsequent `vmmap -summary` snapshot reported 93.4 MiB of live malloc allocations
inside 131.4 MiB of resident malloc zones (38.0 MiB fragmentation), plus 96.7 MiB in
resident `owned unmapped (graphics)` regions. These categories are not disjoint from
the aggregate OS metrics and must not be added to RSS. They illustrate why reducing
the Markdown cache alone cannot control the application's total memory footprint.

The last wheel action ended at 20:41:36.982 local time. Comparing `vmmap` at 20:44:01
and 20:47:31 (about 354 seconds after that action, beyond the five-minute TTL and
30-second sweep interval):

| Native memory metric | Before expiry | After expiry |
| --- | ---: | ---: |
| Live malloc allocations | 93.4 MiB | 88.2 MiB |
| Resident malloc-zone memory | 131.4 MiB | 131.5 MiB |
| Physical footprint | 309.0 MiB | 281.3 MiB |
| Process RSS | approximately 221 MiB | approximately 221 MiB |

This demonstrates that live allocated memory fell during inactivity while RSS stayed
essentially flat. It is consistent with expiry freeing cached data and the allocator
retaining/reclassifying pages, but native snapshots alone cannot attribute all 5.2 MiB
to `PreparationCache` or assert its exact remaining byte count. The cache's deterministic
expiry/count guarantees remain covered by the separate regression tests.

## Where CPU time still goes

The final 60-second sample includes interaction and subsequent idle time. Counts
below are samples, not milliseconds or per-frame timings; inclusive counts overlap
their children and must not be added to exclusive counts.

| Observed stack | Thread | Sample count | Interpretation |
| --- | --- | ---: | --- |
| `SyntaxHighlighter::new` | background | 881 inclusive | Highlighter construction remains repeated on cold fragments. |
| `ts_query__perform_analysis` + `ts_query_new` | background | 540 + 191 exclusive | Compiling Tree-sitter rules contributes visible cold-work cost. |
| `markdown::to_mdast` | background | 223 inclusive | Markdown parsing runs on the background preparation path. |
| `render_code_block` | main | 437 inclusive / 350 exclusive | Building code UI still takes main-thread work. |
| `ChatViewport::sync` | main | 438 inclusive / 200 exclusive | Projection/viewport synchronization also remains on the main thread. |
| `ThmtxTable::GetHhea`, `TMetricsTableContext` constructor | native font stack | 876 / 431 exclusive | Native font measurement is a prominent sampled cost. |

Source inspection confirms that `prepare_markdown` constructs `SyntaxHighlighter`
for each code block, and the installed component's `build_for_language` compiles
Tree-sitter queries. Preparation has moved off the main thread, but repeated setup
still consumes CPU before a cold fragment is ready. Investigate reuse of immutable
language queries before increasing the presentation-cache budget. That is a follow-up
candidate, not a change made or a performance gain demonstrated in this run.

## Evidence and reproduction

Local artifacts are in `target/native-chat-profile-2026-09-16/` (ignored build output):

- `run2-cold.png`, `run2-warm.png`: paired scroll screens.
- `final-B.png`, `final-A-restored.png`, `final-C-long.png`: session and long-scroll screens.
- `resources-run2.csv`, `resources-run3.csv`, `idle-resources.csv`: process samples.
- `vmmap-before-expiry.txt`, `vmmap-after-expiry.txt`: native memory category snapshots.
- `after-idle.png`, `idle-after-expiry.sample.txt`: stationary reading after expiry.
- `actions-run2.json`, `actions-final.json`: actual completed UI action timestamps.
- `cold-scroll.sample.txt`, `scroll-and-switch.sample.txt`,
  `final-interactions.sample.txt`: native stack reports. The second filename predates
  the interrupted switch attempt; only its completed scrolling phase is used above.
- `native_profile_fixture.rs`: one-off fixture generator, retained as evidence rather
  than coupled into the production tree. To regenerate, copy into the desktop crate's
  `examples/`, run `cargo run -p kcastle-desktop --release --locked --example
  native_profile_fixture -- /an/empty/isolated/root`, then remove that temporary example.
- `monitor.py`, `analyze.py`, `stacks.py`: collectors and analysis; adjust their local
  artifact root if replaying elsewhere.

This machine has Command Line Tools but no Instruments/xctrace installation. Native
CPU and memory sampling is complete; display-frame p95/p99, GPU timing and definitive
transient-flash acceptance remain unmeasured. A frame trace or high-frame-rate capture
on the same interactions is required before calling the rendering issue fully resolved.

## Follow-up: reuse source-free syntax highlighters

The next change implements `syntax.rs`, shared by `prepare_markdown` and direct
settled-code rendering. Up to four recently used language highlighters survive between
tasks; every returned object has had its document deleted, leaving compiled rules and
parser scratch state rather than session text. Parsing runs outside the pool mutex.
Large inputs do not return their parser to the pool. The existing presentation budget,
preparation worker count and publication freshness checks are unchanged.

A new regression was first run against fresh construction and failed with “rules
should survive each document”. The fixed test compares fresh and reused highlighting
across incomplete/multibyte code, empty input, themes, aliases and HTML injections,
asserting three constructions for three languages. A second test checks the idle
bound and oversized/cancelled/unknown inputs. All 390 workspace tests passed (four
ignored), along with Clippy `-D warnings`, formatting and both TLA+ commands.

The existing 8 MiB benchmark was then run in alternating before/after processes,
three times each, without concurrent builds. Each process runs the same 39-operation
first and revisit passes. The table gives the median of each run's p50/p95; these are
headless preparation-settle times, **not native frame times**.

| Pass | Before p50 / p95 | After p50 / p95 |
| --- | ---: | ---: |
| First visits | 35.314 / 36.089 ms | 15.708 / 16.623 ms |
| Revisits | 35.190 / 37.208 ms | 15.363 / 16.695 ms |

First-pass preparation starts stayed at 180. Revisit starts ranged from 156–159 before
and 158–161 after; admission depends on the bounded cache. Thus the improvement is not
from preparing fewer fragments. The presentation estimate stayed below 8 MiB, with
1,026,828 bytes after simulated idle expiry in all runs. Test-process peak RSS ranged
from 64.98–67.81 MiB before and 63.61–65.55 MiB after; that small difference is not a
claim of an application-memory improvement. Raw logs are under the local artifact
directory's `highlighter-reuse/{before,after}-{0,1,2}.log`.

The optimized native release was built (SHA-256
`593fa5d34f1b374eef349eefe6cd33620c4235f9880ddb2644571b26a6bf6900`) and launched in
the isolated bundle. The first attempt was blocked by the locked Mac; its idle trace
is excluded. After manual unlock, a fresh process completed A's 100-page upward scroll,
100-page return, and A → B → A, with highlighted content belonging to the correct
session. Initial screenshots had a clipped right edge; zooming the test window after
sampling produced `after-full-window.png`, confirming intact code and composer layout.

The optimized 45-second native sample included these interactions. The inclusive
`SyntaxHighlighter::new` count fell from 303 in the earlier 45-second scroll sample to
19; exclusive `ts_query__perform_analysis` samples fell from 193 to 10. These are
sample observations, not constructor invocation counts or a precise speedup ratio.
The optimized sample also includes B/A switching. Repeated rule compilation is no
longer a prominent sampled cost; font metrics and main-thread code/layout work remain.

| Optimized native workload | Sampled interval | CPU time | Average CPU | RSS start → peak/end |
| --- | ---: | ---: | ---: | ---: |
| A: first upward 100 pages | 9.335 s | 4.32 s | 46.3% | 141.89 → 160.39 MiB |
| A: return 100 pages | 7.009 s | 3.08 s | 43.9% | 160.38 → 161.39 MiB |

**Total native scrolling CPU did not improve in this run**: the earlier measurements
were 4.00 / 1.84 CPU seconds. Automated input pacing also differed (earlier sampled
intervals 6.995 / 5.957 seconds), so these runs do not isolate an end-to-end effect.
The demonstrated gains are faster preparation and removal of repeated compilation,
not a measured reduction in overall scrolling CPU or display-frame latency. Native
frame tracing and font/layout investigation remain necessary before claiming the
original transient-rendering problem fully solved.

Follow-up native evidence: `highlighter-reuse/after-native.sample.txt`,
`after-native.csv`, `after-actions.json`, `after-cold.png`, `after-warm.png`,
`after-B.png`, `after-A-restored.png`, and `after-full-window.png`.
