# Dependency patches

## IME candidate positioning

`gpui-base` is copied from crates.io 0.6.4 (upstream commit
`3c387ae0a3e9b14ee39fe98be2b51a882800aa16` in
https://github.com/longbridge/gpui-kit), retaining its Apache-2.0 license.
Only sources, the manifest, README and license are vendored; unused test/bench
targets are removed from the manifest.

An IME can request the newly inserted caret's bounds before the next paint.
The old shaped lines cannot resolve that offset, and upstream substitutes the
input's origin, making the candidate panel jump left and back after repaint.
The shared input handler now falls back to the last laid-out caret; if only the
range end is unavailable, it uses the resolved start. Input, Textarea and Editor
all use this handler. No composition or undo state changes.

The application regression queries actual composer geometry before repaint,
including CJK/emoji text, wrapped lines and a scrolled multi-line input:

```sh
cargo test --locked -p kcastle-desktop composer_ime_bounds_do_not_jump_before_repaint
```

For native acceptance, type a Chinese sentence, then continue entering pinyin
without committing it. Repeat on a wrapped/scrolled line and in a search field;
the candidate panel should stay by the composition rather than jump to the left
edge. Also check candidate selection and Escape cancellation. Native candidate
panel behavior still requires the affected OS/input method.

Remove this patch when an upstream release passes the regression and native check.

## XIM compound-text decoding

`xim-ctext` is a compatibility bridge: `zed-xim 0.4.0-zed` requires the 0.3 API,
but its decoder rejects GB2312 and panics when IBus commits Chinese on X11.
The bridge re-exports the API-compatible upstream `xim-ctext 0.4.1`, which supports
CJK encodings and charset switches within a commit. No decoder is copied or
maintained locally. Upstream: https://github.com/Riey/xim-rs.

The regression fixtures were generated on Ubuntu 24.04 using libX11's
`Xutf8TextListToTextProperty` with `XCompoundTextStyle`, the conversion used by
`ibus-x11`. They cover Chinese, ASCII transitions, mixed charsets, and emoji;
the old decoder fails the first fixture with `UnsupportedEncoding`.

```sh
cargo test --locked -p xim-ctext@0.3.0
```

For native acceptance, launch a rebuilt app on Ubuntu with X11 and IBus/Rime,
select Chinese candidates in the composer and search inputs, and verify that
the committed text stays intact and no new panic is logged. Also commit a
mixed phrase such as `今天是2026年9月20日`.

Remove the bridge when GPUI's XIM dependency uses the corrected decoder directly.

## RaTeX font-memory patches

These two crates are copied from crates.io RaTeX 0.1.14, upstream commit
`08cae05377938391117913ca4f278e6a3ffb6a8a` in
https://github.com/erweixin/RaTeX. Each retains the upstream MIT license.
Only Cargo manifests, sources, and licenses are vendored; the root Cargo lockfile
pins their dependencies. `[patch.crates-io]` keeps the change reproducible without
editing the Cargo registry or replacing the math renderer.

Local changes:

- `ratex-font-loader`: check actual glyph outlines in the primary Unicode face
  before requesting optional emoji/secondary fonts. Reuse the renderer's outline
  cache and TTC face index. If any candidate lacks an outline (including bitmap
  emoji), retain the original fallback plan. Explicitly required fonts remain
  required.
- `ratex-unicode-font`: without a valid `RATEX_UNICODE_FONT` override, share the
  discovered system font bytes and face index between primary and secondary roles.
  Custom primary fonts retain a separately discovered system fallback.
- `ratex-unicode-font` / `ratex-font-loader`: share `Arc<FontData>` owners, exposing
  borrowed byte slices to renderers. On macOS, canonical paths under
  `/System/Library/Fonts` are mapped only when `fstatfs` on the open descriptor
  confirms a read-only filesystem. Emoji discovery preserves the selected file
  and TTC face index instead of copying `fontdb`'s entire font buffer. All other
  paths/platforms and mapping failures retain owned snapshots. The system volume
  must remain read-only while mappings live; writable custom fonts are never mapped.
  `FontBytes` and the Unicode loader return types change locally, but `FontSet`'s
  slice interface and its `From<HashMap<FontId, Vec<u8>>>` conversion stay intact.

Run regression tests with:

```sh
cargo test --locked -p ratex-font-loader -p ratex-unicode-font
cargo run --locked -p kcastle-desktop --example math_memory -- '\text{中文}'
```

The example reports Rust heap allocations, not total process footprint. Run each
formula in a fresh process because font caches last for the process lifetime.
A second argument writes its SVG for comparison. The macOS coverage test requires
Arial Unicode; platforms without that font still test missing-glyph fallbacks.

Remove these patches when an upstream release provides equivalent behavior and
passes the rendering and cold-process memory checks. Do not remove emoji support
or silently ignore missing outlines to achieve a lower memory number.

## Measured check (macOS 26.5.2)

Fresh-process retained Rust heap after dropping the display list and SVG:

| Formula | Upstream 0.1.14 | Patched |
| --- | ---: | ---: |
| `x` | 0.32 MiB | 0.32 MiB |
| `\text{中文}` / `\text{⌘}` | 227.92 MiB | 22.49 MiB |
| `\text{😀}` | 227.91 MiB | 205.71 MiB |

Before the mapping patch, nine SVGs (ASCII, fraction, Greek letters, Chinese, command symbol, emoji,
mixed Chinese/emoji, summation, and outlined smiley) matched upstream byte for
byte. These numbers depend on installed fonts and are not a total-app memory
budget. See the desktop architecture for the subsequent mapping ownership policy;
native App acceptance measurements are recorded separately in `lesson/`.
