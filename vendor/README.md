# RaTeX font-memory patches

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

Nine SVGs (ASCII, fraction, Greek letters, Chinese, command symbol, emoji,
mixed Chinese/emoji, summation, and outlined smiley) matched upstream byte for
byte. These numbers depend on installed fonts and are not a total-app memory
budget. An actual emoji still retains its source font; file mapping is deferred.
