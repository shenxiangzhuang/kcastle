---
type: bug
created: "2026-09-15"
verified: "2026-09-15"
scope: Kcastle 0.2.0-alpha.24 / RaTeX 0.1.14 字体加载与 macOS 内存诊断
---

# 为什么一条中文公式会让桌面应用多占约 228 MB？

```text
Markdown 中的公式，例如 $\text{中文}$
  → prepare_math_at_size：解析、数学排版，生成 DisplayList
  → render_prepared_math：把字形转换为自包含 SVG
  → RaTeX load_fonts_for_items：决定并加载所需字体
  → 全局字体缓存：持有完整字体文件的 Arc<Vec<u8>>
  → GPUI：显示 SVG，管理窗口和图形资源
```

这条调用链解释了一个容易误判的现象：对话很少，应用却占用 300～500 MB。消息文本很短，不代表渲染它所触发的依赖资源也很小。

本次定位到的主要问题是：RaTeX 0.1.14 遇到需要 Unicode 回退的公式字符时，会把主 Unicode、备用 Unicode 和 Emoji 字体一起加载。当前 macOS 上三份字体约占 228 MiB，并一直保留到进程退出。普通中文公式不需要这么多字体。

本文记录已完成的诊断、补丁和验证。数据来自 Apple M4 Pro、48 GB 统一内存、macOS 26.5.2；不是其他机器或所有会话的内存承诺。文中 `vmmap` 的 `M` 沿用工具显示口径；独立分配计量明确以 MiB（2²⁰ 字节）报告。

## 1. 先分清谁负责什么

Kcastle 的 [crates/desktop/src/dsh_markdown.rs](../crates/desktop/src/dsh_markdown.rs) 有两个相关步骤：

- `prepare_math_at_size` 调用 `ratex_parser::parse`、`ratex_layout::layout` 和 `to_display_list`。`DisplayList` 是待绘制的字形、路径及其位置，不是最终像素。
- `render_prepared_math` 调用 `ratex_svg::render_to_svg`，生成可以独立显示的 SVG。

后一步的关键配置如下，摘自实际调用：

```rust
let svg = render_to_svg(
    &prepared.display_list,
    &SvgOptions {
        font_size: prepared.font_size,
        padding: prepared.padding,
        stroke_width: 1.0,
        embed_glyphs: true,
        font_dir: String::new(),
    },
);
```

工作区启用了 `ratex-svg` 的 `embed-fonts` feature，`embed_glyphs: true` 让渲染器读取字体中的字形轮廓，写入 SVG 路径；彩色 Emoji 可以写成嵌入的图片。这样生成的公式不依赖查看器是否安装了相应的 KaTeX 字体。

这里有两个不同的生命周期：公式的 SVG 可以随着展示对象释放，但它所用的**字体源数据**由依赖库全局缓存持有。关闭会话或丢弃 SVG，并不会自动清空这些字体。

因此，应用层的展示缓存预算不能代替整个进程的资源预算。[docs/architecture/desktop.md](../docs/architecture/desktop.md) 中的准备数据预算本来就不覆盖原生 GPU、字体等资源。

## 2. 从总占用缩小到三块大分配

第一次检查运行进程时，实际运行的是 alpha.23：`Physical footprint` 为 357.7M，历史峰值为 482.5M。用户随后更新到 alpha.24，重新测量仍发现相同的大分配；更新后的一个稳定快照为 351.9M，峰值为 476.7M。

这一步先确认了两个事实：症状确实存在，而且不能把旧进程的内存直接当成当前工作区的结果。

### 2.1 三种测量不能混用

| 测量 | 本次用于回答什么 | 不能单独证明什么 |
| --- | --- | --- |
| `vmmap -summary PID` 的 `Physical footprint` | 系统计入该进程的总体内存负担和分类 | 哪个 Rust 对象或函数分配了内存 |
| `heap` 的大块分配列表 | 堆中是否有特别大的活跃分配 | 被标为 `non-object` 的块究竟是什么类型 |
| 自定义 Rust 全局分配器计量 | 只渲染一条公式会增加多少尚未释放的 Rust 堆请求字节 | GPU、原生库分配、分配器碎片及整个 App 的占用 |

`RSS`、虚拟地址空间大小和 `Physical footprint` 也不是同一个数字。本次前后对照使用相同的指标，没有把虚拟地址空间或所有共享库映射相加当成应用独占内存。

### 2.2 有用的假设与证据

初始候选包括会话数据、富文本缓存、语法高亮、字体和图形缓冲。没有因为“用了 GPU”就默认几百 MB 都正常，也没有因为出现 `Vec`、`HashMap` 就认定它是根因。

本地会话数据库当时有 4 个会话，事件正文总计约 2 MB，搜索文本约 70 KB。这不能排除解析对象膨胀，却说明原始文本本身不足以解释几百 MB 的增量。

接着，堆检查发现三块特别大的活跃分配：

| 堆分配大小 | 与本机文件的对应关系 |
| ---: | --- |
| 192,135,168 字节，约 183.2 MiB | `Apple Color Emoji.ttc` 文件为 192,123,488 字节 |
| 23,281,664 字节，约 22.2 MiB | `Arial Unicode.ttf` 文件为 23,278,008 字节 |
| 23,281,664 字节，约 22.2 MiB | 同样大小的第二份 Unicode 字体 |

差异与分配取整相容，但**大小相近还不是归因完成**。随后用 `leaks --traceTree` 查看引用链，大 Emoji 分配经一个小对象连接到程序全局数据区；再检查 RaTeX 的全局字体缓存和读文件代码，最后用独立公式渲染复现同样的约 228 MiB 增量，才把证据连起来。

这里使用 `leaks` 的引用追踪功能，不等于已经证明发生了泄漏。已定位的这些资源仍被全局缓存引用，问题是加载得太多、持有得太久。

## 3. 根因：optional 表示“可以加载失败”，不表示“按需再加载”

RaTeX 的 `FontLoadPlan` 包含两组 `FontId`：

```rust
pub struct FontLoadPlan {
    required: HashSet<FontId>,
    optional: HashSet<FontId>,
}
```

`FontId` 是字体角色，如 `MainRegular`、`CjkRegular`、`CjkFallback` 和 `EmojiFallback`。一个 `GlyphPath` 指出角色名与字符码，计划据此决定要加载什么。

以下是上游 0.1.14 的关键代码：

```rust
if needs_optional_unicode_fallbacks {
    optional.insert(FontId::CjkRegular);
    optional.insert(FontId::EmojiFallback);
    optional.insert(FontId::CjkFallback);
}

pub fn all(&self) -> HashSet<FontId> {
    self.required.union(&self.optional).copied().collect()
}
```

`load_fonts_for_plan` 随后遍历 `plan.all()`，加载尚未缓存的每种字体；最后才检查 `required` 是否全部存在。也就是说，`optional` 里的字体同样会被尝试加载，只是缺失时不一定导致整个计划失败。

触发条件也不是“出现任意非 ASCII 字符”。具体包括显式使用 CJK/Emoji 字体角色，或者非 ASCII 字符在选定 KaTeX 字体中没有相应度量数据。因此普通 `\alpha + \beta` 不一定触发，而 `\text{中文}`、`\text{⌘}` 在本机触发了回退计划。

完整因果链是：

```text
中文公式触发 Unicode 回退
  → 把三种字体全部放入待加载集合
  → 读取完整 Emoji 文件，约 183 MiB
  → 主、备用 Unicode 分别发现并读取同一文件，各约 22 MiB
  → 三份 Arc<Vec<u8>> 被 OnceLock 等全局缓存持有
  → 释放本条公式的 AST、DisplayList、SVG 后，字体仍存在
```

`OnceLock` 保证初始化一次，`Arc` 允许共享同一次分配；它们不会自动识别“两次读取的内容来自同一文件”。原实现中，主 Unicode 与系统备用字体拥有独立的初始化入口，默认情况下会分别调用 `discover_system_font`。

上游为什么选择这种策略，现有证据没有给出历史动机。可以确认的是它的实际语义，不能把“可能为了兼容性”写成已经证实的设计理由。

## 4. 最小修复：覆盖足够就停止扩展计划，默认字体共享一次发现

应用的 `render_to_svg` 调用和输出接口均未改变。补丁放在字体加载依赖中，让经过这个入口的调用者一起受益。

### 4.1 先检查主 Unicode 字体是否真的能画出来

修改后的计划收集需要回退检查的字符，再决定是否加入额外字体：

```rust
if !fallback_chars.is_empty() {
    optional.insert(FontId::CjkRegular);
    if !primary_unicode_covers(&fallback_chars) {
        optional.insert(FontId::EmojiFallback);
        optional.insert(FontId::CjkFallback);
    }
}
```

`fallback_chars` 使用 `katex_ttf_glyph_char` 转换，与渲染器使用相同的字符映射。覆盖检查摘自 [vendor/ratex-font-loader/src/lib.rs](../vendor/ratex-font-loader/src/lib.rs)：

```rust
fn primary_unicode_covers(chars: &[char]) -> bool {
    let Some(bytes) = ratex_unicode_font::load_unicode_font_arc() else {
        return false;
    };
    let index = ratex_unicode_font::unicode_font_face_index().unwrap_or(0);
    let Ok(font) = FontRef::try_from_slice_and_index(&bytes, index) else {
        return false;
    };
    chars.iter().all(|&ch| {
        let glyph = font.glyph_id(ch);
        glyph.0 != 0
            && outline_cache::get_or_compute_outline(FontId::CjkRegular, &font, glyph).is_some()
    })
}
```

`FontRef` 从字体字节解析出一个可查询的字体视图；TTC 是可包含多个字体面的集合，因此必须使用对应的 face index。`glyph_id == 0` 表示未找到字形。只检查非零 ID 仍不够：位图字形可能没有矢量轮廓，所以这里复用实际渲染器的轮廓缓存进行检查。

这个补丁是**保守地缩小加载计划**，不是全面改写成每个字形逐级动态加载：

- 所有候选字符都有主 Unicode 轮廓：跳过可选 Emoji 和第二 Unicode 字体。
- 任意候选没有轮廓，或者字体读取、解析失败：保留原有可选回退集合。
- 显式列入 `required` 的字体仍然必需，不会被这项优化删除。

这样 `\text{中文}` 不会为了可能遇到 Emoji 而提前付出 183 MiB；真实 Emoji 或未覆盖字符仍有原来的回退机会。代价是规划阶段增加字体和轮廓检查；轮廓缓存由后续渲染复用。本次没有单独测量这个检查的耗时，不能声称已经证明零 CPU 成本。

### 4.2 默认主、备用字体共享同一份字节

主字体初始化仍优先尝试 `RATEX_UNICODE_FONT`。没有有效覆盖配置时，原来的末尾是直接调用 `discover_system_font()`；现在改为：

```rust
SYSTEM_FALLBACK_FONT
    .get_or_init(discover_system_font)
    .clone()
```

这里克隆的是 `Option<(Arc<Vec<u8>>, u32)>`：增加引用计数并复制 face index，不会复制整个字体文件。系统备用字体仍使用同一个 `OnceLock`，无论哪个入口先调用，都共享默认发现结果。

边界见 [vendor/ratex-unicode-font/src/lib.rs](../vendor/ratex-unicode-font/src/lib.rs)：有效的自定义主字体仍单独加载，并保留系统备用字体。**这不是按规范化文件路径进行的全局去重**；若用户显式配置的主字体恰好与系统备用指向同一文件，本补丁没有承诺将它们合并。

## 5. 为什么选择这个修复，而不是其他办法？

| 方案 | 判断与边界 |
| --- | --- |
| 缩小会话或 Markdown 缓存 | 不能释放依赖内部仍由全局引用持有的字体，错过了已测出的主要增量 |
| 禁止 Emoji 或禁用自包含 SVG | 可能减少占用，但会改变渲染能力或字体依赖，不符合保持输出的约束 |
| 主字体已覆盖时跳过额外加载，共享默认字体 | 不改变调用接口，保留缺字回退；针对约 205 MiB 的已知浪费，优先实施 |
| 将字体改为只读文件映射 | 有利于真正需要的大字体，但涉及字节持有类型和生命周期；实际物理收益取决于访问页面，暂未实施 |
| 公式用完就清空全局字体缓存 | 当前持有关系不是公式私有；还会引入重复读盘和解析，需另行设计缓存生命周期，未实施 |

本次 KISS 的具体含义是：复用字形查询、轮廓缓存、`OnceLock` 和 `Arc`，在资源选择处减少工作。它不意味着删掉错误处理、假定所有汉字都存在，或牺牲 Emoji 渲染。

共享默认字体体现的也不是“代码长得一样就合并”，而是两个默认角色实际依赖**同一个系统字体发现结果**。有效的自定义主字体代表另一项输入，不能强行合并。

两个补丁通过根 [Cargo.toml](../Cargo.toml) 的 `[patch.crates-io]` 接入；[vendor/README.md](../vendor/README.md) 记录了上游版本、提交、MIT 许可证和移除条件。没有直接修改本机 Cargo registry，也没有增加一个平行的公式渲染器。代价是需要维护少量 vendored 源码，等待上游具备等价修复后再移除。

## 6. 测试为什么原先没挡住？又如何验证修复？

原依赖的测试检查 ASCII 不需要回退、非 ASCII 缺度量时包含回退字体，以及缺失可选字体的缓存行为。其中一些测试直接把“所有可选回退都被请求”作为预期。

这些测试能验证原策略被执行，却没有问：**主 Unicode 字体明明已覆盖时，还应不应该读取整个 Emoji 文件？** 一般渲染正确性测试也不会自动发现资源浪费，因为多加载字体通常仍然画得对。

### 6.1 先失败，再修复

本次先在未改逻辑的 vendored 源码上加入并运行两个回归：

- `covered_unicode_does_not_request_emoji_or_second_font`：中文和 `⌘` 不应请求额外 Emoji、第二 Unicode 字体。原逻辑的断言失败。
- `default_primary_and_fallback_share_font_bytes`：用 `Arc::ptr_eq` 检查默认主、备用字体是否共享同一次分配。原逻辑的断言失败。

修复后两项通过；同时保留了缺字场景，使用 U+10FFFF 检查未覆盖字符仍然请求回退，显式 CJK 角色仍被列为必需。

测试不是无条件跨平台保证：覆盖测试只在 macOS、有 Arial Unicode 且没有字体覆盖环境变量时执行；共享测试在有覆盖配置或没有可发现字体时不验证指针相等。其他平台还需要自身字体环境下的集成验证。

### 6.2 独立渲染区分字体成本与桌面框架成本

[crates/desktop/examples/math_memory.rs](../crates/desktop/examples/math_memory.rs) 用包装 `System` 的全局分配器累计活跃请求字节，调用解析、排版、SVG 渲染，然后释放本次产物再读数。它没有启动 GPUI 窗口。

```sh
cargo run --locked -p kcastle-desktop --example math_memory -- '\text{中文}'
cargo run --locked -p kcastle-desktop --example math_memory -- '\text{😀}'
```

每种公式都使用新进程，避免前一个用例留下的全局字体缓存污染“首次加载”测量。输出中的 `after_drop_delta_MiB` 是相对渲染前的 Rust 活跃分配增量，不能当成整个进程的物理占用。

| 输入 | 原版产物释放后的增量 | 补丁后增量 |
| --- | ---: | ---: |
| `x` | 0.323 MiB | 0.323 MiB |
| `\frac{a}{b}` | 0.324 MiB | 0.324 MiB |
| `\alpha + \beta` | 0.325 MiB | 0.325 MiB |
| `\text{中文}` | 227.916 MiB | 22.493 MiB |
| `\text{⌘}` | 227.917 MiB | 22.494 MiB |
| `\text{😀}` | 227.914 MiB | 205.714 MiB |
| `\text{中文😀}` | 227.916 MiB | 205.716 MiB |

中文场景减少约 205.4 MiB；真实 Emoji 场景仍加载 Emoji 字体，只节省了约 22.2 MiB 的默认重复字体。这比笼统宣称“所有会话都省 205 MB”准确。

另外，对原版和补丁版本分别生成了 9 组 SVG：ASCII、分数、希腊字母、中文、命令符号、Emoji、中文混合 Emoji、求和式、带轮廓笑脸。输出逐字节一致。这验证了这些样例的表示不变，不等于覆盖了所有 Unicode 字符和操作系统字体。

完整本轮验证结果：

- 工作区测试：89 项 agent 测试、282 项 desktop 测试通过，合计 3 项忽略。
- 两个字体依赖：7 项测试通过，并加入 `just test` 与 CI。
- 格式检查、diff 检查、工作区 Clippy 和 release App 构建、签名验证通过。
- `just tla-check`、`just tla-self-test` 均尝试执行，但因本机无 Java 而未启动模型检查。相关架构说明已同步；现有模型覆盖的 worker、取消和结果身份协议未改变，没有因此新增字体缓存模型。

### 6.3 回到真实 App

从工作区 `target/Kcastle.app` 打开最新构建，确认进程路径，打开“推导下KV Cache”会话并滚动到公式区域。公式显示正常，两次稳定测量的 `Physical footprint` 都为 **126.8M**；本次启动、切换和滚动的历史峰值为 **241.3M**。

大块堆分配检查中仅剩一块 23,281,664 字节的 Unicode 字体，原来的 183 MiB Emoji 和第二份 Unicode 分配不再出现。

这个结果把独立复现带回了真实调用链。但前面的 351.9M 与这里的 126.8M 来自不同运行阶段，不是完全相同 UI 状态下的严格 A/B。约 205 MiB 的因果收益由独立实验支持；不能把整进程数值相减后的全部差额都归给补丁。

## 7. GPU 渲染为什么仍计入“内存”？

这与本次字体堆问题是不同的一层。M4 Pro 使用统一内存，CPU 与 GPU 共用物理内存；GPU 资源可能计入 App 的内存负担。Metal 的 `Private` 指 CPU 不直接访问该资源，不是另有独立显存。[Apple：资源存储模式](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

窗口图形内存更依赖像素尺寸，而不是消息的字数。以约 1180 × 720 逻辑点、2 倍 Retina 缩放和每像素 4 字节估算：

```text
(1180 × 2) × (720 × 2) × 4 ≈ 13 MiB / 张
3 张显示缓冲 ≈ 39 MiB
```

当前 `gpui-pre-apple 0.3.3` 的 `MetalRenderer::new` 使用 `BGRA8Unorm`，并调用 `set_maximum_drawable_count(3)`。上面的估算与实测 `IOSurface` 39.8M 相容，但不是逐个识别了所有 IOSurface。字体图集、中间纹理和其他图形资源另计。[Apple：Metal App 内存分析](https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app)

诊断中曾测到代码会话约 164M 图形相关内存：约 39.8M IOSurface、7.9M IOAccelerator，以及 116.7M `owned unmapped (graphics)`。最后公式会话快照中，同口径图形部分约 55M。

需要保留这个解释边界：164M 是一次快照，不是固定预算。特别是驱动归属的那部分，没有完成逐个资源归因，不能断言全部必要，也不能把它全部归为某张纹理的泄漏。后续若继续优化，应使用 Metal 资源分析验证具体分配、复用和释放，不能靠分类名称猜测。

## 8. 下次遇到类似问题，先看什么？

1. **少量输入触发大幅阶跃增长**：检查是否初始化了字体、解析器、模型或全局资源，不只看输入长度。
2. **大分配有非常稳定的尺寸**：把尺寸与资源文件核对，再追引用链和分配路径；匹配尺寸是线索，独立复现才进一步区分原因。
3. **接口出现 optional、lazy、cache**：读清楚它们实际控制的是失败策略、加载时机还是生命周期。名称不能代替执行语义。
4. **释放局部对象却不回落**：查最后一个所有者。`Arc` 和全局缓存保留的资源，不受会话退出或展示缓存淘汰直接控制。
5. **共享资源之前**：确认它们是否来自同一项配置知识。默认主、备用字体可共享；用户自定义输入必须保留边界。
6. **报告优化收益时**：分开已测出的资源节省、真实应用表现和仍未归因的波动。保留真实 Emoji、不同字体和 GPU 资源这些会改变结果的条件。

本次可复用的判断是：先找到“谁因为什么条件加载资源、谁持有它”，再减少不必要的资源选择。减少工作比给所有缓存再加一层回收策略更直接，但只有保留缺字、错误和自定义配置的语义，这个简化才成立。
