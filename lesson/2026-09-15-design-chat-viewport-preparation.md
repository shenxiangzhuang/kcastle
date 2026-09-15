---
type: design
created: "2026-09-15"
verified: "2026-09-15"
scope: Kcastle desktop ChatViewport and PreparedMarkdown; 41ceec7 to 2c679b2
---

# 会话切换时，怎样让昂贵的富文本准备退出首帧？

先看这次改动涉及的最短调用链。下面是调用关系摘要，不是可执行代码：

```text
旧版 41ceec7
选择 SessionRuntime → 发布 SessionView → 清空消息展示对象
  → 遍历历史消息 → 解析 Markdown → 构造高亮器 → 创建 UI 元素

新版 2c679b2
选择 SessionRuntime → 发布 SessionView → 建立源文本片段索引
  → ListState 请求视口附近的片段 → 先创建纯文本 UI
  → 布局后收集需求 → 后台准备 Markdown / 高亮 / 公式
  → 回到 UI 验证结果仍有效 → 替换对应片段并重新测量
```

`SessionRuntime` 管理会话数据，向桌面发布不可变的 `Arc<SessionView>` 快照；`ChatViewport` 管理当前聊天界面需要展示的片段。这里的“准备”是将源文本变成语法树、代码样式和公式 SVG；“展示”是创建 GPUI 元素、文字排版和绘制。两者原先夹在同一次 UI 更新里。

本篇回答一个问题：**怎样让会话切换只等待必要的可见内容，同时让后续富文本结果保持正确？** 它不讨论如何减少模型响应时间，也不把本次改动等同于数据库分页。

## 1. 先确定用户在等待什么

最初的现象是：切回一个含 Haskell/Python 快排示例的会话，即使此前打开过，仍明显停顿。可能的成本包括读取会话、重放事件、Markdown 解析、代码高亮和 UI 布局。仅凭“切换时发生”不能把原因归给数据库。

旧版 `crates/desktop/src/conversation.rs::chat_timeline` 对完整历史调用 `message_view`。下面省略容器样式，保留实际遍历方式：

```rust
self.core.session_view.conversation.messages
    .iter()
    .chain(self.core.transient_messages.iter())
    .enumerate()
    .map(|(index, message)| {
        self.message_view(index, message, window, cx)
    })
```

`message_view` 调用旧版 `MessagePresentationStore::sync_message(id, generation, revision, source, markdown)`。该方法返回消息的展示对象；首次构造时，`MessagePresentation::new` 会同步执行 `presentation.markdown.update(source)`。

高亮则经 `SelectionFrame::code_styles` 进入。它检查选择对象里的缓存，未命中时执行以下代码；这是 `crates/desktop/src/platform/gpui/text_selection.rs` 中的真实片段，相关旧渲染路径仍供其他调用者使用：

```rust
let mut highlighter = SyntaxHighlighter::new(language);
let completed = highlighter.update(None, &Rope::from(source), None);
debug_assert!(completed, "an unbounded syntax parse always completes");
let styles = highlighter.styles(&(0..source.len()), theme.as_ref());
```

`SyntaxHighlighter::new` 并不只是创建一个很小的 Rust 对象。在锁定的 `gpui-component 0.6.0` 中，它的构造路径调用 `Query::new`，编译语言的 tree-sitter 查询规则。规则大小与当前代码块大小不是同一回事。

旧版还有一个使成本反复出现的生命周期关系：

```rust
// 41ceec7，MessagePresentationStore::activate 的实际逻辑
if self.active_session.as_deref() != Some(session.as_str()) {
    self.entries.clear();
    self.active_session = Some(session);
}
```

会话切换清空展示对象，选择对象及其高亮缓存随之释放。于是，**会话数据可以已在 runtime 中，展示缓存却仍然未命中**。不能用“已经缓存会话”推断切回时不会重做高亮。

### 区分原因的对照

在真实数据的隔离副本上，用 release 构建和 1180 × 720 GPUI 测试窗口测量。两次预热后的结果如下，单位为 ms：

| 同一个快排会话 | 第一次 | 第二次 |
| --- | ---: | ---: |
| 原版完整打开路径并等待测试任务完成 | 245.235 | 248.040 |
| 诊断开关只跳过代码高亮 | 3.865 | 3.802 |

对照保留了文本、Markdown、布局和会话加载。进一步的分段记录显示：两个 Haskell 代码块分别只有 183 和 66 字节，但高亮器构造各需约 115–120 ms；解析代码并生成样式只额外花约 0.07–0.20 ms。预热后的会话加载约 2.1–2.5 ms，文档投影约 0.35–0.37 ms。

这支持一个具体结论：**在这个样本里，反复编译高亮查询是主要阻塞来源。** 它不证明所有长会话都受同一个瓶颈支配。旧公式路径在测试构建与产品构建中的调度还有差异，因此不能用测试里的公式冷启动数值推断原生应用的卡顿。

诊断最初考虑复用语言级查询。随后设计目标扩大为：恢复上次阅读位置、只准备附近内容、先显示纯文本、滚动时继续准备，并控制资源生命周期。这改变了选择方案时的约束。

## 2. 缓存、视口和后台执行各解决什么

| 方案 | 解决的问题 | 单独采用时留下的问题 |
| --- | --- | --- |
| 复用高亮查询或结果 | 避免重复计算 | 首次打开、失效和未命中仍可能阻塞 UI；需要容量和失效规则 |
| 只准备视口附近内容 | 不为当前看不到的内容付出昂贵成本 | 一个可见代码块的查询构造也可能耗时 120 ms |
| 把准备放到后台 | UI 不必同步等待准备完成 | 若后台仍处理整段历史，CPU、内存和过期任务仍会积累 |

因此本次采用“按视口限制需求 + 后台准备 + 纯文本先显示”。缓存仍有价值，但语言级查询复用没有在这次实现中完成。

理解效果可以用一个简化的成本分解：旧首帧包括同步解析和高亮；新首帧包括源索引、附近片段的纯文本排版，以及 UI 本身的更新。昂贵准备移到了后面，计算本身未必减少。后台执行与缓存命中是两种不同的收益。

## 3. 把需要同步完成的工作缩小到片段

新版 `crates/desktop/src/platform/gpui/chat_viewport.rs` 使用原生列表：

```rust
let list = ListState::new(0, ListAlignment::Top, px(600.0));
list.set_follow_mode(gpui_kit::FollowMode::Tail);
```

600 px 是视口前后的预加载余量。`gpui_kit::list` 按需回调 `DesktopApp::render_chat_row`，而不是让调用者遍历整个历史并创建所有消息元素。

仅按“消息”虚拟化还不够：一条回答就可能含 20,000 个段落。因此 `ChatViewport::sync` 为文本建立 `ChatRow` 源片段索引，历史保留片段位置和共享的消息引用。它不为所有片段建立 Markdown 语法树或选择状态。

源片段用以下身份定位，实际定义略去派生属性：

```rust
pub(crate) struct RowKey {
    pub(crate) message: MessageId,
    pub(crate) field: u8,
    pub(crate) start: usize,
}
```

`message` 标识消息，`field` 区分文本字段，`start` 是片段的源字节位置。阅读位置进一步保存 `ScrollAnchor::Block { id, field, source_offset, local_offset }`，最后一项是行内像素偏移。这样，纯文本变成富文本、上方内容高度变化后，仍能根据源位置找到对应内容；单独保存整个文档的像素偏移没有这个语义。

切块也不是无损地支持任意 Markdown 语义：代码片段携带围栏和语言信息；普通表格尽量完整保留；过大的非代码语义块退回纯文本。具体边界见后文。

### 先显示，再准备

`ChatViewport::row` 返回片段、选择状态和可选的 `Arc<PreparedMarkdown>`。`render_chat_row` 的核心分支如下，省略样式和测试标记：

```rust
let content = if let Some(prepared) = prepared {
    dsh_markdown::render_prepared_markdown(
        row.message.key.0,
        &prepared,
        self.core.layout.content_max_width,
        &selection,
        window,
        cx,
    )
} else {
    dsh_markdown::plain_text(row.plain().to_owned().into(), Some(&selection))
        .into_any_element()
};
```

这改变了调用者的契约：`row` 不承诺返回时 Markdown 已经解析完；没有准备结果也是一种正常、可展示的状态。旧版 `sync_message` 同步构造富文本，新版把这项隐含等待从展示对象 API 中移走了。

行回调记录本帧需求，使用 `cx.defer` 在布局之后调用 `finish_chat_frame`。该方法清理离屏展示对象，优先选择真正可见的片段，再处理预加载区域。这时才读取列表的最终布局信息，也避开了 GPUI 布局过程中持有的内部可变借用。

后台入口的实际签名为：

```rust
pub(crate) fn prepare_markdown(
    source: &str,
    theme: &HighlightTheme,
    cancelled: &AtomicBool,
) -> Option<PreparedMarkdown>
```

它读取文本和高亮主题，检查取消标记，返回语法树、代码样式和公式结果；取消时可以返回 `None`。跨线程携带的是这些数据，不是 GPUI 窗口或选择实体。

调度的关键是内层 `background_executor().spawn`。以下摘录省略了测试探针和结果发布代码：

```rust
let executor = cx.background_executor().clone();
let task = cx.spawn(async move |this, cx| {
    let prepared = executor
        .spawn(async move {
            dsh_markdown::prepare_markdown(&source, &theme, &worker_cancel)
        })
        .await;
    // 随后 this.update(cx, ...) 回到 UI，验证并发布结果。
});
```

外层 `async` 只是组织等待与回调。CPU 密集函数如果直接放在前台 future 里，仍会在被轮询时占用 UI 执行器；语法上的 `async` 不会自动把它移到后台。GPUI 元素创建、文字排版和绘制仍留在 UI 线程，本次改变的是它们面对的工作量以及前置等待。

## 4. 离开首帧之后，仍要回答“这个结果还属于谁”

考虑这个操作顺序：

```text
A 的片段开始准备 → 切到 B → A 的准备尚未返回 → 再切到 C
```

如果每次切换都清空“正在工作”的记录并启动新任务，旧的查询编译可能仍在运行。界面只记录了一个任务，并不代表实际上只有一个计算任务。

当前实现让 `InFlight` 持有任务、片段身份、版本、主题和取消标记。`release` 的核心逻辑是：

```rust
self.epoch = self.epoch.wrapping_add(1);
self.presentations.clear();
self.list.remeasure();
self.requested.clear();
if let Some(work) = &self.in_flight {
    work.cancel.store(true, Ordering::Relaxed);
}
```

这里没有把 `in_flight` 置空。取消是“不要继续做或发布这份工作”的请求，不是“昂贵函数已经停止”的事实。只有完成回调返回，才释放这个槽位。代价是 B 的富文本可能要等待 A 的一次不可中断调用结束；B 的纯文本仍可先显示。

完成时还必须通过 `current_result`。其实际逻辑如下：

```rust
if cancelled || self.epoch != epoch || self.dark != dark {
    return None;
}
self.requested.get(&key).copied().filter(|index| {
    self.rows
        .get(*index)
        .is_some_and(|row| row.key == key && row.revision == revision)
})
```

`epoch` 覆盖会话/投影失效，`revision` 约束源片段版本，`dark` 约束高亮主题，`requested` 表示当前还需要这行。通过后才能写入 `Presentation::prepared`，并调用 `list.remeasure_items` 重新测量该行。流式追加时，源内容没变的片段保留原 revision，避免每来一个 token 就否定全部工作。

### 结果输入也必须完整：表格公式的反例

这次异步拆分曾引入一个实际回归：段落公式能显示，表格里的公式却退回灰色 LaTeX 文本。

`PreparedMarkdown` 的公式键包含 `(source, display, font_size.to_bits())`。初版后台遍历给表格单元格继承了段落的 16 px；前台 `render_table` 使用 15 px。公式内容合法，但前台查找的键在准备结果里不存在。

修复是让准备和展示共用 `TABLE_FONT_SIZE`，并在遍历 `Node::TableCell` 时传给子节点，而不是放松查找或忽略字号。这也覆盖了公式嵌套在加粗文字里的情况。

这个反例说明：拆出后台计算时，需要把原来从展示上下文隐式取得的输入明确传过去。主题、字号、语言等是否影响产物，要看具体计算；不能只按源字符串判断结果能否复用。

### 资源也必须有结束时间

当前保留的准备结果有估算的 8 MiB 总预算、1 MiB 单片段准入限制。离屏淘汰或切换会丢弃展示对象；源消息仍由会话快照拥有。展开/评分等轻量交互状态单独保留。

公式 SVG 原来注册在全局资产表中。只释放 Markdown 对象不足以移除那张表里的字节。现在 `register_generated_asset` 返回 `Arc<GeneratedAsset>`，最后一个持有者释放时，由 `GeneratedAsset::drop` 删除注册项；测试通过额外保留一个 Arc 引用，模拟当前帧仍持有资产时它应继续存在。

这里可迁移的原则是：先区分会话数据、可见展示对象和后台产物的生命周期，再决定谁释放谁。预算是估算的准备数据预算，不是整个应用或原生 GPU 缓存的内存上限。

## 5. 怎样证明优化有效，又不把测试结果说得过头

### 首帧变快，不等于总工作量下降

在同一隔离快排会话上的原始 release 探针记录如下：

| 测量边界 | 旧版两次预热值（ms） | 新版两次预热值（ms） |
| --- | --- | --- |
| 应用快照并同步绘制首帧 | 241.616 / 242.279 | 0.704 / 0.660 |
| 打开会话并排空测试后台任务 | 245.235 / 248.040 | 243.716 / 245.726 |

这与代码机制相符：高亮查询依然要编译，但首帧不用等它。完整路径几乎没变，也阻止了“总计算快了几百倍”的错误解释。

这些都是 GPUI 测试窗口数据，不是鼠标点击到屏幕显示的原生延迟。隔离数据的旧诊断补丁只用于当时版本，不是长期可运行的 benchmark。

### 确定性测试守住调度和工作量

旧测试主要检查同一会话 revision 不变时复用，以及切换后展开/评分状态保留。它们没有覆盖“切回后重新构造昂贵高亮器”。只检查最终画面或最后保留多少对象，也看不到中间是否处理了整段历史再全部丢弃。

新增 `crates/desktop/src/conversation/performance.rs::chat_switch_and_scroll_do_not_wait_for_markdown` 使用测试专用 channel 暂停第一个准备任务。以下是其核心步骤的摘要，完整测试还初始化窗口和生成会话快照：

```text
生成两个各含 1,000 条富文本消息的 SessionView
→ 发布第一个快照，暂停准备任务
→ 确认纯文本行已出现在 GPUI 布局中、准备结果为 0
→ 连续切换快照，确认对应内容可见，worker_starts 始终为 1
→ 发送 ScrollWheelEvent，确认源锚点改变
→ 释放 channel，确认当前视口最终出现准备结果
→ 检查累计 worker_starts < 30
```

累计计数包括取消的和已经淘汰的工作，不只是最后留下的对象。已有 `chat_only_prepares_the_viewport` 还对单条 20,000 段消息断言新增准备次数小于 60。这些上限是防止全历史工作的粗粒度回归断言，不是最佳并发度或像素布局的规范。

准备任务内有测试专用的后台执行器断言。验证测试自身时，曾临时把任务放到 `ForegroundExecutor`，测试确实因 `Markdown preparation must run in the background` 失败；随后恢复实现。channel、计数器和该断言不进入产品构建。

表格公式测试 `prepared_table_math_uses_cell_typography` 则要求存在 `math:...` SVG 元素，且不存在 `math-fallback:...` 元素。修复字号前它失败，修复后通过。它验证“展示的是公式”，不只验证“这段源文本没有丢”。

这些测试直接覆盖快照发布与展示边界；它们不代替 `open_session` 的数据库路径测试，也不能保证任何未来代码都不会引入新的同步耗时。

### 可重复基准记录耗时趋势

仓库现有 Rust/GPUI 测试设施已能运行实际布局和受控调度，因此没有增加 Criterion。运行：

```sh
just bench-chat
```

该命令以 release 模式运行默认忽略的 `chat_presentation_benchmark`。三个固定生成样本不依赖私人会话数据；每轮变更 namespace，清空 Chat 展示结果。第 0 轮单独报告，后 20 轮取 nearest-rank p50/p95。它们是进程内预热数据，不是 20 次全新进程冷启动。

核心计时截取自当前基准，省略断言和输出：

```rust
let start = Instant::now();
view.update(cx, |app, cx| publish(app, &snapshot, &namespace, cx));
let first_frame_ms = start.elapsed().as_secs_f64() * 1000.0;

let start = Instant::now();
cx.run_until_parked();
let settle_ms = start.elapsed().as_secs_f64() * 1000.0;
```

这里 `publish` 发布预先构造的 `Arc<SessionView>`，不读数据库。GPUI 测试上下文在 `update` 返回前同步绘制脏窗口；其间没有推进测试执行器。`settle_ms` 随后排空后台准备与渐进重排。测试执行器在测试线程上交错调度前后台，所以这个数值也不是产品环境里单独的后台 CPU 时间。

2026-09-15 的实际基线：Apple M4 Pro、macOS 26.5.2、rustc 1.97.1、1180 × 720 窗口，单位 ms：

| 样本 | 首帧 p50 / p95 | 后续完成 p50 / p95 |
| --- | ---: | ---: |
| 1,000 条富文本消息 | 0.721 / 0.939 | 238.236 / 258.528 |
| 单条 20,000 段消息 | 1.774 / 2.047 | 14.491 / 15.745 |
| Haskell 与公式表格 | 0.549 / 0.656 | 122.741 / 127.222 |

普通 CI 执行确定性测试；耗时基准目前不设硬阈值。比较基线应固定机器、工具链、样本和构建配置。将来若有稳定的专用 runner，可再根据重复测量的波动设门槛；当前没有声称已完成自动化耗时回归判定。

## 6. 回到一次真实切换，以及仍然存在的边界

现在切回长会话时，快照先交给 `ChatViewport`；列表按保存的源锚点请求附近片段。纯文本先出现，一个后台任务从当前需求中逐步准备内容；用户滚动改变需求后，旧结果必须重新通过身份和版本检查。一个片段变成富文本，只使相关行重新测量。离屏内容的展示资源随之释放。

本次隔离原生应用的点击记录确认了会话选择、内容切换及来回切换后的阅读位置恢复。但是第一次视觉检查漏掉了表格内的 LaTeX 回退；用户指出后才补齐根因修复和 SVG 断言。修复后的原生表格复核又受到 `noWindowsAvailable` 阻挡，不能把自动化测试通过写成最终原生画面已确认。原生点击到屏幕显示的可靠时延也没有测得。

当前设计仍有这些明确代价：

- `ChatViewport::sync` 仍在 UI 线程维护历史源索引并扫描变化文本。源数据加载和全历史索引没有变成数据库分页；更大的历史仍需重新测量这一成本。
- 2 KiB 内的源通常整体保留；较大代码按约 24 行/2 KiB 分片。超过 16 KiB 的非代码语义块可能退回纯文本；跨片段 Markdown 引用不解析。这是完整语义与有界工作量之间的取舍。
- 一个 worker 简化并发和资源边界，但旧任务的一次不可中断库调用可能推迟新视口的富文本完成。取消检查不能抢占正在执行的 `Query::new`。
- 字体整形和绘制仍在主线程；估算的准备数据预算不覆盖会话源数据、原生 GPU/字体资源或独立的 Trajectory 渲染路径。

因此，下次遇到同类问题，可以依次问：**用户首次看到内容前必须完成哪些调用？昂贵调用是否只服务当前需求？取消后计算真的停止了吗？结果发布依赖哪些源与样式输入？测试检查的是最终状态、累计工作量，还是实际耗时？** 每个问题对应一个可检查的边界，而不是泛泛地要求“再加一层缓存”。

## 7. 核验范围与代码入口

本文在 `2c679b2` 上核对实现，使用 `41ceec7` 重建旧调用链，并复查本次诊断、失败对照和验证日志；此次写作没有重新实施优化或重新运行整套性能实验。

已有验证记录：工作区 370 项测试通过、3 项忽略（含可选基准）；最终桌面套件 281 项通过；release 基准、工作区 Clippy、格式检查通过。Chat TLA+ 模型检查探索 156 个不同状态，通过当前需求/代际正确性、worker 上限和取消收敛等检查；自检主动注入过期发布、不淘汰及并行任务故障，得到预期反例。模型假设 worker 最终返回，并不证明 Rust 实现与模型完全等价，也不证明像素锚点、公式语义或字节预算正确。

相关仓库入口：

- [crates/desktop/src/app.rs](../crates/desktop/src/app.rs)：`select_runtime`、`sync_message_presentations`、阅读位置保存/恢复。
- [crates/desktop/src/conversation.rs](../crates/desktop/src/conversation.rs)：`chat_timeline`、`render_chat_row`、视口与超长消息回归。
- [crates/desktop/src/platform/gpui/chat_viewport.rs](../crates/desktop/src/platform/gpui/chat_viewport.rs)：源切块、`row`、`current_result`、`finish_chat_frame`。
- [crates/desktop/src/dsh_markdown.rs](../crates/desktop/src/dsh_markdown.rs)：`prepare_markdown`、`render_prepared_markdown`、表格公式测试。
- [crates/desktop/src/assets.rs](../crates/desktop/src/assets.rs)：`GeneratedAsset` 及资产生命周期测试。
- [crates/desktop/src/conversation/performance.rs](../crates/desktop/src/conversation/performance.rs)：受控后台测试、生成样本与基准。
- [docs/architecture/desktop.md](../docs/architecture/desktop.md#chat-viewport)：实现约束和性能基线。
- [docs/architecture/tla/chat-presentation/README.md](../docs/architecture/tla/chat-presentation/README.md)：模型范围、假设和实现映射。

旧版只跳过高亮的诊断、首帧对照与原生检查更正记录保存在本地忽略目录 `target/session-switch-diagnosis/`。它们不是仓库依赖；本文已列出决定结论的数据及测量边界。后续可复现的入口是 `just bench-chat` 和仓库中的回归测试。
