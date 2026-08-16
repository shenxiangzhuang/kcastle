# Kcastle Desktop UX 架构重构方案 v1

## 1. 目标与边界

本方案的目标不是把现有页面“拆成更多文件”，而是让桌面端 UX 具有以下可证明性质：

1. 相同状态和相同测量输入必然得到相同布局与 ViewModel。
2. 用户意图、Agent 事件和异步结果只能通过同一个状态迁移入口改变 UX 状态。
3. 过期异步结果、过期布局测量和过期滚动恢复不能污染新页面。
4. 流式输出按显示帧合并，单个 token 不触发一整棵 UI 的重复解析和布局。
5. 布局响应实际容器尺寸，不依赖散落在 View 中的窗口宽度判断或固定底部留白。
6. 关键行为可以在不启动真实窗口、不访问网络和不依赖时间的情况下测试。
7. GPUI 生命周期 API 只能由平台适配层调用，避免在普通事件回调里触发布局阶段 panic。

v1 不引入 Web 式 CSS 框架，不复制 DSH 特有的 Agent mode、插件或 preset。视觉和交互以 DSH 的通用桌面 UX 为基准，Agent 能力仍由 `kcastle-agent` 提供。

## 2. 第一性原理

### 2.1 状态、计算和副作用必须分离

桌面应用中的复杂 Bug 通常不是“GPUI 性能不够”，而是同一段逻辑同时做了四件事：读取组件尺寸、改业务状态、启动异步任务、再直接滚动。v1 将它们拆为：

```text
Input / AgentEvent / Measurement
              │
              ▼
           Action
              │
              ▼
   reduce(AppState, Action)
        │              │
        ▼              ▼
   new AppState      Effect[]
        │              │
        ▼              ▼
    ViewModel      GPUI Effect Runner
        │              │
        └──────┬───────┘
               ▼
             View
```

Reducer 是纯函数边界：它不读取窗口、不访问文件系统、不启动 Task、不持有 GPUI Entity。Effect Runner 是唯一解释 `CreateSession`、`StartRun`、`OpenSession`、`RenameSession` 和滚动命令的地方。

### 2.2 响应式布局是约束求解，不是 breakpoint 堆叠

布局输入只包含可测量事实：viewport、实际 main container、rem、composer 实际高度、安全区和当前可见 surface。`resolve_layout` 和 `resolve_container` 根据容量求解：

- Sidebar：`Expanded` 或 `Rail`；
- Trajectory：`Ledger`、`Split` 或 `Overlay`；
- Height：`Compact`、`Regular` 或 `Spacious`；
- 阅读列、Composer 列、两侧 padding；
- transcript 尾部 inset。

阈值表达的是“两个 pane 的最小可用宽度之和”，而不是某个设备名。嵌套内容使用实际 main container 测量；窗口扩大时，模式不能退化为更拥挤的模式。

### 2.3 滚动位置是语义状态，不是像素快照

像素 offset 在字体、窗口宽度、Markdown 换行或 Composer 高度变化后没有稳定含义。Chat 使用：

```text
Tail
Message { id, local_offset }
```

布局变化前记录第一条可见消息及其局部偏移，变化后等待新 generation 的 transcript 和 message bounds，再恢复语义锚点。所有测量携带 `LayoutGeneration`；旧 generation 的测量只能被忽略。

### 2.4 流式渲染以帧为背压边界

网络 token 到达频率和屏幕刷新频率是两个系统。正确管线是：

```text
AgentEvent stream
  → StreamBatch（相邻 text/reasoning 合并，结构事件立即 flush）
  → next frame gate
  → Conversation Reducer
  → changed-message projection
  → incremental Markdown frontier
  → one notify / frame
```

没有固定 200 ms 延迟。单帧最多接收 128 个事件，工具开始、审批和完成等结构事件不会被延后。`StreamTelemetry` 分别记录原始 delta、实际投递事件、batch 数和最大 batch，便于后续建立 P95 指标。

## 3. 模块与依赖方向

```text
main.rs
  └─ lib.rs / app.rs                 组合根与 GPUI shell
       ├─ domain/                    纯状态、Action、Reducer、ID
       ├─ application/               纯流式泵、ViewModel / selector
       ├─ layout/                    纯布局、表格、语义滚动算法
       ├─ platform/gpui/             GPUI 生命周期和渲染缓存适配
       └─ *_view modules             只投影 ViewModel 并发送 Action
```

依赖规则：

- `domain`、`layout`、`application` 禁止依赖 GPUI；
- `platform/gpui` 可以依赖纯层并解释 Effect；
- View 不得直接写 `AppState`，只调用 `dispatch` / `dispatch_local`；
- `app.rs` 是组合根，可以持有 Agent、Input Entity、FocusHandle、ScrollHandle 和平台 modal 等运行时对象，但 UX 领域状态必须在 `AppState`；
- `kcastle-agent` 不依赖 Desktop，Desktop 只消费 `AgentEvent`。

`architecture_tests.rs` 自动执行上述依赖守卫，并禁止 `on_next_frame`、`layout_bounds` 等 draw-phase API 逃出 GPUI 适配层。

## 4. 状态模型

### 4.1 AppState

`AppState` 包含：

- `ConversationState`：消息、标题、turn/step、token usage；
- `ComposerState`：打开的菜单和键盘高亮项；
- `SidebarState`：搜索、分组、排序、popover target；
- `Surface`、`DetailsState`、`TrajectoryState`；
- `WorkspaceState`、`SessionState`；
- `RunState` 和带 ID 的 `PendingSessionOperation`；
- `ApprovalState`；
- follow-tail、unread delta；
- `LayoutInput`、`LayoutPlan`、`LayoutGeneration`。

GPUI Entity、输入法状态、FocusHandle、ScrollHandle、Agent、RunControl、Theme 和原生文件选择器不进入 AppState；它们是不可序列化的平台资源。

### 4.2 Action 与 Effect

Action 表达已经发生或用户希望发生的事实，例如：

- `Conversation(TextDelta / ToolStarted / RunFinished)`；
- `ToggleSidebar`、`SetComposerMenu`、`SelectDetails`；
- `LayoutInputChanged`、`Scroll`；
- `BeginSessionCreation`、`BeginOpenSession`、`BeginRenameSession`；
- 带 operation ID 的完成和失败结果。

Effect 表达尚未执行的外部动作：

- `ApplyChatTail`；
- `CreateSession`；
- `StartRun`；
- `OpenSession`；
- `RenameSession`。

View callback、Agent callback 和测量 callback 都不能绕过 Reducer 写领域字段。唯一例外是 `DesktopApp::new` 的原子初始化，以及平台资源自身的更新。

## 5. 异步一致性

### 5.1 Identity token

每个有竞争可能的异步操作在开始时由 Reducer 分配单调 ID：

- `OperationId`：创建、打开、重命名 Session；
- `RunId`：Agent run；
- `LayoutGeneration`：布局/内容测量代次；
- `MessageId`：跨 session reload 仍不复用的渲染身份。

完成 Action 必须同时满足“类型匹配 + ID 匹配 + 必要目标匹配”才会被接受。用户已经切换页面时，旧结果不会更新标题、会话、滚动或 Agent ownership。

### 5.2 Session 创建时机

选择 Project 或点击新建只进入内存态 `New chat`。第一次非空提交才发 `BeginSessionCreation`。创建成功后同一个 Action 原子地写入 session path、刷新列表并发出 `StartRun`；创建失败会还原输入，且不在侧栏留下伪 Session。

### 5.3 Agent ownership

Idle 时 Desktop 持有 `Agent`；run 开始后所有权转入 `ActiveAgent` Task，UI 只持有 `RunControl`。完成时只有匹配 `RunId` 的结果可以归还 Agent。Abort、审批和 queue/steer 通过 RunControl，不共享可变 Agent。

## 6. 渲染投影

领域 `Message` 只保存语义文本和状态，不保存 `SharedString`、Markdown AST 或 GPUI Element。`MessagePresentationStore` 位于 `platform/gpui`：

- 以 `MessageId` 为 key；
- source 未变化时不更新 SharedString；
- 只对 Assistant 文本维护增量 Markdown 状态；
- session 替换后删除失效缓存；
- Markdown 只重解析尚未闭合的 frontier，已稳定 block 冻结复用。

ViewModel 位于 `application/view_model.rs`，把 AppState 投影为标题、empty state、turn/step 和 token status 等只读数据。View 不重复实现统计与格式化规则。

## 7. 响应式与内容布局

### 7.1 测量路径

1. `observe_window_bounds` 提供 viewport 和 rem；
2. `measured_container` 在 prepaint 读取实际 main/composer bounds；
3. 测量差小于 0.5 px 不产生 Action，防止抖动；
4. Reducer 生成新的 LayoutPlan 和 generation；
5. after-layout callback 只在下一帧恢复语义滚动。

测量只能发生在 request-layout/prepaint/paint 生命周期。普通输入、发送消息和异步回调永远不调用 `layout_bounds`，从架构上消除此前的 GPUI panic 路径。

### 7.2 宽度

- Sidebar 只有在“展开宽度 + 主内容最小宽度”可容纳时展开；
- Trajectory 详情只有在 ledger + details 最小宽度可容纳时 split，否则 overlay；
- 阅读列和 Composer 各有独立 max-width，但共享容器求解器；
- 任意宽度（包括 0、NaN、Infinity 输入）均输出有限、非负且不越界的几何；
- Markdown table 为整表分配一次列宽，header 和 row 共享 plan；低于最小宽度时整表横向滚动，不让单元格各自滚动。

### 7.3 高度

短窗口进入 Compact height mode，隐藏非必要 token status，tail inset 同步减少。Composer 尾部空间始终由“实测 Composer 高度 + status + safe area + comfort gap”计算，不使用固定 256 px。

### 7.4 Dialog 与 overlay

Modal、trajectory details 和 popover 应优先使用 `max_w + w_full + min_w(0)`，并在容器容量不足时 overlay/scroll。原生文件选择器、输入法和 focus 对象保留在平台层；打开/关闭意图仍应进入 Reducer 或平台 modal coordinator。

## 8. 滚动规则

- 用户向上滚动立即关闭 follow-tail；
- follow 关闭时，流式 delta 只累加 unread，不与用户“对抗”；
- 回到底部或点击 unread/tail button 才重新启用 follow；
- 流式追加在 follow 模式下每帧最多执行一次 `scroll_to_bottom`；
- Composer reflow、侧栏变化、窗口 resize 和 session 内容替换都通过 generation-aware semantic anchor 恢复；
- stale measurement 不得改变当前 offset；
- Trajectory 与 Details 使用独立 ScrollHandle，overlay 不复用 ledger 的滚动容器。

## 9. 流式性能预算

建议在 debug telemetry 和后续 benchmark 中固定以下验收目标：

- 每显示帧最多一次 App notify 和一次 Markdown frontier update；
- 结构事件到 UI 不跨帧排队；
- 10,000 个小 delta 的累计 Markdown parse work 近似线性；
- 用户离开 tail 后 0 次自动 scroll；
- resize 期间没有同步文件 I/O、网络 I/O 或全 transcript Markdown 重建；
- release 构建下连续流式输出无肉眼可见的 200 ms 阶梯。

## 10. 测试体系

### 10.1 纯单元测试

覆盖 Reducer、Session/Run identity、Conversation delta、标题、turn/step、scroll restore、layout、container、table、ViewModel 和 stream coalescing。

### 10.2 Property test

`proptest` 当前验证：

- 任意有限窗口/容器测量不会生成非法几何；
- 窗口变宽时响应式模式不会退化；
- 任意 UI Action 序列保持 state/layout flags 同步；
- 任意 token chunk 边界产生相同 Assistant 文本；
- 任意消息角色序列 reindex 后 turn 单调；
- 任意表格列约束不会被 allocator 破坏。

失败 seed 保存在 `proptest-regressions`，CI 会永久重放。

### 10.3 Mutation test

`.cargo/mutants.toml` 只扫描纯核心和增量 Markdown，避免把 GPU/平台胶水的编译成本当成测试质量。每周和手动 workflow 执行。首轮关键突变覆盖 operation/run guard、layout generation、frame event 分类；任何存活突变都必须通过补测试或删除无意义分支解决。

### 10.4 GPUI lifecycle test

使用 `gpui::test` 真正执行 layout/prepaint/paint：

- frame gate 可以在 draw phase 之外安全 armed；
- transcript bottom inset 在真实滚动容器中生效；
- measured container 在 prepaint 更新，并在 resize 后给出新尺寸；
- stale generation 不恢复滚动。

### 10.5 视觉与点击矩阵

最终验收必须在真实 app 完成，而不是只看截图：

| 场景 | 窗口 | 验收 |
| --- | --- | --- |
| New chat | 900×620 / 1180×720 / 宽屏 | 无伪 Session；Composer 不遮内容 |
| Sidebar | rail / expanded | traffic lights 不重叠；hover/action 对齐；标题时间不重叠 |
| Streaming | tail / 用户上滑 / 回到底部 | 无对抗；无阶梯卡顿；unread 正确 |
| Composer | 1 行 / 多行 / IME | 高度实测；内容始终可滚到完整末尾 |
| Trajectory | ledger / split / overlay | 每种 cell 可开详情；pane 独立滚动 |
| Markdown | heading/list/code/table/CJK | 节奏一致；代码高亮；表格整体验证 |
| Resize | 连续拖动宽和高 | 不 panic；锚点稳定；模式只按容量切换 |
| Session async | 快速打开/切换/重命名 | 旧结果不能覆盖新状态 |

## 11. 质量门禁

每次合并前：

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

涉及纯核心时增加目标 mutation run；涉及布局和渲染时增加真实点击/resize 与截图矩阵。任何 GPUI panic、stale async 覆盖、内容被 Composer 遮挡或用户滚动对抗均为 release blocker。

## 12. 迁移阶段与完成定义

1. **建立纯核心**：AppState/Action/Reducer/Effect、typed IDs、lib/bin 分离。
2. **收口消息和异步**：Conversation Reducer、Session/Run operation identity、Effect Runner。
3. **重建流式管线**：frame batch、结构事件 flush、增量 Markdown projection、telemetry。
4. **重建布局与滚动**：容器测量、LayoutPlan、height mode、semantic anchor、shared table plan。
5. **收口 View**：通用 selector/ViewModel；View 无领域状态直写。
6. **测试有效性**：property、mutation、架构守卫、真实 GPUI lifecycle test。
7. **全量与人工验收**：workspace 门禁、release build、真实 app 点击和连续 resize。

“代码能编译”不是完成。完成必须同时满足：架构守卫全绿、关键 mutation 无存活、全 workspace 门禁全绿、真实点击矩阵无 panic/遮挡/滚动对抗，并且本文件中的依赖规则与代码一致。

## 13. 参考实现与资料

- GPUI 官方 examples：<https://github.com/zed-industries/zed/blob/main/crates/gpui/examples/README.md>
- GPUI `Window` 生命周期 API：<https://docs.rs/gpui/latest/gpui/struct.Window.html>
- GPUI `ListState`：<https://docs.rs/gpui/latest/gpui/struct.ListState.html>
- gpui-component：<https://github.com/longbridge/gpui-component>
- cargo-mutants 配置：<https://mutants.rs/config-file.html>

## 14. v1 落地状态与验收记录

本方案已按上述依赖方向落地，而不是仅作为未来设计：

- `domain/` 已承载 `AppState`、Action、Effect、Conversation Reducer 和 typed IDs；
- `application/` 已承载按帧 stream pump、telemetry 和 ViewModel；
- `layout/` 已承载容器、主布局、表格列分配和语义滚动纯函数；
- `platform/gpui/` 已承载 Effect Runner、帧时钟、prepaint 测量、layout runtime 和消息渲染投影；
- `architecture_tests.rs` 已把纯层依赖和 draw-phase API 边界变成自动守卫；
- `proptest-regressions/` 已保存真实发现的几何与会话序列失败 seed；
- `.cargo/mutants.toml` 与定时 workflow 已建立纯核心 mutation 门禁。

2026-08-16 的 release app 点击验收实际覆盖了：

1. 1180×720 宽窗和约 900×620 矮窄窗之间连续 resize，Sidebar 自动切换且 traffic lights 不重叠；
2. 发现并修复 Compact mode 误把整个空白页 Hero/Composer 当成 status chrome 隐藏的问题，并增加 ViewModel 回归测试；
3. Project hover 显示 overflow 与新会话笔记按钮；点击仅打开内存草稿，Sidebar 没有 `New Session` 伪记录；
4. 第一次非空提交后才创建持久 Session，标题和时间使用独立宽度槽，不发生重叠；
5. Composer 内联模型/推理强度菜单和仅含 General/Models 的 Settings 均可点击；
6. Chat 长流式输出期间上滚后保持消息锚点，输出继续增长且出现 `Back to bottom`，不会把用户抢回尾部；
7. 回到底部后最终第 400 行和结束段落完整显示在 Composer 上方；
8. Trajectory 的 Tool 与 Message cell 均可打开类型化详情，Tool 五标签和 Message 三标签正确，header 只有 Close；
9. 120 行 Tool Result 在窄窗 Overlay 中可独立滚动到末尾，ledger 与 Composer 保持稳定；
10. 发现并修复三位数 Markdown 有序列表 marker 固定 28px 导致编号折行的问题，最终 `392.`–`400.` 对齐显示。

点击验收的意义是补足纯函数测试无法覆盖的最终组合行为；其中发现的两个缺陷都先保留稳定复现，再补自动测试并重新验证 release bundle。
