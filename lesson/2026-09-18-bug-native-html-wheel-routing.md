---
type: bug
created: "2026-09-18"
verified: "2026-09-18"
scope: Kcastle HTML previews / macOS WKWebView / gpui-pre-macos 0.3.3
---

# 为什么滚动处理函数正确，HTML 和正文仍会轮流卡住？

```text
修复前的 HTML 滚轮路径：
macOS NSEvent
  → AppKit / WebKit 投递 DOM wheel
  → document.js 判断哪个滚动容器能移动
  → 内部移动，或 postMessage → host.html → IPC
  → Rust BrowserEvent::Wheel → GPUI 聊天列表
```

Kcastle 的聊天正文由 GPUI 绘制，HTML 代码块则由嵌入的 WKWebView 渲染。它们不是同一棵 DOM：浏览器滚动到边界，不能指望浏览器自动替 GPUI 滚动聊天列表。`document.js` 因而负责决定内部是否消费滚轮；需要交给正文时，通过宿主页面和进程间消息（IPC）通知 Rust。

这次要回答的问题是：**跨原生视图与网页转发输入时，怎样确认事件既到达了处理入口，又交给了真正的接收者？** 排查先遇到 HTML 内偶发卡住，修复入口后又引入正文不能滚动，最终发现这两个检查缺一不可。

本文对应 `codex/inline-html-previews` 分支、基于 `6c508fe` 的工作区改动；撰写时修复尚未提交。下面的历史错误代码来自本次排查过程，不代表该提交中的代码。实现入口见 [macos.rs](../crates/desktop/src/html_preview/macos.rs) 的 `CursorOwner::update`、[document.js](../crates/desktop/src/html_preview/document.js)、[host.html](../crates/desktop/src/html_preview/host.html) 和 [html_preview.rs](../crates/desktop/src/html_preview.rs) 的 `BrowserEvent::Wheel`。

## 先定义一个事件由谁消费

用户要求的最终行为如下。“到边界”包括内容太短、根本没有内部滚动空间。

| 指针位置 | 仍有内部滚动空间 | 已到内部边界 |
| --- | --- | --- |
| HTML 区域外的正文 | 只滚正文 | 只滚正文 |
| 正文中的 HTML 预览 | 优先滚最内层可移动容器 | 后续事件交给正文 |
| 右侧栏 HTML 预览 | 只滚侧栏内部 | 留在侧栏，不推动正文 |

一次滚轮事件只交给一个滚动层。如果内部距离底部只剩 10 px，而事件请求向下 40 px，本次内部移动 10 px，剩余量不再同时推动正文；下一次事件才尝试交接。这样避免两个独立渲染系统同时移动产生抖动。反向滚动时重新判断，不把所有后续事件永久锁在外层。

实现按位移绝对值选择主轴，防止触控板的小幅横向漂移让横向表格截住垂直滚动。以下是当前代码中消费判定的关键片段，省略了此前的 CSS、范围和方向检查；`vertical` 表示主轴，`dx`、`dy` 是标准化后的像素位移：

```javascript
const before = vertical ? node.scrollTop : node.scrollLeft;
node.scrollBy({
  top: vertical ? dy : 0,
  left: vertical ? 0 : dx,
  behavior: 'instant'
});
if ((vertical ? node.scrollTop : node.scrollLeft) !== before) return;
```

CSS 声称 `overflow:auto`，或者 `scrollHeight > clientHeight`，只说明它值得尝试；只有位置实际改变才算消费。如果没动，就继续检查祖先。短 HTML 没有任何可移动祖先时，内联预览会直接交给正文。

这些规则解决的是“事件进入处理函数以后怎么办”。它们不能保证事件会进入函数。

## 第一个故障：原生输入到了，DOM 入口没有到

最初的 JavaScript 测试直接调用 `wheel` 回调。短内容、横向表格、边界交接和反向滚动都能通过，用户却仍能在“匀速运动”区域复现卡住。诊断版有时又暂时正常，因此“这次能滑”不能作为修复依据。

随后把观测点放在同一条事件链的不同边界：

```text
WHEEL native       AppKit 本地监听器收到事件：位置、位移、是否落在可见预览内
trace / entry      document.js 已进入现有 wheel 回调
trace / attempt    尝试内部滚动，记录 before / after
WHEEL ipc          宿主发出的消息到达 Rust
WHEEL receive      当前文档的回调通过身份检查
WHEEL before/after GPUI 聊天列表的位置变化
```

决定性证据来自 release 诊断版：出现卡住的两段日志里，约 6 秒和 11 秒没有 DOM `entry`，期间仍不断收到落在预览区域内的非零原生滚轮事件。因此，继续修改 DOM 回调内部的方向判断或 `preventDefault()`，无法修复这些根本没有进入回调的事件。

这个证据定位了**原生事件到 DOM 入口之间的投递断点**，没有单独证明 WebKit 内部的具体缺陷。原生子视图随着虚拟列表移动、手势仍沿用旧目标，是合理解释；本次没有通过 WebKit 内部追踪证实它。不能把这个解释写成已经查明的浏览器内部根因。

同样，“CSS 看起来可滚、实际滚不动”是值得防御的另一类错误，已由位置变化检查覆盖，但它不能解释连 DOM 入口日志都缺失的这两段记录。

## 把路由放到已经确认收到事件的入口

最终复用 `CursorOwner` 已有的窗口级本地监听器，增加滚轮路由，没有另外创建全局输入监听器。当前路径为：

```text
macOS NSEvent → CursorOwner::update 安装的监听器
  ├─ 在当前可见 HTML 裁剪区域内
  │    → WKWebView 执行 host.previewWheel(x, y, dx, dy)
  │    → iframe 收到 nativeWheel 消息
  │    → elementFromPoint → 共用的 scroll(...) 策略
  │         ├─ 内部实际移动：结束
  │         ├─ 内联边界：IPC → GPUI 聊天列表
  │         └─ 侧栏边界：结束，不交给正文
  └─ 在 HTML 区域外
       → GPUI 原生输入视图的 scrollWheel:
```

监听器每次按当前可见裁剪矩形选目标，包括惯性事件；把窗口坐标转换为完整浏览器视图中的坐标，再由宿主转换成 iframe 坐标。已接管的事件返回空指针，让 AppKit 不再重复投递给浏览器的默认滚动路径。Ctrl/Command 修饰的事件保留原来的处理路径。

宿主的渲染模式分支如下，`frame` 是有独立、不透明来源的沙箱 iframe，参数使用浏览器坐标和像素位移；源码模式另行处理源码容器：

```javascript
const rect = frame.getBoundingClientRect();
frame.contentWindow.postMessage({
  kind: 'nativeWheel',
  x: x - rect.left,
  y: y - rect.top,
  dx, dy
}, '*');
```

iframe 只接受父窗口发来的这类消息，并复用同一套滚动策略：

```javascript
if (event.source === parent && event.data?.kind === 'nativeWheel') {
  const {x, y, dx, dy} = event.data;
  scroll(document.elementFromPoint(x, y), x, y, dx, dy);
}
```

它不再要求 WebKit 先产生一个 DOM `wheel`。其他平台的 DOM 入口仍调用同一个 `scroll(...)`，避免维护两套边界规则。沙箱没有因此得到原生调用权限：原生脚本只调用受信宿主的固定入口，宿主仍检查 iframe 消息来源，Rust 仍检查宿主令牌和文档 generation。

内联边界进入 `BrowserEvent::Wheel` 后，Rust 通过 `window.defer` 在释放实体可变借用后分发给聊天视口。异步返回时行可能已经移动，所以目标是聊天列表本身，而不是用旧 iframe 坐标重新猜接收者。侧栏事件则在文档、宿主和 Rust 回调处都禁止交给正文。

代价也很明确：macOS 的普通滚轮不再依赖浏览器默认滚动，而由应用控制主轴选择、坐标转换和同步位移。原生路径也不会先执行页面自己的 DOM `wheel` 监听器。这适合当前“预览滚动归属由应用定义”的约束；若以后要支持页面自定义滚轮缩放等交互，必须重新定义契约，不能假定此方案等价于完整浏览器行为。它也仍依赖 JavaScript 能执行，不能保证无限循环的页面脚本继续响应。

## 第二个故障：方法存在，不等于对象负责处理

原生入口接管后，用户确认 HTML 可以滑动，但正文反而不能滑动。这是此次修复引入的回归。

当时区域外的错误分支是：

```rust
// 历史错误代码：content 的类型也是 NSView，但不是 GPUI 的输入视图。
if let Some(content) = state.window.contentView() {
    drop(state);
    content.scrollWheel(native_event);
    return std::ptr::null_mut();
}
```

问题不在事件类型，也不在 delta 正负号，而在 `content` 的身份。`Cargo.lock` 锁定的实际依赖是 `gpui-pre-macos 0.3.3`，它的 `src/window.rs`：

- 从 `native_window.contentView()` 取得 AppKit 外层容器，再创建独立的 `native_view` 加入容器。
- 在 `VIEW_CLASS` 上注册 `scrollWheel:`，对应 `handle_view_event`。
- `MacWindow::window_handle` 返回的是这个 `native_view`。

因此，相关层次是：

```text
NSWindow
  └─ contentView：AppKit 外层容器
       └─ GPUI native_view：注册了 scrollWheel:，也是 raw window handle 指向的视图
            └─ ClipView：Kcastle 的预览裁剪视图
                 └─ WKWebView
```

向外层容器调用同名方法，并没有进入 GPUI 的输入回调；紧接着返回空指针又拦截了原事件，于是正文没有机会滚动。Objective-C 的方法调用在类型层面合法，编译和 Clippy 都不可能据此发现“对象找错了”。

这也说明，检查依赖时必须核对锁文件中的实际包名和版本；机器上另一个 `gpui` 包或历史版本的窗口结构不能替代当前实现的证据。

修正没有猜测窗口的第几个子视图，也没有通过当前焦点寻找接收者，而是沿用已经建立的挂载关系。`ClipView::new` 把裁剪视图直接挂在 raw window handle 对应的 GPUI 视图下，因此其父视图就是需要的接收者。以下是当前关键代码，`clips` 是本窗口可见预览的浏览器集合，`state` 是监听器持有的 `CursorState`：

```rust
// 安装监听器时保存确定的 GPUI 输入视图；原生视图访问在 UI 线程上。
let Some(input_view) = clips
    .first()
    .and_then(|browser| unsafe { browser.clip.0.superview() })
else {
    return;
};
// input_view 存入 CursorState，类型是 Retained<NSView>。
```

```rust
// 区域外的当前处理分支。
let input_view = state.input_view.clone();
drop(state);
input_view.scrollWheel(native_event);
return std::ptr::null_mut();
```

先释放 `RefCell` 借用，再进入 GPUI 回调，避免回调触发布局或其他更新时重入同一借用。保留明确的视图引用，也让切换到没有 HTML 的会话后仍然安装着的监听器有正确的转发目标。

这次 KISS 的具体落点是：利用 `ClipView::new` 已经确定的父子关系，替换一个错误接收者。无需遍历视图树、按类名猜测或维护另一套正文滚动状态。但该关系有条件：如果以后改变 `ClipView` 的挂载层级，就必须同步检查这里的输入路由。

## 侧栏改变的是滚动归属，不只是尺寸

同一轮需求还要求打开侧栏时正文不要留白。当前 [HtmlPreviews::sidebar](../crates/desktop/src/html_preview.rs) 为选中的 HTML 创建一个独立实例，原内联实例继续显示；侧栏使用独立 generation，避免它的测量或旧回调影响正文。两处渲染同一份源码，但 DOM、脚本和滚动状态各自独立，不承诺复制任意动画闭包或同步每个控件值。

这个边界直接影响滚动规则：内联预览属于聊天滚动链，侧栏是独立区域，因此不能把原来的“所有预览到边界都交给聊天”规则套在侧栏。当前侧栏根元素使用自己的垂直滚动条，边界事件不改变聊天位置。生命周期和滚动归属分别由 [html-preview 模型](../docs/architecture/tla/html-preview/README.md) 与 [html-scroll 模型](../docs/architecture/tla/html-scroll/README.md) 描述。

## 测试为什么通过，问题却没有消失

测试的入口决定它能发现哪一段错误。

| 验证 | 实际检查的内容 | 不能据此证明的内容 |
| --- | --- | --- |
| 原来的 JavaScript `wheel` 回调测试 | 事件进入函数后的主轴、边界与短内容行为 | WebKit 是否会投递 DOM 事件 |
| 新增的 `nativeWheel` 消息测试 | 不触发 DOM `wheel`，宿主到 iframe 的路径仍能交接短内容 | AppKit 是否调用监听器、坐标是否对应真实窗口 |
| 无窗口的 GPUI 测试 | 双实例挂载、generation、侧栏事件不推动正文等 | `NSWindow` 与 GPUI 原生子视图的对象层次 |
| TLA+ / TLC | 有限状态范围内的独占消费、短内容交接、侧栏隔离与最终完成 | 实际 Objective-C 接收者、浏览器事件投递和真实 IPC 活性 |
| 打包后的 release 实机操作 | 原生输入、浏览器、正文和窗口生命周期的实际组合 | 所有机器、任意页面及无限事件序列均无故障 |

新增 JavaScript 回归用例直接发送父窗口的 `nativeWheel` 消息，不调用 DOM `wheel` 回调。该用例在旧实现上观察到失败：期待增加一次外层消息，实际没有增加；修复后通过。后续又接通测试中的宿主、iframe 和 IPC，检查这条完整消息路径以及短侧栏不向正文转发。可运行 [html-preview-bootstrap.cjs](../crates/desktop/tests/html-preview-bootstrap.cjs)。

正文回归没有被无窗口测试自动复现；证据是用户报告、当前依赖的视图结构、错误转发代码，以及改正对象后的原生上下滚动对照。没有把新增的 TLA+ `outside` 故障注入说成“测试已经执行过错误的 Objective-C 调用”。它只是确认模型能够发现“区域外输入被消费却没有推动正文”的协议违例。

滚动模型只枚举三个层次（内部控件、文档、正文）、0–2 的范围、±1 的方向和 1–2 的位移，串行处理事件，并对处理与交付作弱公平性假设。检查通过说明这些边界内的协议自洽，不说明模型之外的投递假设已经成立。生命周期模型另有两个待处理回调的队列边界；二者都不是对原生应用的完备证明。

## 本次实际验证与重做方法

2026-09-18 的修复记录中，以下检查已执行并通过；本次写复盘核对了实现、测试文件和已有结果，没有重复构建或重新跑整套测试：

```sh
cargo test -p kcastle-desktop html_preview --locked
cargo clippy -p kcastle-desktop --all-targets -- -D warnings
node crates/desktop/tests/html-preview-bootstrap.cjs
just tla-check
just tla-self-test
just macos-app
```

针对 `html_preview` 的 Rust 检查通过 6 项测试；滚动模型的正常配置检查了 33,696 个不同状态，故障自检也检出了 `outside` 导致的 `NoSwallowedWheel` 违例。TLA+ 的预期反例是自检成功条件，不是正常配置验证失败。

最后一次启动确认使用优化的 release 包，包内程序与 `target/release/kcastle-desktop` 的 Mach-O UUID 均为 `018F22FD-2C5B-34AA-836A-EAFE7478F3DA`。此前原生操作工具一度报 `noWindowsAvailable`，那些失败调用没有计作成功验收；最终工具恢复后实际观察到：

1. 在 HTML 演示会话的预览外正文向上、向下滚动，聊天内容相应移动，滚动中跨过“匀速运动”预览边界。
2. 打开已有的 KV Cache 纯文本会话，在没有可见 HTML 时向上滚动，从第 9 节到第 8 节；反向滚动回到第 9 节。
3. 返回原 HTML 会话。用户随后明确反馈“现在可以了”。

这些事实支持本次修复有效，不证明 WebKit 内部手势锁定假说。正文与侧栏双实例、侧栏边界隔离有代码和自动化用例支持；上述最终原生步骤没有单独记录一轮完整侧栏边界验收，不把它追加成已观察事实。

以后改动原生输入桥接时，可以复用这组最小验收：正文双向滚动 → 短 HTML 内交接 → 长 HTML 内滚动及边界 → 侧栏边界不动正文 → 切到纯文本会话再双向滚动。监听器属于窗口，不能只测创建它的那一个 HTML 页面。

最有用的判断信号是：**如果局部处理逻辑反复“修好又复现”，先沿事件链寻找最后一个确实收到输入的边界；一旦拦截原事件，既要验证处理入口，也要验证最终接收对象。** 模型、单元测试和实机观察各自覆盖不同一段，不能相互冒充。
