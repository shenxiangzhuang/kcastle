---
type: bug
created: "2026-09-16"
verified: "2026-09-16"
scope: Kcastle / RaTeX SVG / GPUI 原生绘制
---

# 为什么 SVG 有彩色 Emoji，App 却显示黑色且被裁切？

## 两个根因

RaTeX 已把 Emoji 编码为 SVG 中的 PNG `<image>`。但 GPUI `svg()` 经 `SvgRenderer::render_alpha_mask` 只取 alpha，再统一染成文字颜色。此前测试只检查 SVG 内容、布局与生成结果，没有经过颜色丢失的绘制边界；独立 SVG 完全正确也不能证明 App 正常。

另一个问题来自位图边界：15px 的 `\text{😀}` 生成的 viewBox 高度为 12.975，图像 y 为 0.7875、高度为 15，下边缘 15.7875 超出画布。较大的实验 padding 会掩盖这个问题。

修复前，`math_mask_excludes_raster_emoji` 明确失败；独立 15px SVG 的边界断言也失败。这两项分别复现颜色丢失和裁切。

## 修复与所有权

- 纯矢量公式保留原始 SVG。含位图的公式分为主题着色的矢量蒙版和透明彩色图层，两层使用相同 viewBox 和原始图像坐标/透明度。
- 画布包含位图及 padding，新增顶部空间同步计入基线。布局使用扩展后的宽高，避免小字号 Emoji 裁切。
- 彩色图层在现有准备 worker 中解码一次，`Arc<RenderImage>` 随 presentation 持有；BGRA 字节计入现有缓存预算，不进入全局 image resource cache。
- 不新增 worker、取消或发布状态转换。架构和 TLA+ 模型映射说明同步更新。

解析依赖固定版本 RaTeX 输出的平铺 SVG 元素；若上游改成嵌套元素，明确返回错误而非静默丢失变换。公式的复杂 Emoji 序列支持仍受 RaTeX/系统字体限制，本次不改变字符布局算法。

## 当前源码 App 验收

在 Apple M2 / macOS 26.6.2 上，从代码提交 `6c40b0a` 构建 release，并通过 `scripts/package-macos-app` 打包签名。验收副本仅修改名称、Bundle ID、可执行文件名及隔离数据目录，未使用 `/Applications/Kcastle.app` 的旧安装包。

源码构建二进制、正式 `target/Kcastle.app` 和实际启动的 `Kcastle Emoji.app` 的链接 UUID 与全部 19 个 file-backed Mach-O section 哈希相同。UUID：`a8933dbd51b03129ac9a4f130ea6b885`，见 [构建身份](assets/2026-09-16-formula-emoji/binary-identity.json)。

实际点击 Emoji 会话和设置，切换浅色/深色主题，再切回中文会话；通过原生窗口缩放触发可靠重绘，截图确认：

- 显示公式、表格内 Emoji 完整且有颜色，混合中文/Emoji 正常。
- 数学符号在浅色主题为深色、深色主题为亮色；Emoji 颜色保持不变。
- 表格分数、Rust/Python 代码、引用/列表和积分仍正常，无新增可见回归。
- 会话切换后重新准备和窗口尺寸变化正常。

图像：[浅色 Emoji](assets/2026-09-16-formula-emoji/light-emoji.jpg) · [深色 Emoji](assets/2026-09-16-formula-emoji/dark-emoji.jpg) · [中文、积分与代码](assets/2026-09-16-formula-emoji/cjk-code.jpg)。这些是原生截图，不是独立 SVG 预览。

主题切换后的 App 物理内存快照为 71.6M，峰值 79.0M；它是运行阶段相关的快照，不把它与前次 65.0M 的全部差额解释成新增图像内存。

验证：285 项 desktop release 测试通过，1 项按原配置忽略；release Clippy all-targets、格式检查、TLA+ 检查和自测通过；release 构建、签名验证通过。测试覆盖实际彩色像素、图层位置/透明度、边界与基线、纯矢量不变及图像字节计量，原有布局与视口测试继续通过。

## Computer Use 故障调查

同一工具会话中，Kcastle 的坐标点击和滚轮返回 `Computer Use server error -10005: noWindowsAvailable`，但能读取 AX 树和窗口截图。独立对照的系统 Calculator 也出现相同坐标错误；其 AX 按钮点击能立即把 0 改为 7，再恢复为 0。这排除了“仅 Kcastle 没有导出可访问窗口”的解释。

本机 `SkyComputerUseService-2026-09-16-094851.ips` 记录工具服务在 09:48:30 发生 `EXC_BAD_ACCESS / SIGSEGV`，空地址访问；故障进程是 SkyComputerUseService，不是 Kcastle。这与此前 native pipe 关闭一致。仅凭崩溃报告无法进一步确定工具服务内部的错误语句。

工具服务的 ScreenCaptureKit 日志还显示该阶段大量窗口 `isOnScreen=0`。GPUI 0.3.3 的 `start_display_link` / `window_did_change_occlusion_state` 会在窗口不可见时停止帧循环，而原生缩放会请求同步帧，符合“AX 操作后需缩放才能看到画面”的现象。当前诊断读到 `IOConsoleLocked=false`，因此不把锁屏当作已证实原因。窗口可见性被外部环境/工具如何影响仍未确定。

结论：已证实的失败发生在跨 App 的工具窗口识别/输入通道及工具服务崩溃中，没有发现可复现的 Kcastle 代码根因，因此未为它创建应用代码修复 PR，也未强制后台重绘绕过平台节能行为。候选 App 的真实滚轮验收仍未完成；已通过的 GPUI 滚动测试不能冒充该项原生验收。完整覆盖仍需 Computer Use 窗口输入通道恢复后复验。

临时原始日志与复现材料位于 `/tmp/kcastle-emoji-verification/`、`/tmp/kcastle-cua-system.log` 及 `/tmp/kcastle-emoji-*.log`。公开记录只保留相关诊断结论，不提交其他应用的日志或窗口内容。
