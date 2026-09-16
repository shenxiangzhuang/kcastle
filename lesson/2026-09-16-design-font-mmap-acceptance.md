---
type: design
created: "2026-09-16"
verified: "2026-09-16"
scope: Kcastle 0.2.0-alpha.25 / macOS 字体映射与原生 App 对照
---

# 字体文件映射：从独立公式实验到当前源码 App

## 结果与边界

在 Apple M2、macOS 26.6.2 (25G83)、rustc 1.97.1 上，将只读系统字体从整份堆副本改为文件映射。基线是提交 `1269183` 的独立 release 构建，候选是当前工作区 release 构建；不是拿 `/Applications/Kcastle.app` 的旧安装包代替基线或候选。

同一隔离 Emoji/中文混合公式会话，在 1180 × 720 窗口中，基线 `vmmap` Physical footprint 为 **287.0M**，初轮候选为 **67.8M**；最后兼容性调整后重建并再次点击该会话，最终候选为 **65.0M**，峰值 **72.7M**。这些是不同进程的原生 App 快照，不是承诺所有会话节省同样数值。图形缓冲和运行阶段也会影响总量，不能把全部约 220M 差额都归给字体。

独立公式实验提供更直接的因果证据：释放本次 AST、布局和 SVG 后，中文公式的 Rust 活跃堆增量由 22.493 MiB 降为 0.293 MiB，Emoji 公式由 205.714 MiB 降至约 0.291 MiB。普通 `x` 保持 0.323 MiB。最终代码的 9 组 SVG 与基线逐字节一致，且每组保留堆均小于 1 MiB。

## 所有权与安全边界

`Arc<FontData>` 持有 owned bytes 或只读映射；调用者仍只借用 `&[u8]`。映射不会在借用期间被释放，主/备用字体默认共享同一个 Arc 的行为不变，TTC face index 不变。

仅在 macOS 同时满足两个条件时映射：

1. 文件规范路径位于 `/System/Library/Fonts` 下。
2. 对已打开文件描述符调用 `fstatfs`，确认所在文件系统具有 `MNT_RDONLY` 标志。

这里信任 OS 的只读系统卷在进程期间不会被改为可写；仅检查文件名、权限位或“它是系统发现的字体”不足以保证映射安全。可修改的自定义字体及其他平台保留内存副本，映射失败也回退读取。先打开原始路径，规范化仅用于判断是否可以映射；规范化失败不应使原本能读取的字体失败。

Emoji 仍由 fontdb 选择文件和 face index；文件来源通过同一所有者加载，避免此前 `data.to_vec()` 的整份复制。内存来源或文件加载失败仍保留原来的字节复制回退。

最终 App 中可看到 22.2M 的 Arial Unicode 和 183.2M 的 Emoji 文件映射，所访问的干净页只占其中一小部分。原生文字系统还可能持有另一份文件映射；虚拟映射大小不能当作独占物理占用相加。

## 构建身份与界面验收

`target/Kcastle.app` 由当前源码通过 `cargo build --release --locked -p kcastle-desktop` 和 `scripts/package-macos-app --binary target/release/kcastle-desktop` 构建、签名并验证。

为避免本机旧 App 的同名注册干扰原生自动化，验收副本使用独立名称、Bundle ID、可执行文件名和 `KCASTLE_DATA_DIR`。这些只修改临时副本的包装元数据；正式 `target/Kcastle.app` 保留正常名称和标识。隔离会话由本地 loopback Responses 夹具通过真实 Agent/Session API 生成，没有复制凭据，也没有向外部模型发送消息。

最终 release 可执行文件、`target/Kcastle.app` 和验收副本的链接 UUID，以及全部 **19 个有文件内容的 Mach-O section 的 SHA-256** 完全一致。UUID：`b3891911a27e3330a0cd6cda7bd7fa28`。完整文件哈希可能因重新签名而不同，所以验证代码/常量 section，而不把签名变化当成代码变化。证据见 [binary-identity.json](assets/2026-09-16-font-mmap/binary-identity.json)。

实际操作与结果：

- 点击中文会话，检查中文公式、命令符号、希腊字母、分数、根号、求和、表格及 Python/Haskell 代码。
- 点击 Emoji 会话，检查公式内 Emoji、中文混合 Emoji、表格分数与 Rust 代码。
- 点击 Trajectory 与 Request #1，确认详情面板可打开、状态和布局与基线相同。
- 切换回中文会话，确认重新准备后的公式和表格仍能显示；执行原生窗口尺寸切换检查重排。
- 初轮候选的 Emoji/表格/代码正文区域 `(350,85)-(1110,565)` 与基线截图 **0 个不同像素**；Trajectory 区域 `(345,75)-(1175,595)` 同样为 **0**。排除了侧栏时间、窗口按钮和光标等非渲染差异，原始 JPEG 未进行图像编辑。见 [pixel-comparison.json](assets/2026-09-16-font-mmap/pixel-comparison.json)。
- 最后的规范化回退调整后重新构建，替换验收副本的可执行文件，再次启动、点击 Emoji 会话、检查最终画面和内存，并重新比对 9 组 SVG。最终截图的侧栏/内容水平位置与初轮不同，不宣称最终截图与初轮整图逐像素一致。

### 工具限制

CUA 坐标点击/滚轮多次返回 `noWindowsAvailable`，并曾出现 native pipe 关闭。可访问性点击与原生窗口缩放仍可执行，因此上述候选界面验收使用这些操作，并以截图确认真实结果。部分后台准备结果需要原生窗口刷新后才能通过该通道可靠捕获。

基线曾成功执行滚轮；**候选版本的真实窗口滚轮验收未完成**。滚动、离屏淘汰、取消和会话切换仍有通过的 GPUI 集成测试，但不把这些测试冒充原生滚轮验收。曾请求授权 AppleScript 作为替代，未收到授权，也未使用 AppleScript。本记录只对已覆盖场景报告未发现新增渲染差异，不保证所有 Unicode/平台/交互组合都没有问题。

### 已存在的 Emoji 显示问题

基线 App 的公式内 Emoji 显示为黑色形状，正文/代码中的 Emoji 正常。候选与基线相同，SVG 字节也一致，因此这是已有的公式显示问题，不是映射改动引入的回归。本次没有更换 SVG 绘制链路，也不把该现象标记为“彩色公式 Emoji 渲染正常”。

## 验证记录

- 工作区 release 测试：agent 89 项、desktop 282 项通过，3 项按原配置忽略。
- 最终字体依赖测试：9 项通过，包括可写字体经符号链接加载后被覆盖/删除仍保持快照、只读系统字体映射与文件内容一致、主/备用共享、缺字回退。
- 最终字体依赖 Clippy `--all-targets -- -D warnings` 通过；工作区与 vendored 源码格式、`git diff --check` 通过。
- 最终 9 组 SVG：ASCII、分数、希腊字母、中文、命令符号、Emoji、中文混合 Emoji、求和、轮廓笑脸，与原始基线逐字节一致。
- `just tla-check`、`just tla-self-test` 均通过。架构与模型说明已更新；字体存储不改变 worker、取消或结果发布状态转换，未修改模型状态机。
- release App 打包和 `codesign --verify --deep --strict` 通过。

## 截图

[基线 Emoji/表格/代码](assets/2026-09-16-font-mmap/baseline-emoji-bottom.jpg) · [初轮候选同一区域](assets/2026-09-16-font-mmap/latest-emoji-bottom.jpg) · [最终代码复验](assets/2026-09-16-font-mmap/final-emoji-bottom.jpg) · [中文与求和](assets/2026-09-16-font-mmap/latest-cjk-return.jpg) · [Trajectory 详情](assets/2026-09-16-font-mmap/latest-trajectory.jpg)

详细临时材料保存在 `/tmp/kcastle-mmap-experiment/`：两个源码构建、独立公式程序、loopback 夹具、SVG、`vmmap` 原始输出和构建/测试日志。它们是本轮诊断材料，不是持久项目依赖。
