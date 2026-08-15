# Desktop interaction QA

Last exercised on macOS with the bundled release application and Computer Use.

## Verified flows

- Click the composer, enter text, and send with either Enter or the send button.
- Receive streamed Markdown and live token/tool counters.
- Stream a response longer than the viewport and verify its final line stays visible above the
  composer while tail-following is active.
- Show a shell approval, deny it, and continue the agent run.
- Stop an agent while it is waiting for tool approval, then continue typing without refocusing.
- Expand and collapse a tool result.
- Inspect the full chronological ledger with the Trajectory tab and filter it by role, tool, or
  content.
- Search the active workspace's sessions and switch their chronological ordering.
- Copy user and assistant messages.
- Switch model and reasoning effort directly from the composer without navigating to Settings.
- Open Settings and verify its supported General and Models pages; unsupported plugin and agent
  preset pages are not rendered.
- Export the current JSONL session through the native Save panel, including canceling without a
  side effect.
- Start an in-memory new chat without creating an empty session file or a synthetic sidebar row;
  the session appears in history only after the first non-empty submission.
- Open a persisted session and restore its transcript, latest usage, and failed tool state.
- Collapse and restore the sidebar.
- Add a project with the native folder picker, switch between projects, and verify their session
  lists remain isolated.
- Hover a workspace row, open its note action, and verify it activates an in-memory new-session
  page without adding a history row until the first non-empty submission.
- Rename and delete a session, including canceling the destructive confirmation first.
- Remove a project from the sidebar, including canceling first, without deleting its files or
  session directory.
- Open Settings, change reasoning effort, close it, and retain the preference across a restart.
- Change System/Light/Dark appearance and verify all primary surfaces use the active theme.
- Resize below 1024 px and verify the sidebar becomes a recoverable 56 px rail.
- Scroll a long Trajectory result independently to its conclusion while the ledger stays fixed.
- Compare Trajectory in both appearances against DSH tokens: business-blue user records,
  violet assistant records, amber tools, green context, neutral active rows, blue active tabs,
  and DSH JSON property/string/keyword syntax colors.
- Click user, context, reasoning, assistant, and tool trajectory cells and verify the inspector
  opens with type-specific tabs. Markdown records use Summary/Preview/Raw; tools use
  Summary/Payload/Result/Schema/Timing, and Summary section headings open their matching tab.
- Verify the trajectory inspector header exposes only Close: the three width adjustment buttons
  are absent, while the divider remains draggable on wide windows.

## DeepSeek Harness alignment pass

- Walked the running DSH Web UI at `127.0.0.1:3080` through the empty state, active chat, expanded
  shell output, workspace search, workspace ordering, General/Models settings, and Trajectory.
- Replaced the separate project/session lists with a workspace-grouped session tree using DSH's
  280px sidebar, 34px workspace rows, and 32px session rows.
- Matched DSH's shared content axis: 748px transcript, 780px composer, 525px user bubbles, 22px
  input and bubble radii, centered empty composer, and docked active composer.
- Removed the empty-session header and switched it to the same centered hero-to-docked-composer
  transition used by DSH.
- Replaced the former tool-only Activity filter with a searchable chronological Trajectory view.
- Expanded Settings into a two-column General/Models surface and added a working session-log
  export action.

## Defects found and fixed

- The 1320×860 default window could extend past a laptop-sized display and leave bottom controls
  outside the clickable window bounds; the centered default is now 1180×720.
- The header marked its entire row as a native drag region, so visible child buttons could hover
  but never receive clicks. Dragging is now limited to the title areas.
- Persisted sessions reset token usage to zero when opened.
- Denied/interrupted/non-zero shell calls were displayed as Completed after reopening.
- The first send could race with a second click while its session file was being created.
- A submitted user message did not immediately scroll into view.
- `gpui-component` uses a trailing 200 ms debounce for updates to an existing Markdown `TextView`:
  every incoming text change restarts that timer. During a continuous stream the rendered document
  and its measured height could therefore remain stale indefinitely, so the old 1 ms scroll retry
  left the actual result below the composer once parsing finally caught up. Streaming now
  follows DSH's append-only pipeline: publish visible deltas once per actual GPUI display frame,
  freeze stable Markdown blocks, re-parse only the trailing two-block frontier, and defer
  open-fence highlighting until settlement. Stable blocks keep their keys, the changing tail
  mounts synchronously, and the pre-layout scroll pin no longer needs a second corrective frame.
  A real upward wheel delta still
  cancels following, and the fixed overlay restores it on demand.
- The composer was fixed-height and could cover transcript/trajectory content. It now auto-grows
  from 1–14 lines inside the flex layout: Enter submits, secondary Enter inserts a newline, and
  focus is restored after every run.
- The Trajectory virtual list could escape its flex allocation and push the composer below the
  window even though the source tree placed it after the content pane. The trajectory root now
  owns overflow clipping and the composer remains visible in the release app.
- Workspace folder/chevron states previously occupied two horizontal slots, and session titles
  competed with a variable-width age/action suffix. Both rows now use fixed optical slots; long
  titles show an explicit width-aware ellipsis and time swaps in place with the action button.
- Switching projects previously rendered the in-memory draft as a selected `New Session · now`
  history item. The sidebar now lists persisted sessions only; first submission remains the point
  where the draft is created and added to history.
- Trajectory previously inherited the generic GPUI chart and selected-button colors, collapsing
  most record roles to blue and rendering active tabs as dark pills. It now uses DSH's dedicated
  light/dark trajectory tokens, including its JSON syntax palette and neutral shell-result color.
- Trajectory previously gave every record the tool-only five-tab inspector, labeled user inputs as
  Step 1, and exposed three width buttons in the header. Details are now cell-kind driven, message
  rows use Message/Step locations correctly, tool schemas render name/description/parameters, and
  only Close remains in the header. Removing the horizontal-scroll wrapper also keeps every
  applicable tab visible instead of clipping the strip after Summary.
- Attachment, search, settings, overflow, and session-info affordances looked clickable but had no behavior.
- Session lists stopped scanning at the first message record, so a title renamed after chatting
  reverted in the sidebar after switching projects.
- `gpui-component` 0.5.1 exposes dialog layer helpers but its `Root` does not attach them to its
  render tree; settings and confirmations now use an application modal layer built from its
  Button and Input components.
- The former UI kept the same header and bottom composer in every state, split workspaces from
  their sessions, capped content at 900px, and exposed no search, trajectory search, message copy,
  or log export. Those structural gaps were fixed instead of only retheming the old layout.

## Automation limitation

GPUI 0.2.2 currently exposes the application window but not its custom controls in the macOS
accessibility tree. Computer Use therefore has to use coordinates to focus the composer before it
can send text or key events. This does not affect physical keyboard input, but it prevents semantic
accessibility automation until GPUI exposes those nodes.
