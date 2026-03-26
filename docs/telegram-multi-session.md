# Telegram Multi-Session Support

The Telegram bot now supports multiple independent sessions per user. This allows you to have different conversations with different contexts without them interfering with each other.

## How It Works

### Session IDs
- **Private chats**: Sessions are identified by `tg-u{user_id}` (base ID)
- **Group chats**: Sessions are identified by `tg-g{chat_id}` (base ID)
- **New sessions**: Get unique IDs with timestamps like `tg-u12345-1703456789`

### Commands

#### `/new [name]` - Create a New Session
Creates a fresh session with optional name:
```
/new
✓ New session started: tg-u12345-1703456789

/new project planning
✓ New session started: tg-u12345-1703456790
```

#### `/sessions` - List Your Sessions
Shows all your sessions with timestamps and active marker:
```
/sessions

**Your sessions:**

• `tg-u12345` — default [default]
• `tg-u12345-1703456789` — [2023-12-24 15:30]
• `tg-u12345-1703456790` — project planning [2023-12-24 15:32] ⬅️ (active)

Use `/switch <session-id>` to switch sessions
Use `/new [name]` to create a new session
```

#### `/switch <session-id>` - Switch Sessions
Switch to a different session:
```
/switch tg-u12345-1703456789
✓ Switched to session: `tg-u12345-1703456789`
```

#### `/model` - Change Model
Changes the model for the currently active session only.

## Features

### Session Persistence
- Sessions persist across bot restarts
- Each session maintains its own conversation history
- Model settings are per-session

### Auto-Detection
After bot restart, when you send a message:
1. Bot checks if you have an active session tracked
2. If not, it automatically uses your most recently active session
3. No need to manually switch after restarts

### Session Isolation
- Each session has completely independent context
- Switching sessions is instant
- No cross-contamination between sessions

## Use Cases

### Multiple Projects
```
/new work project
[discuss work tasks...]

/new personal assistant
[discuss personal tasks...]

/switch tg-u12345-work
[back to work context]
```

### Different Models for Different Tasks
```
/new coding assistant
/model anthropic claude-3-haiku-20240307
[use for coding tasks]

/new creative writing
/model openai gpt-4-turbo-preview
[use for creative tasks]
```

### Testing and Debugging
```
/new test session
[test something risky]

/switch tg-u12345
[back to main session, test session isolated]
```

## Technical Details

### Session ID Structure
- Base ID: Identifies the user/chat
- Timestamp suffix: Makes each session unique
- Example: `tg-u12345-1703456789`
  - `tg-u12345`: User 12345's base ID
  - `1703456789`: Unix timestamp when created

### Active Session Tracking
- Bot maintains a mapping of base ID → active session ID
- Persists in memory during bot lifetime
- Auto-detects most recent on restart

### Storage
- Sessions stored in kcastle's session directory
- Each session has its own `trace.jsonl` and `meta.json`
- Standard kcastle session format

## Troubleshooting

### "Session not found" Error
- Use `/sessions` to see exact session IDs
- Ensure you're copying the full ID including timestamp

### Lost Active Session
- Bot auto-detects most recent session
- Or use `/sessions` and `/switch` to manually select

### Want to Delete Old Sessions
Currently, old sessions need to be manually cleaned from the filesystem.
Future versions may add a `/delete` command.