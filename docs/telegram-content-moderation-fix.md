# Fixing Telegram Content Moderation Errors

## Problem
Some AI providers (especially OpenRouter) have aggressive content moderation that flags legitimate technical discussions. Once a session is flagged, all subsequent messages in that session may be blocked with "Content Exists Risk" errors.

## Quick Fix: Create a New Session
The easiest solution is to start fresh with a new session:
```
/new
```

## Automatic Recovery (Future Enhancement)
We could add auto-recovery when content moderation errors occur:

```python
# In telegram.py _render_events_to_text
case AgentError(error=err):
    error_msg = str(err)
    if "Content Exists Risk" in error_msg:
        # Auto-create new session
        base_sid = ... # get user's base session ID
        new_sid = f"{base_sid}-{int(time.time())}"
        self._castle.session_manager.create(session_id=new_sid, name="auto-recovery")
        self._active_sessions[base_sid] = new_sid

        parts.append(
            "\n⚠️ Content moderation triggered. "
            "I've automatically created a new session. "
            "Please resend your message."
        )
```

## Preventing Content Moderation

### 1. Use Direct API Providers
OpenRouter applies stricter filters than direct providers:
- ❌ OpenRouter (strict moderation)
- ✅ Anthropic API (nuanced moderation)
- ✅ OpenAI API (reasonable moderation)

### 2. Configure Alternative System Prompt
If the default identity triggers moderation, use a simpler prompt:

```yaml
# ~/.kcastle/config.yaml
agent:
  system_prompt: "You are a helpful AI assistant."
```

### 3. Use Different Models for Different Topics
Some models have stricter filters:
```
# For general chat
/model anthropic claude-3-haiku-20240307

# For technical/security discussions
/model openai gpt-4-turbo-preview
```

## Why This Happens

OpenRouter's content moderation may flag:
- Security-related terms (exploit, vulnerability, hack)
- Code that resembles malware patterns
- Discussions about bypassing restrictions
- Technical jargon that seems suspicious

The moderation happens at the provider level, before the AI model sees the request.

## Long-term Solutions

1. **Session Isolation**: Each new session starts with a clean slate
2. **Provider Selection**: Choose providers with appropriate moderation levels
3. **Content Filtering**: Pre-filter messages to avoid trigger words
4. **Fallback Providers**: Automatically switch providers on moderation errors