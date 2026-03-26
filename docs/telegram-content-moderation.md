# Handling Content Moderation Errors in Telegram

If you're getting "Content Exists Risk" errors from the Telegram bot, this is due to the AI provider's content moderation filters blocking the request.

## Quick Solutions

1. **Clear session history**: Use `/clear` command to keep the same session but clear history
2. **Start a new conversation**: Use `/new` command to create a completely new session
3. **Rephrase your message**: Try expressing the same idea differently
4. **Switch to a different model**: Some models have less strict filters

## Switching Models

You can configure a different provider/model for better compatibility:

### Option 1: Use Claude via Anthropic (recommended)
```yaml
# ~/.kcastle/config.yaml
providers:
  anthropic:
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    models:
      - id: claude-3-haiku-20240307
        active: true
      - id: claude-3-sonnet-20240229
        active: true

default_provider: anthropic
default_model: claude-3-haiku-20240307
```

### Option 2: Use OpenAI GPT models
```yaml
# ~/.kcastle/config.yaml
providers:
  openai:
    provider: openai
    api_key: ${OPENAI_API_KEY}
    models:
      - id: gpt-4-turbo-preview
        active: true
      - id: gpt-3.5-turbo
        active: true

default_provider: openai
default_model: gpt-3.5-turbo
```

### Option 3: Configure a different model for specific sessions

If the error persists with certain topics, you can manually switch models for that session in the CLI:
```bash
k -S <session-id>  # Resume the session
/model anthropic claude-3-haiku-20240307  # Switch to a different model
```

## Why This Happens

Some AI providers (especially router services like OpenRouter) apply strict content filters that may flag:
- Technical security discussions
- Code with certain patterns
- Medical or legal topics
- Historical or political content

Different models and providers have different moderation policies. Direct API providers (Anthropic, OpenAI) typically have more nuanced moderation compared to router services.

## Long-term Solution

Consider using:
1. Direct API access (Anthropic/OpenAI) instead of router services
2. Self-hosted models for sensitive technical work
3. Different models for different types of conversations