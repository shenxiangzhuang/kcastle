use kcastle_agent::{Model, ReasoningEffort};

use crate::settings::{ProviderModel, ProviderProfile};

pub(crate) const INSTRUCTIONS: &str = "You are Kcastle, a concise coding agent. Use the shell tool when it helps. Inspect before editing, report tool errors honestly, and stop when the task is complete.";
pub(crate) const DEEPSEEK_PROVIDER_ID: &str = "deepseek-official";
pub(crate) const OPENAI_PROVIDER_ID: &str = "openai";

const DEEPSEEK_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::None,
    ReasoningEffort::Low,
    ReasoningEffort::High,
];
const OPENAI_REASONING_EFFORTS: &[ReasoningEffort] = &[
    ReasoningEffort::None,
    ReasoningEffort::Low,
    ReasoningEffort::Medium,
    ReasoningEffort::High,
    ReasoningEffort::Xhigh,
];

const DEEPSEEK_MODELS: &[(&str, &str, usize)] = &[
    ("deepseek-v4-flash", "DeepSeek-V4-Flash", 1_000_000),
    ("deepseek-v4-pro", "DeepSeek-V4-Pro", 1_000_000),
];
const OPENAI_MODELS: &[(&str, &str, usize)] = &[
    ("gpt-5.6-sol", "GPT-5.6 Sol", 1_050_000),
    ("gpt-5.6-terra", "GPT-5.6 Terra", 1_050_000),
    ("gpt-5.6-luna", "GPT-5.6 Luna", 1_050_000),
];

#[allow(
    clippy::unreachable,
    reason = "callers use the closed built-in provider ids"
)]
pub(crate) fn default_provider_profile(provider_id: &str) -> ProviderProfile {
    let (display_name, api_base, models) = match provider_id {
        DEEPSEEK_PROVIDER_ID => ("DeepSeek", "https://api.deepseek.com", DEEPSEEK_MODELS),
        OPENAI_PROVIDER_ID => ("OpenAI", "https://api.openai.com/v1", OPENAI_MODELS),
        _ => unreachable!("unsupported provider: {provider_id}"),
    };
    ProviderProfile::new(
        provider_id,
        display_name,
        api_base,
        models
            .iter()
            .map(|(id, name, context)| ProviderModel::new(*id, *name, *context, None))
            .collect(),
    )
}

pub(crate) fn build_model(
    provider: &ProviderProfile,
    profile: &ProviderModel,
    api_key: String,
) -> Model {
    let model = Model::new(
        provider.display_name.clone(),
        api_key,
        provider.api_base.clone(),
        profile.model_id.clone(),
        profile.context_window,
    )
    .with_max_output_tokens(profile.max_output_tokens);
    match provider.provider_id.as_str() {
        DEEPSEEK_PROVIDER_ID | "deepseek" => {
            model.with_reasoning_efforts(DEEPSEEK_REASONING_EFFORTS)
        }
        OPENAI_PROVIDER_ID => model.with_reasoning_efforts(OPENAI_REASONING_EFFORTS),
        _ => model,
    }
}

pub(crate) fn default_reasoning_effort(provider_id: &str) -> Option<ReasoningEffort> {
    match provider_id {
        DEEPSEEK_PROVIDER_ID | "deepseek" => Some(ReasoningEffort::High),
        OPENAI_PROVIDER_ID => Some(ReasoningEffort::Medium),
        _ => None,
    }
}

pub(crate) fn initial_session_title(input: &str) -> Option<String> {
    let normalized = input.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = normalized.chars();
    let title = chars.by_ref().take(48).collect::<String>();
    (!title.is_empty()).then(|| {
        if chars.next().is_some() {
            format!("{title}…")
        } else {
            title
        }
    })
}

#[cfg(test)]
mod tests {
    use super::initial_session_title;

    #[test]
    fn initial_title_is_normalized_and_bounded() {
        assert_eq!(
            initial_session_title("  hello\n world  ").as_deref(),
            Some("hello world")
        );
        assert_eq!(
            initial_session_title(&"a".repeat(49)).as_deref(),
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa…")
        );
        assert_eq!(initial_session_title(" \n\t "), None);
    }
}
