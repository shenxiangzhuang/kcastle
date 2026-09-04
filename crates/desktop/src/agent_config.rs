use kcastle_agent::{Model, ReasoningEffort, SessionModelConfig};

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

#[derive(Clone)]
pub(crate) struct ConfiguredModel {
    pub(crate) id: String,
    pub(crate) model: Model,
    pub(crate) provider_id: String,
    pub(crate) profile: ProviderModel,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
}

impl ConfiguredModel {
    pub(crate) fn new(
        provider_id: impl Into<String>,
        profile: ProviderModel,
        model: Model,
    ) -> Self {
        let provider_id = provider_id.into();
        let reasoning_effort = default_reasoning_effort(&provider_id);
        Self {
            id: format!("{provider_id}/{}", profile.model_id),
            model,
            provider_id,
            profile,
            reasoning_effort,
        }
    }

    pub(crate) fn label(&self) -> String {
        let model = if self.profile.display_name.trim().is_empty() {
            &self.profile.model_id
        } else {
            &self.profile.display_name
        };
        format!("{} · {model}", self.model.name())
    }

    pub(crate) fn session_model_config(&self) -> SessionModelConfig {
        SessionModelConfig {
            model_id: Some(self.id.clone()),
            reasoning_effort: self.reasoning_effort.as_ref().map(reasoning_key),
        }
    }
}

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

fn reasoning_key(effort: &ReasoningEffort) -> String {
    serde_json::to_value(effort)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| format!("{effort:?}").to_lowercase())
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
    use super::*;

    #[test]
    fn configured_model_owns_its_session_selection() {
        let profile = ProviderModel::new("gpt-test", "GPT Test", 10_000, None);
        let configured = ConfiguredModel::new(
            OPENAI_PROVIDER_ID,
            profile,
            Model::new("OpenAI", "key", "https://example.com", "gpt-test", 10_000),
        );

        assert_eq!(
            configured.session_model_config(),
            SessionModelConfig {
                model_id: Some("openai/gpt-test".into()),
                reasoning_effort: Some("medium".into()),
            }
        );
    }

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
