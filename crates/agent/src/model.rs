use async_openai::Client;
use async_openai::config::OpenAIConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
        }
    }
}

impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Concrete model configuration for the OpenAI-compatible Responses API.
///
/// Product-owned provider catalogs and defaults live in the desktop crate. The harness only
/// retains the capabilities required to build and execute a request.
#[derive(Clone)]
pub struct Model {
    name: String,
    api_key: String,
    api_base: String,
    pub(crate) client: Client<OpenAIConfig>,
    pub(crate) model: String,
    context_window: usize,
    pub(crate) max_output_tokens: Option<u32>,
    reasoning_efforts: &'static [ReasoningEffort],
}

impl Model {
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        api_base: impl Into<String>,
        model: impl Into<String>,
        context_window: usize,
    ) -> Self {
        let api_key = api_key.into();
        let api_base = api_base.into();
        let config = OpenAIConfig::new()
            .with_api_key(api_key.clone())
            .with_api_base(api_base.clone());
        Self {
            name: name.into(),
            api_key,
            api_base,
            client: Client::with_config(config),
            model: model.into(),
            context_window,
            max_output_tokens: None,
            reasoning_efforts: &[],
        }
    }

    pub fn with_reasoning_efforts(mut self, reasoning_efforts: &'static [ReasoningEffort]) -> Self {
        self.reasoning_efforts = reasoning_efforts;
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    pub fn has_api_key(&self) -> bool {
        !self.api_key.trim().is_empty()
    }

    pub fn context_window(&self) -> usize {
        self.context_window
    }

    pub fn max_output_tokens(&self) -> Option<u32> {
        self.max_output_tokens
    }

    pub fn with_max_output_tokens(mut self, max_output_tokens: Option<u32>) -> Self {
        self.max_output_tokens = max_output_tokens;
        self
    }

    pub fn reasoning_efforts(&self) -> &'static [ReasoningEffort] {
        self.reasoning_efforts
    }

    pub fn reconfigured(
        &self,
        name: impl Into<String>,
        api_key: Option<String>,
        api_base: impl Into<String>,
        model: impl Into<String>,
        context_window: usize,
    ) -> Self {
        let mut configured = Self::new(
            name,
            api_key.unwrap_or_else(|| self.api_key.clone()),
            api_base,
            model,
            context_window,
        );
        configured.reasoning_efforts = self.reasoning_efforts;
        configured.max_output_tokens = self.max_output_tokens;
        configured
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model::new("test", "key", "https://example.com", "test", 1)
    }

    #[test]
    fn reconfiguration_preserves_static_reasoning_capabilities() {
        let model = model().with_reasoning_efforts(&[ReasoningEffort::Low]);
        let reconfigured = model.reconfigured("new", None, "https://new", "new", 2);
        assert_eq!(reconfigured.reasoning_efforts(), &[ReasoningEffort::Low]);
    }
}
