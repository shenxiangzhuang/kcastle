use async_openai::Client;
use async_openai::config::OpenAIConfig;
pub use async_openai::types::responses::ReasoningEffort;

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
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
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
            reasoning_effort: None,
        }
    }

    pub fn with_reasoning(
        mut self,
        reasoning_efforts: &'static [ReasoningEffort],
        reasoning_effort: ReasoningEffort,
    ) -> Self {
        assert!(reasoning_efforts.contains(&reasoning_effort));
        self.reasoning_efforts = reasoning_efforts;
        self.reasoning_effort = Some(reasoning_effort);
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

    pub fn reasoning_effort(&self) -> Option<&ReasoningEffort> {
        self.reasoning_effort.as_ref()
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: ReasoningEffort) {
        assert!(self.reasoning_efforts.contains(&reasoning_effort));
        self.reasoning_effort = Some(reasoning_effort);
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
        configured.reasoning_effort = self.reasoning_effort.clone();
        configured.max_output_tokens = self.max_output_tokens;
        configured
    }
}
