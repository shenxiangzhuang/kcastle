use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use kcastle_agent::{DEEPSEEK_PROVIDER_ID, Model, ReasoningEffort};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProviderModel {
    pub(crate) model_id: String,
    pub(crate) display_name: String,
    pub(crate) context_window: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u32>,
}

impl ProviderModel {
    pub(crate) fn new(
        model_id: impl Into<String>,
        display_name: impl Into<String>,
        context_window: usize,
        max_output_tokens: Option<u32>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            display_name: display_name.into(),
            context_window,
            max_output_tokens,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProviderProfile {
    pub(crate) provider_id: String,
    pub(crate) display_name: String,
    pub(crate) api_base: String,
    #[serde(default)]
    pub(crate) models: Vec<ProviderModel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
    #[serde(default, rename = "model_id", skip_serializing)]
    legacy_model_id: Option<String>,
    #[serde(default, rename = "context_window", skip_serializing)]
    legacy_context_window: Option<usize>,
}

impl ProviderProfile {
    pub(crate) fn new(
        provider_id: impl Into<String>,
        display_name: impl Into<String>,
        api_base: impl Into<String>,
        models: Vec<ProviderModel>,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            display_name: display_name.into(),
            api_base: api_base.into(),
            models,
            api_key: None,
            legacy_model_id: None,
            legacy_context_window: None,
        }
    }

    pub(crate) fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Appearance {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum EnterBehavior {
    #[default]
    Steer,
    Queue,
}

#[derive(Default, Serialize, Deserialize)]
struct StoredSettings {
    /// Kept for migration from the first desktop settings format.
    reasoning_effort: Option<String>,
    #[serde(default)]
    reasoning_efforts: HashMap<String, String>,
    selected_model: Option<String>,
    #[serde(default)]
    allow_all_tools: bool,
    #[serde(default)]
    appearance: Appearance,
    #[serde(default)]
    enter_behavior: EnterBehavior,
    #[serde(default)]
    reduce_motion: bool,
    #[serde(default)]
    trajectory_actual_duration: bool,
    #[serde(default)]
    #[serde(alias = "model_profiles")]
    providers: Vec<ProviderProfile>,
}

pub(crate) struct SettingsStore {
    path: PathBuf,
    stored: StoredSettings,
}

impl SettingsStore {
    pub(crate) fn load(root: PathBuf) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(&root)?;
        let path = root.join("settings.json");
        let mut stored = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => StoredSettings::default(),
            Err(error) => return Err(error.into()),
        };
        let mut providers: Vec<ProviderProfile> = Vec::new();
        for mut provider in std::mem::take(&mut stored.providers) {
            if provider.provider_id == "deepseek" {
                provider.provider_id = DEEPSEEK_PROVIDER_ID.into();
            }
            if provider.models.is_empty()
                && let Some(model_id) = provider.legacy_model_id.take()
            {
                provider.models.push(ProviderModel::new(
                    model_id.clone(),
                    model_id,
                    provider.legacy_context_window.take().unwrap_or(128_000),
                    None,
                ));
            }
            if let Some(existing) = providers
                .iter_mut()
                .find(|existing| existing.provider_id == provider.provider_id)
            {
                for model in provider.models {
                    if !existing
                        .models
                        .iter()
                        .any(|existing| existing.model_id == model.model_id)
                    {
                        existing.models.push(model);
                    }
                }
                if existing.api_key.is_none() {
                    existing.api_key = provider.api_key;
                }
            } else {
                providers.push(provider);
            }
        }
        stored.providers = providers;
        if let Some(selected) = &mut stored.selected_model
            && let Some(model_id) = selected.strip_prefix("deepseek/")
        {
            *selected = format!("{DEEPSEEK_PROVIDER_ID}/{model_id}");
        }
        let legacy_reasoning = std::mem::take(&mut stored.reasoning_efforts);
        for (model_id, effort) in legacy_reasoning {
            let model_id = model_id
                .strip_prefix("deepseek/")
                .map(|model| format!("{DEEPSEEK_PROVIDER_ID}/{model}"))
                .unwrap_or(model_id);
            stored.reasoning_efforts.entry(model_id).or_insert(effort);
        }
        Ok(Self { path, stored })
    }

    pub(crate) fn apply(&self, model_id: &str, model: &mut Model) {
        let selected = self
            .stored
            .reasoning_efforts
            .get(model_id)
            .or_else(|| {
                self.stored
                    .reasoning_efforts
                    .get(&format!("{}/{}", model.name(), model.model()))
            })
            .map(String::as_str)
            .or(self.stored.reasoning_effort.as_deref());
        let Some(selected) = selected else {
            return;
        };
        if let Some(effort) = model
            .reasoning_efforts()
            .iter()
            .find(|effort| reasoning_key(effort) == selected)
        {
            model.set_reasoning_effort(effort.clone());
        }
    }

    pub(crate) fn selected_model(&self) -> Option<&str> {
        self.stored.selected_model.as_deref()
    }

    pub(crate) fn set_selected_model(&mut self, model_id: &str) -> Result<(), Box<dyn Error>> {
        self.stored.selected_model = Some(model_id.into());
        self.save()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_effort(
        &mut self,
        model_id: &str,
        effort: &ReasoningEffort,
    ) -> Result<(), Box<dyn Error>> {
        self.stored
            .reasoning_efforts
            .insert(model_id.into(), reasoning_key(effort).into());
        self.stored.reasoning_effort = None;
        self.save()
    }

    pub(crate) fn provider_profiles(&self) -> &[ProviderProfile] {
        &self.stored.providers
    }

    pub(crate) fn save_provider_profile(
        &mut self,
        mut profile: ProviderProfile,
        api_key: Option<String>,
    ) -> Result<(), Box<dyn Error>> {
        let position = self
            .stored
            .providers
            .iter()
            .position(|stored| stored.provider_id == profile.provider_id);
        profile.api_key = api_key
            .or_else(|| position.and_then(|index| self.stored.providers[index].api_key.clone()));
        if let Some(index) = position {
            self.stored.providers[index] = profile;
        } else {
            self.stored.providers.push(profile);
        }
        self.save()
    }

    pub(crate) fn allow_all_tools(&self) -> bool {
        self.stored.allow_all_tools
    }

    pub(crate) fn set_allow_all_tools(&mut self, allow: bool) -> Result<(), Box<dyn Error>> {
        self.stored.allow_all_tools = allow;
        self.save()
    }

    pub(crate) fn appearance(&self) -> Appearance {
        self.stored.appearance
    }

    pub(crate) fn set_appearance(&mut self, appearance: Appearance) -> Result<(), Box<dyn Error>> {
        self.stored.appearance = appearance;
        self.save()
    }

    pub(crate) fn enter_behavior(&self) -> EnterBehavior {
        self.stored.enter_behavior
    }

    pub(crate) fn set_enter_behavior(
        &mut self,
        behavior: EnterBehavior,
    ) -> Result<(), Box<dyn Error>> {
        self.stored.enter_behavior = behavior;
        self.save()
    }

    pub(crate) fn reduce_motion(&self) -> bool {
        self.stored.reduce_motion
    }

    pub(crate) fn set_reduce_motion(&mut self, reduce: bool) -> Result<(), Box<dyn Error>> {
        self.stored.reduce_motion = reduce;
        self.save()
    }

    pub(crate) fn trajectory_actual_duration(&self) -> bool {
        self.stored.trajectory_actual_duration
    }

    pub(crate) fn set_trajectory_actual_duration(
        &mut self,
        enabled: bool,
    ) -> Result<(), Box<dyn Error>> {
        self.stored.trajectory_actual_duration = enabled;
        self.save()
    }

    fn save(&self) -> Result<(), Box<dyn Error>> {
        let temporary = self.path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&self.stored)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        }
        fs::rename(temporary, &self.path)?;
        Ok(())
    }
}

fn reasoning_key(effort: &ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::None => "none",
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    static NEXT_TEST_ROOT: AtomicU64 = AtomicU64::new(0);

    fn test_root() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = NEXT_TEST_ROOT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "kcastle-settings-test-{}-{suffix}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn settings_round_trip_per_model_and_general_preferences() {
        let root = test_root();
        let mut store = SettingsStore::load(root.clone()).unwrap();
        store.set_selected_model("test/model").unwrap();
        store
            .set_effort("test/model", &ReasoningEffort::Low)
            .unwrap();
        store.set_appearance(Appearance::Dark).unwrap();
        store.set_enter_behavior(EnterBehavior::Queue).unwrap();
        store.set_reduce_motion(true).unwrap();
        store.set_trajectory_actual_duration(true).unwrap();
        store
            .save_provider_profile(
                ProviderProfile::new(
                    DEEPSEEK_PROVIDER_ID,
                    "DeepSeek",
                    "https://api.deepseek.test",
                    vec![
                        ProviderModel::new(
                            "deepseek-v4-flash",
                            "DeepSeek-V4-Flash",
                            1_000_000,
                            None,
                        ),
                        ProviderModel::new(
                            "deepseek-v4-pro",
                            "DeepSeek-V4-Pro",
                            1_000_000,
                            Some(256_000),
                        ),
                    ],
                ),
                Some("secret".into()),
            )
            .unwrap();

        let mut model = Model::new("test", "key", "http://localhost", "model", 10_000)
            .with_reasoning(
                &[ReasoningEffort::None, ReasoningEffort::Low],
                ReasoningEffort::None,
            );
        let store = SettingsStore::load(root.clone()).unwrap();
        store.apply("test/model", &mut model);
        assert_eq!(model.reasoning_effort(), Some(&ReasoningEffort::Low));
        assert_eq!(store.selected_model(), Some("test/model"));
        assert_eq!(store.appearance(), Appearance::Dark);
        assert_eq!(store.enter_behavior(), EnterBehavior::Queue);
        assert!(store.reduce_motion());
        assert!(store.trajectory_actual_duration());
        assert_eq!(store.provider_profiles()[0].api_key(), Some("secret"));
        assert_eq!(store.provider_profiles()[0].models.len(), 2);
        assert_eq!(
            store.provider_profiles()[0].models[1].max_output_tokens,
            Some(256_000)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn updating_a_profile_keeps_its_stored_credential_when_key_is_blank() {
        let root = test_root();
        let mut store = SettingsStore::load(root.clone()).unwrap();
        let profile = ProviderProfile::new(
            "openai",
            "OpenAI",
            "https://api.openai.test/v1",
            vec![ProviderModel::new("gpt-test", "GPT Test", 131_072, None)],
        );
        store
            .save_provider_profile(profile.clone(), Some("secret".into()))
            .unwrap();
        store.save_provider_profile(profile, None).unwrap();

        let store = SettingsStore::load(root.clone()).unwrap();
        assert_eq!(store.provider_profiles()[0].api_key(), Some("secret"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn master_provider_catalog_api_key_loads_without_rewrite() {
        let root = test_root();
        let original = serde_json::json!({
            "reasoning_effort": null,
            "reasoning_efforts": {},
            "selected_model": "deepseek-official/deepseek-v4-flash",
            "allow_all_tools": false,
            "appearance": "system",
            "enter_behavior": "steer",
            "reduce_motion": false,
            "providers": [
                {
                    "provider_id": "deepseek-official",
                    "display_name": "DeepSeek",
                    "api_base": "https://api.deepseek.com",
                    "models": [
                        {
                            "model_id": "deepseek-v4-flash",
                            "display_name": "DeepSeek-V4-Flash",
                            "context_window": 1_000_000
                        }
                    ],
                    "api_key": "master-secret"
                }
            ]
        });
        let path = root.join("settings.json");
        fs::write(&path, serde_json::to_vec_pretty(&original).unwrap()).unwrap();

        let store = SettingsStore::load(root.clone()).unwrap();

        assert_eq!(
            store.selected_model(),
            Some("deepseek-official/deepseek-v4-flash")
        );
        assert_eq!(
            store.provider_profiles()[0].api_key(),
            Some("master-secret")
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap()).unwrap(),
            original
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn legacy_single_model_profiles_are_migrated_to_provider_catalogs() {
        let root = test_root();
        fs::write(
            root.join("settings.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_profiles": [
                    {
                        "provider_id": "deepseek",
                        "display_name": "DeepSeek",
                        "api_base": "https://api.deepseek.test",
                        "model_id": "deepseek-test",
                        "context_window": 131072,
                        "api_key": "secret"
                    },
                    {
                        "provider_id": "deepseek",
                        "display_name": "DeepSeek",
                        "api_base": "https://api.deepseek.test",
                        "model_id": "deepseek-second",
                        "context_window": 256000
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();

        let store = SettingsStore::load(root.clone()).unwrap();
        let provider = &store.provider_profiles()[0];
        assert_eq!(provider.provider_id, DEEPSEEK_PROVIDER_ID);
        assert_eq!(provider.models.len(), 2);
        assert_eq!(provider.models[0].model_id, "deepseek-test");
        assert_eq!(provider.models[0].context_window, 131_072);
        assert_eq!(provider.models[1].model_id, "deepseek-second");
        assert_eq!(provider.api_key(), Some("secret"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn provider_ids_keep_legacy_display_name_reasoning_preferences() {
        let root = test_root();
        let mut store = SettingsStore::load(root.clone()).unwrap();
        store
            .set_effort("DeepSeek/deepseek-test", &ReasoningEffort::Low)
            .unwrap();
        let mut model = Model::new(
            "DeepSeek",
            "key",
            "https://api.deepseek.test",
            "deepseek-test",
            10_000,
        )
        .with_reasoning(
            &[ReasoningEffort::None, ReasoningEffort::Low],
            ReasoningEffort::None,
        );

        store.apply("deepseek/deepseek-test", &mut model);

        assert_eq!(model.reasoning_effort(), Some(&ReasoningEffort::Low));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn first_setting_write_creates_a_missing_root_directory() {
        let root = test_root();
        fs::remove_dir_all(&root).unwrap();

        let mut store = SettingsStore::load(root.clone()).unwrap();
        store.set_selected_model("test/model").unwrap();

        assert_eq!(
            SettingsStore::load(root.clone()).unwrap().selected_model(),
            Some("test/model")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
