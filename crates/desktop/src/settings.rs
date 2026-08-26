use std::collections::HashMap;
use std::error::Error;
use std::fs;

use kcastle_agent::{Model, ReasoningEffort};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::agent_config::DEEPSEEK_PROVIDER_ID;
use crate::app_store::AppStore;

const LEGACY_SETTINGS_MIGRATION: &str = "settings-json-v1";

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
    store: AppStore,
    stored: StoredSettings,
}

impl SettingsStore {
    pub(crate) fn load(source: impl AppStoreSource) -> Result<Self, Box<dyn Error>> {
        let store = source.into_app_store()?;
        migrate_legacy_settings(&store)?;
        let stored = load_settings(&store)?;
        Ok(Self { store, stored })
    }

    fn normalize(mut stored: StoredSettings) -> StoredSettings {
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
        stored
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
        save_settings(&self.store, &self.stored)
    }
}

pub(crate) trait AppStoreSource {
    fn into_app_store(self) -> Result<AppStore, Box<dyn Error>>;
}

impl AppStoreSource for AppStore {
    fn into_app_store(self) -> Result<AppStore, Box<dyn Error>> {
        Ok(self)
    }
}

impl AppStoreSource for std::path::PathBuf {
    fn into_app_store(self) -> Result<AppStore, Box<dyn Error>> {
        AppStore::open(self)
    }
}

fn migrate_legacy_settings(store: &AppStore) -> Result<(), Box<dyn Error>> {
    if !store.migration_complete(LEGACY_SETTINGS_MIGRATION)? {
        let legacy_path = store.root().join("settings.json");
        let stored = match fs::read(&legacy_path) {
            Ok(bytes) => SettingsStore::normalize(serde_json::from_slice(&bytes)?),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => StoredSettings::default(),
            Err(error) => return Err(error.into()),
        };
        store.write(|transaction| {
            save_settings_with_transaction(transaction, &stored)?;
            AppStore::mark_migration_complete(transaction, LEGACY_SETTINGS_MIGRATION)
        })?;
    }
    store.backup_legacy_file("settings.json")?;
    // `config.yaml` belonged to a pre-desktop configuration path and has no reader in the current
    // application. Preserve it for rollback, but do not import stale defaults into the new store.
    store.backup_legacy_file("config.yaml")?;
    Ok(())
}

fn load_settings(store: &AppStore) -> Result<StoredSettings, Box<dyn Error>> {
    let connection = store.connection()?;
    let mut stored = connection
        .query_row(
            "SELECT legacy_reasoning_effort, selected_model, allow_all_tools, appearance,
                    enter_behavior, reduce_motion, trajectory_actual_duration
             FROM app_preferences WHERE singleton = 1",
            [],
            |row| {
                Ok(StoredSettings {
                    reasoning_effort: row.get(0)?,
                    reasoning_efforts: HashMap::new(),
                    selected_model: row.get(1)?,
                    allow_all_tools: row.get::<_, i64>(2)? != 0,
                    appearance: appearance_from_sql(row.get::<_, String>(3)?, 3)?,
                    enter_behavior: enter_behavior_from_sql(row.get::<_, String>(4)?, 4)?,
                    reduce_motion: row.get::<_, i64>(5)? != 0,
                    trajectory_actual_duration: row.get::<_, i64>(6)? != 0,
                    providers: Vec::new(),
                })
            },
        )
        .optional()?
        .unwrap_or_default();

    let mut effort_statement = connection
        .prepare("SELECT model_id, effort FROM model_reasoning_efforts ORDER BY model_id")?;
    let efforts = effort_statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for effort in efforts {
        let (model_id, effort) = effort?;
        stored.reasoning_efforts.insert(model_id, effort);
    }
    drop(effort_statement);

    let mut provider_statement = connection.prepare(
        "SELECT provider_id, display_name, api_base, api_key FROM providers ORDER BY ordinal",
    )?;
    let providers = provider_statement.query_map([], |row| {
        Ok(ProviderProfile {
            provider_id: row.get(0)?,
            display_name: row.get(1)?,
            api_base: row.get(2)?,
            models: Vec::new(),
            api_key: row.get(3)?,
            legacy_model_id: None,
            legacy_context_window: None,
        })
    })?;
    for provider in providers {
        stored.providers.push(provider?);
    }
    drop(provider_statement);

    let mut model_statement = connection.prepare(
        "SELECT model_id, display_name, context_window, max_output_tokens
         FROM provider_models WHERE provider_id = ?1 ORDER BY ordinal",
    )?;
    for provider in &mut stored.providers {
        let models = model_statement.query_map([provider.provider_id.as_str()], |row| {
            let context_window = row.get::<_, i64>(2)?;
            let max_output_tokens = row.get::<_, Option<i64>>(3)?;
            Ok(ProviderModel {
                model_id: row.get(0)?,
                display_name: row.get(1)?,
                context_window: usize::try_from(context_window).map_err(|error| {
                    rusqlite::Error::FromSqlConversionFailure(
                        2,
                        rusqlite::types::Type::Integer,
                        Box::new(error),
                    )
                })?,
                max_output_tokens: max_output_tokens.map(u32::try_from).transpose().map_err(
                    |error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Integer,
                            Box::new(error),
                        )
                    },
                )?,
            })
        })?;
        provider.models = models.collect::<Result<Vec<_>, _>>()?;
    }
    Ok(SettingsStore::normalize(stored))
}

fn save_settings(store: &AppStore, stored: &StoredSettings) -> Result<(), Box<dyn Error>> {
    store.write(|transaction| save_settings_with_transaction(transaction, stored))
}

fn save_settings_with_transaction(
    transaction: &rusqlite::Transaction<'_>,
    stored: &StoredSettings,
) -> Result<(), Box<dyn Error>> {
    transaction.execute(
        "INSERT INTO app_preferences (
            singleton, legacy_reasoning_effort, selected_model, allow_all_tools, appearance,
            enter_behavior, reduce_motion, trajectory_actual_duration
         ) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(singleton) DO UPDATE SET
            legacy_reasoning_effort = excluded.legacy_reasoning_effort,
            selected_model = excluded.selected_model,
            allow_all_tools = excluded.allow_all_tools,
            appearance = excluded.appearance,
            enter_behavior = excluded.enter_behavior,
            reduce_motion = excluded.reduce_motion,
            trajectory_actual_duration = excluded.trajectory_actual_duration",
        params![
            stored.reasoning_effort,
            stored.selected_model,
            i64::from(stored.allow_all_tools),
            appearance_to_sql(stored.appearance),
            enter_behavior_to_sql(stored.enter_behavior),
            i64::from(stored.reduce_motion),
            i64::from(stored.trajectory_actual_duration),
        ],
    )?;
    transaction.execute("DELETE FROM model_reasoning_efforts", [])?;
    for (model_id, effort) in &stored.reasoning_efforts {
        transaction.execute(
            "INSERT INTO model_reasoning_efforts (model_id, effort) VALUES (?1, ?2)",
            params![model_id, effort],
        )?;
    }
    transaction.execute("DELETE FROM providers", [])?;
    for (provider_ordinal, provider) in stored.providers.iter().enumerate() {
        transaction.execute(
            "INSERT INTO providers (provider_id, ordinal, display_name, api_base, api_key)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                provider.provider_id,
                i64::try_from(provider_ordinal)?,
                provider.display_name,
                provider.api_base,
                provider.api_key,
            ],
        )?;
        for (ordinal, model) in provider.models.iter().enumerate() {
            transaction.execute(
                "INSERT INTO provider_models (
                    provider_id, ordinal, model_id, display_name, context_window,
                    max_output_tokens
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    provider.provider_id,
                    ordinal as i64,
                    model.model_id,
                    model.display_name,
                    i64::try_from(model.context_window)?,
                    model.max_output_tokens.map(i64::from),
                ],
            )?;
        }
    }
    Ok(())
}

fn appearance_to_sql(value: Appearance) -> &'static str {
    match value {
        Appearance::System => "system",
        Appearance::Light => "light",
        Appearance::Dark => "dark",
    }
}

fn appearance_from_sql(value: String, column: usize) -> rusqlite::Result<Appearance> {
    match value.as_str() {
        "system" => Ok(Appearance::System),
        "light" => Ok(Appearance::Light),
        "dark" => Ok(Appearance::Dark),
        _ => Err(invalid_text_value(column, "appearance", value)),
    }
}

fn enter_behavior_to_sql(value: EnterBehavior) -> &'static str {
    match value {
        EnterBehavior::Steer => "steer",
        EnterBehavior::Queue => "queue",
    }
}

fn enter_behavior_from_sql(value: String, column: usize) -> rusqlite::Result<EnterBehavior> {
    match value.as_str() {
        "steer" => Ok(EnterBehavior::Steer),
        "queue" => Ok(EnterBehavior::Queue),
        _ => Err(invalid_text_value(column, "enter behavior", value)),
    }
}

fn invalid_text_value(column: usize, label: &str, value: String) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        column,
        rusqlite::types::Type::Text,
        Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid {label}: {value:?}"),
        )),
    )
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
        path::PathBuf,
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
        drop(store);
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
        drop(store);
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

        drop(store);
        let store = SettingsStore::load(root.clone()).unwrap();
        assert_eq!(store.provider_profiles()[0].api_key(), Some("secret"));
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn master_provider_catalog_api_key_is_imported_before_legacy_backup() {
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
        fs::write(root.join("config.yaml"), b"unused: true\n").unwrap();

        let store = SettingsStore::load(root.clone()).unwrap();

        assert_eq!(
            store.selected_model(),
            Some("deepseek-official/deepseek-v4-flash")
        );
        assert_eq!(
            store.provider_profiles()[0].api_key(),
            Some("master-secret")
        );
        assert!(!path.exists());
        let backup = root.join("backups/pre-app-sqlite/settings.json");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(backup).unwrap()).unwrap(),
            original
        );
        assert_eq!(
            fs::read(root.join("backups/pre-app-sqlite/config.yaml")).unwrap(),
            b"unused: true\n"
        );
        drop(store);
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
        drop(store);
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
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn first_setting_write_creates_a_missing_root_directory() {
        let root = test_root();
        fs::remove_dir_all(&root).unwrap();

        let mut store = SettingsStore::load(root.clone()).unwrap();
        store.set_selected_model("test/model").unwrap();

        let reloaded = SettingsStore::load(root.clone()).unwrap();
        assert_eq!(reloaded.selected_model(), Some("test/model"));
        drop(reloaded);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_legacy_settings_remain_in_place_until_a_retry_succeeds() {
        let root = test_root();
        let settings = root.join("settings.json");
        fs::write(&settings, b"not json").unwrap();

        assert!(SettingsStore::load(root.clone()).is_err());
        assert_eq!(fs::read(&settings).unwrap(), b"not json");
        assert!(!root.join("backups/pre-app-sqlite/settings.json").exists());

        fs::write(
            &settings,
            serde_json::to_vec(&serde_json::json!({"appearance": "dark"})).unwrap(),
        )
        .unwrap();
        let store = SettingsStore::load(root.clone()).unwrap();
        assert_eq!(store.appearance(), Appearance::Dark);
        assert!(!settings.exists());
        assert!(root.join("backups/pre-app-sqlite/settings.json").is_file());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}
