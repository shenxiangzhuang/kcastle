use std::collections::HashMap;
use std::error::Error;

use kcastle_agent::{Model, ReasoningEffort};
use rusqlite::{OptionalExtension, params};

use crate::app_store::AppStore;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProviderModel {
    pub(crate) model_id: String,
    pub(crate) display_name: String,
    pub(crate) context_window: usize,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ProviderProfile {
    pub(crate) provider_id: String,
    pub(crate) display_name: String,
    pub(crate) api_base: String,
    pub(crate) models: Vec<ProviderModel>,
    api_key: Option<String>,
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
        }
    }

    pub(crate) fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Appearance {
    #[default]
    System,
    Light,
    Dark,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum EnterBehavior {
    #[default]
    Steer,
    Queue,
}

#[derive(Default)]
struct StoredSettings {
    reasoning_efforts: HashMap<String, String>,
    selected_model: Option<String>,
    allow_all_tools: bool,
    appearance: Appearance,
    enter_behavior: EnterBehavior,
    reduce_motion: bool,
    trajectory_actual_duration: bool,
    providers: Vec<ProviderProfile>,
}

pub(crate) struct SettingsStore {
    store: AppStore,
    stored: StoredSettings,
}

impl SettingsStore {
    pub(crate) fn load(source: impl AppStoreSource) -> Result<Self, Box<dyn Error>> {
        let store = source.into_app_store()?;
        let stored = load_settings(&store)?;
        Ok(Self { store, stored })
    }

    pub(crate) fn reasoning_effort(
        &self,
        model_id: &str,
        model: &Model,
    ) -> Option<ReasoningEffort> {
        let selected = self.stored.reasoning_efforts.get(model_id)?;
        model
            .reasoning_efforts()
            .iter()
            .find(|effort| effort.as_str() == selected)
            .copied()
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
            .insert(model_id.into(), effort.as_str().into());
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

fn load_settings(store: &AppStore) -> Result<StoredSettings, Box<dyn Error>> {
    let connection = store.connection()?;
    let mut stored = connection
        .query_row(
            "SELECT selected_model, allow_all_tools, appearance,
                    enter_behavior, reduce_motion, trajectory_actual_duration
             FROM app_preferences WHERE singleton = 1",
            [],
            |row| {
                Ok(StoredSettings {
                    reasoning_efforts: HashMap::new(),
                    selected_model: row.get(0)?,
                    allow_all_tools: row.get::<_, i64>(1)? != 0,
                    appearance: appearance_from_sql(row.get::<_, String>(2)?, 2)?,
                    enter_behavior: enter_behavior_from_sql(row.get::<_, String>(3)?, 3)?,
                    reduce_motion: row.get::<_, i64>(4)? != 0,
                    trajectory_actual_duration: row.get::<_, i64>(5)? != 0,
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
    Ok(stored)
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
            singleton, selected_model, allow_all_tools, appearance,
            enter_behavior, reduce_motion, trajectory_actual_duration
         ) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(singleton) DO UPDATE SET
            selected_model = excluded.selected_model,
            allow_all_tools = excluded.allow_all_tools,
            appearance = excluded.appearance,
            enter_behavior = excluded.enter_behavior,
            reduce_motion = excluded.reduce_motion,
            trajectory_actual_duration = excluded.trajectory_actual_duration",
        params![
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

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use crate::agent_config::DEEPSEEK_PROVIDER_ID;

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

        let model = Model::new("test", "key", "http://localhost", "model", 10_000)
            .with_reasoning_efforts(&[ReasoningEffort::None, ReasoningEffort::Low]);
        drop(store);
        let store = SettingsStore::load(root.clone()).unwrap();
        assert_eq!(
            store.reasoning_effort("test/model", &model),
            Some(ReasoningEffort::Low)
        );
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
}
