use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::PathBuf;

use kcastle_agent::{Model, ReasoningEffort};
use serde::{Deserialize, Serialize};

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
}

pub(crate) struct SettingsStore {
    path: PathBuf,
    stored: StoredSettings,
}

impl SettingsStore {
    pub(crate) fn load(root: PathBuf) -> Result<Self, Box<dyn Error>> {
        let path = root.join("settings.json");
        let stored = match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => StoredSettings::default(),
            Err(error) => return Err(error.into()),
        };
        Ok(Self { path, stored })
    }

    pub(crate) fn apply(&self, model_id: &str, model: &mut Model) {
        let selected = self
            .stored
            .reasoning_efforts
            .get(model_id)
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

    fn save(&self) -> Result<(), Box<dyn Error>> {
        let temporary = self.path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&self.stored)?)?;
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
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn test_root() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("kcastle-settings-test-{suffix}"));
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
        fs::remove_dir_all(root).unwrap();
    }
}
