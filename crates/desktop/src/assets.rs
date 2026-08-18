use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use gpui::{AssetSource, Result, SharedString};
use gpui_component::IconNamed;
use gpui_component_assets::Assets as ComponentAssets;

const SQUARE_PEN_PATH: &str = "icons/square-pen.svg";
static GENERATED_ASSETS: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();
static NEXT_GENERATED_ASSET: AtomicU64 = AtomicU64::new(0);

pub(crate) fn register_generated_asset(bytes: Vec<u8>) -> SharedString {
    let path = format!(
        "generated/{}.svg",
        NEXT_GENERATED_ASSET.fetch_add(1, Ordering::Relaxed)
    );
    GENERATED_ASSETS
        .get_or_init(Default::default)
        .lock()
        .expect("generated asset cache poisoned")
        .entry(path.clone())
        .or_insert(bytes);
    path.into()
}

pub(crate) struct DesktopAssets;

impl AssetSource for DesktopAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if let Some(bytes) = GENERATED_ASSETS
            .get_or_init(Default::default)
            .lock()
            .expect("generated asset cache poisoned")
            .get(path)
        {
            return Ok(Some(Cow::Owned(bytes.clone())));
        }
        if path == SQUARE_PEN_PATH {
            return Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/square-pen.svg"
            ))));
        }
        ComponentAssets.load(path)
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut assets = ComponentAssets.list(path)?;
        if SQUARE_PEN_PATH.starts_with(path) {
            assets.push(SQUARE_PEN_PATH.into());
        }
        Ok(assets)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum DesktopIconName {
    SquarePen,
}

impl IconNamed for DesktopIconName {
    fn path(self) -> SharedString {
        match self {
            Self::SquarePen => SQUARE_PEN_PATH,
        }
        .into()
    }
}
