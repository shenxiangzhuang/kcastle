use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use gpui_kit::assets::Assets as ComponentAssets;
use gpui_kit::component::IconNamed;
use gpui_kit::{AssetSource, Result, SharedString};

const SQUARE_PEN_PATH: &str = "icons/square-pen.svg";
const DOWNLOAD_PATH: &str = "icons/download.svg";
const ARCHIVE_PATH: &str = "icons/archive.svg";
const CLOCK_PATH: &str = "icons/clock.svg";
static GENERATED_ASSETS: OnceLock<Mutex<HashMap<String, Vec<u8>>>> = OnceLock::new();
static NEXT_GENERATED_ASSET: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub(crate) struct GeneratedAsset {
    pub(crate) path: SharedString,
    pub(crate) bytes: usize,
}

impl Drop for GeneratedAsset {
    fn drop(&mut self) {
        GENERATED_ASSETS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(self.path.as_ref());
    }
}

pub(crate) fn register_generated_asset(bytes: Vec<u8>) -> Arc<GeneratedAsset> {
    let path = format!(
        "generated/{}.svg",
        NEXT_GENERATED_ASSET.fetch_add(1, Ordering::Relaxed)
    );
    let len = bytes.len();
    GENERATED_ASSETS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(path.clone())
        .or_insert(bytes);
    Arc::new(GeneratedAsset {
        path: path.into(),
        bytes: len,
    })
}

pub(crate) struct DesktopAssets;

impl AssetSource for DesktopAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if let Some(bytes) = GENERATED_ASSETS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(path)
        {
            return Ok(Some(Cow::Owned(bytes.clone())));
        }
        if path == SQUARE_PEN_PATH {
            return Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/square-pen.svg"
            ))));
        }
        if path == DOWNLOAD_PATH {
            return Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/download.svg"
            ))));
        }
        if path == ARCHIVE_PATH {
            return Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/archive.svg"
            ))));
        }
        if path == CLOCK_PATH {
            return Ok(Some(Cow::Borrowed(include_bytes!(
                "../assets/icons/clock.svg"
            ))));
        }
        ComponentAssets.load(path)
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut assets = ComponentAssets.list(path)?;
        if SQUARE_PEN_PATH.starts_with(path) {
            assets.push(SQUARE_PEN_PATH.into());
        }
        if DOWNLOAD_PATH.starts_with(path) {
            assets.push(DOWNLOAD_PATH.into());
        }
        if ARCHIVE_PATH.starts_with(path) {
            assets.push(ARCHIVE_PATH.into());
        }
        if CLOCK_PATH.starts_with(path) {
            assets.push(CLOCK_PATH.into());
        }
        Ok(assets)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum DesktopIconName {
    Archive,
    Clock,
    Download,
    SquarePen,
}

impl IconNamed for DesktopIconName {
    fn path(self) -> SharedString {
        match self {
            Self::Archive => ARCHIVE_PATH,
            Self::Clock => CLOCK_PATH,
            Self::Download => DOWNLOAD_PATH,
            Self::SquarePen => SQUARE_PEN_PATH,
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generated_asset_lives_only_as_long_as_its_presentation() {
        let asset = register_generated_asset(b"<svg/>".to_vec());
        let path = asset.path.clone();
        let frame = asset.clone();
        drop(asset);
        assert!(DesktopAssets.load(&path).unwrap().is_some());
        drop(frame);
        assert!(DesktopAssets.load(&path).ok().flatten().is_none());
    }
}
