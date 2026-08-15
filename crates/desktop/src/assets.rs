use std::borrow::Cow;

use gpui::{AssetSource, Result, SharedString};
use gpui_component::IconNamed;
use gpui_component_assets::Assets as ComponentAssets;

const SQUARE_PEN_PATH: &str = "icons/square-pen.svg";

pub(crate) struct DesktopAssets;

impl AssetSource for DesktopAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
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
