use std::{fs::File, io, io::Read, ops::Deref, path::Path};

/// Immutable font bytes whose owner keeps either a snapshot or a mapping alive.
#[derive(Debug)]
pub struct FontData(Storage);

#[derive(Debug)]
enum Storage {
    Owned(Vec<u8>),
    #[cfg(target_os = "macos")]
    Mapped(memmap2::Mmap),
}

impl FontData {
    pub(crate) fn read(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        #[cfg(target_os = "macos")]
        if let Some(mapping) = path
            .canonicalize()
            .ok()
            .and_then(|canonical| map_system_font(&canonical, &file))
        {
            return Ok(Self(Storage::Mapped(mapping)));
        }
        // Custom fonts, writable volumes, unsupported platforms and mapping
        // failures (including canonicalization) retain snapshot semantics.
        // Mapping does not advance the cursor.
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(bytes.into())
    }

    pub fn as_slice(&self) -> &[u8] {
        match &self.0 {
            Storage::Owned(bytes) => bytes,
            #[cfg(target_os = "macos")]
            Storage::Mapped(mapping) => mapping,
        }
    }
}

impl From<Vec<u8>> for FontData {
    fn from(bytes: Vec<u8>) -> Self {
        Self(Storage::Owned(bytes))
    }
}

impl Deref for FontData {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(target_os = "macos")]
fn map_system_font(path: &Path, file: &File) -> Option<memmap2::Mmap> {
    use std::{mem::MaybeUninit, os::fd::AsRawFd};

    if !path.starts_with("/System/Library/Fonts") {
        return None;
    }
    let mut stat = MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: stat points to writable storage and file owns a valid descriptor.
    if unsafe { libc::fstatfs(file.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: successful fstatfs initialized the structure.
    if unsafe { stat.assume_init() }.f_flags & libc::MNT_RDONLY as u32 == 0 {
        return None;
    }
    // SAFETY: only system fonts on a verified read-only filesystem are mapped.
    // The check uses the open descriptor, not path permissions or a discovery
    // hint. These files cannot be modified/truncated during normal execution;
    // changing the OS's read-only system mount is outside this process's trust
    // boundary. Writable/custom fonts take an owned snapshot instead. FontData
    // owns the mapping for the entire lifetime of every borrowed byte slice.
    unsafe { memmap2::MmapOptions::new().map(file) }.ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writable_font_is_a_snapshot_even_through_a_symlink() {
        let root = std::env::temp_dir().join(format!("ratex-font-data-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("font.ttf");
        std::fs::write(&path, b"original font bytes").unwrap();
        #[cfg(unix)]
        let load_path = {
            let link = root.join("alias.ttf");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            link
        };
        #[cfg(not(unix))]
        let load_path = path.clone();
        let bytes = FontData::read(&load_path).unwrap();
        assert!(matches!(bytes.0, Storage::Owned(_)));
        #[cfg(target_os = "macos")]
        assert!(map_system_font(
            Path::new("/System/Library/Fonts/not-the-open-file.ttf"),
            &File::open(&path).unwrap(),
        )
        .is_none());
        std::fs::write(&path, b"changed").unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bytes.as_slice(), b"original font bytes");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn readonly_system_fonts_are_mapped_without_changing_bytes() {
        for path in [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
        ] {
            let path = Path::new(path);
            if !path.exists() {
                continue;
            }
            let file = File::open(path).unwrap();
            if map_system_font(path, &file).is_none() {
                // Machines without an immutable system volume use the safe fallback.
                continue;
            }
            let bytes = FontData::read(path).unwrap();
            assert!(matches!(bytes.0, Storage::Mapped(_)));
            assert_eq!(bytes.as_slice(), std::fs::read(path).unwrap());
        }
    }
}
