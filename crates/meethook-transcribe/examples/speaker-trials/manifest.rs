//! The manifest.

use std::path::{Path, PathBuf};

/// One person as they sounded in one recording session: the unit everything here measures.
pub struct Item {
    pub speaker: String,
    pub session: String,
    pub wavs: Vec<PathBuf>,
}

/// Parses the manifest, grouping lines that share a `(speaker, session)` into one item.
///
/// Items come back in manifest order, and so do the wav files within each -- which is what
/// makes "each speaker's *first* session is the enrolled reference" a reproducible statement
/// about a file the operator wrote rather than about whatever order a hash map felt like.
pub fn read_manifest(path: &Path) -> Result<Vec<Item>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;

    let mut items: Vec<Item> = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
        let [speaker, session, wav] = fields.as_slice() else {
            return Err(format!(
                "{}:{}: expected three tab-separated fields \
                 (speaker, session, wav), found {}",
                path.display(),
                number + 1,
                fields.len()
            ));
        };
        if speaker.is_empty() || session.is_empty() || wav.is_empty() {
            return Err(format!(
                "{}:{}: no field may be empty",
                path.display(),
                number + 1
            ));
        }

        match items
            .iter_mut()
            .find(|item| item.speaker == *speaker && item.session == *session)
        {
            Some(item) => item.wavs.push(PathBuf::from(wav)),
            None => items.push(Item {
                speaker: (*speaker).to_string(),
                session: (*session).to_string(),
                wavs: vec![PathBuf::from(wav)],
            }),
        }
    }

    if items.is_empty() {
        return Err(format!("{} named no audio at all", path.display()));
    }
    Ok(items)
}
