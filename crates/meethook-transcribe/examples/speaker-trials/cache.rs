//! The embedding cache.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::support::fail;
use super::voices::Voice;

#[derive(Serialize, Deserialize)]
struct EmbeddingCache {
    schema_version: u32,
    items: Vec<Voice>,
}

const CACHE_SCHEMA_VERSION: u32 = 1;

/// Reads `--embeddings`, or an empty map if it does not exist yet.
///
/// A file that exists and does not parse is fatal rather than ignored: silently re-measuring
/// everything would be slow and confusing, and silently scoring half a cache would be worse.
pub fn read_cache(path: &Path) -> BTreeMap<(String, String), Voice> {
    if !path.exists() {
        return BTreeMap::new();
    }
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| fail(&format!("could not read {}: {e}", path.display())));
    let cache: EmbeddingCache = serde_json::from_str(&text)
        .unwrap_or_else(|e| fail(&format!("could not parse {}: {e}", path.display())));

    if cache.schema_version != CACHE_SCHEMA_VERSION {
        fail(&format!(
            "{} is schema version {}, this tool writes {CACHE_SCHEMA_VERSION}",
            path.display(),
            cache.schema_version
        ));
    }

    // Two dimensions in one file means two embedding models and two spaces that cannot be
    // compared. `best_match` skips such a reference for exactly this reason; here it would
    // poison a whole run's worth of rates instead of one row, so it stops the run.
    let dimensions: BTreeSet<usize> = cache.items.iter().map(|item| item.dimensions).collect();
    if dimensions.len() > 1 {
        fail(&format!(
            "{} mixes {dimensions:?} dimensions, so it was written by more than one \
             embedding model. Delete it or re-run with --fresh.",
            path.display()
        ));
    }
    for item in &cache.items {
        if item.embedding.len() != item.dimensions {
            fail(&format!(
                "{}: {} / {} says {} dimensions but carries {}",
                path.display(),
                item.speaker,
                item.session,
                item.dimensions,
                item.embedding.len()
            ));
        }
    }

    println!(
        "cache:     {} ({} voice(s) available for re-use)",
        path.display(),
        cache.items.len()
    );
    cache
        .items
        .into_iter()
        .map(|item| ((item.speaker.clone(), item.session.clone()), item))
        .collect()
}

pub fn write_cache(path: &Path, voices: &[Voice]) {
    let cache = EmbeddingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        items: voices.to_vec(),
    };
    let json = serde_json::to_string_pretty(&cache)
        .unwrap_or_else(|e| fail(&format!("could not serialize the embedding cache: {e}")));
    std::fs::write(path, json)
        .unwrap_or_else(|e| fail(&format!("could not write {}: {e}", path.display())));
}
