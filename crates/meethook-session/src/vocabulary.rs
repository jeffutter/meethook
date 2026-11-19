//! `vocabulary.toml`: the words Whisper gets wrong, grouped by the kind of session.
//!
//! One hand-written file under the root, next to `speakers.json`:
//!
//! ```toml
//! [global]
//! terms = ["meethook"]
//!
//! [profile.dnd]
//! terms = ["Strahd", "Barovia"]
//! ```
//!
//! A run applies `[global]` plus at most one profile, chosen by name. Profiles are never merged
//! with each other: a D&D list spent on a work meeting costs prompt window for words that do
//! not occur, and leaks one domain's spellings into another.
//!
//! Load policy matches `exclusions.json`: a missing file is the normal case and means no
//! vocabulary; a file that exists but is malformed, has an unknown key or a non-string term is
//! a hard error naming the file and line, never a silent fallback to no vocabulary.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::{Error, Paths, Result};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Section {
    #[serde(default)]
    terms: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct VocabularyFile {
    global: Option<Section>,
    #[serde(default)]
    profile: BTreeMap<String, Section>,
}

/// A parsed `vocabulary.toml`, or the absence of one.
#[derive(Debug, Default)]
pub struct Vocabulary {
    /// `None` when the file does not exist, which `--vocab` reports differently from an
    /// unknown profile.
    file: Option<VocabularyFile>,
}

/// The terms one run applies, and which profile (if any) contributed them.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Selection {
    /// `[global]` terms followed by the chosen profile's terms.
    pub terms: Vec<String>,
    /// The chosen profile; `None` when only `[global]` applied.
    pub profile: Option<String>,
}

impl Vocabulary {
    /// Reads `<root>/vocabulary.toml`. A missing file is `Ok` with no vocabulary.
    pub fn load(paths: &Paths) -> Result<Vocabulary> {
        let path = paths.vocabulary_toml();
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Vocabulary { file: None });
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        Self::parse(&path, &text)
    }

    fn parse(path: &Path, text: &str) -> Result<Vocabulary> {
        let file: VocabularyFile = toml::from_str(text).map_err(|e| {
            let location = e
                .span()
                .map(|span| {
                    let line = text[..span.start.min(text.len())].matches('\n').count() + 1;
                    format!("line {line}: ")
                })
                .unwrap_or_default();
            Error::Vocabulary {
                path: path.to_path_buf(),
                detail: format!("{location}{}", e.message()),
            }
        })?;
        Ok(Vocabulary { file: Some(file) })
    }

    /// Chooses the terms for one run: `[global]`, then the named profile's terms. Never any
    /// other profile. A name the file does not have is an error listing the ones it does.
    pub fn select(&self, paths: &Paths, profile: Option<&str>) -> Result<Selection> {
        let path = paths.vocabulary_toml();
        let mut terms: Vec<String> = self
            .file
            .as_ref()
            .and_then(|f| f.global.as_ref())
            .map(|g| g.terms.clone())
            .unwrap_or_default();

        let Some(name) = profile else {
            return Ok(Selection {
                terms,
                profile: None,
            });
        };
        let Some(file) = &self.file else {
            return Err(Error::Vocabulary {
                path,
                detail: format!("--vocab {name} was given but the file does not exist"),
            });
        };
        let Some(section) = file.profile.get(name) else {
            let known: Vec<&str> = file.profile.keys().map(String::as_str).collect();
            let known = if known.is_empty() {
                "the file defines no profiles".to_owned()
            } else {
                format!("profiles are: {}", known.join(", "))
            };
            return Err(Error::Vocabulary {
                path,
                detail: format!("no profile named {name:?}; {known}"),
            });
        };
        terms.extend(section.terms.iter().cloned());
        Ok(Selection {
            terms,
            profile: Some(name.to_owned()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths_with(contents: Option<&str>) -> (tempfile::TempDir, Paths) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path().to_path_buf());
        if let Some(c) = contents {
            std::fs::write(paths.vocabulary_toml(), c).unwrap();
        }
        (dir, paths)
    }

    fn select(contents: &str, name: Option<&str>) -> Result<Selection> {
        let (_d, paths) = paths_with(Some(contents));
        Vocabulary::load(&paths)?.select(&paths, name)
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    const FILE: &str = r#"
[global]
terms = ["meethook"]

[profile.dnd]
terms = ["Strahd", "Barovia"]

[profile.work]
terms = ["Kubernetes"]
"#;

    #[test]
    fn missing_file_is_no_vocabulary() {
        let (_d, paths) = paths_with(None);
        let sel = Vocabulary::load(&paths)
            .unwrap()
            .select(&paths, None)
            .unwrap();
        assert_eq!(sel, Selection::default());
    }

    #[test]
    fn vocab_with_no_file_is_an_error() {
        let (_d, paths) = paths_with(None);
        let err = Vocabulary::load(&paths)
            .unwrap()
            .select(&paths, Some("dnd"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("vocabulary.toml") && err.contains("does not exist"),
            "{err}"
        );
    }

    #[test]
    fn no_vocab_uses_global_only() {
        let sel = select(FILE, None).unwrap();
        assert_eq!(sel.terms, strs(&["meethook"]));
        assert_eq!(sel.profile, None);
    }

    #[test]
    fn profile_adds_to_global_in_order_and_never_merges_others() {
        let sel = select(FILE, Some("dnd")).unwrap();
        assert_eq!(sel.terms, strs(&["meethook", "Strahd", "Barovia"]));
        assert_eq!(sel.profile.as_deref(), Some("dnd"));
        assert!(!sel.terms.contains(&"Kubernetes".to_owned()));
    }

    #[test]
    fn global_is_optional() {
        let sel = select("[profile.a]\nterms = [\"X\"]\n", Some("a")).unwrap();
        assert_eq!(sel.terms, strs(&["X"]));
        assert_eq!(select("", None).unwrap(), Selection::default());
    }

    #[test]
    fn unknown_profile_lists_existing_ones() {
        let err = select(FILE, Some("nope")).unwrap_err().to_string();
        assert!(err.contains("nope") && err.contains("dnd, work"), "{err}");
    }

    #[test]
    fn bad_files_name_the_file_and_line() {
        for (bad, line) in [
            ("[global]\nterms = [\"a\"]\nbogus = 1\n", "line 3"),
            ("[profile.x]\nterms = []\nextra = 1\n", "line 3"),
            ("[global]\nterms = [\"a\", 5]\n", "line 2"),
            ("[global]\nterms = [\n\n", "line "),
            ("surprise = 1\n", "line 1"),
        ] {
            let err = select(bad, None).unwrap_err().to_string();
            assert!(err.contains("vocabulary.toml"), "{err}");
            assert!(err.contains(line), "wanted {line} in {err}");
        }
    }
}
