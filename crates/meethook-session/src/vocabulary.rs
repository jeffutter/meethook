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
//! match_title = "(?i)\\bd&d\\b|curse of strahd"
//! ```
//!
//! A run applies `[global]` plus at most one profile. Profiles are never merged with each
//! other: a D&D list spent on a work meeting costs prompt window for words that do not occur,
//! and leaks one domain's spellings into another.
//!
//! The profile is chosen, in order, by an explicit name (`--vocab`), then by the one profile
//! whose `match_title` regex matches the session's calendar event title, then not at all
//! (`[global]` only). `match_title` is a plain, unanchored [`regex`] pattern; write `(?i)`
//! for case-insensitive. Two profiles matching the same title is an error, not a guess.
//!
//! Load policy matches `exclusions.json`: a missing file is the normal case and means no
//! vocabulary; a file that exists but is malformed, has an unknown key, a non-string term or
//! an invalid `match_title` is a hard error naming the file and (where known) the line or
//! profile, never a silent fallback to no vocabulary.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::Deserialize;

use crate::{Error, Paths, Result};

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobalSection {
    #[serde(default)]
    terms: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileSection {
    #[serde(default)]
    terms: Vec<String>,
    #[serde(default)]
    match_title: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct VocabularyFile {
    global: Option<GlobalSection>,
    #[serde(default)]
    profile: BTreeMap<String, ProfileSection>,
}

#[derive(Debug)]
struct Profile {
    terms: Vec<String>,
    match_title: Option<Regex>,
}

#[derive(Debug)]
struct Loaded {
    global: Vec<String>,
    profiles: BTreeMap<String, Profile>,
}

/// A parsed `vocabulary.toml`, or the absence of one.
#[derive(Debug)]
pub struct Vocabulary {
    path: PathBuf,
    /// `None` when the file does not exist, which `--vocab` reports differently from an
    /// unknown profile.
    file: Option<Loaded>,
}

impl Default for Vocabulary {
    fn default() -> Self {
        Vocabulary {
            path: PathBuf::from("vocabulary.toml"),
            file: None,
        }
    }
}

/// The terms one session applies, and which profile (if any) contributed them.
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
                return Ok(Vocabulary { path, file: None });
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        Self::parse(&path, &text)
    }

    fn parse(path: &Path, text: &str) -> Result<Vocabulary> {
        let fail = |detail: String| Error::Vocabulary {
            path: path.to_path_buf(),
            detail,
        };
        let file: VocabularyFile = toml::from_str(text).map_err(|e| {
            let location = e
                .span()
                .map(|span| {
                    let line = text[..span.start.min(text.len())].matches('\n').count() + 1;
                    format!("line {line}: ")
                })
                .unwrap_or_default();
            fail(format!("{location}{}", e.message()))
        })?;
        let mut profiles = BTreeMap::new();
        for (name, section) in file.profile {
            let match_title = section
                .match_title
                .map(|pattern| {
                    Regex::new(&pattern)
                        .map_err(|e| fail(format!("profile {name:?}: invalid match_title: {e}")))
                })
                .transpose()?;
            profiles.insert(
                name,
                Profile {
                    terms: section.terms,
                    match_title,
                },
            );
        }
        Ok(Vocabulary {
            path: path.to_path_buf(),
            file: Some(Loaded {
                global: file.global.map(|g| g.terms).unwrap_or_default(),
                profiles,
            }),
        })
    }

    /// Checks an explicit `--vocab` name against the file, so a typo fails the run before any
    /// engine is opened.
    pub fn check_explicit(&self, name: &str) -> Result<()> {
        self.select(Some(name), None).map(|_| ())
    }

    /// Chooses the terms for one session: `[global]`, then at most one profile.
    ///
    /// An explicit name wins and the title is ignored. Otherwise the profile whose
    /// `match_title` matches `meeting_title` applies; none matching means `[global]` only; two
    /// or more is an error naming them. `None` for the title behaves exactly as no match.
    pub fn select(&self, explicit: Option<&str>, meeting_title: Option<&str>) -> Result<Selection> {
        let fail = |detail: String| Error::Vocabulary {
            path: self.path.clone(),
            detail,
        };
        let mut terms: Vec<String> = self
            .file
            .as_ref()
            .map(|f| f.global.clone())
            .unwrap_or_default();

        let chosen: Option<&str> = if let Some(name) = explicit {
            let Some(file) = &self.file else {
                return Err(fail(format!(
                    "--vocab {name} was given but the file does not exist"
                )));
            };
            if !file.profiles.contains_key(name) {
                let known: Vec<&str> = file.profiles.keys().map(String::as_str).collect();
                let known = if known.is_empty() {
                    "the file defines no profiles".to_owned()
                } else {
                    format!("profiles are: {}", known.join(", "))
                };
                return Err(fail(format!("no profile named {name:?}; {known}")));
            }
            Some(name)
        } else if let (Some(file), Some(title)) = (&self.file, meeting_title) {
            let matching: Vec<&str> = file
                .profiles
                .iter()
                .filter(|(_, p)| p.match_title.as_ref().is_some_and(|re| re.is_match(title)))
                .map(|(name, _)| name.as_str())
                .collect();
            match matching.as_slice() {
                [] => None,
                [one] => Some(*one),
                several => {
                    // The title is deliberately not in the message: it is meeting content.
                    return Err(fail(format!(
                        "the session's calendar event title matches match_title of several                          profiles ({}); narrow the patterns or pass --vocab",
                        several.join(", ")
                    )));
                }
            }
        } else {
            None
        };

        let Some(name) = chosen else {
            return Ok(Selection {
                terms,
                profile: None,
            });
        };
        let profile = &self.file.as_ref().expect("chosen implies a file").profiles[name];
        terms.extend(profile.terms.iter().cloned());
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
        Vocabulary::load(&paths)?.select(name, None)
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    const FILE: &str = r#"
[global]
terms = ["meethook"]

[profile.dnd]
terms = ["Strahd", "Barovia"]
match_title = "(?i)d&d|strahd"

[profile.work]
terms = ["Kubernetes"]
"#;

    #[test]
    fn missing_file_is_no_vocabulary() {
        let (_d, paths) = paths_with(None);
        let sel = Vocabulary::load(&paths)
            .unwrap()
            .select(None, None)
            .unwrap();
        assert_eq!(sel, Selection::default());
    }

    #[test]
    fn vocab_with_no_file_is_an_error() {
        let (_d, paths) = paths_with(None);
        let err = Vocabulary::load(&paths)
            .unwrap()
            .select(Some("dnd"), None)
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

    fn by_title(contents: &str, title: Option<&str>) -> Result<Selection> {
        let (_d, paths) = paths_with(Some(contents));
        Vocabulary::load(&paths)?.select(None, title)
    }

    #[test]
    fn a_matching_title_picks_its_profile() {
        let sel = by_title(FILE, Some("Friday D&D night")).unwrap();
        assert_eq!(sel.profile.as_deref(), Some("dnd"));
        assert_eq!(sel.terms, strs(&["meethook", "Strahd", "Barovia"]));
    }

    #[test]
    fn no_match_or_no_title_is_global_only() {
        for title in [Some("Sprint planning"), None] {
            let sel = by_title(FILE, title).unwrap();
            assert_eq!(sel.terms, strs(&["meethook"]));
            assert_eq!(sel.profile, None);
        }
    }

    #[test]
    fn explicit_name_beats_the_title() {
        let (_d, paths) = paths_with(Some(FILE));
        let sel = Vocabulary::load(&paths)
            .unwrap()
            .select(Some("work"), Some("D&D night"))
            .unwrap();
        assert_eq!(sel.profile.as_deref(), Some("work"));
    }

    #[test]
    fn two_matching_profiles_fail_naming_both_but_not_the_title() {
        let file = format!("{FILE}\n[profile.also]\nmatch_title = \"Secret\"\n");
        let err = by_title(&file, Some("Secret D&D")).unwrap_err().to_string();
        assert!(err.contains("also") && err.contains("dnd"), "{err}");
        assert!(!err.contains("Secret D&D"), "{err}");
    }

    #[test]
    fn an_invalid_pattern_fails_at_load_naming_the_profile() {
        let err = select("[profile.broken]\nmatch_title = \"(\"\n", None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("broken") && err.contains("match_title"),
            "{err}"
        );
    }

    #[test]
    fn global_rejects_match_title() {
        assert!(select("[global]\nmatch_title = \"x\"\n", None).is_err());
    }

    #[test]
    fn check_explicit_reports_unknown_names() {
        let (_d, paths) = paths_with(Some(FILE));
        let v = Vocabulary::load(&paths).unwrap();
        assert!(v.check_explicit("dnd").is_ok());
        assert!(v.check_explicit("nope").is_err());
    }
}
