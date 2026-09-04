//! Apps the user has excluded from the mic-activity trigger.
//!
//! `meethook record` starts a session whenever CoreAudio reports another process capturing
//! input, and that predicate excludes meethook's own helper processes by hardcoded bundle id.
//! This is the escape hatch for everything else: an app like a dictation tool that opens the
//! microphone without it being a meeting. The user names those apps here, in
//! [`Paths::exclusions_json`], and the record crate consults the set on every predicate walk.
//!
//! # Load policy
//!
//! Absent file is the *normal* case and means "no exclusions": with no file, or an empty
//! list, the trigger behaves exactly as before this file existed. That is the whole of what
//! reading may silently do.
//!
//! A file that exists but does not parse -- or that claims a schema version this build does
//! not understand -- is a hard error naming the path, never a fallback to the empty set. The
//! fallback *is* the bug the user was fixing: a `record` that quietly ignored a corrupt
//! exclusion list would keep treating the dictation tool as a meeting, and the user would be
//! debugging why their fix did nothing. Same house rule as the enrolled-speaker database: a
//! user who asked for something and quietly got the default has been lied to.
//!
//! Matching is exact only -- no wildcards, prefixes, or fuzzy executable names. The predicate
//! fails asymmetrically (an over-exclusion costs *every* session; a missed entry costs one
//! stray one), so a user entry must match positively and never act as a catch-all: a
//! `com.apple.` prefix would swallow FaceTime. See the record crate's activity module for
//! the failure asymmetry itself.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Error, Paths, Result};

/// The on-disk shape of `exclusions.json`. Carries the schema version the in-memory type
/// deliberately drops: meethook never writes this file, so there is no version to carry
/// back out.
#[derive(Deserialize)]
struct ExclusionsFile {
    schema_version: u32,
    /// Missing keys read as empty lists rather than as a parse error: a user hand-editing
    /// the file who drops one kind of entry means "none of that kind", not "malformed".
    #[serde(default)]
    bundle_ids: Vec<String>,
    #[serde(default)]
    executables: Vec<PathBuf>,
}

pub const EXCLUSIONS_SCHEMA_VERSION: u32 = 1;

const OLDEST_SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// Apps that never count as the mic-activity signal, as named by the user.
///
/// Consulted per process object by the record crate's predicate: an entry fires only when
/// the fact it keys on is present (a bundle-id entry cannot fire for a process whose bundle
/// id is unreadable; its executable entry can still fire).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppExclusions {
    /// Bundle ids that never start a session, matched exactly.
    pub bundle_ids: Vec<String>,
    /// Executable paths that never start a session, matched exactly.
    ///
    /// List the real executable inside `.app/Contents/MacOS/`, not the bundle directory:
    /// the predicate reads the executable behind the pid, which is what macOS hands out.
    pub executables: Vec<PathBuf>,
}

impl AppExclusions {
    /// Loads `<root>/exclusions.json`; absent means the empty set, anything unreadable or
    /// unrecognised is an error naming the path. See the module docs for why the error
    /// direction matters.
    pub fn read_or_empty(paths: &Paths) -> Result<AppExclusions> {
        let path = paths.exclusions_json();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(AppExclusions::default());
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        let file: ExclusionsFile =
            serde_json::from_slice(&bytes).map_err(|e| Error::json(&path, e))?;
        if !(OLDEST_SUPPORTED_SCHEMA_VERSION..=EXCLUSIONS_SCHEMA_VERSION)
            .contains(&file.schema_version)
        {
            return Err(Error::UnsupportedSchema {
                path,
                found: file.schema_version,
                oldest: OLDEST_SUPPORTED_SCHEMA_VERSION,
                newest: EXCLUSIONS_SCHEMA_VERSION,
            });
        }
        Ok(AppExclusions {
            bundle_ids: file.bundle_ids,
            // Canonicalized where possible because the process paths the predicate compares
            // against arrive canonicalized; an entry that does not resolve is kept raw and
            // simply will not match, which is the honest outcome for a path that is not
            // there.
            executables: file
                .executables
                .into_iter()
                .map(|p| match std::fs::canonicalize(&p) {
                    Ok(canonical) => canonical,
                    Err(_) => p,
                })
                .collect(),
        })
    }

    /// Whether `id` is on the exclusion list. Exact match only.
    pub fn contains_bundle_id(&self, id: &str) -> bool {
        self.bundle_ids.iter().any(|entry| entry == id)
    }

    /// Whether `path` is on the exclusion list. Exact match only; `path` should be the
    /// canonicalized executable behind the pid, which is what the entries were normalized
    /// to at load time.
    pub fn contains_executable(&self, path: &Path) -> bool {
        self.executables.iter().any(|entry| entry == path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A root holding no `exclusions.json`: the normal case.
    fn bare_paths(dir: &std::path::Path) -> Paths {
        Paths::new(dir)
    }

    #[test]
    fn an_absent_file_is_the_empty_set() {
        let dir = tempfile::tempdir().unwrap();
        let read = AppExclusions::read_or_empty(&bare_paths(dir.path())).unwrap();
        assert_eq!(read, AppExclusions::default());
    }

    #[test]
    fn a_populated_file_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let paths = bare_paths(dir.path());
        std::fs::write(
            paths.exclusions_json(),
            r#"{"schema_version": 1, "bundle_ids": ["com.example.voiceink"],
               "executables": ["/Applications/VoiceInk.app/Contents/MacOS/VoiceInk"]}"#,
        )
        .unwrap();

        let read = AppExclusions::read_or_empty(&paths).unwrap();
        assert_eq!(
            read,
            AppExclusions {
                bundle_ids: vec!["com.example.voiceink".to_owned()],
                // The path does not exist in the sandbox, so it is kept raw rather than
                // canonicalized.
                executables: vec![PathBuf::from(
                    "/Applications/VoiceInk.app/Contents/MacOS/VoiceInk"
                )],
            }
        );
        assert!(read.contains_bundle_id("com.example.voiceink"));
        assert!(!read.contains_bundle_id("com.apple.FaceTime"));
    }

    #[test]
    fn empty_lists_are_the_empty_set() {
        let dir = tempfile::tempdir().unwrap();
        let paths = bare_paths(dir.path());
        std::fs::write(
            paths.exclusions_json(),
            r#"{"schema_version": 1, "bundle_ids": [], "executables": []}"#,
        )
        .unwrap();
        assert_eq!(
            AppExclusions::read_or_empty(&paths).unwrap(),
            AppExclusions::default()
        );
    }

    #[test]
    fn missing_keys_default_to_empty_lists() {
        let dir = tempfile::tempdir().unwrap();
        let paths = bare_paths(dir.path());
        std::fs::write(paths.exclusions_json(), r#"{"schema_version": 1}"#).unwrap();
        assert_eq!(
            AppExclusions::read_or_empty(&paths).unwrap(),
            AppExclusions::default()
        );
    }

    #[test]
    fn malformed_json_is_an_error_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let paths = bare_paths(dir.path());
        std::fs::write(paths.exclusions_json(), b"{ not json").unwrap();

        let error = AppExclusions::read_or_empty(&paths).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("malformed JSON"), "{message}");
        assert!(message.contains("exclusions.json"), "{message}");
    }

    #[test]
    fn an_unsupported_schema_version_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let paths = bare_paths(dir.path());
        std::fs::write(
            paths.exclusions_json(),
            format!(
                r#"{{"schema_version": {}, "bundle_ids": [], "executables": []}}"#,
                EXCLUSIONS_SCHEMA_VERSION + 1
            ),
        )
        .unwrap();

        let error = AppExclusions::read_or_empty(&paths).unwrap_err();
        assert!(matches!(error, Error::UnsupportedSchema { .. }));
    }

    #[test]
    fn a_resolvable_executable_entry_is_canonicalized() {
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("some-app");
        std::fs::write(&binary, b"").unwrap();
        let canonical = std::fs::canonicalize(&binary).unwrap();

        let paths = bare_paths(dir.path());
        // A non-canonical spelling of the same path: a redundant `.` component.
        let spelled = dir.path().join(".").join("some-app");
        std::fs::write(
            paths.exclusions_json(),
            format!(
                r#"{{"schema_version": 1, "executables": ["{}"]}}"#,
                spelled.display()
            ),
        )
        .unwrap();

        let read = AppExclusions::read_or_empty(&paths).unwrap();
        assert_eq!(read.executables, vec![canonical.clone()]);
        assert!(read.contains_executable(&canonical));
    }
}
