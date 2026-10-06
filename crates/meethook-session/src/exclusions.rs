//! Apps named by what the microphone trigger saw them doing.
//!
//! Two things in this crate are sets of *app identities* rather than sets of anything else:
//! the user's exclusion list ([`AppExclusions`]) and the apps a session observed holding the
//! microphone (`SessionMetadata::mic_apps`). They are the same two lists, matched the same
//! exact way, and they are therefore one type: [`AppIdentities`]. That is not tidiness -- see
//! [Shared identity](#shared-identity) below for the user-facing property the shared type is
//! what makes true.
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
//!
//! # Shared identity
//!
//! A session records which apps held the microphone. When one of them turns out not to be a
//! meeting, the user's remedy is to name that app in `exclusions.json` -- and the whole design
//! goal of this module is that the remedy is *copy and paste*, not transcription. It holds
//! because both files are serialized from [`AppIdentities`] rather than from two shapes that
//! happen to agree -- that type's doc carries the reasoning in full.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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

/// The two ways the microphone trigger can name a program, and the vocabulary both
/// `exclusions.json` and a session's `mic_apps` are written in.
///
/// Serialized as `{"bundle_ids": [...], "executables": [...]}` in both places, which is what
/// makes moving a list from a session into the user's exclusion file a copy-paste rather than
/// a translation: read the session's `mic_apps`, move its two lists under `exclusions.json`'s
/// two keys, restart `record`. Those two files speak one vocabulary because they are serialized
/// from this one Rust type, so the key names cannot drift apart the way two hand-written shapes
/// eventually do. Both sides normalize executable paths the same way (canonicalized) for the
/// same reason: an entry pasted from a session must match what the predicate compares against,
/// which is the canonical path behind the pid.
///
/// Lists are kept sorted and deduplicated by [`AppIdentities::insert_bundle_id`] and
/// [`AppIdentities::insert_executable`], so a value built by observing many holders serializes
/// deterministically and a diff between two sessions' files means a difference in the world.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppIdentities {
    /// Bundle ids, matched exactly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bundle_ids: Vec<String>,
    /// Executable paths, matched exactly.
    ///
    /// The real executable inside `.app/Contents/MacOS/`, not the bundle directory: the
    /// predicate reads the executable behind the pid, which is what macOS hands out, and both
    /// writers and readers canonicalize so the two agree.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub executables: Vec<PathBuf>,
}

impl AppIdentities {
    /// An empty set.
    ///
    /// `const` so a caller can hold one in a `static` -- which is what lets the record crate's
    /// exclusion-matrix tests keep their shared no-exclusions value rather than rebuilding one
    /// per assertion.
    pub const fn new() -> Self {
        AppIdentities {
            bundle_ids: Vec::new(),
            executables: Vec::new(),
        }
    }

    /// A set spelled out directly, normalized the way the insert methods normalize.
    ///
    /// For callers that have the whole list at once rather than one app at a time -- mostly
    /// tests.
    pub fn from_parts(
        bundle_ids: impl IntoIterator<Item = String>,
        executables: impl IntoIterator<Item = PathBuf>,
    ) -> Self {
        let mut set = AppIdentities::new();
        for id in bundle_ids {
            set.insert_bundle_id(id);
        }
        for exe in executables {
            set.insert_executable(exe);
        }
        set
    }

    /// Adds one bundle id, keeping the list sorted and free of duplicates.
    pub fn insert_bundle_id(&mut self, id: impl Into<String>) {
        insert_sorted(&mut self.bundle_ids, id.into());
    }

    /// Adds one executable path, keeping the list sorted and free of duplicates.
    pub fn insert_executable(&mut self, path: impl Into<PathBuf>) {
        insert_sorted(&mut self.executables, path.into());
    }

    /// Adds every identity in `other`, keeping the sorted-and-unique invariant.
    ///
    /// Used where a set is learned a piece at a time rather than read whole -- a session
    /// accumulating the microphone holders it saw over a meeting.
    pub fn merge(&mut self, other: &AppIdentities) {
        for id in &other.bundle_ids {
            self.insert_bundle_id(id.clone());
        }
        for exe in &other.executables {
            self.insert_executable(exe.clone());
        }
    }

    /// Whether this set names nothing at all. The serialization guard for both files: a
    /// session that observed nothing writes no key, so its `session.json` stays byte-identical
    /// to what builds before this feature wrote.
    pub fn is_empty(&self) -> bool {
        self.bundle_ids.is_empty() && self.executables.is_empty()
    }

    /// Whether `id` is in the set. Exact match only.
    pub fn contains_bundle_id(&self, id: &str) -> bool {
        self.bundle_ids.iter().any(|entry| entry == id)
    }

    /// Whether `path` is in the set. Exact match only; `path` should be the canonicalized
    /// executable behind the pid, which is what entries are normalized to.
    pub fn contains_executable(&self, path: &Path) -> bool {
        self.executables.iter().any(|entry| entry == path)
    }
}

/// Push and de-duplicate in one place, so "sorted and unique" cannot become a rule one writer
/// forgets. Sorted rather than insertion-ordered because these lists are read by eye and
/// diffed: a session's recorded apps should mean the same thing twice.
fn insert_sorted<T: Ord>(list: &mut Vec<T>, value: T) {
    match list.binary_search(&value) {
        Ok(_) => {}
        Err(at) => list.insert(at, value),
    }
}

/// Apps that never count as the mic-activity signal, as named by the user.
///
/// The identity lists are [`AppIdentities`]; what this type adds is the load policy in
/// [`AppExclusions::read_or_empty`] and the meaning "never counts". Read-only from meethook's
/// side: nothing here ever writes `exclusions.json`, so there is no `Serialize` on this type
/// and no schema version to carry back out.
///
/// Consulted per process object by the record crate's predicate: an entry fires only when
/// the fact it keys on is present (a bundle-id entry cannot fire for a process whose bundle
/// id is unreadable; its executable entry can still fire).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppExclusions(AppIdentities);

impl std::ops::Deref for AppExclusions {
    type Target = AppIdentities;

    /// The user's identities, and with them the membership tests. Read access only by
    /// design: adding to an exclusion list means editing `exclusions.json`, and there is no
    /// API here that pretends otherwise.
    fn deref(&self) -> &AppIdentities {
        &self.0
    }
}

impl AppExclusions {
    /// Builds the set from identities whose executable paths are already canonicalized.
    ///
    /// The one non-loading way to construct one, for callers that resolved the list elsewhere
    /// (and for tests). Loading from disk goes through [`AppExclusions::read_or_empty`], which
    /// canonicalizes on the way in.
    /// `const` for the same reason [`AppIdentities::new`] is: a test fixture can be a `static`.
    pub const fn from_identities(identities: AppIdentities) -> Self {
        AppExclusions(identities)
    }

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
        let mut identities = AppIdentities::new();
        file.bundle_ids
            .into_iter()
            .for_each(|id| identities.insert_bundle_id(id));
        // Canonicalized where possible because the process paths the predicate compares
        // against arrive canonicalized; an entry that does not resolve is kept raw and
        // simply will not match, which is the honest outcome for a path that is not
        // there.
        file.executables
            .into_iter()
            .map(|p| match std::fs::canonicalize(&p) {
                Ok(canonical) => canonical,
                Err(_) => p,
            })
            .for_each(|p| identities.insert_executable(p));
        Ok(AppExclusions(identities))
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
            *read,
            // The path does not exist in the sandbox, so it is kept raw rather than
            // canonicalized.
            AppIdentities::from_parts(
                ["com.example.voiceink".to_owned()],
                [PathBuf::from(
                    "/Applications/VoiceInk.app/Contents/MacOS/VoiceInk"
                )],
            )
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

    #[test]
    fn inserts_sort_and_deduplicate() {
        let mut set = AppIdentities::new();
        set.insert_bundle_id("com.zoom.xcode");
        set.insert_bundle_id("com.example.voiceink");
        set.insert_bundle_id("com.zoom.xcode");
        set.insert_executable("/Applications/Zoom.app/Contents/MacOS/Zoom");
        set.insert_executable("/Applications/VoiceInk.app/Contents/MacOS/VoiceInk");
        set.insert_executable("/Applications/Zoom.app/Contents/MacOS/Zoom");

        assert_eq!(
            set.bundle_ids,
            ["com.example.voiceink", "com.zoom.xcode"].map(String::from)
        );
        assert_eq!(set.executables.len(), 2);
        assert!(set.executables[0] < set.executables[1]);
    }

    /// The contract the whole shared type exists for: what a session writes under `mic_apps`
    /// can be pasted under `exclusions.json`'s own keys unchanged.
    ///
    /// Asserted against literal JSON rather than against this crate's own writer, since a
    /// writer checked against itself proves nothing about the file a human edits by hand.
    #[test]
    fn a_sessions_mic_apps_block_is_valid_exclusions_json() {
        let observed = AppIdentities::from_parts(
            ["com.microsoft.teams2".to_owned()],
            [PathBuf::from(
                "/Applications/Microsoft Teams (work or school).app/Contents/MacOS/Teams",
            )],
        );
        // The shape `SessionMetadata::mic_apps` serializes to, wrapped in the schema version
        // `exclusions.json` demands.
        let block = serde_json::to_value(&observed).unwrap();
        let mut file = block.as_object().unwrap().clone();
        file.insert(
            "schema_version".to_owned(),
            serde_json::json!(EXCLUSIONS_SCHEMA_VERSION),
        );
        let dir = tempfile::tempdir().unwrap();
        let paths = bare_paths(dir.path());
        std::fs::write(
            paths.exclusions_json(),
            serde_json::to_vec_pretty(&file).unwrap(),
        )
        .unwrap();

        let read = AppExclusions::read_or_empty(&paths).unwrap();
        assert!(read.contains_bundle_id("com.microsoft.teams2"));
        assert_eq!(read.bundle_ids, observed.bundle_ids);
    }
}
