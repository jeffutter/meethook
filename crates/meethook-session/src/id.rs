use std::fmt;
use std::fs;
use std::io::ErrorKind;

use jiff::Zoned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Error, Paths, Result, SessionPaths};

/// How many same-second collisions we are willing to absorb before giving up. Reaching
/// this means something is generating sessions in a loop, which is a bug worth surfacing
/// rather than papering over with an unbounded retry.
const MAX_COLLISION_SUFFIX: u32 = 999;

/// A session directory name: local-time `YYYYMMDD-HHMMSS`, optionally `-N` when a session
/// starts in the same second as an existing one.
///
/// Local time rather than UTC because the id is the primary thing the user reads and types;
/// the unambiguous instant lives in `session.json`'s `start_time`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(String);

impl SessionId {
    /// Validates the `YYYYMMDD-HHMMSS[-N]` shape.
    ///
    /// Discovery uses this to ignore anything else living under `sessions/`, and the CLI
    /// uses it to reject a typo'd id before touching the filesystem. It checks shape, not
    /// whether the date is real -- a directory named `20261332-256199` is nobody's data.
    pub fn parse(s: &str) -> Result<SessionId> {
        let malformed = || Error::MalformedSessionId(s.to_string());

        let (date, rest) = s.split_once('-').ok_or_else(malformed)?;
        // `rest` is either `HHMMSS` or `HHMMSS-N`.
        let (time, suffix) = match rest.split_once('-') {
            Some((time, suffix)) => (time, Some(suffix)),
            None => (rest, None),
        };

        let digits =
            |part: &str, len: usize| part.len() == len && part.bytes().all(|b| b.is_ascii_digit());
        if !digits(date, 8) || !digits(time, 6) {
            return Err(malformed());
        }
        if let Some(suffix) = suffix {
            let numeric = !suffix.is_empty()
                && suffix.bytes().all(|b| b.is_ascii_digit())
                && !suffix.starts_with('0');
            if !numeric {
                return Err(malformed());
            }
        }

        Ok(SessionId(s.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Serialize for SessionId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for SessionId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        SessionId::parse(&raw).map_err(serde::de::Error::custom)
    }
}

/// Allocates and creates a fresh session directory for `now`, returning its id and paths.
///
/// Creating the directory *is* the collision check: `create_dir` fails with `AlreadyExists`
/// rather than succeeding silently, so there is no stat-then-create window in which two
/// recorders could agree on the same id. On collision the id gains a `-1`, `-2`, ... suffix.
pub fn create_session_dir(paths: &Paths, now: &Zoned) -> Result<(SessionId, SessionPaths)> {
    let sessions_dir = paths.sessions_dir();
    fs::create_dir_all(&sessions_dir).map_err(|e| Error::io(&sessions_dir, e))?;

    let base = now.strftime("%Y%m%d-%H%M%S").to_string();

    for attempt in 0..=MAX_COLLISION_SUFFIX {
        let id = if attempt == 0 {
            base.clone()
        } else {
            format!("{base}-{attempt}")
        };
        let dir = sessions_dir.join(&id);
        match fs::create_dir(&dir) {
            Ok(()) => return Ok((SessionId(id), SessionPaths::new(dir))),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(Error::io(&dir, e)),
        }
    }

    Err(Error::SessionIdExhausted {
        base,
        tried: MAX_COLLISION_SUFFIX + 1,
    })
}

/// Removes a session directory that never became a session.
///
/// The inverse of [`create_session_dir`], for the window between that call and the point where
/// a recorder has both of its capture engines live. A start that fails inside that window
/// leaves two WAV headers and no `session.json` behind, and since such a start is retried,
/// it leaves one such directory per attempt -- on-disk litter that reads exactly like a
/// recording that went wrong rather than one that never began.
///
/// Deliberately narrow, and deliberately infallible:
///
/// - Only the two track files a start can have created are removed, by name, and the directory
///   goes with `remove_dir` rather than `remove_dir_all`. Anything else inside means this is
///   not the directory this function believes it is, and `remove_dir` refuses a non-empty one,
///   so the surprising case leaves the data alone instead of deleting it.
/// - Nothing is reported, because there is nothing a caller could do with it. The error that
///   prompted the discard is the one worth surfacing; returning a `Result` here would only
///   invite a failure to tidy up to shadow the failure that mattered.
pub fn discard_session_dir(paths: &SessionPaths) {
    for track in [paths.mic_wav(), paths.speaker_wav()] {
        let _ = fs::remove_file(track);
    }
    let _ = fs::remove_dir(paths.dir());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_plain_and_suffixed_ids() {
        assert_eq!(
            SessionId::parse("20260809-052600").unwrap().as_str(),
            "20260809-052600"
        );
        assert_eq!(
            SessionId::parse("20260809-052600-1").unwrap().as_str(),
            "20260809-052600-1"
        );
        assert_eq!(
            SessionId::parse("20260809-052600-42").unwrap().as_str(),
            "20260809-052600-42"
        );
    }

    #[test]
    fn rejects_non_session_directory_names() {
        for garbage in [
            "",
            "sessions",
            "models",
            ".DS_Store",
            "2026089-052600",    // 7-digit date
            "20260809-05260",    // 5-digit time
            "20260809_052600",   // wrong separator
            "20260809-052600-",  // empty suffix
            "20260809-052600-0", // leading-zero suffix is not one we ever emit
            "20260809-052600-x", // non-numeric suffix
            "20260809-05260a",
        ] {
            assert!(
                SessionId::parse(garbage).is_err(),
                "expected {garbage:?} to be rejected"
            );
        }
    }

    /// The ordinary case: a start that brought both writers up and then failed leaves the two
    /// WAV headers, and the discard takes the whole directory with them.
    #[test]
    fn discarding_removes_a_directory_holding_only_the_two_track_headers() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let now: Zoned = "2026-08-14T16:45:31-05:00[America/Chicago]"
            .parse()
            .unwrap();
        let (_id, session) = create_session_dir(&paths, &now).unwrap();
        fs::write(session.mic_wav(), b"RIFF").unwrap();
        fs::write(session.speaker_wav(), b"RIFF").unwrap();

        discard_session_dir(&session);

        assert!(
            !session.dir().exists(),
            "the directory outlived the discard"
        );
    }

    /// A start that failed before either writer existed leaves an empty directory, and the
    /// discard has to cope with the files it is asked to remove not being there.
    #[test]
    fn discarding_removes_an_empty_directory() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let now: Zoned = "2026-08-14T16:45:31-05:00[America/Chicago]"
            .parse()
            .unwrap();
        let (_id, session) = create_session_dir(&paths, &now).unwrap();

        discard_session_dir(&session);

        assert!(!session.dir().exists());
    }

    /// The guard that matters. Anything beyond the two track files means this is not the
    /// directory the discard believes it is, and the safe answer is to leave it alone -- so
    /// `remove_dir` rather than `remove_dir_all`, checked here rather than assumed.
    #[test]
    fn discarding_leaves_a_directory_holding_anything_else() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let now: Zoned = "2026-08-14T16:45:31-05:00[America/Chicago]"
            .parse()
            .unwrap();
        let (_id, session) = create_session_dir(&paths, &now).unwrap();
        fs::write(session.mic_wav(), b"RIFF").unwrap();
        fs::write(session.session_json(), b"{}").unwrap();

        discard_session_dir(&session);

        assert!(
            session.dir().exists(),
            "a directory holding session.json was deleted"
        );
        assert!(session.session_json().exists());
    }
}
