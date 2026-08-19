use std::io;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, write_atomic};

/// Bumped whenever `cleaning.json`'s shape changes incompatibly.
///
/// Separate from every other schema version in this crate for the reason they are all
/// separate from each other: this file is written by `transcribe`'s AEC pre-pass and read by
/// nobody in-process today, on its own schedule.
pub const CLEANING_SCHEMA_VERSION: u32 = 1;

/// Why cancellation did not run. Every variant is a normal outcome of a real recording, not an
/// error -- `mic.cleaned.wav` is written on every one of these paths, so the rest of the
/// pipeline has exactly one input to read and no branch to get wrong.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PassThrough {
    /// No speaker track, or one that is digital silence throughout -- nothing was playing, so
    /// nothing bled.
    NoReference,
    /// The two tracks would not align, so there is no reference to subtract.
    Unalignable(NotMeasurable),
    /// AEC3 itself could not produce a cancelled track.
    Cancellation(CancellationFailure),
}

/// Why the two tracks would not align.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotMeasurable {
    /// The tracks cannot supply even one window with the full search range inside them.
    TracksTooShort,
    /// Too few windows produced a peak that survived the guards. The headphones case, where
    /// nothing bled into the microphone, lands here.
    TooFewWindows { survived: usize, examined: usize },
    /// Windows that individually looked convincing disagreed with each other, so no single lag
    /// describes the recording -- the output device changed mid-meeting, or the peaks were
    /// coincidence.
    InconsistentWindows { windows: usize, spread_samples: i64 },
}

/// Why the echo canceller itself failed, as distinct from [`Cleaning::Cancelled`]'s
/// `erle_db: None`, which means it ran to completion but never converged far enough to report
/// a figure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CancellationFailure {
    /// The canceller could not be constructed.
    ProcessorUnavailable,
    /// A frame errored partway through the track.
    FrameFailed,
}

/// What the AEC pre-pass did to one session's mic track, in enough detail to say something
/// truthful to the user without reaching back into the audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Cleaning {
    /// The reference lined up and AEC3 ran over the whole track.
    Cancelled {
        /// How much later the mic track heard the far end than the speaker track recorded it,
        /// in samples, at the midpoint in time of the windows it was measured over.
        lag_samples: i64,
        /// How far the windows the lag was measured over disagreed, after fitting out
        /// whatever drift `drift_ms_per_hour` reports. A wide spread on an accepted
        /// measurement means the correlation was marginal.
        spread_samples: i64,
        /// How fast `lag_samples` itself was moving across the recording, in milliseconds per
        /// hour -- positive if the lag grew over the meeting, negative if it shrank.
        drift_ms_per_hour: f64,
        /// Median echo return loss enhancement across the track, in dB, or `None` if AEC3 never
        /// converged far enough to report one.
        erle_db: Option<f64>,
    },
    /// The mic track was passed through untouched, for this reason.
    PassedThrough { reason: PassThrough },
}

/// `cleaning.json`: a durable answer to "was this session's mic track cleaned, and by how
/// much", written by `transcribe`'s AEC pre-pass.
///
/// Deliberately its own type rather than a `Serialize` bolted onto `meethook-transcribe`'s
/// in-process `aec::Cleaning`: this crate owns every on-disk shape in the session directory,
/// independent of the crate that produces the value, and `meethook-session` cannot depend on
/// `meethook-transcribe` -- the dependency runs the other way. `meethook-transcribe` projects
/// its richer type down to this one with `From` impls; this crate stays ignorant of
/// `meethook-transcribe` entirely.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CleaningRecord {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub outcome: Cleaning,
}

impl CleaningRecord {
    pub fn new(session_id: SessionId, outcome: Cleaning) -> Self {
        CleaningRecord {
            schema_version: CLEANING_SCHEMA_VERSION,
            session_id,
            outcome,
        }
    }

    pub fn write(&self, paths: &SessionPaths) -> Result<()> {
        let path = paths.cleaning_json();
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(&path, e))?;
        json.push(b'\n');
        write_atomic(&path, &json)
    }

    /// Reads this session's cleaning record, treating "there isn't one" as `None` rather than
    /// an error.
    ///
    /// Absent is the normal state of every session transcribed before this file existed --
    /// there is no honest outcome to invent in its place, unlike
    /// [`crate::SpeakerNames::read_or_empty`], where an empty list of hand-given names really
    /// is what "nobody has named a voice yet" looks like. There is no such neutral `Cleaning`
    /// value, so `Option` is the honest return type here instead.
    ///
    /// A file that exists and does not parse stays an error: a `cleaning.json` that is present
    /// but unreadable is not the ordinary "session predates this record" case, and folding it
    /// into `None` would hide a real problem behind a normal one.
    pub fn read_if_present(paths: &SessionPaths) -> Result<Option<CleaningRecord>> {
        let path = paths.cleaning_json();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::io(&path, e)),
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|e| Error::json(&path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_id() -> SessionId {
        SessionId::parse("20260809-052600").unwrap()
    }

    fn cancelled() -> CleaningRecord {
        CleaningRecord::new(
            session_id(),
            Cleaning::Cancelled {
                lag_samples: 1920,
                spread_samples: 12,
                drift_ms_per_hour: -3.4,
                erle_db: Some(26.7),
            },
        )
    }

    #[test]
    fn a_written_cancelled_record_reads_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());
        let written = cancelled();

        written.write(&paths).unwrap();

        let read = CleaningRecord::read_if_present(&paths).unwrap().unwrap();
        assert_eq!(read, written);
        assert_eq!(read.schema_version, CLEANING_SCHEMA_VERSION);
    }

    /// Every reason enumerated in the AC, not just the happy path: each one round-trips.
    #[test]
    fn every_pass_through_reason_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        for reason in [
            PassThrough::NoReference,
            PassThrough::Unalignable(NotMeasurable::TooFewWindows {
                survived: 2,
                examined: 40,
            }),
            PassThrough::Unalignable(NotMeasurable::InconsistentWindows {
                windows: 5,
                spread_samples: 900,
            }),
            PassThrough::Cancellation(CancellationFailure::FrameFailed),
        ] {
            let written = CleaningRecord::new(
                session_id(),
                Cleaning::PassedThrough {
                    reason: reason.clone(),
                },
            );
            written.write(&paths).unwrap();

            let read = CleaningRecord::read_if_present(&paths).unwrap().unwrap();
            assert_eq!(read, written, "reason {reason:?} did not round-trip");
        }
    }

    /// AC #3, at the unit level: a session transcribed before this file existed simply has no
    /// `cleaning.json`, and that has to read as "nothing recorded" rather than as corruption.
    #[test]
    fn an_absent_file_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        assert_eq!(CleaningRecord::read_if_present(&paths).unwrap(), None);
    }

    /// The other half of that: a file that is *there* and unreadable is not the ordinary
    /// absent case, and must not be silently downgraded into one.
    #[test]
    fn a_malformed_file_is_an_error_rather_than_none() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());
        std::fs::write(paths.cleaning_json(), b"{ this is not json").unwrap();

        let error = CleaningRecord::read_if_present(&paths).unwrap_err();

        assert!(
            matches!(error, Error::Json { .. }),
            "expected a JSON error, got {error:?}"
        );
        assert!(error.to_string().contains("cleaning.json"), "{error}");
    }

    /// Rewriting is what a `transcribe --force` re-run does, and it must not append or merge.
    #[test]
    fn rewriting_replaces_the_previous_contents() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        cancelled().write(&paths).unwrap();
        let second = CleaningRecord::new(
            session_id(),
            Cleaning::PassedThrough {
                reason: PassThrough::NoReference,
            },
        );
        second.write(&paths).unwrap();

        let read = CleaningRecord::read_if_present(&paths).unwrap().unwrap();
        assert_eq!(read, second);
    }
}
