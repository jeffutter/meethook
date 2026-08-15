use std::path::Path;

use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, write_atomic};

/// Bumped whenever `session.json`'s shape changes incompatibly.
///
/// `transcript.json` carries its own version; see [`crate::TRANSCRIPT_SCHEMA_VERSION`].
pub const SESSION_SCHEMA_VERSION: u32 = 1;

/// One track's first-sample host timestamp, in the form the hardware reported it.
///
/// Ticks are stored raw, together with the `mach_timebase_info` ratio needed to interpret
/// them, rather than pre-converted to nanoseconds. Converting at write time would round
/// once here and again when `transcribe` computes the mic/speaker offset; keeping the
/// native tick count lets `transcribe` do a single exact rational conversion.
///
/// Nanoseconds = `host_ticks * timebase_numer / timebase_denom`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackSync {
    /// Raw `mach_absolute_time()` value at the track's first delivered sample.
    pub host_ticks: u64,
    /// `mach_timebase_info.numer`.
    pub timebase_numer: u32,
    /// `mach_timebase_info.denom`.
    pub timebase_denom: u32,
}

/// Where one participant of a meeting stood on it.
///
/// Mirrors `EKParticipantStatus` minus its two reminder-only members (`Completed`,
/// `InProcess`), which cannot occur on an event. An unrecognized value from a future macOS
/// reads as [`AttendeeStatus::Unknown`] at the recorder rather than being stored, so this
/// list is closed on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttendeeStatus {
    Unknown,
    Pending,
    Accepted,
    Declined,
    Tentative,
    Delegated,
}

/// One person invited to a meeting.
///
/// `is_you` is the calendar's own answer (`EKParticipant.isCurrentUser`), stored rather than
/// re-derived from an address: it is what decides whether *this user* declined the meeting,
/// and what a later per-session speaker whitelist needs in order to leave the local speaker
/// out of the candidate list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attendee {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The bare address, with any `mailto:` scheme already stripped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    pub status: AttendeeStatus,
    pub is_you: bool,
}

/// The calendar meeting a session was recorded during.
///
/// The attendee list is the load-bearing part rather than decoration: it is the per-session
/// set of people who could plausibly be speaking, which is what a speaker-identification
/// pass needs to avoid matching a voice against every person ever enrolled.
///
/// `notes` is the invite body, stored because it is the field most likely to answer "what
/// was this meeting about" for a transcript that has outlived anyone's memory of it. It is
/// also the least predictable thing here: meeting bodies routinely carry dial-in numbers,
/// conference PINs and one-time passcodes, and every other field on this struct is short and
/// structural by comparison. Two rules follow from that and are enforced elsewhere:
///
/// - It reaches `session.json` and nothing else -- no log line, no terminal, no error report.
///   `meethook-record`'s calendar debug output renders a meeting without it, under test.
/// - `session.json` is therefore a file with meeting *content* in it, not just metadata
///   about a recording. That matters wherever a session directory gets synced or shared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Meeting {
    pub title: String,
    pub start: Timestamp,
    pub end: Timestamp,
    /// The containing calendar's display name ("Work", "Personal"), not its identifier.
    pub calendar: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub organizer: Option<Attendee>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attendees: Vec<Attendee>,
    /// The event's own URL, which for most video-conferencing invites is the join link.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Where the meeting is: a room name, a street address, or a join URL, depending
    /// entirely on who wrote the invite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    /// The invite body, verbatim and unparsed -- the agenda, and whatever else the organizer
    /// put there. Absent rather than empty when the event has none. See this struct's own
    /// documentation for what may and may not be done with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
    /// `EKEvent.eventIdentifier`, stored so a later pass can re-resolve the event against
    /// the live calendar rather than trusting this snapshot of it.
    pub event_id: String,
}

/// `session.json`: the marker that a session shut down cleanly, plus the sync data
/// `transcribe` needs to put the two tracks on one timeline.
///
/// Sample rate, channel count, and bit depth are deliberately absent. They live in the WAV
/// headers, which are the authority; duplicating them here would create two sources of
/// truth that can silently disagree after any format change in the recorder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: SessionId,
    pub schema_version: u32,
    /// Session start as an unambiguous instant, RFC 3339. The local wall-clock time is
    /// already encoded in the session id.
    pub start_time: Timestamp,
    pub mic: TrackSync,
    pub speaker: TrackSync,
    /// The calendar meeting this session was recorded during, if one could be identified.
    ///
    /// Absent, not null, when there is none: a session recorded outside any meeting writes
    /// byte-identical JSON to what this build's predecessors wrote. Together with `default`
    /// on the way in, that is what keeps [`SESSION_SCHEMA_VERSION`] where it is -- the
    /// version marks *incompatible* shape changes, and this field is readable in both
    /// directions by builds on either side of it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meeting: Option<Meeting>,
}

impl SessionMetadata {
    pub fn new(
        session_id: SessionId,
        start_time: Timestamp,
        mic: TrackSync,
        speaker: TrackSync,
    ) -> Self {
        SessionMetadata {
            session_id,
            schema_version: SESSION_SCHEMA_VERSION,
            start_time,
            mic,
            speaker,
            meeting: None,
        }
    }

    /// Attaches the meeting this session was recorded during, if one was found.
    ///
    /// A builder rather than a fifth parameter on [`SessionMetadata::new`]: three of that
    /// constructor's four call sites have no calendar in reach at all -- `transcribe`'s
    /// importer and two test helpers -- and widening the signature would make all of them
    /// pass a `None` to say so.
    #[must_use]
    pub fn with_meeting(mut self, meeting: Option<Meeting>) -> Self {
        self.meeting = meeting;
        self
    }

    /// Writes `session.json` atomically.
    ///
    /// Atomicity is what makes presence-of-file a trustworthy "session is complete" marker:
    /// a reader either sees no file or sees a whole one, never a truncated fragment that
    /// would classify a crashed session as valid.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(path, e))?;
        json.push(b'\n');
        write_atomic(path, &json)
    }

    pub fn read(path: &Path) -> Result<SessionMetadata> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
    }
}
