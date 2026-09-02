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

/// How strongly a session's start supports the meeting it was matched to.
///
/// **The fit is about the start, and only about the start.** A session's *end* is not an input
/// to any variant here, deliberately: meetings run long routinely, so a recording that
/// overruns its event end is the most ordinary outcome there is and must never fit worse for
/// it. A coverage ratio -- session length over event length -- would score exactly that
/// ordinary case as a weak match, which is why there is not one.
///
/// What the start can say is which of three shapes the match has: the recording began with the
/// meeting, it began materially after the meeting had already started, or it was never
/// contained by the meeting at all and was matched by adjacency. The middle case is the one
/// worth marking. "Joined the 2:00 standup twenty minutes late" and "left the standup and took
/// an unrelated call" are identical in start and end times, so no rule sharp enough to reject
/// the second survives the first -- which is far more common. This type therefore does not
/// change *which* meeting is selected. It records how strong the claim is, so a consumer can
/// tell a meeting a session *was* from a meeting a session merely *sat inside*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeetingFit {
    /// Nobody scored this match: a `session.json` written before fits existed, or a meeting
    /// that never went through the recorder's selection. Both read the same way downstream,
    /// which is the point of one variant covering them -- neither is evidence of a good match.
    #[default]
    Unknown,
    /// The recording began at, or shortly after, the meeting's start, inside it. The strong
    /// case, and the one an ordinary join produces.
    Started,
    /// The recording began before the meeting started, and was matched to it by proximity.
    /// Strong -- joining a call early is ordinary -- but distinct from [`MeetingFit::Started`]
    /// because it was never contained by the event.
    StartedEarly,
    /// The recording began well after the meeting had already started, though still inside it.
    /// A late join or an entirely unrelated call; the calendar cannot tell which, and saying so
    /// is the whole of what this variant claims.
    JoinedLate,
    /// The recording began after the meeting had already ended, and was matched to it by
    /// proximity. The ad-hoc call that inherits the invite it happened to follow.
    AfterEnd,
    /// A human said this is the meeting. The strongest claim available, and the only one here
    /// not derived from arithmetic over timestamps: somebody who was in the call is better
    /// evidence about what it was than any rule over start and end times can be. Written only
    /// by [`SessionMetadata::label_by_hand`], and never produced by the recorder's own lookup.
    Confirmed,
}

impl MeetingFit {
    /// Every variant, so a caller iterating outcomes cannot list only the ones it remembered.
    pub const ALL: [MeetingFit; 6] = [
        MeetingFit::Unknown,
        MeetingFit::Started,
        MeetingFit::StartedEarly,
        MeetingFit::JoinedLate,
        MeetingFit::AfterEnd,
        MeetingFit::Confirmed,
    ];

    /// Whether the session's start actually supports this being the meeting.
    ///
    /// Written as an exhaustive `match` rather than a `matches!` with a wildcard so that a
    /// variant added later cannot default into "strong" by omission -- it will not compile
    /// until somebody decides.
    /// [`MeetingFit::Confirmed`] is strong, and deliberately so: withholding
    /// [`Meeting::speaker_roster`] from the one label somebody typed on purpose would invert
    /// the point of the guard, which is to keep a *guess* from seeding an identification pass.
    pub fn is_strong(&self) -> bool {
        match self {
            MeetingFit::Started | MeetingFit::StartedEarly | MeetingFit::Confirmed => true,
            MeetingFit::Unknown | MeetingFit::JoinedLate | MeetingFit::AfterEnd => false,
        }
    }

    /// How to qualify this meeting where it is shown to a person, or `None` when the fit is
    /// strong enough to state the meeting plainly.
    ///
    /// The wording lives here rather than in the CLI, on the precedent this crate's other
    /// user-facing values set: the library owns the sentence, the caller owns the stream, and
    /// the sentence is then testable without a terminal. It names no attendee and no invite
    /// content, so it is safe on any surface the title itself is safe on.
    pub fn caveat(&self) -> Option<&'static str> {
        match self {
            MeetingFit::Started | MeetingFit::StartedEarly | MeetingFit::Confirmed => None,
            MeetingFit::JoinedLate => {
                Some("uncertain: the recording began after this meeting had started")
            }
            MeetingFit::AfterEnd => {
                Some("uncertain: the recording began after this meeting had ended")
            }
            MeetingFit::Unknown => {
                Some("unverified: this session was recorded before meethook scored the match")
            }
        }
    }
}

/// The calendar meeting a session was recorded during.
///
/// The attendee list is the load-bearing part rather than decoration: it is the per-session
/// set of people who could plausibly be speaking, which is what a speaker-identification
/// pass needs to avoid matching a voice against every person ever enrolled. It is reachable
/// as a roster only through [`Meeting::speaker_roster`], which consults [`Meeting::fit`] --
/// see that method for why.
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
    /// Private, and the one field here that is: the list is reachable as a *roster* only
    /// through [`Meeting::speaker_roster`]. It still serializes exactly as it always did, so a
    /// user's own transcript template can print the names into their own notes -- the guard is
    /// on code paths that consume the list to decide who is speaking, not on the file.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    attendees: Vec<Attendee>,
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
    /// How strongly the session's start supports this being the meeting.
    ///
    /// `default` on the way in, with no `skip_serializing_if`: every meeting this build writes
    /// has a fit, so an absent key only ever means a `session.json` written before fits
    /// existed -- which reads as [`MeetingFit::Unknown`], i.e. not a strong match. Living on
    /// [`Meeting`] rather than on [`SessionMetadata`] is what keeps a session with no meeting
    /// writing byte-identical JSON: there is no meeting to hang a fit on.
    #[serde(default)]
    pub fit: MeetingFit,
}

impl Meeting {
    /// The identity of a calendar event, with nobody attached and nothing scored.
    ///
    /// A constructor rather than a struct literal because the `attendees` field is private;
    /// the people, the invite fields and the fit are attached by the builders below, in the
    /// shape [`SessionMetadata::with_meeting`] already establishes here.
    pub fn new(
        event_id: String,
        title: String,
        calendar: String,
        start: Timestamp,
        end: Timestamp,
    ) -> Self {
        Meeting {
            title,
            start,
            end,
            calendar,
            organizer: None,
            attendees: Vec::new(),
            url: None,
            location: None,
            notes: None,
            event_id,
            fit: MeetingFit::Unknown,
        }
    }

    #[must_use]
    pub fn with_people(mut self, organizer: Option<Attendee>, attendees: Vec<Attendee>) -> Self {
        self.organizer = organizer;
        self.attendees = attendees;
        self
    }

    /// Replace the attendee list, leaving every other field -- in particular `fit` and
    /// `organizer` -- untouched.
    ///
    /// The mutation path the record TUI's roster editor (TASK-056.02) rides: an edited roster
    /// must keep the provenance of the meeting it came from. Deliberately not
    /// [`Meeting::with_people`] (which also replaces the organizer) and deliberately not
    /// [`SessionMetadata::label_by_hand`] (which force-stamps [`MeetingFit::Confirmed`] -- a
    /// human correcting the roster is not a human confirming the meeting, decision-009).
    /// Because this touches only the private `attendees` field, [`Meeting::speaker_roster`]
    /// remains the sole accessor and its `is_strong()` gate applies to an edited roster with
    /// no further work.
    #[must_use]
    pub fn with_attendees(mut self, attendees: Vec<Attendee>) -> Self {
        self.attendees = attendees;
        self
    }

    #[must_use]
    pub fn with_invite(
        mut self,
        url: Option<String>,
        location: Option<String>,
        notes: Option<String>,
    ) -> Self {
        self.url = url;
        self.location = location;
        self.notes = notes;
        self
    }

    #[must_use]
    pub fn with_fit(mut self, fit: MeetingFit) -> Self {
        self.fit = fit;
        self
    }

    /// The people who could plausibly be speaking in this session, or `None` when the match is
    /// too weak to say.
    ///
    /// The only way to obtain the attendee list as a whole, and the reason the field is
    /// private. A per-session attendee whitelist is what fixes cross-session speaker
    /// contamination -- doc-001 records that finding -- so seeding one from a meeting the
    /// session merely *sat inside* is that same contamination arriving through the calendar
    /// instead. Routing the roster through [`Meeting::fit`] means a future
    /// speaker-identification pass cannot consume a weak match by not knowing to ask.
    pub fn speaker_roster(&self) -> Option<&[Attendee]> {
        self.fit.is_strong().then_some(self.attendees.as_slice())
    }

    /// How many people were invited -- what a diagnostic line needs, and all it needs.
    ///
    /// Deliberately not a way around [`Meeting::speaker_roster`]: a count names nobody.
    pub fn attendee_count(&self) -> usize {
        self.attendees.len()
    }
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
    /// A human said this session was not recorded during any meeting.
    ///
    /// True only when `meeting` is `None`, and written only by
    /// [`SessionMetadata::label_by_hand`] -- the two are set together so they cannot disagree.
    /// It is the cleared half of the provenance whose attached half is
    /// [`MeetingFit::Confirmed`]: a cleared label has no [`Meeting`] to hang a fit on, so the
    /// fact has to live here instead.
    ///
    /// Skipped when false, which is what keeps a session nobody has corrected writing
    /// byte-identical JSON to what this build's predecessors wrote -- the same equivalence
    /// that keeps [`SESSION_SCHEMA_VERSION`] where it is. Only a session somebody actually
    /// cleared gains the key, and an old file with neither key reads as not settled by hand.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub meeting_cleared: bool,

    /// The user's assertion that this session's speaker track is one person: their name.
    ///
    /// Absent, not null, when nobody has asserted it: a pre-assertion file reads as
    /// unasserted, and a session nobody has asserted about writes byte-identical JSON to what
    /// this build's predecessors wrote -- the same equivalence that keeps
    /// [`SESSION_SCHEMA_VERSION`] where it is. Written only by
    /// [`SessionMetadata::assert_one_remote_speaker`]; re-asserting a different name
    /// overwrites, and the run converges on the new one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub one_remote_speaker: Option<String>,
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
            meeting_cleared: false,
            one_remote_speaker: None,
        }
    }

    /// Attaches the meeting this session was recorded during, if one was found.
    ///
    /// A builder rather than a fifth parameter on [`SessionMetadata::new`]: three of that
    /// constructor's four call sites have no calendar in reach at all -- `transcribe`'s
    /// importer and two test helpers -- and widening the signature would make all of them
    /// pass a `None` to say so.
    ///
    /// **A no-op on a session whose label a human settled.** A label somebody set or cleared
    /// is the strongest evidence this tool holds about what a session was, and an automatic
    /// lookup must never overwrite it by guessing again. The guard lives here rather than in
    /// any future re-guessing pass because this is the one door such a pass would come
    /// through -- there is no re-guessing pass today, and the cheap moment to settle the rule
    /// is before there is one. It is unreachable from the recorder, which only ever calls this
    /// on a freshly [`SessionMetadata::new`]'d value.
    #[must_use]
    pub fn with_meeting(mut self, meeting: Option<Meeting>) -> Self {
        if self.meeting_settled_by_hand() {
            return self;
        }
        self.meeting = meeting;
        self
    }

    /// Records the meeting label a human decided on, or the absence of one they decided on.
    ///
    /// `Some(meeting)` stores it as [`MeetingFit::Confirmed`] whatever fit it arrived with --
    /// a candidate offered for correction carries [`MeetingFit::Unknown`], and what makes this
    /// one a match is the person choosing it. `None` drops the meeting and records that the
    /// absence was decided rather than merely never found.
    ///
    /// The only way to write either half, which is what keeps them consistent: a caller cannot
    /// construct a session that both names a meeting and claims one was cleared.
    pub fn label_by_hand(&mut self, meeting: Option<Meeting>) {
        match meeting {
            Some(meeting) => {
                self.meeting = Some(meeting.with_fit(MeetingFit::Confirmed));
                self.meeting_cleared = false;
            }
            None => {
                self.meeting = None;
                self.meeting_cleared = true;
            }
        }
    }

    /// Records the user's assertion that this session's speaker track is one person, `name`.
    ///
    /// Like a hand-settled meeting label, only the user's word settles this: enrollment runs
    /// apply it on the user's say-so, and nothing else may invent or clear it -- absence means
    /// unasserted, which is how every pre-assertion file reads. Re-asserting a different name
    /// simply overwrites; the next run re-offers every voice against the new one and converges.
    pub fn assert_one_remote_speaker(&mut self, name: String) {
        self.one_remote_speaker = Some(name);
    }

    /// Whether a human has decided this session's meeting label, either way.
    ///
    /// The one question a re-guessing pass has to ask before writing, and the reason both
    /// halves of the provenance are readable through one call rather than two fields.
    pub fn meeting_settled_by_hand(&self) -> bool {
        self.meeting_cleared
            || self
                .meeting
                .as_ref()
                .is_some_and(|meeting| meeting.fit == MeetingFit::Confirmed)
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
