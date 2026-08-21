//! Tests for the on-disk session contract.
//!
//! Every test builds its own root under a `TempDir`; nothing here ever touches
//! `~/meethook`.

use std::fs;
use std::path::Path;

use jiff::Timestamp;
use meethook_session::{
    Attendee, AttendeeStatus, Classification, Meeting, MeetingFit, Paths, SESSION_SCHEMA_VERSION,
    SessionId, SessionMetadata, SessionPaths, TrackSync, create_session_dir, discover_sessions,
    write_atomic,
};
use tempfile::TempDir;

fn temp_root() -> (TempDir, Paths) {
    let dir = TempDir::new().expect("temp dir");
    let paths = Paths::new(dir.path().join("meethook"));
    (dir, paths)
}

fn sample_metadata(id: &str) -> SessionMetadata {
    SessionMetadata::new(
        SessionId::parse(id).unwrap(),
        "2026-08-09T05:26:00Z".parse::<Timestamp>().unwrap(),
        TrackSync {
            // Large enough that a trip through f64 would lose the low bits, which is
            // exactly the pre-conversion mistake this field exists to prevent.
            host_ticks: 9_007_199_254_740_993,
            timebase_numer: 125,
            timebase_denom: 3,
        },
        TrackSync {
            host_ticks: 9_007_199_254_740_995,
            timebase_numer: 125,
            timebase_denom: 3,
        },
    )
}

/// Builds a session directory by hand, the way `record` or a crash would leave it.
fn make_session(paths: &Paths, id: &str, files: &[&str]) -> SessionPaths {
    let dir = paths.sessions_dir().join(id);
    fs::create_dir_all(&dir).unwrap();
    let session = SessionPaths::new(&dir);
    for file in files {
        fs::write(dir.join(file), b"placeholder").unwrap();
    }
    session
}

// --- session ids -------------------------------------------------------------------

#[test]
fn session_id_uses_local_time_yyyymmdd_hhmmss() {
    let (_tmp, paths) = temp_root();
    let now = "2026-08-09T05:26:07-04:00[America/Toronto]"
        .parse::<jiff::Zoned>()
        .unwrap();

    let (id, session) = create_session_dir(&paths, &now).unwrap();

    assert_eq!(id.as_str(), "20260809-052607");
    assert!(session.dir().is_dir());
    assert_eq!(session.dir(), paths.sessions_dir().join("20260809-052607"));
}

#[test]
fn same_second_sessions_get_distinct_directories() {
    let (_tmp, paths) = temp_root();
    let now = "2026-08-09T05:26:07-04:00[America/Toronto]"
        .parse::<jiff::Zoned>()
        .unwrap();

    let (first, first_paths) = create_session_dir(&paths, &now).unwrap();
    let (second, second_paths) = create_session_dir(&paths, &now).unwrap();
    let (third, third_paths) = create_session_dir(&paths, &now).unwrap();

    assert_eq!(first.as_str(), "20260809-052607");
    assert_eq!(second.as_str(), "20260809-052607-1");
    assert_eq!(third.as_str(), "20260809-052607-2");

    assert_ne!(first_paths.dir(), second_paths.dir());
    assert_ne!(second_paths.dir(), third_paths.dir());
    assert!(first_paths.dir().is_dir());
    assert!(second_paths.dir().is_dir());
    assert!(third_paths.dir().is_dir());
}

#[test]
fn suffixed_ids_round_trip_through_parse() {
    assert!(SessionId::parse("20260809-052607").is_ok());
    assert!(SessionId::parse("20260809-052607-2").is_ok());
    assert!(SessionId::parse("not-a-session").is_err());
}

// --- metadata round-trip -----------------------------------------------------------

#[test]
fn metadata_round_trips_with_raw_mach_ticks() {
    let metadata = sample_metadata("20260809-052607");
    let json = serde_json::to_string(&metadata).unwrap();
    let decoded: SessionMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, metadata);
    assert_eq!(decoded.schema_version, SESSION_SCHEMA_VERSION);
    assert_eq!(decoded.session_id.as_str(), "20260809-052607");
    assert_eq!(decoded.start_time.to_string(), "2026-08-09T05:26:00Z");

    // Raw ticks, not nanoseconds: the stored value is the mach counter itself, and the
    // timebase ratio needed to interpret it travels alongside it.
    assert_eq!(decoded.mic.host_ticks, 9_007_199_254_740_993);
    assert_eq!(decoded.speaker.host_ticks, 9_007_199_254_740_995);
    assert_eq!(decoded.mic.timebase_numer, 125);
    assert_eq!(decoded.mic.timebase_denom, 3);

    // Precision survived: these differ by 2, which an f64 nanosecond conversion would erase.
    assert_ne!(decoded.mic.host_ticks, decoded.speaker.host_ticks);

    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let object = value.as_object().unwrap();
    for field in [
        "session_id",
        "schema_version",
        "start_time",
        "mic",
        "speaker",
    ] {
        assert!(object.contains_key(field), "missing field {field}");
    }
}

#[test]
fn metadata_never_duplicates_wav_header_fields() {
    let json = serde_json::to_string(&sample_metadata("20260809-052607")).unwrap();
    for banned in [
        "sample_rate",
        "channels",
        "bit_depth",
        "sampleRate",
        "bitDepth",
    ] {
        assert!(
            !json.contains(banned),
            "session.json must not duplicate {banned}; it belongs to the WAV header"
        );
    }
}

// --- the meeting field, and its compatibility in both directions ---------------------

fn sample_meeting() -> Meeting {
    Meeting::new(
        "EVENT-ABC".to_owned(),
        "Incident review".to_owned(),
        "Work".to_owned(),
        "2026-08-09T05:00:00Z".parse().unwrap(),
        "2026-08-09T06:00:00Z".parse().unwrap(),
    )
    .with_people(
        Some(Attendee {
            name: Some("Ada Lovelace".to_owned()),
            email: Some("ada@example.com".to_owned()),
            status: AttendeeStatus::Accepted,
            is_you: false,
        }),
        vec![Attendee {
            name: Some("Grace Hopper".to_owned()),
            email: Some("grace@example.com".to_owned()),
            status: AttendeeStatus::Tentative,
            is_you: true,
        }],
    )
    .with_invite(
        Some("https://example.com/j/12345".to_owned()),
        Some("Babbage Room".to_owned()),
        Some("Agenda: the pager, then the fix".to_owned()),
    )
    .with_fit(MeetingFit::Started)
}

/// The old-file/new-build direction, written as a literal rather than as a
/// re-serialization: a re-serialization would track the struct through every future change
/// and so could never fail, which is the opposite of what this asserts.
#[test]
fn session_json_written_before_meetings_still_reads() {
    let (_tmp, paths) = temp_root();
    let session = make_session(&paths, "20260809-052607", &[]);
    let before = r#"{
      "session_id": "20260809-052607",
      "schema_version": 1,
      "start_time": "2026-08-09T05:26:00Z",
      "mic": { "host_ticks": 9007199254740993, "timebase_numer": 125, "timebase_denom": 3 },
      "speaker": { "host_ticks": 9007199254740995, "timebase_numer": 125, "timebase_denom": 3 }
    }"#;
    fs::write(session.session_json(), before).unwrap();

    let decoded = SessionMetadata::read(&session.session_json()).unwrap();

    assert_eq!(decoded, sample_metadata("20260809-052607"));
    assert!(decoded.meeting.is_none());
}

/// The new-file/old-build direction: `OldMetadata` is a build predating this field,
/// reproduced. Serde ignores members it does not know, so a `session.json` carrying a
/// meeting must still parse as a whole session for a downgraded binary.
#[test]
fn session_json_with_a_meeting_still_reads_on_a_build_without_one() {
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct OldMetadata {
        session_id: String,
        schema_version: u32,
        start_time: String,
        mic: serde_json::Value,
        speaker: serde_json::Value,
    }

    let json = serde_json::to_string(
        &sample_metadata("20260809-052607").with_meeting(Some(sample_meeting())),
    )
    .unwrap();

    let old: OldMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(old.session_id, "20260809-052607");
    assert_eq!(old.schema_version, SESSION_SCHEMA_VERSION);
}

/// A session recorded outside any meeting must write exactly the bytes it wrote before this
/// field existed -- that equivalence is why `SESSION_SCHEMA_VERSION` did not move.
///
/// The fit lives on the meeting rather than on the session for exactly this reason: with no
/// meeting there is nothing to hang one on, so no `fit` key can appear either.
#[test]
fn a_session_with_no_meeting_writes_no_meeting_key() {
    let json = serde_json::to_string(&sample_metadata("20260809-052607")).unwrap();
    assert!(
        !json.contains("meeting"),
        "an absent meeting must be absent, not null: {json}"
    );
    assert!(
        !json.contains("fit"),
        "a session with no meeting has no fit: {json}"
    );
    // `meeting_cleared` contains "meeting", so the first assertion already covers it -- which
    // is the point worth stating rather than leaving to be rediscovered. A session nobody has
    // corrected serializes byte-identically to one written before the flag existed, and that
    // is why adding it did not need a schema version.
    assert!(
        !json.contains("cleared"),
        "a session nobody corrected must not carry the flag: {json}"
    );
}

// --- a label settled by hand -----------------------------------------------------------

/// The three states a label can be in, and the fact that only two of them are settled.
///
/// "No meeting" is the ambiguous one: it is both what a session recorded outside any meeting
/// looks like and what one the calendar could not be read for looks like. `meeting_cleared` is
/// the whole of the difference, and it is what stops a later pass from filling the second
/// answer into the first.
#[test]
fn a_label_is_settled_by_hand_only_when_somebody_settled_it() {
    let found = sample_metadata("20260809-052607").with_meeting(Some(sample_meeting()));
    assert!(!found.meeting_settled_by_hand());
    assert!(!sample_metadata("20260809-052607").meeting_settled_by_hand());

    let mut attached = sample_metadata("20260809-052607");
    attached.label_by_hand(Some(sample_meeting()));
    assert!(attached.meeting_settled_by_hand());
    assert_eq!(
        attached.meeting.as_ref().unwrap().fit,
        MeetingFit::Confirmed
    );
    assert!(!attached.meeting_cleared);

    let mut cleared = found.clone();
    cleared.label_by_hand(None);
    assert!(cleared.meeting_settled_by_hand());
    assert!(cleared.meeting.is_none());
    assert!(cleared.meeting_cleared);

    // And correcting a correction: attaching after clearing takes the flag back down, so a
    // session cannot end up both cleared and labelled.
    let mut corrected = cleared;
    corrected.label_by_hand(Some(sample_meeting()));
    assert!(!corrected.meeting_cleared);
    assert_eq!(
        corrected.meeting.as_ref().unwrap().fit,
        MeetingFit::Confirmed
    );
}

/// The guard itself: `with_meeting` is the one door an automatic pass writes a label through,
/// and it refuses a session somebody has already settled.
///
/// Structural rather than a rule each future caller has to remember -- which is the only form
/// of "nothing overwrites a human's answer" that survives code nobody has written yet.
#[test]
fn an_automatic_pass_cannot_overwrite_a_label_settled_by_hand() {
    let guess = sample_meeting()
        .with_fit(MeetingFit::Started)
        .with_invite(None, None, None);

    let mut attached = sample_metadata("20260809-052607");
    attached.label_by_hand(Some(sample_meeting()));
    assert_eq!(attached.clone().with_meeting(Some(guess.clone())), attached);
    assert_eq!(attached.clone().with_meeting(None), attached);

    let mut cleared = sample_metadata("20260809-052607");
    cleared.label_by_hand(None);
    assert_eq!(cleared.clone().with_meeting(Some(guess)), cleared);

    // A session nobody has settled is still writable, or the recorder could never label one.
    let fresh = sample_metadata("20260809-052607").with_meeting(Some(sample_meeting()));
    assert!(fresh.meeting.is_some());
}

/// A confirmed label is a strong match, so the attendee roster and the caveat follow from it
/// exactly as they do for the automatic strong fits -- the two properties every consumer of a
/// meeting reads.
#[test]
fn a_confirmed_label_is_strong_and_carries_no_caveat() {
    assert!(MeetingFit::Confirmed.is_strong());
    assert_eq!(MeetingFit::Confirmed.caveat(), None);
    assert!(MeetingFit::ALL.contains(&MeetingFit::Confirmed));

    let meeting = sample_meeting().with_fit(MeetingFit::Confirmed);
    assert_eq!(meeting.speaker_roster().map(<[_]>::len), Some(1));
}

/// A correction survives the round trip through `session.json`, spelled the way a person
/// reading the file would expect.
#[test]
fn a_cleared_label_round_trips_through_session_json() {
    let mut cleared = sample_metadata("20260809-052607").with_meeting(Some(sample_meeting()));
    cleared.label_by_hand(None);
    let json = serde_json::to_string(&cleared).unwrap();

    assert!(json.contains(r#""meeting_cleared":true"#), "{json}");
    assert!(!json.contains("Incident review"), "{json}");
    let decoded: SessionMetadata = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, cleared);
    assert!(decoded.meeting_settled_by_hand());
}

/// A `session.json` written before corrections existed reads as one nobody has corrected,
/// rather than failing to parse -- the same rule every other addition to this file has had to
/// meet, and the reason the schema version did not move.
#[test]
fn session_json_written_before_corrections_still_reads() {
    let (_tmp, paths) = temp_root();
    let session = make_session(&paths, "20260809-052607", &[]);
    let before = r#"{
      "session_id": "20260809-052607",
      "schema_version": 1,
      "start_time": "2026-08-09T05:26:00Z",
      "mic": { "host_ticks": 9007199254740993, "timebase_numer": 125, "timebase_denom": 3 },
      "speaker": { "host_ticks": 9007199254740995, "timebase_numer": 125, "timebase_denom": 3 }
    }"#;
    write_atomic(&session.session_json(), before.as_bytes()).unwrap();

    let metadata = SessionMetadata::read(&session.session_json()).unwrap();

    assert!(!metadata.meeting_cleared);
    assert!(!metadata.meeting_settled_by_hand());
    // So the recorder's own lookup still reaches it.
    assert!(
        metadata
            .with_meeting(Some(sample_meeting()))
            .meeting
            .is_some()
    );
}

// --- the fit ---------------------------------------------------------------------------

/// A `session.json` whose meeting predates fits must not read as a *good* match.
///
/// Written as a literal rather than as a re-serialization, for the same reason
/// `session_json_written_before_meetings_still_reads` gives: a re-serialization would track
/// the struct through every future change and so could never fail.
#[test]
fn a_meeting_written_before_fits_reads_as_unknown_and_yields_no_roster() {
    let (_tmp, paths) = temp_root();
    let session = make_session(&paths, "20260809-052607", &[]);
    let before = r#"{
      "session_id": "20260809-052607",
      "schema_version": 1,
      "start_time": "2026-08-09T05:26:00Z",
      "mic": { "host_ticks": 9007199254740993, "timebase_numer": 125, "timebase_denom": 3 },
      "speaker": { "host_ticks": 9007199254740995, "timebase_numer": 125, "timebase_denom": 3 },
      "meeting": {
        "title": "Incident review",
        "start": "2026-08-09T05:00:00Z",
        "end": "2026-08-09T06:00:00Z",
        "calendar": "Work",
        "attendees": [
          { "name": "Grace Hopper", "status": "tentative", "is_you": true }
        ],
        "event_id": "EVENT-ABC"
      }
    }"#;
    fs::write(session.session_json(), before).unwrap();

    let meeting = SessionMetadata::read(&session.session_json())
        .unwrap()
        .meeting
        .expect("the meeting still reads");

    assert_eq!(meeting.title, "Incident review");
    assert_eq!(meeting.fit, MeetingFit::Unknown);
    assert!(!meeting.fit.is_strong());
    // The people are still on disk -- this is not redaction -- but they are not a roster.
    assert_eq!(meeting.attendee_count(), 1);
    assert_eq!(meeting.speaker_roster(), None);
}

/// Every fit survives a round trip through `session.json`, spelled in snake case.
#[test]
fn every_fit_round_trips_through_session_json() {
    for fit in MeetingFit::ALL {
        let metadata =
            sample_metadata("20260809-052607").with_meeting(Some(sample_meeting().with_fit(fit)));
        let json = serde_json::to_string(&metadata).unwrap();
        let decoded: SessionMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, metadata, "{fit:?}");
        assert_eq!(decoded.meeting.unwrap().fit, fit);
    }
}

/// The guard doc-001's finding asks for: the attendee list is reachable as a speaker roster
/// only through the fit, so a weak match cannot seed an identification pass with the wrong
/// people. Driven over every variant, so a new one has to decide rather than inherit.
#[test]
fn the_attendee_roster_is_only_available_for_a_strong_match() {
    for fit in MeetingFit::ALL {
        let meeting = sample_meeting().with_fit(fit);
        assert_eq!(meeting.attendee_count(), 1, "{fit:?}");

        match meeting.speaker_roster() {
            Some(roster) => {
                assert!(fit.is_strong(), "{fit:?} handed out a roster");
                assert_eq!(roster.len(), 1);
            }
            None => assert!(!fit.is_strong(), "{fit:?} withheld a roster"),
        }
    }
}

/// A weak fit always has a caveat to show a person, and a strong one never does -- the
/// property the record command's finish line and the transcript frontmatter both rely on.
#[test]
fn a_caveat_exists_exactly_when_the_fit_is_weak() {
    for fit in MeetingFit::ALL {
        assert_eq!(
            fit.caveat().is_none(),
            fit.is_strong(),
            "{fit:?}: {:?}",
            fit.caveat()
        );
    }
    // And the wording names the timing rather than any meeting content.
    let caveat = MeetingFit::JoinedLate.caveat().unwrap();
    assert!(
        caveat.contains("after this meeting had started"),
        "{caveat}"
    );
}

#[test]
fn a_meeting_round_trips_with_its_attendees() {
    let metadata = sample_metadata("20260809-052607").with_meeting(Some(sample_meeting()));
    let json = serde_json::to_string(&metadata).unwrap();
    let decoded: SessionMetadata = serde_json::from_str(&json).unwrap();

    assert_eq!(decoded, metadata);
    let meeting = decoded.meeting.unwrap();
    assert_eq!(meeting.title, "Incident review");
    assert_eq!(meeting.calendar, "Work");
    assert_eq!(meeting.event_id, "EVENT-ABC");
    // Read through the guard, so this also asserts that a strong match yields the roster.
    let roster = meeting
        .speaker_roster()
        .expect("a strong match has a roster");
    assert_eq!(roster.len(), 1);
    assert!(roster[0].is_you);
    assert_eq!(roster[0].status, AttendeeStatus::Tentative);
    assert_eq!(
        meeting.organizer.as_ref().unwrap().email.as_deref(),
        Some("ada@example.com")
    );
}

/// The invite body reaches disk verbatim -- it is the agenda, and the best answer a stored
/// session has to "what was this meeting about". Asserted on the serialized form, which is
/// what actually reaches disk.
///
/// This replaces an earlier test asserting the exact opposite. Storing the body was once
/// ruled out because it routinely carries dial-in PINs; the field is now stored deliberately,
/// and the security property that survived the reversal is narrower: notes go to
/// `session.json` and to no log line. That half lives where the rendering does, in
/// `meethook-record`'s `the_debug_line_counts_attendees_without_naming_them`.
#[test]
fn meeting_metadata_stores_the_invite_body_and_location() {
    let json = serde_json::to_string(
        &sample_metadata("20260809-052607").with_meeting(Some(sample_meeting())),
    )
    .unwrap();

    assert!(json.contains("Agenda: the pager, then the fix"), "{json}");
    assert!(json.contains("Babbage Room"), "{json}");
}

/// An event with no body writes no key, rather than `"notes": null` or `"notes": ""`: absent
/// and empty mean different things to anything reading these files later.
#[test]
fn a_meeting_without_notes_writes_no_notes_key() {
    let bare =
        sample_meeting().with_invite(Some("https://example.com/j/12345".to_owned()), None, None);
    let json = serde_json::to_string(&sample_metadata("20260809-052607").with_meeting(Some(bare)))
        .unwrap();

    assert!(!json.contains("notes"), "{json}");
    assert!(!json.contains("location"), "{json}");
}

#[test]
fn metadata_written_to_disk_reads_back_identically() {
    let (_tmp, paths) = temp_root();
    let session = make_session(&paths, "20260809-052607", &["mic.wav", "speaker.wav"]);
    let metadata = sample_metadata("20260809-052607");

    metadata.write(&session.session_json()).unwrap();

    assert_eq!(
        SessionMetadata::read(&session.session_json()).unwrap(),
        metadata
    );
}

// --- atomic write ------------------------------------------------------------------

#[test]
fn atomic_write_replaces_contents_wholly_and_leaves_no_residue() {
    let (_tmp, paths) = temp_root();
    let session = make_session(&paths, "20260809-052607", &[]);
    let target = session.session_json();

    write_atomic(&target, b"a much longer first payload").unwrap();
    write_atomic(&target, b"short").unwrap();

    assert_eq!(fs::read(&target).unwrap(), b"short");
    assert_eq!(temp_file_count(session.dir()), 0, "temp files left behind");
}

#[test]
fn interrupted_write_leaves_no_partial_session_json() {
    let (_tmp, paths) = temp_root();
    let session = make_session(&paths, "20260809-052607", &["mic.wav", "speaker.wav"]);

    // Simulate a crash mid-write: the temp file exists with partial contents, but the
    // rename never happened. This is the exact state that must not read as a valid session.
    let partial = session.dir().join(".meethook-tmp-abc123");
    fs::write(&partial, b"{\"session_id\":\"202608").unwrap();

    assert!(!session.session_json().exists());
    let discovered = discover_sessions(&paths).unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].classification, Classification::Orphaned);

    // And completing the write afterwards still produces a whole, parseable file.
    sample_metadata("20260809-052607")
        .write(&session.session_json())
        .unwrap();
    let discovered = discover_sessions(&paths).unwrap();
    assert_eq!(discovered[0].classification, Classification::Valid);
    assert!(discovered[0].load_metadata().is_ok());
}

fn temp_file_count(dir: &Path) -> usize {
    fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(".meethook-tmp-")
        })
        .count()
}

// --- discovery and classification --------------------------------------------------

#[test]
fn classifies_valid_orphaned_and_transcribed_sessions() {
    let (_tmp, paths) = temp_root();

    make_session(
        &paths,
        "20260809-052601",
        &["mic.wav", "speaker.wav", "session.json"],
    );
    make_session(&paths, "20260809-052602", &["mic.wav", "speaker.wav"]);
    make_session(
        &paths,
        "20260809-052603",
        &["mic.wav", "speaker.wav", "session.json", "transcript.json"],
    );

    let discovered = discover_sessions(&paths).unwrap();
    let seen: Vec<_> = discovered
        .iter()
        .map(|s| (s.id.as_str(), s.classification))
        .collect();

    assert_eq!(
        seen,
        vec![
            ("20260809-052601", Classification::Valid),
            ("20260809-052602", Classification::Orphaned),
            ("20260809-052603", Classification::Transcribed),
        ]
    );
}

#[test]
fn transcribed_wins_over_valid_even_though_both_markers_exist() {
    let (_tmp, paths) = temp_root();
    make_session(
        &paths,
        "20260809-052601",
        &["session.json", "transcript.json"],
    );

    let discovered = discover_sessions(&paths).unwrap();
    assert_eq!(discovered[0].classification, Classification::Transcribed);
}

#[test]
fn missing_and_empty_sessions_roots_are_not_errors() {
    let (_tmp, paths) = temp_root();
    assert!(discover_sessions(&paths).unwrap().is_empty());

    fs::create_dir_all(paths.sessions_dir()).unwrap();
    assert!(discover_sessions(&paths).unwrap().is_empty());
}

#[test]
fn non_session_entries_under_sessions_are_ignored() {
    let (_tmp, paths) = temp_root();
    make_session(&paths, "20260809-052601", &["session.json"]);
    fs::create_dir_all(paths.sessions_dir().join("scratch-notes")).unwrap();
    fs::write(paths.sessions_dir().join(".DS_Store"), b"junk").unwrap();
    fs::write(
        paths.sessions_dir().join("20260809-052699"),
        b"a file, not a dir",
    )
    .unwrap();

    let discovered = discover_sessions(&paths).unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].id.as_str(), "20260809-052601");
}

#[test]
fn discovery_is_sorted_by_session_id() {
    let (_tmp, paths) = temp_root();
    for id in ["20260809-052603", "20260809-052601", "20260809-052602"] {
        make_session(&paths, id, &["session.json"]);
    }

    let ids: Vec<_> = discover_sessions(&paths)
        .unwrap()
        .iter()
        .map(|s| s.id.to_string())
        .collect();
    assert_eq!(
        ids,
        ["20260809-052601", "20260809-052602", "20260809-052603"]
    );
}

// --- path contract -----------------------------------------------------------------

#[test]
fn paths_are_derived_from_the_root_alone() {
    let paths = Paths::new("/tmp/meethook-root");
    let id = SessionId::parse("20260809-052607").unwrap();
    let session = paths.session(&id);

    assert_eq!(
        paths.sessions_dir(),
        Path::new("/tmp/meethook-root/sessions")
    );
    assert_eq!(paths.models_dir(), Path::new("/tmp/meethook-root/models"));
    assert_eq!(
        paths.speakers_json(),
        Path::new("/tmp/meethook-root/speakers.json")
    );
    // Root-level, not per-session, and that is the contract rather than a convenience: it is
    // what lets `enroll` and `forget` re-render a transcript through the same template
    // `transcribe` wrote it with.
    assert_eq!(
        paths.transcript_template(),
        Path::new("/tmp/meethook-root/transcript.md.jinja")
    );
    assert_eq!(
        session.dir(),
        Path::new("/tmp/meethook-root/sessions/20260809-052607")
    );
    assert_eq!(session.mic_wav().file_name().unwrap(), "mic.wav");
    assert_eq!(session.speaker_wav().file_name().unwrap(), "speaker.wav");
    assert_eq!(session.session_json().file_name().unwrap(), "session.json");
    assert_eq!(
        session.transcript_json().file_name().unwrap(),
        "transcript.json"
    );
    assert_eq!(
        session.transcript_md().file_name().unwrap(),
        "transcript.md"
    );
    assert_eq!(
        session.transcript_vtt().file_name().unwrap(),
        "transcript.vtt"
    );
    assert_eq!(
        session.mic_cleaned_wav().file_name().unwrap(),
        "mic.cleaned.wav"
    );
    assert_eq!(
        session.speaker_clusters_json().file_name().unwrap(),
        "speaker_clusters.json"
    );
}
