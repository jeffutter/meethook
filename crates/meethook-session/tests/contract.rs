//! Tests for the on-disk session contract.
//!
//! Every test builds its own root under a `TempDir`; nothing here ever touches
//! `~/meethook`.

use std::fs;
use std::path::Path;

use jiff::Timestamp;
use meethook_session::{
    Attendee, AttendeeStatus, Classification, Meeting, Paths, SESSION_SCHEMA_VERSION, SessionId,
    SessionMetadata, SessionPaths, TrackSync, create_session_dir, discover_sessions, write_atomic,
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
    Meeting {
        title: "Incident review".to_owned(),
        start: "2026-08-09T05:00:00Z".parse().unwrap(),
        end: "2026-08-09T06:00:00Z".parse().unwrap(),
        calendar: "Work".to_owned(),
        organizer: Some(Attendee {
            name: Some("Ada Lovelace".to_owned()),
            email: Some("ada@example.com".to_owned()),
            status: AttendeeStatus::Accepted,
            is_you: false,
        }),
        attendees: vec![Attendee {
            name: Some("Grace Hopper".to_owned()),
            email: Some("grace@example.com".to_owned()),
            status: AttendeeStatus::Tentative,
            is_you: true,
        }],
        url: Some("https://example.com/j/12345".to_owned()),
        event_id: "EVENT-ABC".to_owned(),
    }
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
#[test]
fn a_session_with_no_meeting_writes_no_meeting_key() {
    let json = serde_json::to_string(&sample_metadata("20260809-052607")).unwrap();
    assert!(
        !json.contains("meeting"),
        "an absent meeting must be absent, not null: {json}"
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
    assert_eq!(meeting.attendees.len(), 1);
    assert!(meeting.attendees[0].is_you);
    assert_eq!(meeting.attendees[0].status, AttendeeStatus::Tentative);
    assert_eq!(
        meeting.organizer.as_ref().unwrap().email.as_deref(),
        Some("ada@example.com")
    );
}

/// Meeting bodies carry dial-in numbers and one-time passcodes, so the absence of a notes
/// field is a security property rather than an oversight. Asserted on the serialized form,
/// which is what actually reaches disk.
#[test]
fn meeting_metadata_never_stores_notes() {
    let json = serde_json::to_string(
        &sample_metadata("20260809-052607").with_meeting(Some(sample_meeting())),
    )
    .unwrap();
    for banned in ["notes", "body", "description", "agenda"] {
        assert!(
            !json.contains(banned),
            "session.json must not store the meeting's {banned}: it routinely carries \
             dial-in PINs and passcodes"
        );
    }
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
        session.mic_cleaned_wav().file_name().unwrap(),
        "mic.cleaned.wav"
    );
    assert_eq!(
        session.speaker_clusters_json().file_name().unwrap(),
        "speaker_clusters.json"
    );
}
