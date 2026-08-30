//! The read-only faces of `enroll`, driven through the built binary: `--list` reports every
//! voice a run would offer with its ranked candidates, and `--dry-run` reports what an answer
//! would do without doing it.
//!
//! The proof that "without doing it" is true is on-disk: the root is snapshotted before and
//! after every invocation and asserted byte-identical, rather than trusting that no write
//! path was reached.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use meethook_session::{
    AssignedName, EnrolledSpeaker, EnrolledSpeakers, Paths, RepresentativeSegment, SessionId,
    SessionMetadata, SourceTrack, SpeakerCluster, SpeakerClusters, SpeakerNames, TrackSync,
    Transcript, TranscriptContext, TranscriptTemplate, Turn,
};

/// A unit vector `degrees` away from cluster 0's voice, in the same four dimensions: close
/// enough to rank beside it, far enough (60 degrees) to stay outside identification's reach.
fn nearly(degrees: f32) -> Vec<f32> {
    let radians = degrees.to_radians();
    vec![radians.cos(), radians.sin(), 0.0, 0.0]
}

fn cluster(id: u32, embedding: Vec<f32>, speech_seconds: f64, first_spoke: f64) -> SpeakerCluster {
    SpeakerCluster {
        id,
        embedding,
        speech_seconds,
        first_spoke_seconds: first_spoke,
        heard_at_once_with: Vec::new(),
        representatives: vec![RepresentativeSegment {
            start: 0.0,
            end: 1.0,
        }],
    }
}

fn speaker_turn(start: f64, cluster: u32, speaker: &str, text: &str) -> Turn {
    Turn {
        speaker: speaker.to_string(),
        start,
        end: start + 1.0,
        text: text.to_string(),
        source_track: SourceTrack::Speaker,
        cluster: Some(cluster),
        speaker_id_confidence: None,
    }
}

/// Records that this session's cluster was given this name without enrolling it, the way an
/// answer in an earlier run would have.
fn assign_name(root: &Path, id: &SessionId, cluster: u32, name: &str, embedding: Vec<f32>) {
    let paths = Paths::new(root.to_path_buf());
    SpeakerNames::new(
        id.clone(),
        vec![AssignedName {
            cluster,
            name: name.to_string(),
            embedding,
        }],
    )
    .write(&paths.session(id))
    .unwrap();
}

/// Two voices heard at once under one session, with two enrolled people: Alice's reference
/// sits exactly on cluster 0's voice, which identifies it and so resolves it out of the
/// queue; cluster 1 is 60 degrees away, outside identification's reach but inside the
/// unthresholded ranking; and Milo is orthogonal to both.
///
/// No `speaker.wav`: nothing on the read-only paths reads audio, and a missing track degrades
/// to an empty clip rather than an error.
fn fixture(root: &Path) {
    let paths = Paths::new(root.to_path_buf());
    let id = SessionId::parse("20260809-052600").unwrap();
    let session = paths.session(&id);
    std::fs::create_dir_all(session.dir()).unwrap();

    let sync = TrackSync {
        host_ticks: 1,
        timebase_numer: 125,
        timebase_denom: 3,
    };
    let metadata = SessionMetadata::new(
        id.clone(),
        "2026-08-09T05:26:00Z".parse().unwrap(),
        sync,
        sync,
    );
    metadata.write(&session.session_json()).unwrap();

    let mut clusters = vec![
        cluster(0, vec![1.0, 0.0, 0.0, 0.0], 40.0, 0.0),
        cluster(1, nearly(60.0), 30.0, 1.0),
    ];
    // Written on both sides, as `speaker_clusters.json` documents it.
    clusters[0].heard_at_once_with = vec![1];
    clusters[1].heard_at_once_with = vec![0];
    SpeakerClusters::new(id.clone(), clusters)
        .write(&session)
        .unwrap();

    Transcript::new(
        id,
        vec![
            speaker_turn(0.0, 0, "Unknown 1", "hi there"),
            speaker_turn(1.0, 1, "Unknown 2", "and from me"),
        ],
    )
    .write(
        &session,
        &TranscriptTemplate::resolve(&paths, None).unwrap(),
        &TranscriptContext::now(&metadata),
    )
    .unwrap();

    EnrolledSpeakers::new(vec![
        EnrolledSpeaker {
            name: "Alice".to_string(),
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            clip_seconds: Some(30.0),
        },
        EnrolledSpeaker {
            name: "Milo".to_string(),
            embedding: vec![0.0, 0.0, 0.0, 1.0],
            clip_seconds: Some(20.0),
        },
    ])
    .write(&paths)
    .unwrap();
}

/// Every file under `root`, by path relative to it and by bytes: the whole state a run could
/// have touched.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(dir: &Path, base: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, base, out);
            } else {
                out.insert(
                    path.strip_prefix(base).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// The built binary pointed at this root, ready for the `enroll` flags a test appends.
fn meethook(root: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_meethook"));
    cmd.args(["--root"]).arg(root).args(["enroll"]);
    cmd
}

#[test]
fn list_reports_every_offered_voice_with_its_ranking_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let before = snapshot(dir.path());

    let output = meethook(dir.path())
        .args(["20260809-052600", "--list"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Cluster 0 is identified against Alice's reference before the run starts, so it is
    // resolved and not offered: the list carries the one voice still open, with the full
    // unthresholded ranking beside it.
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "\
20260809-052600  who is Unknown 2?  30s of speech
    Alice                  0.50   1 ref
    Milo                   0.00   1 ref
"
    );
    assert_eq!(snapshot(dir.path()), before, "--list wrote something");
}

#[test]
fn list_json_is_the_versioned_document_a_script_may_parse() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let before = snapshot(dir.path());

    let output = meethook(dir.path())
        .args(["20260809-052600", "--list", "--json"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(doc["schema"], "meethook.enroll.list.v1");
    assert_eq!(doc["sessions"].as_array().unwrap().len(), 1);
    let session = &doc["sessions"][0];
    assert_eq!(session["id"], "20260809-052600");
    assert!(session["meeting"].is_null());
    let voices = session["voices"].as_array().unwrap();
    assert_eq!(voices.len(), 1);
    assert_eq!(voices[0]["number"], "Unknown 2");
    assert_eq!(voices[0]["label"], "Unknown 2");
    assert_eq!(voices[0]["speech_seconds"], 30.0);
    assert_eq!(voices[0]["candidates"][0]["name"], "Alice");
    // The cosine of 60 degrees is 0.5 up to single-precision rounding; the document carries
    // the raw number, so compare it as one.
    let sim = voices[0]["candidates"][0]["similarity"].as_f64().unwrap();
    assert!((sim - 0.5).abs() < 1e-6, "similarity {sim} is not ~0.5");
    assert_eq!(voices[0]["candidates"][0]["references"], 1);
    assert_eq!(voices[0]["candidates"][1]["name"], "Milo");
    assert_eq!(
        snapshot(dir.path()),
        before,
        "--list --json wrote something"
    );
}

#[test]
fn list_correct_asks_the_confirmation_question_for_a_named_voice() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let id = SessionId::parse("20260809-052600").unwrap();
    // A name given in an earlier run, reached only through `--correct`, asks "is this right"
    // rather than "who is this".
    assign_name(dir.path(), &id, 1, "Nate", nearly(60.0));
    let before = snapshot(dir.path());

    let output = meethook(dir.path())
        .args(["20260809-052600", "--list", "--correct"])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // `--correct` widens the queue to every voice carrying a label -- confirmed identifications
    // among them -- and each asks "is this right".
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "\
20260809-052600  is Unknown 1 Alice?  40s of speech
    Alice                  1.00   1 ref
    Milo                   0.00   1 ref

20260809-052600  is Unknown 2 Nate?  30s of speech
    Alice                  0.50   1 ref
    Milo                   0.00   1 ref
"
    );
    assert_eq!(
        snapshot(dir.path()),
        before,
        "--list --correct wrote something"
    );
}

#[test]
fn dry_run_reports_the_veto_refusal_without_writing() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let before = snapshot(dir.path());

    // Cluster 1 is heard at once with cluster 0, whose voice Alice's reference sits on: the
    // name stays where it is, and the refusal names the voice that holds it.
    let output = meethook(dir.path())
        .args([
            "20260809-052600",
            "--voice",
            "2",
            "--name",
            "Alice",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "\
Unknown 2 in 20260809-052600, answering \"Alice\":
  unavailable: Unknown 1 was heard at the same time as this voice and would keep the name
"
    );
    assert_eq!(snapshot(dir.path()), before, "--dry-run wrote something");
}

#[test]
fn dry_run_reports_the_stored_outcome_without_writing() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let before = snapshot(dir.path());

    let output = meethook(dir.path())
        .args([
            "20260809-052600",
            "--voice",
            "2",
            "--name",
            "Milo",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "\
Unknown 2 in 20260809-052600, answering \"Milo\":
  stores another recording of them, 2 in all
"
    );
    assert_eq!(snapshot(dir.path()), before, "--dry-run wrote something");
}

#[test]
fn dry_run_json_is_the_versioned_document_a_script_may_parse() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let before = snapshot(dir.path());

    let output = meethook(dir.path())
        .args([
            "20260809-052600",
            "--voice",
            "2",
            "--name",
            "Milo",
            "--dry-run",
            "--json",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(doc["schema"], "meethook.enroll.dry-run.v1");
    assert_eq!(doc["voice"]["session"], "20260809-052600");
    assert_eq!(doc["voice"]["number"], "Unknown 2");
    assert_eq!(doc["name"], "Milo");
    assert!(doc["consequence"]["refused"].is_null());
    assert_eq!(doc["consequence"]["stored"]["Added"]["held"], 2);
    assert_eq!(doc["consequence"]["session_only"], false);
    assert_eq!(
        snapshot(dir.path()),
        before,
        "--dry-run --json wrote something"
    );
}

#[test]
fn a_selector_matching_nothing_fails_loudly_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    fixture(dir.path());
    let before = snapshot(dir.path());

    let output = meethook(dir.path())
        .args([
            "20260809-052600",
            "--voice",
            "9",
            "--name",
            "Alice",
            "--dry-run",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("could not be served"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        snapshot(dir.path()),
        before,
        "a failed dry run wrote something"
    );
}
