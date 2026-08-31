//! Scratch: dump the three-voice group-commit fixture to the root named in argv[1].
//! Used to drive the real binary by hand while shaping the interrupt test. Not shipped.

use meethook_session::{
    Paths, RepresentativeSegment, SessionId, SessionMetadata, SourceTrack, SpeakerCluster,
    SpeakerClusters, TrackSync, Transcript, TranscriptContext, TranscriptTemplate, Turn,
};

fn cluster(id: u32, embedding: [f32; 4], speech_seconds: f64, first_spoke: f64) -> SpeakerCluster {
    SpeakerCluster {
        id,
        embedding: embedding.into(),
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

fn main() {
    let root = std::env::args()
        .nth(1)
        .expect("usage: dump_group_fixture <root>");
    let paths = Paths::new(root.clone());
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

    // Three orthogonal voices: no reference can ever identify any of them. The one person
    // this meeting actually held is split across clusters 1 and 2, which segmentation heard
    // at once -- the veto a lone naming cannot pass and the group answer overrides.
    let mut clusters = vec![
        cluster(0, [1.0, 0.0, 0.0, 0.0], 40.0, 0.0),
        cluster(1, [0.0, 1.0, 0.0, 0.0], 30.0, 1.0),
        cluster(2, [0.0, 0.0, 1.0, 0.0], 20.0, 2.0),
    ];
    clusters[1].heard_at_once_with = vec![2];
    clusters[2].heard_at_once_with = vec![1];
    SpeakerClusters::new(id.clone(), clusters)
        .write(&session)
        .unwrap();

    Transcript::new(
        id,
        vec![
            speaker_turn(0.0, 0, "Unknown 1", "hi there"),
            speaker_turn(1.0, 1, "Unknown 2", "and from me"),
            speaker_turn(2.0, 2, "Unknown 3", "counting in"),
            speaker_turn(3.0, 0, "Unknown 1", "let us start"),
        ],
    )
    .write(
        &session,
        &TranscriptTemplate::resolve(&paths, None).unwrap(),
        &TranscriptContext::now(&metadata),
    )
    .unwrap();

    println!("fixture written to {root}");
}
