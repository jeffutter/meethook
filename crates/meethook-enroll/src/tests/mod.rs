use std::cell::Cell;
use std::collections::VecDeque;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use meethook_session::{
    Attendee, AttendeeStatus, EnrolledSpeaker, MAX_REFERENCES_PER_SPEAKER, Meeting, MeetingFit,
    RepresentativeSegment, SPEAKER_YOU, SessionMetadata, SessionPaths, SourceTrack, SpeakerCluster,
    SpeakerClusters, SpeakerNames, Stored, TRANSCRIPT_SCHEMA_VERSION, TrackSync, Transcript,
    TranscriptContext, Turn, unknown_labels, unknown_speaker,
};
// The cut the ranking is deliberately *not* made at, named rather than spelled 0.40, so
// the fixtures below still mean "outside identification's reach" if it moves.
use meethook_transcribe::{Attribution, IDENTIFY_DISTANCE, Resemblance, identify_clusters};

use super::*;

mod assertions;
mod corrections;
mod deferral;
mod floor;
mod meetings;
mod naming;
mod narration;
mod preview;
mod prompts;
mod references;
mod refusals;
mod selectors;

/// One row of the queue a prompt was shown, owned so it can outlive the call.
///
/// The whole [`Attribution`] rather than its label, because what a queue pane needs is the
/// basis as well as the name -- "identified at 0.91" and "named for this session" are two
/// different rows however identically they read.
#[derive(Debug, PartialEq)]
struct Row {
    number: String,
    attribution: Attribution,
    speech_seconds: f64,
    below_floor: bool,
}

/// A voice recorded exactly as it was shown, so a test can assert on what the user would
/// have been looking at rather than only on what they answered.
#[derive(Debug, PartialEq)]
struct Shown {
    session: String,
    /// The meeting the prompt was told this session was recorded during -- or that it was
    /// not labelled with one at all. The only way a test can check that the value crosses
    /// the Interviewer seam rather than being re-read from `session.json` behind it.
    meeting: Option<MeetingLabel>,
    /// Which of this session's questions this was, and how many there were, exactly as the
    /// prompt was handed it.
    position: Position,
    /// What the prompt was told this voice is called and on what basis -- which is the only
    /// way a test can check that a correction prompt asked "is this right" rather than
    /// "who is this", and that a voice named for one session says so.
    attribution: Attribution,
    /// The handle the prompt was given, which is the only way a test can check that it does
    /// not move when the voice is named.
    number: String,
    speech_seconds: f64,
    /// Every voice of the session as the prompt was shown them, so a test can check both
    /// what a queue pane would hold and that it is current.
    queue: Vec<Row>,
    snippets: Vec<String>,
    /// Each snippet's `(start, duration)`, so a test can prove the prompt was handed track
    /// time rather than timeline time -- the one failure no assertion about text can see.
    snippet_times: Vec<(f64, f64)>,
    /// How many samples each snippet carried, which is what says the audio was cut from
    /// the stretch those times name.
    snippet_samples: Vec<usize>,
    clip_samples: usize,
    /// Who the prompt was told this voice resembles, in the order it was handed them --
    /// which is the only way a test can check that an [`Interviewer`] can offer names
    /// without ever reading `speakers.json`.
    resembles: Vec<Resemblance>,
    /// Every enrolled name the prompt was handed -- the universe [`resolve()`] requires,
    /// which is not the same list as `resembles`.
    enrolled: Vec<String>,
}

impl Shown {
    fn label(&self) -> &str {
        self.attribution.label()
    }

    fn confidence(&self) -> Option<f32> {
        self.attribution.confidence()
    }

    /// The queue as a pane would list it: the handle, what the row reads as, and whether
    /// the floor held it back. For the assertions that are about the shape of the queue
    /// rather than about the basis of one row.
    fn rows(&self) -> Vec<(&str, &str, bool)> {
        self.queue
            .iter()
            .map(|row| {
                (
                    row.number.as_str(),
                    row.attribution.label(),
                    row.below_floor,
                )
            })
            .collect()
    }

    /// The ranking as a prompt would list it: who, and how many recordings of them.
    fn offered(&self) -> Vec<(&str, usize)> {
        self.resembles
            .iter()
            .map(|r| (r.name.as_str(), r.references))
            .collect()
    }
}

/// An interviewer that answers from a queue and remembers every voice it was asked about.
/// Answers past the end of the script are skips, so a test that expects no prompt at all
/// fails on `seen` rather than on a panic somewhere else.
#[derive(Default)]
struct Scripted {
    answers: VecDeque<Answer>,
    seen: Vec<Shown>,
    /// How many more stalled passes this answerer claims to still be working through. A
    /// countdown rather than a flag so that a test which gets the arithmetic wrong fails
    /// instead of hanging: once it reaches zero the session ends however the script reads.
    working_passes: Cell<usize>,
}

impl Scripted {
    fn answering(answers: Vec<Answer>) -> Scripted {
        Scripted {
            answers: answers.into(),
            seen: Vec::new(),
            working_passes: Cell::new(0),
        }
    }

    /// Say "still working" for the next `passes` stalled passes and finished after that,
    /// standing in for the cursor of a full-screen frame.
    fn working_for(self, passes: usize) -> Scripted {
        Scripted {
            working_passes: Cell::new(passes),
            ..self
        }
    }

    fn labels(&self) -> Vec<&str> {
        self.seen.iter().map(Shown::label).collect()
    }

    /// The positions as the user reads them, through [`Display`] rather than as a pair, so
    /// an assertion covers the form on the screen and not only the two numbers.
    fn positions(&self) -> Vec<String> {
        self.seen.iter().map(|v| v.position.to_string()).collect()
    }
}

impl Interviewer for Scripted {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        self.seen.push(Shown {
            session: voice.session.to_string(),
            meeting: voice.meeting.cloned(),
            position: voice.position,
            attribution: voice.attribution.clone(),
            number: voice.number.to_string(),
            speech_seconds: voice.speech_seconds,
            queue: voice
                .queue
                .iter()
                .map(|row| Row {
                    number: row.number.to_string(),
                    attribution: row.attribution.clone(),
                    speech_seconds: row.speech_seconds,
                    below_floor: row.below_floor,
                })
                .collect(),
            snippets: voice.snippets.iter().map(|s| s.text.to_string()).collect(),
            snippet_times: voice
                .snippets
                .iter()
                .map(|s| (s.start, s.duration))
                .collect(),
            snippet_samples: voice.snippets.iter().map(|s| s.audio.len()).collect(),
            clip_samples: voice.clip.len(),
            resembles: voice.resembles.clone(),
            enrolled: voice.enrolled.iter().map(|n| n.to_string()).collect(),
        });
        self.answers.pop_front().unwrap_or(Answer::Skip)
    }

    fn still_working(&self) -> bool {
        let left = self.working_passes.get();
        self.working_passes.set(left.saturating_sub(1));
        left > 0
    }
}

fn named(name: &str) -> Answer {
    Answer::Named {
        name: name.to_string(),
        anyway: false,
    }
}

/// The same answer, insisted on: honour it even where it takes a name off a voice the user
/// was not asked about. Only [`Refusal::Taken`] is in reach, which is what the veto tests
/// below use this to pin.
fn named_anyway(name: &str) -> Answer {
    Answer::Named {
        name: name.to_string(),
        anyway: true,
    }
}

/// A distinct unit vector per cluster id, so enrolling one of these voices matches that
/// cluster and nobody else's.
pub(crate) fn voice(id: u32) -> Vec<f32> {
    let mut embedding = vec![0.0f32; 4];
    embedding[id as usize % 4] = 1.0;
    embedding
}

/// A unit vector `degrees` away from cluster 0's, for the fixtures that are about how
/// close two voices are: one person clustering split in two, or one reference that matches
/// both halves. 0.35 of cosine distance is `IDENTIFY_DISTANCE`, so 49 degrees is the edge.
pub(crate) fn nearly(degrees: f32) -> Vec<f32> {
    let radians = degrees.to_radians();
    vec![radians.cos(), radians.sin(), 0.0, 0.0]
}

fn cluster(id: u32, first_spoke: f64, representative: (f64, f64)) -> SpeakerCluster {
    SpeakerCluster {
        id,
        embedding: voice(id),
        speech_seconds: 10.0 + f64::from(id),
        first_spoke_seconds: first_spoke,
        heard_at_once_with: Vec::new(),
        representatives: vec![RepresentativeSegment {
            start: representative.0,
            end: representative.1,
        }],
    }
}

/// `cluster` is the voice the turn came from, exactly as `merge` would have recorded it,
/// and `speaker` is what that voice was called when the transcript was written. The two
/// have to agree for a fixture to mean anything: the tests below read a label off the
/// file and expect the cluster underneath it to be the one they named.
pub(crate) fn speaker_turn(start: f64, cluster: u32, speaker: &str, text: &str) -> Turn {
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

pub(crate) fn mic_turn(start: f64, text: &str) -> Turn {
    Turn {
        speaker: SPEAKER_YOU.to_string(),
        start,
        end: start + 1.0,
        text: text.to_string(),
        source_track: SourceTrack::Mic,
        cluster: None,
        speaker_id_confidence: None,
    }
}

/// Six seconds of 16 kHz mono tone: real audio, so a clip sliced out of it has the
/// samples a test can count.
fn write_speaker_wav(path: &Path) {
    let samples: Vec<f32> = (0..16_000 * 6)
        .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.3)
        .collect();
    write_clip(path, &samples).unwrap();
}

/// A transcribed two-voice session: cluster 0 speaks first, cluster 1 answers, and the
/// local speaker is in there too so tests can prove the mic track is never touched.
///
/// The transcript is written with the labels `transcribe` would have given it against an
/// empty database, which is the state `enroll` is for.
/// The `session.json` a fixture session carries.
///
/// A real one rather than the `{}` placeholder this used to be: classification still only
/// checks the file's presence, but re-rendering a `transcript.md` reads the session's start
/// time and its meeting out of it.
pub(crate) fn session_metadata(id: &SessionId) -> SessionMetadata {
    let sync = TrackSync {
        host_ticks: 1,
        timebase_numer: 125,
        timebase_denom: 3,
    };
    SessionMetadata::new(
        id.clone(),
        "2026-08-09T05:26:00Z".parse().unwrap(),
        sync,
        sync,
    )
}

/// Writes both transcript files the way `transcribe` does: through whatever template the
/// root resolves to.
///
/// Going through [`TranscriptTemplate::resolve`] rather than always taking the built-in is
/// what lets a test drop a `transcript.md.jinja` into the root and have the fixture itself
/// honour it, exactly as the CLI does.
pub(crate) fn write_transcript(
    transcript: &Transcript,
    paths: &Paths,
    session: &SessionPaths,
    metadata: &SessionMetadata,
) {
    transcript
        .write(
            session,
            &TranscriptTemplate::resolve(paths, None).unwrap(),
            &TranscriptContext::now(metadata),
        )
        .unwrap();
}

pub(crate) fn make_session(paths: &Paths, id: &str) -> SessionPaths {
    let id = SessionId::parse(id).unwrap();
    let session = paths.session(&id);
    std::fs::create_dir_all(session.dir()).unwrap();
    let metadata = session_metadata(&id);
    metadata.write(&session.session_json()).unwrap();
    write_speaker_wav(&session.speaker_wav());

    SpeakerClusters::new(
        id.clone(),
        vec![cluster(0, 0.0, (0.5, 2.5)), cluster(1, 3.0, (3.0, 5.0))],
    )
    .write(&session)
    .unwrap();

    write_transcript(
        &Transcript::new(
            id,
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "  hi there  "),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(4.0, 0, "Unknown 1", "let us start"),
            ],
        ),
        paths,
        &session,
        &metadata,
    );

    session
}

/// One voice worth naming and three fragments under the floor, which is the shape real
/// clustering leaves a meeting in: a handful of speakers and a tail of turns too short
/// for any distance rule to place.
fn make_fragmented_session(paths: &Paths, id: &str) -> SessionPaths {
    let session = make_session(paths, id);
    let parsed = SessionId::parse(id).unwrap();

    let mut clusters = vec![
        cluster(0, 0.0, (0.5, 2.5)),
        cluster(1, 3.0, (3.0, 5.0)),
        cluster(2, 3.5, (1.0, 2.0)),
        cluster(3, 4.5, (2.0, 3.0)),
    ];
    for (cluster, seconds) in clusters.iter_mut().zip([40.0, 1.5, 0.9, 2.0]) {
        cluster.speech_seconds = seconds;
    }
    SpeakerClusters::new(parsed.clone(), clusters)
        .write(&session)
        .unwrap();

    write_transcript(
        &Transcript::new(
            parsed.clone(),
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(3.5, 2, "Unknown 3", "mm"),
                speaker_turn(4.5, 3, "Unknown 4", "yes"),
            ],
        ),
        paths,
        &session,
        &session_metadata(&parsed),
    );

    session
}

fn run(paths: &Paths, ids: &[&str], interviewer: &mut Scripted) -> (EnrollReport, String) {
    run_asking(paths, ids, Offer::default(), interviewer)
}

/// `run`, with the widening flags exposed. Separate so that the dozen tests that have
/// nothing to do with the floor or with corrections do not carry an [`Offer`] each.
fn run_asking(
    paths: &Paths,
    ids: &[&str],
    offer: Offer,
    interviewer: &mut Scripted,
) -> (EnrollReport, String) {
    run_enrolling(paths, ids, offer, Enrolment::default(), interviewer)
}

/// `run_asking`, with the write-side override exposed too. Separate again for the same
/// reason: only the tests about what an answer *writes* care which of the two it is.
fn run_enrolling(
    paths: &Paths,
    ids: &[&str],
    offer: Offer,
    enrolment: Enrolment,
    interviewer: &mut Scripted,
) -> (EnrollReport, String) {
    run_over(
        paths,
        ids,
        None,
        offer,
        visits(offer),
        enrolment,
        interviewer,
    )
}

/// Which sessions the CLI visits for a given [`Offer`], so the helpers above stay the plain
/// command: both halves come off `--correct` there, and a test that wants them apart -- the
/// one about [`Sessions`] being separately decidable -- goes through [`run_over`] directly.
fn visits(offer: Offer) -> Sessions {
    if offer.named {
        Sessions::Every
    } else {
        Sessions::Unresolved
    }
}

/// `run`, aimed at one voice. One helper per axis, like the two above, so that the tests
/// that do not target a voice keep their short signature -- and a default [`Offer`], since
/// the point of a selector is that it needs no flags to reach a voice.
fn run_targeting(
    paths: &Paths,
    ids: &[&str],
    voice: &str,
    interviewer: &mut Scripted,
) -> (EnrollReport, String) {
    let selector = VoiceSelector::from(voice);
    run_over(
        paths,
        ids,
        Some(Selection::Voice(selector)),
        Offer::default(),
        // Irrelevant beside a selector, which stands in for the queue and its gates alike.
        Sessions::default(),
        Enrolment::default(),
        interviewer,
    )
}

/// `run_targeting`'s sibling, aimed at whoever was speaking at one moment. `at` is written
/// exactly as a user would copy it off `transcript.md`, so the tests exercise the spelling
/// as well as the lookup.
fn run_at(
    paths: &Paths,
    ids: &[&str],
    at: &str,
    interviewer: &mut dyn Interviewer,
) -> (EnrollReport, String) {
    run_over(
        paths,
        ids,
        Some(Selection::At(at.parse().unwrap())),
        Offer::default(),
        Sessions::default(),
        Enrolment::default(),
        interviewer,
    )
}

/// The whole non-interactive command: a moment, and the name of whoever was speaking then.
fn run_naming_at(paths: &Paths, ids: &[&str], at: &str, name: &str) -> (EnrollReport, String) {
    run_at(paths, ids, at, &mut GivenName::new(name))
}

#[allow(clippy::too_many_arguments)]
fn run_over(
    paths: &Paths,
    ids: &[&str],
    selection: Option<Selection>,
    offer: Offer,
    sessions: Sessions,
    enrolment: Enrolment,
    interviewer: &mut dyn Interviewer,
) -> (EnrollReport, String) {
    let requested: Vec<SessionId> = ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
    let mut out = Vec::new();
    let report = run_enroll(
        paths,
        &requested,
        EnrollRules {
            selector: selection,
            offer,
            sessions,
            enrolment,
            relabel_transcript: true,
            one_speaker: None,
            // Resolved from the root, exactly as the CLI does, so a test that puts a
            // template there is testing the path a user takes.
            template: &TranscriptTemplate::resolve(paths, None).unwrap(),
        },
        interviewer,
        &mut Lines::new(&mut out),
    )
    .unwrap();
    (report, String::from_utf8(out).unwrap())
}

/// The database `enroll` would have written by naming these clusters, so a test can start
/// from "the wrong person is already on this voice" without running a first pass.
pub(crate) fn enrolled(entries: &[(&str, Vec<f32>)], paths: &Paths) {
    EnrolledSpeakers::new(
        entries
            .iter()
            .map(|(name, embedding)| EnrolledSpeaker {
                name: name.to_string(),
                embedding: embedding.clone(),
                clip_seconds: None,
            })
            .collect(),
    )
    .write(paths)
    .unwrap();
}

/// `--correct` on its own: reach the already-named voices, leave the floor where it is.
const CORRECT: Offer = Offer {
    quiet: false,
    named: true,
};

/// `--all` on its own: reach the quiet voices. Since [`PROMPT_FLOOR_SECONDS`] and
/// [`REFERENCE_FLOOR_SECONDS`] are the same number, this is also the only flag that
/// reaches a voice quiet enough for an answer to be recorded against the session alone.
const ALL: Offer = Offer {
    quiet: true,
    named: false,
};

/// `--all --correct`: the only way back to a voice already named for its session, which is
/// by construction both named *and* under the prompt floor, so either flag alone misses it.
const ALL_AND_CORRECT: Offer = Offer {
    quiet: true,
    named: true,
};

/// Rewrites this session's clusters with the talk times given, ids in order, leaving
/// first appearances and representatives as [`make_session`] wrote them.
///
/// The fixture's default is `10.0 + id`, which clears the floor for every voice; the
/// floor tests are the ones that need to say otherwise.
fn with_speech_seconds(session: &SessionPaths, seconds: &[f64]) {
    let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    for (cluster, seconds) in clusters.clusters.iter_mut().zip(seconds) {
        cluster.speech_seconds = *seconds;
    }
    clusters.write(session).unwrap();
}

/// Rewrites this session's transcript, leaving its clusters and its metadata as
/// [`make_session`] wrote them. Both files, through the same template `transcribe` uses, so
/// the timestamps a test then points at are the ones `transcript.md` actually prints.
///
/// The timestamp tests are the ones that need a timeline other than the fixture's four turns
/// in its first five seconds.
fn with_turns(paths: &Paths, session: &SessionPaths, id: &str, turns: Vec<Turn>) {
    let parsed = SessionId::parse(id).unwrap();
    write_transcript(
        &Transcript::new(parsed.clone(), turns),
        paths,
        session,
        &session_metadata(&parsed),
    );
}

/// Rewrites this session's cluster embeddings, ids in order, leaving everything else as
/// [`make_session`] wrote it. The fixture's default is one orthogonal vector per cluster;
/// the tests about near voices are the ones that need to say otherwise.
pub(crate) fn with_embeddings(session: &SessionPaths, embeddings: &[Vec<f32>]) {
    let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    for (cluster, embedding) in clusters.clusters.iter_mut().zip(embeddings) {
        cluster.embedding = embedding.clone();
    }
    clusters.write(session).unwrap();
}

/// Records that segmentation heard these two voices speaking at once.
///
/// That relation is the one piece of evidence proving two clusters are different people,
/// and it is what the heard-at-once veto acts on -- so it is also the one way an answer can
/// still cost another voice its name once references accumulate rather than replace.
/// Written on both sides, as `speaker_clusters.json` documents it.
pub(crate) fn heard_at_once(session: &SessionPaths, a: u32, b: u32) {
    let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    for cluster in &mut clusters.clusters {
        if cluster.id == a {
            cluster.heard_at_once_with.push(b);
        } else if cluster.id == b {
            cluster.heard_at_once_with.push(a);
        }
    }
    clusters.write(session).unwrap();
}

/// A unit vector on one axis of an `axes`-wide space: every pair of these is orthogonal, so
/// no two of them can ever be matched to one another however many references pile up.
/// [`voice`] is the same idea fixed at four dimensions.
pub(crate) fn axis(which: usize, axes: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; axes];
    embedding[which] = 1.0;
    embedding
}

pub(crate) fn transcript_of(session: &SessionPaths) -> Transcript {
    Transcript::read(&session.transcript_json()).unwrap()
}

/// This session's hand-given names as they stand on disk, which is where an answer to a
/// voice too quiet for a reference goes instead of into `speakers.json`.
pub(crate) fn assigned_in(session: &SessionPaths, id: &str) -> SpeakerNames {
    SpeakerNames::read_or_empty(session, &SessionId::parse(id).unwrap()).unwrap()
}

/// Every file under a directory, by path and by contents, so a comparison covers a file
/// created or removed as well as one rewritten.
///
/// Here rather than in one test module because "wrote nothing" is a claim two commands make,
/// and it has to mean the same thing in both: byte-for-byte over the whole root rather than
/// over the files each was expected to touch.
pub(crate) fn files_under(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(dir: &Path, into: &mut Vec<(PathBuf, Vec<u8>)>) {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                walk(&path, into);
            } else {
                into.push((path.clone(), std::fs::read(&path).unwrap()));
            }
        }
    }
    let mut files = Vec::new();
    walk(root, &mut files);
    files
}

/// Turns as (speaker, text, confidence), which is what a reader of the transcript sees.
pub(crate) fn said(transcript: &Transcript) -> Vec<(&str, &str, Option<f32>)> {
    transcript
        .turns
        .iter()
        .map(|t| (t.speaker.as_str(), t.text.as_str(), t.speaker_id_confidence))
        .collect()
}

/// A session whose second voice is under both floors and has already been named for this
/// session alone -- the state the tests below start from. Cluster 0 is left unresolved on
/// purpose, so each of them can also show what happens to a voice nobody named.
pub(crate) fn named_for_its_session(paths: &Paths, id: &str) -> SessionPaths {
    let session = make_session(paths, id);
    with_speech_seconds(&session, &[40.0, 1.5]);

    let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
    let (report, output) = run_asking(paths, &[], ALL, &mut interviewer);
    assert_eq!(report.session_only, 1, "{output}");
    session
}
