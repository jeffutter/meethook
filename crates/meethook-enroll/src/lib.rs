//! Naming the voices transcription could not identify.
//!
//! This is the only interactive path in meethook, and it is built so that almost none of it
//! is interactive. Which sessions get visited, which voices get asked about, in what order,
//! and what each answer writes to disk are all decided here, against the one-method
//! [`Interviewer`] seam -- with no terminal and no audio device on this side of it. The live
//! implementation, which prints, plays and reads a line, lives in the CLI crate; the test one
//! answers from a script, which is what makes the sequencing decidable in `cargo test` the
//! way the record loop's already is.
//!
//! Diarization never runs. Everything a prompt needs -- the voice's embedding and the bounds
//! of a clip to play -- was written to `speaker_clusters.json` when the session was
//! transcribed, which is the whole reason that file is on disk.
//!
//! Two rules are worth stating before the code:
//!
//! *Unresolved* is decided against the database as it stands right now, not against the text
//! of the transcript. Name someone in the first session and their voice in the third is
//! matched and passed over, with no cross-session comparison of unnamed voices anywhere: the
//! deduplication is enrollment itself.
//!
//! A rewritten transcript is exactly what `transcribe --force` would now produce. That is the
//! invariant everything below is implemented against, because it is what stops `enroll` and
//! `transcribe` from becoming two sources of truth about a transcript. It applies to every
//! session this reads, not only to the one an answer was given in: a transcript written
//! before its speaker was enrolled is brought up to date on the way past, since a session
//! with nothing left to ask about would otherwise keep calling a named colleague "Unknown 2"
//! for good. Files that already agree are left alone, byte for byte.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::io::Write;
use std::path::{Path, PathBuf};

use meethook_session::{
    Classification, DiscoveredSession, EnrolledSpeaker, EnrolledSpeakers, Paths, SessionId,
    SourceTrack, SpeakerCluster, SpeakerClusters, Transcript, discover_sessions, unknown_labels,
};
use meethook_transcribe::{TARGET_RATE, identify_clusters, read_track_16k_mono};

/// How many of a voice's lines to show before asking who it is.
///
/// Enough to hear a person in the words -- what they said, what they were asked -- without
/// turning a prompt into a page of transcript that hides the question at the bottom of it.
const SNIPPETS: usize = 3;

/// How much of one line to show. Long enough for a sentence, short enough to stay on a line.
const SNIPPET_CHARS: usize = 100;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Session(#[from] meethook_session::Error),

    #[error("could not write the clip to {path}: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },

    #[error("could not write output: {0}")]
    Output(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One unresolved voice, and everything needed to ask who it is.
///
/// Deliberately one value rather than a play-then-ask pair of calls: the order those two
/// would have to be made in is exactly the sort of thing a seam should not be leaking.
pub struct UnknownVoice<'a> {
    pub session: &'a SessionId,

    /// Exactly as the transcript reads -- "Unknown 2" -- so the user can find this voice in
    /// the file in front of them.
    pub label: &'a str,

    /// Total speech attributed to this voice, in seconds. How the user tells a participant
    /// from someone who coughed once.
    pub speech_seconds: f64,

    /// Up to `SNIPPETS` of what this voice said, whitespace-trimmed and cut to
    /// `SNIPPET_CHARS` characters. Empty if the recogniser heard nothing over it.
    pub snippets: Vec<&'a str>,

    /// The longest representative clip: 16 kHz mono, the same rate everything else in
    /// meethook works in.
    ///
    /// Empty when `speaker.wav` is missing or unreadable, which is a voice that can still be
    /// named from its snippets rather than a session that has to fail.
    pub clip: &'a [f32],
}

/// What the user said when asked who a voice is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Named(String),
    Skip,
    /// End the run here. A variant rather than an error because stopping early is an
    /// ordinary outcome -- everything accepted so far is already on disk.
    Quit,
}

/// Asks a user who one voice is.
///
/// Infallible on purpose. A terminal that cannot play audio still has an answer, and one
/// that cannot be read has `Quit`; making this fallible would push terminal errors into the
/// sequencing, which is the one place this design keeps them out of.
pub trait Interviewer {
    fn identify(&mut self, voice: &UnknownVoice<'_>) -> Answer;
}

/// What a run did, so the caller can pick an exit status without re-deriving it.
///
/// `named` and `skipped` count *voices*; `passed_over` counts *sessions* that were never
/// asked about at all; `failed` counts sessions that could not be read, plus ids that were
/// requested and are not on disk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnrollReport {
    pub named: usize,
    pub skipped: usize,
    pub passed_over: usize,
    pub failed: usize,
}

/// A speaker label, and how confident the identity claim in it is: `None` for an "Unknown N",
/// which claims no identity at all.
type Label = (String, Option<f32>);

/// Whether the queue should carry on to the next session.
enum Outcome {
    Finished,
    Quit,
}

/// Works through every unresolved voice in a selection of sessions, asking who each one is.
///
/// With no `requested` ids every discovered session is considered, in session-id order;
/// naming ids scopes the run to those, and one that is not on disk is reported individually
/// rather than ignored -- enrolling three of four requested sessions and exiting 0 would look
/// like success.
///
/// The enrolled database is read once and carried through the run, updated in memory by each
/// accepted name and written before anything else. That is what makes the second session's
/// copy of a person somebody was just named in the first one a match rather than a second
/// prompt.
pub fn run_enroll(
    paths: &Paths,
    requested: &[SessionId],
    interviewer: &mut dyn Interviewer,
    out: &mut dyn Write,
) -> Result<EnrollReport> {
    let discovered = discover_sessions(paths)?;
    let mut report = EnrollReport::default();

    for id in requested {
        if !discovered.iter().any(|session| &session.id == id) {
            writeln!(out, "{id}  not found")?;
            report.failed += 1;
        }
    }

    let selected: Vec<&DiscoveredSession> = if requested.is_empty() {
        discovered.iter().collect()
    } else {
        discovered
            .iter()
            .filter(|session| requested.contains(&session.id))
            .collect()
    };

    if selected.is_empty() && requested.is_empty() {
        writeln!(
            out,
            "No sessions found in {}",
            paths.sessions_dir().display()
        )?;
        return Ok(report);
    }

    let mut speakers = EnrolledSpeakers::read_or_empty(paths)?;

    for session in selected {
        match enroll_session(paths, session, &mut speakers, interviewer, out, &mut report)? {
            Outcome::Finished => {}
            Outcome::Quit => break,
        }
    }

    Ok(report)
}

/// Asks about every unresolved voice in one session, writing after each accepted name.
///
/// The two files are written in a fixed order -- `speakers.json`, then this session's
/// transcript -- and after every single name rather than once at the end. `speakers.json` is
/// what the next identification reads, so an interrupt between the two writes leaves a name
/// the next run simply re-applies, rather than a transcript naming somebody the database has
/// never heard of. It is also what makes ending a run early cost nothing that was already
/// answered.
///
/// A session this cannot read is reported and counted, and the queue carries on: one session
/// transcribed by a build too old to have recorded first appearances must not end the run.
fn enroll_session(
    paths: &Paths,
    session: &DiscoveredSession,
    speakers: &mut EnrolledSpeakers,
    interviewer: &mut dyn Interviewer,
    out: &mut dyn Write,
    report: &mut EnrollReport,
) -> Result<Outcome> {
    match session.classification {
        Classification::Orphaned => {
            writeln!(
                out,
                "{}  passed over: no session.json (the recorder crashed mid-session)",
                session.id
            )?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Valid => {
            writeln!(out, "{}  passed over: not transcribed yet", session.id)?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Transcribed => {}
    }

    let clusters = match SpeakerClusters::read(&session.paths.speaker_clusters_json()) {
        Ok(clusters) => clusters,
        // The expected instance of this is a `speaker_clusters.json` from before first
        // appearances were recorded: without them an "Unknown 2" cannot be mapped back to a
        // voice at all, so the file is refused rather than read with a defaulted zero.
        Err(e) => {
            writeln!(
                out,
                "{}  failed: {e} -- re-transcribe this session with --force",
                session.id
            )?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };
    let mut transcript = match Transcript::read(&session.paths.transcript_json()) {
        Ok(transcript) => transcript,
        Err(e) => {
            writeln!(out, "{}  failed: {e}", session.id)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };

    // The "Unknown N" numbering the transcript was written with, recovered from the clusters
    // file by the one function `transcribe` labels with. Fixed for the whole session: it is a
    // fact about when each voice first spoke, which no answer below changes.
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    // What each voice should be called given the database as it stands.
    let mut shown = effective_labels(&clusters.clusters, &unknown, speakers);

    // The transcript may predate an answer given in an earlier session -- name somebody in
    // January's meeting and February's transcript still calls them Unknown 2 -- so it is
    // brought in line before anything is asked. Doing it here rather than only after a name
    // is what stops a session with nothing left to ask about from keeping a stale label
    // forever, since it would be passed over on every later run too. Nothing is written when
    // nothing differs.
    if relabel(&mut transcript, &unknown, &shown, &shown) {
        transcript.write(&session.paths)?;
        writeln!(out, "{}  transcript brought up to date", session.id)?;
    }

    // First-appearance order, which is "Unknown 1, Unknown 2, ..." -- the order the user
    // reads the transcript in. Talk-time order would put the most-worth-naming voice first
    // and jump around relative to the file they are looking at.
    let mut order: Vec<&SpeakerCluster> = clusters.clusters.iter().collect();
    order.sort_by(|a, b| {
        a.first_spoke_seconds
            .total_cmp(&b.first_spoke_seconds)
            .then(a.id.cmp(&b.id))
    });

    let unresolved = order.iter().filter(|c| shown[&c.id].1.is_none()).count();
    if unresolved == 0 {
        writeln!(out, "{}  passed over: nothing unresolved", session.id)?;
        report.passed_over += 1;
        return Ok(Outcome::Finished);
    }
    writeln!(out, "{}  {unresolved} unresolved voice(s)", session.id)?;

    // Read after that check, so a session with nothing to ask about never resamples an hour
    // of audio in order to then ask nothing. Unreadable is empty rather than fatal: a voice
    // with no clip can still be named from its snippets.
    let track = read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();

    for cluster in order {
        // Identified before this run started, or named by an answer given a moment ago:
        // clustering that split one person in two must not ask about them twice.
        if shown[&cluster.id].1.is_some() {
            continue;
        }

        // Scoped so the borrows of `transcript` and `shown` inside the voice end before the
        // answer is acted on.
        let answer = {
            let label = shown[&cluster.id].0.as_str();
            let snippets: Vec<&str> = transcript
                .turns
                .iter()
                .filter(|turn| turn.source_track == SourceTrack::Speaker && turn.speaker == label)
                .map(|turn| snippet(&turn.text))
                .filter(|text| !text.is_empty())
                .take(SNIPPETS)
                .collect();

            interviewer.identify(&UnknownVoice {
                session: &session.id,
                label,
                speech_seconds: cluster.speech_seconds,
                snippets,
                clip: clip_for(&track, cluster),
            })
        };

        let name = match answer {
            Answer::Quit => return Ok(Outcome::Quit),
            Answer::Skip => {
                report.skipped += 1;
                continue;
            }
            Answer::Named(name) => name,
        };
        // A name of nothing but spaces is somebody pressing Enter with a stray keystroke in
        // the buffer, not a request for an entry called "".
        let name = name.trim();
        if name.is_empty() {
            report.skipped += 1;
            continue;
        }

        // An existing name is replaced rather than appended to or averaged with: typing a
        // name already in the database means the stored reference failed to match this voice,
        // and appending would leave two entries under one name. Matching is exact, so "alice"
        // and "Alice" are two people.
        match speakers.speakers.iter_mut().find(|s| s.name == name) {
            Some(entry) => {
                entry.embedding = cluster.embedding.clone();
                writeln!(out, "{}  updated {name}", session.id)?;
            }
            None => {
                speakers.speakers.push(EnrolledSpeaker {
                    name: name.to_string(),
                    embedding: cluster.embedding.clone(),
                });
                writeln!(out, "{}  enrolled {name}", session.id)?;
            }
        }
        report.named += 1;
        speakers.write(paths)?;

        // Re-identified against the updated database rather than assumed: naming one voice
        // can also name a second cluster in this session, if clustering split that person in
        // two, and a `--force` re-transcribe would name both.
        let now = effective_labels(&clusters.clusters, &unknown, speakers);
        if relabel(&mut transcript, &unknown, &shown, &now) {
            transcript.write(&session.paths)?;
        }
        shown = now;
    }

    Ok(Outcome::Finished)
}

/// What each voice is called given the database as it stands: an enrolled name where one
/// matched, otherwise the "Unknown N" its first appearance earned it.
///
/// This is the labelling `merge` performs when it writes a transcript, reached through the
/// same two functions, which is what makes a rewrite here and a `--force` re-transcribe agree
/// on the answer.
fn effective_labels(
    clusters: &[SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    speakers: &EnrolledSpeakers,
) -> BTreeMap<u32, Label> {
    let identified = identify_clusters(clusters, speakers);
    clusters
        .iter()
        .map(|cluster| {
            let label = match identified.get(&cluster.id) {
                Some(who) => (who.name.clone(), Some(who.similarity)),
                // `unknown` was built from these same clusters, so the lookup is total.
                None => (unknown[&cluster.id].clone(), None),
            };
            (cluster.id, label)
        })
        .collect()
}

/// Rewrites the speaker-track turns whose voice should now read something else, reporting
/// whether anything changed.
///
/// Turns are found by the label they currently read, because that is the only handle on them:
/// `transcript.json` records what each turn was called, not which cluster it came from. A
/// voice can be sitting under either of two labels -- the "Unknown N" it was written with, or
/// a name identification gave it -- so both are accepted as the source, which is what lets
/// this repair a transcript written before its speaker was ever enrolled as well as one
/// written a moment ago.
///
/// Two voices sharing a source label -- both matched to one enrolled person -- are
/// indistinguishable here, so if the database now sends them to *different* labels neither is
/// rewritten. That can only happen when replacing a reference stops one of them matching, and
/// guessing which turns belong to which half would put one person's name on another person's
/// words; `transcribe --force` re-derives it from the audio.
///
/// Nothing is written when nothing changed, which is what makes a skipped session leave its
/// files byte-identical rather than merely equivalent.
fn relabel(
    transcript: &mut Transcript,
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Label>,
    now: &BTreeMap<u32, Label>,
) -> bool {
    let mut destination: BTreeMap<&str, Option<&Label>> = BTreeMap::new();
    for (id, was) in shown {
        let is_now = now.get(id).unwrap_or(was);
        let sources = [was.0.as_str(), unknown[id].as_str()];
        for source in sources {
            match destination.entry(source) {
                Entry::Vacant(slot) => {
                    slot.insert(Some(is_now));
                }
                Entry::Occupied(mut slot) => {
                    if slot.get() != &Some(is_now) {
                        slot.insert(None);
                    }
                }
            }
        }
    }

    let mut changed = false;
    for turn in &mut transcript.turns {
        if turn.source_track != SourceTrack::Speaker {
            continue;
        }
        let Some(Some(label)) = destination.get(turn.speaker.as_str()) else {
            continue;
        };
        if turn.speaker != label.0 || turn.speaker_id_confidence != label.1 {
            turn.speaker = label.0.clone();
            turn.speaker_id_confidence = label.1;
            changed = true;
        }
    }
    changed
}

/// One line of transcript, trimmed and cut to something that fits a prompt.
fn snippet(text: &str) -> &str {
    let trimmed = text.trim();
    match trimmed.char_indices().nth(SNIPPET_CHARS) {
        Some((cut, _)) => &trimmed[..cut],
        None => trimmed,
    }
}

/// The audio to play for one voice: its longest representative, cut out of the speaker track.
///
/// The clip is sliced rather than seeked to because `afplay` cannot seek -- it has no start
/// offset at all -- so somebody has to extract it either way. Slicing the 16 kHz track
/// diarization itself ran on is what makes the seconds in a [`meethook_session::RepresentativeSegment`]
/// impossible to misinterpret: they are offsets into exactly this buffer.
///
/// A range running off the end of the track is clipped to what is there, and anything left
/// empty is a voice asked about without audio rather than a session that fails.
fn clip_for<'a>(track: &'a [f32], cluster: &SpeakerCluster) -> &'a [f32] {
    let Some(segment) = cluster.representatives.first() else {
        return &[];
    };
    let start = sample_at(segment.start).min(track.len());
    let end = sample_at(segment.end).min(track.len());
    if end <= start {
        return &[];
    }
    &track[start..end]
}

fn sample_at(seconds: f64) -> usize {
    (seconds.max(0.0) * f64::from(TARGET_RATE)).round() as usize
}

/// Writes a clip where an external player can reach it: mono, 16 kHz, 32-bit float.
///
/// Here rather than in the caller because the format is this crate's knowledge -- the clip in
/// an [`UnknownVoice`] is 16 kHz mono because that is the track it was cut from -- and a
/// player that had to be told the rate could be told the wrong one.
pub fn write_clip(path: &Path, clip: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let wav = |source| Error::Wav {
        path: path.to_path_buf(),
        source,
    };

    // Not `hound::WavWriter::create`: it tags a mono stream `SPEAKER_FRONT_LEFT`, and a clip
    // that exists so a human can recognise a voice is the last place to send it to one ear.
    let mut writer = meethook_session::wav::create(path, spec).map_err(wav)?;
    for sample in clip {
        writer.write_sample(*sample).map_err(wav)?;
    }
    writer.finalize().map_err(wav)
}

/// The sequencing and the writes, exercised without a terminal and without an audio device.
///
/// Every test below drives [`run_enroll`] against a scripted answerer over real session
/// directories on a temporary disk. What is *not* decidable here is whether a human can name
/// a colleague from what a prompt shows -- the audio, the snippet length, the wording -- which
/// needs a real recording and a real person.
#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use meethook_session::{
        RepresentativeSegment, SPEAKER_YOU, SessionPaths, TRANSCRIPT_SCHEMA_VERSION, Turn,
    };

    use super::*;

    /// A voice recorded exactly as it was shown, so a test can assert on what the user would
    /// have been looking at rather than only on what they answered.
    #[derive(Debug, PartialEq)]
    struct Shown {
        session: String,
        label: String,
        speech_seconds: f64,
        snippets: Vec<String>,
        clip_samples: usize,
    }

    /// An interviewer that answers from a queue and remembers every voice it was asked about.
    /// Answers past the end of the script are skips, so a test that expects no prompt at all
    /// fails on `seen` rather than on a panic somewhere else.
    #[derive(Default)]
    struct Scripted {
        answers: VecDeque<Answer>,
        seen: Vec<Shown>,
    }

    impl Scripted {
        fn answering(answers: Vec<Answer>) -> Scripted {
            Scripted {
                answers: answers.into(),
                seen: Vec::new(),
            }
        }

        fn labels(&self) -> Vec<&str> {
            self.seen.iter().map(|v| v.label.as_str()).collect()
        }
    }

    impl Interviewer for Scripted {
        fn identify(&mut self, voice: &UnknownVoice<'_>) -> Answer {
            self.seen.push(Shown {
                session: voice.session.to_string(),
                label: voice.label.to_string(),
                speech_seconds: voice.speech_seconds,
                snippets: voice.snippets.iter().map(|s| s.to_string()).collect(),
                clip_samples: voice.clip.len(),
            });
            self.answers.pop_front().unwrap_or(Answer::Skip)
        }
    }

    fn named(name: &str) -> Answer {
        Answer::Named(name.to_string())
    }

    /// A distinct unit vector per cluster id, so enrolling one of these voices matches that
    /// cluster and nobody else's.
    fn voice(id: u32) -> Vec<f32> {
        let mut embedding = vec![0.0f32; 4];
        embedding[id as usize % 4] = 1.0;
        embedding
    }

    fn cluster(id: u32, first_spoke: f64, representative: (f64, f64)) -> SpeakerCluster {
        SpeakerCluster {
            id,
            embedding: voice(id),
            speech_seconds: 10.0 + f64::from(id),
            first_spoke_seconds: first_spoke,
            representatives: vec![RepresentativeSegment {
                start: representative.0,
                end: representative.1,
            }],
        }
    }

    fn speaker_turn(start: f64, speaker: &str, text: &str) -> Turn {
        Turn {
            speaker: speaker.to_string(),
            start,
            end: start + 1.0,
            text: text.to_string(),
            source_track: SourceTrack::Speaker,
            speaker_id_confidence: None,
        }
    }

    fn mic_turn(start: f64, text: &str) -> Turn {
        Turn {
            speaker: SPEAKER_YOU.to_string(),
            start,
            end: start + 1.0,
            text: text.to_string(),
            source_track: SourceTrack::Mic,
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
    fn make_session(paths: &Paths, id: &str) -> SessionPaths {
        let id = SessionId::parse(id).unwrap();
        let session = paths.session(&id);
        std::fs::create_dir_all(session.dir()).unwrap();
        // Only its presence is read here; classification never parses it.
        std::fs::write(session.session_json(), b"{}").unwrap();
        write_speaker_wav(&session.speaker_wav());

        SpeakerClusters::new(
            id.clone(),
            vec![cluster(0, 0.0, (0.5, 2.5)), cluster(1, 3.0, (3.0, 5.0))],
        )
        .write(&session)
        .unwrap();

        Transcript::new(
            id,
            vec![
                speaker_turn(0.0, "Unknown 1", "  hi there  "),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, "Unknown 2", "and from me"),
                speaker_turn(4.0, "Unknown 1", "let us start"),
            ],
        )
        .write(&session)
        .unwrap();

        session
    }

    fn run(paths: &Paths, ids: &[&str], interviewer: &mut Scripted) -> (EnrollReport, String) {
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut out = Vec::new();
        let report = run_enroll(paths, &requested, interviewer, &mut out).unwrap();
        (report, String::from_utf8(out).unwrap())
    }

    fn transcript_of(session: &SessionPaths) -> Transcript {
        Transcript::read(&session.transcript_json()).unwrap()
    }

    /// Turns as (speaker, text, confidence), which is what a reader of the transcript sees.
    fn said(transcript: &Transcript) -> Vec<(&str, &str, Option<f32>)> {
        transcript
            .turns
            .iter()
            .map(|t| (t.speaker.as_str(), t.text.as_str(), t.speaker_id_confidence))
            .collect()
    }

    /// A clip exists to be handed to `afplay`, so its header is part of what it is for: a
    /// mono stream tagged `SPEAKER_FRONT_LEFT` reaches the listener in one ear.
    #[test]
    fn a_clip_is_tagged_mono_so_a_player_does_not_put_it_in_one_ear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.wav");
        write_clip(&path, &[0.0, 0.25, -0.25, 0.5]).unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(
            meethook_session::wav::channel_mask_of(&wav),
            Some(meethook_session::wav::MONO_CHANNEL_MASK)
        );
    }

    /// Acceptance criteria #5 and #6, at the level a user meets them: one answer puts a
    /// person in the database and their name on their own turns, and on nobody else's.
    #[test]
    fn naming_a_voice_enrolls_them_and_rewrites_that_sessions_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Alice");
        assert_eq!(speakers.speakers[0].embedding, voice(0));

        assert_eq!(
            said(&transcript_of(&session)),
            [
                ("Alice", "  hi there  ", Some(1.0)),
                ("You", "morning", None),
                ("Unknown 2", "and from me", None),
                ("Alice", "let us start", Some(1.0)),
            ]
        );
        // The rendering is rewritten from the turns, not patched line by line.
        let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
        assert_eq!(markdown, transcript_of(&session).render_markdown());
        assert!(markdown.contains("Alice"), "{markdown}");
        assert!(!markdown.contains("Unknown 1"), "{markdown}");
    }

    /// Acceptance criterion #6's actual claim, which the assertion above only illustrates:
    /// the rewritten transcript is what `transcribe --force` would now produce. Checked by
    /// deriving the labels the way `merge` does -- `unknown_labels` over the clusters,
    /// `identify_clusters` against the database -- rather than by restating the expected
    /// strings, so the two paths cannot drift without this failing.
    #[test]
    fn the_rewritten_transcript_is_what_a_force_re_transcribe_would_produce() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
        run(&paths, &[], &mut interviewer);

        let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let identified = identify_clusters(&clusters.clusters, &speakers);
        let unknown = unknown_labels(
            clusters
                .clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds)),
        );
        // The transcript's speaker turns, in order, are cluster 0, 1, 0.
        let expected: Vec<(String, Option<f32>)> = [0u32, 1, 0]
            .iter()
            .map(|id| match identified.get(id) {
                Some(who) => (who.name.clone(), Some(who.similarity)),
                None => (unknown[id].clone(), None),
            })
            .collect();

        let written: Vec<(String, Option<f32>)> = transcript_of(&session)
            .turns
            .iter()
            .filter(|t| t.source_track == SourceTrack::Speaker)
            .map(|t| (t.speaker.clone(), t.speaker_id_confidence))
            .collect();
        assert_eq!(written, expected);
    }

    /// Acceptance criterion #7: a skip changes nothing, and "nothing" is byte-for-byte. A
    /// rewrite that happened to produce equivalent turns would still churn the files.
    #[test]
    fn skipping_every_voice_leaves_the_files_byte_identical() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let before = (
            std::fs::read(session.transcript_json()).unwrap(),
            std::fs::read(session.transcript_md()).unwrap(),
            std::fs::read(session.speaker_clusters_json()).unwrap(),
        );

        let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(report.named, 0);
        assert_eq!(
            (
                std::fs::read(session.transcript_json()).unwrap(),
                std::fs::read(session.transcript_md()).unwrap(),
                std::fs::read(session.speaker_clusters_json()).unwrap(),
            ),
            before
        );
        assert!(
            !paths.speakers_json().exists(),
            "a run that named nobody must not create a database"
        );
    }

    /// Acceptance criterion #4, and the boundary the clusters file exists to defend: enroll
    /// reads it and never writes it, so nothing here can start depending on a name being in
    /// there.
    #[test]
    fn a_run_that_names_everybody_still_leaves_the_clusters_file_untouched() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let before = std::fs::read(session.speaker_clusters_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
        run(&paths, &[], &mut interviewer);

        assert_eq!(
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            before
        );
    }

    /// Acceptance criterion #1, and the deduplication rule: the same person in two sessions is
    /// asked about once, because the second session identifies them from the answer given in
    /// the first. Sessions are worked through in id order.
    #[test]
    fn a_person_named_in_one_session_is_matched_rather_than_asked_about_again() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");

        // One name, then skips: whoever is asked about after Alice is somebody else.
        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        let sessions: Vec<&str> = interviewer
            .seen
            .iter()
            .map(|v| v.session.as_str())
            .collect();
        assert_eq!(
            sessions,
            ["20260809-052600", "20260809-052600", "20260809-052700"],
            "expected both voices of the first session, then the second session's other voice"
        );
        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2", "Unknown 2"],
            "the second session's Alice must not be asked about again"
        );

        // ...and her name reaches the second session's transcript anyway, on the way past.
        for session in [&first, &second] {
            assert_eq!(
                transcript_of(session).turns[0].speaker,
                "Alice",
                "in {}",
                session.dir().display()
            );
        }
    }

    /// Acceptance criterion #8: nothing to ask about is passed over silently rather than
    /// prompting, and so is a session nobody has transcribed yet.
    #[test]
    fn sessions_with_nothing_to_ask_about_are_passed_over_without_prompting() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());

        // Already fully identified.
        let resolved = make_session(&paths, "20260809-052600");
        EnrolledSpeakers::new(vec![
            EnrolledSpeaker {
                name: "Alice".to_string(),
                embedding: voice(0),
            },
            EnrolledSpeaker {
                name: "Bob".to_string(),
                embedding: voice(1),
            },
        ])
        .write(&paths)
        .unwrap();

        // Recorded but never transcribed.
        let untranscribed = paths.session(&SessionId::parse("20260809-052700").unwrap());
        std::fs::create_dir_all(untranscribed.dir()).unwrap();
        std::fs::write(untranscribed.session_json(), b"{}").unwrap();

        // The recorder died mid-session.
        let orphan = paths.session(&SessionId::parse("20260809-052800").unwrap());
        std::fs::create_dir_all(orphan.dir()).unwrap();

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert!(interviewer.seen.is_empty(), "{:?}", interviewer.seen);
        assert_eq!(report.passed_over, 3, "{output}");
        assert_eq!(report.failed, 0, "{output}");
        assert!(output.contains("nothing unresolved"), "{output}");
        assert!(output.contains("not transcribed yet"), "{output}");
        assert!(output.contains("no session.json"), "{output}");
        // Nobody was asked, and the transcript still caught up with the database: a session
        // where everyone is already known is exactly the one that would otherwise be passed
        // over on every future run, keeping its stale labels for good.
        assert_eq!(
            said(&transcript_of(&resolved))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Bob", "Alice"]
        );
        assert!(output.contains("brought up to date"), "{output}");
    }

    /// Acceptance criterion #2: ids scope the run, and one that is not on disk is named
    /// rather than quietly doing less than was asked.
    #[test]
    fn ids_scope_the_run_and_an_unknown_id_is_reported() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let untouched = make_session(&paths, "20260809-052700");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(
            &paths,
            &["20260809-052600", "20260809-999999"],
            &mut interviewer,
        );

        assert!(output.contains("20260809-999999  not found"), "{output}");
        assert_eq!(report.failed, 1);
        assert_eq!(report.named, 1);
        for voice in &interviewer.seen {
            assert_eq!(voice.session, "20260809-052600", "{voice:?}");
        }
        assert_eq!(transcript_of(&untouched).turns[0].speaker, "Unknown 1");
    }

    /// Acceptance criterion #9: ending the run early keeps everything already answered. The
    /// name given before the quit is on disk in both files, and nothing after it was asked.
    #[test]
    fn quitting_keeps_every_name_accepted_so_far() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");

        let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Quit]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(interviewer.seen.len(), 2, "{:?}", interviewer.seen);

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(transcript_of(&first).turns[0].speaker, "Alice");
        assert!(
            std::fs::read_to_string(first.transcript_md())
                .unwrap()
                .contains("Alice")
        );
        // The queue stopped where it was told to, rather than carrying on to the next session.
        assert_eq!(transcript_of(&second).turns[0].speaker, "Unknown 1");
    }

    /// Acceptance criterion #5's other half, and the drift case: typing a name already in the
    /// database replaces that person's reference instead of leaving two entries under it.
    #[test]
    fn naming_someone_already_enrolled_replaces_their_reference() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Alice, enrolled from a voice that matches neither cluster here.
        EnrolledSpeakers::new(vec![EnrolledSpeaker {
            name: "Alice".to_string(),
            embedding: voice(3),
        }])
        .write(&paths)
        .unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(output.contains("updated Alice"), "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1, "{:?}", speakers.speakers);
        assert_eq!(speakers.speakers[0].embedding, voice(0));
    }

    /// Acceptance criterion #3 and the queue order: each prompt carries that voice's own
    /// lines and its own clip, and they arrive in "Unknown N" order rather than in talk-time
    /// order.
    ///
    /// Cluster 0 is the first to speak and cluster 1 the second, so the labels below are also
    /// the assertion that first-appearance order is what the queue follows.
    #[test]
    fn each_prompt_carries_that_voices_snippets_and_clip_in_unknown_order() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::default();
        run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"]);
        assert_eq!(
            interviewer.seen[0].snippets,
            ["hi there", "let us start"],
            "only this voice's lines, whitespace trimmed"
        );
        assert_eq!(interviewer.seen[1].snippets, ["and from me"]);
        assert_eq!(interviewer.seen[0].speech_seconds, 10.0);
        // The representative spans 0.5 s to 2.5 s of a 16 kHz track.
        assert_eq!(interviewer.seen[0].clip_samples, 32_000);
        assert_eq!(interviewer.seen[1].clip_samples, 32_000);
    }

    /// Acceptance criterion #11: no audio is not a failed session. The prompt still happens,
    /// still carries the snippets, and an answer still lands on disk.
    #[test]
    fn a_session_with_no_speaker_wav_is_still_asked_about_with_an_empty_clip() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        std::fs::remove_file(session.speaker_wav()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.failed, 0, "{output}");
        assert_eq!(interviewer.seen[0].clip_samples, 0);
        assert_eq!(interviewer.seen[0].snippets, ["hi there", "let us start"]);
        assert_eq!(transcript_of(&session).turns[0].speaker, "Alice");
    }

    /// A representative that runs off the end of the track -- a truncated `speaker.wav` -- is
    /// clipped to what is there rather than refused, for the same reason as above.
    #[test]
    fn a_representative_past_the_end_of_the_track_plays_what_is_there() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let id = SessionId::parse("20260809-052600").unwrap();
        SpeakerClusters::new(
            id,
            vec![
                cluster(0, 0.0, (5.0, 90.0)),
                cluster(1, 3.0, (600.0, 620.0)),
            ],
        )
        .write(&session)
        .unwrap();

        let mut interviewer = Scripted::default();
        run(&paths, &[], &mut interviewer);

        // The track is six seconds long: one second of the first clip survives, none of the
        // second.
        assert_eq!(interviewer.seen[0].clip_samples, 16_000);
        assert_eq!(interviewer.seen[1].clip_samples, 0);
    }

    /// A session transcribed by a build that did not record first appearances cannot be
    /// mapped from "Unknown 2" back to a voice, so it is reported and counted -- and the
    /// session after it is still asked about.
    #[test]
    fn a_stale_clusters_file_fails_its_session_without_ending_the_queue() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let stale = make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");
        std::fs::write(
            stale.speaker_clusters_json(),
            br#"{
              "schema_version": 1,
              "session_id": "20260809-052600",
              "clusters": [
                {
                  "id": 0,
                  "embedding": [1.0, 0.0, 0.0, 0.0],
                  "speech_seconds": 42.5,
                  "representatives": [{ "start": 1.0, "end": 3.0 }]
                }
              ]
            }"#,
        )
        .unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.failed, 1, "{output}");
        assert!(output.contains("--force"), "{output}");
        assert_eq!(report.named, 1, "{output}");
        for voice in &interviewer.seen {
            assert_eq!(voice.session, "20260809-052700", "{voice:?}");
        }
    }

    /// A blank answer is a skip, not an entry called "". Somebody pressing Enter with a stray
    /// space in the buffer must not end up in the database.
    #[test]
    fn a_blank_name_is_a_skip_rather_than_an_empty_entry() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("   "), named("  Bob  ")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.skipped, 1, "{output}");
        assert_eq!(report.named, 1, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        // Trimmed, so the transcript does not read "**[00:03]   Bob  :**".
        assert_eq!(speakers.speakers[0].name, "Bob");
    }

    /// One person clustering split in two is named once and lands on both halves, because
    /// that is what a `--force` re-transcribe would do with the reference this answer just
    /// stored.
    #[test]
    fn naming_a_split_voice_names_its_other_half_without_asking_twice() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // Two clusters a few degrees apart: one voice the clusterer did not join up.
        let nearly = |degrees: f32| {
            let radians: f32 = degrees.to_radians();
            vec![radians.cos(), radians.sin(), 0.0, 0.0]
        };
        let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        clusters.clusters[0].embedding = nearly(0.0);
        clusters.clusters[1].embedding = nearly(20.0);
        clusters.write(&session).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            interviewer.labels(),
            ["Unknown 1"],
            "the second half of one voice must not be asked about"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Alice", "Alice"]
        );
    }

    /// The transcript's schema version survives a rewrite: `enroll` edits turns, it does not
    /// re-stamp the file as something it is not.
    #[test]
    fn a_rewritten_transcript_keeps_its_schema_version_and_session_id() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        run(&paths, &[], &mut interviewer);

        let transcript = transcript_of(&session);
        assert_eq!(transcript.schema_version, TRANSCRIPT_SCHEMA_VERSION);
        assert_eq!(transcript.session_id.as_str(), "20260809-052600");
    }

    /// An empty meethook directory is a first run, not an error.
    #[test]
    fn no_sessions_at_all_is_reported_rather_than_failing() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report, EnrollReport::default());
        assert!(output.contains("No sessions found"), "{output}");
    }

    /// A long line is cut to something that fits a prompt, on a character boundary rather
    /// than a byte one.
    #[test]
    fn a_long_snippet_is_cut_to_a_readable_length() {
        let long = "é".repeat(SNIPPET_CHARS * 2);
        assert_eq!(snippet(&long).chars().count(), SNIPPET_CHARS);
        assert_eq!(snippet("  short  "), "short");
    }
}
