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
//! deduplication is enrollment itself. The one exception is [`Offer::named`], which asks about
//! resolved voices too so that an identification the database got wrong can be answered --
//! without it a false accept would be permanent short of hand-editing `speakers.json`.
//!
//! A rewritten transcript is exactly what `transcribe --force` would now produce. That is the
//! invariant everything below is implemented against, because it is what stops `enroll` and
//! `transcribe` from becoming two sources of truth about a transcript. It applies to every
//! session this reads, not only to the one an answer was given in: a transcript written
//! before its speaker was enrolled is brought up to date on the way past, since a session
//! with nothing left to ask about would otherwise keep calling a named colleague "Unknown 2"
//! for good. Files that already agree are left alone, byte for byte.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use meethook_session::{
    AssignedName, Classification, DiscoveredSession, EnrolledSpeaker, EnrolledSpeakers, Paths,
    SessionId, SourceTrack, SpeakerCluster, SpeakerClusters, SpeakerNames, Transcript,
    discover_sessions, unknown_labels,
};
use meethook_transcribe::{
    Attribution, Naming, TARGET_RATE, attributions, identify_clusters, read_track_16k_mono,
};

/// How many of a voice's lines to show before asking who it is.
///
/// Enough to hear a person in the words -- what they said, what they were asked -- without
/// turning a prompt into a page of transcript that hides the question at the bottom of it.
const SNIPPETS: usize = 3;

/// How much of one line to show. Long enough for a sentence, short enough to stay on a line.
const SNIPPET_CHARS: usize = 100;

/// How much a voice has to have spoken before it is worth a question.
///
/// A rule about the prompt queue and nothing else. A cluster below this still keeps its
/// "Unknown N", still holds its turns, and is still relabelled when somebody else's answer
/// turns out to name it -- it is only not *asked about* unless `enroll --all` is passed.
/// Nothing on disk depends on it.
///
/// # Where 5 s comes from
///
/// Clustering emits one cluster per voice it is sure of plus a long tail of fragments it
/// cannot place, and the tail is not a tuning failure: a one-second embedding describes a
/// phoneme and a prosody rather than a person, so no distance rule puts it anywhere. On
/// session `20260810-093047` -- seven people, 1368.7 s of speech -- the shipped clustering
/// leaves 56 clusters, 8 of which identification resolves, so without a floor `enroll` asks
/// 48 questions about a meeting with seven people in it.
///
/// Sorted by talk time those 56 clusters run 426.8 / 423.7 / 124.8 / 119.5 / 96.0 / 51.5 --
/// the six voices the user confirms are the six main speakers, 1242.2 s between them -- and
/// then fall off a cliff to 8.6 / 8.5 / 7.8 / 7.5 / 6.0 / 5.6 / 5.4 / 4.9 / 4.2 / 3.9 / ...
/// into a tail where 29 of the 56 hold under two seconds and 126.5 s covers all fifty of them.
/// Of the 48 clusters left unresolved after identification, **every floor `f` with
/// `4.9 < f <= 5.9` offers the same seven voices and holds back the same 41**; over all 56
/// clusters, ignoring which happen to be enrolled, the partition is fixed across
/// `4.9 < f <= 5.4`. 5 s is the round number in that band rather than a value fitted to this
/// recording.
///
/// Both edges are consequences. Above 7.8 s Alex -- a real seventh participant,
/// 9.8 s of speech split across clusters of 7.83 s and 1.99 s -- stops being offered and can
/// only be reached through `--all`, which is the failure TASK-021 AC #3 names; he happens to
/// be enrolled already in this session, so the cost lands on the next participant like him.
/// Below it the tail arrives fast: 9 voices at a 4 s floor, 15 at 3 s, 21 at 2 s, which is the
/// 48-question prompt again with a smaller number on it.
///
/// # Not [`meethook_transcribe::SPEAKER_FLOOR_SECONDS`], and not [`REFERENCE_FLOOR_SECONDS`]
///
/// Same units, three different questions, and they do not imply one another:
///
/// - `SPEAKER_FLOOR_SECONDS` (30 s) decides **which clusters are solid enough to adopt
///   fragments into** -- how much evidence a centroid rests on before it is allowed to claim
///   somebody else's turns. It is necessarily the larger: at 30 s the seventh participant
///   would not be asked about at all.
/// - This one decides **which voices are worth asking about**. Getting it wrong costs a
///   question, in one direction or the other, and nothing else.
/// - [`REFERENCE_FLOOR_SECONDS`] decides **which answers become references in
///   `speakers.json`**. Naming somebody who spoke 8 s is right; storing a reference built from
///   8 s of audio is what TASK-019 measured going wrong. It landed on the same 5.0 s this one
///   sits on, which is why both state the same boundary convention below: a value offered here
///   and then refused there would be a question asked for nothing.
///
/// The comparison is `speech_seconds >= PROMPT_FLOOR_SECONDS`, the same convention
/// `SPEAKER_FLOOR_SECONDS` states: a cluster sitting exactly on the floor is offered. Two
/// floors in one codebase disagreeing about their own boundary is a bug waiting to happen.
const PROMPT_FLOOR_SECONDS: f64 = 5.0;

/// How much a voice has to have spoken before an answer about it becomes a *reference* in
/// `speakers.json`.
///
/// A rule about what an accepted name writes, and nothing else. Below this a name is recorded
/// against the session in `speaker_names.json` -- the voice still reads as that person in this
/// transcript -- and `speakers.json` is not touched; [`Enrolment::Always`] overrides that.
///
/// # Where 5 s comes from
///
/// TASK-019.01 measured 104 reference-versus-turn similarities on real sessions and put the
/// usable floor in the band **(2.4 s, 5.2 s]**, recommending **5.0 s**, with `>=` meaning
/// above it. Both edges are priced.
///
/// Below 2.4 s, four measured references failed outright: 0.95 s scored 0.702 against its own
/// owner and was rejected, 1.05 s scored 0.807 with the *wrong* argmax and was rejected, 1.3 s
/// scored 0.441, 2.4 s scored 0.392. A reference built from a fragment that short describes a
/// phoneme and a prosody rather than a person, so it neither matches its owner nor stays out
/// of everybody else's way.
///
/// Above 5.2 s it starts refusing references that demonstrably worked: a 5.2 s prefix matched
/// its own speaker at 0.160, and one voice's references spread over 2.9 / 3.2 / 4.6 s all held.
/// Past ~7.8 s the cost has a name on it -- Alex's three ~8 s fragments identify
/// their owner at 0.321 / 0.173 / 0.295, and a floor above them would refuse a reference that
/// works.
///
/// Why 5.0 rather than the middle of the band: everything inside it is selection-biased low by
/// the measured 0.26-0.37, while the only references measured free of that bias succeeded at
/// 7.8 s and failed at 1.05 s -- so the honest reading of the evidence sits at the top of the
/// band rather than in the middle of it. Three caveats travel with the number: that selection
/// effect; one call on one microphone is the easiest condition a reference will ever face; and
/// 104 points are readings rather than rates.
///
/// # Three floors, same units, three questions
///
/// None of them implies another, and they are deliberately not derived from each other:
///
/// - [`meethook_transcribe::SPEAKER_FLOOR_SECONDS`] (30 s) decides **which clusters are solid
///   enough to adopt fragments into**. Necessarily the largest: it is about a centroid being
///   allowed to claim somebody else's turns.
/// - [`PROMPT_FLOOR_SECONDS`] (5 s) decides **which voices are worth asking about**. Getting it
///   wrong costs a question in one direction or the other, and nothing on disk.
/// - This one decides **which answers become references**. Naming somebody who spoke 8 s is
///   right; storing a reference built from 8 s of audio is what TASK-019 measured going wrong.
///
/// # Why it sits on the same value as [`PROMPT_FLOOR_SECONDS`]
///
/// Not a coincidence to be tidied away into one constant, and not a duplication: they answer
/// different questions and would move independently -- a better prompt heuristic does not
/// change what a reference needs, and vice versa. That they coincide today has a consequence
/// worth stating, because it is what bounds this path's exposure: a default run offers only
/// voices at or above 5.0 s, so every answer it collects clears this floor and the
/// session-scoped branch never fires. What reaches it is `enroll --all`, plus any small voice
/// reached through `--correct`.
///
/// Two floors on one value must also agree about their own boundary, or a cluster of exactly
/// 5.0 s would be asked about and then have its answer refused. The comparison here is
/// `speech_seconds >= REFERENCE_FLOOR_SECONDS`, matching both neighbours.
const REFERENCE_FLOOR_SECONDS: f64 = 5.0;

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

/// One voice being asked about, and everything needed to ask.
///
/// Usually a voice nothing in the database matched, which is what `enroll` exists for. Under
/// [`Offer::named`] it can also be one the database has already put a name to, and then the
/// question is a different one -- not "who is this" but "is this right" -- which is what
/// `confidence` tells the caller.
///
/// Deliberately one value rather than a play-then-ask pair of calls: the order those two
/// would have to be made in is exactly the sort of thing a seam should not be leaking.
pub struct Voice<'a> {
    pub session: &'a SessionId,

    /// What this voice is currently called, and on what basis.
    ///
    /// One field rather than a label plus a confidence, because the prompt turns on *which
    /// kind* of label this is and only one of the three carries a number. Reading "has a
    /// confidence" as "already has an answer" is right for exactly as long as an identification
    /// is the only way to get a name onto a voice; [`Attribution::Assigned`] is a name with no
    /// similarity behind it, and a prompt written against the number would ask "who is this?"
    /// about a voice this command named an hour ago.
    ///
    /// [`Attribution::label`] is exactly as the transcript reads -- "Unknown 2", or the name --
    /// so the user can find the voice in the file in front of them.
    pub attribution: &'a Attribution,

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
    fn identify(&mut self, voice: &Voice<'_>) -> Answer;
}

/// Which voices a run offers beyond the ones it offers by default.
///
/// Two orthogonal questions -- how quiet a voice may be, and whether the database has already
/// named it -- deliberately not one flag, because `--all` already answers the first and a user
/// who wants to correct one identification is not asking to be shown the two-second fragments
/// as well. The two filters compose: the floor decides whether a voice is worth a question
/// whatever put it in the list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Offer {
    /// `--all`: voices below `PROMPT_FLOOR_SECONDS`, which are normally held back.
    pub quiet: bool,

    /// `--correct`: voices the database has already put a name to, so a wrong identification
    /// can be answered instead of being permanent.
    pub named: bool,
}

/// When an accepted name is allowed to become a reference in `speakers.json`.
///
/// A separate axis from [`Offer`] rather than a third field on it, because the two answer
/// different questions: `Offer` decides *which voices a run asks about*, and this decides
/// *what an answer writes*. Folding them together would make `--all` -- a request to be shown
/// the quiet voices -- also a request to store references built from two seconds of audio,
/// which is the failure `REFERENCE_FLOOR_SECONDS` exists to prevent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Enrolment {
    /// The default: a voice with at least `REFERENCE_FLOOR_SECONDS` of speech becomes a
    /// reference, and a quieter one is named for its own session only.
    #[default]
    AboveTheFloor,

    /// `--force-reference`: every accepted name becomes a reference, whatever the duration.
    ///
    /// The explicit override TASK-019 requires before a short reference is stored. Worth
    /// having because the floor is a rule about the average case and the user may know better
    /// in a particular one -- the clip is the only recording of somebody, or a legacy
    /// reference built from that same fragment needs replacing, which the default path can
    /// only report and not fix.
    Always,
}

/// How a run is configured: which voices it offers, and what an answer to one writes.
///
/// Private, and a bundle rather than two parameters, because it is threaded through the walk
/// over sessions unchanged and every function on that path would otherwise carry both. The two
/// axes stay separate in the public signature, where a caller has to say which is which.
#[derive(Debug, Clone, Copy, Default)]
struct Rules {
    offer: Offer,
    enrolment: Enrolment,
}

/// What a run did, so the caller can pick an exit status without re-deriving it.
///
/// `named`, `skipped`, `kept` and `held_back` count *voices*; `passed_over` counts *sessions*
/// that were never asked about at all; `failed` counts sessions that could not be read, plus
/// ids that were requested and are not on disk.
///
/// `held_back` is unresolved voices that sat under `PROMPT_FLOOR_SECONDS` and so were never
/// asked about. Reported rather than merely not-counted, because a run that asked seven
/// questions about a meeting of fifty-six voices should say what it did not ask about.
///
/// `kept` is already-named voices the user left as they were -- an answer, and the common one
/// under [`Offer::named`]. Counted apart from `skipped` because they write the same nothing
/// but mean opposite things: a kept voice *has* a name, and folding it into the skipped count
/// would have the summary report a named voice as unnamed.
///
/// `session_only` is a **sub-count of `named`**, not a category beside it: those voices were
/// named, and the name is in their session's `speaker_names.json` rather than in
/// `speakers.json` because they spoke less than `REFERENCE_FLOOR_SECONDS`. A caller adding
/// the fields up would double-count them; a caller checking "did this run name anybody" keeps
/// using `named`, which is what makes it a sub-count rather than a seventh bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnrollReport {
    pub named: usize,
    pub session_only: usize,
    pub skipped: usize,
    pub kept: usize,
    pub held_back: usize,
    pub passed_over: usize,
    pub failed: usize,
}

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
///
/// [`Offer`] widens which voices get asked about -- the quiet ones, the already-named ones, or
/// both. It changes which questions get asked and nothing else: the same answers write the
/// same files however a voice came to be offered. [`Enrolment`] is the other axis, and the
/// only one that changes *what an answer writes* -- which is exactly why the override is a
/// parameter of its own instead of a third field on `Offer`. There are three files an answer
/// can land in now (`speakers.json`, a session's `speaker_names.json`, and its transcript),
/// and which of the first two it is depends on the voice's duration and on this.
pub fn run_enroll(
    paths: &Paths,
    requested: &[SessionId],
    offer: Offer,
    enrolment: Enrolment,
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
    let rules = Rules { offer, enrolment };

    for session in selected {
        match enroll_session(
            paths,
            session,
            rules,
            &mut speakers,
            interviewer,
            out,
            &mut report,
        )? {
            Outcome::Finished => {}
            Outcome::Quit => break,
        }
    }

    Ok(report)
}

/// Asks about every unresolved voice in one session, writing after each accepted name.
///
/// The files are written in a fixed order -- whichever of `speakers.json` and this session's
/// `speaker_names.json` the answer belongs in, then the transcript -- and after every single
/// name rather than once at the end. The name file is what the next labelling reads, so an
/// interrupt between the two writes leaves a name the next run simply re-applies, rather than
/// a transcript naming somebody nothing on disk records. It is also what makes ending a run
/// early cost nothing that was already answered.
///
/// A session this cannot read is reported and counted, and the queue carries on: one session
/// transcribed by a build too old to have recorded first appearances must not end the run.
fn enroll_session(
    paths: &Paths,
    session: &DiscoveredSession,
    rules: Rules,
    speakers: &mut EnrolledSpeakers,
    interviewer: &mut dyn Interviewer,
    out: &mut dyn Write,
    report: &mut EnrollReport,
) -> Result<Outcome> {
    let offer = rules.offer;
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
        // As above, and with the same remedy: the expected instance is a `transcript.json`
        // from before turns recorded which cluster they came from. A user told only "missing
        // field `cluster`" has been given a diagnosis with no next step.
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

    // Voices somebody named in this session without enrolling them. Read here, beside the
    // clusters, so the relabel below already honours them -- a name given in an earlier run is
    // part of what this session's transcript should say, exactly as an enrolled name is.
    let mut assigned = match SpeakerNames::read_or_empty(&session.paths, &session.id) {
        Ok(assigned) => assigned,
        // Unlike the two failures above, no re-transcribe recovers this one: this file holds
        // names a person typed and nothing else can regenerate them, so the only honest
        // instruction is to go and look at it.
        Err(e) => {
            writeln!(
                out,
                "{}  failed: {e} -- fix or delete that file",
                session.id
            )?;
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
    let mut shown = effective_labels(&clusters.clusters, &unknown, speakers, &assigned.names);

    // The transcript may predate an answer given in an earlier session -- name somebody in
    // January's meeting and February's transcript still calls them Unknown 2 -- so it is
    // brought in line before anything is asked. Doing it here rather than only after a name
    // is what stops a session with nothing left to ask about from keeping a stale label
    // forever, since it would be passed over on every later run too. Nothing is written when
    // nothing differs.
    if relabel(&mut transcript, &shown) {
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

    // The one place "already named" is decided. Everything below -- the floor, the in-run
    // guard, the prompt -- treats a voice the same however it got into this list, which is
    // what lets `--all` and `--correct` compose without either knowing about the other.
    let candidates: Vec<&SpeakerCluster> = order
        .into_iter()
        .filter(|c| offer.named || !shown[&c.id].is_named())
        .collect();
    if candidates.is_empty() {
        // A session whose voices are all identified is exactly where somebody stands when one
        // of those identifications is wrong, and this line is the only thing it prints -- so
        // it names the escape, the way the held-back line already names `--all`.
        let named = shown.values().filter(|label| label.is_named()).count();
        if named == 0 {
            writeln!(out, "{}  passed over: nothing unresolved", session.id)?;
        } else {
            writeln!(
                out,
                "{}  passed over: nothing unresolved ({named} named voice(s) -- \
                 meethook enroll --correct)",
                session.id
            )?;
        }
        report.passed_over += 1;
        return Ok(Outcome::Finished);
    }

    let queued = candidates.len();

    // Only the voices worth a question, unless the user asked for the rest. Clustering emits a
    // long tail of one- and two-second fragments it cannot place -- 48 unresolved clusters for
    // a meeting of seven people, measured on `20260810-093047` -- and asking about each of
    // them is how a five-minute job becomes an hour. Filtering preserves first-appearance
    // order, which is what the user reads the transcript in.
    let mut offered: Vec<&SpeakerCluster> = if offer.quiet {
        candidates.clone()
    } else {
        candidates
            .iter()
            .copied()
            .filter(|c| c.speech_seconds >= PROMPT_FLOOR_SECONDS)
            .collect()
    };
    // A floor that hides every voice in a session is not a floor, it is a command that does
    // nothing. A short recording where nobody clears it -- the three-second fixtures the
    // end-to-end tests are built on, and any real meeting that ran for a minute -- offers
    // everybody instead. Decided here rather than defended against, because the alternative is
    // `enroll` reporting "nothing to do" on a session with unnamed people in it.
    if offered.is_empty() {
        offered = candidates;
    }
    let held_back = queued - offered.len();
    report.held_back += held_back;

    // "Unresolved" is false under `--correct`, where most of the queue is resolved and the
    // point is to review it. The default wording is left exactly as it was.
    let counted = if offer.named {
        let already = offered.iter().filter(|c| shown[&c.id].is_named()).count();
        format!(
            "{} voice(s) to review, {already} of them already named",
            offered.len()
        )
    } else {
        format!("{} unresolved voice(s)", offered.len())
    };
    if held_back == 0 {
        writeln!(out, "{}  {counted}", session.id)?;
    } else {
        // Naming the escape rather than only the count: a voice nobody is told about is not
        // reachable, which is what AC #3 asks for.
        writeln!(
            out,
            "{}  {counted}, {held_back} quieter voice(s) not offered -- meethook enroll --all",
            session.id
        )?;
    }

    // Read after that check, so a session with nothing to ask about never resamples an hour
    // of audio in order to then ask nothing. Unreadable is empty rather than fatal: a voice
    // with no clip can still be named from its snippets.
    let track = read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();

    // What each voice was called when this queue was built. The guard below compares against
    // *this* rather than against the live labels, because under `--correct` a queued voice may
    // legitimately be one the database had already named.
    let baseline = shown.clone();

    for cluster in offered {
        // A voice an answer given earlier in this run has already put a name to: clustering
        // that split one person in two must not ask about them twice. Only an in-run answer
        // can have moved a label since `baseline` was taken, so "named, and not the name it
        // had when we queued it" is exactly that case and nothing else. The `is_named()`
        // half matters as much: an in-run answer can also *un*-name a voice -- re-anchoring a
        // reference to another cluster drops this one back to its "Unknown N" -- and that is a
        // question this run created and has not answered.
        if shown[&cluster.id].is_named() && shown[&cluster.id] != baseline[&cluster.id] {
            continue;
        }

        // Scoped so the borrows of `transcript` and `shown` inside the voice end before the
        // answer is acted on.
        let answer = {
            let attribution = &shown[&cluster.id];
            // Keyed on the cluster, not on the label text: under `--correct` two voices can
            // sit under one enrolled name -- which is the false accept being corrected -- and
            // a prompt showing the other person's lines cannot be answered.
            let snippets: Vec<&str> = transcript
                .turns
                .iter()
                .filter(|turn| {
                    turn.source_track == SourceTrack::Speaker && turn.cluster == Some(cluster.id)
                })
                .map(|turn| snippet(&turn.text))
                .filter(|text| !text.is_empty())
                .take(SNIPPETS)
                .collect();

            interviewer.identify(&Voice {
                session: &session.id,
                attribution,
                speech_seconds: cluster.speech_seconds,
                snippets,
                clip: clip_for(&track, cluster),
            })
        };

        // Leaving an already-named voice alone is keeping that identification, which is an
        // answer; leaving an unnamed one alone is the question going unanswered. Same write --
        // none -- and different enough that the summary must not conflate them.
        let left_alone = if shown[&cluster.id].is_named() {
            &mut report.kept
        } else {
            &mut report.skipped
        };

        let name = match answer {
            Answer::Quit => return Ok(Outcome::Quit),
            Answer::Skip => {
                *left_alone += 1;
                continue;
            }
            Answer::Named(name) => name,
        };
        // A name of nothing but spaces is somebody pressing Enter with a stray keystroke in
        // the buffer, not a request for an entry called "".
        let name = name.trim();
        if name.is_empty() {
            *left_alone += 1;
            continue;
        }

        // Naming a voice and storing a reference built from it are two different acts, and
        // this is where they come apart. Below the floor the name is recorded against the
        // session and `speakers.json` is not touched at all -- see `REFERENCE_FLOOR_SECONDS`
        // for what a reference built from two seconds of speech does to every future meeting.
        if cluster.speech_seconds < REFERENCE_FLOOR_SECONDS && rules.enrolment != Enrolment::Always
        {
            // The case being given up by not touching the database here: a legacy reference
            // that *is* this exact fragment (built before the floor existed) stays, and goes
            // on competing as an argmax under somebody else's name. Reported rather than
            // silently left, with the override that fixes it, because an enrollment that is
            // wrong and unmentioned is worse than one that is wrong and named.
            let stale: Vec<&str> = speakers
                .speakers
                .iter()
                .filter(|s| s.name != name && s.embedding == cluster.embedding)
                .map(|s| s.name.as_str())
                .collect();
            for who in stale {
                writeln!(
                    out,
                    "{}  {who} still has a reference built from this voice -- \
                     meethook enroll --force-reference to replace it with {name}",
                    session.id
                )?;
            }

            assigned.assign(cluster.id, name, cluster.embedding.clone());
            assigned.write(&session.paths)?;
            writeln!(
                out,
                "{}  named {name} in this session only: {:.1} s of speech is under the \
                 {REFERENCE_FLOOR_SECONDS} s reference floor -- \
                 meethook enroll --force-reference to store a reference anyway",
                session.id, cluster.speech_seconds
            )?;
            report.named += 1;
            report.session_only += 1;
        } else {
            // A reference identical to this cluster was built from this voice, and the user has
            // just told us this voice is somebody else -- so it is a stored claim about a person
            // it is not of, and it competes as an argmax in every future meeting, winning
            // whenever its name sorts first (`identify::best_match`'s tie-break). Exact equality
            // is the whole condition: a reference derived from another recording of that person
            // is a different vector and a legitimate one, and is left alone.
            let displaced: Vec<String> = speakers
                .speakers
                .iter()
                .filter(|s| s.name != name && s.embedding == cluster.embedding)
                .map(|s| s.name.clone())
                .collect();
            speakers
                .speakers
                .retain(|s| s.name == name || s.embedding != cluster.embedding);
            for who in displaced {
                // An enrollment that vanishes without a line about it is worse than the bug.
                writeln!(
                    out,
                    "{}  {who} no longer has a reference: that voice is {name}",
                    session.id
                )?;
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

            // One voice, one record. A voice named for this session only and then enrolled
            // properly -- the same fragment reached again with `--force-reference`, or a later
            // clustering that gave it enough speech -- must stop also being an assignment, or
            // the two could be made to disagree about who it is.
            if assigned.forget(cluster.id) {
                assigned.write(&session.paths)?;
            }
        }

        // Re-identified against the updated database rather than assumed: naming one voice
        // can also name a second cluster in this session, if clustering split that person in
        // two, and a `--force` re-transcribe would name both.
        let now = effective_labels(&clusters.clusters, &unknown, speakers, &assigned.names);
        if relabel(&mut transcript, &now) {
            transcript.write(&session.paths)?;
        }
        shown = now;
    }

    Ok(Outcome::Finished)
}

/// What each voice is called given the database and this session's hand-given names as they
/// stand: a name the user assigned, else an enrolled name where one matched, else the
/// "Unknown N" its first appearance earned it.
///
/// This is the labelling `merge` performs when it writes a transcript, reached through the
/// same [`attributions`], which is what makes a rewrite here and a `--force` re-transcribe
/// agree on the answer rather than merely be written to. The precedence between the three is
/// stated there and nowhere else.
///
/// `clusters` is what identification runs over and what `assigned` is resolved against;
/// `unknown` is what the transcript was written with, and is the key set of the result. Those
/// two are built from the same file, so every voice gets an entry.
fn effective_labels(
    clusters: &[SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    speakers: &EnrolledSpeakers,
    assigned: &[AssignedName],
) -> BTreeMap<u32, Attribution> {
    attributions(
        unknown,
        Naming::new(clusters, &identify_clusters(clusters, speakers), assigned),
    )
}

/// Rewrites every speaker-track turn to what `labels` says its voice should now be called,
/// reporting whether anything changed.
///
/// Turns are found by the cluster they were attributed to, which `transcript.json` records for
/// exactly this: it is an exact handle on one voice's turns, so what a turn currently *reads*
/// never enters into it. That matters most in the case a label lookup cannot survive -- two
/// voices both matched to one enrolled person, then corrected so they belong to different
/// people. Keyed on text those turns are indistinguishable and the only safe answer is to
/// rewrite neither; keyed on the cluster there is no ambiguity to resolve, and correcting one
/// voice leaves the other's turns exactly where they were.
///
/// The cluster is never written back. `merge` is the sole producer of that field and `enroll`
/// only ever changes what a cluster is *called*, which is what keeps a transcript rewritten
/// here identical to what `transcribe --force` would now produce.
///
/// A turn with no cluster is left alone: on the mic track that is the local speaker, whose
/// name is not `enroll`'s to change, and on the speaker track it only arises in a session
/// where diarization found no clusters -- which has no labels to map and nothing to ask about.
/// A cluster absent from `labels` is left alone for the same reason `merge` ignores an
/// identification for a cluster diarization did not produce.
///
/// Nothing is written when nothing changed, which is what makes a skipped session leave its
/// files byte-identical rather than merely equivalent.
fn relabel(transcript: &mut Transcript, labels: &BTreeMap<u32, Attribution>) -> bool {
    let mut changed = false;
    for turn in &mut transcript.turns {
        if turn.source_track != SourceTrack::Speaker {
            continue;
        }
        let Some(label) = turn.cluster.and_then(|id| labels.get(&id)) else {
            continue;
        };
        if turn.speaker != label.label() || turn.speaker_id_confidence != label.confidence() {
            turn.speaker = label.label().to_string();
            turn.speaker_id_confidence = label.confidence();
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
/// a [`Voice`] is 16 kHz mono because that is the track it was cut from -- and a player that
/// had to be told the rate could be told the wrong one.
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
        /// What the prompt was told this voice is called and on what basis -- which is the only
        /// way a test can check that a correction prompt asked "is this right" rather than
        /// "who is this", and that a voice named for one session says so.
        attribution: Attribution,
        speech_seconds: f64,
        snippets: Vec<String>,
        clip_samples: usize,
    }

    impl Shown {
        fn label(&self) -> &str {
            self.attribution.label()
        }

        fn confidence(&self) -> Option<f32> {
            self.attribution.confidence()
        }
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
            self.seen.iter().map(Shown::label).collect()
        }
    }

    impl Interviewer for Scripted {
        fn identify(&mut self, voice: &Voice<'_>) -> Answer {
            self.seen.push(Shown {
                session: voice.session.to_string(),
                attribution: voice.attribution.clone(),
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

    /// A unit vector `degrees` away from cluster 0's, for the fixtures that are about how
    /// close two voices are: one person clustering split in two, or one reference that matches
    /// both halves. 0.35 of cosine distance is `IDENTIFY_DISTANCE`, so 49 degrees is the edge.
    fn nearly(degrees: f32) -> Vec<f32> {
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

    fn mic_turn(start: f64, text: &str) -> Turn {
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
                speaker_turn(0.0, 0, "Unknown 1", "  hi there  "),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(4.0, 0, "Unknown 1", "let us start"),
            ],
        )
        .write(&session)
        .unwrap();

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

        Transcript::new(
            parsed,
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(3.5, 2, "Unknown 3", "mm"),
                speaker_turn(4.5, 3, "Unknown 4", "yes"),
            ],
        )
        .write(&session)
        .unwrap();

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
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut out = Vec::new();
        let report =
            run_enroll(paths, &requested, offer, enrolment, interviewer, &mut out).unwrap();
        (report, String::from_utf8(out).unwrap())
    }

    /// The database `enroll` would have written by naming these clusters, so a test can start
    /// from "the wrong person is already on this voice" without running a first pass.
    fn enrolled(entries: &[(&str, Vec<f32>)], paths: &Paths) {
        EnrolledSpeakers::new(
            entries
                .iter()
                .map(|(name, embedding)| EnrolledSpeaker {
                    name: name.to_string(),
                    embedding: embedding.clone(),
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

    /// Rewrites this session's cluster embeddings, ids in order, leaving everything else as
    /// [`make_session`] wrote it. The fixture's default is one orthogonal vector per cluster;
    /// the tests about near voices are the ones that need to say otherwise.
    fn with_embeddings(session: &SessionPaths, embeddings: &[Vec<f32>]) {
        let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        for (cluster, embedding) in clusters.clusters.iter_mut().zip(embeddings) {
            cluster.embedding = embedding.clone();
        }
        clusters.write(session).unwrap();
    }

    fn transcript_of(session: &SessionPaths) -> Transcript {
        Transcript::read(&session.transcript_json()).unwrap()
    }

    /// This session's hand-given names as they stand on disk, which is where an answer to a
    /// voice too quiet for a reference goes instead of into `speakers.json`.
    fn assigned_in(session: &SessionPaths, id: &str) -> SpeakerNames {
        SpeakerNames::read_or_empty(session, &SessionId::parse(id).unwrap()).unwrap()
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

    /// The case this whole handle exists for, and the one a label lookup cannot survive: a
    /// false accept has filed cluster 3's voice under the name of the person who is really
    /// cluster 1, so two clusters read "Andrew", and the correction sends them to
    /// different names. Keyed on the label text both turn-groups are one indistinguishable
    /// bucket and the only safe answer is to rewrite neither -- silently leaving the user
    /// looking at an uncorrected transcript. Keyed on the cluster the two are simply
    /// different turns.
    #[test]
    fn correcting_one_of_two_voices_sharing_a_label_leaves_the_other_alone() {
        let mut transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                speaker_turn(0.0, 1, "Andrew", "the real one"),
                mic_turn(1.0, "morning"),
                speaker_turn(2.0, 3, "Andrew", "actually Ryan"),
                speaker_turn(3.0, 1, "Andrew", "the real one again"),
            ],
        );
        for turn in &mut transcript.turns {
            if turn.source_track == SourceTrack::Speaker {
                turn.speaker_id_confidence = Some(0.71);
            }
        }

        // The database after the correction: cluster 3 is Ryan, cluster 1 is still Andrew.
        let labels: BTreeMap<u32, Attribution> = [
            (
                1,
                Attribution::Identified {
                    name: "Andrew".to_string(),
                    similarity: 0.71,
                },
            ),
            (
                3,
                Attribution::Identified {
                    name: "Ryan".to_string(),
                    similarity: 0.88,
                },
            ),
        ]
        .into_iter()
        .collect();

        assert!(
            relabel(&mut transcript, &labels),
            "the correction must be reported as a change, not silently declined"
        );
        assert_eq!(
            said(&transcript),
            [
                ("Andrew", "the real one", Some(0.71)),
                ("You", "morning", None),
                ("Ryan", "actually Ryan", Some(0.88)),
                ("Andrew", "the real one again", Some(0.71)),
            ]
        );
    }

    /// The guard on `merge` staying the sole producer of a turn's provenance: `enroll` changes
    /// what a cluster is called and never which cluster a turn came from. That is what keeps
    /// a rewritten transcript identical to a `--force` re-transcribe, since the field would
    /// otherwise be one `enroll` could drift.
    #[test]
    fn a_rewrite_leaves_every_turns_cluster_exactly_as_it_was() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let before: Vec<Option<u32>> = transcript_of(&session)
            .turns
            .iter()
            .map(|t| t.cluster)
            .collect();

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 2, "{output}");
        let after: Vec<Option<u32>> = transcript_of(&session)
            .turns
            .iter()
            .map(|t| t.cluster)
            .collect();
        assert_eq!(after, before);
        assert_eq!(before, [Some(0), None, Some(1), Some(0)]);
    }

    /// The compatibility decision on `TRANSCRIPT_SCHEMA_VERSION`, at the level a user meets
    /// it: a transcript written before turns recorded their cluster is refused rather than
    /// read with that provenance fabricated, it says how to fix it, and the session after it
    /// is still asked about.
    #[test]
    fn a_transcript_without_clusters_fails_its_session_without_ending_the_queue() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let stale = make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");
        std::fs::write(
            stale.transcript_json(),
            br#"{
              "schema_version": 1,
              "session_id": "20260809-052600",
              "turns": [
                {
                  "speaker": "Unknown 1",
                  "start": 0.0,
                  "end": 1.0,
                  "text": "hi there",
                  "source_track": "speaker",
                  "speaker_id_confidence": null
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
        assert!(
            !session.speaker_names_json().exists(),
            "a run that named nobody must not create a names file either"
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
        // A session where everybody is already named is the one somebody is looking at when
        // one of those names is wrong, and this line is all it prints.
        assert!(
            output.contains("2 named voice(s) -- meethook enroll --correct"),
            "a correction nobody is told how to reach is not reachable: {output}"
        );
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
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

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

    /// TASK-019.03 acceptance criteria #1 and #2, which is the whole ticket in one test: a
    /// voice the database has named the wrong person is reached, corrected, and lands in both
    /// files -- and a later default run does not ask about it again.
    #[test]
    fn correcting_a_named_voice_updates_the_database_and_this_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // Cluster 0 is enrolled under the wrong name.
        let mut first = Scripted::answering(vec![named("Alice"), named("Carol")]);
        run(&paths, &[], &mut first);

        let mut interviewer = Scripted::answering(vec![named("Bob")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        // The question that was asked: a name, and how confident the claim behind it was.
        assert_eq!(interviewer.labels(), ["Alice", "Carol"], "{output}");
        assert_eq!(interviewer.seen[0].confidence(), Some(1.0), "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.kept, 1, "{output}");

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [("Carol", voice(1).as_slice()), ("Bob", voice(0).as_slice())],
            "the corrected name owns this voice, and the wrong one no longer claims it"
        );
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Bob", "You", "Carol", "Bob"]
        );

        // ...and the correction sticks: a later default run has nothing to ask about.
        let mut again = Scripted::default();
        let (report, output) = run(&paths, &[], &mut again);
        assert!(again.seen.is_empty(), "{:?}", again.seen);
        assert_eq!(report.passed_over, 1, "{output}");
    }

    /// Acceptance criterion #3: reaching an already-named voice takes an explicit request. A
    /// default run over a half-identified session offers only the half nothing matched.
    #[test]
    fn a_default_run_still_asks_only_about_unresolved_voices() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);

        let mut default = Scripted::default();
        let (_, output) = run(&paths, &[], &mut default);
        assert_eq!(default.labels(), ["Unknown 2"], "{output}");
        assert!(output.contains("1 unresolved voice(s)"), "{output}");

        let mut correcting = Scripted::default();
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut correcting);
        assert_eq!(correcting.labels(), ["Alice", "Unknown 2"], "{output}");
        assert_eq!(correcting.seen[0].confidence(), Some(1.0), "{output}");
        assert_eq!(correcting.seen[1].confidence(), None, "{output}");
        assert!(
            output.contains("2 voice(s) to review, 1 of them already named"),
            "{output}"
        );
        assert_eq!(report.kept, 1, "{output}");
        assert_eq!(report.skipped, 1, "{output}");
    }

    /// Acceptance criterion #4's other half: pressing Enter on an already-named voice keeps
    /// that identification. The same nothing a skip writes -- byte for byte -- and counted
    /// apart from it, because a kept voice has a name and a skipped one does not.
    #[test]
    fn keeping_an_identification_writes_nothing_and_is_not_counted_as_a_skip() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0)), ("Bob", voice(1))], &paths);

        // A default run first, so the snapshot below is of a transcript already in step with
        // the database and any difference is the correcting run's doing.
        run(&paths, &[], &mut Scripted::default());
        let before = (
            std::fs::read(session.transcript_json()).unwrap(),
            std::fs::read(session.transcript_md()).unwrap(),
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            std::fs::read(paths.speakers_json()).unwrap(),
        );

        // Enter, then Enter with a stray space in the buffer.
        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("   ")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Alice", "Bob"], "{output}");
        assert_eq!(report.kept, 2, "{output}");
        assert_eq!(report.skipped, 0, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert_eq!(
            (
                std::fs::read(session.transcript_json()).unwrap(),
                std::fs::read(session.transcript_md()).unwrap(),
                std::fs::read(session.speaker_clusters_json()).unwrap(),
                std::fs::read(paths.speakers_json()).unwrap(),
            ),
            before
        );
    }

    /// Acceptance criterion #5 under `--correct`, which is where it could regress: the in-run
    /// guard no longer looks at "is this named" alone, so the split-voice case has to be
    /// checked with the flag on as well as off.
    #[test]
    fn correcting_still_asks_once_about_one_voice_clustering_split_in_two() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1"],
            "the second half of one voice must not be asked about: {output}"
        );
        assert_eq!(report.named, 1, "{output}");
    }

    /// The other half of that guard, and the reason it is two conditions rather than one: an
    /// answer can *un*-name a voice. Re-affirming cluster 0 re-anchors Alice's reference to it,
    /// which puts cluster 1 out of range and back to its "Unknown N" -- a question this run
    /// created and has not answered, so it must still be asked.
    #[test]
    fn a_voice_an_answer_unnamed_is_still_asked_about() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // 80 degrees apart, with Alice's reference sitting between them: inside
        // `IDENTIFY_DISTANCE` of both now, and of only cluster 0 once it is re-anchored there.
        with_embeddings(&session, &[nearly(0.0), nearly(80.0)]);
        enrolled(&[("Alice", nearly(40.0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Alice", "Unknown 2"], "{output}");
        assert!(interviewer.seen[0].confidence().is_some(), "{output}");
        assert_eq!(
            interviewer.seen[1].confidence(),
            None,
            "the answer took this voice's name away, so the prompt must not claim one"
        );
        assert_eq!(report.kept, 0, "{output}");
        assert_eq!(report.skipped, 1, "{output}");
    }

    /// One voice cannot be two people's stored reference. Correcting a voice enrolled under
    /// the wrong name leaves that name holding a reference built from somebody else's audio,
    /// which then competes as an exact tie in every future meeting -- and wins whenever it
    /// sorts first. Both orderings are checked, so the fix cannot be about the alphabet.
    #[test]
    fn correcting_a_voice_removes_the_reference_the_wrong_name_kept_of_it() {
        for correction in ["Ryan", "Aaron"] {
            let root = tempfile::tempdir().unwrap();
            let paths = Paths::new(root.path());
            let session = make_session(&paths, "20260809-052600");
            enrolled(&[("Nate", voice(0))], &paths);

            let mut interviewer = Scripted::answering(vec![named(correction)]);
            let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

            assert_eq!(report.named, 1, "{output}");
            assert!(
                output.contains(&format!(
                    "Nate no longer has a reference: that voice is {correction}"
                )),
                "an enrollment must not vanish without a line about it: {output}"
            );
            let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
            let stored: Vec<&str> = speakers.speakers.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(stored, [correction], "{output}");
            assert_eq!(
                transcript_of(&session).turns[0].speaker,
                correction,
                "{output}"
            );
        }
    }

    /// A reference built from a *different* recording of the same person is a legitimate one
    /// and is left alone: only a reference identical to this cluster is a claim about a voice
    /// the user has just said is somebody else.
    #[test]
    fn correcting_a_voice_leaves_the_wrong_names_other_reference_alone() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Nate, enrolled from audio that is not either cluster here, matched to cluster 0 by
        // being merely close to it -- which is the false accept this ticket opens with.
        with_embeddings(
            &paths.session(&SessionId::parse("20260809-052600").unwrap()),
            &[nearly(0.0), nearly(80.0)],
        );
        enrolled(&[("Nate", nearly(20.0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Ryan")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(!output.contains("no longer has a reference"), "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [
                ("Nate", nearly(20.0).as_slice()),
                ("Ryan", nearly(0.0).as_slice())
            ],
            "Nate's own enrollment must survive somebody else's correction"
        );
    }

    /// The prompt finds its lines by the cluster the turns came from, not by what they read.
    /// Two voices under one enrolled name is exactly the case a correction is for, and keyed
    /// on the label text both prompts would show the same person's words.
    #[test]
    fn each_correction_prompt_carries_only_its_own_voices_lines() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // One reference matching both clusters: two voices, one name in the transcript.
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Andrew", nearly(10.0))], &paths);

        let mut interviewer = Scripted::default();
        let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Andrew", "Andrew"],
            "{output}"
        );
        assert_eq!(interviewer.seen[0].snippets, ["hi there", "let us start"]);
        assert_eq!(interviewer.seen[1].snippets, ["and from me"]);
    }

    /// The two flags stay orthogonal: `--correct` reaches the named voices, the floor still
    /// decides which are worth a question, and only `--all` lifts it.
    #[test]
    fn correcting_does_not_lift_the_floor_and_all_does_not_reach_named_voices() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        enrolled(&[("Bob", voice(1))], &paths);

        let mut correcting = Scripted::default();
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut correcting);
        assert_eq!(correcting.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.held_back, 1, "{output}");
        assert!(output.contains("meethook enroll --all"), "{output}");

        let mut both = Scripted::default();
        let (report, output) = run_asking(
            &paths,
            &[],
            Offer {
                quiet: true,
                named: true,
            },
            &mut both,
        );
        assert_eq!(both.labels(), ["Unknown 1", "Bob"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
    }

    /// TASK-021 acceptance criterion #1, at the scale a unit test can hold it: a voice under
    /// [`PROMPT_FLOOR_SECONDS`] is not asked about, and the run says both how many it held
    /// back and how to get at them.
    #[test]
    fn a_voice_too_quiet_to_be_worth_a_question_is_not_asked_about() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.held_back, 1, "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("1 unresolved voice(s), 1 quieter voice(s) not offered"),
            "{output}"
        );
        assert!(
            output.contains("meethook enroll --all"),
            "a held-back voice nobody is told how to reach is not reachable: {output}"
        );
    }

    /// The escape the line above advertises actually reaches them, in the same
    /// first-appearance order the queue always follows.
    #[test]
    fn all_asks_about_the_voices_the_floor_holds_back() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::default();
        let (report, output) = run_asking(
            &paths,
            &[],
            Offer {
                quiet: true,
                ..Offer::default()
            },
            &mut interviewer,
        );

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
        assert!(!output.contains("not offered"), "{output}");
    }

    /// TASK-021 acceptance criterion #2, which is the one that matters: the floor filters
    /// *questions*. Nothing is merged, deleted, renumbered or re-attributed, so the clusters
    /// file is byte-identical and every held-back voice still reads the "Unknown N" it was
    /// written with -- while the voice that was named reads their name.
    #[test]
    fn holding_a_voice_back_changes_no_cluster_and_no_unknown_numbering() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");
        let before = std::fs::read(session.speaker_clusters_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.held_back, 3, "{output}");
        assert_eq!(
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            before,
            "the floor must not touch the clustering"
        );
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Unknown 2", "Unknown 3", "Unknown 4"],
            "held-back voices keep the labels transcribe gave them"
        );
    }

    /// The proof that the floor is a filter on questions and not on labelling: one person
    /// clustering split into a large half and a fragment is named once, from the half that
    /// was offered, and the held-back half is relabelled with them -- exactly as a `--force`
    /// re-transcribe would do it.
    #[test]
    fn naming_an_offered_voice_still_relabels_its_held_back_half() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Alice", "Alice"],
            "the floor decides which voices are asked about, not which turns are labelled"
        );
    }

    /// A floor that hides every voice in a session would be a command that does nothing, so
    /// a recording where nobody clears it offers everybody. This is what keeps the
    /// end-to-end tests -- three seconds of synthesised audio apiece -- meaningful.
    #[test]
    fn a_session_where_nobody_clears_the_floor_offers_everybody() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[1.0, 2.0]);

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
        assert!(output.contains("2 unresolved voice(s)"), "{output}");
        assert!(!output.contains("not offered"), "{output}");
    }

    /// A session whose second voice is under both floors and has already been named for this
    /// session alone -- the state the tests below start from. Cluster 0 is left unresolved on
    /// purpose, so each of them can also show what happens to a voice nobody named.
    fn named_for_its_session(paths: &Paths, id: &str) -> SessionPaths {
        let session = make_session(paths, id);
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_asking(paths, &[], ALL, &mut interviewer);
        assert_eq!(report.session_only, 1, "{output}");
        session
    }

    /// TASK-019 acceptance criteria #1 and #2: an answer about a voice with 1.5 s of speech is
    /// kept, and kept *here* -- the transcript reads as the person the user named, and the
    /// database that every future meeting is matched against is byte-for-byte what it was.
    ///
    /// The two acts the floor separates. Before it, this answer wrote a reference built from a
    /// fragment; now it writes a row in this session's own file and says so.
    #[test]
    fn naming_a_voice_under_the_reference_floor_names_the_session_and_not_the_database() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        // Somebody unrelated is already enrolled, so "unchanged" is a real claim about a real
        // file rather than about one that was never created.
        enrolled(&[("Bob", voice(3))], &paths);
        let before = std::fs::read(paths.speakers_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            report.session_only, 1,
            "a name given to a voice under the floor is a session-scoped one: {output}"
        );
        assert_eq!(
            std::fs::read(paths.speakers_json()).unwrap(),
            before,
            "a voice this quiet must not change the enrolled database at all"
        );

        let assigned = assigned_in(&session, "20260809-052600");
        assert_eq!(
            assigned
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str(), &row.embedding))
                .collect::<Vec<_>>(),
            [(1, "Silas", &voice(1))]
        );

        // Only that voice's turns move, and they carry no confidence: nothing was matched.
        assert_eq!(
            said(&transcript_of(&session)),
            [
                ("Unknown 1", "  hi there  ", None),
                ("You", "morning", None),
                ("Silas", "and from me", None),
                ("Unknown 1", "let us start", None),
            ]
        );

        // Which of the two it did is not something a user should have to infer from a file.
        assert!(
            output.contains("named Silas in this session only"),
            "{output}"
        );
        assert!(output.contains("1.5 s of speech"), "{output}");
        assert!(output.contains("--force-reference"), "{output}");
        assert!(!output.contains("enrolled Silas"), "{output}");
    }

    /// The override the line above advertises: `--force-reference` writes the reference the
    /// floor would have withheld, and then there is nothing session-scoped to record.
    #[test]
    fn force_reference_stores_the_reference_the_floor_would_have_withheld() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_enrolling(&paths, &[], ALL, Enrolment::Always, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.session_only, 0, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Silas");
        assert_eq!(speakers.speakers[0].embedding, voice(1));
        assert!(
            !session.speaker_names_json().exists(),
            "an enrolled voice is not also a session-scoped name: {output}"
        );
        assert!(output.contains("enrolled Silas"), "{output}");

        // And the turns now carry a similarity, because this is an identification.
        assert_eq!(
            said(&transcript_of(&session))[2],
            ("Silas", "and from me", Some(1.0))
        );
    }

    /// TASK-019 acceptance criterion #5: an answer is an answer. A voice named for its session
    /// is not asked about again -- not even by `--all`, which is what reached it in the first
    /// place -- and `--correct` is the way back to it, with the prompt saying what it knows.
    #[test]
    fn a_voice_named_for_its_session_is_asked_about_again_only_by_correct() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");

        let mut again = Scripted::default();
        let (report, output) = run_asking(&paths, &[], ALL, &mut again);
        assert_eq!(
            again.labels(),
            ["Unknown 1"],
            "only the voice nobody named should still be asked about: {output}"
        );
        assert_eq!(report.skipped, 1, "{output}");

        let mut correcting = Scripted::default();
        let (_, output) = run_asking(&paths, &[], ALL_AND_CORRECT, &mut correcting);
        assert_eq!(correcting.labels(), ["Unknown 1", "Silas"], "{output}");
        assert_eq!(
            correcting.seen[1].attribution,
            Attribution::Assigned {
                name: "Silas".to_string()
            },
            "the prompt has to say this name was given to this session, not matched"
        );
        assert_eq!(correcting.seen[1].confidence(), None, "{output}");
        assert_eq!(
            transcript_of(&session).turns[2].speaker,
            "Silas",
            "a run that answered nothing must leave the name where it was"
        );
    }

    /// Correcting one: the row is replaced rather than appended to, so a voice answered twice
    /// is one claim about one voice and not two rows racing to label it.
    #[test]
    fn re_answering_a_voice_named_for_its_session_replaces_its_row() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");

        let mut correcting = Scripted::answering(vec![Answer::Skip, named("Alex")]);
        let (report, output) = run_asking(&paths, &[], ALL_AND_CORRECT, &mut correcting);

        assert_eq!(report.session_only, 1, "{output}");
        assert_eq!(
            assigned_in(&session, "20260809-052600")
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Alex")]
        );
        assert_eq!(transcript_of(&session).turns[2].speaker, "Alex");
    }

    /// One voice, one record. The same fragment reached again with `--force-reference` is a
    /// promotion: the reference is written and the session-scoped row it replaces is dropped,
    /// so the two can never be made to disagree about who this voice is.
    #[test]
    fn enrolling_a_voice_that_was_named_for_its_session_drops_its_row() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_enrolling(
            &paths,
            &[],
            ALL_AND_CORRECT,
            Enrolment::Always,
            &mut interviewer,
        );

        assert_eq!(report.session_only, 0, "{output}");
        assert!(
            assigned_in(&session, "20260809-052600").names.is_empty(),
            "an enrolled voice must stop being an assignment too: {output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].embedding, voice(1));
        assert_eq!(
            said(&transcript_of(&session))[2],
            ("Silas", "and from me", Some(1.0)),
            "the same name, now on the basis of a match"
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
