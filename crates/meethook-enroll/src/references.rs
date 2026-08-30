//! What every stored reference is currently naming, derived rather than recorded.
//!
//! `speakers.json` holds one row per confirmed recording of a person, so "Alice" can be five
//! rows that are indistinguishable from each other in the file. At the reference cap that is
//! exactly the moment somebody has to choose one to drop, with nothing on the screen to choose
//! on. The useful fact about a reference is not its vector: it is **which voices in which
//! sessions read a name because of it**.
//!
//! That is computable without provenance and without audio. A session's labelling is a pure
//! function of three things on disk -- `speaker_clusters.json`, `speaker_names.json` and the
//! database in memory -- which is what [`crate::effective_labels`] is, and what makes labelling
//! one session cost a few hundred dot products and no model load. Label every session with the
//! database as it stands, label it again with one row removed, and the diff is what that row is
//! buying. `enroll` already performs exactly this two-labelling diff over one session before it
//! honours an answer; this is the same move over every session under the root.
//!
//! # Nothing is written, and that is structural rather than a promise
//!
//! [`scan`] reads three kinds of file and constructs [`EnrolledSpeakers`] values in memory.
//! There is no `write` on any path out of this module, which is why the listing is separable
//! from the removal that consults it: there is no half-finished on-disk state to leave behind.
//!
//! That claim is worth keeping literally true, which is why [`crate::forget`] -- a write path --
//! lives beside this module rather than in it, and reaches the derivation through
//! [`label_sessions`], [`Labelled::labels`] and [`Labelled::moved`]. Sharing the derivation rather
//! than copying it is what guarantees that the preview a user consents to before a removal and the
//! listing they chose a reference from are the *same* labelling, and that neither can drift into
//! disagreeing with what `merge` writes.
//!
//! # What it can and cannot speak for, said out loud
//!
//! A report whose whole claim is completeness has to name its own edges, or its silence gets
//! read as "nothing depends on this":
//!
//! - **Its scope.** Only the transcribed sessions under this root. One that was deleted, or
//!   lives under another `--root`, or has not been transcribed yet, is one it has no opinion
//!   about -- so the counts it read are printed above everything derived from them.
//! - **The sessions it could not read.** A `speaker_clusters.json` from before first appearances
//!   were recorded, or a malformed `speaker_names.json`, is a session with no opinion in it.
//!   Each is named with the same remedy `enroll` gives for the same broken file, and the scan
//!   carries on.
//!
//! # `transcript.json` is deliberately not read
//!
//! The label a voice "reads as" here is derived through [`crate::effective_labels`], which is
//! the labelling `merge` writes a transcript with -- so the derived label is what the file says,
//! or what the next `enroll` will make it say if the file is stale. Reading the transcript
//! instead would report a stale label as fact and would double the I/O for nothing.

use std::collections::BTreeMap;
use std::io::Write;

use meethook_session::{
    AssignedName, Classification, EnrolledSpeakers, Paths, SessionId, SpeakerClusters,
    SpeakerNames, discover_sessions, unknown_labels,
};
use meethook_transcribe::Attribution;

use crate::{Result, effective_labels};

/// Everything one scan found: who is enrolled, what each of their references is naming, and
/// what the scan was able to look at.
///
/// A value rather than a streaming report -- unlike [`crate::EnrollReport`], which counts a walk
/// as it happens -- because the whole thing is derived before a line of it is printed, and
/// because a caller that wants to pick an exit status or (later) preview a removal wants the
/// derivation without the printing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scan {
    /// Grouped by person, in the order enrolment put them in. Empty when nobody is enrolled,
    /// which is the one case that short-circuits before any session is read.
    pub people: Vec<Enrolled>,

    /// Session directories under `sessions/`, whatever state each is in.
    pub sessions_found: usize,

    /// Those of them with a transcript, which are the only ones that have a labelling to diff.
    pub sessions_transcribed: usize,

    /// Those the scan actually labelled: `sessions_transcribed` less the ones in `unreadable`.
    pub sessions_read: usize,

    /// Transcribed sessions the scan has no opinion about, each with its reason and remedy.
    pub unreadable: Vec<Unreadable>,
}

/// One person, and every recording of them the database holds.
#[derive(Debug, Clone, PartialEq)]
pub struct Enrolled {
    pub name: String,

    /// In file order, which is enrolment order, and which is the order `handle` counts in.
    pub references: Vec<Reference>,
}

/// One stored recording of one person, described by what it is currently doing.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    /// 1-based position among the rows bearing this name, in file order: the handle this
    /// listing prints and [`EnrolledSpeakers::without`] accepts.
    pub handle: usize,

    /// How much speech this row was built from, or `None` for one stored before meethook
    /// recorded that.
    ///
    /// Printed because this listing exists to be read by somebody choosing which reference to
    /// drop, and length is half of that judgement: what a reference is *naming* says what
    /// removing it costs today, and how long a recording it came from says how good it is
    /// likely to be tomorrow. It is also what `enroll` compares at the cap, so a listing that
    /// hid it would leave the user unable to predict the one decision the tool makes for them.
    pub clip_seconds: Option<f64>,

    /// Voices that read this person's name and would stop reading it if this row were removed.
    ///
    /// Empty is the answer somebody at the cap is looking for: a reference that is naming
    /// nothing in any session the scan could read.
    pub depends: Vec<VoiceChange>,

    /// Every *other* voice whose label would move, which is not the same question and is not
    /// the smaller one.
    ///
    /// Removing one row moves labels in three directions rather than one, because
    /// identification is an argmax over rows that is then vetoed per name: a voice reverts to
    /// its "Unknown N"; a voice reads *somebody else*, because the second-nearest row belongs to
    /// them; and a voice **gains** a name, because this row was winning a name the heard-at-once
    /// veto then denied to a contender. The first is `depends`. The other two land here whenever
    /// the voice was not reading this person to begin with.
    ///
    /// Reported even though nothing asks for it, because the diff produces it for free and a
    /// listing that showed only the reverting voices would let somebody believe a removal
    /// touches nothing else.
    pub elsewhere: Vec<VoiceChange>,
}

/// One voice whose label would move, and how.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceChange {
    pub session: SessionId,

    /// The "Unknown N" this voice's first appearance earned it -- the handle that reaches a
    /// voice whatever it is called, and what `enroll --voice` accepts. The same choice
    /// `enroll`'s refusal lines make, for the same reason.
    pub voice: String,

    /// What it reads as now.
    pub reads: String,

    /// What it would read as with this reference gone. The whole difference between "this
    /// reverts to a number" and "this starts reading somebody else", which is what a user at the
    /// cap is choosing between.
    pub would_read: String,
}

/// A transcribed session the scan has no opinion about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unreadable {
    pub session: SessionId,

    /// The error and its remedy, worded exactly as `enroll_session` words the same two failures.
    /// Two commands giving different instructions for one broken file is the inconsistency the
    /// wording is copied to avoid.
    pub why: String,
}

/// One session as the diff needs it: everything to label it again, and what it reads now.
pub(crate) struct Labelled {
    pub(crate) session: SessionId,
    pub(crate) clusters: SpeakerClusters,
    /// The "Unknown N" numbering the transcript was written with. Also the key set of both
    /// labellings, so a change can always be named by the voice it happened to.
    pub(crate) unknown: BTreeMap<u32, String>,
    pub(crate) assigned: Vec<AssignedName>,
    /// The session's one-remote-speaker assertion, if there is one. Both labellings honour it,
    /// which is what keeps a removal from being reported as moving a label the assertion --
    /// not the removed row -- still holds in place.
    pub(crate) one_remote_speaker: Option<String>,
    /// What every voice reads with the database exactly as it stands on disk.
    pub(crate) baseline: BTreeMap<u32, Attribution>,
}

impl Labelled {
    /// What every voice in this session reads with `speakers` in place of the database on disk.
    ///
    /// The counterfactual half of the diff. Returned as the map rather than folded into
    /// [`Self::moved`] so a caller that is going to *act* on the result -- rewriting this
    /// session's transcript to match -- can hand the very labelling it inspected to
    /// [`crate::relabel`] instead of deriving it a second time and hoping the two agree.
    pub(crate) fn labels(&self, speakers: &EnrolledSpeakers) -> BTreeMap<u32, Attribution> {
        effective_labels(
            &self.clusters.clusters,
            &self.unknown,
            speakers,
            &self.assigned,
            self.one_remote_speaker.as_deref(),
        )
    }

    /// Every voice whose label differs between [`Self::baseline`] and `after`, in cluster order.
    ///
    /// Compared on the label rather than on the whole attribution: a voice that still reads
    /// "Alice" through another of her recordings, at a different similarity, has not changed what
    /// it says. Both maps are keyed by `unknown`, which is built over every cluster in the
    /// session, so the lookups are total.
    ///
    /// Nothing here is ever an [`Attribution::Assigned`] voice on the removal paths that use it:
    /// no change to the database can move a hand-given name, so a session-only name never appears
    /// in a diff -- which is how it stays out of every report without a rule saying so.
    pub(crate) fn moved(&self, after: &BTreeMap<u32, Attribution>) -> Vec<VoiceChange> {
        let mut changes = Vec::new();
        for (id, reads) in &self.baseline {
            let Some(would_read) = after.get(id) else {
                continue;
            };
            if reads.label() == would_read.label() {
                continue;
            }
            changes.push(VoiceChange {
                session: self.session.clone(),
                voice: self.unknown[id].clone(),
                reads: reads.label().to_string(),
                would_read: would_read.label().to_string(),
            });
        }
        changes
    }
}

/// Every transcribed session a labelling can speak for, and the ones it cannot.
///
/// The scope statement and the per-session baselines together, because they are two halves of one
/// claim: a report that printed "no voice changes" without the count it read would be silence
/// mistaken for evidence.
pub(crate) struct Labelling {
    /// One entry per session actually labelled, in discovery order.
    pub(crate) sessions: Vec<Labelled>,
    /// Session directories under `sessions/`, whatever state each is in.
    pub(crate) found: usize,
    /// Those of them with a transcript, which are the only ones that have a labelling at all.
    pub(crate) transcribed: usize,
    /// Transcribed sessions with no opinion in them, each with its reason and remedy.
    pub(crate) unreadable: Vec<Unreadable>,
}

/// Labels every transcribed session under the root against `speakers`, naming the ones it could
/// not read.
///
/// The shared half of the derivation: [`scan`] is this plus one counterfactual per stored
/// reference, and [`crate::forget`] is this plus one counterfactual per removal. A session that
/// cannot be read is named in [`Labelling::unreadable`] and skipped rather than failing the run,
/// because one session transcribed by a build too old to have recorded first appearances must not
/// cost the report on all the others.
pub(crate) fn label_sessions(paths: &Paths, speakers: &EnrolledSpeakers) -> Result<Labelling> {
    let discovered = discover_sessions(paths)?;
    let mut labelling = Labelling {
        sessions: Vec::new(),
        found: discovered.len(),
        transcribed: 0,
        unreadable: Vec::new(),
    };

    for session in &discovered {
        if session.classification != Classification::Transcribed {
            continue;
        }
        labelling.transcribed += 1;

        let clusters = match SpeakerClusters::read(&session.paths.speaker_clusters_json()) {
            Ok(clusters) => clusters,
            // The expected instance is a file from before first appearances were recorded:
            // without them an "Unknown 2" cannot be mapped back to a voice at all.
            Err(e) => {
                labelling.unreadable.push(Unreadable {
                    session: session.id.clone(),
                    why: format!("{e} -- re-transcribe this session with --force"),
                });
                continue;
            }
        };
        let assigned = match SpeakerNames::read_or_empty(&session.paths, &session.id) {
            Ok(assigned) => assigned.names,
            // No re-transcribe recovers this one: the file holds names a person typed, so the
            // only honest instruction is to go and look at it.
            Err(e) => {
                labelling.unreadable.push(Unreadable {
                    session: session.id.clone(),
                    why: format!("{e} -- fix or delete that file"),
                });
                continue;
            }
        };

        // Read beside the other two files the labelling turns on: a session that asserts one
        // remote speaker would otherwise be diffed through a rule the transcript does not use,
        // and every reference in it would report naming voices the assertion, not the row, holds.
        let one_remote_speaker = match session.load_metadata() {
            Ok(metadata) => metadata.one_remote_speaker,
            Err(e) => {
                labelling.unreadable.push(Unreadable {
                    session: session.id.clone(),
                    why: format!("{e} -- fix that file"),
                });
                continue;
            }
        };

        let unknown = unknown_labels(
            clusters
                .clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds)),
        );
        let baseline = effective_labels(
            &clusters.clusters,
            &unknown,
            speakers,
            &assigned,
            one_remote_speaker.as_deref(),
        );
        labelling.sessions.push(Labelled {
            session: session.id.clone(),
            clusters,
            unknown,
            assigned,
            one_remote_speaker,
            baseline,
        });
    }

    Ok(labelling)
}

/// What every stored reference is currently naming, derived from the sessions on disk.
///
/// Reads `speakers.json` and, per transcribed session, `speaker_clusters.json` and
/// `speaker_names.json`. Writes nothing.
///
/// A database that exists and cannot be read is the one outcome worth interrupting for, so it is
/// an `Err` here exactly as it is in every other reader. A *session* that cannot be read is not:
/// it is named in [`Scan::unreadable`] and the scan carries on, because one session transcribed
/// by a build too old to have recorded first appearances must not cost the report on all the
/// others.
///
/// # Cost
///
/// `(references + 1)` labellings per session, each an argmax over rows: a 25-row database over a
/// 56-cluster session is a few million multiply-adds, no model load and no `speaker.wav` read.
/// That is what makes this a plain synchronous command with no progress reporting.
///
/// Two things it deliberately does not do. It does not deduplicate identical rows -- if a
/// hand-edited file holds one vector twice under one name, removing either alone changes nothing
/// and both report as naming nothing, which is the honest answer to the question actually asked,
/// since a removal removes *one* reference. And it does not try to share labellings between
/// sessions: each session has its own clusters, so there is nothing to share.
pub fn scan(paths: &Paths) -> Result<Scan> {
    let speakers = EnrolledSpeakers::read_or_empty(paths)?;
    // Nobody enrolled short-circuits before discovery: there is no question the scope counts
    // would answer, and reading a hundred session directories to say nothing would be work for
    // nothing. Absent and empty collapse into one case because `read_or_empty` already collapses
    // them, and telling them apart here would need a second `stat` to say something no user
    // needs to know.
    if speakers.speakers.is_empty() {
        return Ok(Scan::default());
    }

    let labelling = label_sessions(paths, &speakers)?;
    let mut found = Scan {
        sessions_found: labelling.found,
        sessions_transcribed: labelling.transcribed,
        // Read off the list that was built rather than subtracted from the counts, so the two
        // cannot drift apart.
        sessions_read: labelling.sessions.len(),
        unreadable: labelling.unreadable,
        ..Scan::default()
    };

    for name in speakers.enrolled_names() {
        let held = speakers.references(name);
        let mut references = Vec::with_capacity(held);
        for handle in 1..=held {
            // `held` is exactly how many rows this name has, so this cannot miss; the `else`
            // exists so a future change to either side is a shorter listing rather than a panic.
            let Some(rest) = speakers.without(name, handle) else {
                continue;
            };
            let mut depends = Vec::new();
            let mut elsewhere = Vec::new();
            for session in &labelling.sessions {
                for change in session.moved(&session.labels(&rest)) {
                    // A voice reading *this* person is what this reference is naming; anything
                    // else moving is a consequence of the removal rather than its subject.
                    if change.reads == name {
                        depends.push(change);
                    } else {
                        elsewhere.push(change);
                    }
                }
            }
            references.push(Reference {
                handle,
                clip_seconds: speakers
                    .speakers
                    .iter()
                    .filter(|speaker| speaker.name == name)
                    .nth(handle - 1)
                    .and_then(|speaker| speaker.clip_seconds),
                depends,
                elsewhere,
            });
        }
        found.people.push(Enrolled {
            name: name.to_string(),
            references,
        });
    }

    Ok(found)
}

/// [`scan`], printed. Returns the scan so a caller can pick an exit status without re-deriving
/// it.
///
/// Shaped like [`crate::run_enroll`] -- `paths`, an `out` to write to, a report back -- for the
/// same reason: the terminal stays in the CLI crate, and every line this prints is decidable in
/// `cargo test`.
///
/// The scope comes first, because it is what every "names nothing" below it has to be read
/// against.
pub fn run_speakers(paths: &Paths, out: &mut dyn Write) -> Result<Scan> {
    let found = scan(paths)?;

    if found.people.is_empty() {
        write_nobody_enrolled(out, paths)?;
        return Ok(found);
    }

    let held: usize = found.people.iter().map(|p| p.references.len()).sum();
    writeln!(
        out,
        "{} person(s) enrolled, {held} reference(s) between them",
        found.people.len()
    )?;
    writeln!(
        out,
        "Read {} of {} transcribed session(s), of {} found in {}",
        found.sessions_read,
        found.sessions_transcribed,
        found.sessions_found,
        paths.sessions_dir().display()
    )?;
    for session in &found.unreadable {
        writeln!(
            out,
            "{}  could not be read: {}",
            session.session, session.why
        )?;
    }

    writeln!(out)?;
    for person in &found.people {
        writeln!(
            out,
            "{}  {} reference(s)",
            person.name,
            person.references.len()
        )?;
        for reference in &person.references {
            // "unknown length", not a blank: absent is a fact about rows stored before meethook
            // recorded it, and it is the fact that keeps such a row from ever being evicted.
            let built_from = match reference.clip_seconds {
                Some(seconds) => format!("from {seconds:.1} s"),
                None => "from a recording of unknown length".to_string(),
            };
            // "in any session read", not "names nothing": the clause that keeps the sentence
            // honest about the scope printed two lines above it.
            if reference.depends.is_empty() {
                writeln!(
                    out,
                    "  reference {}  {built_from}, names nothing in any session read",
                    reference.handle
                )?;
            } else {
                writeln!(
                    out,
                    "  reference {}  {built_from}, names {} voice(s):",
                    reference.handle,
                    reference.depends.len()
                )?;
                for change in &reference.depends {
                    write_change(out, change)?;
                }
            }
            // Only when there is something to say, and subordinated under the reference rather
            // than given equal weight: it is not what that reference is *naming*.
            if !reference.elsewhere.is_empty() {
                writeln!(
                    out,
                    "    also moves {} other voice(s):",
                    reference.elsewhere.len()
                )?;
                for change in &reference.elsewhere {
                    write_change(out, change)?;
                }
            }
        }
    }

    Ok(found)
}

/// One moved voice on one line: where it is, which voice it is, and both labels.
///
/// One writer for both lists, so the two cannot end up describing the same fact differently --
/// and visible to the crate for the same reason, so a voice reads identically in this listing and
/// in the removal a user reaches from it.
pub(crate) fn write_change(out: &mut dyn Write, change: &VoiceChange) -> Result<()> {
    writeln!(
        out,
        "      {}  {} reads {}, would read {}",
        change.session, change.voice, change.reads, change.would_read
    )?;
    Ok(())
}

/// The scan could not speak for every session, said once for every caller that has to say it.
///
/// `meethook speakers` fails with this sentence and the enrolment frame draws it, about the same
/// incompleteness in the same [`Scan`]. Two callers wording one fact differently is what makes a
/// user wonder whether they mean the same thing -- the reason `write_nobody_enrolled` exists in
/// this module, and the reason a caller that has a [`Scan::unreadable`] to report should reach for
/// this rather than format its own.
///
/// A `String` rather than a write, because the two callers put it in different places: one exits
/// with it, the other draws it as the last line of a pane.
pub fn incomplete(count: usize) -> String {
    format!("{count} session(s) could not be read, so this listing is incomplete")
}

/// Nobody is enrolled at all, said once for every command that has to say it.
///
/// A listing and a removal both reach this state, and two commands wording the same fact
/// differently is what makes a user wonder whether they mean the same thing.
pub(crate) fn write_nobody_enrolled(out: &mut dyn Write, paths: &Paths) -> Result<()> {
    writeln!(
        out,
        "Nobody is enrolled: {} holds no references -- meethook enroll names the voices in a \
         session",
        paths.speakers_json().display()
    )?;
    Ok(())
}

/// The derivation and the listing, over real session directories on a temporary disk.
///
/// The fixtures come from the crate's own test module rather than being copied here: the point
/// of the diff is that it reads the same files `enroll` does, and a second set of fixtures could
/// drift into describing a session neither command would ever see.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{
        assigned_in, axis, enrolled, files_under, heard_at_once, make_session,
        named_for_its_session, nearly, voice, with_embeddings,
    };

    /// The scan and its listing together, since every test below wants both: a report whose
    /// wording is half its value should not be assertable only through its fields.
    fn scanned(paths: &Paths) -> (Scan, String) {
        let mut out = Vec::new();
        let found = run_speakers(paths, &mut out).unwrap();
        (found, String::from_utf8(out).unwrap())
    }

    /// What one reference is naming, as `(session, voice, reads, would read)`, which is the
    /// line the listing prints.
    fn naming(reference: &Reference) -> Vec<(String, &str, &str, &str)> {
        reference
            .depends
            .iter()
            .map(|c| {
                (
                    c.session.to_string(),
                    c.voice.as_str(),
                    c.reads.as_str(),
                    c.would_read.as_str(),
                )
            })
            .collect()
    }

    fn person<'a>(found: &'a Scan, name: &str) -> &'a Enrolled {
        found
            .people
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("{name} is not in {:?}", found.people))
    }

    /// Acceptance criterion #1: a person holding several recordings is one entry with one
    /// numbered handle per recording, in file order, and the handles are what the listing
    /// prints.
    #[test]
    fn a_person_with_several_references_lists_each_one_separately() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Three recordings of Alice: the first is this session's cluster 0, and the other two
        // are voices no session on disk has.
        enrolled(
            &[
                ("Alice", voice(0)),
                ("Alice", axis(5, 8)),
                ("Alice", axis(6, 8)),
                ("Bob", voice(1)),
            ],
            &paths,
        );

        let (found, output) = scanned(&paths);

        let names: Vec<&str> = found.people.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["Alice", "Bob"], "{output}");
        let handles: Vec<usize> = person(&found, "Alice")
            .references
            .iter()
            .map(|r| r.handle)
            .collect();
        assert_eq!(handles, [1, 2, 3], "{output}");
        assert!(output.contains("Alice  3 reference(s)"), "{output}");
        assert!(output.contains("Bob  1 reference(s)"), "{output}");
        assert!(
            output.contains("2 person(s) enrolled, 4 reference(s) between them"),
            "{output}"
        );
        assert!(output.contains("  reference 3  "), "{output}");
    }

    /// Acceptance criterion #2: what a reference is naming is the voices in the sessions on
    /// disk, named by session id and by the label they read as, and one recording can be doing
    /// that in more than one meeting.
    #[test]
    fn a_reference_that_names_voices_in_two_sessions_names_both() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260810-101500");
        enrolled(&[("Alice", voice(0))], &paths);

        let (found, output) = scanned(&paths);

        let reference = &person(&found, "Alice").references[0];
        assert_eq!(
            naming(reference),
            [
                (
                    "20260809-052600".to_string(),
                    "Unknown 1",
                    "Alice",
                    "Unknown 1"
                ),
                (
                    "20260810-101500".to_string(),
                    "Unknown 1",
                    "Alice",
                    "Unknown 1"
                ),
            ],
            "{output}"
        );
        assert!(
            output.contains("reference 1  from a recording of unknown length, names 2 voice(s):"),
            "{output}"
        );
        assert!(
            output.contains("20260810-101500  Unknown 1 reads Alice, would read Unknown 1"),
            "{output}"
        );
        assert!(reference.elsewhere.is_empty(), "{output}");
    }

    /// Acceptance criterion #3: the reference somebody at the cap should drop is the one naming
    /// nothing, and it has to say so rather than simply printing no lines under itself -- which
    /// is indistinguishable from a listing that ran out of room.
    #[test]
    fn a_reference_naming_nothing_says_so() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // The second recording is orthogonal to every cluster in the session, so it matches
        // nothing and nothing reads a name because of it.
        enrolled(&[("Alice", voice(0)), ("Alice", axis(5, 8))], &paths);

        let (found, output) = scanned(&paths);

        let references = &person(&found, "Alice").references;
        assert_eq!(references[0].depends.len(), 1, "{output}");
        assert!(references[1].depends.is_empty(), "{output}");
        assert!(
            output.contains(
                "reference 2  from a recording of unknown length, names nothing in any session read"
            ),
            "{output}"
        );
    }

    /// The other half of the choice this listing exists to inform. What a reference is naming
    /// says what dropping it costs now; how long a recording it came from says how good it is,
    /// and is what `enroll` itself compares at the cap -- so both belong on the line, and a
    /// measured length must print as a number rather than falling back to the unknown wording.
    #[test]
    fn a_reference_says_how_long_a_recording_it_was_built_from() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", voice(0), 42.5);
        speakers.store_reference("Alice", axis(5, 8), 7.25);
        speakers.write(&paths).unwrap();

        let (found, output) = scanned(&paths);

        let references = &person(&found, "Alice").references;
        assert_eq!(references[0].clip_seconds, Some(42.5), "{output}");
        assert_eq!(references[1].clip_seconds, Some(7.25), "{output}");
        assert!(
            output.contains("reference 1  from 42.5 s, names"),
            "{output}"
        );
        assert!(
            output.contains("reference 2  from 7.2 s, names nothing in any session read"),
            "one decimal place, like every other duration this tool prints: {output}"
        );
    }

    /// Acceptance criterion #7, and the invariant that makes it free: a voice named by hand
    /// against one session reads that name because of `speaker_names.json`, not because of any
    /// reference -- so removing any row leaves it alone and it appears under nobody.
    #[test]
    fn a_voice_named_for_its_session_belongs_to_no_reference() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");
        // The assignment is the only thing naming that voice: nothing was stored for it.
        let assigned = assigned_in(&session, "20260809-052600");
        assert_eq!(assigned.names.len(), 1);
        assert_eq!(assigned.names[0].name, "Silas");
        // Somebody with a real reference too, so the scan has a reference to file voices under
        // at all and "under nobody" is a claim about a listing that has entries in it.
        enrolled(&[("Alice", voice(0))], &paths);

        let (found, output) = scanned(&paths);

        assert_eq!(
            naming(&person(&found, "Alice").references[0]),
            [(
                "20260809-052600".to_string(),
                "Unknown 1",
                "Alice",
                "Unknown 1"
            )],
            "{output}"
        );
        for person in &found.people {
            for reference in &person.references {
                for change in reference.depends.iter().chain(&reference.elsewhere) {
                    assert_ne!(
                        change.reads, "Silas",
                        "a hand-given name depends on no reference: {change:?}"
                    );
                }
            }
        }
    }

    /// The second direction a removal moves a label in: the voice does not revert to a number,
    /// it starts reading somebody else, because the second-nearest row belongs to them. A
    /// listing that modelled only reversion would understate what dropping this row does.
    #[test]
    fn a_voice_that_would_read_someone_else_says_so() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // Cluster 0 sits between two references: 10 degrees from Alice's and 20 from Bob's, so
        // Alice wins the argmax and Bob is the runner-up that both clear the cut.
        with_embeddings(&session, &[nearly(10.0), voice(3)]);
        enrolled(&[("Alice", nearly(0.0)), ("Bob", nearly(30.0))], &paths);

        let (found, output) = scanned(&paths);

        assert_eq!(
            naming(&person(&found, "Alice").references[0]),
            [("20260809-052600".to_string(), "Unknown 1", "Alice", "Bob")],
            "{output}"
        );
        assert!(
            output.contains("Unknown 1 reads Alice, would read Bob"),
            "{output}"
        );
        // The same removal is what puts Bob on that voice, which is a move Bob's own reference
        // is not responsible for -- so it belongs to nobody's `depends`.
        assert!(
            person(&found, "Bob").references[0].depends.is_empty(),
            "{output}"
        );
    }

    /// The third direction, and the one only the veto produces: this row is winning a name the
    /// heard-at-once veto then denies to a contender, so removing it makes another voice *gain*
    /// a name. Filed under `elsewhere`, because it is not what this reference is naming.
    #[test]
    fn a_reference_holding_a_name_off_another_voice_reports_that_too() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // Both voices claim Alice, cluster 0 the nearer of the two, and segmentation heard them
        // speaking at once -- so the veto awards Alice to cluster 0 and leaves cluster 1 with
        // its number.
        with_embeddings(&session, &[nearly(5.0), nearly(20.0)]);
        heard_at_once(&session, 0, 1);
        // Two of Alice's recordings, each nearest one of the two voices, so removing the first
        // hands the name to the voice the veto was holding it off.
        enrolled(&[("Alice", nearly(0.0)), ("Alice", nearly(25.0))], &paths);

        let (found, output) = scanned(&paths);

        let first = &person(&found, "Alice").references[0];
        assert_eq!(
            naming(first),
            [(
                "20260809-052600".to_string(),
                "Unknown 1",
                "Alice",
                "Unknown 1"
            )],
            "{output}"
        );
        let gained: Vec<(&str, &str, &str)> = first
            .elsewhere
            .iter()
            .map(|c| (c.voice.as_str(), c.reads.as_str(), c.would_read.as_str()))
            .collect();
        assert_eq!(
            gained,
            [("Unknown 2", "Unknown 2", "Alice")],
            "removing the awarded row lets the vetoed contender gain the name: {output}"
        );
        assert!(output.contains("also moves 1 other voice(s):"), "{output}");
    }

    /// Acceptance criterion #4: a session the scan has no opinion about is named, with the same
    /// remedy `enroll` gives for the same file, the counts say one fewer was read, and the
    /// listing still prints -- a report that gave up on the whole database because one session
    /// was stale would be worse than the stale session.
    #[test]
    fn a_session_that_cannot_be_read_is_named_with_its_reason() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let stale = make_session(&paths, "20260809-052600");
        make_session(&paths, "20260810-101500");
        // A clusters file from before first appearances were recorded: it parses as far as the
        // schema and then has no `first_spoke_seconds` to map "Unknown 2" back with.
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
        enrolled(&[("Alice", voice(0))], &paths);

        let (found, output) = scanned(&paths);

        assert_eq!(found.sessions_transcribed, 2, "{output}");
        assert_eq!(found.sessions_read, 1, "{output}");
        assert_eq!(found.unreadable.len(), 1, "{output}");
        assert_eq!(
            found.unreadable[0].session.to_string(),
            "20260809-052600",
            "{output}"
        );
        assert!(found.unreadable[0].why.contains("--force"), "{output}");
        assert!(
            output.contains("20260809-052600  could not be read:"),
            "{output}"
        );
        assert!(
            output.contains("Read 1 of 2 transcribed session(s)"),
            "{output}"
        );
        // The readable session is still reported on, and only it.
        assert_eq!(
            naming(&person(&found, "Alice").references[0]),
            [(
                "20260810-101500".to_string(),
                "Unknown 1",
                "Alice",
                "Unknown 1"
            )],
            "{output}"
        );
    }

    /// Acceptance criterion #4, the other half: the scope is what makes every "names nothing"
    /// below it readable, so the sessions the scan could not speak for are counted rather than
    /// quietly dropped out of the denominator.
    #[test]
    fn the_scope_line_counts_what_it_could_not_speak_for() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Recorded but never transcribed: no labelling exists to diff.
        let untranscribed = paths.session(&SessionId::parse("20260810-101500").unwrap());
        std::fs::create_dir_all(untranscribed.dir()).unwrap();
        std::fs::write(untranscribed.session_json(), b"{}").unwrap();
        // The recorder died mid-session: not even a session.json.
        let orphaned = paths.session(&SessionId::parse("20260811-090000").unwrap());
        std::fs::create_dir_all(orphaned.dir()).unwrap();
        enrolled(&[("Alice", voice(0))], &paths);

        let (found, output) = scanned(&paths);

        assert_eq!(found.sessions_found, 3, "{output}");
        assert_eq!(found.sessions_transcribed, 1, "{output}");
        assert_eq!(found.sessions_read, 1, "{output}");
        assert!(found.unreadable.is_empty(), "{output}");
        assert!(
            output.contains(&format!(
                "Read 1 of 1 transcribed session(s), of 3 found in {}",
                paths.sessions_dir().display()
            )),
            "{output}"
        );
    }

    /// Acceptance criterion #5: nobody enrolled is a sentence rather than an empty listing, and
    /// it is the same sentence whether the file is absent or present and empty -- which is the
    /// distinction `read_or_empty` already collapses and no user needs back.
    #[test]
    fn nobody_enrolled_is_reported_rather_than_an_empty_listing() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let (absent, absent_output) = scanned(&paths);
        enrolled(&[], &paths);
        let (empty, empty_output) = scanned(&paths);

        assert!(absent.people.is_empty(), "{absent_output}");
        assert!(empty.people.is_empty(), "{empty_output}");
        assert_eq!(absent_output, empty_output);
        assert!(
            absent_output.contains("Nobody is enrolled"),
            "{absent_output}"
        );
        assert!(
            absent_output.contains("meethook enroll"),
            "the line names the way out: {absent_output}"
        );
        // Short-circuited before discovery: there is no scope line, because there is no
        // question it would answer.
        assert!(
            !absent_output.contains("transcribed session(s)"),
            "{absent_output}"
        );
    }

    /// Acceptance criterion #1's other half: this command writes nothing, and "nothing" is
    /// byte-for-byte over every file under the root rather than over the ones it was expected
    /// to touch.
    #[test]
    fn a_listing_leaves_every_file_byte_identical() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0)), ("Bob", axis(5, 8))], &paths);
        let before = files_under(root.path());
        assert!(
            before.len() > 5,
            "the fixture should have written a database, a names file and a session: {before:?}"
        );

        let (found, output) = scanned(&paths);

        assert_eq!(found.people.len(), 2, "{output}");
        assert_eq!(files_under(root.path()), before);
        // Named so the assertion above is about the files that exist rather than about a path
        // that was never written.
        assert!(session.speaker_names_json().is_file());
    }
}
