//! Removing a stored reference, or a person, after saying what it costs.
//!
//! The write half of the pair whose read half is [`crate::references`]. Freeing a slot at the
//! reference cap, or deleting a reference built from audio that turned out to be somebody else,
//! used to mean opening `speakers.json` in a text editor; this is the command that replaces that.
//!
//! The removal itself is nearly free. A person is every row bearing their name, so dropping their
//! last row *is* removing the person, and [`EnrolledSpeakers`] already owns that invariant behind
//! [`EnrolledSpeakers::without`] and [`EnrolledSpeakers::without_person`]. What this module is
//! actually about is the consequence and the consent.
//!
//! # The consequence: three directions, not one
//!
//! Removing a reference does not only revert voices to their "Unknown N". Identification is an
//! argmax over rows that is then vetoed per name, so a removal can also make a voice read
//! *somebody else* -- the second-nearest row belongs to them -- and can let a voice the
//! heard-at-once veto was refusing **gain** the name the removed row was holding. All three fall
//! out of one before/after labelling diff, and all three are printed: reversion and "reads
//! somebody else" are told apart inside [`Removal::reverting`] by what the voice `would_read`,
//! and a gain is in [`Removal::elsewhere`].
//!
//! # The consent: consequences first, write second
//!
//! A reference cannot be regenerated -- the audio it was built from may be long deleted, which is
//! the same reasoning that makes `speakers.json` migrate an old version on read rather than refuse
//! it -- and the user cannot know what a removal costs until the tool has computed it. So
//! [`run_forget`] is **one function rather than a `preview` and an `execute`**: with two, "print
//! the consequences before writing" would be a convention a caller could get wrong; with one, the
//! write is physically after the print and unreachable without it. It is also what keeps the
//! labellings the diff produced in scope for the transcript pass, so nothing is derived twice.
//!
//! # Nothing is ever refused
//!
//! `enroll` refuses an answer that would cost a third party their name, because there the user's
//! intent was to name one voice and the cost is a surprise. Here the cost *is* the request: the
//! user has asked to remove something, has been shown everything it takes with it, and has said
//! `--yes`. A veto on this path would be the tool overruling an informed decision about its own
//! data. The asymmetry is deliberate rather than an oversight.
//!
//! # Session-only names are not touched, and that is not a gap
//!
//! Removing a person leaves any `speaker_names.json` row naming them alone, so a transcript can
//! still read that name in the session somebody typed it into. `transcribe --force` would still
//! produce that name there, because the session's own file supports it -- what a removal must not
//! leave behind is a transcript resting on a *database* claim that is gone. Removing a hand-given
//! name is a different act on a different file, and the way to do it is to re-answer the voice
//! (`meethook enroll --correct --all --voice`).

use std::collections::BTreeMap;
use std::io::Write;

use meethook_session::{
    EnrolledSpeakers, Paths, SessionId, SessionMetadata, SessionPaths, Transcript,
    TranscriptContext, TranscriptTemplate,
};
use meethook_transcribe::Attribution;

use crate::references::{
    Unreadable, VoiceChange, label_sessions, write_change, write_nobody_enrolled,
};
use crate::{Result, relabel};

/// What one removal names: a person, and optionally which one of their recordings.
pub struct Target {
    /// Exactly as the user typed it. Names match exactly everywhere else in the tool -- `alice`
    /// and `Alice` are two people, as [`EnrolledSpeakers::store_reference`] documents -- and a
    /// removal must not be the one place that guesses.
    pub name: String,

    /// The 1-based handle `meethook speakers` printed. `None` removes the person outright.
    pub reference: Option<usize>,
}

/// Whether this run may write. [`Confirm::Preview`] is the default and the safe one.
pub enum Confirm {
    Preview,
    Confirmed,
}

/// How a removal ended, so a caller can pick an exit status without re-deriving anything.
pub enum Forgotten {
    /// Nothing matched: that name, or that reference of it, is not in the database. The lines
    /// naming the path and what *is* stored have been printed, and no session was read.
    NotStored,

    /// What the removal would do, printed. Nothing was written.
    Previewed(Removal),

    /// Written: `speakers.json`, then the transcripts of every session whose labelling moved.
    Removed(Removal),
}

/// Everything a removal does, derived before a byte is written.
pub struct Removal {
    pub name: String,

    /// Echoed back from the [`Target`], so a stale handle is visible in the output that acted on
    /// it: a user working from a listing taken before an earlier removal sees which row this
    /// number now addresses, in the voices printed underneath it.
    pub reference: Option<usize>,

    /// Rows going: 1, or every one this name holds.
    pub removed: usize,

    /// References this name keeps. Zero is "no longer enrolled", and is what the wording keys on
    /// -- so `forget Alice --reference 1` on a one-reference Alice reads as removing the person,
    /// which is what it is doing.
    pub remaining: usize,

    /// Voices that stop reading this person: reverting to their "Unknown N", or reading somebody
    /// else because the second-nearest row belongs to them.
    pub reverting: Vec<VoiceChange>,

    /// Voices whose label moves some other way -- one that *gains* a name this row was holding
    /// off through the heard-at-once veto, or one that loses a third party's name because the
    /// removal re-runs identification.
    pub elsewhere: Vec<VoiceChange>,

    /// Sessions whose `transcript.json` could not be read, so their transcripts will keep
    /// claiming a name the database no longer supports. Named *before* the write, not after: the
    /// user is asked to accept this, rather than told about it once it is too late.
    pub unwritable: Vec<Unreadable>,

    /// The same scope statement `meethook speakers` makes, for the same reason: every "no voice
    /// changes" has to be read against how much was looked at.
    pub sessions_found: usize,
    pub sessions_transcribed: usize,
    pub sessions_read: usize,

    /// Transcribed sessions the labelling has no opinion about, each with its reason and remedy.
    pub unreadable: Vec<Unreadable>,
}

impl Removal {
    /// Sessions whose labelling moves, in id order: exactly the transcripts a removal rewrites.
    ///
    /// Derived from the two change lists rather than stored beside them, so the count printed and
    /// the files written cannot disagree.
    pub fn sessions_changed(&self) -> Vec<&SessionId> {
        let mut ids: Vec<&SessionId> = self
            .reverting
            .iter()
            .chain(&self.elsewhere)
            .map(|change| &change.session)
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }
}

/// One changed session, carrying everything the write needs and nothing it would have to re-derive.
///
/// The transcript is read during the *preview*, which is what makes the preview exact: the user is
/// told which transcripts will be brought in line and which cannot be, before consenting.
struct Pending {
    paths: SessionPaths,
    session: SessionId,
    /// The post-removal labelling this session was diffed against -- handed straight to
    /// [`relabel`] rather than derived a second time, so what was previewed is what is written.
    labels: BTreeMap<u32, Attribution>,
    transcript: Transcript,
    /// What re-rendering the markdown needs beyond the turns. Read during the preview beside
    /// the transcript, and for the same reason: a session whose `session.json` cannot be read
    /// is named before the database moves rather than after.
    metadata: SessionMetadata,
}

/// Derives what a removal would do, prints it, and -- only with [`Confirm::Confirmed`] -- performs
/// it.
///
/// # Order of operations, and why it is not a caller's business
///
/// 1. Read the database. An unreadable or unsupported-version file is an `Err`, as in every other
///    reader; a v1 file is migrated on read, so a removal works on one without a version complaint.
/// 2. **Fail fast, before any session is read.** A name that is not stored, or a handle that name
///    does not hold, prints the path and what *is* stored and returns [`Forgotten::NotStored`].
///    Labelling a hundred sessions to answer a typo is work for nothing, and the user's next move
///    is to re-type the name rather than to read a dependency report.
/// 3. Label every transcribed session under the root, then label each again against the database
///    with the chosen rows absent, and partition the differences.
/// 4. Read the `transcript.json` of every changed session, so an unreadable one is named in the
///    preview rather than discovered after the database has already moved.
/// 5. Print everything.
/// 6. Write, or not. Under [`Confirm::Preview`] nothing on any path above has written anything --
///    `without` clones, `labels` derives, `Transcript::read` reads -- so "writes nothing" is
///    structural rather than a branch that has to be got right.
///
/// # The write order, and what an interrupt between the writes costs
///
/// `speakers.json` first, then the transcripts, which is `enroll_session`'s order and for its
/// reason: the database is what the next labelling reads, so an interrupt after it leaves
/// transcripts that a later `meethook enroll <id>` simply brings in line, by this crate's
/// documented invariant that a transcript on disk is what `transcribe --force` would now produce.
/// The reverse order would leave transcripts no on-disk state justifies. Neither order is atomic
/// across N files and this does not pretend otherwise: it makes the surviving state the one an
/// existing command recovers.
pub fn run_forget(
    paths: &Paths,
    target: &Target,
    confirm: Confirm,
    template: &TranscriptTemplate,
    out: &mut dyn Write,
) -> Result<Forgotten> {
    let speakers = EnrolledSpeakers::read_or_empty(paths)?;
    let held = speakers.references(&target.name);

    let rest = match target.reference {
        Some(handle) => speakers.without(&target.name, handle),
        None => speakers.without_person(&target.name),
    };
    // The same `Option` either way, so the two acts are one branch apart rather than two paths.
    let Some(rest) = rest else {
        write_not_stored(out, paths, &speakers, target, held)?;
        return Ok(Forgotten::NotStored);
    };

    let labelling = label_sessions(paths, &speakers)?;
    let mut removal = Removal {
        name: target.name.clone(),
        reference: target.reference,
        removed: held - rest.references(&target.name),
        remaining: rest.references(&target.name),
        reverting: Vec::new(),
        elsewhere: Vec::new(),
        unwritable: Vec::new(),
        sessions_found: labelling.found,
        sessions_transcribed: labelling.transcribed,
        sessions_read: labelling.sessions.len(),
        unreadable: labelling.unreadable,
    };

    let mut pending: Vec<Pending> = Vec::new();
    for session in &labelling.sessions {
        let after = session.labels(&rest);
        let changes = session.moved(&after);
        if changes.is_empty() {
            continue;
        }
        for change in changes {
            // A voice that was reading this person is what the removal takes away; anything else
            // moving is a consequence of it. The identical rule `scan` partitions on.
            if change.reads == target.name {
                removal.reverting.push(change);
            } else {
                removal.elsewhere.push(change);
            }
        }
        // Read now, before anything is written, so an unreadable transcript is named in the
        // preview rather than discovered once the database has already moved. The remedy is the
        // one `enroll_session` gives for this same file, so two commands do not hand out
        // different instructions for one problem.
        let session_paths = paths.session(&session.session);
        // Both files the rewrite needs, read now and on the same terms, before anything is
        // written: an unreadable one is named in the preview rather than discovered once the
        // database has already moved. Each carries the remedy `enroll_session` gives for that
        // same file, so two commands do not hand out different instructions for one problem.
        let readable = Transcript::read(&session_paths.transcript_json())
            .map_err(|e| format!("{e} -- re-transcribe this session with --force"))
            .and_then(|transcript| {
                SessionMetadata::read(&session_paths.session_json())
                    .map(|metadata| (transcript, metadata))
                    .map_err(|e| format!("{e} -- fix that file"))
            });
        match readable {
            Ok((transcript, metadata)) => pending.push(Pending {
                paths: session_paths,
                session: session.session.clone(),
                labels: after,
                transcript,
                metadata,
            }),
            Err(why) => removal.unwritable.push(Unreadable {
                session: session.session.clone(),
                why,
            }),
        }
    }

    write_removal(out, paths, &removal, &confirm)?;

    match confirm {
        Confirm::Preview => Ok(Forgotten::Previewed(removal)),
        Confirm::Confirmed => {
            rest.write(paths)?;
            for entry in &mut pending {
                if relabel(&mut entry.transcript, &entry.labels) {
                    entry.transcript.write(
                        &entry.paths,
                        template,
                        &TranscriptContext::now(&entry.metadata),
                    )?;
                    writeln!(out, "{}  transcript brought in line", entry.session)?;
                } else {
                    // The labelling moved but no turn was attributed to the voice that moved --
                    // a cluster with nothing said under it. Said out loud, because a session
                    // counted above and then silent here would read as a failed write.
                    writeln!(out, "{}  transcript already agreed", entry.session)?;
                }
            }
            Ok(Forgotten::Removed(removal))
        }
    }
}

/// The removal that is not there: the path, and what the database actually holds.
///
/// This is what makes the usual mistake visible. Names match exactly, so a case or spelling slip
/// is a different person, and seeing the enrolled names beside the one that was typed is the whole
/// fix. It deliberately does *not* print `meethook speakers`'s per-reference listing: that needs
/// the scan this path has just declined to run, and it answers a question ("which one should I
/// drop") the user has not reached yet.
fn write_not_stored(
    out: &mut dyn Write,
    paths: &Paths,
    speakers: &EnrolledSpeakers,
    target: &Target,
    held: usize,
) -> Result<()> {
    if speakers.speakers.is_empty() {
        return write_nobody_enrolled(out, paths);
    }
    match target.reference {
        // The name is there; the handle is not. Say how many there are, since that is the whole
        // correction.
        Some(handle) if held > 0 => writeln!(
            out,
            "{} holds no reference {handle}: {} has {held} reference(s) in {}",
            target.name,
            target.name,
            paths.speakers_json().display()
        )?,
        _ => writeln!(
            out,
            "Nobody called {} is enrolled in {}",
            target.name,
            paths.speakers_json().display()
        )?,
    }
    writeln!(out, "Enrolled:")?;
    for name in speakers.enrolled_names() {
        writeln!(out, "  {name}  {} reference(s)", speakers.references(name))?;
    }
    writeln!(
        out,
        "meethook speakers says what each of those recordings is naming"
    )?;
    Ok(())
}

/// What is going, then the scope, then the consequences, then the transcripts, then the one line
/// that says whether anything happened.
///
/// Scope before consequences for the reason `run_speakers` prints it there: every "no voice
/// changes" below it has to be read against it. And "nothing" is always a sentence rather than
/// blank space -- a report whose value is its completeness cannot express nothing by omission.
fn write_removal(
    out: &mut dyn Write,
    paths: &Paths,
    removal: &Removal,
    confirm: &Confirm,
) -> Result<()> {
    let name = &removal.name;
    let held = removal.removed + removal.remaining;
    writeln!(
        out,
        "{name} holds {held} reference(s) in {}",
        paths.speakers_json().display()
    )?;
    // The two wordings AC #6 asks for, and they key on the *outcome* rather than on which flag
    // was passed: what is going echoes the handle, what is left says whether a person survives.
    let going = match removal.reference {
        Some(handle) => format!("Removing reference {handle}"),
        None => format!("Removing all {} reference(s)", removal.removed),
    };
    if removal.remaining == 0 {
        writeln!(
            out,
            "{going} leaves {name} no longer enrolled: nothing will name that voice in a future \
             meeting"
        )?;
    } else {
        writeln!(out, "{going} leaves {name} with {}", removal.remaining)?;
    }

    writeln!(
        out,
        "Read {} of {} transcribed session(s), of {} found in {}",
        removal.sessions_read,
        removal.sessions_transcribed,
        removal.sessions_found,
        paths.sessions_dir().display()
    )?;
    for session in &removal.unreadable {
        writeln!(
            out,
            "{}  could not be read: {}",
            session.session, session.why
        )?;
    }

    writeln!(out)?;
    if removal.reverting.is_empty() {
        writeln!(out, "No voice stops reading {name} in any session read")?;
    } else {
        writeln!(
            out,
            "{} voice(s) stop reading {name}:",
            removal.reverting.len()
        )?;
        for change in &removal.reverting {
            write_change(out, change)?;
        }
    }
    if !removal.elsewhere.is_empty() {
        writeln!(out, "{} other voice(s) move:", removal.elsewhere.len())?;
        for change in &removal.elsewhere {
            write_change(out, change)?;
        }
    }

    let changed = removal.sessions_changed();
    if changed.is_empty() {
        writeln!(out, "No transcript to bring in line")?;
    } else {
        let ids: Vec<String> = changed.iter().map(|id| id.to_string()).collect();
        writeln!(
            out,
            "{} transcript(s) to bring in line: {}",
            changed.len(),
            ids.join(", ")
        )?;
    }
    // The one respect in which a removal leaves the tool inconsistent, on its own line and above
    // the confirmation, because the user is being asked to accept it rather than told afterwards.
    if !removal.unwritable.is_empty() {
        writeln!(
            out,
            "{} transcript(s) cannot be brought in line and will keep the name they have:",
            removal.unwritable.len()
        )?;
        for session in &removal.unwritable {
            writeln!(out, "      {}  {}", session.session, session.why)?;
        }
    }

    writeln!(out)?;
    match confirm {
        // The irreversibility clause sits here, where somebody who has not yet typed `--yes` is
        // looking, rather than in the header where it would be read once and then skipped.
        Confirm::Preview => {
            let reference = match removal.reference {
                Some(handle) => format!(" --reference {handle}"),
                None => String::new(),
            };
            writeln!(
                out,
                "Nothing was written: meethook forget {name}{reference} --yes removes it. A \
                 removed reference cannot be rebuilt -- the audio it was made from is not \
                 consulted and may be long deleted."
            )?;
        }
        Confirm::Confirmed if removal.remaining == 0 => writeln!(
            out,
            "Removed {} reference(s): {name} is no longer enrolled",
            removal.removed
        )?,
        Confirm::Confirmed => writeln!(
            out,
            "Removed {} reference(s): {name} keeps {}",
            removal.removed, removal.remaining
        )?,
    }
    Ok(())
}

/// The preview, the write, and the wording, over real session directories on a temporary disk.
///
/// The fixtures come from the crate's own test module rather than being copied here, for the reason
/// `references::tests` gives: a removal has to be shown acting on the same session directories
/// `enroll` and `speakers` read, and a second set of fixtures could drift into describing a session
/// neither command would ever produce.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::references::scan;
    use crate::tests::{
        axis, enrolled, files_under, heard_at_once, make_session, named_for_its_session, nearly,
        said, session_metadata, transcript_of, voice, with_embeddings, write_transcript,
    };

    /// One removal and everything it printed, since a report whose wording is half its value
    /// should not be assertable only through its fields.
    fn forgetting(
        paths: &Paths,
        name: &str,
        reference: Option<usize>,
        confirm: Confirm,
    ) -> (Forgotten, String) {
        let target = Target {
            name: name.to_string(),
            reference,
        };
        let mut out = Vec::new();
        let forgotten = run_forget(
            paths,
            &target,
            confirm,
            // Resolved from the root, exactly as the CLI does.
            &TranscriptTemplate::resolve(paths, None).unwrap(),
            &mut out,
        )
        .unwrap();
        (forgotten, String::from_utf8(out).unwrap())
    }

    /// The [`Removal`] out of a preview or a write, whichever this was.
    fn removal<'a>(forgotten: &'a Forgotten, output: &str) -> &'a Removal {
        match forgotten {
            Forgotten::Previewed(removal) | Forgotten::Removed(removal) => removal,
            Forgotten::NotStored => panic!("expected a removal, got NotStored: {output}"),
        }
    }

    /// One change as `(session, voice, reads, would read)`, which is the line that is printed.
    fn moving(changes: &[VoiceChange]) -> Vec<(String, &str, &str, &str)> {
        changes
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

    /// Brings every session's transcript in line with the database as it now stands -- which is
    /// the state a previous `enroll` run leaves, and the state this crate's invariant says every
    /// transcript on disk is in.
    ///
    /// Every test that asserts a removal *changes* a transcript needs the file to start out
    /// claiming the name, and `make_session` writes it as `transcribe` would against an empty
    /// database. Reached through the same [`label_sessions`] and [`relabel`] the tool uses, so the
    /// fixture cannot set up a state the tool would never produce.
    fn transcripts_in_line(paths: &Paths) {
        let speakers = EnrolledSpeakers::read_or_empty(paths).unwrap();
        for session in &label_sessions(paths, &speakers).unwrap().sessions {
            let session_paths = paths.session(&session.session);
            let mut transcript = Transcript::read(&session_paths.transcript_json()).unwrap();
            if relabel(&mut transcript, &session.baseline) {
                write_transcript(
                    &transcript,
                    paths,
                    &session_paths,
                    &session_metadata(&session.session),
                );
            }
        }
    }

    /// Who is stored, as `(name, embedding)` in file order.
    fn stored(paths: &Paths) -> Vec<(String, Vec<f32>)> {
        EnrolledSpeakers::read_or_empty(paths)
            .unwrap()
            .speakers
            .into_iter()
            .map(|s| (s.name, s.embedding))
            .collect()
    }

    /// Acceptance criteria #1 and #6: thinning drops the row the handle addresses and nothing
    /// else, and it reads as thinning -- "keeps 2" -- rather than as removing a person.
    #[test]
    fn thinning_a_reference_set_keeps_the_others_and_says_how_many_remain() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(
            &[
                ("Alice", voice(0)),
                ("Alice", axis(5, 8)),
                ("Alice", axis(6, 8)),
            ],
            &paths,
        );

        let (forgotten, output) = forgetting(&paths, "Alice", Some(2), Confirm::Confirmed);

        let removal = removal(&forgotten, &output);
        assert!(matches!(forgotten, Forgotten::Removed(_)), "{output}");
        assert_eq!((removal.removed, removal.remaining), (1, 2), "{output}");
        assert_eq!(
            stored(&paths),
            [
                ("Alice".to_string(), voice(0)),
                ("Alice".to_string(), axis(6, 8)),
            ],
            "the survivors keep their file order: {output}"
        );
        assert!(output.contains("Alice holds 3 reference(s) in"), "{output}");
        assert!(
            output.contains("Removing reference 2 leaves Alice with 2"),
            "{output}"
        );
        assert!(
            output.contains("Removed 1 reference(s): Alice keeps 2"),
            "{output}"
        );
        assert!(
            !output.contains("no longer enrolled"),
            "Alice is still enrolled, so nothing may say otherwise: {output}"
        );
    }

    /// Acceptance criteria #1, #4 and #6: a person is every row bearing their name, so removing
    /// the last one removes the person -- and both transcript files stop claiming a name the
    /// database no longer supports.
    #[test]
    fn removing_the_last_reference_removes_the_person() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);
        transcripts_in_line(&paths);
        assert_eq!(
            said(&transcript_of(&session))[0].0,
            "Alice",
            "the fixture has to start out claiming the name"
        );
        assert!(
            std::fs::read_to_string(session.transcript_md())
                .unwrap()
                .contains("Alice:")
        );

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Confirmed);

        let removal = removal(&forgotten, &output);
        assert_eq!((removal.removed, removal.remaining), (1, 0), "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 0, "{output}");
        assert!(speakers.enrolled_names().is_empty(), "{output}");
        assert!(
            output.contains("Removing all 1 reference(s) leaves Alice no longer enrolled"),
            "{output}"
        );
        assert!(
            output.contains("Removed 1 reference(s): Alice is no longer enrolled"),
            "{output}"
        );
        // Both files, because a reader of a meeting reads the markdown: a transcript.md still
        // saying Alice would be the same lie in the file people actually open.
        let turns = transcript_of(&session);
        assert_eq!(
            said(&turns).iter().map(|t| t.0).collect::<Vec<&str>>(),
            ["Unknown 1", "You", "Unknown 2", "Unknown 1"],
            "{output}"
        );
        let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
        assert!(!markdown.contains("Alice"), "{markdown}");
        assert!(markdown.contains("Unknown 1:"), "{markdown}");
        assert!(
            output.contains("20260809-052600  transcript brought in line"),
            "{output}"
        );
    }

    /// Acceptance criterion #2, and the fact a plan here can get wrong: **a person is not the
    /// union of their references.** Two of Alice's rows both within the cut of one voice means
    /// dropping either alone leaves the other naming it -- so every per-reference diff is empty
    /// while removing *Alice* reverts that voice. Aggregating what `speakers` prints would have
    /// reported no cost at all.
    #[test]
    fn removing_a_person_costs_more_than_removing_each_of_their_references_alone() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // Cluster 0 sits between two recordings of Alice, 10 degrees from one and 10 from the
        // other, so both clear the cut on their own. Cluster 1 is kept clear of both.
        with_embeddings(&session, &[nearly(10.0), voice(3)]);
        enrolled(&[("Alice", nearly(0.0)), ("Alice", nearly(20.0))], &paths);

        let found = scan(&paths).unwrap();
        for reference in &found.people[0].references {
            assert!(
                reference.depends.is_empty(),
                "each reference alone names nothing, which is the trap: {:?}",
                reference
            );
        }

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Preview);

        let removal = removal(&forgotten, &output);
        assert_eq!(removal.removed, 2, "{output}");
        assert_eq!(
            moving(&removal.reverting),
            [(
                "20260809-052600".to_string(),
                "Unknown 1",
                "Alice",
                "Unknown 1"
            )],
            "removing the person is a different counterfactual from removing each row: {output}"
        );
        assert!(
            output.contains("1 voice(s) stop reading Alice:"),
            "{output}"
        );
    }

    /// Acceptance criteria #2 and #4, second direction: the voice does not revert to a number, it
    /// starts reading somebody else, because the second-nearest row belongs to them. The preview
    /// says so and the transcript afterwards agrees with the preview -- which is what sharing one
    /// labelling between the two buys.
    #[test]
    fn a_voice_that_would_read_someone_else_is_previewed_and_then_reads_them() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(10.0), voice(3)]);
        enrolled(&[("Alice", nearly(0.0)), ("Bob", nearly(30.0))], &paths);
        transcripts_in_line(&paths);
        assert_eq!(said(&transcript_of(&session))[0].0, "Alice");

        let (previewed, preview) = forgetting(&paths, "Alice", None, Confirm::Preview);

        assert_eq!(
            moving(&removal(&previewed, &preview).reverting),
            [("20260809-052600".to_string(), "Unknown 1", "Alice", "Bob")],
            "{preview}"
        );
        assert!(
            preview.contains("Unknown 1 reads Alice, would read Bob"),
            "{preview}"
        );

        let (_, output) = forgetting(&paths, "Alice", None, Confirm::Confirmed);

        assert_eq!(
            said(&transcript_of(&session))[0].0,
            "Bob",
            "the write has to land where the preview said it would: {output}"
        );
        assert!(
            !std::fs::read_to_string(session.transcript_md())
                .unwrap()
                .contains("Alice"),
            "{output}"
        );
    }

    /// Acceptance criterion #7: a name somebody typed against one session rests on
    /// `speaker_names.json`, not on any reference, so removing a person cannot touch it -- and it
    /// appears in neither change list, because no database change can move a hand-given name.
    #[test]
    fn a_voice_named_for_its_session_keeps_its_name() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);
        transcripts_in_line(&paths);

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Confirmed);

        let removal = removal(&forgotten, &output);
        for change in removal.reverting.iter().chain(&removal.elsewhere) {
            assert_ne!(
                change.reads, "Silas",
                "a hand-given name depends on no reference: {change:?}"
            );
            assert_ne!(change.would_read, "Silas", "{change:?}");
        }
        let names = crate::tests::assigned_in(&session, "20260809-052600");
        assert_eq!(names.names.len(), 1, "{output}");
        assert_eq!(names.names[0].name, "Silas", "{output}");
        assert!(
            said(&transcript_of(&session))
                .iter()
                .any(|t| t.0 == "Silas"),
            "the session's own file still supports that name: {output}"
        );
    }

    /// Acceptance criteria #2 and #4, third direction, and the only one the veto produces: this
    /// row was winning a name the heard-at-once veto then denied to a contender, so removing it
    /// makes another voice **gain** the name. Filed under the other movers, because it is not what
    /// this reference was naming.
    #[test]
    fn a_voice_the_veto_was_holding_a_name_off_gains_it_and_is_reported() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // Both voices claim Alice, cluster 0 the nearer, and segmentation heard them at once --
        // so the veto awards Alice to cluster 0 and leaves cluster 1 with its number.
        with_embeddings(&session, &[nearly(5.0), nearly(20.0)]);
        heard_at_once(&session, 0, 1);
        enrolled(&[("Alice", nearly(0.0)), ("Alice", nearly(25.0))], &paths);
        transcripts_in_line(&paths);
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|t| t.0)
                .collect::<Vec<&str>>(),
            ["Alice", "You", "Unknown 2", "Alice"]
        );

        let (forgotten, output) = forgetting(&paths, "Alice", Some(1), Confirm::Confirmed);

        let removal = removal(&forgotten, &output);
        assert_eq!(
            moving(&removal.reverting),
            [(
                "20260809-052600".to_string(),
                "Unknown 1",
                "Alice",
                "Unknown 1"
            )],
            "{output}"
        );
        assert_eq!(
            moving(&removal.elsewhere),
            [(
                "20260809-052600".to_string(),
                "Unknown 2",
                "Unknown 2",
                "Alice"
            )],
            "the vetoed contender gains the name the removed row was holding off: {output}"
        );
        assert!(output.contains("1 other voice(s) move:"), "{output}");
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|t| t.0)
                .collect::<Vec<&str>>(),
            ["Unknown 1", "You", "Alice", "Unknown 1"],
            "the transcript reads the gain too: {output}"
        );
    }

    /// Acceptance criterion #3: a mistyped name is the usual mistake, since names match exactly,
    /// and the fix is the enrolled names on the screen beside the one that was typed. Nothing is
    /// written and -- deliberately -- no session is even read.
    #[test]
    fn a_removal_naming_a_person_who_is_not_stored_writes_nothing_and_says_what_is() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(
            &[
                ("Alice", voice(0)),
                ("Alice", axis(5, 8)),
                ("Alice", axis(6, 8)),
                ("Bob", voice(1)),
            ],
            &paths,
        );
        let before = files_under(root.path());

        let (forgotten, output) = forgetting(&paths, "alice", None, Confirm::Confirmed);

        assert!(matches!(forgotten, Forgotten::NotStored), "{output}");
        assert!(
            output.contains(&format!(
                "Nobody called alice is enrolled in {}",
                paths.speakers_json().display()
            )),
            "{output}"
        );
        assert!(output.contains("Alice  3 reference(s)"), "{output}");
        assert!(output.contains("Bob  1 reference(s)"), "{output}");
        assert!(output.contains("meethook speakers"), "{output}");
        // The scan is declined rather than run: labelling a hundred sessions to answer a typo is
        // work for nothing, and its absence is what this line proves.
        assert!(
            !output.contains("transcribed session(s)"),
            "no session should have been read: {output}"
        );
        assert_eq!(
            files_under(root.path()),
            before,
            "a failed removal writes nothing, even under --yes"
        );
    }

    /// Acceptance criterion #3, the other handle: the name is there and the number is not, so the
    /// correction is how many there actually are.
    #[test]
    fn a_removal_naming_a_reference_that_is_not_held_says_how_many_there_are() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(
            &[
                ("Alice", voice(0)),
                ("Alice", axis(5, 8)),
                ("Alice", axis(6, 8)),
            ],
            &paths,
        );

        let (forgotten, output) = forgetting(&paths, "Alice", Some(7), Confirm::Confirmed);

        assert!(matches!(forgotten, Forgotten::NotStored), "{output}");
        assert!(
            output.contains(&format!(
                "Alice holds no reference 7: Alice has 3 reference(s) in {}",
                paths.speakers_json().display()
            )),
            "{output}"
        );
        assert_eq!(
            EnrolledSpeakers::read_or_empty(&paths)
                .unwrap()
                .speakers
                .len(),
            3,
            "{output}"
        );
    }

    /// Acceptance criterion #3, the empty case: a removal against a database with nobody in it is
    /// a failure rather than a quiet success, and it is the same sentence `meethook speakers`
    /// prints -- two commands wording one fact differently is what makes a user wonder whether
    /// they mean the same thing.
    #[test]
    fn nobody_enrolled_is_a_failed_removal_rather_than_a_quiet_one() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Confirmed);

        assert!(matches!(forgotten, Forgotten::NotStored), "{output}");
        let mut listing = Vec::new();
        crate::run_speakers(&paths, &mut listing).unwrap();
        assert_eq!(output, String::from_utf8(listing).unwrap());
        assert!(output.contains("Nobody is enrolled"), "{output}");
    }

    /// Acceptance criterion #2's "writes nothing until the user has confirmed", and "nothing" is
    /// byte-for-byte over every file under the root rather than over the ones a preview was
    /// expected to leave alone.
    #[test]
    fn nothing_is_written_without_confirmation() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);
        transcripts_in_line(&paths);
        let before = files_under(root.path());
        assert!(before.len() > 4, "{before:?}");

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Preview);

        assert!(matches!(forgotten, Forgotten::Previewed(_)), "{output}");
        assert_eq!(
            removal(&forgotten, &output).reverting.len(),
            1,
            "the preview has to have found the change it declined to make: {output}"
        );
        assert_eq!(files_under(root.path()), before);
        // The irreversibility clause sits on the confirmation line, where somebody who has not
        // yet typed --yes is looking.
        assert!(
            output.contains("Nothing was written: meethook forget Alice --yes removes it."),
            "{output}"
        );
        assert!(
            output.contains("cannot be rebuilt"),
            "the line has to say a reference cannot be got back: {output}"
        );
        assert!(
            !output.contains("Removed"),
            "a preview must not claim a removal happened: {output}"
        );
    }

    /// Acceptance criteria #2 and #4: "nothing changes" is a sentence rather than blank space --
    /// a report whose value is its completeness cannot express nothing by omission -- and a
    /// removal that changes no labelling touches no transcript.
    #[test]
    fn a_removal_that_changes_nothing_says_so_and_touches_no_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // The second recording is orthogonal to every cluster in the session, so it names nothing
        // and removing it moves no label at all.
        enrolled(&[("Alice", voice(0)), ("Alice", axis(5, 8))], &paths);
        transcripts_in_line(&paths);
        let before = files_under(root.path());

        let (previewed, preview) = forgetting(&paths, "Alice", Some(2), Confirm::Preview);

        let removal = removal(&previewed, &preview);
        assert!(removal.reverting.is_empty(), "{preview}");
        assert!(removal.elsewhere.is_empty(), "{preview}");
        assert!(removal.sessions_changed().is_empty(), "{preview}");
        assert!(
            preview.contains("No voice stops reading Alice in any session read"),
            "{preview}"
        );
        assert!(
            preview.contains("No transcript to bring in line"),
            "{preview}"
        );

        let (_, output) = forgetting(&paths, "Alice", Some(2), Confirm::Confirmed);

        let after = files_under(root.path());
        let differing: Vec<&std::path::PathBuf> = after
            .iter()
            .zip(&before)
            .filter(|(now, then)| now != then)
            .map(|(now, _)| &now.0)
            .collect();
        assert_eq!(
            differing,
            [&paths.speakers_json()],
            "only the database may change: {output}"
        );
        assert_eq!(said(&transcript_of(&session))[0].0, "Alice", "{output}");
    }

    /// Acceptance criterion #2: a session the labelling has no opinion about is named with the
    /// same remedy `enroll` gives for the same file, the scope says one fewer was read, and the
    /// rest of the report stands -- giving up on the whole removal because one session was
    /// transcribed by an older build would be worse than the stale session.
    #[test]
    fn a_session_that_cannot_be_read_is_named_and_the_removal_still_says_what_it_can() {
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

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Preview);

        let removal = removal(&forgotten, &output);
        assert_eq!(removal.sessions_transcribed, 2, "{output}");
        assert_eq!(removal.sessions_read, 1, "{output}");
        assert_eq!(removal.unreadable.len(), 1, "{output}");
        assert!(removal.unreadable[0].why.contains("--force"), "{output}");
        assert!(
            output.contains("20260809-052600  could not be read:"),
            "{output}"
        );
        assert!(
            output.contains("Read 1 of 2 transcribed session(s)"),
            "{output}"
        );
        // The session it could read is still reported on, and only it.
        assert_eq!(
            moving(&removal.reverting),
            [(
                "20260810-101500".to_string(),
                "Unknown 1",
                "Alice",
                "Unknown 1"
            )],
            "{output}"
        );
    }

    /// Acceptance criterion #4's honest edge: a transcript that cannot be read is named in the
    /// **preview**, so the user accepts it rather than discovering it after the database has
    /// already moved -- and the removal still happens, leaving that one file untouched.
    #[test]
    fn a_transcript_that_cannot_be_read_is_named_before_the_write_rather_than_after() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);
        transcripts_in_line(&paths);
        std::fs::write(session.transcript_json(), b"{ this is not a transcript").unwrap();
        let corrupt = std::fs::read(session.transcript_json()).unwrap();

        let (previewed, preview) = forgetting(&paths, "Alice", None, Confirm::Preview);

        let removal = removal(&previewed, &preview);
        assert_eq!(removal.reverting.len(), 1, "{preview}");
        assert_eq!(removal.unwritable.len(), 1, "{preview}");
        assert_eq!(
            removal.unwritable[0].session.to_string(),
            "20260809-052600",
            "{preview}"
        );
        assert!(
            preview.contains(
                "1 transcript(s) cannot be brought in line and will keep the name they have:"
            ),
            "{preview}"
        );
        assert!(
            preview.contains("re-transcribe this session with --force"),
            "{preview}"
        );

        let (forgotten, output) = forgetting(&paths, "Alice", None, Confirm::Confirmed);

        assert!(matches!(forgotten, Forgotten::Removed(_)), "{output}");
        assert_eq!(
            EnrolledSpeakers::read_or_empty(&paths)
                .unwrap()
                .references("Alice"),
            0,
            "the database is still written: the user was told and said yes: {output}"
        );
        assert_eq!(
            std::fs::read(session.transcript_json()).unwrap(),
            corrupt,
            "a file that could not be read is not rewritten from a guess: {output}"
        );
    }
}
