//! The read-only faces of `enroll`: what a run would offer, and what one answer would do.
//!
//! A scripted driver -- a shell script, CI, or an LLM deciding on a person's behalf -- can
//! already *execute* a naming decision (`--voice 2 --name Ryan` writes everything typing the
//! same name at the prompt would), but it has nothing to decide *with*: which enrolled people
//! a voice resembles, how wide the similarity gap is, whether the answer it is about to give
//! will be refused by the heard-at-once veto or displace somebody's shortest reference. This
//! module hands those two facts over without a terminal and without writing anything:
//!
//! - [`run`] with `--list` walks the unchanged rules construction and records every voice the
//!   run would offer, each with its ranked candidates.
//! - `run` with `--dry-run` computes the full consequence of the proposed answer through the
//!   real preview and reports it.
//!
//! Both go through [`run_enroll`] with a recording interviewer
//! rather than reaching for the session files themselves: the facts then cross the public
//! [`Interviewer`] seam exactly as the interactive frame reads
//! them, so the headless report cannot drift from what the frame shows.
//!
//! # stdout is the document
//!
//! Run commentary goes to stderr (the narrator is a [`Lines`] pointed
//! at it); stdout carries only the result, line-oriented by default and a single versioned
//! JSON document under `--json`. That purity is what makes `meethook enroll --list --json |
//! jq` work, and it is why the summary the ordinary run prints after `ask` never runs here.
//!
//! # What this module deliberately does not do
//!
//! It writes nothing, on any path. The interviewers answer [`Skip`](meethook_enroll::Answer::Skip)
//! and [`Quit`](meethook_enroll::Answer::Quit) -- decisions that commit no voice -- and the
//! integration test proves the stronger claim: the root is byte-identical before and after.
//! `--one-speaker` is out of reach for the reason its flag gives: assertion mode commits every
//! voice without asking, so there is no Interviewer door into a preview of it.

use std::io::{self, Write};

use anyhow::{Result, bail};
use meethook_enroll::{Answer, Interviewer, Lines, Refusal, Stored, Voice, run_enroll, speech};
use meethook_session::{Displaced, Paths, SessionId, TranscriptTemplate};
use meethook_transcribe::Resemblance;
use serde::Serialize;

use crate::EnrollArgs;

/// Runs the read-only half of `enroll` and prints its document to stdout.
///
/// Called from [`crate::commands::enroll`] in place of the answerer choice when `--list` or
/// `--dry-run` is set, which is also where the two are kept away from every writing path:
/// this function never reaches `answerer`, so neither a terminal
/// nor a pipe changes what happens here.
pub(crate) fn run(
    paths: &Paths,
    requested: &[SessionId],
    args: &EnrollArgs,
    template: &TranscriptTemplate,
) -> Result<()> {
    // clap cannot say "requires one of these two", so the pairing is checked here, before any
    // filesystem work: `--json` alone would otherwise walk every session to print nothing.
    if args.json && !(args.list || args.dry_run) {
        bail!("--json needs --list or --dry-run: there is no other document to print");
    }
    // No screen widening: the frame is the only thing that widens `Offer` for itself, and it
    // is not here. Everything else the flags mean is honoured unchanged.
    let mut rules = crate::commands::enroll_rules(args, false, template);
    // The one write a run makes before asking -- bringing a stale transcript in line with the
    // database -- is declined here: these faces promise the root exactly as found, and a query
    // must not be a writer however small the write. Nothing they report depends on it, since
    // the labels are computed from the database either way.
    rules.relabel_transcript = false;

    // Commentary to stderr, always: stdout carries only the document, whatever its shape.
    let mut stderr = io::stderr();

    if args.list {
        let mut surveyor = Surveyor::new();
        let report = run_enroll(
            paths,
            requested,
            rules,
            &mut surveyor,
            &mut Lines::new(&mut stderr),
        )?;
        if report.failed > 0 {
            bail!("{} enroll request(s) could not be served", report.failed);
        }
        let doc = ListOutput::from_surveyor(&surveyor);
        if args.json {
            return print_json(&doc);
        }
        doc.print_lines(&mut io::stdout())?;
        return Ok(());
    }

    // clap requires `--name` beside `--dry-run`, so the unwrap is a pairing guarantee rather
    // than a second validation.
    let name = args
        .name
        .as_deref()
        .expect("clap refuses --dry-run without --name");
    let mut dry = DryRun::new(name);
    let report = run_enroll(
        paths,
        requested,
        rules,
        &mut dry,
        &mut Lines::new(&mut stderr),
    )?;
    if report.failed > 0 {
        bail!("{} enroll request(s) could not be served", report.failed);
    }
    // A selector that reached a session the run passed over (orphaned, not transcribed)
    // offers no voice and so previews nothing: the narration said why, this says the
    // consequence for the request.
    let captured = dry
        .captured
        .ok_or_else(|| anyhow::anyhow!("no voice was offered, so nothing could be previewed"))?;
    if args.json {
        return print_json(&DryRunOutput {
            schema: DRY_RUN_SCHEMA,
            voice: DryRunVoice {
                session: captured.session.clone(),
                number: captured.number.clone(),
            },
            name: name.to_string(),
            consequence: captured.consequence.as_ref().map(DryRunConsequence::from),
        });
    }
    // The sentences come off the consequence itself -- the refusal, or what would be written
    // -- so the headless print cannot restate them. Only the header and the indent live here.
    let lines = match &captured.consequence {
        Some(consequence) => consequence.outcome_lines(),
        None => vec!["a name of nothing but spaces writes nothing".to_string()],
    };
    let mut stdout = io::stdout();
    print_dry_run_lines(&captured, name, &lines, &mut stdout)?;
    Ok(())
}

/// The `--json` half of the print: one pretty-printed document on stdout, the only thing that
/// ever goes there in these modes.
fn print_json(doc: &impl Serialize) -> Result<()> {
    let mut stdout = io::stdout();
    writeln!(stdout, "{}", serde_json::to_string_pretty(doc)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The interviewers
// ---------------------------------------------------------------------------

/// Records every voice a run offers, and answers none of them.
///
/// [`Skip`](meethook_enroll::Answer::Skip) rather than [`Quit`](meethook_enroll::Answer::Quit):
/// a skip is a decision that writes nothing and lets the run carry on to the next session,
/// which is exactly what a survey wants -- every question the run would have asked, in the
/// order it would have asked them, across every session it would have opened.
struct Surveyor {
    sessions: Vec<SurveyedSession>,
}

impl Surveyor {
    fn new() -> Self {
        Surveyor {
            sessions: Vec::new(),
        }
    }
}

/// One session of a survey, in the order the run visits its sessions.
struct SurveyedSession {
    id: String,
    meeting: Option<String>,
    voices: Vec<SurveyedVoice>,
}

/// One voice a run would offer, with what it would show beside the question.
struct SurveyedVoice {
    number: String,
    label: String,
    speech_seconds: f64,
    candidates: Vec<Resemblance>,
}

impl Interviewer for Surveyor {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        // Voices arrive per session, in session order, so a new session id means a new entry.
        // Checked before pushing: the run never revisits a finished session, so the last entry
        // is the current one whenever its id matches.
        if self
            .sessions
            .last()
            .is_none_or(|current| current.id != voice.session.to_string())
        {
            self.sessions.push(SurveyedSession {
                id: voice.session.to_string(),
                meeting: voice.meeting.map(|meeting| meeting.title.clone()),
                voices: Vec::new(),
            });
        }
        self.sessions
            .last_mut()
            .unwrap()
            .voices
            .push(SurveyedVoice {
                number: voice.number.to_string(),
                label: voice.attribution.label().to_string(),
                speech_seconds: voice.speech_seconds,
                candidates: voice.resembles.clone(),
            });
        Answer::Skip
    }
}

/// Previews one proposed answer on the first voice it is shown, then ends the run.
///
/// [`needs_one_voice`](Interviewer::needs_one_voice) is what inherits the selector rule for
/// free: without `--voice`/`--at` the run refuses to start with the same note it gives a
/// bare `--name`, and a selector matching nothing or several inherits today's not-served
/// failure the same way.
struct DryRun {
    name: String,
    captured: Option<CapturedAnswer>,
}

impl DryRun {
    fn new(name: &str) -> Self {
        DryRun {
            name: name.trim().to_string(),
            captured: None,
        }
    }
}

impl Interviewer for DryRun {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        // The preview reflects the database as it stands now, which for the first voice of the
        // run is the database as the run began it: nothing has been committed yet.
        if self.captured.is_none() {
            self.captured = Some(CapturedAnswer {
                session: voice.session.to_string(),
                number: voice.number.to_string(),
                consequence: voice.preview.of(&self.name),
            });
        }
        // Nothing was accepted, so nothing is written; ending the run costs nothing.
        Answer::Quit
    }

    fn needs_one_voice(&self) -> bool {
        true
    }
}

/// What the dry run saw, keyed on the stable "Unknown N" handle for the reason the frame keys
/// its own state on it.
struct CapturedAnswer {
    session: String,
    number: String,
    /// `None` for a name of nothing but spaces: that is a skip, not an answer, and the
    /// preview says so by refusing to compute one.
    consequence: Option<meethook_enroll::Consequence>,
}

// ---------------------------------------------------------------------------
// The documents
// ---------------------------------------------------------------------------

/// The stability contract both documents share.
///
/// A field added to a document does not bump its tag; a field renamed, retyped, or given a
/// different meaning does, and takes the next `.vN`. Scripts should match on `schema` before
/// trusting any other key.
const LIST_SCHEMA: &str = "meethook.enroll.list.v1";
const DRY_RUN_SCHEMA: &str = "meethook.enroll.dry-run.v1";

/// Every voice a run would offer, with the enrolled speakers each resembles.
#[derive(Debug, Serialize)]
struct ListOutput {
    schema: &'static str,
    /// One entry per session with at least one offered voice, in the order the run visits
    /// them. Empty when the run offers nothing at all.
    sessions: Vec<ListSession>,
}

#[derive(Debug, Serialize)]
struct ListSession {
    id: String,
    /// The meeting title if the session carries one, which most do not.
    meeting: Option<String>,
    voices: Vec<ListVoice>,
}

#[derive(Debug, Serialize)]
struct ListVoice {
    /// The stable "Unknown N" handle, the one `--voice` accepts.
    number: String,
    /// What the voice currently reads as in the transcript, which equals `number` unless the
    /// voice already carries a name (reachable under `--correct`).
    label: String,
    speech_seconds: f64,
    /// The enrolled speakers this voice sounds like, nearest first, unthresholded: the same
    /// ranking the frame's candidate pane shows, including people far outside the automatic
    /// match cut. Empty when nobody is enrolled.
    candidates: Vec<Resemblance>,
}

impl ListOutput {
    fn from_surveyor(surveyor: &Surveyor) -> ListOutput {
        ListOutput {
            schema: LIST_SCHEMA,
            sessions: surveyor
                .sessions
                .iter()
                .map(|session| ListSession {
                    id: session.id.clone(),
                    meeting: session.meeting.clone(),
                    voices: session
                        .voices
                        .iter()
                        .map(|voice| ListVoice {
                            number: voice.number.clone(),
                            label: voice.label.clone(),
                            speech_seconds: voice.speech_seconds,
                            candidates: voice.candidates.clone(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl ListOutput {
    /// The line form: one block per offered voice, headed the way the frame phrases its
    /// question and listing candidates in the frame's columns.
    fn print_lines(&self, out: &mut dyn Write) -> io::Result<()> {
        let mut first = true;
        for session in &self.sessions {
            for voice in &session.voices {
                if !first {
                    writeln!(out)?;
                }
                first = false;
                writeln!(
                    out,
                    "{}  {}  {} of speech",
                    session.id,
                    question(&voice.number, &voice.label),
                    speech(voice.speech_seconds)
                )?;
                for candidate in &voice.candidates {
                    writeln!(
                        out,
                        "    {:<20} {:>6.2}  {:>2} ref",
                        candidate.name, candidate.similarity, candidate.references
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// The dry-run line form: the header names the voice and the proposed answer, and the body is
/// the consequence's own [`outcome_lines`](meethook_enroll::Consequence) -- or the skip
/// sentence for a name of nothing but spaces -- indented under it.
fn print_dry_run_lines(
    captured: &CapturedAnswer,
    name: &str,
    lines: &[String],
    out: &mut dyn Write,
) -> io::Result<()> {
    writeln!(
        out,
        "{} in {}, answering \"{}\":",
        captured.number, captured.session, name
    )?;
    for line in lines {
        writeln!(out, "  {line}")?;
    }
    Ok(())
}

/// The frame's question wording, so the headless list asks the same question the pane does:
/// "who is this" for an unnamed voice, "is this right" for one that already reads a name.
fn question(number: &str, label: &str) -> String {
    if label == number {
        format!("who is {number}?")
    } else {
        format!("is {number} {label}?")
    }
}

/// What answering one selected voice with one name would do, computed by the real preview and
/// written nowhere.
#[derive(Debug, Serialize)]
struct DryRunOutput {
    schema: &'static str,
    /// The voice the selection resolved to.
    voice: DryRunVoice,
    /// The proposed answer, trimmed the way the run trims it.
    name: String,
    /// `null` for a name of nothing but spaces, which is a skip rather than an answer.
    consequence: Option<DryRunConsequence>,
}

#[derive(Debug, Serialize)]
struct DryRunVoice {
    session: String,
    number: String,
}

/// The public face of a [`Consequence`](meethook_enroll::Consequence).
///
/// Mapped field by field rather than derived on the struct itself: two of its fields are
/// crate-visible inside `meethook-enroll` because an interviewer able to read them could write
/// them, and deriving here would leak both into the document.
#[derive(Debug, Serialize)]
struct DryRunConsequence {
    /// Why the answer would not be honoured, if it would not be: the heard-at-once veto, or a
    /// name taken off another voice. Externally tagged, as serde renders enums by default.
    refused: Option<Refusal>,
    /// What `speakers.json` would record, or `null` for an answer that goes no further than
    /// this session.
    stored: Option<Stored>,
    /// Whether the name lands in this session alone, storing no reference.
    session_only: bool,
    /// The people who would lose a reference to this answer's correction.
    displaced: Vec<Displaced>,
    /// Names that would still hold a reference built from this exact voice afterwards.
    stale: Vec<String>,
}

impl From<&meethook_enroll::Consequence> for DryRunConsequence {
    fn from(consequence: &meethook_enroll::Consequence) -> Self {
        DryRunConsequence {
            refused: consequence.refused.clone(),
            stored: consequence.stored.clone(),
            session_only: consequence.session_only(),
            displaced: consequence.displaced.clone(),
            stale: consequence.stale.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn resemblance(name: &str, similarity: f32, references: usize) -> Resemblance {
        Resemblance {
            name: name.to_string(),
            similarity,
            references,
        }
    }

    /// Two voices in one session, then one in the next: the shape a multi-session run
    /// leaves, with the second session's voice already named.
    fn surveyed() -> Surveyor {
        let mut surveyor = Surveyor::new();
        surveyor_from_parts::push_session(
            &mut surveyor,
            "20260809-052600",
            None,
            &[
                (
                    "Unknown 1",
                    "Unknown 1",
                    37.4,
                    vec![resemblance("Ryan", 0.62, 3)],
                ),
                ("Unknown 2", "Unknown 2", 12.0, Vec::new()),
            ],
        );
        surveyor_from_parts::push_session(
            &mut surveyor,
            "20260810-093047",
            Some("Design Review"),
            &[(
                "Unknown 1",
                "Milo",
                205.0,
                vec![resemblance("Milo", 0.91, 1)],
            )],
        );
        surveyor
    }

    /// Test-side access to the grouping `identify` does against live `Voice`s, which a unit
    /// test cannot construct across the seam.
    mod surveyor_from_parts {
        use super::*;

        pub(super) fn push_session(
            surveyor: &mut Surveyor,
            id: &str,
            meeting: Option<&str>,
            voices: &[(&str, &str, f64, Vec<Resemblance>)],
        ) {
            surveyor.sessions.push(SurveyedSession {
                id: id.to_string(),
                meeting: meeting.map(str::to_string),
                voices: voices
                    .iter()
                    .map(|(number, label, seconds, candidates)| SurveyedVoice {
                        number: number.to_string(),
                        label: label.to_string(),
                        speech_seconds: *seconds,
                        candidates: candidates.clone(),
                    })
                    .collect(),
            });
        }
    }

    #[test]
    fn the_list_document_carries_every_offered_voice_with_its_ranking() {
        let doc = ListOutput::from_surveyor(&surveyed());
        let mut out = Vec::new();
        serde_json::to_writer(&mut out, &doc).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            r#"{"schema":"meethook.enroll.list.v1","sessions":[{"id":"20260809-052600","meeting":null,"voices":[{"number":"Unknown 1","label":"Unknown 1","speech_seconds":37.4,"candidates":[{"name":"Ryan","similarity":0.62,"references":3}]},{"number":"Unknown 2","label":"Unknown 2","speech_seconds":12.0,"candidates":[]}]},{"id":"20260810-093047","meeting":"Design Review","voices":[{"number":"Unknown 1","label":"Milo","speech_seconds":205.0,"candidates":[{"name":"Milo","similarity":0.91,"references":1}]}]}]}"#
        );
    }

    #[test]
    fn the_empty_survey_is_an_empty_session_list() {
        let doc = ListOutput::from_surveyor(&Surveyor::new());
        let mut out = Vec::new();
        serde_json::to_writer(&mut out, &doc).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            r#"{"schema":"meethook.enroll.list.v1","sessions":[]}"#
        );
    }

    #[test]
    fn the_list_lines_ask_the_frame_s_question_in_the_frame_s_columns() {
        let doc = ListOutput::from_surveyor(&surveyed());
        let mut out = Vec::new();
        doc.print_lines(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\
20260809-052600  who is Unknown 1?  37s of speech
    Ryan                   0.62   3 ref

20260809-052600  who is Unknown 2?  12s of speech

20260810-093047  is Unknown 1 Milo?  3m 25s of speech
    Milo                   0.91   1 ref
"
        );
    }

    fn dry_run_output(consequence: Option<DryRunConsequence>) -> DryRunOutput {
        DryRunOutput {
            schema: DRY_RUN_SCHEMA,
            voice: DryRunVoice {
                session: "20260809-052600".to_string(),
                number: "Unknown 2".to_string(),
            },
            name: "Ryan".to_string(),
            consequence,
        }
    }

    #[test]
    fn the_dry_run_document_maps_the_consequence_field_by_field() {
        let doc = dry_run_output(Some(DryRunConsequence {
            refused: Some(Refusal::Vetoed {
                holder: Some("Unknown 5".to_string()),
            }),
            stored: Some(Stored::Added { held: 4 }),
            session_only: false,
            displaced: vec![Displaced {
                name: "Milo".to_string(),
                remaining: 2,
            }],
            stale: Vec::new(),
        }));
        let mut out = Vec::new();
        serde_json::to_writer(&mut out, &doc).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            r#"{"schema":"meethook.enroll.dry-run.v1","voice":{"session":"20260809-052600","number":"Unknown 2"},"name":"Ryan","consequence":{"refused":{"Vetoed":{"holder":"Unknown 5"}},"stored":{"Added":{"held":4}},"session_only":false,"displaced":[{"name":"Milo","remaining":2}],"stale":[]}}"#
        );
    }

    #[test]
    fn a_whitespace_name_is_a_null_consequence() {
        let doc = dry_run_output(None);
        let mut out = Vec::new();
        serde_json::to_writer(&mut out, &doc).unwrap();
        assert!(
            String::from_utf8(out)
                .unwrap()
                .ends_with("\"consequence\":null}")
        );
    }

    /// The line printer takes the consequence's own sentences, which a unit test cannot build
    /// across the seam -- `Consequence`'s state fields are crate-visible inside
    /// `meethook-enroll` -- so it asserts the header and the indent, and the sentence content
    /// is pinned where it is stated: on the library side.
    fn captured() -> CapturedAnswer {
        CapturedAnswer {
            session: "20260809-052600".to_string(),
            number: "Unknown 2".to_string(),
            consequence: None,
        }
    }

    #[test]
    fn a_refused_answer_prints_the_refusal_and_nothing_else() {
        let mut out = Vec::new();
        print_dry_run_lines(
            &captured(),
            "Ryan",
            &[Refusal::Taken {
                voice: "Unknown 1".to_string(),
                losing: "Ryan".to_string(),
            }
            .sentence()],
            &mut out,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\
Unknown 2 in 20260809-052600, answering \"Ryan\":
  unavailable: Unknown 1 would stop reading Ryan
"
        );
    }

    #[test]
    fn an_accepted_answer_prints_its_sentences_indented_under_the_header() {
        let mut out = Vec::new();
        print_dry_run_lines(
            &captured(),
            "Ryan",
            &[
                "stores this recording in place of their shortest, 12s, 4 in all".to_string(),
                "takes a recording off Milo, leaving them 2".to_string(),
                "leaves a recording of this voice standing under Nate".to_string(),
            ],
            &mut out,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\
Unknown 2 in 20260809-052600, answering \"Ryan\":
  stores this recording in place of their shortest, 12s, 4 in all
  takes a recording off Milo, leaving them 2
  leaves a recording of this voice standing under Nate
"
        );
    }

    #[test]
    fn a_whitespace_name_prints_the_skip_sentence() {
        let mut out = Vec::new();
        print_dry_run_lines(
            &captured(),
            "   ",
            &["a name of nothing but spaces writes nothing".to_string()],
            &mut out,
        )
        .unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "\
Unknown 2 in 20260809-052600, answering \"   \":
  a name of nothing but spaces writes nothing
"
        );
    }
}
