//! Subcommand bodies.
//!
//! All six are thin: the rules they enforce live in `meethook-record`,
//! `meethook-transcribe` and `meethook-enroll`, where they can be tested without a terminal.
//! What is left here is the terminal itself -- printing, prompting, and playing audio --
//! which is exactly the part no test can decide.

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use meethook_enroll::{
    Answer, Confirm, EnrollReport, EnrollRules, Enrolment, Forgotten, GivenName, Interviewer,
    Labelled, Lines, MeetingChoice, MeetingSource, Offer, Selection, Sessions, Target, Voice,
    VoiceSelector, incomplete, run_enroll, run_forget, run_meeting, run_speakers, speech,
};
use meethook_models::{ModelSpec, ensure_model};
use meethook_session::{Paths, SessionId, TranscriptTemplate};

use crate::EnrollArgs;
use crate::clips::Clips;
use crate::screen::{Interface, Shared};
use meethook_transcribe::{
    Attribution, EMBEDDING_MODEL, Engines, OnnxDiarizer, SEGMENTATION_MODEL, SILERO_VAD_MODEL,
    WHISPER_MODEL, WhisperEngine, run_batch,
};

/// Transcribes recorded sessions.
///
/// Fully non-interactive, deliberately: this is meant to be aimed at a directory of
/// meetings and left alone. All four models are acquired lazily through the one factory
/// below, so a run that turns out to have nothing to do never pays for a download.
pub fn transcribe(
    paths: &Paths,
    session_ids: &[String],
    force: bool,
    template: Option<&Path>,
    mixdown_settings: meethook_transcribe::mixdown::Settings,
) -> Result<()> {
    let requested = parse_session_ids(session_ids)?;
    let template = TranscriptTemplate::resolve(paths, template)?;

    let models_dir = paths.models_dir();
    let mut open_engine =
        || -> std::result::Result<Engines, Box<dyn std::error::Error + Send + Sync>> {
            // Smallest first, so that a first run reaches its slowest download last -- by which
            // point everything else is known to be in place. The VAD weights are 885 KB,
            // diarization's two models are 32 MB, and Whisper is 1.6 GB.
            let silero = fetch(&models_dir, &SILERO_VAD_MODEL)?;
            let segmentation = fetch(&models_dir, &SEGMENTATION_MODEL)?;
            let embedding = fetch(&models_dir, &EMBEDDING_MODEL)?;
            let whisper = fetch(&models_dir, &WHISPER_MODEL)?;

            let diarizer = OnnxDiarizer::load(&segmentation, &embedding)?;
            // Only meaningful where a CoreML EP was compiled in: off macOS `accelerated` is
            // always false by construction of the build, and printing "CoreML declined"
            // there would name a component the platform never had.
            #[cfg(target_os = "macos")]
            if !diarizer.accelerated() {
                // Correct but several times slower. Worth a line: the alternative is a user
                // wondering why transcribing a meeting suddenly takes minutes.
                eprintln!("Note: CoreML declined these graphs; diarization is running on CPU.");
            }

            let asr = WhisperEngine::load(&whisper, &silero)?;
            // Same reasoning: off macOS the CPU is the only path, so a non-accelerated
            // engine is not a reportable choice -- it is the build.
            #[cfg(target_os = "macos")]
            if !asr.accelerated() {
                // Only reachable via MEETHOOK_CPU, so this confirms an explicit choice rather
                // than reporting a surprise -- and says out loud what that choice costs.
                eprintln!(
                    "Note: MEETHOOK_CPU is set; speech recognition is running on the CPU and \
                     will be much slower."
                );
            }

            Ok(Engines {
                asr: Box::new(asr),
                diarizer: Box::new(diarizer),
            })
        };

    let stdout = io::stdout();
    let report = run_batch(
        paths,
        &requested,
        force,
        &template,
        mixdown_settings,
        &mut open_engine,
        &mut stdout.lock(),
    )?;

    // Skips -- including orphans -- are normal and stay at exit 0; a session that genuinely
    // failed is what makes the run unsuccessful.
    if report.failed > 0 {
        bail!(
            "{} of {} session(s) failed to transcribe",
            report.failed,
            report.failed + report.transcribed
        );
    }
    Ok(())
}

/// Acquires one model, reporting its download identically to every other model's.
fn fetch(
    models_dir: &std::path::Path,
    spec: &'static ModelSpec,
) -> meethook_models::Result<PathBuf> {
    let mut report = DownloadProgress::new(spec);
    ensure_model(models_dir, spec, &mut |done, total| {
        report.update(done, total)
    })
}

/// Prints download progress on stderr, one line, rewritten in place.
///
/// stderr rather than stdout so the batch's own output stays a clean, greppable record of
/// what happened to each session. Throttled to whole percent because a 1.6 GB download at a
/// megabyte a chunk would otherwise emit sixteen hundred lines.
struct DownloadProgress {
    spec: &'static ModelSpec,
    last_percent: u64,
    started: bool,
}

impl DownloadProgress {
    fn new(spec: &'static ModelSpec) -> DownloadProgress {
        DownloadProgress {
            spec,
            last_percent: 0,
            started: false,
        }
    }

    fn update(&mut self, done: u64, total: u64) {
        if !self.started {
            self.started = true;
            // Megabytes for the two diarization graphs, gigabytes for Whisper: "0.0 GB"
            // against a 6 MB file reads as a bug in the downloader.
            let size = if total >= 1_000_000_000 {
                format!("{:.1} GB", total as f64 / 1e9)
            } else {
                format!("{:.0} MB", total as f64 / 1e6)
            };
            eprintln!("Fetching {} ({size}, one time)", self.spec.file_name);
        }
        let percent = (done * 100).checked_div(total).unwrap_or(0);
        if percent == self.last_percent && done < total {
            return;
        }
        self.last_percent = percent;
        let mut stderr = io::stderr();
        let _ = write!(stderr, "\r  {percent}%");
        if done >= total {
            let _ = writeln!(stderr);
        }
        let _ = stderr.flush();
    }
}

pub fn enroll(paths: &Paths, args: &EnrollArgs, template: Option<&Path>) -> Result<()> {
    let requested = parse_session_ids(&args.session_ids)?;
    let template = TranscriptTemplate::resolve(paths, template)?;
    // The read-only faces take over before any answerer is chosen: neither prompts nor opens
    // a frame, and neither prints the run summary below -- stdout carries only their document.
    if args.list || args.dry_run {
        return crate::headless::run(paths, &requested, args, &template);
    }
    // Which answerer this run has, by the rule in `answerer`. Decided here and passed in so that
    // this function's only remaining job is the summary below -- which has to print with nothing
    // of `ask`'s still holding the screen.
    let chosen = answerer(args.name.is_some(), args.plain, Tty::current());
    let report = ask(paths, &requested, args, &template, chosen)?;
    // The last thing the command writes, so the lock on stdout dies with the function.
    let mut out = io::stdout().lock();
    run_summary(&mut out, &report)
}

/// What a finished run says about itself, off the report alone.
///
/// A function rather than inline because every clause is conditional on its own count and the
/// wording of each one is pinned by test -- including the ones this ticket added -- and a seam
/// is what lets a test hold the whole block without a terminal behind it. The `bail!` at the
/// end is the exit status: a request that could not be served makes the run unsuccessful,
/// exactly as in `transcribe`.
fn run_summary(out: &mut dyn Write, report: &EnrollReport) -> Result<()> {
    writeln!(
        out,
        "\n{} named, {} skipped, {} session(s) passed over",
        report.named, report.skipped, report.passed_over
    )?;
    // A sub-count of `named` rather than a separate outcome, so this reads as a qualification
    // of the line above rather than as more voices. Says where the name went, because "named"
    // on its own would leave a user expecting those people to be recognised in the next
    // meeting -- which is exactly what a session-scoped name does not do.
    if report.session_only > 0 {
        writeln!(
            out,
            "{} of those named in their own session only, with no reference stored -- \
             each said why on its own line above",
            report.session_only
        )?;
    }
    // A voice left as it was found is a kept identification, not an unanswered question, and
    // only ever arises under `--correct`.
    if report.kept > 0 {
        writeln!(out, "{} identification(s) kept as they were", report.kept)?;
    }
    // Only when there were any: a run that asked about everything should not end on a line
    // about the nothing it held back.
    if report.held_back > 0 {
        writeln!(
            out,
            "{} quieter voice(s) not offered -- meethook enroll --all asks about those too",
            report.held_back
        )?;
    }
    // An answer declined because honouring it would have taken a name off another voice. Only
    // when there were any, like the two above, and worth its own line rather than being folded
    // into the skips: those voices are still unnamed *and* the user has already answered them,
    // so the next step is to read the refusal lines rather than to answer again.
    if report.refused > 0 {
        writeln!(
            out,
            "{} answer(s) refused: honouring them would have un-named another voice",
            report.refused
        )?;
    }
    // Tentative guesses the user turned down. Honoured answers that wrote something -- the
    // suppression row and the demoted label -- so they are neither refused nor skipped, and
    // worth their own line rather than disappearing into the named count they are not in.
    if report.denied > 0 {
        writeln!(out, "{} guess(es) denied", report.denied)?;
    }
    // The run-wide half of the veto override, beside the per-session lines the narration
    // already printed: an assertion that never hit a pair it overrode says nothing here, which
    // is the same as saying nothing happened.
    if report.vetoes_overridden > 0 {
        writeln!(
            out,
            "overrode the heard-at-once veto on {} voice(s)",
            report.vetoes_overridden
        )?;
    }
    // Skips and pass-overs are ordinary; a request that could not be served -- a session that
    // could not be read, an id that is not on disk, a `--voice` matching no voice or several --
    // is what makes the run unsuccessful, exactly as in `transcribe`. Each has already printed
    // the line saying which it was, so this only has to make the exit status say so too.
    if report.failed > 0 {
        bail!("{} enroll request(s) could not be served", report.failed);
    }
    Ok(())
}

/// How a run is configured, off the flags alone: the one place that reads each flag into the
/// rules bundle, so the ordinary run and the read-only ones cannot configure the same flags
/// differently.
///
/// `screen` is the frame's self-widening: it asks about every voice and every named one for its
/// queue pane, so `Offer` is widened for it rather than for `--all`/`--correct`. The read-only
/// paths pass `false` -- the frame is not among their answerers.
pub(crate) fn enroll_rules<'a>(
    args: &'a EnrollArgs,
    screen: bool,
    template: &'a TranscriptTemplate,
) -> EnrollRules<'a> {
    // Unlike a session id there is nothing to validate about a `--voice`: a selector that matches
    // nothing is answered against the session's actual voices, which is a better message than
    // anything this edge could produce without having read them. `--at` is the other way round --
    // a malformed timestamp has nothing to be compared against -- so clap has already parsed it.
    let selector = args.voice.as_deref().map(VoiceSelector::from);
    EnrollRules {
        selector: match (selector, args.at) {
            (Some(selector), _) => Some(Selection::Voice(selector)),
            (None, Some(at)) => Some(Selection::At(at)),
            (None, None) => None,
        },
        // Which flag answers which question, readable here rather than positional.
        offer: Offer {
            quiet: args.all || screen,
            named: args.correct || screen,
        },
        // The other half of what `--correct` means, and a separate axis from `offer` for the
        // reason `Sessions` gives: `offer` says which voices a session asks about, this says
        // whether a session with nothing unresolved is opened at all. Not widened for the frame:
        // a session with nothing left to answer is one the user did not ask to revisit, and
        // opening it would put an empty queue on the screen for every finished meeting on disk.
        sessions: if args.correct {
            Sessions::Every
        } else {
            Sessions::Unresolved
        },
        // A separate axis from `offer`: that one decides which voices are asked about, this
        // one what an answer to a quiet voice writes.
        enrolment: if args.force_reference {
            Enrolment::Always
        } else {
            Enrolment::AboveTheFloor
        },
        // Every run this function serves may write answers, so every one of them brings a stale
        // transcript in line first; the read-only faces decline it where they take over.
        relabel_transcript: true,
        // The CLI refuses this flag alongside any selector or up-front name (see the flag), so
        // none of them can be set beside it: the assertion stands in for the queue and its
        // gates alike, and composing it with a selector would give every voice two answers.
        one_speaker: args.one_speaker.as_deref(),
        template,
    }
}

/// Runs the questions and returns what they came to, with nothing on the screen afterwards.
///
/// Split out of [`enroll`] rather than inlined for one reason: the full-screen answerer holds the
/// terminal for as long as it is alive, and the run summary has to land on the restored screen
/// below the narration. A function boundary is what makes that ordering structural -- the frame
/// cannot outlive this call -- instead of a `drop` somebody has to remember not to move.
fn ask(
    paths: &Paths,
    requested: &[SessionId],
    args: &EnrollArgs,
    template: &TranscriptTemplate,
    chosen: Answerer,
) -> Result<meethook_enroll::EnrollReport> {
    // The frame navigates rather than being fed a queue, so it takes every voice the session has
    // and decides for itself which to show; `--all` and `--correct` are how the *line* prompt
    // widens what it is offered, and AC #2 is that the frame needs neither.
    let rules = enroll_rules(args, chosen == Answerer::Screen, template);

    // Prompt-free by construction: the assertion names every voice without asking about any,
    // so the full-screen interface has nothing to show -- and a scripted driver must reach the
    // same writes from either terminal. This is the seam the key handler reaches mid-run;
    // one body of work, two doors into it. The answerer below is a `Terminal` rather than a
    // silent one on purpose: the run never consults it, and if a future change ever did, a
    // question appearing on screen beats a fabricated answer.
    if args.one_speaker.is_some() {
        return Ok(run_enroll(
            paths,
            requested,
            rules,
            &mut Terminal::default(),
            &mut Lines::new(&mut io::stdout()),
        )?);
    }

    // `--name` is refused without a selector by `run_enroll`, which is where both halves of that
    // rule are in hand. A match rather than `expect(..)`: `answerer` returns `Given` only when
    // `named` is true, and `named` is `args.name.is_some()`, so the pairing cannot fail -- and
    // were it ever to, falling through to the line prompt is better than a panic in a command
    // that may already have written names to disk.
    match (chosen, args.name.as_deref()) {
        (Answerer::Given, Some(name)) => Ok(run_enroll(
            paths,
            requested,
            rules,
            &mut GivenName::new(name),
            &mut Lines::new(&mut io::stdout()),
        )?),
        (Answerer::Screen, _) => {
            // A frame cannot share stdout with `Lines`, so the two share a buffer instead and
            // the frame draws the narration in a pane. `finish` writes the whole buffer out once
            // the terminal is back, so a full-screen run leaves the same scrollback a plain one
            // does -- and it runs whether or not the run itself succeeded, because narration
            // already written describes work already done to the disk.
            let narration = Shared::default();
            let mut narrator = narration.clone();
            let mut frame = Interface::new(narration, paths.clone());
            let outcome = run_enroll(
                paths,
                requested,
                rules,
                &mut frame,
                &mut Lines::new(&mut narrator),
            );
            let flushed = frame.finish(&mut io::stdout());
            let report = outcome?;
            flushed?;
            Ok(report)
        }
        _ => Ok(run_enroll(
            paths,
            requested,
            rules,
            &mut Terminal::default(),
            &mut Lines::new(&mut io::stdout()),
        )?),
    }
}

/// Reports who is enrolled and what each of their stored recordings is currently naming.
///
/// The thinnest of the five, and read-only: everything printed comes back from one call, and
/// the only decision left here is the exit status.
pub fn speakers(paths: &Paths) -> Result<()> {
    let scan = run_speakers(paths, &mut io::stdout())?;

    // A report whose entire claim is its completeness must not exit 0 while admitting it could
    // not read three sessions. The same rule `enroll` and `transcribe` apply to a request they
    // could not serve, and as there, each one has already printed the line saying which it was
    // -- above the whole listing, which is still printed.
    //
    // The sentence itself comes from `meethook-enroll`, not from here: the enrolment frame draws
    // the same fact about the same scan, and one wording is what stops the two from disagreeing.
    if !scan.unreadable.is_empty() {
        bail!("{}", incomplete(scan.unreadable.len()));
    }
    Ok(())
}

/// Removes one stored recording of somebody, or all of them, having first printed what that costs.
///
/// Thin for the same reason `speakers` is: every line, including the one telling the user that
/// nothing was written and how to confirm, comes back from `run_forget`, which is what makes the
/// wording decidable in `cargo test`. The one decision left here is the exit status.
pub fn forget(
    paths: &Paths,
    name: &str,
    reference: Option<usize>,
    yes: bool,
    template: Option<&Path>,
) -> Result<()> {
    let template = TranscriptTemplate::resolve(paths, template)?;
    let target = Target {
        name: name.to_string(),
        reference,
    };
    // `--yes` is the only thing that lets a write happen, and it is read here rather than passed
    // through as a bool so the library's own type says which of the two a run is.
    let confirm = if yes {
        Confirm::Confirmed
    } else {
        Confirm::Preview
    };

    let removal = match run_forget(paths, &target, confirm, &template, &mut io::stdout())? {
        // The detail -- the path, and what *is* stored -- has already been printed, so this only
        // has to make the exit status say the request was not served.
        Forgotten::NotStored => match reference {
            Some(handle) => bail!("{name} holds no reference {handle}"),
            None => bail!("nobody called {name} is enrolled"),
        },
        Forgotten::Previewed(removal) | Forgotten::Removed(removal) => removal,
    };

    // The rule `enroll`, `transcribe` and `speakers` already apply: a request that could not be
    // fully served makes the run unsuccessful, having first printed the line saying which part it
    // was. A removal that happened alongside an incomplete scope is not a new shape of outcome --
    // `enroll` already exits non-zero on a failed session after writing the names it did take.
    if !removal.unreadable.is_empty() || !removal.unwritable.is_empty() {
        bail!(
            "{} session(s) could not be read and {} transcript(s) could not be brought in line, \
             so this removal is not complete",
            removal.unreadable.len(),
            removal.unwritable.len()
        );
    }
    Ok(())
}

/// Corrects, or clears, the meeting a session was labelled with.
///
/// Thin like `speakers` and `forget`: every printed line, including the numbered offer and the
/// instruction for acting on it, comes back from `run_meeting`, which is what makes the whole
/// wording decidable in `cargo test` on a machine with no calendar grant. Left here are the
/// three things that genuinely need this crate -- the real calendar, the optional grant prompt,
/// and the exit status.
///
/// The grant is asked for only when a listing is needed. `--clear` reaching a calendar prompt
/// would be the opposite of the point: it is the path for the user whose calendar is refused,
/// unreadable, or simply does not contain the meeting they were in.
pub fn meeting(
    paths: &Paths,
    session_id: &str,
    event: Option<u32>,
    clear: bool,
    template: Option<&Path>,
) -> Result<()> {
    // Before any filesystem work, so a typo fails as a typo, exactly as the positional ids of
    // `transcribe` and `enroll` do.
    let session = match SessionId::parse(session_id) {
        Ok(session) => session,
        Err(e) => bail!(e),
    };
    let template = TranscriptTemplate::resolve(paths, template)?;
    let choice = match (clear, event) {
        // clap refuses the two flags together, so this order decides nothing a user can reach.
        (true, _) => MeetingChoice::Clear,
        (false, Some(nth)) => MeetingChoice::Event(nth as usize),
        (false, None) => MeetingChoice::Show,
    };

    // The grant is asked for only when a listing is needed, and only where a calendar exists
    // to grant access to: off macOS there is no backend at all, so the prompt would ask for
    // nothing. `--clear` reaching a calendar prompt anywhere would be the opposite of the
    // point: it is the path for the user whose calendar is refused, unreadable, or simply
    // does not contain the meeting they were in.
    #[cfg(target_os = "macos")]
    if !matches!(choice, MeetingChoice::Clear)
        && let Some(problem) = meethook_record::request_calendar_access()
    {
        // Guidance, not an error: a refused grant means an empty offer, which `run_meeting`
        // prints as a sentence pointing at `--clear`. Never fatal, for the reason `record`
        // gives at its own call to this -- the calendar is an enrichment, not a prerequisite.
        println!("{problem}");
    }

    match run_meeting(
        paths,
        &session,
        choice,
        &Calendar,
        &template,
        &mut io::stdout(),
    )? {
        // The count and the way to see the list have already been printed; this only makes the
        // exit status say the request was not served, as `forget` does for a name nobody holds.
        Labelled::NoSuchEvent { offered } => bail!(
            "there is no meeting {} to attach: {offered} were offered around {session}",
            event.unwrap_or_default()
        ),
        Labelled::Shown | Labelled::Written(_) => Ok(()),
    }
}

/// The real calendar, behind `meethook-enroll`'s one-method seam.
///
/// The whole of what the correction command needs a calendar for, and the only place in this
/// binary that knows where meetings come from. Total by construction on the other side: a
/// refused grant, an unreadable store and a session with nothing booked around it are all an
/// empty list, so no correction fails because the calendar was unavailable.
struct Calendar;

#[cfg(target_os = "macos")]
impl MeetingSource for Calendar {
    fn around(&self, at: jiff::Timestamp) -> Vec<meethook_session::Meeting> {
        meethook_record::meetings_around(at)
    }
}

// Off macOS there is no calendar backend at all -- EventKit is one of the frameworks the
// capture crate cannot compile without -- so the seam answers the total-by-construction
// empty list rather than an error: `meeting <id>` lists nothing and points at `--clear`,
// which remains fully functional because clearing never consults the calendar.
#[cfg(not(target_os = "macos"))]
impl MeetingSource for Calendar {
    fn around(&self, _at: jiff::Timestamp) -> Vec<meethook_session::Meeting> {
        Vec::new()
    }
}

/// Whether the run is attached to a person, as the two streams `enroll` and `record` actually
/// use.
///
/// A struct rather than two bools passed positionally, for the reason [`crate::EnrollArgs`] gives
/// for itself: adjacent bools transpose silently, and a transposed one here would open a
/// full-screen interface onto a pipe.
///
/// Not stderr. Narration goes to stdout, through `Lines::new(&mut io::stdout())` in [`enroll`],
/// and [`Terminal::identify`] asks its question with `println!` -- so stdout is the stream a
/// full-screen frame would have to fight over, and stderr says nothing about whether it could.
/// `record`'s line-based output is the same story: its status lines and its finish summary all
/// go to stdout.
///
/// Shared with `record` rather than duplicated there: "is this run attached to a person" is one
/// fact about the process, and reading `is_terminal()` twice in two modules is where the two
/// answers drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tty {
    pub(crate) stdin: bool,
    pub(crate) stdout: bool,
}

impl Tty {
    /// What this process is actually attached to. The one line of this decision no test can
    /// decide, which is why it is the only thing in here that reads the process. Shared with
    /// `record` rather than re-read there: both presenters must answer the same question about
    /// the same two streams.
    pub(crate) fn current() -> Tty {
        Tty {
            stdin: io::stdin().is_terminal(),
            stdout: io::stdout().is_terminal(),
        }
    }

    /// Whether both ends are attached to a terminal: the only case a full-screen interface may
    /// open in. A pipe on either end keeps the run on its line output.
    ///
    /// Only `record` calls this, and only under its macOS-only presenter, so a non-macOS,
    /// non-test build never reaches it -- hence the same cfg as that call site, or clippy
    /// reports it dead.
    #[cfg(any(target_os = "macos", test))]
    pub(crate) fn is_attached(self) -> bool {
        self.stdin && self.stdout
    }
}

/// Which of `enroll`'s answerers a run gets.
///
/// Three distinct answerers: [`GivenName`], the line-based [`Terminal`], and the full-screen
/// [`Interface`]. Keeping the choice in its own type is what lets [`answerer`] state the rule
/// once and be tested against it without a terminal, while [`enroll`] does the constructing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Answerer {
    /// `--name`: the answer was given up front, so nothing is asked at all.
    Given,
    /// The line-based prompt: a question printed to stdout, an answer read from stdin.
    Prompt,
    /// The full-screen interface.
    Screen,
}

/// Which answerer a run gets, given `--name`, `--plain`, and what the streams are attached to.
///
/// The order is the rule, and it is deliberate:
///
/// 1. `--name` wins over everything. A name given up front is never shown the voice it lands
///    on, so there is no question to ask on any path and no interface to open.
/// 2. A pipe on *either* end is the line prompt. Both streams, not either: the prompt writes
///    the question and the snippets to stdout and reads the answer from stdin, so a run being
///    driven -- by CI, by a shell pipeline, by a subprocess -- must not write escape sequences
///    into a captured buffer, and must not wait for a keypress that a script cannot send.
/// 3. `--plain` is the explicit override, for somebody on a real terminal who wants the old
///    prompt back, or who needs the interface out of a reproduction.
///
/// A function rather than an inline `if` for the reason `meeting_line` is one: the rule is
/// then decidable in `cargo test` with no terminal in front of it.
fn answerer(named: bool, plain: bool, tty: Tty) -> Answerer {
    if named {
        Answerer::Given
    } else if !tty.stdin || !tty.stdout || plain {
        Answerer::Prompt
    } else {
        Answerer::Screen
    }
}

/// How many of a voice's lines this prompt shows before asking who it is.
///
/// Enough to hear a person in the words -- what they said, what they were asked -- without
/// turning a prompt into a page of transcript that hides the question at the bottom of it.
///
/// Here rather than in `meethook-enroll`, which hands over every snippet a voice has: this is a
/// fact about one screenful of scrollback, and so about *this* answerer. A frame that can scroll
/// takes a different number.
const SNIPPETS: usize = 3;

/// The interactive half of `enroll`: what a prompt looks like, and how a clip gets played.
///
/// Everything about *which* voice is asked about, and what an answer writes, is on the other
/// side of the `Interviewer` seam in `meethook-enroll`. What is left here needs a person in
/// front of it, so none of it is tested; keeping it this small is what makes that acceptable.
#[derive(Default)]
struct Terminal {
    clips: Clips,
}

impl Terminal {
    /// Plays a clip, reporting anything that stopped it under the snippets.
    fn play(&mut self, clip: &[f32]) {
        if clip.is_empty() {
            println!("    (no audio for this voice)");
            return;
        }
        if let Err(e) = self.clips.play(clip) {
            println!("    (could not play the clip: {e})");
        }
    }
}

impl Interviewer for Terminal {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        // An already-named voice is a different question -- "is this right", not "who is
        // this" -- and asking the second one with a name already on the screen invites the
        // user to type that name straight back in. The basis clause says which question it is.
        let (label, basis) = match voice.attribution {
            Attribution::Identified { name, similarity } => (
                name.as_str(),
                format!(", identified at {similarity:.2} confidence"),
            ),
            // No confidence to print, and saying so matters: this name is here because
            // somebody typed it, and it is recorded against this session rather than as a
            // reference, so it will not follow the person into the next meeting.
            Attribution::Assigned { name } => {
                (name.as_str(), ", named for this session".to_string())
            }
            // A guess is an open question, not an assertion: `is_named()` below is false for
            // it, so this asks "who is this" and Enter keeps the guess standing -- the same
            // epistemic status the transcript carries in the label's question mark.
            Attribution::Tentative { name, similarity } => (
                name.as_str(),
                format!(", tentatively {name} at {similarity:.2}"),
            ),
            Attribution::Unknown(label) => (label.as_str(), String::new()),
        };
        // The position sits right after the session id, at a fixed column: what it is for is
        // being findable by eye on every header, which a suffix after a variable-length speech
        // time would not be.
        println!(
            "\n{}  {}  {label} -- {} of speech{basis}",
            voice.session,
            voice.position,
            speech(voice.speech_seconds)
        );
        if voice.snippets.is_empty() {
            println!("    (nothing was transcribed for this voice)");
        }
        // Cut to `SNIPPETS` here rather than across the seam, so a voice with fifty lines still
        // leaves the question visible at the bottom of the screen.
        for snippet in voice.snippets.iter().take(SNIPPETS) {
            println!("    \"{}\"", snippet.text);
        }
        self.play(voice.clip);

        // Enter is `Answer::Skip` either way, and `Skip` writes nothing -- which is exactly
        // what keeping an identification means, so no new answer variant is needed.
        if voice.attribution.is_named() {
            print!(
                "Who is this? (name to correct, Enter to keep {}, Ctrl-D to stop) ",
                voice.attribution.label()
            );
        } else {
            print!("Who is this? (name, Enter to skip, Ctrl-D to stop) ");
        }
        // Without this the question sits in the buffer behind the answer.
        if io::stdout().flush().is_err() {
            return Answer::Quit;
        }

        // End of input is somebody stopping, not a failure: every name accepted so far is
        // already on disk, and so is the transcript it changed. Ctrl-C, which kills the
        // process outright rather than arriving here, is safe for the same reason.
        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => Answer::Quit,
            Ok(_) if line.trim().is_empty() => Answer::Skip,
            // Never insists. This prompt reports what an answer did after the fact and shows no
            // cost before it, so a user typing here has not been shown the third voice an
            // override would cost -- the frame is the interface that has, and it is the one
            // that can set `anyway`.
            Ok(_) => Answer::Named {
                name: line.trim().to_string(),
                anyway: false,
            },
        }
    }
}

/// Validates positional ids before any filesystem work, so a typo fails immediately with a
/// message about the typo rather than as a confusing "not found" much later.
fn parse_session_ids(raw: &[String]) -> Result<Vec<SessionId>> {
    let mut ids = Vec::with_capacity(raw.len());
    for value in raw {
        match SessionId::parse(value) {
            Ok(id) => ids.push(id),
            Err(e) => bail!(e),
        }
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::{Answerer, EnrollReport, Tty, answerer};

    /// Every shape the two streams can be in, so a rule is asserted over all of them rather
    /// than at the one point a spot check happened to pick.
    const STREAMS: [Tty; 4] = [
        Tty {
            stdin: true,
            stdout: true,
        },
        Tty {
            stdin: true,
            stdout: false,
        },
        Tty {
            stdin: false,
            stdout: true,
        },
        Tty {
            stdin: false,
            stdout: false,
        },
    ];

    /// A terminal at both ends: the only shape a full-screen interface may be opened onto.
    const ATTACHED: Tty = Tty {
        stdin: true,
        stdout: true,
    };

    /// The whole of `--name`'s guarantee: it is answered up front on *every* combination of
    /// the other two inputs, so no stream shape and no flag can route it into an interface.
    #[test]
    fn a_name_given_up_front_is_never_asked_for_again() {
        for tty in STREAMS {
            for plain in [false, true] {
                assert_eq!(
                    answerer(true, plain, tty),
                    Answerer::Given,
                    "--name lost to plain={plain} {tty:?}"
                );
            }
        }
    }

    /// A pipe on either end is a run being driven rather than used: escape sequences would go
    /// into somebody's captured buffer, and a keypress the driver cannot send would be waited
    /// for. Both ends are checked because guarding only one is the usual way this goes wrong.
    #[test]
    fn a_pipe_on_either_end_is_the_line_prompt() {
        for tty in STREAMS {
            if tty == ATTACHED {
                continue;
            }
            assert_eq!(
                answerer(false, false, tty),
                Answerer::Prompt,
                "a full-screen interface was chosen for {tty:?}"
            );
        }
    }

    /// The override, which is the only reason the flag exists: a person at a real terminal
    /// asking for the plain prompt gets it.
    #[test]
    fn plain_forces_the_line_prompt_on_a_terminal() {
        assert_eq!(answerer(false, true, ATTACHED), Answerer::Prompt);
    }

    /// The arm that has nothing behind it yet, and the one assertion that would go quiet if
    /// somebody deleted it: without this, dropping the interactive decision entirely would
    /// still pass every other test in this file.
    #[test]
    fn a_terminal_with_no_flags_is_the_interactive_arm() {
        assert_eq!(answerer(false, false, ATTACHED), Answerer::Screen);
    }

    /// Every clause of the run summary is conditional on its own count, and each wording is
    /// what a user copies into an issue report -- including the denied-guess line, whose only
    /// producer is the full-screen reject answer no scripted answerer can give.
    #[test]
    fn the_summary_says_what_a_run_denied_and_stays_quiet_when_it_denied_nothing() {
        let mut out = Vec::new();
        super::run_summary(
            &mut out,
            &EnrollReport {
                named: 1,
                skipped: 2,
                refused: 1,
                denied: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(
            text.contains("\n1 named, 2 skipped, 0 session(s) passed over"),
            "{text}"
        );
        assert!(text.contains("1 answer(s) refused"), "{text}");
        assert!(text.contains("3 guess(es) denied"), "{text}");

        let mut quiet = Vec::new();
        super::run_summary(&mut quiet, &EnrollReport::default()).unwrap();
        let quiet = String::from_utf8(quiet).unwrap();
        assert!(!quiet.contains("denied"), "{quiet}");
        assert!(!quiet.contains("refused"), "{quiet}");
    }

    /// The off-macOS calendar seam, pinned at its source: whatever instant is asked about, the
    /// offer is empty and total -- which is why `meeting <id>` on Linux lists nothing and points
    /// at `--clear` instead of failing or guessing. Only this side makes the promise: the macOS
    /// implementation may legitimately find meetings, so the test holds here alone.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn off_macos_the_calendar_offers_nothing_for_any_instant() {
        // Local rather than at the module top: the test is gated out on macOS, where the
        // import would be unused.
        use meethook_enroll::MeetingSource;

        for at in [
            "2026-08-09T05:26:00Z",
            "2026-08-09T00:00:00Z",
            "2030-01-01T12:00:00Z",
        ] {
            let offered = super::Calendar.around(at.parse().unwrap());
            assert!(offered.is_empty(), "{at}");
        }
    }
}
