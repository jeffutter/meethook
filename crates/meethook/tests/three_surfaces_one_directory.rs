//! One unfinished directory, shown by all three shipped surfaces, in the same words.
//!
//! The design record for this family (`decision-001`) chose a *report* over a marker on the
//! strength of there being one home for the wording: `interrupted.rs` renders it, and every
//! command prints what that renderer says. Until now that promise was pinned only from the
//! inside. Each consumer byte-pinned its own copy of one sentence, each fixture a different
//! shape, and `RootNow`'s unit test proved only that two cadences of one probe agree per
//! directory. Every one of those pins could sit green while `sessions`, `transcribe <id>` and
//! `enroll --list` drifted apart on a shape none of them happened to hold.
//!
//! So this file puts a single root in front of the built binary three ways and compares what a
//! user actually sees, across processes, over the four forensic shapes `common::mixed_root`
//! builds. Nothing here needs model weights: `transcribe` reaches them lazily, and a session it
//! passes over as an orphan never asks for one. That is also why the ids are named rather than
//! letting `transcribe` scan the root -- a batch run with no ids would open the models for the
//! `valid` directory and fetch 1.6 GB. [`run`] therefore carries the no-download tripwire, so a
//! regression in the lazy fetch fails on the command that caused it rather than at the end of
//! some other test.
//!
//! The streams differ per surface and are not a defect: `sessions` writes its report to stdout,
//! `transcribe` its skip lines to stdout, and `enroll --list` its narration to stderr because
//! stdout carries only the `--list` document (see `headless.rs`'s "Commentary to stderr, always").
//! Every assertion here therefore searches both, and changes neither.
//!
//! One condition is deliberately absent: no test in this file holds the root's lock. While a
//! recorder is live, the per-directory answer is suppressed and replaced by the once-per-run
//! hedge, which would leave nothing to compare. That side is already proven cross-process by
//! `sessions_report.rs`'s `a_recorder_holding_the_root_elsewhere_is_said_once_and_nothing_is_called_interrupted`.

mod common;

use std::path::Path;
use std::process::{Command, Stdio};

use common::mixed_root;
use meethook_session::wav::{TrackEvidence, unfinished};
use meethook_session::{SessionId, interrupted_brief, interrupted_detail};

/// The two directories `mixed_root` leaves without a `session.json`, oldest first.
const ORPHANS: [&str; 2] = ["20260809-052500", "20260809-052600"];

/// One run of the built binary over `root`, with both streams collected and the model-fetch
/// tripwire applied.
///
/// Piped stdio with stdin closed is enough for these three faces: none of them prompts without a
/// terminal, which is why `sessions_report.rs` can block on `output()` today. The pty driver in
/// `common` is the wrong tool here -- it exists for frames that only draw on a tty.
///
/// Returns `(exit code, stdout, stderr)`. AC#3 lives inside this helper rather than in a test of
/// its own: after every run, the root must still hold no `models/` and stderr must not have said
/// it was fetching anything. A stub weight file is not offered as a shortcut either -- hash
/// verification would reject it, so such a guard would be asserting nothing.
fn run(root: &Path, args: &[&str]) -> (Option<i32>, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_meethook"))
        .arg("--root")
        .arg(root)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("running the built meethook");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let called = format!("meethook {}", args.join(" "));

    assert!(
        !stderr.contains("Fetching "),
        "{called} started a model download -- the bug AC#3 exists to catch. stderr:\n{stderr}"
    );
    assert!(
        !root.join("models").exists(),
        "{called} created <root>/models, so it reached the weights. stderr:\n{stderr}"
    );

    (output.status.code(), stdout, stderr)
}

/// The body of a `<id>  <verb>: <body>` line, keyed by id.
///
/// The prefix is what makes the comparison meaningful: each surface owns its verb (`skipped`,
/// `passed over`), and only the body is shared. A line that does not carry the expected prefix is
/// a surface rewording itself, which is reported rather than papered over.
fn bodies(text: &str, verb: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, "  ");
            let id = parts.next()?;
            let rest = parts.next()?.strip_prefix(verb)?;
            Some((id.to_string(), rest.strip_prefix(": ")?.to_string()))
        })
        .collect()
}

/// AC#1 and AC#2: through the binary, `transcribe` and `enroll --list` pass over the same two
/// orphans in byte-identical words, and `sessions` has read every directory in the root.
#[test]
fn transcribe_and_enroll_pass_over_the_same_orphan_in_the_same_words() {
    let dir = tempfile::tempdir().unwrap();
    mixed_root(dir.path());

    // Named ids on both batch commands, so neither opens the models for the `valid` directory.
    let (transcribe_code, transcribe_out, transcribe_err) =
        run(dir.path(), &["transcribe", ORPHANS[0], ORPHANS[1]]);
    let (enroll_code, enroll_out, enroll_err) =
        run(dir.path(), &["enroll", "--list", ORPHANS[0], ORPHANS[1]]);
    let (sessions_code, sessions_out, _) = run(dir.path(), &["sessions"]);

    assert_eq!(
        transcribe_code,
        Some(0),
        "transcribe stderr:\n{transcribe_err}"
    );
    assert_eq!(enroll_code, Some(0), "enroll stderr:\n{enroll_err}");
    assert_eq!(sessions_code, Some(0));

    let skipped = bodies(&transcribe_out, "skipped");
    let passed = bodies(&enroll_err, "passed over");
    assert_eq!(
        skipped.len(),
        ORPHANS.len(),
        "expected one skip line per orphan; stdout was:\n{transcribe_out}"
    );
    assert_eq!(
        passed.len(),
        ORPHANS.len(),
        "expected one pass-over line per orphan; stderr was:\n{enroll_err}"
    );

    for id in ORPHANS {
        let t = skipped
            .iter()
            .find(|(seen, _)| seen == id)
            .unwrap_or_else(|| panic!("transcribe did not name {id}:\n{transcribe_out}"));
        let e = passed
            .iter()
            .find(|(seen, _)| seen == id)
            .unwrap_or_else(|| panic!("enroll --list did not name {id}:\n{enroll_err}"));
        assert_eq!(
            t.1, e.1,
            "{id}: the two batch surfaces disagree about the same directory"
        );
        // Neither surface may quietly describe a directory that stopped as one still running.
        assert!(
            !t.1.contains("a recorder holds this root"),
            "{id}: a batch surface printed the liveness hedge with nobody holding the lock"
        );
    }

    // A fixture that silently stopped being discovered would shrink the comparison above to zero
    // rows, and both asserts on length would still pass if only one orphan survived. The census is
    // the fourth surface's testimony that all four shapes really were there.
    assert!(
        sessions_out.starts_with("4 session(s) in "),
        "sessions lost the census header:\n{sessions_out}"
    );
    for id in ORPHANS {
        assert!(
            sessions_out.contains(&format!("{id}  orphaned")),
            "sessions did not list {id} as orphaned:\n{sessions_out}"
        );
    }

    // `enroll --list` keeps stdout pure: the document only, no narration.
    assert!(
        !enroll_out.contains("passed over"),
        "enroll --list narrated on stdout, where only the document belongs:\n{enroll_out}"
    );
}

/// AC#1: what `sessions` prints per directory is `interrupted_detail` for that directory's own
/// files, and the batch bodies are `interrupted_brief` for the same files -- so the two forms a
/// user may see side by side are computed from one reading of the tracks, not retyped twice.
///
/// The pairing is asserted structurally rather than against prose: brief and detail are both
/// rendered here from the `wav::unfinished` answer for the real fixture, exactly as the commands
/// do, and then related to each other the way `interrupted.rs` documents. That keeps the claim
/// true for evidence states this fixture does not contain, instead of pinning one sentence to
/// another. `interrupted.rs`'s own `the_two_forms_never_disagree_about_a_track` covers the unit
/// level; what nothing else covered is that these two CLIs print those two renderings.
#[test]
fn what_each_surface_prints_is_what_the_renderers_say_about_those_exact_files() {
    let dir = tempfile::tempdir().unwrap();
    let paths = mixed_root(dir.path());

    let (_, _, enroll_err) = run(dir.path(), &["enroll", "--list", ORPHANS[0], ORPHANS[1]]);
    let (_, transcribe_out, _) = run(dir.path(), &["transcribe", ORPHANS[0], ORPHANS[1]]);
    let (_, sessions_out, _) = run(dir.path(), &["sessions"]);

    let skipped = bodies(&transcribe_out, "skipped");
    let passed = bodies(&enroll_err, "passed over");

    for id in ORPHANS {
        let session = paths.session(&SessionId::parse(id).unwrap());
        let tracks = unfinished(&session);
        let brief = interrupted_brief(&tracks);
        let detail = interrupted_detail(&tracks);

        let transcribed = skipped
            .iter()
            .find(|(seen, _)| seen == id)
            .map(|(_, body)| body.clone())
            .expect("named above");
        let enrolled = passed
            .iter()
            .find(|(seen, _)| seen == id)
            .map(|(_, body)| body.clone())
            .expect("named above");
        assert_eq!(transcribed, brief, "{id}: transcribe's body is not brief()");
        assert_eq!(enrolled, brief, "{id}: enroll --list's body is not brief()");

        // The report block, unindented back to what the renderer returned. Four spaces is how the
        // command marks these lines as belonging to the id above them; the renderer knows nothing
        // about that indent, so it is stripped rather than added to the expectation.
        let start = sessions_out
            .find(&format!("{id}  orphaned\n"))
            .unwrap_or_else(|| panic!("sessions did not list {id}:\n{sessions_out}"));
        let block = sessions_out[start + id.len() + "  orphaned".len() + 1..]
            .lines()
            .take_while(|line| line.starts_with("    "))
            .map(|line| line.trim_start_matches("    ").to_string())
            .collect::<Vec<_>>();
        assert_eq!(block, detail, "{id}: sessions' block is not detail()");

        // And the two forms relate to each other, mechanically, for these files.
        assert!(
            brief.starts_with("no session.json: no transcript is possible; "),
            "{id}: brief does not open on the reason: {brief}"
        );
        assert_eq!(
            detail[0], "no session.json: no transcript is possible.",
            "{id}: detail opens differently from the brief's stem"
        );
        let both_absent = matches!(
            (tracks.mic, tracks.speaker),
            (TrackEvidence::Absent, TrackEvidence::Absent)
        );
        if both_absent {
            // The one wording that covers the pair, in a form that names neither track -- while the
            // detail still answers for each file separately.
            assert!(brief.ends_with("neither track reached disk"), "{brief}");
            assert_eq!(
                detail.len(),
                5,
                "reason, clock, one line per track, outcome"
            );
            for line in &detail[2..4] {
                assert!(line.contains("never reached disk"), "{line}");
            }
        } else {
            for (name, evidence, line) in [
                ("mic", tracks.mic, detail[2].clone()),
                ("speaker", tracks.speaker, detail[3].clone()),
            ] {
                assert!(line.starts_with(&format!("The {name} track ")), "{line}");
                // A measurement in one form is the same measurement in the other, and a track the
                // detail finds fault with is a track the brief had to mention.
                match evidence {
                    // The three states that print a figure. `NoDeclaredLength` belongs here even
                    // though no fixture makes one yet -- the relation is stated once here so the
                    // compiler keeps finding drift when TASK-067.05.09 puts that state in front of
                    // a surface.
                    TrackEvidence::ShortBy(_)
                    | TrackEvidence::BeyondDeclaration(_)
                    | TrackEvidence::NoDeclaredLength(_) => {
                        // Compare the printed figure rather than recomputing it: the renderer's
                        // rounding is its own business, and what must not drift is the number a
                        // user sees in the two places.
                        let measured = seconds_figure(&line)
                            .unwrap_or_else(|| panic!("{id}: detail reports no figure: {line}"));
                        assert!(
                            brief.contains(name) && brief.contains(measured),
                            "{id}: brief dropped {name}'s {measured}: {brief}"
                        );
                    }
                    TrackEvidence::Absent
                    | TrackEvidence::NotAWav
                    | TrackEvidence::Unreadable
                    | TrackEvidence::Unknown
                    | TrackEvidence::HeaderOnly => assert!(
                        brief.contains(name),
                        "{id}: brief dropped {name}, which the detail reports: {brief}"
                    ),
                    TrackEvidence::CompleteAsDeclared => {
                        assert!(
                            !brief.contains(&format!("the {name} track")),
                            "{id}: brief reports a track the detail calls whole: {brief}"
                        );
                    }
                }
            }
        }
    }
}

/// AC#2: the four shapes are present and remain distinguishable, so parity is proven over real
/// forensic states rather than one toy directory.
#[test]
fn the_four_shapes_are_all_there_to_be_read() {
    let dir = tempfile::tempdir().unwrap();
    let paths = mixed_root(dir.path());

    let (_, out, _) = run(dir.path(), &["sessions"]);
    assert!(
        out.starts_with(&format!(
            "4 session(s) in {}: 1 transcribed, 1 valid, 2 orphaned\n",
            paths.sessions_dir().display()
        )),
        "the census itself changed: {out}"
    );
    for (id, kind) in [
        ("20260809-052500", "orphaned"),
        ("20260809-052600", "orphaned"),
        ("20260809-052700", "transcribed"),
        ("20260809-052800", "valid"),
    ] {
        assert!(
            out.contains(&format!("{id}  {kind}")),
            "{id} is not {kind}:\n{out}"
        );
    }

    // The two orphans say different things -- one has tracks with measurements, one has no tracks
    // at all -- which is what makes the byte-equality in the parity test more than one comparison.
    let short = unfinished(&paths.session(&SessionId::parse(ORPHANS[0]).unwrap()));
    let empty = unfinished(&paths.session(&SessionId::parse(ORPHANS[1]).unwrap()));
    assert_eq!(
        short.mic,
        TrackEvidence::ShortBy(meethook_session::wav::TrackSpan {
            bytes: 12_800,
            millis: 200,
        }),
        "the fixture's mic is no longer 0.2 s short of its declaration"
    );
    assert!(matches!(short.speaker, TrackEvidence::BeyondDeclaration(_)));
    assert_eq!(empty.mic, TrackEvidence::Absent);
    assert_eq!(empty.speaker, TrackEvidence::Absent);
    assert_ne!(
        interrupted_brief(&short),
        interrupted_brief(&empty),
        "two shapes that render the same sentence prove nothing together"
    );

    // And the batch runs named exactly these two, no more.
    let (_, transcribe_out, _) = run(dir.path(), &["transcribe", ORPHANS[0], ORPHANS[1]]);
    assert_eq!(
        bodies(&transcribe_out, "skipped")
            .into_iter()
            .map(|(id, _)| id)
            .collect::<Vec<_>>(),
        ORPHANS.to_vec(),
        "the batch run skipped something other than the two orphans"
    );
}

/// The first `"<digits>[.<digits>] s"` figure in a rendered sentence.
///
/// Scanned rather than parsed: the tests only ever need the one figure a track line carries, and
/// pulling it out of the printed string is what keeps the comparison about what the user reads
/// instead of about `TrackSpan`'s rounding, which is the renderer's own decision.
fn seconds_figure(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    let mut start = None;
    for (i, byte) in bytes.iter().enumerate() {
        let digit = byte.is_ascii_digit() || *byte == b'.';
        match (start, digit) {
            (None, true) => start = Some(i),
            (Some(_), false) => {
                let end = i;
                if let Some(s) = start
                    && bytes[end..].starts_with(b" s")
                    && bytes[s..end].iter().any(u8::is_ascii_digit)
                {
                    return Some(&line[s..end]);
                }
                start = None;
            }
            _ => {}
        }
    }
    None
}
