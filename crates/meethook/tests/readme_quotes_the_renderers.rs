//! The README's two rendered samples, compared against what the binary prints.
//!
//! README is where most users meet these sentences, and it holds two of them as verbatim samples:
//! the `meethook sessions` listing and one `transcribe` skip line. Both are transcriptions of real
//! output, which means both are a copy that nothing checks. This file makes them quotations rather
//! than transcriptions by running the commands over fixtures built to match, and diffing line for
//! line.
//!
//! Two rules follow from that being the whole point. The test never rewrites README -- a green run
//! means the prose still quotes the code, and a red one is a drift report, not something to
//! auto-repair into whichever direction happens to be convenient. And the comparison is made
//! against what *that* run printed: the shapes that produce these two sentences are close to each
//! other but not interchangeable (`common::mixed_root`'s orphan has a mic short of its declaration
//! as well as a speaker track past its own, so it renders a longer skip line), so an expectation
//! carried over from the wrong fixture would fail while both surfaces stayed in agreement.
//!
//! README is pulled in with `include_str!` rather than opened by path, so moving the file breaks
//! the build loudly instead of turning these into tests that quietly read some other README. Each
//! sample is then located by an anchor string verified unique in the document; a missing anchor
//! panics naming itself, which is the failure wanted when somebody edits around a sample.

mod common;

use std::path::Path;
use std::process::{Command, Stdio};

use common::{clip, mixed_root};
use meethook_session::{Paths, SessionId};

/// The whole document, compiled in.
const README: &str = include_str!("../../../README.md");

/// The directory `sessions` reports in, spelled the way README spells it.
///
/// A sample block quoting `/Users/you/meethook/sessions` cannot be reproduced byte for byte from a
/// tempdir, and rewriting README to say `<temp>` would make it quote no one. Substituting the
/// produced path with README's placeholder keeps the header line in the comparison instead of
/// excusing the one line that carries the census AC#5 asks be checked.
const README_ROOT: &str = "/Users/you/meethook/sessions";

/// The bytes of README's single elision line: four spaces and one U+2026, not four spaces and
/// three full stops. Verified with `od -c`, because the two look identical here and differ in the
/// file.
const ELISION: &str = "    \u{2026}";

/// README's `sessions` sample block, between its fences, anchored on its census header.
fn readme_sessions_block() -> Vec<&'static str> {
    let start = README
        .find("4 session(s) in ")
        .expect("README's sessions sample anchor `4 session(s) in ` is gone");
    let body = &README[start..];
    let end = body
        .find("\n```")
        .expect("README's sessions sample has no closing fence");
    let block = &body[..end];
    let lines: Vec<&str> = block.lines().collect();
    assert!(
        lines.len() > 3,
        "the sessions sample collapsed to {} lines",
        lines.len()
    );
    lines
}

/// Run the built binary over `root`.
fn run(root: &Path, args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_meethook"))
        .arg("--root")
        .arg(root)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("running the built meethook");
    assert!(
        output.status.success(),
        "meethook {} failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// AC#5: the `meethook sessions` block in README is what the binary prints over the four-shape
/// root, line for line -- including the census header, modulo the path placeholder, and excepting
/// exactly one elided row.
#[test]
fn the_sessions_sample_is_what_the_binary_prints() {
    let dir = tempfile::tempdir().unwrap();
    let paths = mixed_root(dir.path());

    let produced = run(dir.path(), &["sessions"])
        .replace(&paths.sessions_dir().display().to_string(), README_ROOT);
    let produced_lines: Vec<&str> = produced.trim_end_matches('\n').lines().collect();
    let readme_lines = readme_sessions_block();

    // One README line stands for five produced ones, so the totals have to account for exactly
    // that. Stated before the loop so a fixture that grew a sixth shape fails here rather than as
    // a confusing off-by-N inside the walk.
    assert_eq!(
        produced_lines.len(),
        readme_lines.len() + 4,
        "README elides one row of five; produced {} lines against README's {}: {:#?}",
        produced_lines.len(),
        readme_lines.len(),
        produced_lines
    );

    let mut produced_at = 0;
    let mut elisions = 0;
    for (number, line) in readme_lines.iter().enumerate() {
        if *line == ELISION {
            elisions += 1;
            // What the ellipsis stands for, asserted rather than assumed: the five indented lines
            // of the second orphan's detail, following the id row already matched above it.
            let collapsed = &produced_lines[produced_at..produced_at + 5];
            assert!(
                collapsed.iter().all(|l| l.starts_with("    ")),
                "the elision covers something other than an indented detail block: {collapsed:?}"
            );
            assert_eq!(collapsed.len(), 5);
            produced_at += 5;
            continue;
        }
        assert!(
            produced_at < produced_lines.len(),
            "README's sample is longer than what the command printed"
        );
        assert_eq!(
            produced_lines[produced_at],
            *line,
            "README line {} of the sessions sample no longer quotes the binary\n  README:   \
             {line}\n  printed:  {}\n  (README line {} of the block)",
            number + 1,
            produced_lines[produced_at],
            number + 1
        );
        produced_at += 1;
    }

    assert_eq!(
        elisions, 1,
        "expected exactly one `{ELISION}` line in the sessions sample"
    );
    assert_eq!(
        produced_at,
        produced_lines.len(),
        "README's sample stopped short of the end of what the command printed"
    );
}

/// AC#5: README's skip line is what `transcribe` prints for the one orphan shape it describes.
///
/// That shape is not `mixed_root`'s orphan. README's line names only the speaker track, which
/// means the mic reached disk whole -- so this fixture writes the mic complete and appends 275 200
/// bytes past the speaker track's declaration, the amount that renders as the `4.3 s` README
/// quotes. Reusing the shared root here would print `, and the mic track declares 0.2 s more than
/// it holds` into the middle of the sentence and fail for a reason that has nothing to do with
/// either renderer. It stays a local builder for the same reason: only this test wants a fifth
/// shape, and folding it into `mixed_root` would churn the golden listing every other target pins.
#[test]
fn the_skip_line_is_what_the_binary_prints() {
    const ID: &str = "20260818-143027";

    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::new(dir.path());
    std::fs::create_dir_all(paths.sessions_dir()).unwrap();
    let session = paths.session(&SessionId::parse(ID).unwrap());
    std::fs::create_dir_all(session.dir()).unwrap();
    clip(&session.mic_wav(), 1.0);
    clip(&session.speaker_wav(), 1.0);
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(session.speaker_wav())
        .unwrap()
        .write_all(&[0u8; 275_200])
        .unwrap();

    let stdout = run(dir.path(), &["transcribe", ID]);
    let printed = stdout
        .lines()
        .find(|line| line.starts_with(&format!("{ID}  skipped: ")))
        .unwrap_or_else(|| panic!("transcribe printed no skip line for {ID}:\n{stdout}"));

    let anchor = format!("{ID}  skipped: ");
    let start = README
        .find(&anchor)
        .expect("README's skip-line sample anchor is gone");
    let quoted = README[start..]
        .lines()
        .next()
        .expect("anchor inside README");

    assert_eq!(
        printed, quoted,
        "README's skip line no longer quotes what `transcribe` prints"
    );
    // Belt for the anchor: the README line must be the whole sample, not a prefix of a longer
    // paragraph that happens to open the same way.
    assert!(
        !quoted.contains("Nothing to transcribe."),
        "the README sample swallowed more than one line: {quoted}"
    );
}
