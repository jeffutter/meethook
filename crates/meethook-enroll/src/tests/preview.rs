//! dry-run previews of what an answer would write.

use super::*;

/// An [`Interviewer`] that asks what one name would do before deciding what to answer, and
/// keeps every answer it got back.
///
/// The type is the point. It holds no [`EnrolledSpeakers`], no [`Paths`], no session
/// directory and no `&mut` anything -- a [`Voice`] is the whole of what it is handed -- so a
/// test that reads a [`Consequence`] out of it has shown that the seam carries the preview
/// rather than that this module went and computed one.
///
/// It answers the first voice it is shown and skips the rest, because these tests are about
/// one answer landing and the fixture session has two voices.
struct Previewing {
    asking: String,
    answer: Answer,
    previews: Vec<Option<Consequence>>,
}

impl Previewing {
    fn asking(name: &str, answer: Answer) -> Previewing {
        Previewing {
            asking: name.to_string(),
            answer,
            previews: Vec::new(),
        }
    }

    /// What the first voice's preview said, which is the one every test here asserts on.
    fn first(&self) -> &Consequence {
        self.previews[0]
            .as_ref()
            .expect("a name that is not blank has a consequence")
    }
}

impl Interviewer for Previewing {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        self.previews.push(voice.preview.of(&self.asking));
        if self.previews.len() == 1 {
            self.answer.clone()
        } else {
            Answer::Skip
        }
    }
}

fn run_previewing(paths: &Paths, interviewer: &mut Previewing) -> (EnrollReport, String) {
    run_over(
        paths,
        &[],
        None,
        Offer::default(),
        Sessions::default(),
        Enrolment::default(),
        interviewer,
    )
}

/// Acceptance criterion #1, at the strongest available reading of "writes nothing": not
/// "`speakers.json` is unchanged" but "no file under the root changed by one byte", over a
/// run that previewed a name for every voice it was shown and then answered none of them.
#[test]
fn asking_what_a_name_would_do_writes_nothing() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let before = files_under(root.path());

    let mut interviewer = Previewing::asking("Alice", Answer::Skip);
    let (report, output) = run_previewing(&paths, &mut interviewer);

    assert_eq!(report.named, 0, "{output}");
    assert_eq!(report.skipped, 2, "{output}");
    assert_eq!(interviewer.previews.len(), 2, "{output}");
    assert_eq!(
        interviewer.first().stored,
        Some(Stored::Enrolled),
        "{output}"
    );
    assert_eq!(
        files_under(root.path()),
        before,
        "asking what an answer would do may not write one byte: {output}"
    );
}

/// Acceptance criterion #5: the outcome a preview reported is the outcome the write
/// produced. Agreement is structural -- the commit takes the copies the dry run built -- so
/// what this pins is that a later refactor cannot go back to deriving the two separately.
#[test]
fn a_preview_of_an_enrollment_is_what_the_answer_then_writes() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Previewing::asking("Alice", named("Alice"));
    let (report, output) = run_previewing(&paths, &mut interviewer);

    assert_eq!(
        interviewer.first().stored,
        Some(Stored::Enrolled),
        "{output}"
    );
    assert!(!interviewer.first().session_only(), "{output}");
    assert!(output.contains("enrolled Alice"), "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.session_only, 0, "{output}");
    assert_eq!(
        EnrolledSpeakers::read_or_empty(&paths)
            .unwrap()
            .references("Alice"),
        1,
        "{output}"
    );
}

/// The same agreement over the outcome that is easiest to get wrong, because the name still
/// lands while nothing is stored: at the cap, the preview must say so *before* the user
/// commits to a name that will not help recognise anybody next time.
#[test]
fn a_preview_at_the_reference_cap_is_the_session_only_name_that_follows() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let axes = MAX_REFERENCES_PER_SPEAKER + 2;
    let session = make_session(&paths, "20260809-052600");
    // Every voice on its own axis, so nothing Alice holds is within reach of the voice being
    // asked about and the question really is asked.
    with_embeddings(&session, &[axis(axes - 2, axes), axis(axes - 1, axes)]);
    let held: Vec<(&str, Vec<f32>)> = (0..MAX_REFERENCES_PER_SPEAKER)
        .map(|i| ("Alice", axis(i, axes)))
        .collect();
    enrolled(&held, &paths);

    let mut interviewer = Previewing::asking("Alice", named("Alice"));
    let (report, output) = run_previewing(&paths, &mut interviewer);

    assert_eq!(
        interviewer.first().stored,
        Some(Stored::AtCapacity {
            held: MAX_REFERENCES_PER_SPEAKER,
            shortest: None,
        }),
        "{output}"
    );
    assert!(interviewer.first().session_only(), "{output}");
    assert!(
        output.contains(&format!(
            "named Alice in this session only: Alice already holds \
             {MAX_REFERENCES_PER_SPEAKER} reference(s)"
        )),
        "{output}"
    );
    assert_eq!(report.session_only, 1, "{output}");
    assert_eq!(
        EnrolledSpeakers::read_or_empty(&paths)
            .unwrap()
            .references("Alice"),
        MAX_REFERENCES_PER_SPEAKER,
        "{output}"
    );
}

/// The override crosses the seam on the answer, with no interface anywhere in the test.
///
/// [`Previewing`] holds no [`Paths`], no database and no session directory, so a refusal it
/// can read came through [`Voice::preview`] and an answer the library honoured came back
/// through [`Interviewer::identify`]. That is the whole claim: the answerer saw the cost and
/// said to pay it, and the library needed to know nothing about who was asking. Which is why
/// the line prompt and any scripted driver reach the same behaviour as the frame does --
/// nothing about it is decided in the frame.
#[test]
fn an_override_crosses_the_seam_on_the_answer() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Bob", nearly(60.0))], &paths);

    let mut interviewer = Previewing::asking("Alice", named_anyway("Alice"));
    let (report, output) = run_previewing(&paths, &mut interviewer);

    assert_eq!(
        interviewer.first().refused,
        Some(Refusal::Taken {
            voice: "Unknown 2".to_string(),
            losing: "Bob".to_string(),
        }),
        "the answerer has to be able to see the cost, or insisting is uninformed: {output}"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.refused, 0, "{output}");
}
