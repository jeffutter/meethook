//! What a user reads about a session directory that was never finished.
//!
//! One home for these sentences. `transcribe`'s skip line, `enroll`'s pass-over line and the
//! standing `sessions` report all render from here, because three commands describing the same
//! directory in three vocabularies is the defect this module exists to end -- and because a
//! consumer that composes its own sentence about an unfinished session is free to invent a cause
//! it cannot observe. **Consumers do not write this prose.** Call [`unfinished_now`] and print
//! what it says; reach for [`interrupted_brief`], [`interrupted_detail`] or
//! [`recording_in_progress`] directly only from inside this module's own chooser. The one
//! exception is [`RootNow::note`], which is the sanctioned way to print the hedge once above a
//! list of directories -- reaching for [`recording_in_progress`] yourself is still an
//! inside-this-module affair, because a caller that has the sentence but not the probe is a
//! caller that can print it unhedged. If the sentence you need is not here, add it here so the
//! next command gets it too.
//!
//! **Consumers do not branch on [`crate::LockState`] either.** Which of the two sentences is true
//! depends on whether a recorder is live *now*, and that question has one answer per directory:
//! [`unfinished_now`] asks the kernel and hands back the wording the answer licenses, so a
//! command cannot print the interruption sentence without having earned it. A command that
//! re-implements that branch is one forgotten probe away from lying about a call in progress.
//! Two shapes ask the question, and both ask it here: a batch line passing one session over
//! calls [`unfinished_now`], and a report that lists many directories calls [`RootNow::ask`] once
//! and [`RootNow::detail_for`] per directory. Neither ever sees [`crate::LockState`]; neither
//! chooses the wording.
//!
//! # The rules these sentences keep
//!
//! - **No cause, ever.** The retired wording said the recorder had crashed partway through a
//!   session, which is a story about what happened rather than a fact about the directory: the
//!   same bytes appear when
//!   a machine loses power, when a process is killed, and when a call is being captured at this
//!   very second. Say what is on disk and what follows from it. Never *crashed*, *died*, *killed*
//!   or *interrupted*, and never anything that sounds like the user's fault.
//! - **The brief form begins with the literal `no session.json`.** Two assertions outside this
//!   crate already pin that substring, and it is the phrase a user learns to grep for.
//! - **No promised interval.** The recorder rewrites its header every few seconds, and that
//!   number stays in the recorder: quoting it here would give it a second home *and* would be
//!   false for a track whose header still declares zero, where seconds of real audio sit on disk
//!   undeclared. Report the measured gap instead -- [`crate::wav::track`] hands it over in bytes
//!   and milliseconds. Where a sentence must speak of playback at all, condition it on the reader
//!   trusting that declaration, and never assert what players do in general: measured readers
//!   disagree (doc-008 §3), so a population claim here is one reader away from being false.
//! - **Three magnitudes stay distinct:** audio kept *and playable*, audio kept *but undeclared*
//!   (quantified, and named as undeclared rather than as unplayable -- how much of it a listener
//!   reaches depends on the reader), and audio that *never reached disk* at all. The last
//!   one is unknowable in degree, so it gets no number and no reason -- only which track is
//!   missing. A dead input device leaves exactly one track on disk, and saying "both tracks" then
//!   is the same class of untruth as saying "crashed".
//! - **Per track, never blended.** The two tracks are written by independent threads that
//!   checkpoint independently, so their headers may disagree with each other; reporting both
//!   separately is information, and reporting a sum would hide it.
//! - **No repair is advertised.** Nothing here offers to rebuild a header or recover a track: the
//!   design record refuses reopening finalized files, and naming a capability the tool does not
//!   have manufactures the support request.
//!
//! The long form carries the tone a standing report needs -- an unfinished session is an expected
//! shape, not a failure -- without using the word "error" to get there.

use crate::paths::{Paths, SessionPaths};
use crate::record_lock::RecordLock;
use crate::wav::{TrackEvidence, TrackGap, Unfinished};

/// What an unfinished directory means *right now*, which is the question its wording depends on.
///
/// A directory holding WAVs and no `session.json` is what a call in progress looks like as often
/// as it is what an abandoned one left behind, and the two deserve different sentences. Carrying
/// the tracks inside the variant that has them makes it impossible to render the stopped case
/// without having asked, and impossible to render the live case with a measurement attached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnfinishedNow {
    /// A recorder holds the root, or the kernel could not be asked. Say nothing about stopping.
    RecorderMayBeLive,
    /// Nothing holds the root, so these tracks are all the evidence the directory left behind.
    NoRecorderHoldsRoot(Unfinished),
}

/// Ask the kernel before reading the tracks: the answer decides which sentence is true.
///
/// Probed per unfinished directory rather than once per run, because the question is only worth
/// asking where its answer changes a line -- an ordinary root answers `Free` from one `ENOENT`
/// open ([`crate::RecordLock::probe`]) and pays nothing further. A recorder that starts mid-run
/// therefore makes later lines hedge while earlier ones did not: toward saying less, which is the
/// direction worth being wrong in. While a recorder *is* live, the `Held` answer reads the holder
/// metadata and costs up to that read's timeout per directory, which a batch line can afford.
///
/// A run that lists many directories asks [`RootNow::ask`] once instead and gets the same answer
/// per directory from [`RootNow::unfinished_at`], which is what this delegates to.
pub fn unfinished_now(paths: &Paths, session: &SessionPaths) -> UnfinishedNow {
    RootNow::ask(paths).unfinished_at(session)
}

impl UnfinishedNow {
    /// The one-line version: what a command prints while passing a session over.
    pub fn brief(&self) -> String {
        match self {
            UnfinishedNow::RecorderMayBeLive => recording_in_progress(),
            UnfinishedNow::NoRecorderHoldsRoot(tracks) => interrupted_brief(tracks),
        }
    }

    /// The multi-line version: what a standing report prints, one entry per line.
    ///
    /// The live answer carries the hedge alone. A report that lists many unfinished directories
    /// may prefer to say it once per run rather than once per directory -- that is the report's
    /// call about its own shape, not something this function can decide from one directory.
    pub fn detail(&self) -> Vec<String> {
        match self {
            UnfinishedNow::RecorderMayBeLive => vec![recording_in_progress()],
            UnfinishedNow::NoRecorderHoldsRoot(tracks) => interrupted_detail(tracks),
        }
    }
}

/// What the kernel said about a whole root, asked once for a run that lists many directories.
///
/// [`unfinished_now`] asks per directory because a batch line pays nothing unless its own
/// directory is unfinished. A standing report wants the opposite trade: one answer for every
/// directory, and the hedge said once above them all rather than repeated under every row. This
/// is that other shape, and it lives beside the first rather than in the report because the rule
/// it enforces -- nobody asserts that something stopped without having asked -- is the module's,
/// not one caller's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootNow {
    /// A recorder holds the root, or the kernel could not be asked. Nothing below may say that a
    /// directory was left behind.
    RecorderMayBeLive,
    /// Nothing holds the root, so every unfinished directory under it is evidence of a recording
    /// that stopped.
    NoRecorderHoldsRoot,
}

impl RootNow {
    /// Ask the kernel once for the whole root.
    ///
    /// Carries the same asymmetry as [`crate::LockState::recorder_may_be_live`]: an answer we
    /// could not get keeps the report from asserting anything stopped, so a root on a filesystem
    /// that will not answer is reported as possibly-live rather than as settled.
    pub fn ask(paths: &Paths) -> Self {
        if RecordLock::probe(paths).recorder_may_be_live() {
            RootNow::RecorderMayBeLive
        } else {
            RootNow::NoRecorderHoldsRoot
        }
    }

    /// The line a report prints above its entries, or `None` when nothing needs hedging.
    ///
    /// A report that prints this must print [`RootNow::detail_for`] rather than
    /// [`UnfinishedNow::detail`] underneath it, or the hedge arrives twice: once as the header and
    /// once again as the only thing an unfinished row had to say.
    pub fn note(self) -> Option<String> {
        match self {
            RootNow::RecorderMayBeLive => Some(recording_in_progress()),
            RootNow::NoRecorderHoldsRoot => None,
        }
    }

    /// The per-directory answer, given what was already asked of the kernel.
    ///
    /// Byte-identical to what [`unfinished_now`] returns for the same root and directory -- the
    /// two entry points differ in how often they ask, never in what the answer means.
    pub fn unfinished_at(self, session: &SessionPaths) -> UnfinishedNow {
        match self {
            RootNow::RecorderMayBeLive => UnfinishedNow::RecorderMayBeLive,
            RootNow::NoRecorderHoldsRoot => {
                UnfinishedNow::NoRecorderHoldsRoot(crate::wav::unfinished(session))
            }
        }
    }

    /// What this directory adds to a report that has already printed [`RootNow::note`].
    ///
    /// Empty while a recorder may be live: the note already said the only thing that is licensed
    /// about such a directory, and repeating it under every id teaches a reader to skim past the
    /// one sentence that has to be read. Stopped directories get [`interrupted_detail`] unchanged,
    /// so the same facts reach the eye whichever way the probe answered -- only the certainty
    /// moves.
    pub fn detail_for(self, session: &SessionPaths) -> Vec<String> {
        match self.unfinished_at(session) {
            UnfinishedNow::RecorderMayBeLive => Vec::new(),
            UnfinishedNow::NoRecorderHoldsRoot(tracks) => interrupted_detail(&tracks),
        }
    }
}

/// The one-line version: what a command prints while passing a session over.
///
/// Always opens `no session.json: no transcript is possible`, then one clause per track with
/// something to report. A track that agrees with its own header contributes no clause of its own;
/// when both agree, that is worth saying once for the pair.
pub fn interrupted_brief(tracks: &Unfinished) -> String {
    // Both tracks missing is one fact about the directory rather than two facts about two files,
    // and saying it once is also the only wording here that does not read like a list of excuses.
    if matches!(
        (tracks.mic, tracks.speaker),
        (TrackEvidence::Absent, TrackEvidence::Absent)
    ) {
        return "no session.json: no transcript is possible; neither track reached disk"
            .to_string();
    }

    let mut clauses = Vec::new();
    for (name, evidence) in [("mic", tracks.mic), ("speaker", tracks.speaker)] {
        if !matches!(evidence, TrackEvidence::CompleteAsDeclared) {
            let (brief, _) = track_says(name, evidence);
            clauses.push(brief);
        }
    }

    let body = match clauses.as_slice() {
        // Nothing to report means both tracks agree with their own headers, which -- with the
        // empty directory already answered above -- is worth saying once for the pair.
        [] => "both tracks are on disk whole as recorded".to_string(),
        [one] => one.clone(),
        [a, b] => format!("{a}, and {b}"),
        // Unreachable with two tracks; joined rather than indexed past the end.
        rest => rest.join(", and "),
    };

    format!("no session.json: no transcript is possible; {body}")
}

/// The multi-line version: what a standing report prints, one entry per line.
///
/// Opens with the reason a transcript is impossible, stated as something the user can check, then
/// one line per track, then what happens to the audio that did survive. Print the entries in
/// order and add no prose of your own around them.
pub fn interrupted_detail(tracks: &Unfinished) -> Vec<String> {
    let mut lines = vec![
        "no session.json: no transcript is possible.".to_string(),
        "That file held the single clock both tracks share, so neither can be placed on a common \
         timeline however much of either one plays."
            .to_string(),
    ];
    for (name, evidence) in [("mic", tracks.mic), ("speaker", tracks.speaker)] {
        let (_, detail) = track_says(name, evidence);
        lines.push(detail);
    }
    lines.push(
        "Nothing about this needs fixing: the audio that reached disk is kept as recorded.".into(),
    );
    lines
}

/// The hedge a report prints when a recorder holds the root right now.
///
/// A directory holding WAVs and no `session.json` describes a call being captured at this second
/// as often as it describes one that stopped, and only [`crate::RecordLock::probe`] can tell those
/// apart. Print this *instead of* asserting interruption whenever
/// [`crate::LockState::recorder_may_be_live`] is true -- including when the probe could not get an
/// answer, which is why that question leans toward held.
pub fn recording_in_progress() -> String {
    "a recorder holds this root right now: a session directory without session.json may be the \
     call happening now rather than one that stopped"
        .to_string()
}

/// What one track has to say, in both lengths at once.
///
/// Returning the short and long predicates from a single match is the point: the two forms a
/// caller prints side by side must not disagree about what the same header means, and a second
/// `match` for the long form is exactly where that drift would start. The short predicate is
/// lower-case, unpunctuated, and safe to join with "and"; the long one is a full sentence.
fn track_says(name: &str, evidence: TrackEvidence) -> (String, String) {
    let (short, long) = match evidence {
        TrackEvidence::Absent => (
            "never reached disk".to_string(),
            "never reached disk, so there is nothing to place on the timeline".to_string(),
        ),
        TrackEvidence::NotAWav => (
            "is not a readable audio file".to_string(),
            "is not a readable audio file".to_string(),
        ),
        TrackEvidence::Unreadable => (
            "is there and cannot be read".to_string(),
            "is on disk and cannot be read right now".to_string(),
        ),
        TrackEvidence::Unknown => (
            "holds audio its header cannot describe".to_string(),
            "holds audio its own header cannot describe, so how much of it survives is unknown"
                .to_string(),
        ),
        TrackEvidence::HeaderOnly => (
            "holds no audio".to_string(),
            "holds a header and no audio".to_string(),
        ),
        TrackEvidence::CompleteAsDeclared => (
            "is whole as recorded".to_string(),
            "is on disk whole as its own header records, and plays through".to_string(),
        ),
        TrackEvidence::ShortBy(gap) => (
            format!("declares {} more than it holds", human_audio(gap)),
            format!(
                "declares {} more audio than the file holds, and that part is not on disk",
                human_audio(gap)
            ),
        ),
        TrackEvidence::BeyondDeclaration(gap) => (
            format!(
                "holds {} past the end its header declares",
                human_audio(gap)
            ),
            format!(
                "holds {} past the end its header declares; a player that trusts that number \
                 stops there, so that part does not play",
                human_audio(gap)
            ),
        ),
    };
    (
        format!("the {name} track {short}"),
        format!("The {name} track {long}."),
    )
}

/// A gap rendered for a person: decimal seconds under a minute, `h:mm:ss` beyond.
///
/// Takes the whole [`TrackGap`] rather than a millisecond count so a caller cannot print one
/// half of a measurement without the other. Deliberately not [`crate::TranscriptTime`], which
/// parses positions inside a transcript and has nothing to do with durations.
fn human_audio(gap: TrackGap) -> String {
    if gap.millis < 60_000 {
        let tenths = gap.millis / 100;
        return format!("{}.{:01} s", tenths / 10, tenths % 10);
    }
    let seconds = gap.millis / 1_000;
    format!(
        "{}:{:02}:{:02}",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn both(mic: TrackEvidence, speaker: TrackEvidence) -> Unfinished {
        Unfinished { mic, speaker }
    }

    fn gap(millis: u64, bytes: u64) -> TrackGap {
        TrackGap { millis, bytes }
    }

    /// The substring two assertions outside this crate already pin, plus the shape the rest of
    /// these tests assume: one line, one prefix, clauses joined rather than stacked.
    #[test]
    fn the_brief_form_opens_with_the_phrase_consumers_grep_for() {
        let line = interrupted_brief(&both(
            TrackEvidence::CompleteAsDeclared,
            TrackEvidence::CompleteAsDeclared,
        ));
        assert_eq!(
            line,
            "no session.json: no transcript is possible; both tracks are on disk whole as recorded"
        );
        assert!(line.starts_with("no session.json"));
        assert!(!line.contains('\n'), "the brief form is one line");
    }

    /// An empty directory is not reported as "both tracks are on disk", which is what a naive
    /// "no clauses means nothing to say" would produce.
    #[test]
    fn a_directory_with_nothing_in_it_says_so_instead_of_claiming_the_tracks_are_there() {
        assert_eq!(
            interrupted_brief(&both(TrackEvidence::Absent, TrackEvidence::Absent)),
            "no session.json: no transcript is possible; neither track reached disk"
        );
    }

    /// One track and not the other -- the dead-input-device shape -- names only the one side, and
    /// the undeclared-audio shape carries its measurement.
    #[test]
    fn a_missing_track_is_named_without_anything_claimed_about_the_other() {
        assert_eq!(
            interrupted_brief(&both(
                TrackEvidence::Absent,
                TrackEvidence::CompleteAsDeclared
            )),
            "no session.json: no transcript is possible; the mic track never reached disk"
        );
        assert_eq!(
            interrupted_brief(&both(
                TrackEvidence::CompleteAsDeclared,
                TrackEvidence::BeyondDeclaration(gap(4_800, 230_400))
            )),
            "no session.json: no transcript is possible; the speaker track holds 4.8 s past the \
             end its header declares"
        );
    }

    /// Two tracks with two different complaints: both are said, and neither swallows the other.
    #[test]
    fn two_tracks_with_two_complaints_are_reported_separately() {
        assert_eq!(
            interrupted_brief(&both(
                TrackEvidence::HeaderOnly,
                TrackEvidence::ShortBy(gap(250, 4_000))
            )),
            "no session.json: no transcript is possible; the mic track holds no audio, and the \
             speaker track declares 0.2 s more than it holds"
        );
    }

    /// The long form leads with the reason a transcript is impossible -- the shared clock, which
    /// a user verifies by looking for the file -- and ends on what happens to the audio.
    #[test]
    fn the_long_form_explains_the_transcript_before_describing_the_audio() {
        let lines = interrupted_detail(&both(
            TrackEvidence::CompleteAsDeclared,
            TrackEvidence::BeyondDeclaration(gap(5_000, 240_000)),
        ));

        assert!(lines[0].starts_with("no session.json"));
        assert!(
            lines[1].contains("single clock"),
            "the reason comes before the audio: {}",
            lines[1]
        );
        assert_eq!(lines.len(), 5, "reason, clock, one line per track, outcome");
        assert_eq!(
            lines[2],
            "The mic track is on disk whole as its own header records, and plays through."
        );
        assert_eq!(
            lines[3],
            "The speaker track holds 5.0 s past the end its header declares; a player that trusts \
             that number stops there, so that part does not play."
        );
        assert!(lines[4].contains("kept"), "{}", lines[4]);
    }

    /// The two forms are read side by side, so they must not disagree about what a header means:
    /// a measurement quoted in one is the same measurement in the other, and each names the
    /// track it is talking about.
    #[test]
    fn the_two_forms_never_disagree_about_a_track() {
        for evidence in [
            TrackEvidence::Absent,
            TrackEvidence::NotAWav,
            TrackEvidence::Unreadable,
            TrackEvidence::Unknown,
            TrackEvidence::HeaderOnly,
            TrackEvidence::CompleteAsDeclared,
            TrackEvidence::ShortBy(gap(4_800, 76_800)),
            TrackEvidence::BeyondDeclaration(gap(4_800, 76_800)),
        ] {
            let (short, long) = track_says("mic", evidence);
            assert!(short.starts_with("the mic track "), "{short}");
            assert!(long.starts_with("The mic track "), "{long}");
            // Whatever number the brief form prints, the detail prints the same one: a report
            // shows both, and two different measurements of one file is the defect here.
            if let TrackEvidence::ShortBy(gap) | TrackEvidence::BeyondDeclaration(gap) = evidence {
                let measured = human_audio(gap);
                assert!(
                    short.contains(&measured) && long.contains(&measured),
                    "{evidence:?}: {measured} should appear in both forms\n  short: {short}\n  \
                     long: {long}"
                );
            }
        }
    }

    /// Neither form blames anyone, promises anything, or quotes the recorder's cadence.
    #[test]
    fn nothing_in_either_form_blames_anyone_or_promises_a_fix() {
        let samples = [
            TrackEvidence::Absent,
            TrackEvidence::NotAWav,
            TrackEvidence::Unreadable,
            TrackEvidence::Unknown,
            TrackEvidence::HeaderOnly,
            TrackEvidence::CompleteAsDeclared,
            TrackEvidence::ShortBy(gap(1_500, 24_000)),
            TrackEvidence::BeyondDeclaration(gap(7_200, 115_200)),
        ];

        for mic in samples {
            for speaker in samples {
                let text = interrupted_detail(&both(mic, speaker)).join(" ");
                let text = format!("{text} {}", interrupted_brief(&both(mic, speaker)));
                for forbidden in [
                    "crash",
                    "died",
                    "kill",
                    "interrupt",
                    "error",
                    "fail",
                    "corrupt",
                    "recover",
                    "repair",
                    "salvage",
                ] {
                    assert!(
                        !text.to_lowercase().contains(forbidden),
                        "the wording claims something it cannot know ({forbidden}): {text}"
                    );
                }
            }
        }
    }

    /// Sub-minute gaps read as decimal seconds; everything else as a clock, so a wrap-long
    /// recording does not print as 21600 seconds.
    #[test]
    fn durations_render_at_a_size_a_person_can_read() {
        assert_eq!(human_audio(gap(250, 4_000)), "0.2 s");
        assert_eq!(human_audio(gap(4_800, 230_400)), "4.8 s");
        assert_eq!(human_audio(gap(59_999, 1)), "59.9 s");
        assert_eq!(human_audio(gap(60_000, 1)), "0:01:00");
        assert_eq!(human_audio(gap(3_725_000, 1)), "1:02:05");
    }

    /// The hedge is its own sentence, because a caller prints it *instead of* the assertion
    /// above, never alongside it.
    #[test]
    fn the_live_recorder_hedge_does_not_assert_that_anything_stopped() {
        let line = recording_in_progress();
        assert!(line.contains("may be"), "{line}");
        assert!(line.contains("session.json"), "{line}");
    }

    /// The guard against hedging forever: a root that has never recorded must still get the real
    /// sentence, which it does because the probe reads absence as `Free` rather than as silence.
    #[test]
    fn a_root_with_no_lock_file_at_all_is_answered_as_stopped() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let session = paths.session(&crate::SessionId::parse("20260809-052500").unwrap());
        std::fs::create_dir_all(session.dir()).unwrap();

        assert_eq!(
            unfinished_now(&paths, &session),
            UnfinishedNow::NoRecorderHoldsRoot(Unfinished {
                mic: TrackEvidence::Absent,
                speaker: TrackEvidence::Absent,
            })
        );
        assert!(!paths.record_lock().exists(), "asking writes nothing");
    }

    /// Liveness comes from the kernel, not from the file's existence: the same root flips back
    /// and forth as the second descriptor is held and dropped.
    #[test]
    fn a_held_root_reads_as_live_and_a_released_one_reads_as_stopped_again() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let session = paths.session(&crate::SessionId::parse("20260809-052500").unwrap());
        std::fs::create_dir_all(session.dir()).unwrap();

        let lock = RecordLock::acquire(&paths).unwrap();
        assert!(
            matches!(lock, crate::Acquisition::Held(_)),
            "a test that silently failed to hold the lock proves nothing"
        );
        assert_eq!(
            unfinished_now(&paths, &session),
            UnfinishedNow::RecorderMayBeLive
        );

        drop(lock);
        assert!(matches!(
            unfinished_now(&paths, &session),
            UnfinishedNow::NoRecorderHoldsRoot(_)
        ));
    }

    /// The live answer must not read like the stopped one in any particular: no leading `no
    /// session.json`, and no claim about what a transcript can or cannot do.
    #[test]
    fn the_brief_under_a_live_recorder_says_neither_of_the_stopped_form_facts() {
        let line = UnfinishedNow::RecorderMayBeLive.brief();
        assert!(line.contains("may be"), "{line}");
        assert!(!line.starts_with("no session.json"), "{line}");
        assert!(!line.contains("no transcript is possible"), "{line}");
        assert!(!line.contains('\n'), "the brief form is one line");
    }

    /// The chooser adds no vocabulary of its own: with nobody holding the root it is byte-for-byte
    /// the renderer every other caller already reads.
    #[test]
    fn the_brief_when_nothing_holds_the_root_is_the_stopped_form_exactly() {
        let tracks = Unfinished {
            mic: TrackEvidence::CompleteAsDeclared,
            speaker: TrackEvidence::BeyondDeclaration(TrackGap {
                millis: 4_300,
                bytes: 137_600,
            }),
        };
        assert_eq!(
            UnfinishedNow::NoRecorderHoldsRoot(tracks).brief(),
            interrupted_brief(&tracks)
        );
        assert_eq!(
            UnfinishedNow::NoRecorderHoldsRoot(tracks).detail(),
            interrupted_detail(&tracks)
        );
        assert_eq!(
            UnfinishedNow::RecorderMayBeLive.detail(),
            vec![recording_in_progress()]
        );
    }

    // --- the run-level shape, asked once for a whole root --------------------------------------

    /// The report's own entry point must keep the guard against hedging forever: a root that has
    /// never recorded gets the real block, not a note, because absence answers `Free`.
    #[test]
    fn a_root_with_no_lock_answers_the_report_as_stopped_and_offers_no_note() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let session = paths.session(&crate::SessionId::parse("20260809-052500").unwrap());
        std::fs::create_dir_all(session.dir()).unwrap();

        let now = RootNow::ask(&paths);
        assert_eq!(now, RootNow::NoRecorderHoldsRoot);
        assert_eq!(now.note(), None, "nothing to hedge, so nothing hedged");
        assert_eq!(
            now.detail_for(&session),
            interrupted_detail(&Unfinished {
                mic: TrackEvidence::Absent,
                speaker: TrackEvidence::Absent,
            }),
            "the stopped answer is the renderer every other caller already reads"
        );
        assert!(!paths.record_lock().exists(), "asking writes nothing");
    }

    /// And the other direction, in the same test: holding the root turns the block into the note
    /// and releasing it turns the note back into the block. A test that silently failed to hold
    /// the lock would otherwise pass by never seeing the live case at all.
    #[test]
    fn a_held_root_gives_the_report_a_note_and_no_block_and_a_released_one_gives_them_back() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let session = paths.session(&crate::SessionId::parse("20260809-052500").unwrap());
        std::fs::create_dir_all(session.dir()).unwrap();

        let lock = RecordLock::acquire(&paths).unwrap();
        assert!(
            matches!(lock, crate::Acquisition::Held(_)),
            "a test that silently failed to hold the lock proves nothing"
        );
        let live = RootNow::ask(&paths);
        assert_eq!(live, RootNow::RecorderMayBeLive);
        assert_eq!(live.note(), Some(recording_in_progress()));
        assert!(
            live.detail_for(&session).is_empty(),
            "the note is said once, above the rows"
        );

        drop(lock);
        let stopped = RootNow::ask(&paths);
        assert_eq!(stopped, RootNow::NoRecorderHoldsRoot);
        assert!(!stopped.detail_for(&session).is_empty());
    }

    /// The two entry points are one decision asked at two cadences. If they ever drift, the
    /// standing report and the batch line start describing the same directory differently --
    /// which is the defect this whole module exists to end.
    #[test]
    fn asking_once_per_run_and_once_per_directory_agree_on_every_directory() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let session = paths.session(&crate::SessionId::parse("20260809-052500").unwrap());
        std::fs::create_dir_all(session.dir()).unwrap();

        for held in [false, true] {
            let _lock = held.then(|| match RecordLock::acquire(&paths).unwrap() {
                crate::Acquisition::Held(guard) => guard,
                crate::Acquisition::Taken(holder) => {
                    panic!("expected to hold the lock: {holder:?}")
                }
            });

            let per_run = RootNow::ask(&paths).unfinished_at(&session);
            let per_directory = unfinished_now(&paths, &session);
            assert_eq!(per_run.brief(), per_directory.brief(), "held: {held}");
            assert_eq!(per_run.detail(), per_directory.detail(), "held: {held}");
        }
    }

    /// While a recorder may be live the report says one sentence and nothing else: no row adds a
    /// track, a measurement, or an impossibility, whatever those tracks actually look like.
    #[test]
    fn under_a_live_recorder_no_track_state_earns_a_line_of_its_own() {
        for evidence in [
            TrackEvidence::Absent,
            TrackEvidence::NotAWav,
            TrackEvidence::Unreadable,
            TrackEvidence::Unknown,
            TrackEvidence::HeaderOnly,
            TrackEvidence::CompleteAsDeclared,
            TrackEvidence::ShortBy(TrackGap {
                millis: 200,
                bytes: 6_400,
            }),
            TrackEvidence::BeyondDeclaration(TrackGap {
                millis: 100,
                bytes: 3_200,
            }),
        ] {
            assert!(
                RootNow::RecorderMayBeLive
                    .detail_for(&SessionPaths::new("sessions/20260809-052500"))
                    .is_empty(),
                "{evidence:?} still printed a line under a live recorder"
            );
        }

        // And the note alone is what the forbidden-word table already governs, since it is the
        // only prose the report prints in this state.
        let note = RootNow::RecorderMayBeLive.note().unwrap();
        for forbidden in ["crash", "died", "kill", "interrupt", "error", "fail"] {
            assert!(
                !note.to_lowercase().contains(forbidden),
                "the note claims something it cannot know ({forbidden}): {note}"
            );
        }
    }
}
