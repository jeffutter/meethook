//! What a user reads about a session directory that was never finished.
//!
//! One home for these sentences. `transcribe`'s skip line, `enroll`'s pass-over line and the
//! standing `sessions` report all render from here, because three commands describing the same
//! directory in three vocabularies is the defect this module exists to end -- and because a
//! consumer that composes its own sentence about an unfinished session is free to invent a cause
//! it cannot observe. **Consumers do not write this prose.** Call [`interrupted_brief`],
//! [`interrupted_detail`] or [`recording_in_progress`]; if the sentence you need is not there,
//! add it here so the next command gets it too.
//!
//! # The rules these sentences keep
//!
//! - **No cause, ever.** The retired wording said the recorder "crashed mid-session", which is a
//!   story about what happened rather than a fact about the directory: the same bytes appear when
//!   a machine loses power, when a process is killed, and when a call is being captured at this
//!   very second. Say what is on disk and what follows from it. Never *crashed*, *died*, *killed*
//!   or *interrupted*, and never anything that sounds like the user's fault.
//! - **The brief form begins with the literal `no session.json`.** Two assertions outside this
//!   crate already pin that substring, and it is the phrase a user learns to grep for.
//! - **No promised interval.** The recorder rewrites its header every few seconds, and that
//!   number stays in the recorder: quoting it here would give it a second home *and* would be
//!   false for a track whose header still declares zero, where seconds of real audio sit on disk
//!   that no player will find at all. Report the measured gap instead -- [`crate::wav::track`]
//!   hands it over in bytes and milliseconds -- and say that players stop at the declaration.
//! - **Three magnitudes stay distinct:** audio kept *and playable*, audio kept *but undeclared*
//!   (quantified, and named as unplayable), and audio that *never reached disk* at all. The last
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

use crate::wav::{TrackEvidence, TrackGap, Unfinished};

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
                "holds {} past the end its header declares; players stop at the declaration, so \
                 that part does not play",
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
            "The speaker track holds 5.0 s past the end its header declares; players stop at the \
             declaration, so that part does not play."
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
}
