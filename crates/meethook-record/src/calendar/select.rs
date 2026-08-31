//! The choice policy: pick and order meetings from candidates.
//!
//! No EventKit, no environment, no I/O -- pure arithmetic over start/end/all-day/declined,
//! which is what lets the tests below run on a machine with no calendar grant at all.

use std::cmp::Ordering;

use jiff::{SignedDuration, Timestamp};
use meethook_session::{Meeting, MeetingFit};

/// How far outside a meeting a session may start and still be attributed to it, and how far
/// *inside* the start it may begin and still be called a clean join.
///
/// Covers three ordinary human cases in one number, so the whole policy is one sentence --
/// "the recording began near the meeting's start" -- rather than two tunables whose
/// relationship to each other nobody could state: joining a call a few minutes early, carrying
/// on recording a conversation that outlived the invite, and starting a minute or two after
/// the hour, which is indistinguishable from starting on it. Past this much of the meeting the
/// start no longer supports the match; see [`MeetingFit::JoinedLate`].
const NEAR_WINDOW: SignedDuration = SignedDuration::from_secs(15 * 60);

/// One event from the calendar, converted to owned Rust, plus the two facts that can
/// disqualify it.
///
/// `all_day` and `declined` are kept beside the meeting rather than folded into it because
/// they are inputs to the choice and not worth writing to `session.json`: by the time a
/// meeting is stored, it has already been chosen.
#[derive(Debug, Clone)]
pub(crate) struct Candidate {
    pub(crate) meeting: Meeting,
    pub(crate) all_day: bool,
    /// The event was cancelled, or the current user declined it. Either way it is not the
    /// meeting this recording is of, even when the times line up perfectly.
    pub(crate) declined: bool,
}

/// Picks the meeting a session starting at `at` belongs to, or none.
///
/// The whole policy, with no configuration knob and no framework in reach:
///
/// 1. An event whose interval contains `at`, shortest first -- so a 30-minute standup beats
///    the two-hour block it was scheduled inside.
/// 2. Otherwise the next event starting within [`NEAR_WINDOW`], nearest first. Joining early
///    is ordinary, and a session that starts just before a meeting is that meeting.
/// 3. Otherwise the most recent event that ended within [`NEAR_WINDOW`], latest first, which
///    is the conversation that ran past its invite.
///
/// All-day events and declined ones are dropped before any of that: an all-day "Conference"
/// block contains every session recorded that day and would otherwise win rule 1 against
/// nothing at all.
///
/// Every comparison ends in a total tie-break, on start and then on event id, because the
/// framework guarantees no order at all for its results -- a rule that resolved ties by
/// input position would pass a test one day and fail it the next.
///
/// The chosen meeting also carries a [`MeetingFit`], which is a *description* of the winning
/// rule and never an input to it. **Which meeting is selected does not depend on the fit, and
/// does not depend on the session's end.** A session running 2:24-3:40 against a 2:00-3:00
/// standup and a 3:30 invite still resolves to the standup it started inside; it is simply
/// marked [`MeetingFit::JoinedLate`] rather than presented as a meeting it certainly was.
pub(crate) fn select(candidates: Vec<Candidate>, at: Timestamp) -> Option<Meeting> {
    let usable: Vec<Candidate> = candidates
        .into_iter()
        .filter(|c| !c.all_day && !c.declined)
        .collect();

    choose(&usable, at).map(|(candidate, fit)| candidate.meeting.clone().with_fit(fit))
}

/// Which of the three rules wins, and what that says about the match.
///
/// Split out of [`select`] so that a stored [`Meeting`] is constructed in exactly one place
/// and cannot escape without a fit having been decided for it.
fn choose(usable: &[Candidate], at: Timestamp) -> Option<(&Candidate, MeetingFit)> {
    // Note that a zero-length event can never satisfy this (`at < end` fails) and falls
    // through to rules 2 and 3, which is what should happen to a bare calendar marker.
    let containing = usable
        .iter()
        .filter(|c| c.meeting.start <= at && at < c.meeting.end)
        .min_by(|a, b| {
            let (a_len, b_len) = (length(a), length(b));
            a_len.cmp(&b_len).then_with(|| tie_break(a, b))
        });
    if let Some(candidate) = containing {
        // The one place the fit carries information the selection did not already have: a
        // session that began within `NEAR_WINDOW` of the start is the meeting, and one that
        // began deep inside it may be a late join or an unrelated call.
        let fit = if at.duration_since(candidate.meeting.start) <= NEAR_WINDOW {
            MeetingFit::Started
        } else {
            MeetingFit::JoinedLate
        };
        return Some((candidate, fit));
    }

    let upcoming = usable
        .iter()
        .filter(|c| c.meeting.start > at && at.duration_until(c.meeting.start) <= NEAR_WINDOW)
        .min_by(|a, b| {
            a.meeting
                .start
                .cmp(&b.meeting.start)
                .then_with(|| tie_break(a, b))
        });
    if let Some(candidate) = upcoming {
        return Some((candidate, MeetingFit::StartedEarly));
    }

    usable
        .iter()
        .filter(|c| c.meeting.end <= at && c.meeting.end.duration_until(at) <= NEAR_WINDOW)
        // Reversed, so that `min_by` -- which returns the *first* of several equal elements,
        // where `max_by` returns the last -- still picks the latest end deterministically.
        .min_by(|a, b| {
            b.meeting
                .end
                .cmp(&a.meeting.end)
                .then_with(|| tie_break(a, b))
        })
        .map(|candidate| (candidate, MeetingFit::AfterEnd))
}

fn length(candidate: &Candidate) -> SignedDuration {
    candidate
        .meeting
        .start
        .duration_until(candidate.meeting.end)
}

fn tie_break(a: &Candidate, b: &Candidate) -> Ordering {
    a.meeting
        .start
        .cmp(&b.meeting.start)
        .then_with(|| a.meeting.event_id.cmp(&b.meeting.event_id))
}

/// Which candidates may be offered, and in what order.
///
/// Drops all-day and declined events -- [`select`]'s own two disqualifications, applied here
/// from the same place so a listing cannot offer something the automatic lookup would never
/// have picked. Everything else survives, including events no rule of `select` would choose:
/// the whole point of a listing is that the rules got it wrong.
///
/// Ordered by [`tie_break`] -- start, then event id -- because `eventsMatchingPredicate:`
/// guarantees no order at all, and a numbered list whose numbering moved between two runs of
/// the same command would be worse than no numbering.
pub(crate) fn offerable(candidates: Vec<Candidate>) -> Vec<Meeting> {
    let mut usable: Vec<Candidate> = candidates
        .into_iter()
        .filter(|c| !c.all_day && !c.declined)
        .collect();
    usable.sort_by(tie_break);
    usable.into_iter().map(|c| c.meeting).collect()
}

/// The policy, decided with no calendar anywhere near it.
///
/// Every case here is built from plain [`Candidate`] values, which is the point of splitting
/// the module: these run on a machine with no calendar grant, no events and no EventKit, and
/// they are the whole of what "which meeting" means.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn at(rfc3339: &str) -> Timestamp {
        rfc3339.parse().expect("a valid timestamp")
    }

    /// The session start every test is written around.
    pub(crate) fn session_start() -> Timestamp {
        at("2026-08-15T10:00:00Z")
    }

    pub(crate) fn candidate(id: &str, start: &str, end: &str) -> Candidate {
        Candidate {
            meeting: Meeting::new(
                id.to_owned(),
                id.to_owned(),
                "Work".to_owned(),
                at(start),
                at(end),
            ),
            all_day: false,
            declined: false,
        }
    }

    fn chosen(candidates: Vec<Candidate>) -> Option<String> {
        select(candidates, session_start()).map(|meeting| meeting.title)
    }

    /// The chosen meeting's title *and* the fit that was recorded for it.
    fn chosen_with_fit(candidates: Vec<Candidate>, at: Timestamp) -> Option<(String, MeetingFit)> {
        select(candidates, at).map(|meeting| (meeting.title, meeting.fit))
    }

    /// Asserts the answer does not depend on the order the framework happened to return the
    /// events in -- which it explicitly does not guarantee. The fit is compared alongside the
    /// title, because a fit that flipped with input order would be no better than a title that
    /// did.
    fn chosen_either_way(candidates: Vec<Candidate>) -> Option<String> {
        let forwards = chosen_with_fit(candidates.clone(), session_start());
        let mut reversed = candidates;
        reversed.reverse();
        assert_eq!(
            forwards,
            chosen_with_fit(reversed, session_start()),
            "the answer depends on input order"
        );
        forwards.map(|(title, _)| title)
    }

    #[test]
    fn no_candidates_is_no_meeting() {
        assert_eq!(chosen(Vec::new()), None);
    }

    #[test]
    fn the_event_containing_the_session_start_wins() {
        let candidates = vec![candidate(
            "standup",
            "2026-08-15T09:55:00Z",
            "2026-08-15T10:25:00Z",
        )];
        assert_eq!(chosen_either_way(candidates), Some("standup".to_owned()));
    }

    #[test]
    fn the_shortest_containing_event_wins() {
        let candidates = vec![
            candidate("block", "2026-08-15T09:00:00Z", "2026-08-15T11:00:00Z"),
            candidate("standup", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z"),
        ];
        assert_eq!(chosen_either_way(candidates), Some("standup".to_owned()));
    }

    /// AC #3: an all-day event overlaps every session recorded that day, so it must never
    /// take one from the meeting that actually contains it.
    #[test]
    fn an_all_day_event_never_beats_the_meeting_containing_the_start() {
        let mut all_day = candidate("conference", "2026-08-15T00:00:00Z", "2026-08-16T00:00:00Z");
        all_day.all_day = true;
        let candidates = vec![
            all_day,
            candidate("standup", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z"),
        ];
        assert_eq!(chosen_either_way(candidates), Some("standup".to_owned()));
    }

    /// And it does not win by default either: no meeting is the right answer for a day-long
    /// block, not "you were in the conference".
    #[test]
    fn an_all_day_event_alone_selects_nothing() {
        let mut all_day = candidate("conference", "2026-08-15T00:00:00Z", "2026-08-16T00:00:00Z");
        all_day.all_day = true;
        assert_eq!(chosen(vec![all_day]), None);
    }

    /// AC #3: a declined event is never selected, checked at each of the three rules so a
    /// later reordering of them cannot reintroduce it.
    #[test]
    fn a_declined_event_is_never_selected() {
        for (rule, start, end) in [
            ("containing", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z"),
            ("upcoming", "2026-08-15T10:05:00Z", "2026-08-15T10:35:00Z"),
            ("recent", "2026-08-15T09:25:00Z", "2026-08-15T09:55:00Z"),
        ] {
            let mut declined = candidate("declined", start, end);
            declined.declined = true;
            assert_eq!(
                chosen(vec![declined]),
                None,
                "selected a declined event by the {rule} rule"
            );
        }
    }

    /// The titles a listing would offer, in the order it would number them.
    fn offers(candidates: Vec<Candidate>) -> Vec<String> {
        offerable(candidates)
            .into_iter()
            .map(|meeting| meeting.title)
            .collect()
    }

    /// A listing exists because `select` got it wrong, so it must keep the candidates `select`
    /// passed over -- the ones it *disqualified* are a different matter, and go.
    #[test]
    fn a_listing_offers_everything_but_the_all_day_and_declined_events() {
        let mut all_day = candidate("conference", "2026-08-15T00:00:00Z", "2026-08-16T00:00:00Z");
        all_day.all_day = true;
        let mut declined = candidate("declined", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z");
        declined.declined = true;
        let candidates = vec![
            candidate("containing", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z"),
            all_day,
            // The one `select` would never choose over the containing event, and precisely the
            // one a user in a double-booked hour is here to pick.
            candidate(
                "also containing",
                "2026-08-15T09:50:00Z",
                "2026-08-15T11:00:00Z",
            ),
            declined,
            candidate("upcoming", "2026-08-15T10:05:00Z", "2026-08-15T10:35:00Z"),
            candidate("ended", "2026-08-15T09:25:00Z", "2026-08-15T09:55:00Z"),
        ];

        assert_eq!(
            offers(candidates),
            vec!["ended", "also containing", "containing", "upcoming"]
        );
    }

    /// The numbering a user types `--event 2` against has to address the same meeting on the
    /// next run, and `eventsMatchingPredicate:` guarantees no order whatsoever.
    #[test]
    fn the_offered_order_does_not_depend_on_the_order_the_framework_returned() {
        let candidates = vec![
            candidate("second", "2026-08-15T10:00:00Z", "2026-08-15T10:30:00Z"),
            candidate("first", "2026-08-15T09:30:00Z", "2026-08-15T10:30:00Z"),
            // Same start as "second": the id breaks the tie, so the order is total rather than
            // merely usually-stable.
            candidate("third", "2026-08-15T10:00:00Z", "2026-08-15T11:00:00Z"),
        ];
        let mut reversed = candidates.clone();
        reversed.reverse();

        assert_eq!(offers(candidates), vec!["first", "second", "third"]);
        assert_eq!(offers(reversed), vec!["first", "second", "third"]);
    }

    #[test]
    fn nothing_on_the_calendar_offers_nothing() {
        assert!(offers(Vec::new()).is_empty());
    }

    /// An offered candidate is not a match: it is one of several a person is about to choose
    /// between, and nothing here has decided anything about it. The fit it eventually carries
    /// is `Confirmed`, and only whoever picks it can put that there.
    #[test]
    fn an_offered_candidate_carries_no_fit_of_its_own() {
        let offered = offerable(vec![candidate(
            "containing",
            "2026-08-15T09:55:00Z",
            "2026-08-15T10:25:00Z",
        )]);
        assert_eq!(offered[0].fit, MeetingFit::Unknown);
    }

    #[test]
    fn a_meeting_about_to_start_beats_one_that_just_ended() {
        let candidates = vec![
            candidate("ended", "2026-08-15T09:25:00Z", "2026-08-15T09:55:00Z"),
            candidate("upcoming", "2026-08-15T10:05:00Z", "2026-08-15T10:35:00Z"),
        ];
        assert_eq!(chosen_either_way(candidates), Some("upcoming".to_owned()));
    }

    #[test]
    fn the_nearest_upcoming_meeting_wins() {
        let candidates = vec![
            candidate("later", "2026-08-15T10:12:00Z", "2026-08-15T10:42:00Z"),
            candidate("sooner", "2026-08-15T10:03:00Z", "2026-08-15T10:33:00Z"),
        ];
        assert_eq!(chosen_either_way(candidates), Some("sooner".to_owned()));
    }

    #[test]
    fn the_most_recently_ended_meeting_wins() {
        let candidates = vec![
            candidate("earlier", "2026-08-15T09:00:00Z", "2026-08-15T09:48:00Z"),
            candidate("later", "2026-08-15T09:20:00Z", "2026-08-15T09:56:00Z"),
        ];
        assert_eq!(chosen_either_way(candidates), Some("later".to_owned()));
    }

    #[test]
    fn nothing_within_a_quarter_hour_is_no_meeting() {
        let candidates = vec![
            candidate("long_over", "2026-08-15T09:00:00Z", "2026-08-15T09:44:00Z"),
            candidate("far_off", "2026-08-15T10:16:00Z", "2026-08-15T10:46:00Z"),
        ];
        assert_eq!(chosen_either_way(candidates), None);
    }

    /// A zero-length event is a marker, not a meeting: it cannot contain the start, so it is
    /// judged by proximity like anything else.
    #[test]
    fn a_zero_length_event_at_the_session_start_is_treated_as_recent() {
        let candidates = vec![candidate(
            "marker",
            "2026-08-15T10:00:00Z",
            "2026-08-15T10:00:00Z",
        )];
        assert_eq!(chosen(candidates), Some("marker".to_owned()));
    }

    // --- the fit ----------------------------------------------------------------------
    //
    // Every outcome below is decided from plain `Candidate` values: no calendar, no grant, no
    // recording, no session length anywhere in reach.

    /// One 10:00-11:00 event, and every fit it can produce, keyed on where the session began.
    ///
    /// The boundaries sit exactly on `NEAR_WINDOW` on both sides of the start, which is where
    /// a change to the policy would show up first.
    #[test]
    fn the_fit_is_decided_by_where_the_session_started() {
        for (start, expected) in [
            // Before the event: matched by proximity, and strong -- joining early is ordinary.
            ("2026-08-15T09:50:00Z", Some(MeetingFit::StartedEarly)),
            ("2026-08-15T09:59:59Z", Some(MeetingFit::StartedEarly)),
            // On the start, and up to `NEAR_WINDOW` inside it.
            ("2026-08-15T10:00:00Z", Some(MeetingFit::Started)),
            ("2026-08-15T10:15:00Z", Some(MeetingFit::Started)),
            // One second past `NEAR_WINDOW`: a late join, or a different call entirely.
            ("2026-08-15T10:15:01Z", Some(MeetingFit::JoinedLate)),
            ("2026-08-15T10:40:00Z", Some(MeetingFit::JoinedLate)),
            // After the event ended, within `NEAR_WINDOW` of its end.
            ("2026-08-15T11:05:00Z", Some(MeetingFit::AfterEnd)),
            // Out of range on either side: no meeting at all, hence no fit.
            ("2026-08-15T09:44:00Z", None),
            ("2026-08-15T11:16:00Z", None),
        ] {
            let candidates = vec![candidate(
                "standup",
                "2026-08-15T10:00:00Z",
                "2026-08-15T11:00:00Z",
            )];
            assert_eq!(
                chosen_with_fit(candidates, at(start)).map(|(_, fit)| fit),
                expected,
                "a session starting at {start}"
            );
        }
    }

    /// The ticket's motivating case: a hop from one call to another inside a booked hour.
    ///
    /// The 2:00 standup is still selected -- the calendar cannot tell a late join from an
    /// incident bridge, and rejecting this would reject the far more common late join too --
    /// but it is no longer stated as though the session had been that meeting all along.
    #[test]
    fn a_session_starting_deep_inside_a_meeting_keeps_it_but_is_marked_uncertain() {
        let standup = || {
            vec![candidate(
                "standup",
                "2026-08-15T14:00:00Z",
                "2026-08-15T15:00:00Z",
            )]
        };

        assert_eq!(
            chosen_with_fit(standup(), at("2026-08-15T14:24:00Z")),
            Some(("standup".to_owned(), MeetingFit::JoinedLate)),
            "the meeting must be kept, and marked"
        );
        assert!(!MeetingFit::JoinedLate.is_strong());

        // The honest late-ish join of the same meeting is unaffected.
        assert_eq!(
            chosen_with_fit(standup(), at("2026-08-15T14:05:00Z")),
            Some(("standup".to_owned(), MeetingFit::Started))
        );
    }

    /// The claim the whole design rests on, asserted where a change would trip over it: the
    /// session's *end* is not an input, so a recording that overran its event -- the most
    /// ordinary outcome there is -- fits exactly as well as one that stopped on time.
    ///
    /// `select` is not even given a session end to consider; this test states that fact so
    /// that any future attempt to introduce one has to delete an assertion that says why not.
    #[test]
    fn a_session_that_overruns_its_meeting_fits_no_worse_for_it() {
        // The same session start against a meeting it covers entirely, a meeting it stops
        // short of, and a meeting it runs hours past -- one input, so one answer.
        for (end, note) in [
            ("2026-08-15T10:05:00Z", "the meeting ended five minutes in"),
            ("2026-08-15T11:00:00Z", "the meeting ran its full hour"),
            (
                "2026-08-15T18:00:00Z",
                "the meeting was booked all afternoon",
            ),
        ] {
            let candidates = vec![candidate("standup", "2026-08-15T10:00:00Z", end)];
            assert_eq!(
                chosen_with_fit(candidates, at("2026-08-15T10:00:00Z")).map(|(_, fit)| fit),
                Some(MeetingFit::Started),
                "{note}"
            );
        }
    }

    /// Selection is unchanged by the fit: the shortest containing event still wins even when
    /// the session started deep inside it and the longer one would have scored better.
    #[test]
    fn the_fit_never_decides_which_meeting_is_selected() {
        let candidates = vec![
            // Starts at the session start -- a `Started` fit, if it were allowed to compete.
            candidate("block", "2026-08-15T10:00:00Z", "2026-08-15T12:00:00Z"),
            // Shorter, so it wins rule 1, even though the session joined it late.
            candidate("standup", "2026-08-15T09:30:00Z", "2026-08-15T11:00:00Z"),
        ];
        assert_eq!(
            chosen_with_fit(candidates, session_start()),
            Some(("standup".to_owned(), MeetingFit::JoinedLate))
        );
    }
}
