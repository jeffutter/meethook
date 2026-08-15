//! The calendar meeting a session was recorded during.
//!
//! A finished session otherwise records *when* it happened and nothing about *what* it was,
//! so finding "the incident review last Tuesday" means opening transcripts until one looks
//! right. macOS already knows the answer: EventKit reads the same store Calendar.app does,
//! including any Exchange or Google accounts synced into it, and a session's start time is
//! enough to pick the meeting it belongs to.
//!
//! Three things about how that is done here are load-bearing.
//!
//! **The query runs at finish, against the stored start time.** `record` auto-starts on mic
//! activity, so anything slow on the start path is lost audio -- and the first `EKEventStore`
//! access can wake the calendar daemon. Asking at the end for the meeting nearest to the
//! recorded start answers the same question off the hot path. The cost is that an event
//! edited during the meeting resolves to its edited form, which is the better of the two
//! errors.
//!
//! **Nothing here can prompt, and nothing here can be killed.** Only
//! `authorizationStatusForEntityType:` is called, which is a pure read: no prompt, no daemon
//! wake, no Info.plist requirement. Both of EventKit's *request* APIs are hazardous in ways
//! that would land inside [`crate::RunningSession::finish`], between the last audio buffer
//! and the `session.json` write: `requestFullAccessToEventsWithCompletion:` requires
//! `NSCalendarsFullAccessUsageDescription` in the responsible process's Info.plist on macOS
//! 14+ and *terminates the process* when the key is absent, which most terminal emulators do
//! not ship. A termination there would convert a complete recording into an orphaned
//! directory. So the grant is read, never asked for; producing it is a separate job.
//!
//! **Failure is never fatal.** A missing permission, no match, an Objective-C raise, or a
//! panic out of a binding all degrade to `None` and a finished recording. Losing a recording
//! because the calendar was unreachable would invert this crate's whole priority ordering,
//! and unlike the microphone and screen grants there is nothing here worth stopping a
//! meeting for -- which is also why calendar access is not part of [`crate::preflight()`].
//!
//! The framework half returns candidates and the safe half picks between them.
//! `eventsMatchingPredicate:` documents that it returns events in no guaranteed order, and
//! the choice policy is pure arithmetic over start/end/all-day/declined -- so splitting there
//! is what makes the policy testable on a machine with no calendar at all.

use std::cmp::Ordering;
use std::fmt::Write as _;
use std::panic::AssertUnwindSafe;

use jiff::{SignedDuration, Timestamp};
use meethook_session::{Attendee, AttendeeStatus, Meeting};
use objc2_event_kit::{
    EKAuthorizationStatus, EKEntityType, EKEvent, EKEventStatus, EKEventStore, EKParticipant,
    EKParticipantStatus,
};
use objc2_foundation::NSDate;

/// How far outside a meeting a session may start and still be attributed to it.
///
/// Covers the two ordinary human cases in one number: joining a call a few minutes early,
/// and carrying on recording a conversation that outlived the invite.
const NEAR_WINDOW: SignedDuration = SignedDuration::from_secs(15 * 60);

/// How much calendar to fetch on either side of the session start.
///
/// Wider than [`NEAR_WINDOW`] on purpose, and it does not need to be exact: the predicate
/// returns events *overlapping* the range, so a two-hour meeting that began 50 minutes
/// before the session must still be inside it to be found. Policy lives in [`select`], not
/// in the query, so making this generous costs a few discarded candidates and nothing else.
const QUERY_WINDOW: SignedDuration = SignedDuration::from_secs(60 * 60);

/// One event from the calendar, converted to owned Rust, plus the two facts that can
/// disqualify it.
///
/// `all_day` and `declined` are kept beside the meeting rather than folded into it because
/// they are inputs to the choice and not worth writing to `session.json`: by the time a
/// meeting is stored, it has already been chosen.
#[derive(Debug, Clone)]
struct Candidate {
    meeting: Meeting,
    all_day: bool,
    /// The event was cancelled, or the current user declined it. Either way it is not the
    /// meeting this recording is of, even when the times line up perfectly.
    declined: bool,
}

/// The meeting `at` fell within, if the calendar can be read and one matches.
///
/// Total by construction -- no `Result`, no error type -- because there is no caller
/// decision to make. A missing grant, an empty calendar, a raise and a panic are all the
/// same outcome to the one function whose failure would cost a recording, so the branch
/// belongs here rather than at the call site.
pub(crate) fn meeting_at(at: Timestamp) -> Option<Meeting> {
    // SAFETY: a class-level status read taking only the entity type. It touches no store,
    // no calendar and no user data, and is the one EventKit entry point that cannot prompt.
    let status = unsafe { EKEventStore::authorizationStatusForEntityType(EKEntityType::Event) };
    if status != EKAuthorizationStatus::FullAccess {
        debug(&format!(
            "calendar access is not granted (status {}); no meeting will be recorded",
            status.0
        ));
        return None;
    }

    // SAFETY: every call inside is a read against a freshly created store, made from the
    // finishing thread with both capture engines already stopped. The store, the dates and
    // the predicate all outlive the query, and every event is converted to owned Rust before
    // any of them is dropped.
    let candidates = caught(|| unsafe { candidates_around(at) })?;
    if debugging() {
        for candidate in &candidates {
            debug(&summarize(candidate));
        }
    }

    let chosen = select(candidates, at);
    debug(&match chosen.as_ref() {
        Some(meeting) => format!("selected {:?}", meeting.title),
        None => "no candidate matched the session start".to_owned(),
    });
    chosen
}

/// Runs the EventKit lookup with both ways out of it blocked.
///
/// Two nets, because there are two ways a framework call can leave without returning.
/// [`crate::exception::catching`] turns an Objective-C raise into an error -- an uncaught one
/// aborts the process outright, as that module records. It deliberately does *not* catch Rust
/// panics, which is the other way out: the bindings return non-optional `Retained<T>` for
/// several properties Apple declares nonnull, and each of those panics rather than returning
/// if the framework ever hands back nil. A panic escaping here would unwind out of `finish`
/// and lose a recording over a missing event title, so it is caught and reported instead --
/// the panic message still reaches stderr, so the bug stays visible without being fatal.
///
/// Unwind safety is asserted for the same reason [`crate::exception::catching`] does: on
/// either failure path every object the closure touched is dropped unused, and nothing it
/// wrote is observed afterwards.
///
/// The lookup is a parameter rather than the body so that both failure paths are decidable
/// in `cargo test`: a claim that a raise costs a field instead of a recording is worth
/// nothing until something has actually raised here.
fn caught(lookup: impl FnOnce() -> Vec<Candidate>) -> Option<Vec<Candidate>> {
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        crate::exception::catching("EKEventStore.eventsMatchingPredicate", lookup)
    }));

    match outcome {
        Ok(Ok(candidates)) => Some(candidates),
        Ok(Err(raise)) => {
            debug(&format!("{raise}"));
            None
        }
        Err(_) => {
            debug("the EventKit lookup panicked; continuing without a meeting");
            None
        }
    }
}

/// Every event overlapping [`QUERY_WINDOW`] around `at`, converted to owned Rust.
///
/// # Safety
///
/// Must be called through [`crate::exception::catching`]: `eventsMatchingPredicate:` raises
/// if handed a predicate it did not create, and a raise that reaches Rust aborts.
unsafe fn candidates_around(at: Timestamp) -> Vec<Candidate> {
    unsafe {
        let store = EKEventStore::new();
        let seconds = at.as_duration().as_secs_f64();
        let from = NSDate::dateWithTimeIntervalSince1970(seconds - QUERY_WINDOW.as_secs_f64());
        let to = NSDate::dateWithTimeIntervalSince1970(seconds + QUERY_WINDOW.as_secs_f64());

        // `None` means every calendar the grant covers, which is the point: a meeting can
        // live in any of the accounts synced into the store, and asking about a subset would
        // be a configuration knob this needs no answer for.
        let predicate = store.predicateForEventsWithStartDate_endDate_calendars(&from, &to, None);
        store
            .eventsMatchingPredicate(&predicate)
            .to_vec()
            .iter()
            .filter_map(|event| candidate(event))
            .collect()
    }
}

/// Converts one `EKEvent`, or drops it.
///
/// A candidate whose dates will not convert is dropped rather than defaulted: an event with
/// no usable interval cannot win any of [`select`]'s rules anyway, and inventing a timestamp
/// for it could let it win one it should not.
///
/// # Safety
///
/// As [`candidates_around`].
unsafe fn candidate(event: &EKEvent) -> Option<Candidate> {
    unsafe {
        let start = timestamp(&event.startDate())?;
        let end = timestamp(&event.endDate())?;

        let attendees: Vec<Attendee> = event
            .attendees()
            .map(|list| list.to_vec().iter().map(|p| attendee(p)).collect())
            .unwrap_or_default();

        // A cancelled event and one this user declined are the same thing for this purpose:
        // whatever was being recorded, it was not that. Other participants' answers say
        // nothing -- a meeting everyone else declined still happened if this user was in it.
        let declined = event.status() == EKEventStatus::Canceled
            || attendees
                .iter()
                .any(|a| a.is_you && a.status == AttendeeStatus::Declined);

        Some(Candidate {
            meeting: Meeting {
                title: event.title().to_string(),
                start,
                end,
                calendar: event
                    .calendar()
                    .map(|calendar| calendar.title().to_string())
                    .unwrap_or_default(),
                organizer: event.organizer().map(|organizer| attendee(&organizer)),
                attendees,
                url: event
                    .URL()
                    .and_then(|url| url.absoluteString())
                    .map(|url| url.to_string()),
                // `eventIdentifier` is nil only for an event not yet saved to a store, which
                // one fetched *from* a store cannot be. The fallback is the calendar item's
                // own identifier rather than an empty string so that the field keeps its
                // meaning -- something a later pass can look the event back up by.
                event_id: event.eventIdentifier().map_or_else(
                    || event.calendarItemIdentifier().to_string(),
                    |id| id.to_string(),
                ),
            },
            all_day: event.isAllDay(),
            declined,
        })
    }
}

/// # Safety
///
/// As [`candidates_around`].
unsafe fn attendee(participant: &EKParticipant) -> Attendee {
    unsafe {
        Attendee {
            name: participant
                .name()
                .map(|name| name.to_string())
                .filter(|name| !name.is_empty()),
            // `EKParticipant` has no address property: the address is the participant's URL,
            // a `mailto:` one for everybody who is not a room or a resource. Stripping the
            // scheme is this module's job because `Attendee::email` is documented as bare.
            email: participant
                .URL()
                .absoluteString()
                .and_then(|url| mail_address(&url.to_string())),
            status: attendee_status(participant.participantStatus()),
            is_you: participant.isCurrentUser(),
        }
    }
}

/// The bare address behind a participant URL, or `None` for a participant that is not a
/// mailbox at all (a room, a resource, a phone number).
fn mail_address(url: &str) -> Option<String> {
    let address = match url.get(..7) {
        Some(scheme) if scheme.eq_ignore_ascii_case("mailto:") => &url[7..],
        _ => return None,
    };
    (!address.is_empty()).then(|| address.to_owned())
}

/// Maps EventKit's participant status onto the stored one.
///
/// The two reminder-only members (`Completed`, `InProcess`) and anything a later macOS adds
/// fall to `Unknown` rather than being stored as a number nothing can interpret.
fn attendee_status(status: EKParticipantStatus) -> AttendeeStatus {
    match status {
        EKParticipantStatus::Pending => AttendeeStatus::Pending,
        EKParticipantStatus::Accepted => AttendeeStatus::Accepted,
        EKParticipantStatus::Declined => AttendeeStatus::Declined,
        EKParticipantStatus::Tentative => AttendeeStatus::Tentative,
        EKParticipantStatus::Delegated => AttendeeStatus::Delegated,
        _ => AttendeeStatus::Unknown,
    }
}

/// `NSDate` carries seconds as an `f64`, which is exact to the microsecond for any date this
/// tool will see; a value that will not convert at all (infinite, or centuries out of range)
/// yields `None` so its event can be dropped.
fn timestamp(date: &NSDate) -> Option<Timestamp> {
    SignedDuration::try_from_secs_f64(date.timeIntervalSince1970())
        .ok()
        .and_then(|since_epoch| Timestamp::from_duration(since_epoch).ok())
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
fn select(candidates: Vec<Candidate>, at: Timestamp) -> Option<Meeting> {
    let usable: Vec<Candidate> = candidates
        .into_iter()
        .filter(|c| !c.all_day && !c.declined)
        .collect();

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
        return Some(candidate.meeting.clone());
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
        return Some(candidate.meeting.clone());
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
        .map(|candidate| candidate.meeting.clone())
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

fn debugging() -> bool {
    std::env::var_os("MEETHOOK_CALENDAR_DEBUG").is_some()
}

/// Prints one diagnostic line, gated behind `MEETHOOK_CALENDAR_DEBUG`.
///
/// "Why did my session get no meeting?" is otherwise unanswerable from the outside: the
/// grant, the candidate list and the rule that fired are all invisible in the result.
fn debug(message: &str) {
    if debugging() {
        eprintln!("[calendar] {message}");
    }
}

/// Renders a candidate for [`debug`] with its attendees *counted*, never named.
///
/// Attendee names and addresses go to `session.json` because speaker identification needs
/// them; they do not go to a terminal, a log file, or anywhere a screen share or a pasted
/// error report can carry them. Keeping the rendering in one tested function is what makes
/// that a property of the code rather than a promise -- see the test below.
fn summarize(candidate: &Candidate) -> String {
    let mut line = String::new();
    let _ = write!(
        line,
        "{} .. {}  {:?}  [{}]  {} attendee(s)",
        candidate.meeting.start,
        candidate.meeting.end,
        candidate.meeting.title,
        candidate.meeting.calendar,
        candidate.meeting.attendees.len(),
    );
    if candidate.all_day {
        line.push_str("  all-day");
    }
    if candidate.declined {
        line.push_str("  declined");
    }
    line
}

/// The policy, decided with no calendar anywhere near it.
///
/// Every case here is built from plain [`Candidate`] values, which is the point of splitting
/// the module: these run on a machine with no calendar grant, no events and no EventKit, and
/// they are the whole of what "which meeting" means.
#[cfg(test)]
mod tests {
    use objc2_event_kit::EKCalendar;
    use objc2_foundation::NSString;

    use super::*;

    fn at(rfc3339: &str) -> Timestamp {
        rfc3339.parse().expect("a valid timestamp")
    }

    /// The session start every test is written around.
    fn session_start() -> Timestamp {
        at("2026-08-15T10:00:00Z")
    }

    fn candidate(id: &str, start: &str, end: &str) -> Candidate {
        Candidate {
            meeting: Meeting {
                title: id.to_owned(),
                start: at(start),
                end: at(end),
                calendar: "Work".to_owned(),
                organizer: None,
                attendees: Vec::new(),
                url: None,
                event_id: id.to_owned(),
            },
            all_day: false,
            declined: false,
        }
    }

    fn chosen(candidates: Vec<Candidate>) -> Option<String> {
        select(candidates, session_start()).map(|meeting| meeting.title)
    }

    /// Asserts the answer does not depend on the order the framework happened to return the
    /// events in -- which it explicitly does not guarantee.
    fn chosen_either_way(candidates: Vec<Candidate>) -> Option<String> {
        let forwards = chosen(candidates.clone());
        let mut reversed = candidates;
        reversed.reverse();
        assert_eq!(
            forwards,
            chosen(reversed),
            "the answer depends on input order"
        );
        forwards
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

    /// AC #5, in the only place a meeting is ever rendered for a human: attendee names and
    /// addresses reach `session.json` and nothing else.
    #[test]
    fn the_debug_line_counts_attendees_without_naming_them() {
        let mut candidate = candidate("standup", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z");
        candidate.meeting.attendees = vec![
            Attendee {
                name: Some("Grace Hopper".to_owned()),
                email: Some("grace@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: false,
            },
            Attendee {
                name: Some("Ada Lovelace".to_owned()),
                email: Some("ada@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: true,
            },
        ];
        candidate.meeting.organizer = Some(Attendee {
            name: Some("Alan Turing".to_owned()),
            email: Some("alan@example.com".to_owned()),
            status: AttendeeStatus::Accepted,
            is_you: false,
        });

        let line = summarize(&candidate);

        assert!(
            line.contains("standup"),
            "the title is what makes the line useful: {line}"
        );
        assert!(
            line.contains("2 attendee(s)"),
            "the count is missing: {line}"
        );
        for secret in [
            "Grace",
            "Hopper",
            "grace@example.com",
            "Ada",
            "Lovelace",
            "ada@example.com",
            "Alan",
            "Turing",
            "alan@example.com",
            "@",
        ] {
            assert!(
                !line.contains(secret),
                "the debug line leaks {secret:?}: {line}"
            );
        }
    }

    #[test]
    fn a_participant_url_yields_a_bare_address() {
        assert_eq!(
            mail_address("mailto:grace@example.com"),
            Some("grace@example.com".to_owned())
        );
        // Exchange and Google both spell the scheme this way at times.
        assert_eq!(
            mail_address("MAILTO:grace@example.com"),
            Some("grace@example.com".to_owned())
        );
        // A room, a resource or a phone participant is not a mailbox and gets no address.
        assert_eq!(mail_address("tel:+15551234567"), None);
        assert_eq!(mail_address("mailto:"), None);
        assert_eq!(mail_address(""), None);
    }

    #[test]
    fn the_reminder_only_participant_statuses_read_as_unknown() {
        assert_eq!(
            attendee_status(EKParticipantStatus::Accepted),
            AttendeeStatus::Accepted
        );
        assert_eq!(
            attendee_status(EKParticipantStatus::Declined),
            AttendeeStatus::Declined
        );
        assert_eq!(
            attendee_status(EKParticipantStatus::Completed),
            AttendeeStatus::Unknown
        );
        // A value from a macOS this build has never seen.
        assert_eq!(
            attendee_status(EKParticipantStatus(99)),
            AttendeeStatus::Unknown
        );
    }

    /// AC #2, the framework-failure arm: an Objective-C raise inside the lookup costs the
    /// meeting field and nothing else, and the caller keeps running afterwards.
    ///
    /// An unrecognized selector stands in for the raise for the reason
    /// [`crate::exception`]'s own tests give: it is a genuine `NSException` travelling the
    /// genuine unwind path, and nothing downstream can tell it from the one EventKit throws.
    #[test]
    fn a_framework_raise_costs_the_meeting_and_not_the_recording() {
        use objc2::rc::Retained;
        use objc2::runtime::NSObject;

        let answer = caught(|| {
            let object = NSObject::new();
            // SAFETY: none required -- sending a selector `NSObject` does not implement is
            // exactly the misuse being provoked, and the raise it produces is the subject.
            let _: Retained<NSObject> = unsafe { objc2::msg_send![&*object, copy] };
            Vec::new()
        });

        assert!(answer.is_none(), "a raise must not produce a meeting");
        // The property this whole arrangement exists for: control reached this line, so a
        // recording being finalized around it would have gone on to be written.
        assert_eq!(caught(Vec::new).map(|c| c.len()), Some(0));
    }

    /// AC #2 again, for the other way out. The bindings return non-optional `Retained<T>`
    /// for properties Apple declares nonnull, so a framework that ever returns nil panics
    /// rather than raising -- and a panic here would unwind out of `finish` and lose the
    /// recording. (The panic message this prints is expected test output.)
    #[test]
    fn a_panic_in_the_lookup_costs_the_meeting_and_not_the_recording() {
        let answer = caught(|| panic!("nil where the binding demands a value"));

        assert!(answer.is_none(), "a panic must not produce a meeting");
        assert_eq!(caught(Vec::new).map(|c| c.len()), Some(0));
    }

    /// The conversion half, against real framework objects rather than a stand-in for them.
    ///
    /// An unsaved `EKEvent` can be built from an ungranted store -- creating a store neither
    /// prompts nor reads anything -- so the `EKEvent` -> [`Meeting`] mapping is decidable
    /// here even on a machine where fetching a real meeting is not. What this cannot reach is
    /// the attendee list and the organizer, which EventKit exposes read-only and populates
    /// only from a synced invitation; that part of the conversion is the hardware check.
    ///
    /// The whole body goes through [`crate::exception::catching`] because a raise from an
    /// unsaved-object setter would otherwise abort the test process outright rather than
    /// failing this test.
    #[test]
    fn an_event_converts_to_the_meeting_that_is_stored() {
        let converted = crate::exception::catching("EKEvent (test fixture)", || {
            // SAFETY: creating a store and an unsaved event reads no calendar data and
            // cannot prompt -- only the request APIs, which this module never calls, can do
            // either. Every object stays alive for the whole conversion.
            unsafe {
                let store = EKEventStore::new();
                let calendar =
                    EKCalendar::calendarForEntityType_eventStore(EKEntityType::Event, &store);
                calendar.setTitle(&NSString::from_str("Work"));

                let event = EKEvent::eventWithEventStore(&store);
                event.setTitle(Some(&NSString::from_str("Incident review")));
                event.setCalendar(Some(&calendar));
                event.setStartDate(Some(&NSDate::dateWithTimeIntervalSince1970(
                    at("2026-08-15T09:55:00Z").as_duration().as_secs_f64(),
                )));
                event.setEndDate(Some(&NSDate::dateWithTimeIntervalSince1970(
                    at("2026-08-15T10:25:00Z").as_duration().as_secs_f64(),
                )));

                super::candidate(&event)
            }
        })
        .expect("building an in-memory event must not raise")
        .expect("an event with both dates set must convert");

        assert_eq!(converted.meeting.title, "Incident review");
        assert_eq!(converted.meeting.start, at("2026-08-15T09:55:00Z"));
        assert_eq!(converted.meeting.end, at("2026-08-15T10:25:00Z"));
        assert_eq!(converted.meeting.calendar, "Work");
        assert!(!converted.all_day);
        assert!(!converted.declined);
        // An unsaved event has no invitation behind it, so these are empty rather than wrong.
        assert!(converted.meeting.attendees.is_empty());
        assert!(converted.meeting.organizer.is_none());
        // And the identifier fallback holds: something a later pass can look the event up by,
        // never an empty string.
        assert!(!converted.meeting.event_id.is_empty());

        // The whole chain, end to end on this side of the grant: a converted event is chosen
        // by the policy and reaches `session.json` with the fields AC #1 names.
        let chosen = select(vec![converted], session_start()).expect("the meeting contains 10:00");
        let json = serde_json::to_string(&chosen).unwrap();
        for field in ["Incident review", "2026-08-15T09:55:00Z", "Work"] {
            assert!(json.contains(field), "{field} is missing from {json}");
        }
    }

    /// The permission arm of AC #2, exercised for real rather than simulated: on any machine
    /// without a full-access calendar grant -- which includes every CI runner -- the lookup
    /// must answer `None` rather than prompt, block, or fail.
    #[test]
    fn a_lookup_without_a_grant_answers_none() {
        // SAFETY: as `meeting_at`.
        let status = unsafe { EKEventStore::authorizationStatusForEntityType(EKEntityType::Event) };
        if status == EKAuthorizationStatus::FullAccess {
            // A granted machine cannot assert absence; the hardware check covers that side.
            return;
        }
        assert!(meeting_at(session_start()).is_none());
    }
}
