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
//! **The lookup only ever reads the grant; asking for one is a different call at a different
//! time.** [`meeting_at`] calls `authorizationStatusForEntityType:` and nothing else, which is
//! a pure read: no prompt, no daemon wake, no Info.plist requirement. That matters because the
//! lookup runs inside [`crate::RunningSession::finish`], between the last audio buffer and the
//! `session.json` write, where a modal dialog would cost a human's full attention span and a
//! process termination would cost the recording outright.
//!
//! [`request_calendar_access`] is where the asking happens, and it is called **once per
//! process at record start** -- right after [`crate::preflight()`], before the activity
//! watcher is installed, so nothing is being captured while the prompt is up and the exposure
//! is identical to the microphone prompt `preflight` already blocks on at the same point.
//!
//! It calls the *deprecated* `requestAccessToEntityType:completion:`, deliberately. The modern
//! `requestFullAccessToEventsWithCompletion:` requires `NSCalendarsFullAccessUsageDescription`
//! in the *responsible* process's Info.plist on macOS 14+ and *terminates the process* when
//! the key is absent -- which most terminal emulators, including the one this was measured on,
//! do not ship. The deprecated selector needs only `NSCalendarsUsageDescription`, which they
//! do ship. Apple documents that selector as granting write-only access for events on macOS
//! 14+, and write-only cannot read events; TASK-030.01.01 measured it granting `FullAccess`
//! instead, on hardware, with events readable afterwards.
//!
//! That measurement contradicts the documentation, so it is guarded rather than trusted: the
//! status is **re-read after the request** and anything other than `FullAccess` -- including
//! `WriteOnly` -- is treated as no access, by the same rule [`meeting_at`] already applies. If
//! a future macOS starts behaving as documented, the result is a degradation to `meeting:
//! None` and a guidance line, not a store that silently returns nothing.
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
use std::fmt;
use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;

use block2::RcBlock;
use jiff::{SignedDuration, Timestamp};
use meethook_session::{Attendee, AttendeeStatus, Meeting, MeetingFit};
use objc2::runtime::Bool;
use objc2_event_kit::{
    EKAuthorizationStatus, EKEntityType, EKEvent, EKEventStatus, EKEventStore, EKParticipant,
    EKParticipantStatus,
};
use objc2_foundation::{NSDate, NSError};

use crate::preflight::PROMPT_TIMEOUT;

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
    let status = status();
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
    let candidates = caught("EKEventStore.eventsMatchingPredicate", || unsafe {
        candidates_around(at)
    })?;
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

/// The calendar authorization status.
///
/// One expression with two readers -- [`meeting_at`] and [`request_calendar_access`] -- rather
/// than the same `unsafe` call written twice, because getting the entity type wrong in one of
/// them would answer a question about reminders.
fn status() -> EKAuthorizationStatus {
    // SAFETY: a class-level status read taking only the entity type. It touches no store,
    // no calendar and no user data, and is the one EventKit entry point that cannot prompt.
    unsafe { EKEventStore::authorizationStatusForEntityType(EKEntityType::Event) }
}

/// Calendar access is not available, and sessions will not be named after their meetings.
///
/// A value rather than a printed line, mirroring [`crate::MissingPermissions`]: this crate
/// owns the wording, the CLI owns the output stream. It is not an [`crate::Error`] variant and
/// is never returned through `Result` -- calendar access is not a precondition for recording,
/// and making it one would invert this crate's priority ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoCalendarAccess;

impl fmt::Display for NoCalendarAccess {
    /// Says all three things a user needs: what is lost, what is *not* lost, and where the
    /// fix is -- including the terminal-inheritance trap, in the same words
    /// [`crate::MissingPermissions`] uses, because macOS will never create a "meethook" entry
    /// in the Calendars pane for the user to look for.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Calendar access is not available, so sessions will not be named after the meeting"
        )?;
        writeln!(
            f,
            "they were recorded during. Recording itself is unaffected."
        )?;
        writeln!(f)?;
        writeln!(f, "  System Settings > Privacy & Security > Calendars")?;
        writeln!(f)?;
        writeln!(
            f,
            "meethook is a command-line tool, so macOS attributes this permission to the"
        )?;
        writeln!(
            f,
            "terminal application you launched it from -- grant it to that app, not to a"
        )?;
        write!(
            f,
            "\"meethook\" entry. You may need to quit and reopen the terminal afterwards."
        )
    }
}

/// Asks macOS for calendar access if it has never been asked, and reports whether the store
/// can be read afterwards.
///
/// `None` means a meeting can be looked up. `Some(_)` carries the guidance for the user, which
/// the caller prints; nothing here is fatal and nothing here returns a `Result`.
///
/// Called once at record start rather than from [`crate::RunningSession::finish`], and that
/// placement is the whole reason this function exists separately from the lookup. The
/// lookup was put in `finish` because the start path is where lost audio comes from -- but
/// that argument is about a several-millisecond store read, and it does not survive a
/// two-minute human decision landing between the last audio buffer and the `session.json`
/// write. At record start nothing is being captured yet, so the wait costs nothing.
///
/// The accepted cost, recorded rather than solved: somebody who runs `meethook record` for the
/// first time *while already in a call* and then walks away from the prompt loses up to
/// two minutes before watching begins. First run only, and [`crate::preflight()`]'s microphone
/// prompt already has exactly this exposure on the same line of the same function.
pub fn request_calendar_access() -> Option<NoCalendarAccess> {
    let granted = resolve(status(), || {
        // A raise or a panic out of the request costs the prompt and not the process -- the
        // same trade the lookup already makes. It reads as `NotDetermined` because that is
        // what it left behind: nothing was resolved, and no access follows either way.
        caught("EKEventStore.requestAccessToEntityType", ask)
            .unwrap_or(EKAuthorizationStatus::NotDetermined)
    });

    if granted {
        debug("calendar access is available");
        return None;
    }
    debug(&format!(
        "calendar access is unavailable (status {})",
        status().0
    ));
    Some(NoCalendarAccess)
}

/// Whether the store is readable, given the status before asking and a way to ask.
///
/// The whole policy, with no framework in reach -- the same split this module already makes
/// between [`select`] and [`candidates_around`], and for the same reason: every rule below is
/// then decidable in `cargo test` on a machine with no calendar, no grant, and no prompt
/// anywhere near it.
///
/// - `FullAccess` -- already granted, and `ask` is never called.
/// - `NotDetermined` -- `ask`, and the answer is whatever status it leaves behind.
/// - anything else (`Denied`, `Restricted`, `WriteOnly`, a value from a later macOS) -- no
///   access, and `ask` is never called. **A resolved denial is never re-asked**: macOS would
///   not prompt for it anyway, so asking only converts a settled answer into a nuisance.
///
/// `WriteOnly` is not access. Apple documents the deprecated request path as producing exactly
/// that on macOS 14+, and a write-only store cannot read events, so it is rejected on both
/// sides -- as a starting status and as an answer.
fn resolve(before: EKAuthorizationStatus, ask: impl FnOnce() -> EKAuthorizationStatus) -> bool {
    match before {
        EKAuthorizationStatus::FullAccess => true,
        EKAuthorizationStatus::NotDetermined => ask() == EKAuthorizationStatus::FullAccess,
        _ => false,
    }
}

/// Raises the OS prompt, waits for the answer, and returns the status it left behind.
///
/// The status re-read is the authoritative answer rather than the block's `granted` boolean --
/// identical in shape and reasoning to [`crate::preflight()`]'s microphone request. It covers a
/// block that never fired, and it is what turns a documented write-only downgrade into a
/// reported failure instead of a store that answers nothing.
fn ask() -> EKAuthorizationStatus {
    // The status is `NotDetermined`, so a dialog is about to appear over whatever the user is
    // looking at. This line lives here rather than in the CLI because this is the only code
    // that knows a prompt is coming; a caller-side check would duplicate the status read.
    // Flushed before the call on the probe's precedent -- cheap insurance on a call family
    // where one member terminates the process, even though this member does not.
    println!("Asking macOS for calendar access, so sessions can be named after their meeting.");
    let _ = std::io::stdout().flush();

    let (tx, rx) = mpsc::channel::<(bool, Option<String>)>();
    // Two arguments, unlike `AVCaptureDevice`'s one-argument block: EventKit's completion
    // handler is `(BOOL, NSError *)`.
    let handler = RcBlock::new(move |granted: Bool, error: *mut NSError| {
        // SAFETY: EventKit passes either null or a valid autoreleased `NSError` here, and the
        // description is copied into an owned `String` before this block returns.
        let message =
            unsafe { error.as_ref() }.map(|error| error.localizedDescription().to_string());
        let _ = tx.send((granted.as_bool(), message));
    });

    // The store is created here and dropped at the end of this function, so it stays alive
    // behind the completion block for the whole of the wait.
    //
    // SAFETY: `handler` is a live block for the whole call -- `RcBlock` owns it and this
    // function blocks until the block has fired or the wait times out -- and the selector
    // accepts a completion handler of exactly this signature.
    unsafe {
        let store = EKEventStore::new();
        #[expect(
            deprecated,
            reason = "the modern requestFullAccessToEventsWithCompletion: requires \
                      NSCalendarsFullAccessUsageDescription in the responsible process and \
                      terminates the process outright without it, which terminal emulators do \
                      not ship. This selector needs only NSCalendarsUsageDescription, and \
                      TASK-030.01.01 measured it granting FullAccess on hardware rather than \
                      the write-only access Apple documents -- which the status re-read below \
                      guards against regardless"
        )]
        store.requestAccessToEntityType_completion(EKEntityType::Event, RcBlock::as_ptr(&handler));

        // Ignored on purpose: the re-read below is the authoritative answer, and a timeout
        // here is indistinguishable from a prompt nobody answered -- which reads as
        // `NotDetermined`, i.e. no access, which is the right outcome for both.
        match rx.recv_timeout(PROMPT_TIMEOUT) {
            Ok((granted, error)) => debug(&format!(
                "the calendar request returned granted {granted}, error {error:?}"
            )),
            Err(_) => debug(&format!(
                "the calendar prompt went unanswered for {}s",
                PROMPT_TIMEOUT.as_secs()
            )),
        }

        status()
    }
}

/// Runs an EventKit call with both ways out of it blocked.
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
/// The call is a parameter rather than the body so that both failure paths are decidable
/// in `cargo test`: a claim that a raise costs a field instead of a recording is worth
/// nothing until something has actually raised here. `api` names the raising call, as
/// [`crate::exception::catching`] documents, so the two callers are distinguishable in the
/// debug line.
fn caught<T>(api: &'static str, call: impl FnOnce() -> T) -> Option<T> {
    let outcome =
        std::panic::catch_unwind(AssertUnwindSafe(|| crate::exception::catching(api, call)));

    match outcome {
        Ok(Ok(value)) => Some(value),
        Ok(Err(raise)) => {
            debug(&format!("{raise}"));
            None
        }
        Err(_) => {
            debug(&format!("{api} panicked; continuing without a meeting"));
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

        // No fit is set here, and `Meeting::new` leaves it `Unknown`: a converted event is a
        // *candidate*, not yet a match, and only `select` knows how the session start sat
        // against it.
        let meeting = Meeting::new(
            // `eventIdentifier` is nil only for an event not yet saved to a store, which
            // one fetched *from* a store cannot be. The fallback is the calendar item's
            // own identifier rather than an empty string so that the field keeps its
            // meaning -- something a later pass can look the event back up by.
            event.eventIdentifier().map_or_else(
                || event.calendarItemIdentifier().to_string(),
                |id| id.to_string(),
            ),
            event.title().to_string(),
            event
                .calendar()
                .map(|calendar| calendar.title().to_string())
                .unwrap_or_default(),
            start,
            end,
        )
        .with_people(
            event.organizer().map(|organizer| attendee(&organizer)),
            attendees,
        )
        .with_invite(
            event
                .URL()
                .and_then(|url| url.absoluteString())
                .map(|url| url.to_string()),
            // Both are absent-or-present, never present-and-empty: EventKit hands back an
            // empty string for a field the organizer left blank in some accounts and nil
            // in others, and `"notes": ""` in `session.json` would read as "the agenda
            // was empty" rather than "there was no agenda".
            text(event.location()),
            text(event.notes()),
        );

        Some(Candidate {
            meeting,
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

/// An optional framework string as owned Rust, with the empty one treated as absent.
fn text(value: Option<objc2::rc::Retained<objc2_foundation::NSString>>) -> Option<String> {
    value
        .map(|value| value.to_string())
        .filter(|value| !value.is_empty())
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
///
/// The chosen meeting also carries a [`MeetingFit`], which is a *description* of the winning
/// rule and never an input to it. **Which meeting is selected does not depend on the fit, and
/// does not depend on the session's end.** A session running 2:24-3:40 against a 2:00-3:00
/// standup and a 3:30 invite still resolves to the standup it started inside; it is simply
/// marked [`MeetingFit::JoinedLate`] rather than presented as a meeting it certainly was.
fn select(candidates: Vec<Candidate>, at: Timestamp) -> Option<Meeting> {
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

/// Renders a candidate for [`debug`] with its attendees *counted*, never named, and its
/// notes and location omitted entirely.
///
/// Attendee names and addresses go to `session.json` because speaker identification needs
/// them; the invite body goes there because it is the meeting's agenda. Neither goes to a
/// terminal, a log file, or anywhere a screen share or a pasted error report can carry them
/// -- an invite body is the single most likely field here to contain a dial-in PIN. What is
/// left is what someone debugging "why did my session get no meeting?" actually needs: the
/// times, the title, the calendar and the two disqualifying flags. Keeping the rendering in
/// one tested function is what makes that a property of the code rather than a promise --
/// see the test below.
fn summarize(candidate: &Candidate) -> String {
    let mut line = String::new();
    let _ = write!(
        line,
        "{} .. {}  {:?}  [{}]  {} attendee(s)",
        candidate.meeting.start,
        candidate.meeting.end,
        candidate.meeting.title,
        candidate.meeting.calendar,
        candidate.meeting.attendee_count(),
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

    /// AC #5, in the only place a meeting is ever rendered for a human: attendee names and
    /// addresses, the invite body and the location reach `session.json` and nothing else.
    #[test]
    fn the_debug_line_counts_attendees_without_naming_them() {
        let mut candidate = candidate("standup", "2026-08-15T09:55:00Z", "2026-08-15T10:25:00Z");
        candidate.meeting = candidate
            .meeting
            .with_invite(
                None,
                Some("Babbage Room, 12 Ada Street".to_owned()),
                Some("Dial-in 555-0100, passcode 481516".to_owned()),
            )
            .with_people(
                Some(Attendee {
                    name: Some("Alan Turing".to_owned()),
                    email: Some("alan@example.com".to_owned()),
                    status: AttendeeStatus::Accepted,
                    is_you: false,
                }),
                vec![
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
                ],
            );

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
            // The invite body and the location, which are why this test matters most.
            "Dial-in",
            "555-0100",
            "passcode",
            "481516",
            "Babbage",
            "12 Ada Street",
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

        let answer = caught("test", || {
            let object = NSObject::new();
            // SAFETY: none required -- sending a selector `NSObject` does not implement is
            // exactly the misuse being provoked, and the raise it produces is the subject.
            let _: Retained<NSObject> = unsafe { objc2::msg_send![&*object, copy] };
            Vec::<Candidate>::new()
        });

        assert!(answer.is_none(), "a raise must not produce a meeting");
        // The property this whole arrangement exists for: control reached this line, so a
        // recording being finalized around it would have gone on to be written.
        assert_eq!(
            caught("test", Vec::<Candidate>::new).map(|c| c.len()),
            Some(0)
        );
    }

    /// AC #2 again, for the other way out. The bindings return non-optional `Retained<T>`
    /// for properties Apple declares nonnull, so a framework that ever returns nil panics
    /// rather than raising -- and a panic here would unwind out of `finish` and lose the
    /// recording. (The panic message this prints is expected test output.)
    #[test]
    fn a_panic_in_the_lookup_costs_the_meeting_and_not_the_recording() {
        let answer = caught("test", || -> Vec<Candidate> {
            panic!("nil where the binding demands a value")
        });

        assert!(answer.is_none(), "a panic must not produce a meeting");
        assert_eq!(
            caught("test", Vec::<Candidate>::new).map(|c| c.len()),
            Some(0)
        );
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
                event.setNotes(Some(&NSString::from_str("Agenda: the pager, then the fix")));
                // Blank rather than unset, which is how some accounts spell "no location":
                // it must convert to absent, not to `Some("")`.
                event.setLocation(Some(&NSString::from_str("")));

                super::candidate(&event)
            }
        })
        .expect("building an in-memory event must not raise")
        .expect("an event with both dates set must convert");

        assert_eq!(converted.meeting.title, "Incident review");
        assert_eq!(converted.meeting.start, at("2026-08-15T09:55:00Z"));
        assert_eq!(converted.meeting.end, at("2026-08-15T10:25:00Z"));
        assert_eq!(converted.meeting.calendar, "Work");
        assert_eq!(
            converted.meeting.notes.as_deref(),
            Some("Agenda: the pager, then the fix")
        );
        assert_eq!(
            converted.meeting.location, None,
            "an empty location is none"
        );
        assert!(!converted.all_day);
        assert!(!converted.declined);
        // An unsaved event has no invitation behind it, so these are empty rather than wrong.
        assert_eq!(converted.meeting.attendee_count(), 0);
        assert!(converted.meeting.organizer.is_none());
        // A converted event is a candidate, not yet a match: nothing has scored it.
        assert_eq!(converted.meeting.fit, MeetingFit::Unknown);
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
    ///
    /// It is also the standing proof that [`meeting_at`] did not acquire a request path when
    /// [`request_calendar_access`] was added: a lookup that asked would hang this test on an
    /// ungranted machine instead of returning.
    #[test]
    fn a_lookup_without_a_grant_answers_none() {
        if status() == EKAuthorizationStatus::FullAccess {
            // A granted machine cannot assert absence; the hardware check covers that side.
            return;
        }
        assert!(meeting_at(session_start()).is_none());
    }

    // The request-path rules, every one of them driven through `resolve` with a closure.
    //
    // Hard rule: **no test may call the real request path.** A test that prompted would hang
    // CI, and on a developer machine a reflexive "Don't Allow" writes a `Denied` into TCC that
    // is far stickier than the `NotDetermined` it replaced -- the same reasoning that kept
    // TASK-030.01 from running its `request-legacy` mode from a sandbox. `ask` is the only
    // function here that touches EventKit's request selector, and nothing below reaches it.

    /// A grant already in place is used, not re-requested: asking again would raise a second
    /// dialog on every single run of `meethook record`.
    #[test]
    fn a_grant_already_in_place_is_not_asked_for_again() {
        assert!(resolve(EKAuthorizationStatus::FullAccess, || unreachable!(
            "an existing grant must not be re-requested"
        )));
    }

    /// A settled answer is never re-asked. Table-driven so that a status a later macOS adds --
    /// which lands in the same catch-all arm -- cannot quietly become a re-ask either.
    #[test]
    fn a_resolved_denial_is_never_re_asked() {
        for status in [
            EKAuthorizationStatus::Denied,
            EKAuthorizationStatus::Restricted,
            EKAuthorizationStatus(99),
        ] {
            assert!(
                !resolve(status, || unreachable!(
                    "status {} must not be re-requested",
                    status.0
                )),
                "status {} was treated as access",
                status.0
            );
        }
    }

    /// The guard against the macOS 14+ downgrade Apple documents for the deprecated request
    /// path: a write-only store cannot read events, so it is not access -- neither as a
    /// starting status nor as the answer a request comes back with.
    ///
    /// This is the one behaviour that cannot be checked on the machine where the contrary
    /// observation was made, which is exactly why it is asserted here.
    #[test]
    fn write_only_access_is_not_access() {
        assert!(!resolve(EKAuthorizationStatus::WriteOnly, || unreachable!(
            "a write-only grant is settled and must not be re-requested"
        )));
        assert!(!resolve(EKAuthorizationStatus::NotDetermined, || {
            EKAuthorizationStatus::WriteOnly
        }));
    }

    /// AC #2's timeout arm. A `recv_timeout` expiry and a completion block that never fired
    /// both leave the status where it started, and that shape must read as no access rather
    /// than as a grant.
    #[test]
    fn an_unanswered_prompt_reads_as_no_access() {
        assert!(!resolve(EKAuthorizationStatus::NotDetermined, || {
            EKAuthorizationStatus::NotDetermined
        }));
    }

    #[test]
    fn a_granted_prompt_is_access() {
        assert!(resolve(EKAuthorizationStatus::NotDetermined, || {
            EKAuthorizationStatus::FullAccess
        }));
    }

    /// The sentence is the whole value of the type, so a refactor must not drop it. Same
    /// shape, and same reason, as `preflight`'s two `MissingPermissions` tests.
    #[test]
    fn the_guidance_names_the_pane_and_the_terminal_trap() {
        let message = NoCalendarAccess.to_string();
        assert!(
            message.contains("Privacy & Security > Calendars"),
            "the pane is missing: {message}"
        );
        assert!(
            message.contains("terminal application"),
            "the inheritance trap is missing: {message}"
        );
        assert!(
            message.contains("Recording itself is unaffected"),
            "the message must say recording still works: {message}"
        );
    }
}
