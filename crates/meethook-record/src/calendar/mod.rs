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
//! There is one other caller of the same read, and it is mid-recording rather than at finish:
//! the record interface asks [`meetings_for`] once when a session opens, so the user can see
//! what the automatic rule would attach and pick a different event by hand while the call is
//! still live. That pause bounds the main thread's event loop, not the capture -- the engines
//! run on their own threads -- and the *ask* for the grant happened once at record start,
//! long before this read, so nothing here prompts.
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
//! the choice policy is pure arithmetic over start/end/all-day/declined -- so the split is
//! what makes the policy testable on a machine with no calendar at all; it lives in
//! [`select`].

mod select;

use std::fmt;
use std::fmt::Write as _;
use std::io::Write as _;
use std::panic::AssertUnwindSafe;
use std::sync::mpsc;

use block2::RcBlock;
use jiff::{SignedDuration, Timestamp};
use meethook_session::{Attendee, AttendeeStatus, Meeting};
use objc2::runtime::Bool;
use objc2_event_kit::{
    EKAuthorizationStatus, EKEntityType, EKEvent, EKEventStatus, EKEventStore, EKParticipant,
    EKParticipantStatus,
};
use objc2_foundation::{NSDate, NSError};

use crate::preflight::PROMPT_TIMEOUT;
use select::{Candidate, offerable, select};

/// How much calendar to fetch on either side of the session start.
///
/// Wider than [`select::NEAR_WINDOW`] on purpose, and it does not need to be exact: the predicate
/// returns events *overlapping* the range, so a two-hour meeting that began 50 minutes
/// before the session must still be inside it to be found. Policy lives in [`select::select`], not
/// in the query, so making this generous costs a few discarded candidates and nothing else.
const QUERY_WINDOW: SignedDuration = SignedDuration::from_secs(60 * 60);

/// The meeting `at` fell within, if the calendar can be read and one matches.
///
/// Total by construction -- no `Result`, no error type -- because there is no caller
/// decision to make. A missing grant, an empty calendar, a raise and a panic are all the
/// same outcome to the one function whose failure would cost a recording, so the branch
/// belongs here rather than at the call site.
pub(crate) fn meeting_at(at: Timestamp) -> Option<Meeting> {
    meetings_for(at).chosen
}

/// Every meeting worth offering as a correction for a session that started at `at`.
///
/// The listing half of `meeting_at`: that one picks, this one shows the choices, and both
/// read the same `QUERY_WINDOW` of calendar around the same instant. `meethook meeting` is
/// the caller -- a session recorded during a double-booked hour resolves to whichever
/// candidate `select` prefers, and the only remaining way to fix that is a person who knows
/// which of the two it was.
///
/// Total, exactly as `meeting_at` is and for the same reason: a missing grant, an empty
/// calendar, a raise and a panic are all an empty `Vec`. Nothing offered is a correction
/// command with nothing to list, never a failed one -- and its other half, clearing a label,
/// needs no calendar at all.
///
/// Each meeting keeps [`meethook_session::MeetingFit::Unknown`]: a candidate is not a
/// match, and only a person choosing one makes it
/// [`meethook_session::MeetingFit::Confirmed`].
pub fn meetings_around(at: Timestamp) -> Vec<Meeting> {
    meetings_for(at).offered
}

/// Both halves of the question "what meeting does this session belong to", answered by one
/// status check and one store query.
///
/// [`meetings_for`] is the shared heart of [`meeting_at`] and [`meetings_around`], and the
/// answer the record interface asks mid-recording: the full-screen frame wants *both* halves
/// at once -- the one the automatic rule would attach, to show with its fit, and everything
/// worth offering, so a wrong guess can be corrected by hand without leaving the call. Two
/// separate calls would take the store pass twice for the same window, and the split into one
/// named value is what says they come from one query and cannot disagree about the window.
///
/// Total, exactly as each half is and for the same reasons: a missing grant, an empty
/// calendar, a raise and a panic all leave both fields empty. The `chosen` meeting carries
/// the fit [`select`] decided; every `offered` meeting keeps
/// [`meethook_session::MeetingFit::Unknown`], and `chosen` is always one of `offered` when it
/// is present -- `select` and `offerable` filter the same candidates, so a meeting the rules
/// would attach is always one a person may offer themselves.
pub struct MeetingLookup {
    /// Every meeting worth offering as a hand correction, in [`select::offerable`]'s stable
    /// order.
    pub offered: Vec<Meeting>,
    /// The one the automatic rule would attach, with the fit that rule decided.
    pub chosen: Option<Meeting>,
}

pub fn meetings_for(at: Timestamp) -> MeetingLookup {
    let status = status();
    if status != EKAuthorizationStatus::FullAccess {
        debug(&format!(
            "calendar access is not granted (status {}); no meeting will be recorded",
            status.0
        ));
        return MeetingLookup {
            offered: Vec::new(),
            chosen: None,
        };
    }

    // SAFETY: every call inside is a read against a freshly created store, made from the
    // calling thread with every event converted to owned Rust before the store, the dates or
    // the predicate are dropped. At finish the engines are already stopped; mid-recording
    // they run on their own threads, which is why this pause costs attention, not audio.
    let candidates = caught("EKEventStore.eventsMatchingPredicate", || unsafe {
        candidates_around(at)
    })
    .unwrap_or_default();
    if debugging() {
        for candidate in &candidates {
            debug(&summarize(candidate));
        }
    }

    let chosen = select(candidates.clone(), at);
    debug(&match chosen.as_ref() {
        Some(meeting) => format!("selected {:?}", meeting.title),
        None => "no candidate matched the session start".to_owned(),
    });
    MeetingLookup {
        offered: offerable(candidates),
        chosen,
    }
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

/// The framework side: raises, panics, conversion and the permission arms, exercised as far
/// as a machine without a calendar grant allows.
#[cfg(test)]
mod tests {
    use objc2_event_kit::EKCalendar;
    use objc2_foundation::NSString;

    use super::select::tests::{at, candidate, session_start};
    use super::*;

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
        assert_eq!(converted.meeting.fit, meethook_session::MeetingFit::Unknown);
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

    /// The same for the listing, and the reason `meethook meeting --clear` needs no calendar:
    /// on an ungranted machine the offer is empty rather than a prompt, a hang or a failure,
    /// so the correction command degrades to the half that needs no events.
    #[test]
    fn nothing_is_offered_without_a_grant() {
        if status() == EKAuthorizationStatus::FullAccess {
            return;
        }
        assert!(meetings_around(session_start()).is_empty());
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
