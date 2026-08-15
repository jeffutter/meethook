//! Reports whether EventKit is reachable from this CLI, and what its store contains.
//!
//! Two questions decide whether a session can be named after the meeting it was recorded
//! during, and neither is answerable from Apple's documentation.
//!
//! **Can a calendar grant reach a non-bundled binary at all.** meethook is a plain
//! executable, so macOS attributes privacy grants to the terminal that launched it -- the
//! trap [`meethook_record::MissingPermissions`] already spells out for the microphone and
//! screen grants. On macOS 14+ `requestFullAccessToEventsWithCompletion:` additionally
//! requires `NSCalendarsFullAccessUsageDescription` in the *responsible* process's
//! `Info.plist` and terminates the process outright when the key is absent, which most
//! terminal emulators do not ship. Which of the two request paths survives is an empirical
//! fact about a particular terminal on a particular macOS.
//!
//! **Are the meetings in the store.** EventKit reads the same database Calendar.app does,
//! so a work calendar that never synced into Calendar.app is not there to be found -- and
//! new Outlook does not sync into it by default. If the meetings are absent, there is
//! nothing to name a session after and the whole approach is wrong.
//!
//! ```text
//! cargo run -p meethook-record --example calendar-access
//! cargo run -p meethook-record --example calendar-access -- request-legacy
//! cargo run -p meethook-record --example calendar-access -- request-full
//! ```
//!
//! The default mode requests nothing. It reads the authorization status, the calendars and
//! the events, and cannot prompt, cannot wake a daemon into asking, and cannot be killed --
//! so it is the safe baseline to run first, and on an ungranted machine its empty listings
//! are the control the other two modes are read against.
//!
//! `request-legacy` calls the deprecated `requestAccessToEntityType:completion:`, which
//! needs only `NSCalendarsUsageDescription`. macOS 14+ is documented to downgrade it to
//! write-only access, and write-only cannot read events -- so a `WriteOnly (4)` status
//! afterwards is a *failure* for this purpose even though the call succeeded.
//!
//! `request-full` is the one that can die. The line naming the selector is printed and
//! flushed immediately before the call, so if macOS kills the process there the last line
//! on the terminal says which call did it.
//!
//! Every mode prints the full read section afterwards, so one invocation answers both
//! questions at once.
//!
//! # What this prints about people
//!
//! This output is meant to be pasted into a ticket, so it is a log line in every sense that
//! matters. It follows the rule `meethook_record`'s calendar module already sets for its own
//! diagnostics: attendees are **counted, never named**. No attendee name and no attendee
//! address is printed, the organizer is reported only as present or absent, and the one
//! person-shaped fact that does appear is the reader's *own* participant status, which is a
//! fact about the reader.
//!
//! Calendar titles, account titles and meeting titles *are* printed, because they are the
//! answer to the second question and redacting them would destroy the only thing this output
//! is for. Redact any that are sensitive by hand before pasting.

use std::ffi::{OsString, c_void};
use std::io::Write as _;
use std::os::unix::ffi::OsStringExt as _;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use block2::RcBlock;
use jiff::{SignedDuration, Timestamp, tz::TimeZone};
use objc2::runtime::Bool;
use objc2_event_kit::{
    EKAuthorizationStatus, EKCalendar, EKCalendarType, EKEntityType, EKEventStatus, EKEventStore,
    EKParticipantStatus, EKSourceType,
};
use objc2_foundation::{NSDate, NSError};

/// How long to wait for a request's completion block.
///
/// Short compared with [`meethook_record::preflight`]'s two minutes, because nothing is
/// being lost while this waits: whoever runs this was told to expect a prompt and is looking
/// at it. Finite so that a block which never fires reports itself instead of hanging.
const PROMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// How much calendar to list on either side of now.
const WINDOW: SignedDuration = SignedDuration::from_secs(60 * 60);

/// Which request path to take, if any.
///
/// A mode rather than a default, and the default is the harmless one. `request-full` is the
/// hazard this whole probe exists to measure, so it must be provokable -- but a probe that
/// provoked it by default would destroy its own baseline output before printing it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Read,
    RequestLegacy,
    RequestFull,
}

impl Mode {
    /// An unrecognized argument is a usage error rather than a silent fall-back to `Read`: a
    /// typo that quietly took the safe path would be indistinguishable from a request that
    /// did nothing, which is exactly the distinction this probe exists to draw.
    fn parse(arg: Option<&str>) -> Result<Self, String> {
        match arg {
            None | Some("read") => Ok(Self::Read),
            Some("request-legacy") => Ok(Self::RequestLegacy),
            Some("request-full") => Ok(Self::RequestFull),
            Some(other) => Err(format!("unknown mode {other:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::RequestLegacy => "request-legacy",
            Self::RequestFull => "request-full",
        }
    }
}

fn main() {
    let arg = std::env::args().nth(1);
    let mode = match Mode::parse(arg.as_deref()) {
        Ok(mode) => mode,
        Err(problem) => {
            eprintln!("calendar-access: {problem}");
            eprintln!("usage: calendar-access [read|request-legacy|request-full]");
            std::process::exit(2);
        }
    };

    header(mode);

    // Creating a store reads no calendar data and cannot prompt -- only the request APIs can
    // do either -- and the same instance is used for the request and both listings so that a
    // grant which lands mid-run is visible to the reads that follow it.
    //
    // SAFETY: a plain allocation. No entitlement, no I/O, no user data.
    let store = unsafe { EKEventStore::new() };

    // Both status lines and the request-path line print in every mode, including the one
    // that requests nothing. An unconditional "before" and "after" is what makes a mode that
    // changed the status distinguishable from one that did not *by reading the output*,
    // rather than by knowing which mode does what -- and "none (read mode requests nothing)"
    // is itself the answer to which path was taken.
    println!();
    status_line("before");

    println!();
    if mode == Mode::Read {
        println!("request path: none (read mode requests nothing and cannot prompt)");
    } else {
        let _ = guarded("the request", || request(&store, mode));
    }

    println!();
    status_line("after");

    println!();
    let calendars = guarded("the calendar listing", || calendars(&store));
    println!();
    let events = guarded("the event listing", || events(&store));

    println!();
    println!(
        "verdict: mode {}, status {}, {}, {}",
        mode.as_str(),
        authorization(),
        count("calendar", calendars),
        count("event", events),
    );
}

/// Who is asking, which is half of the attribution question.
///
/// The parent process is the responsible-process candidate a TCC grant will be attributed
/// to, so naming it here is what makes "did the grant land on the terminal" answerable from
/// this output alone. Failing to name it is reported and survived: this is a diagnostic, not
/// a precondition.
fn header(mode: Mode) {
    println!("calendar-access, mode {}", mode.as_str());
    println!("  pid          {}", std::process::id());
    println!(
        "  executable   {}",
        std::env::current_exe().map_or_else(
            |e| format!("<unreadable: {e}>"),
            |path| path.display().to_string()
        )
    );
    // SAFETY: `getppid` takes no arguments, touches no memory and cannot fail.
    let parent = unsafe { libc::getppid() };
    println!(
        "  parent       pid {parent}, {}",
        executable(parent).map_or_else(
            || "<unreadable>".to_owned(),
            |path| path.display().to_string()
        )
    );
}

/// The executable behind a pid, or `None` for one that cannot be read.
///
/// The same `proc_pidpath` call `meethook_record`'s activity watcher uses to name the
/// process behind a capturing pid; it is duplicated rather than shared because that helper
/// is private to the crate and an example is an outside caller.
fn executable(pid: i32) -> Option<PathBuf> {
    let mut buffer = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    // SAFETY: `buffer` is owned, writable storage of exactly the length passed alongside it.
    // `proc_pidpath` writes at most that many bytes and returns the length written.
    let length = unsafe {
        libc::proc_pidpath(
            pid,
            buffer.as_mut_ptr().cast::<c_void>(),
            buffer.len() as u32,
        )
    };
    if length <= 0 {
        return None;
    }
    buffer.truncate(length as usize);
    Some(PathBuf::from(OsString::from_vec(buffer)))
}

/// Prints the current authorization status, labelled.
fn status_line(label: &str) {
    println!("authorization {label:<6} {}", authorization());
}

/// The calendar authorization status, as *name and number*.
///
/// Both, deliberately: the name is what a human reads, and the number is what survives being
/// pasted into a ticket by somebody who does not know the enum. Anything a later macOS adds
/// renders as `unknown (n)` rather than being dropped.
fn authorization() -> String {
    // SAFETY: a class-level status read taking only the entity type. It touches no store, no
    // calendar and no user data, and is the one EventKit entry point that cannot prompt.
    let status = unsafe { EKEventStore::authorizationStatusForEntityType(EKEntityType::Event) };
    let name = match status {
        EKAuthorizationStatus::NotDetermined => "NotDetermined",
        EKAuthorizationStatus::Restricted => "Restricted",
        EKAuthorizationStatus::Denied => "Denied",
        EKAuthorizationStatus::FullAccess => "FullAccess",
        EKAuthorizationStatus::WriteOnly => "WriteOnly",
        _ => "unknown",
    };
    format!("{name} ({})", status.0)
}

/// Makes the request the mode selected, and waits for its answer.
///
/// The completion block fires on an arbitrary dispatch queue, so the answer comes back over
/// a channel rather than by mutating shared state -- the same arrangement
/// [`meethook_record::preflight`] uses for the microphone prompt, and for the same reason.
/// The block's `granted` boolean is printed but is not the authoritative answer; the status
/// re-read by the caller afterwards is, which also covers a block that never fired at all.
///
/// A timeout is printed distinctly from a `false`: a prompt nobody answered and a call that
/// never called back are different findings.
fn request(store: &EKEventStore, mode: Mode) {
    let (tx, rx) = mpsc::channel::<(bool, Option<String>)>();
    let handler = RcBlock::new(move |granted: Bool, error: *mut NSError| {
        // SAFETY: EventKit passes either null or a valid autoreleased `NSError` here, and
        // the description is copied into an owned `String` before this block returns.
        let message =
            unsafe { error.as_ref() }.map(|error| error.localizedDescription().to_string());
        let _ = tx.send((granted.as_bool(), message));
    });

    let selector = match mode {
        Mode::RequestFull => "requestFullAccessToEventsWithCompletion:",
        _ => "requestAccessToEntityType:completion:",
    };
    // The single most load-bearing line in this file. `requestFullAccessToEventsWithCompletion:`
    // terminates the process when the responsible app lacks NSCalendarsFullAccessUsageDescription,
    // and a buffered stdout would take this line to the grave with it -- leaving an operator
    // with a dead process and no record of which call killed it. Print, then flush, then call.
    println!("request path: {selector} -- calling it now");
    let _ = std::io::stdout().flush();

    // SAFETY: `handler` is a live block for the whole call and outlives it -- `RcBlock` owns
    // it and this function blocks until the block has fired or the wait times out. Both
    // selectors accept a completion handler of exactly this signature.
    unsafe {
        match mode {
            Mode::RequestFull => {
                store.requestFullAccessToEventsWithCompletion(RcBlock::as_ptr(&handler));
            }
            _ => {
                #[expect(
                    deprecated,
                    reason = "the deprecated path is the subject: it needs only the usage-description \
                              key a terminal actually ships, and whether that is enough to read \
                              events is the question this mode exists to answer"
                )]
                store.requestAccessToEntityType_completion(
                    EKEntityType::Event,
                    RcBlock::as_ptr(&handler),
                );
            }
        }
    }

    println!("survived the call");
    match rx.recv_timeout(PROMPT_TIMEOUT) {
        Ok((granted, error)) => {
            println!("  completion block: granted {granted}");
            match error {
                Some(message) => println!("  completion block: error {message:?}"),
                None => println!("  completion block: no error"),
            }
        }
        Err(_) => println!(
            "  completion block never fired within {}s -- an unanswered prompt and a call that \
             never called back look the same from here",
            PROMPT_TIMEOUT.as_secs()
        ),
    }
}

/// Every calendar in the store, then every account behind them.
///
/// The two listings are separate on purpose. An account that synced with zero event
/// calendars and an account that is not configured at all produce the same empty calendar
/// list, and only the source list tells them apart -- which is precisely the difference
/// between "the work calendar did not sync" and "there is no work account here".
///
/// Returns the number of calendars, for the verdict line.
fn calendars(store: &EKEventStore) -> usize {
    // SAFETY: every call here is a read against a live store. Each returned object is
    // converted to owned Rust before the store is dropped.
    unsafe {
        let calendars = store.calendarsForEntityType(EKEntityType::Event).to_vec();
        if calendars.is_empty() {
            println!("calendars: the store returned none");
        } else {
            println!("calendars: {}", calendars.len());
            for calendar in &calendars {
                println!("  {:?}", calendar.title().to_string());
                println!("      type        {}", calendar_type(calendar.r#type()));
                println!("      account     {}", account(calendar));
                println!(
                    "      writable    {}   subscribed {}",
                    calendar.allowsContentModifications(),
                    calendar.isSubscribed()
                );
                println!("      identifier  {}", calendar.calendarIdentifier());
            }
        }

        let sources = store.sources().to_vec();
        if sources.is_empty() {
            println!("accounts: the store returned none");
        } else {
            println!("accounts: {}", sources.len());
            for source in &sources {
                println!(
                    "  {:?}  {}",
                    source.title().to_string(),
                    source_type(source.sourceType())
                );
            }
        }

        println!(
            "default calendar for new events: {}",
            store.defaultCalendarForNewEvents().map_or_else(
                || "<none>".to_owned(),
                |calendar| format!("{:?}", calendar.title().to_string())
            )
        );

        calendars.len()
    }
}

/// The account a calendar belongs to.
///
/// `EKCalendar::type()` alone answers CalDAV/Exchange/Local but not *which* account, and a
/// machine with several configured accounts needs the distinction to answer whether the work
/// meetings are here.
///
/// # Safety
///
/// As [`calendars`].
unsafe fn account(calendar: &EKCalendar) -> String {
    unsafe {
        calendar.source().map_or_else(
            || "<none>".to_owned(),
            |source| {
                format!(
                    "{:?}  {}",
                    source.title().to_string(),
                    source_type(source.sourceType())
                )
            },
        )
    }
}

/// Every event overlapping [`WINDOW`] either side of now.
///
/// Deliberately the *raw* store rather than `meethook_record`'s own selection: a probe that
/// went through the production filter could not show an event the production filter drops,
/// and "is the meeting in the store" is the question here.
///
/// Sorted by start before printing because `eventsMatchingPredicate:` documents no order at
/// all, and an unsorted listing would be harder to compare against Calendar.app.
///
/// Returns the number of events, for the verdict line.
fn events(store: &EKEventStore) -> usize {
    let now = Timestamp::now();

    // SAFETY: the store, both dates and the predicate all outlive the query, and the
    // predicate is one this same store created -- handing `eventsMatchingPredicate:` a
    // foreign predicate is what makes it raise. Every event is converted to owned Rust
    // before any of them is dropped.
    let mut rows: Vec<(Timestamp, String)> = unsafe {
        let seconds = now.as_duration().as_secs_f64();
        let from = NSDate::dateWithTimeIntervalSince1970(seconds - WINDOW.as_secs_f64());
        let to = NSDate::dateWithTimeIntervalSince1970(seconds + WINDOW.as_secs_f64());
        // `None` means every calendar the grant covers, which is the point: the meetings can
        // live in any account, and asking about a subset would presume the answer.
        let predicate = store.predicateForEventsWithStartDate_endDate_calendars(&from, &to, None);

        store
            .eventsMatchingPredicate(&predicate)
            .to_vec()
            .iter()
            .map(|event| {
                let start = event.startDate();
                let attendees = event
                    .attendees()
                    .map(|list| list.to_vec())
                    .unwrap_or_default();
                // The reader's own answer, which is a fact about the reader. Every other
                // participant contributes only to the count.
                let yours = attendees
                    .iter()
                    .find(|participant| participant.isCurrentUser())
                    .map_or_else(
                        || "not an attendee".to_owned(),
                        |participant| participant_status(participant.participantStatus()),
                    );
                let row = format!(
                    "  {} .. {}  {:?}\n      calendar {:?}  {}{}\n      {} attendee(s), you: {}, organizer: {}",
                    local(&start),
                    local(&event.endDate()),
                    event.title().to_string(),
                    event
                        .calendar()
                        .map_or_else(String::new, |calendar| calendar.title().to_string()),
                    event_status(event.status()),
                    if event.isAllDay() { "  all-day" } else { "" },
                    attendees.len(),
                    yours,
                    if event.organizer().is_some() {
                        "present"
                    } else {
                        "absent"
                    },
                );
                (timestamp(&start).unwrap_or(now), row)
            })
            .collect()
    };

    if rows.is_empty() {
        println!("events within an hour of now: the store returned none");
        return 0;
    }

    rows.sort_by_key(|row| row.0);
    println!("events within an hour of now: {}", rows.len());
    for (_, row) in &rows {
        println!("{row}");
    }
    rows.len()
}

/// Runs one section with both ways out of it blocked, and reports rather than aborts.
///
/// Two nets, because there are two ways a framework call can leave without returning.
/// `objc2::exception::catch` turns an Objective-C raise into an error -- an uncaught one
/// aborts the process outright. It deliberately does not catch Rust panics, which is the
/// other way out: these bindings return non-optional `Retained<T>` for properties Apple
/// declares nonnull, and each panics rather than returning if the framework hands back nil.
///
/// A raise while listing calendars must not cost the event listing. This is a diagnostic,
/// and half an answer beats an abort.
///
/// Unwind safety is asserted because on either failure path every object the closure touched
/// is dropped unused and nothing it wrote is read afterwards.
fn guarded<R>(section: &str, body: impl FnOnce() -> R) -> Option<R> {
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        objc2::exception::catch(AssertUnwindSafe(body))
    }));

    match outcome {
        Ok(Ok(value)) => Some(value),
        Ok(Err(exception)) => {
            println!(
                "{section} raised: {}",
                render_exception(exception.as_deref())
            );
            None
        }
        Err(_) => {
            println!("{section} panicked; the message above says where");
            None
        }
    }
}

fn render_exception(exception: Option<&objc2::exception::Exception>) -> String {
    exception.map_or_else(
        || "a nil exception".to_owned(),
        |exception| exception.to_string(),
    )
}

/// `n calendar(s)`, or `calendars: failed` for a section that raised or panicked.
fn count(noun: &str, counted: Option<usize>) -> String {
    match counted {
        Some(n) => format!("{n} {noun}(s)"),
        None => format!("{noun}s: failed"),
    }
}

/// An `NSDate` rendered in the machine's own time zone, which is what Calendar.app shows and
/// therefore what this output has to be comparable against.
fn local(date: &NSDate) -> String {
    match timestamp(date) {
        Some(at) => at
            .to_zoned(TimeZone::system())
            .strftime("%Y-%m-%d %H:%M %Z")
            .to_string(),
        None => format!("<unconvertible {}>", date.timeIntervalSince1970()),
    }
}

fn timestamp(date: &NSDate) -> Option<Timestamp> {
    SignedDuration::try_from_secs_f64(date.timeIntervalSince1970())
        .ok()
        .and_then(|since_epoch| Timestamp::from_duration(since_epoch).ok())
}

fn calendar_type(kind: EKCalendarType) -> String {
    let name = match kind {
        EKCalendarType::Local => "Local",
        EKCalendarType::CalDAV => "CalDAV",
        EKCalendarType::Exchange => "Exchange",
        EKCalendarType::Subscription => "Subscription",
        EKCalendarType::Birthday => "Birthday",
        _ => "unknown",
    };
    format!("{name} ({})", kind.0)
}

fn source_type(kind: EKSourceType) -> String {
    let name = match kind {
        EKSourceType::Local => "Local",
        EKSourceType::Exchange => "Exchange",
        EKSourceType::CalDAV => "CalDAV",
        EKSourceType::MobileMe => "MobileMe",
        EKSourceType::Subscribed => "Subscribed",
        EKSourceType::Birthdays => "Birthdays",
        _ => "unknown",
    };
    format!("{name} ({})", kind.0)
}

fn event_status(status: EKEventStatus) -> String {
    let name = match status {
        EKEventStatus::None => "NoStatus",
        EKEventStatus::Confirmed => "Confirmed",
        EKEventStatus::Tentative => "Tentative",
        EKEventStatus::Canceled => "Canceled",
        _ => "unknown",
    };
    format!("{name} ({})", status.0)
}

fn participant_status(status: EKParticipantStatus) -> String {
    let name = match status {
        EKParticipantStatus::Unknown => "Unknown",
        EKParticipantStatus::Pending => "Pending",
        EKParticipantStatus::Accepted => "Accepted",
        EKParticipantStatus::Declined => "Declined",
        EKParticipantStatus::Tentative => "Tentative",
        EKParticipantStatus::Delegated => "Delegated",
        EKParticipantStatus::Completed => "Completed",
        EKParticipantStatus::InProcess => "InProcess",
        _ => "unknown",
    };
    format!("{name} ({})", status.0)
}
