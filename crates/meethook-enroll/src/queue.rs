//! Which voices a run asks about, in what order, and how a targeted run finds its one voice.
//!
//! The types here are the queue's data: where a voice sits ([`Position`]), how a row of the
//! queue reads ([`Queued`]), which voices get offered beyond the default ([`Offer`],
//! [`Sessions`]), and how a run narrowed to one voice arrived there ([`Selection`],
//! [`VoiceSelector`]). The decision functions below decide the list itself: [`queue`] walks a
//! whole session, and [`targeted`] and [`at_timestamp`] are its two siblings for a run aimed at
//! one voice.

use std::collections::BTreeMap;

use meethook_session::{
    DiscoveredSession, SpeakerCluster, Transcript, TranscriptTime, VoiceAt, unknown_speaker,
};
use meethook_transcribe::Attribution;

use crate::narration::{
    Narrator, Nearest, NotSelected, PassedOver, SessionNote, VoiceDescription, about,
};
use crate::{EnrollReport, MeetingLabel, Result};

#[cfg(doc)]
use crate::interview::{Answer, Interviewer};
#[cfg(doc)]
use crate::prompt::Voice;
#[cfg(doc)]
use crate::{Confirm, Enrolment, REFERENCE_FLOOR_SECONDS};
#[cfg(doc)]
use meethook_session::unknown_labels;

/// How much a voice has to have spoken before it is worth a question.
///
/// A rule about the prompt queue and nothing else. A cluster below this still keeps its
/// "Unknown N", still holds its turns, and is still relabelled when somebody else's answer
/// turns out to name it -- it is only not *asked about* unless `enroll --all` is passed.
/// Nothing on disk depends on it.
///
/// # Where 5 s comes from
///
/// Clustering emits one cluster per voice it is sure of plus a long tail of fragments it
/// cannot place, and the tail is not a tuning failure: a one-second embedding describes a
/// phoneme and a prosody rather than a person, so no distance rule puts it anywhere. On
/// session `20260810-093047` -- seven people, 1368.7 s of speech -- the shipped clustering
/// leaves 56 clusters, 8 of which identification resolves, so without a floor `enroll` asks
/// 48 questions about a meeting with seven people in it.
///
/// Sorted by talk time those 56 clusters run 426.8 / 423.7 / 124.8 / 119.5 / 96.0 / 51.5 --
/// the six voices the user confirms are the six main speakers, 1242.2 s between them -- and
/// then fall off a cliff to 8.6 / 8.5 / 7.8 / 7.5 / 6.0 / 5.6 / 5.4 / 4.9 / 4.2 / 3.9 / ...
/// into a tail where 29 of the 56 hold under two seconds and 126.5 s covers all fifty of them.
/// Of the 48 clusters left unresolved after identification, **every floor `f` with
/// `4.9 < f <= 5.9` offers the same seven voices and holds back the same 41**; over all 56
/// clusters, ignoring which happen to be enrolled, the partition is fixed across
/// `4.9 < f <= 5.4`. 5 s is the round number in that band rather than a value fitted to this
/// recording.
///
/// Both edges are consequences. Above 7.8 s Alex -- a real seventh participant,
/// 9.8 s of speech split across clusters of 7.83 s and 1.99 s -- stops being offered and can
/// only be reached through `--all`, which is the failure TASK-021 AC #3 names; he happens to
/// be enrolled already in this session, so the cost lands on the next participant like him.
/// Below it the tail arrives fast: 9 voices at a 4 s floor, 15 at 3 s, 21 at 2 s, which is the
/// 48-question prompt again with a smaller number on it.
///
/// # Not [`meethook_transcribe::SPEAKER_FLOOR_SECONDS`], and not [`REFERENCE_FLOOR_SECONDS`]
///
/// Same units, three different questions, and they do not imply one another:
///
/// - `SPEAKER_FLOOR_SECONDS` (30 s) decides **which clusters are solid enough to adopt
///   fragments into** -- how much evidence a centroid rests on before it is allowed to claim
///   somebody else's turns. It is necessarily the larger: at 30 s the seventh participant
///   would not be asked about at all.
/// - This one decides **which voices are worth asking about**. Getting it wrong costs a
///   question, in one direction or the other, and nothing else.
/// - [`REFERENCE_FLOOR_SECONDS`] decides **which answers become references in
///   `speakers.json`**. Naming somebody who spoke 8 s is right; storing a reference built from
///   8 s of audio is what TASK-019 measured going wrong. It landed on the same 5.0 s this one
///   sits on, which is why both state the same boundary convention below: a value offered here
///   and then refused there would be a question asked for nothing.
/// - [`meethook_transcribe::TENTATIVE_FLOOR_SECONDS`] decides **which fragments the tentative
///   band may guess at**, and it is this floor's own number: the band scores exactly the
///   voices held back here. It lives in `meethook-transcribe` because the band does, so the
///   parity is pinned by a test in `meethook-enroll` rather than by a shared definition.
///
/// The comparison is `speech_seconds >= PROMPT_FLOOR_SECONDS`, the same convention
/// `SPEAKER_FLOOR_SECONDS` states: a cluster sitting exactly on the floor is offered. Two
/// floors in one codebase disagreeing about their own boundary is a bug waiting to happen.
pub(crate) const PROMPT_FLOOR_SECONDS: f64 = 5.0;

/// Where a voice sits in the questions this run has for one session: "the 2nd of 9".
///
/// Reads as `2/9`. The point of it is that an interview otherwise has no visible end -- the
/// count printed on the session line has scrolled away behind the snippets and the clips by
/// the second or third question -- so every prompt carries the same number back.
///
/// Two things it deliberately is not:
///
/// - `of` counts the voices this run *offered for this session*, which is the number the
///   session line already printed. It is not a run-wide total, because that would mean reading
///   every session up front, and it does not include the voices held back under
///   `PROMPT_FLOOR_SECONDS`, which are reported on their own clause and are not questions this
///   run will ask.
/// - `nth` is the voice's place in that queue, not a tally of the questions actually asked. An
///   answer can name a voice further down the queue -- clustering splitting one person in two
///   -- and that voice is then passed over, so a number can be skipped: 1/4, 2/4, 4/4. The gap
///   is the honest reading, because it means `of - nth` is a true upper bound on the questions
///   left rather than a promise of more questions than the run will ask.
///
/// `nth` is fixed when the queue is built, so [`Answer::Later`] does not renumber anything: a
/// voice deferred at 3/9 comes back at 3/9, however many passes it takes. The alternative --
/// numbering the passes -- would have the same voice arrive as 3/9 and then 1/2, which reads as
/// a different voice in a different session rather than as the one question still open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// 1-based place in the queue this run offered for this session. Never greater than `of`.
    pub nth: usize,
    /// How many voices this run offered for this session.
    pub of: usize,
}

impl std::fmt::Display for Position {
    /// One place decides the form, so no two [`Interviewer`]s can disagree about it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.nth, self.of)
    }
}

/// One voice of a session as a queue pane lists it, which is not the same thing as a question.
///
/// Every voice the session has, including the ones this run is not asking about: the quiet
/// fragments held back under the prompt floor, and the ones the database has already named.
/// An interface that draws a queue needs all of them at once -- a pane showing only the voices
/// currently being asked about would leave the user unable to see that the two-second fragment
/// they are looking for exists at all -- whereas [`Voice`] is one question.
///
/// Four fields and no methods on purpose: it is a row, and what a row *reads like* belongs to
/// whatever is drawing it.
pub struct Queued<'a> {
    /// The "Unknown N" this voice was transcribed with, which does not move when it is named.
    /// The same handle [`Voice::number`] carries, so an interface can match a row against the
    /// question it is being asked.
    pub number: &'a str,

    /// What this voice currently reads as and on what basis, exactly as
    /// [`Voice::attribution`] means it -- and as the database and this run's answers stand
    /// right now, not as they stood when the session was opened.
    pub attribution: &'a Attribution,

    /// Total speech attributed to this voice, in seconds. What tells a participant from
    /// somebody who coughed once, and so what a queue is worth sorting or dimming by.
    pub speech_seconds: f64,

    /// Whether the prompt floor would have held this voice back -- so a queue can say why a
    /// row is not among this run's questions, and offer `--all` by name.
    ///
    /// A boolean rather than `PROMPT_FLOOR_SECONDS` made public: where the floor sits, and that
    /// a voice sitting exactly on it is offered, stay this library's decisions. An interface
    /// comparing its own copy of the number would be a second answer to the same question.
    ///
    /// True even under `--all`, which changes which voices are *asked about* and not which ones
    /// are quiet.
    pub below_floor: bool,
}

/// Which one voice a run is about, when it is about one voice.
///
/// `--voice`. The queue is the right shape for "I have not named anybody here yet" and the
/// wrong one for the commonest follow-up -- one voice the user can now place, or one name
/// that is wrong -- where reaching it means pressing Enter past everybody else, and every one
/// of those presses is a chance to type a name onto the wrong person.
///
/// # What it selects
///
/// One selector matched two ways, so the user does not have to know which kind of thing they
/// are holding:
///
/// - **A number** is the number in "Unknown 3", not the cluster id. The cluster id appears in
///   `transcript.json` and nowhere a person reads, while the "Unknown N" is on every prompt
///   header and every unnamed line of the transcript -- so accepting both would be two
///   numbering systems on one flag, silently targeting the wrong voice whenever they disagree.
///   The number comes from [`unknown_labels`], which ranks *every* voice by first appearance
///   whether or not it has a name, so it is defined for named voices too and does not move
///   when one of them is named.
/// - **A name** is what the voice currently reads as: the enrolled name that matched it, the
///   name somebody gave it for this session, or its own "Unknown 3" written out.
///
/// Matching is exact after trimming -- `alice` and `Alice` are two people here as everywhere
/// else in this file. A miss costs one retry, because it prints what the session does contain.
///
/// # What it overrides
///
/// Both [`Offer`] filters, for its one voice: a targeted voice is asked about whether it is
/// under `PROMPT_FLOOR_SECONDS` and whether the database has already named it. Naming somebody
/// specific is exactly the judgement those two gates make on the user's behalf when they have
/// not made it themselves. It does not touch [`Enrolment`], which is the other axis: what an
/// answer *writes* is the same however the question came to be asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceSelector(String);

impl VoiceSelector {
    /// Whether this selector means this voice, given the "Unknown N" it was transcribed with
    /// and what it currently reads as.
    ///
    /// Both arms, so that a number keeps pointing at the same voice after that voice has been
    /// named, and a name reaches a voice whose number the user never saw.
    fn matches(&self, unknown: &str, shown: &Attribution) -> bool {
        self.0 == unknown || self.0 == shown.label()
    }
}

impl From<&str> for VoiceSelector {
    /// Normalises to a label, so `3` and `Unknown 3` are the same selector from here on.
    ///
    /// Infallible: everything that is not a number is a name, and a name that matches nothing
    /// is reported against the session's actual voices rather than refused at the edge, where
    /// there is nothing to compare it to yet.
    fn from(raw: &str) -> VoiceSelector {
        let trimmed = raw.trim();
        match trimmed.parse::<usize>() {
            Ok(number) => VoiceSelector(unknown_speaker(number)),
            Err(_) => VoiceSelector(trimmed.to_string()),
        }
    }
}

impl std::fmt::Display for VoiceSelector {
    /// The normalised form, which is what was matched against: a user who passed `3` and
    /// missed is told that "Unknown 3" is what was looked for, beside the labels that exist.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// How a run that is about one voice arrived at that voice.
///
/// Two ways in, because there are two things a user is looking at when they want to name
/// somebody. [`Voice`](Self::Voice) is the prompt queue's own vocabulary -- "Unknown 3", or the
/// name a voice currently reads as -- and is right while the queue is on screen.
/// [`At`](Self::At) is the transcript's: a moment in the session, for the far commoner case of
/// somebody reading `transcript.md`, seeing that whoever spoke at 12:34 is Alice, and neither
/// knowing nor caring which Unknown number that voice ended up as.
///
/// One enum rather than two fields beside each other, so that "one voice, selected one way" is a
/// property of the type instead of a rule two `Option`s have to be checked against. What each
/// arm resolves *through* is different -- a label is compared against the session's voices, a
/// timestamp is looked up in its transcript -- but everything downstream of the resolution is
/// the same one voice, which is why this changes nothing about what an answer writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// `--voice`: the label the voice reads as. See [`VoiceSelector`].
    Voice(VoiceSelector),

    /// `--at`: the moment the voice was speaking at, in the `MM:SS` spelling `transcript.md`
    /// prints. Resolved through [`meethook_session::Transcript::voice_at`], which owns the rule
    /// for turning a printed label back into a turn.
    At(TranscriptTime),
}

impl Selection {
    /// The flag this arrived on, so a message about the request names what the user typed.
    pub(crate) fn flag(&self) -> &'static str {
        match self {
            Selection::Voice(_) => "--voice",
            Selection::At(_) => "--at",
        }
    }

    /// Why one session id and not several. Two different reasons, and a user who passed one flag
    /// is not helped by the other's.
    pub(crate) fn why_one_session(&self) -> &'static str {
        match self {
            Selection::Voice(_) => {
                "a voice belongs to one session, so its number and its name mean nothing across \
                 several"
            }
            Selection::At(_) => {
                "a timestamp is an offset into one recording, so it lands somewhere different in \
                 each of several"
            }
        }
    }
}

/// Which voices a run offers beyond the ones it offers by default.
///
/// Two orthogonal questions -- how quiet a voice may be, and whether the database has already
/// named it -- deliberately not one flag, because `--all` already answers the first and a user
/// who wants to correct one identification is not asking to be shown the two-second fragments
/// as well. The two filters compose: the floor decides whether a voice is worth a question
/// whatever put it in the list.
///
/// Both filters are overridden, for one voice, by a [`VoiceSelector`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Offer {
    /// `--all`: voices below `PROMPT_FLOOR_SECONDS`, which are normally held back.
    pub quiet: bool,

    /// `--correct`: voices the database has already put a name to, so a wrong identification
    /// can be answered instead of being permanent.
    pub named: bool,
}

/// Whether a session with nothing left unresolved is opened at all.
///
/// The other half of what [`Offer::named`] used to decide on its own, pulled out because they
/// are two questions: *which voices does a session offer*, which is `Offer`'s own subject, and
/// *is this session worth visiting*. They coincide for the two combinations the CLI shipped, and
/// come apart the moment an interface wants every voice in the queue pane -- widening `Offer`
/// for that would also, silently, have `meethook enroll` over a directory of finished sessions
/// open one on each of them.
///
/// An enum rather than a bool, following [`Enrolment`] and [`Confirm`]: at the call site
/// `Sessions::Every` says what it does, where `true` would need the parameter name to be read.
///
/// Nothing here applies to a run with a [`Selection`]: pointing at a voice or a moment has
/// already made this judgement, so neither of those paths has ever had this gate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Sessions {
    /// The default: pass over a session where every voice already carries a name, because there
    /// is no question left to ask about it.
    #[default]
    Unresolved,

    /// `--correct`: opened anyway. A session where nothing is unresolved is exactly where a
    /// wrong identification sits, so it is the one a user correcting an identification is
    /// reaching for.
    ///
    /// Not what the full-screen frame asks for, which is the whole point of this enum existing
    /// separately from [`Offer::named`]. The frame widens `Offer` so its queue pane can reach
    /// every voice in a session it *did* open; widening this as well would open one on every
    /// finished meeting on disk.
    Every,
}

/// The voices one session's run will ask about, in first-appearance order, and the line
/// saying so -- or `None` for a session with nothing to ask about, which has been reported
/// and counted.
///
/// Separated from the asking so that the one decision a [`VoiceSelector`] changes is made in
/// one place: [`targeted`] is the sibling of this, and everything downstream of both is shared.
#[allow(clippy::too_many_arguments)]
pub(crate) fn queue<'c>(
    order: &[&'c SpeakerCluster],
    shown: &BTreeMap<u32, Attribution>,
    offer: Offer,
    sessions: Sessions,
    meeting: Option<MeetingLabel>,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    // The one place "already named" is decided. Everything below -- the floor, the in-run
    // guard, the prompt -- treats a voice the same however it got into this list, which is
    // what lets `--all` and `--correct` compose without either knowing about the other.
    let candidates: Vec<&SpeakerCluster> = order
        .iter()
        .copied()
        .filter(|c| offer.named || !shown[&c.id].is_named())
        .collect();
    // The two halves of the pass-over, which used to be one: a session with no candidates at
    // all, and a session whose candidates are all already named. Only the second is
    // [`Sessions`]'s to overrule -- a session with no clusters in it has nothing to draw
    // however hard a caller asks -- which is why the emptiness test stays rather than folding
    // into the count.
    //
    // Behaviour is unchanged for the two combinations that predate the split, and the third is
    // the reason for it. With `offer.named` false -- the plain path -- every candidate is
    // unresolved, so `unresolved == 0` holds exactly when the list is empty and this is the
    // `candidates.is_empty()` gate it replaces. With `--correct` it is true alongside
    // `Sessions::Every`, so the count is not consulted. The full-screen frame is the third: it
    // sets `offer.named` so its queue pane can reach every voice, but leaves
    // `Sessions::Unresolved`, and this count is what then keeps it from opening on every
    // finished meeting on disk.
    let unresolved = candidates
        .iter()
        .filter(|c| !shown[&c.id].is_named())
        .count();
    if candidates.is_empty() || (unresolved == 0 && sessions == Sessions::Unresolved) {
        // A session whose voices are all identified is exactly where somebody stands when one
        // of those identifications is wrong, and this note is the only thing it produces -- so
        // it carries the count that reaches the escape, the way the held-back one names `--all`.
        let named = shown.values().filter(|label| label.is_named()).count();
        about(
            notes,
            &session.id,
            SessionNote::PassedOver(PassedOver::NothingUnresolved { named }),
        )?;
        report.passed_over += 1;
        return Ok(None);
    }

    let queued = candidates.len();

    // Only the voices worth a question, unless the user asked for the rest. Clustering emits a
    // long tail of one- and two-second fragments it cannot place -- 48 unresolved clusters for
    // a meeting of seven people, measured on `20260810-093047` -- and asking about each of
    // them is how a five-minute job becomes an hour. Filtering preserves first-appearance
    // order, which is what the user reads the transcript in.
    let mut offered: Vec<&SpeakerCluster> = if offer.quiet {
        candidates.clone()
    } else {
        candidates
            .iter()
            .copied()
            .filter(|c| c.speech_seconds >= PROMPT_FLOOR_SECONDS)
            .collect()
    };
    // A floor that hides every voice in a session is not a floor, it is a command that does
    // nothing. A short recording where nobody clears it -- the three-second fixtures the
    // end-to-end tests are built on, and any real meeting that ran for a minute -- offers
    // everybody instead. Decided here rather than defended against, because the alternative is
    // `enroll` reporting "nothing to do" on a session with unnamed people in it.
    if offered.is_empty() {
        offered = candidates;
    }
    let held_back = queued - offered.len();
    report.held_back += held_back;

    // `offered.len()` here is the same number every prompt below carries as its [`Position`]
    // total, because both read this list. Anything that computes this count independently
    // breaks that.
    //
    // `already_named` is `Some` exactly under `--correct`, which is what makes the queue a
    // review rather than a list of unknowns -- and so is what picks between the two wordings.
    about(
        notes,
        &session.id,
        SessionNote::Queue {
            offered: offered.len(),
            already_named: offer
                .named
                .then(|| offered.iter().filter(|c| shown[&c.id].is_named()).count()),
            held_back,
            meeting,
        },
    )?;

    Ok(Some(offered))
}

/// The one voice a [`VoiceSelector`] names, or `None` when it named none or several -- which
/// is reported and counted as a request that could not be served.
///
/// No floor, no `--correct` gate and no "nothing unresolved" pass-over: a user who named a
/// voice has already decided it is worth a question, and a session where everybody is already
/// named is exactly where `--voice "Alice"` gets used. Nothing is counted as held back either;
/// this run was aimed at one voice rather than filtered down to it, so a summary line offering
/// `--all` would be answering a question nobody asked.
pub(crate) fn targeted<'c>(
    selector: &VoiceSelector,
    order: &[&'c SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Attribution>,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    let matched: Vec<&SpeakerCluster> = order
        .iter()
        .copied()
        .filter(|c| selector.matches(&unknown[&c.id], &shown[&c.id]))
        .collect();

    let describe = |c: &SpeakerCluster| describe(c, unknown, shown);

    match matched.len() {
        1 => {
            about(
                notes,
                &session.id,
                SessionNote::Selected {
                    at: None,
                    voice: describe(matched[0]),
                },
            )?;
            Ok(Some(matched))
        }
        0 => {
            // Every voice, quiet ones included: a miss is usually a number off by one or a name
            // spelled as the user remembers it rather than as the transcript has it -- and the
            // quiet voices are exactly what somebody is reaching for when they miss. Fifty-odd
            // lines on a real session is still far cheaper than fifty-odd prompts.
            about(
                notes,
                &session.id,
                SessionNote::NotSelected(NotSelected::NoVoiceMatched {
                    selector,
                    voices: order.iter().copied().map(describe).collect(),
                }),
            )?;
            report.failed += 1;
            Ok(None)
        }
        _ => {
            about(
                notes,
                &session.id,
                SessionNote::NotSelected(NotSelected::SeveralVoicesMatched {
                    selector,
                    voices: matched.iter().copied().map(describe).collect(),
                }),
            )?;
            report.failed += 1;
            Ok(None)
        }
    }
}

/// How one voice reads in a message about several: the number it is reachable by, the name it
/// currently carries, and how much it spoke.
///
/// Shared by both selectors so that a list of candidates reads the same however the user missed:
/// the number is what the message hands back, and it has to be the same number in both. Which of
/// the three fields end up in a sentence, and in what order, belongs to the `narration` module.
pub(crate) fn describe(
    cluster: &SpeakerCluster,
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Attribution>,
) -> VoiceDescription {
    VoiceDescription {
        number: unknown[&cluster.id].clone(),
        label: shown[&cluster.id].label().to_string(),
        speech_seconds: cluster.speech_seconds,
    }
}

/// The one voice speaking at a moment of this session, or `None` when that moment names no voice
/// -- which is reported and counted as a request that could not be served.
///
/// The third sibling of [`queue`] and [`targeted`], and deliberately nothing more than that:
/// it produces the same one-element list they do, so a timestamp is a way of *arriving* at a
/// voice rather than a second way of enrolling one. Everything downstream -- the prompt, the
/// pre-flight, the refusal, the three writes -- is shared, which is what makes the reference
/// floor, the already-enrolled safeguards and the two transcript files behave here exactly as
/// they do everywhere else.
///
/// No floor, no `--correct` gate and no pass-over, for the reason [`targeted`] gives: pointing at
/// a moment is already the judgement those gates make on the user's behalf.
///
/// Every refusal names the timestamp back in the spelling it was given, so the line can be read
/// beside the transcript the user copied it from.
#[allow(clippy::too_many_arguments)]
pub(crate) fn at_timestamp<'c>(
    at: TranscriptTime,
    transcript: &Transcript,
    order: &[&'c SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Attribution>,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    // Each non-answer says which of them it was and what to do about it: they are four
    // different situations, and only one of them is the user's mistake.
    let voice = match transcript.voice_at(at) {
        VoiceAt::Cluster(id) => id,
        VoiceAt::LocalSpeaker => {
            missed(notes, session, NotSelected::OnTheMicrophone { at }, report)?;
            return Ok(None);
        }
        VoiceAt::NoCluster => {
            missed(notes, session, NotSelected::NoClusters { at }, report)?;
            return Ok(None);
        }
        VoiceAt::Silence => {
            // A miss here is usually a second or two off, and the user is holding the file with
            // the right timestamp in it, so the nearest turn is worth more than the refusal.
            let nearest = transcript
                .turns
                .iter()
                .min_by(|a, b| gap_to(a, at).total_cmp(&gap_to(b, at)))
                .map(|turn| Nearest {
                    speaker: turn.speaker.clone(),
                    at: TranscriptTime::of(turn.start),
                });
            missed(notes, session, NotSelected::Silence { at, nearest }, report)?;
            return Ok(None);
        }
        VoiceAt::PastEnd { last } => {
            missed(
                notes,
                session,
                NotSelected::PastEnd {
                    at,
                    last: TranscriptTime::of(last),
                },
                report,
            )?;
            return Ok(None);
        }
    };

    // Two voices can print the same label -- turns a fraction of a second apart round to the
    // same second -- and then the timestamp names neither of them on its own. That is a question
    // this command cannot answer for the user, so it hands back the thing that tells them apart,
    // exactly as an ambiguous `--voice` does.
    let candidates = transcript.clusters_at(at);
    if candidates.len() > 1 {
        // The count comes off the transcript rather than off the voices below, which are looked
        // up in `speaker_clusters.json`: a transcript naming a cluster that file no longer has
        // would otherwise be reported as fewer turns than it has.
        missed(
            notes,
            session,
            NotSelected::SeveralVoicesAt {
                at,
                count: candidates.len(),
                voices: candidates
                    .iter()
                    .filter_map(|id| order.iter().find(|c| c.id == *id))
                    .map(|c| describe(c, unknown, shown))
                    .collect(),
            },
            report,
        )?;
        return Ok(None);
    }

    // A voice the transcript names and the clusters file does not is the stale-file failure the
    // rest of this crate already has wording for, reached from the other side.
    let Some(cluster) = order.iter().copied().find(|c| c.id == voice) else {
        missed(
            notes,
            session,
            NotSelected::VoiceNotInClusters { at },
            report,
        )?;
        return Ok(None);
    };

    // The same note [`targeted`] produces, plus the moment it was reached by: the user named a
    // timestamp and gets told which voice that turned out to be, which is the one thing they
    // did not already know.
    about(
        notes,
        &session.id,
        SessionNote::Selected {
            at: Some(at),
            voice: describe(cluster, unknown, shown),
        },
    )?;
    Ok(Some(vec![cluster]))
}

/// A request that could not be served: the reason, and the one counter every one of them lands
/// in. Together, because the two have never come apart -- see [`EnrollReport::failed`].
pub(crate) fn missed(
    notes: &mut dyn Narrator,
    session: &DiscoveredSession,
    why: NotSelected<'_>,
    report: &mut EnrollReport,
) -> Result<()> {
    about(notes, &session.id, SessionNote::NotSelected(why))?;
    report.failed += 1;
    Ok(())
}

/// How far a turn is from an instant: zero while the instant is inside it.
pub(crate) fn gap_to(turn: &meethook_session::Turn, at: TranscriptTime) -> f64 {
    let instant = at.seconds();
    if instant < turn.start {
        turn.start - instant
    } else {
        (instant - turn.end).max(0.0)
    }
}
