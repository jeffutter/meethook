//! Deriving who the highlighted candidate already is, across every session the scan could read.
//!
//! Owns the scan-derived identity report: the [`Context`] the shell hands over, the [`Who`] a
//! highlighted candidate turns out to be, and the pure [`who`] pass between them -- arithmetic
//! over a borrowed [`Scan`] with no I/O behind it. The scan itself is gathered by the shell on a
//! thread of its own and arrives as data somebody else read; the state machine that highlights a
//! candidate and shows the answer lives in [`super::state`].

use std::collections::BTreeMap;

use meethook_enroll::Scan;
use meethook_session::SessionId;

/// What the cross-session scan has to say about the enrolled speakers, or why it has nothing to
/// say yet.
///
/// The seam that keeps the scan's I/O out of the state machine the way [`super::state::Costs`] keeps
/// [`Preview`](meethook_enroll::Preview)'s out: the shell gathers a [`Scan`] on a thread of its
/// own and hands it over as a borrow. Unlike `Costs` this needs no trait, because `Scan` and
/// everything under it is fully public with public fields, so a test can build one.
///
/// [`Context::Reading`] is the first fraction of a second of a run rather than an error, and
/// [`Context::Failed`] is a database that could not be read at all -- which is a pane that says so
/// and a frame that carries on answering, because every answer already given is on disk.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Context<'a> {
    Reading,
    Read(&'a Scan),
    Failed(&'a str),
}

/// Who the highlighted candidate turns out to be, across every session the scan could read.
///
/// The answer to "who is Ivan again?": how many recordings of them the database holds, and which
/// voices in which meetings read that name because of those recordings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Who {
    /// Nothing is highlighted, so there is nobody to be.
    Nobody,
    Reading,
    Failed(String),
    /// Enrolled since the scan ran -- named during this very run, and so absent from a snapshot
    /// taken before that answer. Its own case because the one wrong answer here that would look
    /// like a fact is "0 references, naming nothing".
    Unrecorded,
    Known {
        references: usize,
        /// Distinct voices those references are naming, across every session read.
        voices: usize,
        /// Most recent session first.
        sessions: Vec<Named>,
        /// Transcribed sessions the scan has no opinion about, so a reader knows the answer above
        /// is over less than everything on disk.
        unreadable: usize,
    },
}

/// One session a person's references are naming voices in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Named {
    pub session: String,
    /// The "Unknown N" of each voice reading this name here, deduped, in the scan's own order.
    pub voices: Vec<String>,
}

/// Who a candidate is, derived from one scan.
///
/// Matched on the name exactly and never fuzzily: [`super::state::Candidate::name`] is documented as the
/// `speakers.json` spelling, which is the same spelling [`Enrolled::name`](meethook_enroll::Enrolled)
/// carries, so anything looser could only turn one person into another. A name the scan does not
/// have is [`Who::Unrecorded`] rather than an empty `Known`, because `Scan::people` holds every
/// enrolled name at scan time even when it names nothing -- so absent means "enrolled since",
/// not "names nothing".
///
/// Only `depends` is read, never `elsewhere`: what a reference *names* is the question, and
/// `elsewhere` is what a removal would *also* move, which is a different one nobody asked here.
///
/// Deliberately not memoised, unlike [`super::state::Costs::of`]. This is a linear pass over a few dozen people
/// and a few hundred [`VoiceChange`](meethook_enroll::VoiceChange)s with no I/O behind it, so a
/// memo would buy nothing and would add an invalidation rule to get wrong -- and the scan under it
/// moves after every accepted answer.
///
/// One limitation, inherited from the scan rather than papered over: a voice named by hand in some
/// session's `speaker_names.json` never appears here, because no change to the database can move a
/// hand-given name and so the diff never reports one. `meethook speakers` says exactly as much,
/// which is the point -- diverging from the scan to look more complete would make the two disagree.
pub(crate) fn who(context: Context<'_>, highlighted: Option<&str>) -> Who {
    let Some(name) = highlighted else {
        return Who::Nobody;
    };
    let scan = match context {
        Context::Reading => return Who::Reading,
        Context::Failed(why) => return Who::Failed(why.to_string()),
        Context::Read(scan) => scan,
    };
    let Some(person) = scan.people.iter().find(|person| person.name == name) else {
        return Who::Unrecorded;
    };

    // Keyed by `SessionId`, which orders as `YYYYMMDD-HHMMSS` does, so the reverse of the map's
    // own order is chronological with the most recent meeting first.
    let mut by_session: BTreeMap<&SessionId, Vec<String>> = BTreeMap::new();
    for reference in &person.references {
        for change in &reference.depends {
            let voices = by_session.entry(&change.session).or_default();
            // Two of somebody's recordings can both be naming one voice. It is one voice.
            if !voices.contains(&change.voice) {
                voices.push(change.voice.clone());
            }
        }
    }

    Who::Known {
        references: person.references.len(),
        voices: by_session.values().map(Vec::len).sum(),
        sessions: by_session
            .into_iter()
            .rev()
            .map(|(session, voices)| Named {
                session: session.to_string(),
                voices,
            })
            .collect(),
        unreadable: scan.unreadable.len(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use meethook_enroll::{Enrolled, Queued, Reference, Scan, Unreadable, VoiceChange};
    use meethook_session::SessionId;
    use meethook_transcribe::{Attribution, Resemblance};

    use super::super::state::tests::{Free, queue, resembles, rows, session, view};
    use super::super::state::{Event, Screen, VoiceView};
    use super::{Context, Named, Who};

    /// One voice a reference is naming: `voice` in `session` reads `name` today and would go back
    /// to its number without this reference, which is the shape `scan` produces.
    pub fn names(session: &str, voice: &str, name: &str) -> VoiceChange {
        VoiceChange {
            session: SessionId::parse(session).expect("a well-formed session id"),
            voice: voice.to_string(),
            reads: name.to_string(),
            would_read: voice.to_string(),
        }
    }

    /// One enrolled person, described by what each of their references is naming. Handles are
    /// 1-based and in file order, as the scan numbers them.
    pub fn holding(name: &str, references: &[&[VoiceChange]]) -> Enrolled {
        Enrolled {
            name: name.to_string(),
            references: references
                .iter()
                .enumerate()
                .map(|(index, depends)| Reference {
                    handle: index + 1,
                    clip_seconds: None,
                    depends: depends.to_vec(),
                    elsewhere: Vec::new(),
                })
                .collect(),
        }
    }

    /// A scan as `meethook speakers` would have produced it, over three transcribed sessions of
    /// which `unreadable` had no opinion in them.
    pub fn scanned(people: Vec<Enrolled>, unreadable: &[&str]) -> Scan {
        Scan {
            people,
            sessions_found: 3,
            sessions_transcribed: 3,
            sessions_read: 3 - unreadable.len(),
            unreadable: unreadable
                .iter()
                .map(|session| Unreadable {
                    session: SessionId::parse(session).expect("a well-formed session id"),
                    why: "the clusters file is stale -- re-transcribe this session with --force"
                        .to_string(),
                })
                .collect(),
        }
    }

    /// A question about "Unknown 1", ranked against Milo alone, with Ivan enrolled and reachable
    /// only by typing. The fixture every "who is this" test below wants: one candidate the ranking
    /// has a count for and one it does not.
    fn asked<'a>(
        session: &'a SessionId,
        queue: &'a [Queued<'a>],
        similar: &'a [Resemblance],
        enrolled: &'a [&'a str],
        attribution: &'a Attribution,
    ) -> VoiceView<'a> {
        view(
            session,
            "Unknown 1",
            1,
            queue,
            &[],
            similar,
            enrolled,
            attribution,
        )
    }

    /// AC #1: the highlighted candidate says how many recordings of them the database holds --
    /// including for somebody the ranking has no count for at all, which is the case the candidate
    /// row can only draw as `--`.
    #[test]
    fn the_highlighted_candidate_says_how_many_references_they_hold() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo", "Ivan"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![
                holding(
                    "Milo",
                    &[&[names("20260819-100000", "Unknown 1", "Milo")], &[]],
                ),
                holding("Ivan", &[&[], &[], &[]]),
            ],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        let derived = screen.view(&voice, &Free, Context::Read(&found));
        assert_eq!(
            derived.who,
            Who::Known {
                references: 2,
                voices: 1,
                sessions: vec![Named {
                    session: "20260819-100000".to_string(),
                    voices: vec!["Unknown 1".to_string()],
                }],
                unreadable: 0,
            }
        );

        // Ivan is absent from the ranking, so the candidate row has no count to show. The scan has
        // one for everybody enrolled, which is what makes this pane worth its rows.
        for c in "Ivan".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        let derived = screen.view(&voice, &Free, Context::Read(&found));
        assert_eq!(derived.candidates[0].name, "Ivan");
        assert_eq!(
            derived.candidates[0].references, None,
            "the ranking has no count for somebody reachable only by typing"
        );
        assert!(
            matches!(derived.who, Who::Known { references: 3, .. }),
            "{:?}",
            derived.who
        );
    }

    /// AC #2: which sessions and which voices, grouped by meeting, most recent first, and one
    /// voice counted once however many of somebody's recordings are naming it.
    #[test]
    fn the_highlighted_candidate_says_which_sessions_and_voices_it_names() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![holding(
                "Milo",
                &[
                    &[
                        names("20260810-101500", "Unknown 1", "Milo"),
                        names("20260809-052600", "Unknown 3", "Milo"),
                    ],
                    &[
                        // The same voice as the first reference names, and one more beside it.
                        names("20260810-101500", "Unknown 1", "Milo"),
                        names("20260810-101500", "Unknown 4", "Milo"),
                    ],
                ],
            )],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 2,
                voices: 3,
                sessions: vec![
                    Named {
                        session: "20260810-101500".to_string(),
                        voices: vec!["Unknown 1".to_string(), "Unknown 4".to_string()],
                    },
                    Named {
                        session: "20260809-052600".to_string(),
                        voices: vec!["Unknown 3".to_string()],
                    },
                ],
                unreadable: 0,
            }
        );
    }

    /// The answer somebody at the reference cap is looking for, and the one that has to keep its
    /// scope clause: a reference naming nothing *in any session read* is not the same claim as one
    /// naming nothing.
    #[test]
    fn a_reference_naming_nothing_says_so_with_the_scope_it_was_read_against() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 2)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(vec![holding("Milo", &[&[], &[]])], &[]);
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 2,
                voices: 0,
                sessions: Vec::new(),
                unreadable: 0,
            }
        );
    }

    /// AC #5: a session the scan could not read leaves the answer incomplete and says so, rather
    /// than failing the run or quietly reporting over less than it claims.
    #[test]
    fn a_session_that_could_not_be_read_leaves_the_context_incomplete() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 1)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![holding(
                "Milo",
                &[&[names("20260810-101500", "Unknown 1", "Milo")]],
            )],
            &["20260809-052600"],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        // What it could read is still reported, which is the whole point of not failing.
        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 1,
                voices: 1,
                sessions: vec![Named {
                    session: "20260810-101500".to_string(),
                    voices: vec!["Unknown 1".to_string()],
                }],
                unreadable: 1,
            }
        );
    }

    /// The one wrong answer here that would look like a fact. Somebody enrolled during this very
    /// run is absent from a scan taken before that answer, and "0 reference(s), naming nothing"
    /// would be read as a person whose recordings do nothing -- which is the opposite of true.
    #[test]
    fn a_candidate_enrolled_during_this_run_is_not_reported_as_naming_nothing() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Maya", 0.64, 1)]);
        let enrolled = ["Maya"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        // The scan ran before Maya existed, so it knows only Milo.
        let found = scanned(
            vec![holding(
                "Milo",
                &[&[names("20260819-100000", "Unknown 2", "Milo")]],
            )],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Unrecorded
        );
    }

    /// The pane is about the highlighted candidate, so both things that move the highlight move
    /// it: the candidate keys and the filter. And with nothing highlighted there is nobody to
    /// report on rather than an empty listing.
    #[test]
    fn moving_the_highlight_and_typing_both_move_the_context() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        // Two people who share a prefix, so the filter below leaves both candidates standing and
        // what moves the pane is the highlight itself rather than the list emptying out.
        let similar = resembles(&[("Marco", 0.71, 2), ("Marcel", 0.38, 1)]);
        let enrolled = ["Marco", "Marcel"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![
                holding("Marco", &[&[], &[]]),
                holding(
                    "Marcel",
                    &[&[names("20260819-100000", "Unknown 2", "Marcel")]],
                ),
            ],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert!(
            matches!(
                screen.view(&voice, &Free, Context::Read(&found)).who,
                Who::Known { references: 2, .. }
            ),
            "the top of the ranking"
        );

        screen.answer(&voice, Event::CandidateDown, &Free);
        assert!(matches!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known { references: 1, .. }
        ));

        screen.answer(&voice, Event::Filter('M'), &Free);
        assert!(
            matches!(
                screen.view(&voice, &Free, Context::Read(&found)).who,
                Who::Known { references: 2, .. }
            ),
            "the filter moved the highlight back to Marco"
        );

        for c in "Quentin".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        let derived = screen.view(&voice, &Free, Context::Read(&found));
        assert!(derived.candidates.is_empty());
        assert_eq!(derived.who, Who::Nobody);
    }

    /// What a reference *names* is the question, and `elsewhere` is not an answer to it: those are
    /// the voices a removal would *also* move, which is what `meethook forget` asks and nothing on
    /// this frame does. Counting them here would put voices in the pane that do not read this name
    /// at all.
    #[test]
    fn elsewhere_is_not_what_a_reference_names() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 1)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![Enrolled {
                name: "Milo".to_string(),
                references: vec![Reference {
                    handle: 1,
                    clip_seconds: Some(42.5),
                    depends: Vec::new(),
                    // Removing this row would hand Ivan a name the veto is holding off. Real, and
                    // not what this row is naming.
                    elsewhere: vec![names("20260810-101500", "Unknown 7", "Ivan")],
                }],
            }],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 1,
                voices: 0,
                sessions: Vec::new(),
                unreadable: 0,
            }
        );
    }
}
