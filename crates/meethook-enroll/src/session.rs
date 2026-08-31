//! The per-session engine and the write path.
//!
//! [`enroll_session`] is the state machine: the passes over a shrinking queue, the dry run on
//! copies, and the fixed-order writes after every accepted name. The labelling pair
//! ([`effective_labels`], [`relabel`]) is shared with the `references` and `forget` paths,
//! because a transcript rewritten any other way would be a second producer of the labels
//! `merge` writes.

use std::collections::BTreeMap;

use meethook_session::{
    AssignedName, Classification, DiscoveredSession, EnrolledSpeakers, Paths, SessionMetadata,
    SourceTrack, SpeakerCluster, SpeakerClusters, SpeakerNames, Transcript, TranscriptContext,
    unknown_labels,
};
use meethook_transcribe::{
    Attribution, Naming, attributions, heard_at_once, identify_clusters, rank_enrolled,
    read_track_16k_mono, speaker_offset_seconds,
};

use crate::consequence::{Preview, Refusal, handle};
use crate::interview::{Answer, Interviewer, MeetingLabel};
use crate::narration::{
    self, AnswerNote, Narrator, PassedOver, SessionFile, SessionNote, about, after,
};
use crate::prompt::{Voice, clip_for, snippets_for};
use crate::queue::{
    PROMPT_FLOOR_SECONDS, Position, Queued, Selection, at_timestamp, queue, targeted,
};
use crate::{EnrollReport, EnrollRules, Outcome, Result};

#[cfg(doc)]
use crate::{forget, references};

/// Asks about every unresolved voice in one session, writing after each accepted name.
///
/// The files are written in a fixed order -- whichever of `speakers.json` and this session's
/// `speaker_names.json` the answer belongs in, then the transcript -- and after every single
/// name rather than once at the end. The name file is what the next labelling reads, so an
/// interrupt between the two writes leaves a name the next run simply re-applies, rather than
/// a transcript naming somebody nothing on disk records. It is also what makes ending a run
/// early cost nothing that was already answered.
///
/// A session this cannot read is reported and counted, and the queue carries on: one session
/// transcribed by a build too old to have recorded first appearances must not end the run.
pub(crate) fn enroll_session(
    paths: &Paths,
    session: &DiscoveredSession,
    rules: EnrollRules<'_>,
    speakers: &mut EnrolledSpeakers,
    interviewer: &mut dyn Interviewer,
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Outcome> {
    match session.classification {
        Classification::Orphaned => {
            about(
                notes,
                &session.id,
                SessionNote::PassedOver(PassedOver::Orphaned),
            )?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Valid => {
            about(
                notes,
                &session.id,
                SessionNote::PassedOver(PassedOver::NotTranscribed),
            )?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Transcribed => {}
    }

    let clusters = match SpeakerClusters::read(&session.paths.speaker_clusters_json()) {
        Ok(clusters) => clusters,
        // The expected instance of this is a `speaker_clusters.json` from before first
        // appearances were recorded: without them an "Unknown 2" cannot be mapped back to a
        // voice at all, so the file is refused rather than read with a defaulted zero.
        Err(e) => {
            unreadable(notes, session, SessionFile::Clusters, &e)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };
    // What a re-rendered `transcript.md` needs beyond the turns: the session's start time and
    // the meeting it was recorded during. Read here, beside the clusters, so a session whose
    // `session.json` has gone bad is reported and skipped like every other unreadable one
    // rather than ending the queue -- and so nothing is read inside the naming loop below,
    // where a failure would arrive after names had already been written.
    // Mutable for the one-remote-speaker assertion, which may land on disk here before any
    // voice is named.
    let mut metadata = match session.load_metadata() {
        Ok(metadata) => metadata,
        // No re-transcribe recovers this: `session.json` is the recorder's own output and the
        // marker that this directory is a session at all, so the only honest instruction is to
        // go and look at it.
        Err(e) => {
            unreadable(notes, session, SessionFile::Metadata, &e)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };
    // The meeting, projected to what a terminal may see and built here, beside the load: the
    // queue announcement below gets it from this value rather than any surface reaching back
    // for `session.json`, and nothing off the roster ever crosses with it. It is handed to the
    // voices too -- the frame shows it across the Interviewer seam -- so the announcement takes
    // a clone and the original outlives the asking loop.
    let meeting = metadata.meeting.as_ref().map(MeetingLabel::from);
    // The assertion, when the run carries one, is on disk before anything derived from it is
    // written: an interrupt between the two would leave a label the next run could not explain,
    // and the note below goes out before the first commit so every line that follows is already
    // readable against it. Writing only where the file differs keeps a re-run byte-identical.
    let mut assertion: Option<String> = None;
    if let Some(name) = rules.one_speaker {
        if metadata.one_remote_speaker.as_deref() != Some(name) {
            metadata.assert_one_remote_speaker(name.to_string());
            metadata.write(&session.paths.session_json())?;
        }
        assertion = Some(name.to_string());
        about(
            notes,
            &session.id,
            SessionNote::AssertingOneSpeaker {
                name,
                voices: clusters.clusters.len(),
            },
        )?;
    }
    let mut transcript = match Transcript::read(&session.paths.transcript_json()) {
        Ok(transcript) => transcript,
        // As above, and with the same remedy: the expected instance is a `transcript.json`
        // from before turns recorded which cluster they came from. A user told only "missing
        // field `cluster`" has been given a diagnosis with no next step.
        Err(e) => {
            unreadable(notes, session, SessionFile::Transcript, &e)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };

    // Voices somebody named in this session without enrolling them. Read here, beside the
    // clusters, so the relabel below already honours them -- a name given in an earlier run is
    // part of what this session's transcript should say, exactly as an enrolled name is.
    let mut assigned = match SpeakerNames::read_or_empty(&session.paths, &session.id) {
        Ok(assigned) => assigned,
        // Unlike the two failures above, no re-transcribe recovers this one: this file holds
        // names a person typed and nothing else can regenerate them, so the only honest
        // instruction is to go and look at it.
        Err(e) => {
            unreadable(notes, session, SessionFile::Names, &e)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };

    // The "Unknown N" numbering the transcript was written with, recovered from the clusters
    // file by the one function `transcribe` labels with. Fixed for the whole session: it is a
    // fact about when each voice first spoke, which no answer below changes.
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    // What each voice should be called given the database as it stands -- and, under an
    // assertion, the asserted person rather than whatever the rule would otherwise decide.
    let mut shown = effective_labels(
        &clusters.clusters,
        &unknown,
        speakers,
        &assigned.names,
        assertion.as_deref(),
        None,
        None,
    );

    // The transcript may predate an answer given in an earlier session -- name somebody in
    // January's meeting and February's transcript still calls them Unknown 2 -- so it is
    // brought in line before anything is asked. Doing it here rather than only after a name
    // is what stops a session with nothing left to ask about from keeping a stale label
    // forever, since it would be passed over on every later run too. Nothing is written when
    // nothing differs.
    //
    // Skipped under an assertion, because there it would write the transcript from the
    // assertion before the first commit had happened -- and nothing derived from the fact may
    // land before the first of its commits, so an interrupt between the two leaves a state
    // that explains itself. The skip is safe rather than lossy: assertion mode commits every
    // voice in the session, and each commit relabels the transcript through the assertion,
    // so a stale label cannot outlive the run the way it can in a session that gets passed
    // over.
    if assertion.is_none() && rules.relabel_transcript && relabel(&mut transcript, &shown) {
        transcript.write(
            &session.paths,
            rules.template,
            &TranscriptContext::now(&metadata),
        )?;
        about(notes, &session.id, SessionNote::BroughtUpToDate)?;
    }

    // First-appearance order, which is "Unknown 1, Unknown 2, ..." -- the order the user
    // reads the transcript in. Talk-time order would put the most-worth-naming voice first
    // and jump around relative to the file they are looking at.
    let mut order: Vec<&SpeakerCluster> = clusters.clusters.iter().collect();
    order.sort_by(|a, b| {
        a.first_spoke_seconds
            .total_cmp(&b.first_spoke_seconds)
            .then(a.id.cmp(&b.id))
    });

    // Which voices this run is about: one the user named, or the queue. The only thing a
    // selector changes -- everything from here down runs on whichever list comes back, so a
    // targeted prompt is not a second implementation of a prompt, it is the same one asked
    // about a shorter list. `None` is a session that is finished and has said why.
    // The assertion stands in for the queue and its gates alike -- every cluster, in the order
    // the user reads the transcript in, below the prompt floor included -- so it takes the
    // first branch rather than flowing through `queue`: none of the floor, the unresolved gate,
    // or the pass-over applies to a session the user has just said is one person.
    let offered = if assertion.is_some() {
        // Every cluster, quiet ones included: the assertion is about the whole track, so the
        // queue is the session itself rather than a filtered view of it.
        Some(order.to_vec())
    } else {
        match &rules.selector {
            Some(Selection::Voice(selector)) => {
                targeted(selector, &order, &unknown, &shown, session, notes, report)?
            }
            Some(Selection::At(at)) => at_timestamp(
                *at,
                &transcript,
                &order,
                &unknown,
                &shown,
                session,
                notes,
                report,
            )?,
            None => queue(
                &order,
                &shown,
                rules.offer,
                rules.sessions,
                meeting.clone(),
                session,
                notes,
                report,
            )?,
        }
    };
    let Some(offered) = offered else {
        return Ok(Outcome::Finished);
    };

    // Read after that check, so a session with nothing to ask about never resamples an hour
    // of audio in order to then ask nothing. Unreadable is empty rather than fatal: a voice
    // with no clip can still be named from its snippets.
    let track = read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();

    // Turn times are on the session timeline; a snippet's are offsets into `speaker.wav`.
    // `Err` is a degenerate timebase in `session.json`: the two clocks cannot be related at
    // all, so the snippets get no audio rather than audio a second out -- the same tolerance
    // an unreadable `speaker.wav` already gets a line above, and for the same reason. `clip`
    // is unaffected either way, because a representative's seconds are already track time.
    let offset = speaker_offset_seconds(&metadata);
    let snippet_track: &[f32] = if offset.is_ok() { &track } else { &[] };
    let offset = offset.unwrap_or(0.0);

    // What each voice was called when this queue was built. The guard below compares against
    // *this* rather than against the live labels, because under `--correct` a queued voice may
    // legitimately be one the database had already named.
    let baseline = shown.clone();

    // The total every prompt below carries. Read off the same list the session line counted
    // one call ago, so the two cannot drift apart.
    let of = offered.len();

    // The queue, numbered once. `nth` travels with the voice from here on rather than being
    // counted per pass, which is what makes a deferred voice come back as the same question --
    // see [`Position`].
    let mut pending: Vec<(usize, &SpeakerCluster)> = offered
        .into_iter()
        .enumerate()
        .map(|(index, cluster)| (index + 1, cluster))
        .collect();

    // Assertion-mode bookkeeping. `committed` is every voice this run has put a name on so far,
    // in commit order: the override report below is O(1) per voice off this list and each
    // cluster's own overlap data -- the veto predicate is exactly "heard at once with a holder
    // of the name", and under the assertion every committed voice holds the same name. The two
    // counters are session-local because the summary line says what the assertion did *here*;
    // the run-wide halves live in `report`.
    let mut committed: Vec<&SpeakerCluster> = Vec::new();
    let mut asserted_voices = 0;
    let mut vetoes_overridden = 0;

    // Passes over a shrinking list rather than one walk, so that [`Answer::Later`] can put a
    // voice back. Each pass asks about whatever is still pending; anything deferred is asked
    // again on the next one.
    loop {
        let asked = pending.len();
        let mut deferred: Vec<(usize, &SpeakerCluster)> = Vec::new();

        // Iterated by reference rather than consumed, because [`Answer::Leave`] has to reach the
        // tail of the queue -- the voices this pass has not asked about yet -- from inside the
        // body, and a consuming `for` has thrown them away by then. `(usize, &SpeakerCluster)` is
        // `Copy`, so the pattern costs nothing, and nothing in the body mutates `pending`: it is
        // reassigned only below the loop.
        for (index, &(nth, cluster)) in pending.iter().enumerate() {
            // A voice an answer given earlier in this run has already put a name to: clustering
            // that split one person in two must not ask about them twice. Only an in-run answer
            // can have moved a label since `baseline` was taken, so "named, and not the name it
            // had when we queued it" is exactly that case and nothing else. The `is_named()`
            // half matters as much: an in-run answer can also *un*-name a voice -- re-anchoring a
            // reference to another cluster drops this one back to its "Unknown N" -- and that is a
            // question this run created and has not answered.
            //
            // This `continue` is the one place a [`Position`] skips a number, and it is skipped
            // rather than compressed on purpose: the voice really was in the queue and really is
            // now answered, so the gap says work disappeared and the end came closer.
            //
            // Never under an assertion: there, the first commit lands the asserted name on
            // every voice at once, and reading that as "already answered" would skip the rest
            // of the track -- the opposite of what the assertion asks. Each voice still needs
            // its own row and its own reference offer, so the walk commits all of them.
            //
            // Or a voice this run has committed through any door: a group member re-named
            // identically to what it read when the queue was built (shown == baseline) is
            // answered too, and only `committed` says so.
            if assertion.is_none()
                && (shown[&cluster.id].is_named() && shown[&cluster.id] != baseline[&cluster.id]
                    || committed.iter().any(|done| done.id == cluster.id))
            {
                continue;
            }

            // Under the assertion there is no question at all: the assertion is the answer for
            // every voice, which is what makes the run prompt-free -- so the seam is not reached
            // and nothing below depends on the answerer from here on.
            let answer = if let Some(asserted) = assertion.as_deref() {
                Answer::Named {
                    name: asserted.to_string(),
                    anyway: false,
                }
            } else {
                // Scoped so the borrows of `transcript` and `shown` inside the voice end before
                // the answer is acted on.
                let answer = {
                    let attribution = &shown[&cluster.id];
                    // Keyed on the cluster, not on the label text: under `--correct` two voices can
                    // sit under one enrolled name -- which is the false accept being corrected -- and
                    // a prompt showing the other person's lines cannot be answered.
                    let snippets = snippets_for(&transcript, cluster.id, snippet_track, offset);

                    // Every voice in the session, built here and now rather than once above the
                    // loop, for the reason `Voice::queue` gives and because the borrow checker
                    // insists: `shown` is *reassigned* at the end of an accepted answer, so rows
                    // borrowing it cannot outlive one question. That is the same thing as the rows
                    // being current, which is why no separate refresh exists.
                    //
                    // `order` and not `pending`: a queue pane is the session, so the quiet voices
                    // and the already-named ones are in it whether or not this run asks about them.
                    let rows: Vec<Queued<'_>> = order
                        .iter()
                        .map(|c| Queued {
                            number: &unknown[&c.id],
                            attribution: &shown[&c.id],
                            speech_seconds: c.speech_seconds,
                            // Strictly less than the floor: a cluster sitting exactly on it is
                            // offered, which is the convention every floor in this codebase states.
                            below_floor: c.speech_seconds < PROMPT_FLOOR_SECONDS,
                        })
                        .collect();

                    // Computed eagerly for every voice, including on the `--name` path where nothing
                    // reads it, and deliberately not deferred behind a closure: two dozen people at
                    // 256 dimensions is a few thousand multiply-adds, against a run that has already
                    // read and resampled the whole speaker track a few lines above. An owned `Vec`
                    // rather than a borrow of the database, so the reborrow ends here and nothing
                    // downstream -- least of all an `Interviewer` -- has to reason about the write
                    // that replaces `speakers` once this answer is accepted.
                    interviewer.identify(&Voice {
                        session: &session.id,
                        meeting: meeting.as_ref(),
                        position: Position { nth, of },
                        attribution,
                        number: &unknown[&cluster.id],
                        speech_seconds: cluster.speech_seconds,
                        queue: &rows,
                        snippets,
                        clip: clip_for(&track, cluster),
                        resembles: rank_enrolled(&cluster.embedding, speakers),
                        // The universe `resolve()` requires, and not the ranking above -- see
                        // `Voice::enrolled`. Owned borrows, like `resembles`, so the reborrow of
                        // the database ends with this block.
                        enrolled: speakers.enrolled_names(),
                        // Six borrows and no work: what an answer would do is computed only if the
                        // answerer asks. Nothing is written by asking, so this is safe to hand out
                        // even though it holds the database -- see `Voice::preview`.
                        preview: Preview::new(
                            &clusters.clusters,
                            &unknown,
                            speakers,
                            &assigned,
                            cluster,
                            rules.enrolment,
                            None,
                            &committed[..],
                        ),
                    })
                };
                match answer {
                    // The frame's half of the assertion. The fact goes to disk before the first
                    // commit below -- the interrupt rule the headless path states -- and from
                    // here on this voice and every voice left in the queue are answered with the
                    // asserted name rather than asked about.
                    Answer::OneSpeaker(raw) => {
                        let raw = raw.trim();
                        if raw.is_empty() {
                            // A name of nothing but spaces is the question going unanswered, the
                            // same way a blank typed answer is: counted, and nothing written.
                            left_unanswered(std::iter::once(cluster), &shown, &baseline, report);
                            continue;
                        }
                        if metadata.one_remote_speaker.as_deref() != Some(raw) {
                            metadata.assert_one_remote_speaker(raw.to_string());
                            metadata.write(&session.paths.session_json())?;
                        }
                        assertion = Some(raw.to_string());
                        about(
                            notes,
                            &session.id,
                            SessionNote::AssertingOneSpeaker {
                                name: raw,
                                voices: order.len(),
                            },
                        )?;
                        // The assertion is about the whole track, not the queue this run was
                        // offering: a default run only offers the voices above the floor, and
                        // the headless flag reaches the quiet ones alike. Widen the walk to
                        // every voice not already committed and not already queued, through
                        // the deferral set -- the one place the loop already knows how to pick
                        // up again on the next pass -- so both doors into this mode land the
                        // same state.
                        for c in order.iter() {
                            if c.id != cluster.id
                                && !committed.iter().any(|done| done.id == c.id)
                                && !pending.iter().any(|&(_, queued)| queued.id == c.id)
                                && !deferred.iter().any(|&(_, held)| held.id == c.id)
                            {
                                deferred.push((0, c));
                            }
                        }
                        Answer::Named {
                            name: raw.to_string(),
                            anyway: false,
                        }
                    }
                    other => other,
                }
            };

            // Decided after the switch above, so the voice the key was pressed on counts too:
            // the assertion names it as well as the ones after it.
            let under_assertion = assertion.is_some();

            match answer {
                // Consumed by the assertion switch above: it is either gone or already turned
                // into a [`Answer::Named`]. Present because Rust checks exhaustiveness against
                // the variant set rather than against what that switch left.
                Answer::OneSpeaker(_) => {
                    debug_assert!(false, "OneSpeaker survives past the assertion switch");
                    unreachable!()
                }
                Answer::Quit => return Ok(Outcome::Quit),
                // The rest of this session, in the three groups it comes in and no fourth: the
                // voice that was on the screen when the key was pressed -- asked about, and
                // decided against, which is the same thing a deferral with no later turns out to
                // be -- then the ones this pass has not reached, then the ones it has already
                // deferred. Voices the guard above took out of the pass are in none of them,
                // which is what makes the counts add up.
                //
                // Returning from inside the loop is the whole implementation of leaving: every
                // write already happened per accepted name, and there is nothing between here and
                // the end of the function, so nothing is skipped by going early.
                Answer::Leave => {
                    let rest = std::iter::once(cluster)
                        .chain(pending[index + 1..].iter().map(|&(_, c)| c))
                        .chain(deferred.iter().map(|&(_, c)| c));
                    let left = left_unanswered(rest, &shown, &baseline, report);
                    about(notes, &session.id, SessionNote::Left { left })?;
                    return Ok(Outcome::Finished);
                }
                Answer::Skip => {
                    left_unanswered(std::iter::once(cluster), &shown, &baseline, report);
                    continue;
                }
                // Back into the queue with the number it already has, and counted as nothing:
                // it has not been answered yet. The pass that finds nobody willing to answer is
                // where these turn into skips -- see the fixed point below.
                Answer::Later => {
                    deferred.push((nth, cluster));
                    continue;
                }
                // One voice through the dry run and the fixed-order writes, exactly as before:
                // no forcing, so a preview and a write see it identically. `continue` rather
                // than falling through because every arm of this match now acts and moves on,
                // and the body of the pass ends right here.
                Answer::Named { name, anyway } => {
                    commit_named(
                        &clusters.clusters,
                        &unknown,
                        speakers,
                        &mut assigned,
                        &mut transcript,
                        &mut shown,
                        &baseline,
                        &mut committed,
                        report,
                        &mut vetoes_overridden,
                        &mut asserted_voices,
                        notes,
                        session,
                        paths,
                        &rules,
                        &metadata,
                        assertion.as_deref(),
                        cluster,
                        name,
                        anyway,
                        under_assertion,
                        None,
                    )?;
                    continue;
                }
                // A user-chosen group of voices named together with one name: the generalisation
                // of the one-remote-speaker assertion from the whole track to a subset the user
                // marked. Resolved and walked in queue order through the same commit a single
                // naming uses, with the growing forced set the aggregate preview folds over, so
                // a preview and a write cannot disagree about sequence or cost. Only reached when
                // there is no assertion -- under one the seam is never consulted -- so `anyway`
                // and `under_assertion` are both `false` here: a group names its members anyway
                // at the veto (that is its authority) but never overrides a `Taken` refusal.
                Answer::Group { name, members } => {
                    // Trimmed once here rather than in each member's dry run, so the forced
                    // map's key and the names committed agree on the same normalisation.
                    let group_name = name.trim().to_string();
                    let members =
                        resolve_group_members(&clusters.clusters, &unknown, &group_name, &members);
                    match members {
                        // The group is real: walk it in queue order, committing each member
                        // through the same path a single naming uses. One context for the whole
                        // walk rather than one per member, since nothing between members touches
                        // the files the context borrows.
                        Some(resolved) => {
                            // Veto authority iff the group names two or more voices: one member
                            // is a plain naming, and the veto refuses it exactly as today.
                            let authority = resolved.len() >= 2;
                            // The declared members committed so far, keyed to the group's name:
                            // the forced tier each member's dry run honours, grown as members
                            // land and shrunken back when one is refused -- the fold
                            // `Preview::group` applies, mirrored here so the two agree.
                            let mut forced: BTreeMap<u32, String> = BTreeMap::new();
                            for member in &resolved {
                                // Answered earlier in this run, by any door: counted then, not
                                // committed again -- and not inserted into the forced set,
                                // because the tier claims the voice reads the group's name, and
                                // a voice committed under another name does not.
                                if committed.iter().any(|done| done.id == member.id) {
                                    continue;
                                }
                                if authority {
                                    forced.insert(member.id, group_name.clone());
                                }
                                let fref: Option<&BTreeMap<u32, String>> =
                                    if authority { Some(&forced) } else { None };
                                let outcome = commit_named(
                                    &clusters.clusters,
                                    &unknown,
                                    speakers,
                                    &mut assigned,
                                    &mut transcript,
                                    &mut shown,
                                    &baseline,
                                    &mut committed,
                                    report,
                                    &mut vetoes_overridden,
                                    &mut asserted_voices,
                                    notes,
                                    session,
                                    paths,
                                    &rules,
                                    &metadata,
                                    assertion.as_deref(),
                                    member,
                                    group_name.clone(),
                                    false,
                                    false,
                                    fref,
                                )?;
                                // A refused member is not committed, so it stops being a
                                // declared holder for the members after it -- the state the run
                                // actually reaches, and the one the next member previews against.
                                if authority && !matches!(outcome, CommitOutcome::Committed) {
                                    forced.remove(&member.id);
                                }
                            }
                            // The voice the answer was given on, if the group did not name it:
                            // asked, went unanswered, counted the way every other unanswered
                            // voice is -- the group walked its members, not the anchor.
                            if !resolved.iter().any(|m| m.id == cluster.id) {
                                left_unanswered(
                                    std::iter::once(cluster),
                                    &shown,
                                    &baseline,
                                    report,
                                );
                            }
                        }
                        // A group that cannot say who its members are -- a blank name, an empty
                        // mark, or a handle nothing resolves -- goes unanswered rather than
                        // partially answered: the blank-name precedent reached with a different
                        // input, counted the way every other unanswered voice is.
                        None => {
                            left_unanswered(std::iter::once(cluster), &shown, &baseline, report);
                        }
                    }
                    continue;
                }
            }
        }

        // The fixed point: a pass that moved nobody out of the deferred set, *and* an answerer
        // that says it has nothing left to do. Every other pass leaves `deferred.len() < asked`,
        // and the set can only shrink, so the first half terminates on its own; the second is
        // [`Interviewer::still_working`]'s contract to keep bounded.
        //
        // The size of the set and not "no answer other than `Later` came back", because the
        // in-run guard above takes a voice out of a pass without any answer being given -- an
        // earlier answer named it -- and that is progress too. Counting answers would end a
        // session while there were still questions the user had not been asked.
        //
        // And the answerer as well as the set, because a stalled pass is not the same fact as a
        // finished session for an interface with a cursor: it defers voices in order to reach
        // another one, so a pass where the user only moved around produces no answer and is not
        // the user being done. An empty queue is decided here rather than there -- nothing is
        // left to offer, so no further prompt could change the answer or carry an
        // [`Answer::Quit`], and consulting the answerer could only spin.
        //
        // Still only about a pass that produced no answer: [`Answer::Leave`] is an answer and
        // returns above this, so leaving a session never reaches the question this asks.
        if deferred.len() == asked && (asked == 0 || !interviewer.still_working()) {
            // Deferred with no later left is the skip -- or the kept identification -- it has
            // turned out to be, counted through the same rule every other unanswered voice goes
            // through so no two of them can disagree about which bucket a named voice is in.
            left_unanswered(
                deferred.iter().map(|&(_, cluster)| cluster),
                &shown,
                &baseline,
                report,
            );
            break;
        }
        pending = deferred;
    }

    // What the assertion came to, said once for the session: the per-voice lines carry the
    // detail, this carries the shape of it, and the reference count is read off the database
    // the run left rather than re-derived from the commits -- the stated rule, D4 of the
    // plan, is that the existing cap does the bounding, so the count it holds is the answer.
    // The run-wide half of the session-local count: the summary line reads off the local
    // because it says what the assertion did *here*, and the report carries it for the caller,
    // which has no other view of the run. A group's overrides feed the same counter; with
    // neither assertion nor group the addend is zero.
    report.vetoes_overridden += vetoes_overridden;
    if let Some(name) = assertion.as_deref() {
        about(
            notes,
            &session.id,
            SessionNote::OneSpeakerSummary {
                name,
                voices: asserted_voices,
                vetoes_overridden,
                references_stored: speakers.references(name),
            },
        )?;
    }

    Ok(Outcome::Finished)
}

/// What committing one voice through the dry run and the fixed-order writes left behind.
///
/// Three outcomes, and only three: a name landed and was written, a refusal declined it and
/// nothing was written, or the name was not one at all and the voice is counted as unanswered.
/// The caller -- the ordinary queue walk for a single [`Answer::Named`], or the group walk for
/// a member of [`Answer::Group`] -- decides what each means to it, which is why the group can
/// carry on past a refused member while a single answer simply moves to the next voice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CommitOutcome {
    /// Named and written; the database, the names file, and the transcript are in line.
    Committed,
    /// Refused; counted, nothing written, the voice keeps whatever it read.
    Refused,
    /// A name of nothing but spaces; counted as an unanswered voice, nothing written.
    Unanswered,
}

/// Names one voice and writes what it wrote, working out everything on copies first.
///
/// One function rather than two implementations of the same mutation sequence is the module
/// invariant the `consequence` doc states, reached from the write side: a preview and a write
/// that disagreed would be two producers of one rule, and the symptom of those disagreeing is a
/// name on the wrong person's turns. The ordinary queue walk calls it once per [`Answer::Named`];
/// the group walk calls it once per member of [`Answer::Group`]. Every file it touches is passed
/// as a reference scoped to this call, so the borrow ends the moment a commit returns and the
/// next commit -- or the group's next member -- sees the state this one left.
///
/// The dry run the `consequence` module holds, and the same one an [`Interviewer`] may have
/// already run through `Voice::preview`: two files can carry a name and both feed the same
/// labelling, so the only way to know what an answer *does* is to build the state it would leave
/// and label the session through it; and the answer is not simply written and inspected
/// afterwards because undoing a write that turned out to cost somebody their name means writing
/// three files back, with a run interrupted mid-undo leaving exactly the mess this prevents.
///
/// `name` is trimmed here, the same normalisation [`crate::GivenName`] applies on the way in, so
/// a typed answer and one supplied up front are decided identically; `None` from the dry run is
/// a name of nothing but spaces, and that is a skip rather than an answer. `under_assertion`
/// and `forced` are the two ways a naming outranks the heard-at-once veto, and they are mutually
/// exclusive: the assertion names the whole track, the group a user-chosen subset of it, and a
/// run never carries both. `anyway` overrides a `Taken` refusal only, and only the single-voice
/// path ever sets it -- a group names its members at the veto (that is its authority) but never
/// pays a third party's name to do it.
#[allow(clippy::too_many_arguments)]
fn commit_named<'d, 'm>(
    clusters: &'d [SpeakerCluster],
    unknown: &'d BTreeMap<u32, String>,
    speakers: &'m mut EnrolledSpeakers,
    assigned: &'m mut SpeakerNames,
    transcript: &'m mut Transcript,
    shown: &'m mut BTreeMap<u32, Attribution>,
    baseline: &'d BTreeMap<u32, Attribution>,
    committed: &'m mut Vec<&'d SpeakerCluster>,
    report: &'m mut EnrollReport,
    vetoes_overridden: &'m mut usize,
    asserted_voices: &'m mut usize,
    notes: &'m mut dyn Narrator,
    session: &'d DiscoveredSession,
    paths: &'d Paths,
    rules: &'d EnrollRules<'d>,
    metadata: &SessionMetadata,
    assertion: Option<&str>,
    cluster: &'d SpeakerCluster,
    name: String,
    anyway: bool,
    under_assertion: bool,
    forced: Option<&BTreeMap<u32, String>>,
) -> Result<CommitOutcome> {
    // Built here rather than held from the prompt above because the commit below needs
    // `speakers` mutably and a live `Preview` would keep it borrowed. It is the same six
    // references and the same `of`, so the preview an answerer saw and the write cannot
    // disagree. `with_forced` threads the group's declared members through every labelling the
    // dry run does, so a member previewed against the running state honours the same tier the
    // write applies; `None` on the ordinary path labels exactly as before.
    let Some(consequence) = Preview::new(
        clusters,
        unknown,
        speakers,
        assigned,
        cluster,
        rules.enrolment,
        assertion,
        &committed[..],
    )
    .with_forced(forced)
    .of(&name) else {
        left_unanswered(std::iter::once(cluster), shown, baseline, report);
        return Ok(CommitOutcome::Unanswered);
    };
    let name = name.trim();

    // The refusal. An answer that would take a name off a voice the user is not answering
    // about is not honoured -- see `Refusal` for the three ways that can happen and why one
    // check covers them -- unless the answer itself says otherwise. Written as one total match
    // rather than as a guard plus an exception, because the rule is which of the three cases an
    // answer falls into and reading it should not require holding a negation.
    match &consequence.refused {
        // Shown what it costs and asked for it anyway. `Answer::anyway` is only ever set by an
        // interface that displayed the paying voice and what it loses before a key was pressed,
        // which makes this `forget --yes`'s argument reached from the other side: see
        // `forget.rs`'s "Nothing is ever refused". Everything below runs exactly as it does for
        // an answer nothing refused -- honouring an override is skipping this guard, not a
        // second write path.
        Some(Refusal::Taken { voice, losing }) if anyway => {
            after(
                notes,
                &session.id,
                AnswerNote::Overrode {
                    name,
                    answered: &handle(cluster.id, unknown),
                    voice,
                    losing,
                },
            )?;
        }
        // Every other refusal: a `Taken` nobody insisted on, and a `Vetoed` however insistent
        // the answer was. Nothing is written, the voice keeps whatever it read, and the note
        // names the voice that would have paid.
        Some(refusal) => {
            let answered = handle(cluster.id, unknown);
            after(
                notes,
                &session.id,
                AnswerNote::Refused {
                    name,
                    voice: &answered,
                    refusal,
                },
            )?;
            report.refused += 1;
            return Ok(CommitOutcome::Refused);
        }
        None => {}
    }

    // The override report, and the only line a veto produces in assertion mode: this voice was
    // heard at once with a voice the run has already put a name on, which is exactly the pair
    // the heard-at-once rule refuses to put under one name. It is named anyway, and said so here
    // rather than silently overridden -- naming the voices it overlapped, which is the evidence
    // the veto would have acted on. No refusal ever arises on this path, because the labelling
    // the dry run uses honours the assertion: the guard above finds the name where it belongs
    // and declines nothing.
    if under_assertion {
        let overlapped: Vec<String> = committed
            .iter()
            .filter(|c| heard_at_once(cluster, c))
            .map(|c| handle(c.id, unknown))
            .collect();
        if !overlapped.is_empty() {
            let answered = handle(cluster.id, unknown);
            after(
                notes,
                &session.id,
                AnswerNote::VetoOverridden {
                    name,
                    answered: &answered,
                    speech_seconds: cluster.speech_seconds,
                    overlapped: &overlapped,
                },
            )?;
            *vetoes_overridden += 1;
        }
    } else if let Some(forced_set) = forced {
        // The group's half of the same report: this member was heard at once with a voice that
        // already holds the group's name, and the group's authority named it anyway. Measured
        // against the holders of the name in the running pre-state labelling under the
        // previous forced set -- the declared members committed so far, minus this one --
        // which is exactly the count `Preview::group` folds, so a preview and a write report
        // the same override. The pre-state labelling rather than `shown`: a member committed
        // earlier in the walk can hold the name only through the tier -- the exclusion takes
        // it off again in the unforced relabel -- and `shown` no longer shows it there.
        let mut previous_forced: BTreeMap<u32, String> = forced_set.clone();
        previous_forced.remove(&cluster.id);
        let pre = effective_labels(
            clusters,
            unknown,
            speakers,
            &assigned.names,
            assertion,
            Some(&previous_forced),
            None,
        );
        let mut overlapped: Vec<String> = Vec::new();
        for (id, label) in pre.iter() {
            if *id != cluster.id
                && label.label() == name
                && clusters
                    .iter()
                    .any(|c| c.id == *id && heard_at_once(cluster, c))
            {
                overlapped.push(handle(*id, unknown));
            }
        }
        if !overlapped.is_empty() {
            let answered = handle(cluster.id, unknown);
            after(
                notes,
                &session.id,
                AnswerNote::GroupVetoOverridden {
                    name,
                    answered: &answered,
                    speech_seconds: cluster.speech_seconds,
                    overlapped: &overlapped,
                },
            )?;
            *vetoes_overridden += 1;
        }
    }

    // Everything this answer wrote, as one note rather than as the four to six lines it used to
    // print, because that is the block an interface lays out together.
    //
    // Narrated *before* the copies are taken out of the consequence below: nothing between here
    // and there writes a byte, so the order the user sees is unchanged, and a partially moved
    // `Consequence` can no longer be borrowed.
    after(
        notes,
        &session.id,
        AnswerNote::Committed {
            name,
            speech_seconds: cluster.speech_seconds,
            consequence: &consequence,
        },
    )?;
    // A sub-count of `named`, and now read off the type rather than off the two arms of the
    // match that used to print those two sentences -- which is what `Consequence::session_only`
    // is documented to be.
    if consequence.session_only() {
        report.session_only += 1;
    }
    report.named += 1;
    // A sub-count of `named`, like `session_only`: the naming came from the assertion rather
    // than from an answer given per voice -- see `EnrollReport::asserted` for why the summary
    // needs the split.
    if under_assertion {
        report.asserted += 1;
        *asserted_voices += 1;
    }

    // Committed by taking the copies the dry run produced, so what lands on disk is the state
    // that was checked rather than a second construction of it.
    let speakers_changed = *speakers != consequence.speakers;
    let assignments_changed = assigned.names != consequence.assigned.names;
    *speakers = consequence.speakers;
    *assigned = consequence.assigned;

    // Written in a fixed order -- the database, then this session's names, then the transcript
    // -- and only where something changed, so a skipped write leaves a file byte-identical
    // rather than merely equivalent.
    if speakers_changed {
        speakers.write(paths)?;
    }
    if assignments_changed {
        assigned.write(&session.paths)?;
    }
    // Recorded after the writes, so the override report measures against voices whose names are
    // actually on disk, never against answers that were refused or skipped.
    committed.push(cluster);

    // Re-identified against the updated database rather than assumed: naming one voice can also
    // name a second cluster in this session, if clustering split that person in two, and a
    // `--force` re-transcribe would name both. Read with the forced tier the dry run used: a
    // group member heard at once with another holder of the name would otherwise be dropped back
    // to its number by the heard-at-once exclusion -- which is exactly the evidence the group's
    // authority overrides -- and the transcript would not say what the user decided. On the
    // ordinary path `forced` is `None`, so this reads exactly as before.
    let now = effective_labels(
        clusters,
        unknown,
        speakers,
        &assigned.names,
        assertion,
        forced,
        None,
    );
    if relabel(transcript, &now) {
        transcript.write(
            &session.paths,
            rules.template,
            &TranscriptContext::now(metadata),
        )?;
    }
    // Only on the timestamp path. A user who pointed at a moment did not choose the voice, so
    // how far the rename reached is the one thing they cannot infer -- whereas the queue and
    // `--voice` both showed them the voice first, and several tests pin their output exactly as
    // it is. `assertion.is_none()` because the assertion stands in for the selector: a library
    // caller that passed both gets the assertion's walk, and the timestamp's rename line is for
    // the one voice a moment pointed at, which this is not. A group is its own unit, so its
    // members get no per-member rename line either.
    if assertion.is_none() && forced.is_none() && matches!(rules.selector, Some(Selection::At(_))) {
        report_rename(transcript, shown, &now, name, session, notes)?;
    }
    *shown = now;
    Ok(CommitOutcome::Committed)
}

/// Resolves a group's "Unknown N" handles to the clusters they name, deduplicated and sorted
/// into queue order -- or `None` when the group cannot say who its members are.
///
/// `None` is the whole question going unanswered rather than a partial group: a blank name, an
/// empty mark, or a handle nothing in this session resolves. That is the blank-name precedent
/// reached with a different input, and deciding it here -- before the answerer is consulted --
/// is what keeps a group from committing some of its members and silently dropping the rest.
///
/// Handles are the values of `unknown`, built over every cluster in the session, so a quiet
/// member below the offer floor resolves alike. First-appearance order keeps the deduplication
/// deterministic; the queue sort re-orders by first appearance regardless, which is the order
/// the commit walks in and the one a preview folds over, so the two cannot disagree about
/// sequence.
fn resolve_group_members<'a>(
    clusters: &'a [SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    name: &str,
    members: &[String],
) -> Option<Vec<&'a SpeakerCluster>> {
    let name = name.trim();
    if name.is_empty() || members.is_empty() {
        return None;
    }
    let mut ids: Vec<u32> = Vec::new();
    for handle in members {
        match unknown
            .iter()
            .find(|(_, label)| label.as_str() == handle.as_str())
            .map(|(&id, _)| id)
        {
            Some(id) if !ids.contains(&id) => ids.push(id),
            Some(_) => {}
            None => return None,
        }
    }
    let mut resolved: Vec<&SpeakerCluster> = ids
        .iter()
        .filter_map(|id| clusters.iter().find(|c| c.id == *id))
        .collect();
    resolved.sort_by(|a, b| {
        a.first_spoke_seconds
            .total_cmp(&b.first_spoke_seconds)
            .then(a.id.cmp(&b.id))
    });
    Some(resolved)
}

/// Counts voices that were offered and not answered into the buckets they have turned out to
/// belong in, and says how many that was.
///
/// Leaving an already-named voice alone is keeping that identification, which is an answer;
/// leaving an unnamed one alone is the question going unanswered. Same write -- none -- and
/// different enough that the summary must not conflate them. One function rather than the rule
/// written out at each of the four places that needs it -- a skip, a name of nothing but spaces,
/// [`Answer::Leave`]'s tail, and the pass loop's fixed point -- so none of them can disagree
/// with the others about which bucket a voice belongs in.
///
/// `shown` is what each voice reads now and `baseline` what it read when the queue was built. A
/// voice that is named and has *moved* since the baseline was taken was named by an answer given
/// earlier in this run -- clustering split one person in two, so naming one half named the other
/// -- and has already been counted under `named`. It is counted here as nothing at all, because
/// reporting it as an identification this run left alone would put one voice in two buckets.
///
/// That guard is load-bearing only for [`Answer::Leave`], whose tail can hold a voice this same
/// pass has just named. Everywhere else it cannot fire -- the pass loop's own guard takes such a
/// voice out before it can be asked about or deferred -- which is what keeps every existing
/// count byte-identical.
fn left_unanswered<'c>(
    voices: impl IntoIterator<Item = &'c SpeakerCluster>,
    shown: &BTreeMap<u32, Attribution>,
    baseline: &BTreeMap<u32, Attribution>,
    report: &mut EnrollReport,
) -> usize {
    let mut counted = 0;
    for cluster in voices {
        let named = shown[&cluster.id].is_named();
        if named && shown[&cluster.id] != baseline[&cluster.id] {
            continue;
        }
        counted += 1;
        if named {
            report.kept += 1;
        } else {
            report.skipped += 1;
        }
    }
    counted
}

/// A session file that would not read, reported against the remedy its kind has.
fn unreadable(
    notes: &mut dyn Narrator,
    session: &DiscoveredSession,
    file: SessionFile,
    error: &meethook_session::Error,
) -> Result<()> {
    about(notes, &session.id, SessionNote::Unreadable { file, error })
}

/// Says how much of the transcript naming one voice just rewrote.
///
/// Only the timestamp path prints this, and the reason is what it is measured from: the
/// difference between what every voice read *before* the answer and what it reads *after*, not
/// the voice that was selected. Naming one cluster can name a second when clustering split that
/// person in two, so the selection is not the blast radius -- the label diff is.
///
/// The turns are counted and their durations summed rather than the clusters'
/// `speech_seconds` taken: the claim is about the lines this command rewrote in the file the
/// user is reading, and those are two different quantities.
fn report_rename(
    transcript: &Transcript,
    before: &BTreeMap<u32, Attribution>,
    after: &BTreeMap<u32, Attribution>,
    name: &str,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
) -> Result<()> {
    let mut renamed: Vec<u32> = Vec::new();
    for (id, label) in after {
        if before.get(id) != Some(label) {
            renamed.push(*id);
        }
    }
    let (turns, seconds) = transcript
        .turns
        .iter()
        .filter(|turn| {
            turn.source_track == SourceTrack::Speaker
                && turn.cluster.is_some_and(|id| renamed.contains(&id))
        })
        .fold((0usize, 0.0f64), |(count, total), turn| {
            (count + 1, total + (turn.end - turn.start))
        });

    // Spelled out because `after` is also the name of this function's label map.
    narration::after(
        notes,
        &session.id,
        AnswerNote::Renamed {
            name,
            turns,
            seconds,
        },
    )
}

/// What each voice is called given the database and this session's hand-given names as they
/// stand: a name the user assigned, else an enrolled name where one matched, else the
/// "Unknown N" its first appearance earned it.
///
/// This is the labelling `merge` performs when it writes a transcript, reached through the
/// same [`attributions`], which is what makes a rewrite here and a `--force` re-transcribe
/// agree on the answer rather than merely be written to. The precedence between the three is
/// stated there and nowhere else; above all of it sits the one-remote-speaker assertion,
/// passed in as `one_remote_speaker`, which when present names every voice and makes the rest
/// of the rule moot -- see [`Naming::with_one_remote_speaker`] for why. Between the assertion
/// and the hand-given names sits the forced-label tier, passed in as `forced`: the voices a
/// group commit has declared one person, exempt from the exclusions for the duration of the
/// walk -- see [`Naming::with_forced`] for the rank and the reason.
///
/// `clusters` is what identification runs over and what `assigned` is resolved against;
/// `unknown` is what the transcript was written with, and is the key set of the result. Those
/// two are built from the same file, so every voice gets an entry.
///
/// Visible to the crate rather than to this file because [`references`] labels sessions through
/// exactly this too: the claim that a reference is naming some voice is only as good as its
/// being the same labelling the transcript is written with.
pub(crate) fn effective_labels(
    clusters: &[SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    speakers: &EnrolledSpeakers,
    assigned: &[AssignedName],
    one_remote_speaker: Option<&str>,
    forced: Option<&BTreeMap<u32, String>>,
    pending: Option<u32>,
) -> BTreeMap<u32, Attribution> {
    // `pending` is the voice an in-progress answer is about: its hand-given row is what the
    // answer would create rather than a standing declaration, so it earns no co-declaration
    // pass in the assignment award below -- which is what leaves the vetoed demotion for the
    // dry run's refusal to fire off. Every other reading passes `None`, where every standing
    // row stands.
    let identified = identify_clusters(clusters, speakers);
    let mut naming =
        Naming::new(clusters, &identified, assigned).with_one_remote_speaker(one_remote_speaker);
    if let Some(forced) = forced {
        naming = naming.with_forced(forced);
    }
    if let Some(pending) = pending {
        naming = naming.with_pending(pending);
    }
    attributions(unknown, naming)
}

/// Rewrites every speaker-track turn to what `labels` says its voice should now be called,
/// reporting whether anything changed.
///
/// Turns are found by the cluster they were attributed to, which `transcript.json` records for
/// exactly this: it is an exact handle on one voice's turns, so what a turn currently *reads*
/// never enters into it. That matters most in the case a label lookup cannot survive -- two
/// voices both matched to one enrolled person, then corrected so they belong to different
/// people. Keyed on text those turns are indistinguishable and the only safe answer is to
/// rewrite neither; keyed on the cluster there is no ambiguity to resolve, and correcting one
/// voice leaves the other's turns exactly where they were.
///
/// The cluster is never written back. `merge` is the sole producer of that field and `enroll`
/// only ever changes what a cluster is *called*, which is what keeps a transcript rewritten
/// here identical to what `transcribe --force` would now produce.
///
/// A turn with no cluster is left alone: on the mic track that is the local speaker, whose
/// name is not `enroll`'s to change, and on the speaker track it only arises in a session
/// where diarization found no clusters -- which has no labels to map and nothing to ask about.
/// A cluster absent from `labels` is left alone for the same reason `merge` ignores an
/// identification for a cluster diarization did not produce.
///
/// Nothing is written when nothing changed, which is what makes a skipped session leave its
/// files byte-identical rather than merely equivalent.
///
/// Visible to the crate rather than to this file because [`forget`] brings a transcript in line
/// through exactly this too: a removal that rewrote transcripts any other way would be a second
/// producer of the labels `merge` writes.
pub(crate) fn relabel(transcript: &mut Transcript, labels: &BTreeMap<u32, Attribution>) -> bool {
    let mut changed = false;
    for turn in &mut transcript.turns {
        if turn.source_track != SourceTrack::Speaker {
            continue;
        }
        let Some(label) = turn.cluster.and_then(|id| labels.get(&id)) else {
            continue;
        };
        if turn.speaker != label.label() || turn.speaker_id_confidence != label.confidence() {
            turn.speaker = label.label().to_string();
            turn.speaker_id_confidence = label.confidence();
            changed = true;
        }
    }
    changed
}
#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use meethook_session::{SessionId, SourceTrack, Transcript};
    use meethook_transcribe::Attribution;

    use super::*;
    use crate::tests::{mic_turn, said, speaker_turn};

    /// The case this whole handle exists for, and the one a label lookup cannot survive: a
    /// false accept has filed cluster 3's voice under the name of the person who is really
    /// cluster 1, so two clusters read "Andrew", and the correction sends them to
    /// different names. Keyed on the label text both turn-groups are one indistinguishable
    /// bucket and the only safe answer is to rewrite neither -- silently leaving the user
    /// looking at an uncorrected transcript. Keyed on the cluster the two are simply
    /// different turns.
    #[test]
    fn correcting_one_of_two_voices_sharing_a_label_leaves_the_other_alone() {
        let mut transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                speaker_turn(0.0, 1, "Andrew", "the real one"),
                mic_turn(1.0, "morning"),
                speaker_turn(2.0, 3, "Andrew", "actually Ryan"),
                speaker_turn(3.0, 1, "Andrew", "the real one again"),
            ],
        );
        for turn in &mut transcript.turns {
            if turn.source_track == SourceTrack::Speaker {
                turn.speaker_id_confidence = Some(0.71);
            }
        }

        // The database after the correction: cluster 3 is Ryan, cluster 1 is still Andrew.
        let labels: BTreeMap<u32, Attribution> = [
            (
                1,
                Attribution::Identified {
                    name: "Andrew".to_string(),
                    similarity: 0.71,
                },
            ),
            (
                3,
                Attribution::Identified {
                    name: "Ryan".to_string(),
                    similarity: 0.88,
                },
            ),
        ]
        .into_iter()
        .collect();

        assert!(
            relabel(&mut transcript, &labels),
            "the correction must be reported as a change, not silently declined"
        );
        assert_eq!(
            said(&transcript),
            [
                ("Andrew", "the real one", Some(0.71)),
                ("You", "morning", None),
                ("Ryan", "actually Ryan", Some(0.88)),
                ("Andrew", "the real one again", Some(0.71)),
            ]
        );
    }
}
