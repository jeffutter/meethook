//! Building the prompt payload and the audio under it.
//!
//! [`Voice`] is one question assembled for the seam: the labels, the queue rows, the snippets
//! with their samples, the representative clip, and the database ranking. [`Snippet`] is one
//! line with the audio under it, and [`write_clip`] is the only place a clip leaves memory.

use std::path::Path;

use meethook_session::{SessionId, SourceTrack, SpeakerCluster, Transcript};
use meethook_transcribe::{Attribution, Resemblance, TARGET_RATE};

use crate::consequence::Preview;
use crate::groups::FragmentGroup;
use crate::interview::MeetingLabel;
use crate::queue::{Position, Queued};
use crate::{Error, Result};

#[cfg(doc)]
use crate::interview::{GivenName, Interviewer};
#[cfg(doc)]
use crate::queue::{Offer, VoiceSelector};
#[cfg(doc)]
use crate::resolve;
#[cfg(doc)]
use meethook_session::unknown_labels;
#[cfg(doc)]
use meethook_transcribe::rank_enrolled;

/// How much of one line to show. Long enough for a sentence, short enough to stay on a line.
const SNIPPET_CHARS: usize = 100;

/// One line a voice said, and the audio under it.
///
/// Four fields and no methods, for the reason [`Queued`]'s doc gives: it is a row, and what a
/// row reads like belongs to whatever is drawing it. Four things a reader cannot get from the
/// type, each of them a place this goes wrong quietly rather than loudly:
///
/// - **`start` is track time, not timeline time.** A [`meethook_session::Turn`]'s seconds are
///   on the session timeline, which begins at whichever track started first;
///   `start` here is an offset into `speaker.wav`, the same space a
///   [`meethook_session::RepresentativeSegment`] is in and the same space [`Voice::clip`] was
///   cut from. The difference between the two is
///   [`meethook_transcribe::speaker_offset_seconds`]. Nothing fails if it is not applied --
///   the words and the sound simply drift apart by however long the microphone led by, and
///   only a listener would notice.
///
/// - **`duration` is what the transcript says, `audio.len()` is what there is to play, and
///   they can disagree.** A truncated `speaker.wav` gives a full `duration` and short `audio`;
///   a missing one gives a full `duration` and empty `audio`. That split is deliberate: one
///   says how long the line took, the other says how much of it survives on disk, so anything
///   sizing a progress bar wants `audio.len() / TARGET_RATE` and not `duration`.
///
/// - **Empty `audio` is normal**, exactly as an empty [`Voice::clip`] is: a voice that can
///   still be named from its lines rather than a session that has to fail.
///
/// - **`text` is the same text a prompt always had** -- whitespace-trimmed, cut to
///   `SNIPPET_CHARS` characters, and never empty, because a line the recogniser heard nothing
///   over is dropped before a snippet is built at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snippet<'a> {
    /// What was said, trimmed and cut. A borrow of the transcript, so a voice that talked for
    /// ten minutes costs pointers rather than text.
    pub text: &'a str,

    /// When it was said, in seconds from the first sample of `speaker.wav`.
    pub start: f64,

    /// How long the line lasted, in seconds, as the transcript has it.
    pub duration: f64,

    /// The samples for exactly that stretch: 16 kHz mono, the rate everything else in meethook
    /// works in. A borrow of the resampled track, not a copy.
    pub audio: &'a [f32],
}

/// One voice being asked about, and everything needed to ask.
///
/// Usually a voice nothing in the database matched, which is what `enroll` exists for. Under
/// [`Offer::named`] it can also be one the database has already put a name to, and then the
/// question is a different one -- not "who is this" but "is this right" -- which is what
/// `confidence` tells the caller.
///
/// Deliberately one value rather than a play-then-ask pair of calls: the order those two
/// would have to be made in is exactly the sort of thing a seam should not be leaking.
pub struct Voice<'a> {
    pub session: &'a SessionId,

    /// The meeting this session was recorded during, as far as a terminal may see it -- or
    /// `None`, which is the common case rather than the exception.
    ///
    /// The same value the queue announcement was handed, built from the same
    /// `session.json` load: an interface that shows it does so across this seam rather than
    /// reaching back for the file itself, and nothing off the roster crosses with it.
    /// Absent costs nothing downstream -- no reserved row, no empty label -- so a run over
    /// sessions without meetings behaves exactly as before.
    pub meeting: Option<&'a MeetingLabel>,

    /// Which of this session's questions this is, and how many there are.
    ///
    /// Carried across the seam rather than counted on the far side, because an [`Interviewer`]
    /// counting its own calls would be counting a different thing -- the questions asked, not
    /// the queue -- and could not know the total at all. Restarts at 1 for each session of a
    /// run; `session` above says which one.
    pub position: Position,

    /// What this voice is currently called, and on what basis.
    ///
    /// One field rather than a label plus a confidence, because the prompt turns on *which
    /// kind* of label this is and only one of the three carries a number. Reading "has a
    /// confidence" as "already has an answer" is right for exactly as long as an identification
    /// is the only way to get a name onto a voice; [`Attribution::Assigned`] is a name with no
    /// similarity behind it, and a prompt written against the number would ask "who is this?"
    /// about a voice this command named an hour ago.
    ///
    /// [`Attribution::label`] is exactly as the transcript reads -- "Unknown 2", or the name --
    /// so the user can find the voice in the file in front of them.
    pub attribution: &'a Attribution,

    /// The "Unknown N" this voice was transcribed with -- a handle that does not move.
    ///
    /// [`attribution`](Voice::attribution)'s label is what the voice reads as *now*, so it
    /// changes the moment the voice is named. This does not: it comes from [`unknown_labels`],
    /// which ranks every voice by first appearance whether or not it has a name, and is fixed
    /// for the session. An [`Interviewer`] with state of its own -- a cursor, a row it has
    /// marked, a name half-typed -- has to key that state on something stable across
    /// [`identify`](Interviewer::identify) calls, and this is the only field that qualifies.
    ///
    /// Not the cluster id, for the reason [`VoiceSelector`] gives: the id appears in
    /// `transcript.json` and nowhere a person reads, and two numbering systems reachable from
    /// one interface is how a cursor lands on the wrong voice.
    pub number: &'a str,

    /// Total speech attributed to this voice, in seconds. How the user tells a participant
    /// from someone who coughed once.
    pub speech_seconds: f64,

    /// Every voice in this session, in first-appearance order -- the order the transcript
    /// reads in -- whether or not this run is asking about it.
    ///
    /// What lets an interface draw the whole session beside the one question, which is what
    /// makes the quiet voices and the already-named ones visible rather than merely reachable.
    /// It includes the voice being asked about, so a queue pane needs nothing stitched onto it.
    ///
    /// Rebuilt for every call rather than handed over once per session, which is what makes it
    /// current: an answer accepted a moment ago has already been written and the session
    /// relabelled through it, so a voice named by the previous question arrives here under its
    /// new name.
    pub queue: &'a [Queued<'a>],

    /// What this voice said, whitespace-trimmed and cut to `SNIPPET_CHARS` characters each,
    /// with the lines the recogniser heard nothing over dropped -- and the audio under each.
    /// Empty if it heard nothing at all.
    ///
    /// Every snippet, uncapped. How many will fit is a fact about the thing displaying them --
    /// a line prompt has one screenful of scrollback and takes three; a pane can scroll -- so
    /// capping here would decide it for both. Both the text and the samples are borrows, of
    /// the transcript and of the resampled track, so a voice that talked for ten minutes costs
    /// pointers rather than text or audio.
    ///
    /// See [`Snippet`] for what "when it was said" means here, which is not the same clock a
    /// transcript reads in.
    pub snippets: Vec<Snippet<'a>>,

    /// The longest representative clip: 16 kHz mono, the same rate everything else in
    /// meethook works in.
    ///
    /// Empty when `speaker.wav` is missing or unreadable, which is a voice that can still be
    /// named from its snippets rather than a session that has to fail.
    pub clip: &'a [f32],

    /// Who this voice sounds like, nearest first, so a prompt can offer names instead of
    /// demanding one be typed.
    ///
    /// This is what makes an [`Interviewer`] able to answer without any access to
    /// `speakers.json`: the names and the numbers are already here, owned, and the database
    /// stays on this side of the seam where the writes are.
    ///
    /// Four things a reader cannot get from the type, all of them [`rank_enrolled`]'s
    /// decisions rather than this field's:
    ///
    /// - **Unthresholded.** Every comparable enrolled person is here, including ones far
    ///   outside `IDENTIFY_DISTANCE`. That cut keeps `transcribe`'s automatic pass
    ///   conservative; a person reading a list is the case it was biased against serving, and
    ///   cutting here would hide the near-miss they are being asked to adjudicate.
    /// - **Entries are people, not references.** Somebody holding several recordings appears
    ///   once, scored at their nearest, with `references` saying how many they hold.
    /// - **Order** is descending similarity, ties by ascending name, so the first entry is the
    ///   person `identify_clusters` would have awarded had it cleared the cut. Empty for an
    ///   install where nobody is enrolled yet.
    /// - **As the database stands now**, not as it stood when the run began. A name given
    ///   earlier in this same run is in here, which is the useful behaviour rather than an
    ///   accident: clustering splits one person in two, and the second half should offer the
    ///   name the first half was just given.
    pub resembles: Vec<Resemblance>,

    /// Every enrolled name, deduplicated, in enrolment order -- the universe [`resolve()`]
    /// requires.
    ///
    /// Not [`resembles`](Voice::resembles) with the numbers stripped off, and the difference is
    /// the whole reason this field exists: ranking a voice against the database drops a person
    /// whose every stored recording is a stale embedding dimension, and that person is still
    /// real, so resolving a typed name against the ranked list would silently enrol a second
    /// copy of them. `resembles` answers "who does this sound like"; this answers "who is
    /// there", which is the question a name being typed is about.
    ///
    /// As the database stands now, like `resembles`: a name given earlier in this same run is
    /// in here.
    pub enrolled: Vec<&'a str>,

    /// What answering with a given name *would* do, asked without writing anything.
    ///
    /// [`resembles`](Voice::resembles) says who this voice sounds like; this says what happens
    /// if you say so. An [`Interviewer`] holding it can show the consequence of the name under
    /// the cursor -- that it enrols somebody new, that it takes a reference off Milo, that the
    /// veto will refuse it -- before the user commits to it, which is the whole difference
    /// between a prompt that reports what it did and one that can be answered.
    ///
    /// The three things the type cannot say:
    ///
    /// - **Asking writes nothing.** Not `speakers.json`, not this session's
    ///   `speaker_names.json`, not the transcript. It runs on copies, which is why an
    ///   `Interviewer` may ask about as many candidates as it likes.
    /// - **One call is one answer's worth of work**, not one keystroke's: a database clone and
    ///   two full labellings of the session. Preview the candidate the user has settled on;
    ///   [`resolve()`] is the cheap thing to run per keystroke.
    /// - **An answerer that never asks pays nothing.** [`GivenName`] does not, and neither does
    ///   a line prompt that only reports outcomes after the fact.
    pub preview: Preview<'a>,

    /// The bundles of below-floor fragments this run asks about together, if it groups them at
    /// all.
    ///
    /// Empty for every run but one that asked for them -- a headless run answers per voice and
    /// prints what it always printed -- and otherwise the full picture as the queue was built:
    /// every multi-member bundle, not just the one the current question is about, because the
    /// pane shows the whole queue rather than one row. Built once when the queue is built and
    /// carried unmodified into every later question; [`bundle_members`](Self::bundle_members)
    /// says which of these the *current* question is about, filtered to the members still open.
    pub fragment_groups: Vec<FragmentGroup>,

    /// The stable "Unknown N" handles the current question covers, in queue order -- or
    /// `None` when the question is about one voice only, which is every question in a run that
    /// does not group fragments.
    ///
    /// Carried across the seam rather than re-derived on the far side, because the interface
    /// cannot see the clusters the bundle was built from: its rows carry attributions that move
    /// as the run names voices, and membership worked out from them would drift from the
    /// members the commit actually walks. This is the live set -- members already settled are
    /// out -- so answering it commits exactly the walk the library will run over it.
    pub bundle_members: Option<Vec<String>>,
}

/// How much somebody said, in the units a person would say it in.
///
/// Public because the prompt in the CLI and the rename line above both print a duration, and one
/// tool printing a duration two ways is a defect rather than a style.
pub fn speech(seconds: f64) -> String {
    let seconds = seconds.round() as u64;
    match seconds / 60 {
        0 => format!("{seconds}s"),
        minutes => format!("{minutes}m {:02}s", seconds % 60),
    }
}

/// One line of transcript, trimmed and cut to something that fits a prompt.
pub(crate) fn snippet(text: &str) -> &str {
    let trimmed = text.trim();
    match trimmed.char_indices().nth(SNIPPET_CHARS) {
        Some((cut, _)) => &trimmed[..cut],
        None => trimmed,
    }
}

/// The samples between two offsets into the 16 kHz speaker track.
///
/// The one place the clamping rule lives, because both things that cut audio out of a track --
/// a voice's clip and a line's snippet -- have to obey the same one: a range running off the
/// end of the track yields what is there, and anything left empty is audio that is missing
/// rather than a session that fails.
pub(crate) fn samples_between(track: &[f32], start: f64, end: f64) -> &[f32] {
    let start = sample_at(start).min(track.len());
    let end = sample_at(end).min(track.len());
    if end <= start {
        return &[];
    }
    &track[start..end]
}

/// The audio to play for one voice: its longest representative, cut out of the speaker track.
///
/// The clip is sliced rather than seeked to because the plainest players in the search list
/// (`afplay`, `paplay`, `aplay`) take no start offset at all -- so somebody has to extract it
/// either way. Slicing the 16 kHz track
/// diarization itself ran on is what makes the seconds in a [`meethook_session::RepresentativeSegment`]
/// impossible to misinterpret: they are offsets into exactly this buffer.
pub(crate) fn clip_for<'a>(track: &'a [f32], cluster: &SpeakerCluster) -> &'a [f32] {
    let Some(segment) = cluster.representatives.first() else {
        return &[];
    };
    samples_between(track, segment.start, segment.end)
}

/// Every line one voice said, as a prompt needs them: the text, when it was said, and the
/// samples for it.
///
/// `offset` is [`meethook_transcribe::speaker_offset_seconds`] -- what turns a
/// session-timeline second into an offset into `track`. Same filter, same trim, same order as
/// the text-only list this replaced, so the lines a prompt shows do not move.
pub(crate) fn snippets_for<'a>(
    transcript: &'a Transcript,
    cluster: u32,
    track: &'a [f32],
    offset: f64,
) -> Vec<Snippet<'a>> {
    transcript
        .turns
        .iter()
        .filter(|turn| turn.source_track == SourceTrack::Speaker && turn.cluster == Some(cluster))
        .filter_map(|turn| {
            let text = snippet(&turn.text);
            if text.is_empty() {
                return None;
            }
            // Both clamps are the habit `sample_at` and `samples_between` already have of
            // defining the edge away rather than checking for it. A speaker-track turn cannot
            // begin before the speaker track by construction, so neither is a real case.
            let start = (turn.start - offset).max(0.0);
            let end = (turn.end - offset).max(start);
            Some(Snippet {
                text,
                start,
                duration: end - start,
                audio: samples_between(track, start, end),
            })
        })
        .collect()
}

pub(crate) fn sample_at(seconds: f64) -> usize {
    (seconds.max(0.0) * f64::from(TARGET_RATE)).round() as usize
}

/// Writes a clip where an external player can reach it: mono, 16 kHz, 32-bit float.
///
/// Here rather than in the caller because the format is this crate's knowledge -- the clip in
/// a [`Voice`] is 16 kHz mono because that is the track it was cut from -- and a player that
/// had to be told the rate could be told the wrong one.
pub fn write_clip(path: &Path, clip: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let wav = |source| Error::Wav {
        path: path.to_path_buf(),
        source,
    };

    // Not `hound::WavWriter::create`: it tags a mono stream `SPEAKER_FRONT_LEFT`, and a clip
    // that exists so a human can recognise a voice is the last place to send it to one ear.
    let mut writer = meethook_session::wav::create(path, spec).map_err(wav)?;
    for sample in clip {
        writer.write_sample(*sample).map_err(wav)?;
    }
    writer.finalize().map_err(wav)
}
#[cfg(test)]
mod tests {
    use meethook_session::{SessionId, Transcript};

    use super::*;
    use crate::tests::{mic_turn, speaker_turn};

    /// Six seconds of a 16 kHz track whose every sample says which sample it is, so a slice
    /// out of it can be checked for *where* it came from and not merely how long it is.
    fn counted_track() -> Vec<f32> {
        (0..16_000 * 6).map(|i| i as f32).collect()
    }

    /// A transcript with one speaker turn per `(start, text)`, all from cluster 0.
    fn transcript_of_lines(lines: &[(f64, &str)]) -> Transcript {
        Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            lines
                .iter()
                .map(|(start, text)| speaker_turn(*start, 0, "Unknown 1", text))
                .collect(),
        )
    }

    /// Acceptance criterion #2: a snippet's seconds are offsets into `speaker.wav`, not into the
    /// session timeline, and its samples come from where those seconds point.
    ///
    /// The first-sample assertion is the point of the counted track: a length-only check passes
    /// just as happily when the offset is applied with the wrong sign.
    #[test]
    fn snippet_times_are_track_time_and_the_audio_starts_there() {
        let track = counted_track();
        let transcript = transcript_of_lines(&[(3.0, "and from me")]);

        let snippets = snippets_for(&transcript, 0, &track, 1.0);

        assert_eq!(snippets.len(), 1);
        assert_eq!((snippets[0].start, snippets[0].duration), (2.0, 1.0));
        assert_eq!(snippets[0].audio.len(), 16_000);
        assert_eq!(
            snippets[0].audio[0], 32_000.0,
            "two seconds into the track, which is the turn's three seconds less the offset"
        );
    }

    /// Acceptance criterion #3: a line running off the end of a truncated track keeps the
    /// duration the transcript gave it and plays what survives -- nothing at all when the line
    /// is entirely past the end.
    #[test]
    fn a_snippet_past_the_end_of_the_track_carries_what_is_there() {
        let track = counted_track();
        let transcript = transcript_of_lines(&[(5.5, "trailing off"), (30.0, "long gone")]);

        let snippets = snippets_for(&transcript, 0, &track, 0.0);

        assert_eq!((snippets[0].start, snippets[0].duration), (5.5, 1.0));
        assert_eq!(
            snippets[0].audio.len(),
            8_000,
            "half the line is on disk; the duration still says how long it took"
        );
        assert_eq!((snippets[1].start, snippets[1].duration), (30.0, 1.0));
        assert!(snippets[1].audio.is_empty());
    }

    /// Acceptance criterion #3, the other half: no track is no audio and not a lost line. The
    /// times are still there, which is what leaves a later run able to say what could not be
    /// played.
    #[test]
    fn an_empty_track_leaves_every_snippet_with_its_times_and_no_audio() {
        let transcript = transcript_of_lines(&[(0.0, "hi there"), (4.0, "let us start")]);

        let snippets = snippets_for(&transcript, 0, &[], 0.0);

        assert_eq!(
            snippets
                .iter()
                .map(|s| (s.start, s.duration))
                .collect::<Vec<_>>(),
            [(0.0, 1.0), (4.0, 1.0)]
        );
        assert!(snippets.iter().all(|s| s.audio.is_empty()));
    }

    /// Acceptance criterion #4: the same lines, the same text and the same order as the
    /// text-only list this replaced -- this voice's turns only, trimmed, cut, and with the ones
    /// the recogniser heard nothing over dropped rather than carried as empty rows.
    #[test]
    fn the_snippets_are_the_same_lines_in_the_same_order_as_before() {
        let long = "x".repeat(SNIPPET_CHARS + 20);
        let transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "  hi there  "),
                mic_turn(1.0, "morning"),
                speaker_turn(2.0, 1, "Unknown 2", "and from me"),
                speaker_turn(3.0, 0, "Unknown 1", "   "),
                speaker_turn(4.0, 0, "Unknown 1", &long),
            ],
        );

        let snippets = snippets_for(&transcript, 0, &[], 0.0);

        assert_eq!(
            snippets.iter().map(|s| s.text).collect::<Vec<_>>(),
            ["hi there", &"x".repeat(SNIPPET_CHARS)],
            "the mic turn, the other voice and the line with nothing in it are all out"
        );
    }

    /// The clamping rule both a clip and a snippet obey, stated once where it lives.
    #[test]
    fn samples_between_clamps_to_what_the_track_holds() {
        let track = counted_track();

        assert_eq!(samples_between(&track, 1.0, 2.0).len(), 16_000);
        assert_eq!(samples_between(&track, 1.0, 2.0)[0], 16_000.0);
        assert_eq!(samples_between(&track, 5.0, 90.0).len(), 16_000);
        assert!(samples_between(&track, 600.0, 620.0).is_empty());
        assert!(samples_between(&track, 2.0, 2.0).is_empty());
        assert!(samples_between(&[], 0.0, 1.0).is_empty());
    }

    /// A long line is cut to something that fits a prompt, on a character boundary rather
    /// than a byte one.
    #[test]
    fn a_long_snippet_is_cut_to_a_readable_length() {
        let long = "é".repeat(SNIPPET_CHARS * 2);
        assert_eq!(snippet(&long).chars().count(), SNIPPET_CHARS);
        assert_eq!(snippet("  short  "), "short");
    }

    /// A clip exists to be handed to an audio player, so its header is part of what it is for: a
    /// mono stream tagged `SPEAKER_FRONT_LEFT` reaches the listener in one ear.
    #[test]
    fn a_clip_is_tagged_mono_so_a_player_does_not_put_it_in_one_ear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.wav");
        write_clip(&path, &[0.0, 0.25, -0.25, 0.5]).unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(
            meethook_session::wav::channel_mask_of(&wav),
            Some(meethook_session::wav::MONO_CHANNEL_MASK)
        );
    }
}
