use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, write_atomic};
// Re-exported from the render layer so the crate root keeps naming them through this
// module; they are defined where they are used, in `transcript_render`.
pub use super::transcript_render::{TranscriptContext, TranscriptTemplate};

/// Deserializes an `Option` field that must nonetheless be *present*.
///
/// This looks like a no-op and is not one. serde treats a missing `Option<T>` field as `None`
/// rather than as an error, which for [`Turn::cluster`] would quietly turn "written by a tool
/// that did not record provenance" into the positive claim "came from no cluster" -- exactly
/// the defaulting [`TRANSCRIPT_SCHEMA_VERSION`] refuses. Naming a `deserialize_with` is what
/// makes serde emit a real missing-field error instead, because it can no longer route the
/// absent field through its `Option`-aware fallback.
///
/// Do not remove this in the belief that it is a redundant wrapper: deleting it silently
/// restores the default and no round-trip test can see the difference.
fn present_option<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::deserialize(deserializer)
}

/// Bumped whenever `transcript.json`'s shape changes incompatibly.
///
/// Deliberately separate from [`crate::SESSION_SCHEMA_VERSION`]: `session.json` and
/// `transcript.json` are written by different commands at different times and evolve
/// independently, so one shared number would force a lie in one of the two files.
///
/// Version 2 added [`Turn::cluster`], and added it as a required field: a version 1 file now
/// fails to parse rather than being read with every turn's cluster defaulted to `null`.
/// Defaulting would be worse than refusing, and specifically so. `null` on this field is a
/// positive assertion -- "this turn's voice came from no cluster" -- which is true only for
/// the mic track and for a session diarization found nobody in. Fabricating it across a whole
/// speaker track would leave `enroll` with no handle on those turns at all, so it would have
/// to go back to guessing which voice a turn belongs to from the label text it currently
/// reads; that guess is exactly what recording the cluster exists to delete, and its failure
/// mode is one person's name written onto another person's words. Refusing also keeps `null`
/// meaning "not applicable" and never "written by an older tool", which is the distinction
/// [`Turn::speaker_id_confidence`] already pays a present-null to preserve. Re-transcribing
/// the session with `--force` rewrites the file correctly.
///
/// The refusal comes from serde's missing-field error rather than from a version comparison,
/// which is how [`crate::SPEAKER_CLUSTERS_SCHEMA_VERSION`]'s two bumps work as well: one file
/// in the session contract checking its version while its neighbours do not would cost more
/// than the better message buys.
pub const TRANSCRIPT_SCHEMA_VERSION: u32 = 2;

/// The speaker label used for every turn recognised on the microphone track.
///
/// The mic track is, by construction, the machine's own user; nothing about the recording
/// can make it anyone else.
pub const YOU: &str = "You";

/// The label for a voice on the speaker track that nobody has named yet.
///
/// `number` is the speaker's rank by *first appearance* in the session, starting at 1, so
/// "Unknown 1" means "the first unidentified person to speak". Numbering by first
/// appearance rather than by cluster id is what makes the label stable across a `--force`
/// re-transcribe and meaningful to a user who is reading the transcript top to bottom.
///
/// It lives here, beside [`YOU`], because `enroll` has to recognise the label it is about
/// to replace with a real name -- that makes the format part of the on-disk contract rather
/// than a detail of whatever produced the transcript.
///
/// Numbers are assigned over *all* speaker-track voices, named or not. When `enroll`
/// substitutes a name it leaves the number it replaced unused rather than renumbering the
/// rest, so naming one person never silently changes anybody else's label.
pub fn unknown_speaker(number: usize) -> String {
    format!("Unknown {number}")
}

/// Which captured track a turn came from.
///
/// Kept as an explicit field even though it is currently derivable from
/// `speaker == "You"`. That equivalence is an artifact of today's pipeline, not a property
/// of the format, and baking it into every downstream consumer of `transcript.json` would
/// make it expensive to stop being true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceTrack {
    Mic,
    Speaker,
}

/// One contiguous stretch of speech attributed to one speaker.
///
/// `start` and `end` are seconds from session start, where session start is the earliest
/// first sample across both tracks. They are *not* offsets into either WAV file: the two
/// tracks begin at different instants, and putting both on one timeline is precisely what
/// `session.json`'s stored host ticks exist for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub speaker: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub source_track: SourceTrack,
    /// The id of the `speaker_clusters.json` cluster this turn's voice was attributed to.
    ///
    /// This is the turn's provenance, not its name: it says which voice said the words, while
    /// `speaker` says what that voice is currently called. `enroll` needs the first to change
    /// the second exactly -- two voices can sit under one label, and without a cluster the
    /// only handle on a turn is the label text, which cannot tell them apart.
    ///
    /// `null` for mic-track turns, where the local speaker is known by construction and comes
    /// from no cluster, and for a speaker-track turn in a session where diarization found no
    /// clusters at all.
    ///
    /// Written by transcription's merge step and never rewritten afterwards: `enroll` changes
    /// only what a cluster is *called*, which is what keeps a rewritten transcript identical
    /// to what `transcribe --force` would produce. Ids are only meaningful within their own
    /// session -- cluster 3 in two meetings is two different people.
    ///
    /// No `skip_serializing_if`, for the same reason as `speaker_id_confidence` below, and a
    /// `deserialize_with` on the way in so that an absent key is refused rather than read as
    /// a null -- see `present_option`, which exists only for that.
    #[serde(deserialize_with = "present_option")]
    pub cluster: Option<u32>,
    /// How confident the claim that `speaker` names the right *person* is: the cosine
    /// similarity between this voice and that person's enrolled reference, in `[-1, 1]` and
    /// in practice near 1.
    ///
    /// `null` wherever no such claim is being made, which is every turn that is not a matched
    /// enrolled speaker: mic-track turns, where the speaker is known by construction rather
    /// than inferred; "Unknown N" turns, where the label names nobody; and turns whose speaker
    /// the user named by hand against this session (`speaker_names.json`), where the name came
    /// from a person listening to the clip rather than from a comparison with a reference. A
    /// number on any of them would be a number about a different question -- and note that the
    /// last case means a named turn may legitimately carry no confidence, so this field is not
    /// a way to tell whether `speaker` is a person's name.
    ///
    /// No `skip_serializing_if`: the key must be present and null rather than absent, so a
    /// consumer can tell "not applicable" from "written by an older tool".
    pub speaker_id_confidence: Option<f32>,
}

/// `transcript.json`: the canonical transcription result for one session.
///
/// Its presence in a session directory is what marks that session as already transcribed,
/// which is why [`Transcript::write`] writes it last.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub turns: Vec<Turn>,
}

impl Transcript {
    pub fn new(session_id: SessionId, turns: Vec<Turn>) -> Self {
        Transcript {
            schema_version: TRANSCRIPT_SCHEMA_VERSION,
            session_id,
            turns,
        }
    }

    /// Writes all three transcript files atomically.
    ///
    /// The readable renderings first, JSON last, and the order is load-bearing:
    /// `transcript.json`'s presence is the "already transcribed" marker, so a crash partway
    /// leaves a session that still re-transcribes rather than one that is marked done but
    /// missing a rendering.
    ///
    /// The markdown is rendered before any write, so a template that fails leaves all three
    /// files exactly as they were rather than truncating what it could not replace.
    pub fn write(
        &self,
        paths: &SessionPaths,
        template: &TranscriptTemplate,
        ctx: &TranscriptContext<'_>,
    ) -> Result<()> {
        let rendered = self.render_markdown(template, ctx)?;

        let md = paths.transcript_md();
        write_atomic(&md, rendered.as_bytes())?;

        let vtt = paths.transcript_vtt();
        write_atomic(&vtt, self.render_vtt().as_bytes())?;

        let json_path = paths.transcript_json();
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(&json_path, e))?;
        json.push(b'\n');
        write_atomic(&json_path, &json)
    }

    pub fn read(path: &Path) -> Result<Transcript> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
    }

    /// Which voice was speaking at `at`, for a timestamp a user read off `transcript.md`.
    ///
    /// This is the inverse of the label [`TranscriptTime`] prints, and it lives here rather
    /// than in the command that asks because the timeline is this crate's: a caller holding a
    /// cluster id can look up a name, but only the transcript knows which turn an instant
    /// belongs to.
    ///
    /// Two rules, both about matching what the user is looking at rather than what the
    /// numbers literally say:
    ///
    /// - **A printed label wins over containment.** `[12:34]` is printed by a turn starting at
    ///   754.3 s, and `12:34` parsed back is 754.0 -- an instant that turn does not contain.
    ///   Resolving by containment alone would refuse the exact line the user copied, so a turn
    ///   whose *printed* label equals `at` is preferred over one that merely covers it.
    /// - **A speaker-track turn wins over a mic-track one.** The two tracks are recorded
    ///   simultaneously and merged onto one timeline, so an instant can sit inside both. The
    ///   mic track is never a nameable voice, so preferring it would answer "you" to a
    ///   question that has a real answer.
    ///
    /// Ties beyond those are broken by the earlier `start`. The turns are scanned linearly and
    /// are not assumed to be sorted: `Transcript` is a deserialized file, and a meeting's worth
    /// of turns is a trivial scan.
    pub fn voice_at(&self, at: TranscriptTime) -> VoiceAt {
        let instant = at.seconds();

        if let Some(turn) = preferred(self.candidates_at(at).into_iter()) {
            return match (turn.source_track, turn.cluster) {
                (SourceTrack::Mic, _) => VoiceAt::LocalSpeaker,
                (SourceTrack::Speaker, Some(cluster)) => VoiceAt::Cluster(cluster),
                (SourceTrack::Speaker, None) => VoiceAt::NoCluster,
            };
        }

        // Checked after the two lookups above so a turn that both covers `at` and ends the
        // session still resolves to that turn. An empty transcript lands here with `last` 0.0,
        // which reads correctly: there is no session after which anything was said.
        let last = self.turns.iter().map(|turn| turn.end).fold(0.0, f64::max);
        if instant >= last {
            VoiceAt::PastEnd { last }
        } else {
            VoiceAt::Silence
        }
    }

    /// Every voice a timestamp could have meant: the distinct clusters among the turns
    /// [`voice_at`](Self::voice_at) chose between, in transcript order.
    ///
    /// `voice_at` answers with one of these. **More than one means the timestamp does not name a
    /// single voice** -- two turns a fraction of a second apart print the same `MM:SS`, and a
    /// caller about to rename somebody cannot pick between them on the user's behalf. Empty
    /// wherever `voice_at` did not resolve to a cluster at all.
    ///
    /// A separate method rather than a sixth [`VoiceAt`] arm because ambiguity is not a
    /// different *answer* -- it is a fact about the candidates behind the answer, which only a
    /// caller that has to act on the answer needs. Both read the same candidate set, so neither
    /// can drift from the other's idea of which turns a timestamp reaches.
    pub fn clusters_at(&self, at: TranscriptTime) -> Vec<u32> {
        let mut voices = Vec::new();
        for turn in self.candidates_at(at) {
            if turn.source_track != SourceTrack::Speaker {
                continue;
            }
            if let Some(cluster) = turn.cluster
                && !voices.contains(&cluster)
            {
                voices.push(cluster);
            }
        }
        voices
    }

    /// The turns a timestamp reaches, before any preference between them is applied: the ones
    /// whose *printed* label is `at`, or -- when no turn prints it -- the ones covering the
    /// instant.
    ///
    /// The one statement of what a timestamp reaches, so that the voice a lookup answers with
    /// and the voices a caller is told it chose between cannot disagree. Containment is
    /// half-open, so a turn's end instant belongs to whatever follows rather than to two turns
    /// at once.
    fn candidates_at(&self, at: TranscriptTime) -> Vec<&Turn> {
        let labelled: Vec<&Turn> = self
            .turns
            .iter()
            .filter(|turn| TranscriptTime::of(turn.start) == at)
            .collect();
        if !labelled.is_empty() {
            return labelled;
        }
        let instant = at.seconds();
        self.turns
            .iter()
            .filter(|turn| turn.start <= instant && instant < turn.end)
            .collect()
    }
}

/// Picks the turn a timestamp should resolve to out of the candidates that matched: the
/// speaker-track one before the mic-track one, then the earliest `start`.
fn preferred<'t>(turns: impl Iterator<Item = &'t Turn>) -> Option<&'t Turn> {
    // Speaker before mic. `min_by` keeps the first of equal elements, so transcript order is
    // the final tiebreak after `start`.
    let rank = |track| match track {
        SourceTrack::Speaker => 0u8,
        SourceTrack::Mic => 1,
    };
    turns.min_by(|a, b| {
        rank(a.source_track)
            .cmp(&rank(b.source_track))
            .then(a.start.total_cmp(&b.start))
    })
}

/// A whole second from session start, in the `MM:SS` spelling every `transcript.md` prints.
///
/// The one owner of that format: it both writes the label a transcript shows and reads a label
/// back off one, so the string a user copies out of a transcript and the string a command
/// accepts cannot drift apart. A parser written anywhere else would be a second statement of
/// the same format, and a second statement is the thing that goes stale.
///
/// **Minutes are not wrapped at 60**, so a 90-minute meeting prints `[90:05]` rather than
/// `[01:30:05]`. One unambiguous format beats an `HH:MM:SS`/`MM:SS` switch that a reader has to
/// detect -- and it is why parsing refuses both a third field and a bare integer, either of
/// which would reintroduce exactly the ambiguity that decision exists to avoid.
///
/// Whole seconds, because that is the resolution a transcript prints at: a turn starting at
/// 754.3 s is labelled `[12:34]`, and 754.3 is not recoverable from what the user read. Turning
/// the label back into a turn is [`Transcript::voice_at`], which knows about that truncation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TranscriptTime(u64);

impl TranscriptTime {
    /// The label a turn starting at `seconds` from session start is printed with.
    ///
    /// Floors to the whole second and clamps a negative to zero; nothing before session start
    /// is representable, and a turn cannot begin there.
    pub fn of(seconds: f64) -> Self {
        TranscriptTime(seconds.max(0.0).floor() as u64)
    }

    /// The instant this label names, in seconds from session start.
    pub fn seconds(&self) -> f64 {
        self.0 as f64
    }
}

impl fmt::Display for TranscriptTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let minutes = self.0 / 60;
        let seconds = self.0 % 60;
        write!(f, "{minutes:02}:{seconds:02}")
    }
}

impl FromStr for TranscriptTime {
    type Err = TimestampError;

    /// Reads a label back off a transcript.
    ///
    /// Accepts what a transcript line actually contains, brackets and surrounding whitespace
    /// included, so `[12:34]` pasted straight out of `transcript.md` works as well as `12:34`.
    /// Any number of minute digits (`90:05`, `120:00`), and seconds as two digits in `00..=59`.
    ///
    /// Refuses anything no transcript prints -- a third field, a bare integer, one-digit
    /// seconds -- rather than guessing: see the type's documentation for why an `HH:MM:SS`
    /// reading must not be available here.
    fn from_str(s: &str) -> std::result::Result<Self, TimestampError> {
        let malformed = || TimestampError(s.to_string());

        let trimmed = s.trim();
        let inner = match (trimmed.strip_prefix('['), trimmed.strip_suffix(']')) {
            (Some(_), Some(_)) => trimmed[1..trimmed.len() - 1].trim(),
            (None, None) => trimmed,
            // One bracket without the other is a truncated paste, not a spelling.
            _ => return Err(malformed()),
        };

        let (minutes, seconds) = inner.split_once(':').ok_or_else(malformed)?;
        let digits = |part: &str| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit());
        if !digits(minutes) || seconds.len() != 2 || !digits(seconds) {
            return Err(malformed());
        }

        let minutes: u64 = minutes.parse().map_err(|_| malformed())?;
        let seconds: u64 = seconds.parse().map_err(|_| malformed())?;
        if seconds > 59 {
            return Err(malformed());
        }
        minutes
            .checked_mul(60)
            .and_then(|m| m.checked_add(seconds))
            .map(TranscriptTime)
            .ok_or_else(malformed)
    }
}

/// A string that is not a timestamp any transcript prints.
///
/// Its own type rather than a variant of [`Error`], which is about session *files*: this is a
/// user typing at a command edge, and answering it needs no session to compare against.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "malformed timestamp {0:?}: expected MM:SS as a transcript prints it, such as 12:34 or 90:05"
)]
pub struct TimestampError(String);

/// Who was speaking at a given instant -- or, when nobody nameable was, which of the four ways
/// that happened.
///
/// Deliberately not an `Option<u32>`. All four non-answers mean "no cluster id", but they are
/// four different things to tell a user: only one of them is a mistake on their part, and each
/// of the others suggests a different next move. Collapsing them here would leave every caller
/// able to say nothing but "no".
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoiceAt {
    /// A speaker-track turn attributed to a voice. The one answer that names something.
    Cluster(u32),
    /// A mic-track turn: the machine's own user, [`YOU`] by construction and belonging to no
    /// cluster, so there is nothing here to rename.
    LocalSpeaker,
    /// A speaker-track turn whose `cluster` is null -- diarization found no voices in this
    /// session, so its turns have no provenance to hang a name on.
    NoCluster,
    /// Inside the session, but between turns: nobody was speaking at that instant.
    Silence,
    /// At or after the end of every turn. `last` is when the last one ended, which is what
    /// lets a caller say how far past the end the timestamp was.
    PastEnd { last: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SessionMetadata, VoiceAt};
    use jiff::Timestamp;

    fn mic_turn(start: f64, end: f64, text: &str) -> Turn {
        Turn {
            speaker: YOU.to_string(),
            start,
            end,
            text: text.to_string(),
            source_track: SourceTrack::Mic,
            cluster: None,
            speaker_id_confidence: None,
        }
    }

    fn session_id() -> SessionId {
        SessionId::parse("20260809-052600").unwrap()
    }

    /// A round-trip cannot distinguish `"speaker_id_confidence": null` from the key being
    /// absent -- both deserialize to `None` -- so the serialized text is what has to be
    /// asserted on.
    #[test]
    fn null_confidence_is_written_as_a_present_key() {
        let transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![mic_turn(0.0, 1.0, "hello")],
        );
        let json = serde_json::to_string(&transcript).unwrap();
        assert!(
            json.contains(r#""speaker_id_confidence":null"#),
            "confidence key missing from {json}"
        );
        assert!(json.contains(r#""source_track":"mic""#), "{json}");
    }

    /// The same argument as above, for the same reason: a mic turn's `null` cluster is the
    /// claim "this voice came from no cluster", and a reader can only tell that from "written
    /// by a tool that did not record provenance" if the key is there.
    #[test]
    fn a_null_cluster_is_written_as_a_present_key() {
        let transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![mic_turn(0.0, 1.0, "hello")],
        );
        let json = serde_json::to_string(&transcript).unwrap();
        assert!(
            json.contains(r#""cluster":null"#),
            "cluster key missing from {json}"
        );
    }

    /// The compatibility decision recorded on [`TRANSCRIPT_SCHEMA_VERSION`], made visible:
    /// a version 1 transcript is refused rather than read with its turns' provenance
    /// fabricated as "no cluster".
    #[test]
    fn a_transcript_without_clusters_is_refused_rather_than_defaulted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("transcript.json");
        std::fs::write(
            &path,
            br#"{
              "schema_version": 1,
              "session_id": "20260809-052600",
              "turns": [
                {
                  "speaker": "Unknown 1",
                  "start": 0.0,
                  "end": 1.0,
                  "text": "hi there",
                  "source_track": "speaker",
                  "speaker_id_confidence": null
                }
              ]
            }"#,
        )
        .unwrap();

        let error = Transcript::read(&path).unwrap_err().to_string();
        assert!(error.contains("cluster"), "{error}");
    }

    /// Acceptance criterion #7: `transcript.json` is the machine-readable artifact and templates
    /// are not its business, so its bytes are pinned against a literal rather than against
    /// whatever the code currently produces.
    #[test]
    fn transcript_json_is_unaffected_by_templating() {
        let dir = tempfile::tempdir().unwrap();
        let session = SessionPaths::new(dir.path());
        let md = metadata();
        Transcript::new(session_id(), vec![mic_turn(12.34, 14.0, "first")])
            .write(
                &session,
                &TranscriptTemplate::builtin(),
                &TranscriptContext::at(&md, rendered_at()),
            )
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(session.transcript_json()).unwrap(),
            r#"{
  "schema_version": 2,
  "session_id": "20260809-052600",
  "turns": [
    {
      "speaker": "You",
      "start": 12.34,
      "end": 14.0,
      "text": "first",
      "source_track": "mic",
      "cluster": null,
      "speaker_id_confidence": null
    }
  ]
}
"#
        );
    }

    fn speaker_turn(start: f64, end: f64, cluster: Option<u32>) -> Turn {
        Turn {
            speaker: unknown_speaker(cluster.unwrap_or(1) as usize),
            start,
            end,
            text: "words".to_string(),
            source_track: SourceTrack::Speaker,
            cluster,
            speaker_id_confidence: None,
        }
    }

    fn at(s: &str) -> TranscriptTime {
        s.parse().unwrap()
    }

    /// A fixed render instant, so two renderings of the same input are comparable.
    fn rendered_at() -> Timestamp {
        "2026-08-09T07:00:00Z".parse().unwrap()
    }

    fn metadata() -> SessionMetadata {
        let sync = crate::TrackSync {
            host_ticks: 1,
            timebase_numer: 125,
            timebase_denom: 3,
        };
        SessionMetadata::new(
            session_id(),
            "2026-08-09T05:26:00Z".parse().unwrap(),
            sync,
            sync,
        )
    }

    /// The point of giving the format one owner: whatever a transcript prints, this reads back
    /// to the second it named. Asserted as a round trip rather than as two hand-written
    /// literals agreeing, because two literals only pin today's behaviour of both sides at once
    /// and would keep agreeing if the pair drifted together.
    ///
    /// The print side is also pinned independently, against `[90:05]` in
    /// `the_default_template_renders_one_line_per_turn_under_frontmatter`.
    #[test]
    fn a_printed_timestamp_parses_back_to_the_second_it_names() {
        for seconds in [0.0, 12.34, 59.9, 60.0, 754.3, 3600.0, 5405.0, 7205.0] {
            let printed = TranscriptTime::of(seconds);
            let parsed: TranscriptTime = printed.to_string().parse().unwrap();
            assert_eq!(parsed, printed, "{seconds} printed as {printed}");
            assert_eq!(parsed.seconds(), seconds.floor(), "{printed}");
        }
    }

    /// Minutes past 60 are not wrapped on the way out, so they must not be on the way back in.
    #[test]
    fn minutes_past_an_hour_are_read_unwrapped() {
        assert_eq!(TranscriptTime::of(5405.0).to_string(), "90:05");
        assert_eq!(at("90:05").seconds(), 5405.0);
        assert_eq!(at("120:00").seconds(), 7200.0);
        // A single-digit minute, which is not a spelling any transcript prints but is one a
        // user types.
        assert_eq!(at("1:05").seconds(), 65.0);
    }

    /// A label copied straight off a transcript line brings its brackets with it.
    #[test]
    fn a_bracketed_label_parses() {
        assert_eq!(at("[12:34]"), at("12:34"));
        assert_eq!(at("  [12:34] "), at("12:34"));
    }

    /// Refused at parse time, with a message about the spelling and without a session to
    /// compare against. `12:34:56` and `12` are refused specifically: reading either would
    /// reintroduce the `HH:MM:SS` ambiguity that unwrapped minutes exist to avoid.
    #[test]
    fn a_malformed_timestamp_is_refused_with_the_spelling() {
        for bad in [
            "", "12", "12:5", "12:345", "12:60", "12:99", "12:34:56", "12x34", "-1:00", ":34",
            "12:", "ab:cd", "[12:34", "12:34]",
        ] {
            let error = bad.parse::<TranscriptTime>().unwrap_err().to_string();
            assert!(error.contains("MM:SS"), "{bad:?} gave {error}");
            assert!(error.contains(bad), "{bad:?} gave {error}");
        }
    }

    /// The truncated-label case, which containment alone gets wrong: this turn *prints*
    /// `[12:34]`, but 754.0 is a fraction of a second before it starts.
    #[test]
    fn a_timestamp_resolves_to_the_turn_whose_printed_label_it_is() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                speaker_turn(750.0, 754.0, Some(1)),
                speaker_turn(754.3, 758.0, Some(7)),
            ],
        );
        assert_eq!(transcript.voice_at(at("12:34")), VoiceAt::Cluster(7));
        // And containment still answers for an instant no label names.
        assert_eq!(transcript.voice_at(at("12:35")), VoiceAt::Cluster(7));
    }

    /// Both tracks are recorded at once and merged onto one timeline, so an instant can sit
    /// inside a turn from each. The mic track is never nameable, so preferring it would refuse
    /// a question that has an answer.
    #[test]
    fn an_instant_in_both_tracks_resolves_to_the_speaker_track() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(100.0, 105.0, "me talking over them"),
                speaker_turn(102.0, 104.0, Some(3)),
            ],
        );
        assert_eq!(transcript.voice_at(at("01:43")), VoiceAt::Cluster(3));
        // Outside the speaker turn, the mic turn is the honest answer.
        assert_eq!(transcript.voice_at(at("01:41")), VoiceAt::LocalSpeaker);
    }

    /// The preference also applies to the label match, not just to containment: two turns can
    /// start within the same second.
    #[test]
    fn a_label_shared_by_both_tracks_resolves_to_the_speaker_track() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(60.0, 62.0, "me"),
                speaker_turn(60.4, 61.0, Some(2)),
            ],
        );
        assert_eq!(transcript.voice_at(at("01:00")), VoiceAt::Cluster(2));
    }

    /// The four non-answers, each distinguishable from the others rather than collapsed into
    /// one word.
    #[test]
    fn the_four_non_answers_are_distinguishable() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(0.0, 1.0, "me"),
                speaker_turn(10.0, 11.0, None),
                speaker_turn(20.0, 21.0, Some(4)),
            ],
        );

        assert_eq!(transcript.voice_at(at("00:00")), VoiceAt::LocalSpeaker);
        assert_eq!(transcript.voice_at(at("00:10")), VoiceAt::NoCluster);
        assert_eq!(transcript.voice_at(at("00:05")), VoiceAt::Silence);
        assert_eq!(
            transcript.voice_at(at("00:30")),
            VoiceAt::PastEnd { last: 21.0 }
        );
        // ... and the answer that names something, so the five are five.
        assert_eq!(transcript.voice_at(at("00:20")), VoiceAt::Cluster(4));
    }

    /// The end instant belongs to whatever follows rather than to two turns at once, and the
    /// end of the last turn is already past the end of the session.
    #[test]
    fn a_turns_end_instant_belongs_to_what_follows() {
        let transcript = Transcript::new(session_id(), vec![speaker_turn(0.0, 5.0, Some(1))]);
        assert_eq!(transcript.voice_at(at("00:04")), VoiceAt::Cluster(1));
        assert_eq!(
            transcript.voice_at(at("00:05")),
            VoiceAt::PastEnd { last: 5.0 }
        );
    }

    /// A transcript with no turns has no instant anybody spoke at, and "past the end of
    /// nothing" is the reading that does not claim a silence inside a session.
    #[test]
    fn an_empty_transcript_is_past_its_end_everywhere() {
        let transcript = Transcript::new(session_id(), Vec::new());
        assert_eq!(
            transcript.voice_at(at("00:00")),
            VoiceAt::PastEnd { last: 0.0 }
        );
    }

    /// `Transcript` is a deserialized file, so the lookup must not lean on merge's sorting.
    #[test]
    fn the_lookup_does_not_assume_the_turns_are_sorted() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                speaker_turn(20.0, 21.0, Some(4)),
                speaker_turn(10.0, 11.0, Some(2)),
            ],
        );
        assert_eq!(transcript.voice_at(at("00:10")), VoiceAt::Cluster(2));
        assert_eq!(transcript.voice_at(at("00:20")), VoiceAt::Cluster(4));
    }

    /// Two voices can print the same label, and then the timestamp names neither of them on its
    /// own. `voice_at` still answers -- it has to answer something -- so the fact that it was
    /// choosing is what `clusters_at` exists to report.
    #[test]
    fn a_label_two_voices_share_lists_both_of_them() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(754.0, 754.2, "me"),
                speaker_turn(754.1, 754.5, Some(3)),
                speaker_turn(754.6, 758.0, Some(7)),
            ],
        );

        assert_eq!(transcript.voice_at(at("12:34")), VoiceAt::Cluster(3));
        assert_eq!(transcript.clusters_at(at("12:34")), [3, 7]);
    }

    /// The unambiguous cases: one voice is one voice, and a timestamp that names no voice at all
    /// names no voices at all.
    #[test]
    fn a_timestamp_that_names_one_voice_or_none_says_so() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(0.0, 1.0, "me"),
                speaker_turn(10.0, 11.0, None),
                speaker_turn(20.0, 21.0, Some(4)),
            ],
        );

        assert_eq!(transcript.clusters_at(at("00:20")), [4]);
        for nothing in ["00:00", "00:05", "00:10", "00:30"] {
            assert!(
                transcript.clusters_at(at(nothing)).is_empty(),
                "{nothing} reaches no voice"
            );
        }
    }
}
