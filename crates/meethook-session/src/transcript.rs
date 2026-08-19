use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use jiff::Timestamp;
use minijinja::{Environment, UndefinedBehavior, Value};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    Error, Meeting, Paths, Result, SessionId, SessionMetadata, SessionPaths, write_atomic,
};

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

    /// Renders the human-readable `transcript.md` body through `template`.
    ///
    /// A pure function of the turns and of `ctx` -- both of which every caller that
    /// re-renders an existing transcript can reach -- which is what lets `enroll` and
    /// `forget` rewrite the file in place after renaming speakers by calling this again
    /// rather than patching lines. It was a pure function of the turns alone before the
    /// output shape became the user's to choose; the template is resolved from the root
    /// rather than recorded per session precisely so that this stayed true.
    ///
    /// `turn.time` is a [`TranscriptTime`], which owns that format; computing it here rather
    /// than in the template means no template does clock arithmetic.
    pub fn render_markdown(
        &self,
        template: &TranscriptTemplate,
        ctx: &TranscriptContext<'_>,
    ) -> Result<String> {
        template.render(self, ctx)
    }

    /// Writes both transcript files atomically.
    ///
    /// Markdown first, JSON second, and the order is load-bearing: `transcript.json`'s
    /// presence is the "already transcribed" marker, so a crash between the two writes
    /// leaves a session that still re-transcribes rather than one that is marked done but
    /// missing its readable rendering.
    ///
    /// The rendering happens before either write, so a template that fails leaves both files
    /// exactly as they were rather than truncating the markdown it could not replace.
    pub fn write(
        &self,
        paths: &SessionPaths,
        template: &TranscriptTemplate,
        ctx: &TranscriptContext<'_>,
    ) -> Result<()> {
        let rendered = self.render_markdown(template, ctx)?;

        let md = paths.transcript_md();
        write_atomic(&md, rendered.as_bytes())?;

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

/// The template `transcript.md` is rendered through when the user has supplied none.
///
/// Shipped as a file rather than as a string literal so that it is also the worked example a
/// user copies to `<root>/transcript.md.jinja` and edits.
const BUILTIN_TEMPLATE: &str = include_str!("transcript.md.jinja");

/// What the built-in template is called in an error, since it has no path.
const BUILTIN_NAME: &str = "<meethook's built-in transcript.md.jinja>";

/// A compiled `transcript.md` template, ready to render any session's turns.
///
/// Compiled at construction rather than at render time, which is what lets the CLI resolve one
/// of these before a command does any work: a template with a syntax error then costs a
/// millisecond rather than an hour of transcription, and "an unusable template leaves the
/// existing `transcript.md` untouched" is a property of when this is built rather than a branch
/// somebody has to get right at each of the four write sites.
///
/// # The context a template receives
///
/// This is a user-facing surface -- changing it breaks templates people have written -- so it
/// is specified here rather than left to be read off the renderer:
///
/// - `session_id` -- the session directory's name, e.g. `20260809-052600`.
/// - `created` -- session start, RFC 3339 in the machine's local zone.
/// - `updated` -- the instant this rendering happened, same format. It moves on every
///   re-render, which is what makes it honest and what stops a `transcript.md` re-rendered by
///   `enroll` from claiming it is as old as the recording.
/// - `meeting` -- the calendar meeting, exactly as `session.json` spells it (`meeting.title`,
///   `meeting.notes`, `meeting.attendees`, ...), so a template and the JSON use one name per
///   concept. **Undefined**, not null, when the session was not recorded during a meeting,
///   which most are not.
/// - `meeting.fit` -- how strongly the session's start supports that being the meeting, as one
///   of `started`, `started_early`, `joined_late`, `after_end` or `unknown`. The first two are
///   strong; the last three are tentative and are why the shipped default emits a
///   `meeting_match:` key for them and nothing for the others. A recording that *overran* its
///   meeting is not tentative -- the fit is a function of the start alone. `unknown` is what a
///   `session.json` written before fits existed reads as, so it is never evidence of a good
///   match. See `MeetingFit`.
/// - `turns` -- each with `time` (the `MM:SS` label), `speaker`, `text`, `start`, `end`,
///   `source_track`, `cluster` and `speaker_id_confidence`.
/// - `blocks` -- the same turns, with each run of consecutive same-speaker turns collapsed into
///   one entry: `time` (the `MM:SS` label of the run's *first* turn), `speaker` (the shared
///   label), `text` (the run's texts joined by a single space), `start` (the first turn's),
///   `end` (the last turn's), and `turns`, the run's own turns in the shape above.
///
/// `turns` is the record and `blocks` is the reading of it. The shipped default renders
/// `blocks`, so a speaker who talks for a minute is one paragraph rather than a wall of
/// near-identical lines; a template that wants a line per turn keeps looping `turns` and gets
/// exactly what it always got. A block carries no `cluster`, `source_track` or
/// `speaker_id_confidence` because a run under one label can come from two clusters -- one
/// person whose voice clustering split in two, both named by `enroll` -- so no single value
/// would be a claim the block could make; those are reachable per turn through `block.turns`.
///
/// Undefined values are semi-strict: a template may test one for truth (`{% if meeting %}`),
/// which is how the built-in default emits meeting keys only when there is a meeting, but
/// *printing* one is an error rather than an empty string. A typo in a template name is worth
/// an error message; it is not worth an empty value silently written into somebody's notes.
///
/// A `yaml` filter is registered, and is how the default template puts meeting values into
/// YAML frontmatter -- see `yaml_filter` below for why the standard `tojson` is not used.
pub struct TranscriptTemplate {
    env: Environment<'static>,
    /// Named in every error this template produces. The built-in's is [`BUILTIN_NAME`].
    source: PathBuf,
    /// What the template is registered as in `env`, which is also what minijinja's own
    /// diagnosis cites the line number against. The template file's own name, so that
    /// "(in mine.jinja:3)" names the file the user is about to open rather than the file
    /// meethook was trying to write.
    name: String,
}

impl std::fmt::Debug for TranscriptTemplate {
    /// Hand-written because `minijinja::Environment`'s own `Debug` prints the compiled
    /// template source, and a user template is not this type's interesting part.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptTemplate")
            .field("source", &self.source)
            .finish()
    }
}

impl TranscriptTemplate {
    /// The shipped default: frontmatter, then one line per run of consecutive same-speaker
    /// turns -- see `blocks` in the context above.
    ///
    /// Infallible, and it has to stay that way -- it is the fallback the resolution below lands
    /// on for the overwhelmingly common case of a user who has never heard of templates. The
    /// test suite compiles it, so a syntax error in it fails `cargo test` rather than a run.
    pub fn builtin() -> Self {
        Self::compile(BUILTIN_TEMPLATE.to_string(), PathBuf::from(BUILTIN_NAME))
            .expect("the built-in transcript template must compile")
    }

    /// Reads and compiles a template file.
    ///
    /// Missing or unreadable is an [`Error::Io`], which already names the path; a syntax error
    /// is an [`Error::Template`], which names the path and minijinja's diagnosis.
    pub fn load(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        Self::compile(source, path.to_path_buf())
    }

    /// Picks the template a run should use: `explicit`, else `<root>/transcript.md.jinja`, else
    /// the built-in default.
    ///
    /// A named `explicit` that is missing or malformed is an error and never falls back. A user
    /// who asked for a template and quietly got the built-in one has been lied to, and would
    /// find out by reading a transcript rendered in the wrong shape.
    ///
    /// Every command that writes a `transcript.md` resolves through here, against the same
    /// root, which is what makes a re-render by `enroll` or `forget` use the template the
    /// transcript was originally written with.
    pub fn resolve(paths: &Paths, explicit: Option<&Path>) -> Result<Self> {
        if let Some(path) = explicit {
            return Self::load(path);
        }
        let in_root = paths.transcript_template();
        if in_root.exists() {
            return Self::load(&in_root);
        }
        Ok(Self::builtin())
    }

    fn compile(source: String, path: PathBuf) -> Result<Self> {
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into(),
        );

        let mut env = Environment::new();
        // See this type's documentation: testable for truth, an error to print. What makes
        // "no empty or null frontmatter values" a property of the engine rather than of the
        // data each template happens to be handed.
        env.set_undefined_behavior(UndefinedBehavior::SemiStrict);
        env.add_filter("yaml", yaml_filter);
        env.add_template_owned(name.clone(), source)
            .map_err(|e| Error::template(&path, e))?;
        Ok(TranscriptTemplate {
            env,
            source: path,
            name,
        })
    }

    fn render(&self, transcript: &Transcript, ctx: &TranscriptContext<'_>) -> Result<String> {
        let template = self
            .env
            .get_template(&self.name)
            .map_err(|e| Error::template(&self.source, e))?;
        template
            .render(RenderContext::new(transcript, ctx))
            .map_err(|e| Error::template(&self.source, e))
    }
}

/// Encodes a value as JSON, for interpolation into the YAML frontmatter of a transcript.
///
/// JSON is a subset of YAML, so a JSON-encoded scalar is a valid YAML double-quoted scalar:
/// this is what lets a meeting title containing `: ` or a multi-line invite body land in
/// frontmatter without breaking it. Every meeting field a template puts up there should go
/// through it.
///
/// minijinja's own `tojson` would do the same job and is deliberately not used. It is
/// HTML-safe, escaping `<`, `>`, `&` and `'` as `\uXXXX` -- valid YAML, and it renders the
/// ordinary meeting title "Design & Review" as `"Design & Review"`. Frontmatter is read
/// by people.
fn yaml_filter(value: Value) -> std::result::Result<String, minijinja::Error> {
    // No `with_source`: a transcript template's errors are printed, and everything reachable
    // from here is meeting content. The value cannot be encoded is the whole of what a user
    // needs, and serde_json's own message would be about the same value.
    serde_json::to_string(&value).map_err(|_| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "the yaml filter was given a value that cannot be encoded",
        )
    })
}

/// The session a transcript is being rendered for, and when.
///
/// The render instant is carried rather than read from the clock inside the renderer so that a
/// test can pin it, which is the only way a byte-for-byte comparison of two renderings can
/// still mean something now that `updated` exists.
#[derive(Debug, Clone, Copy)]
pub struct TranscriptContext<'a> {
    session: &'a SessionMetadata,
    rendered_at: Timestamp,
}

impl<'a> TranscriptContext<'a> {
    /// The ordinary constructor: this rendering is happening now.
    pub fn now(session: &'a SessionMetadata) -> Self {
        TranscriptContext {
            session,
            rendered_at: Timestamp::now(),
        }
    }

    /// A rendering pinned to a given instant, for tests and for any caller that wants two
    /// renderings to be comparable.
    pub fn at(session: &'a SessionMetadata, rendered_at: Timestamp) -> Self {
        TranscriptContext {
            session,
            rendered_at,
        }
    }
}

/// What a template actually sees. Documented on [`TranscriptTemplate`], which is where a reader
/// looking for the template contract will go.
#[derive(Serialize)]
struct RenderContext<'a> {
    session_id: &'a SessionId,
    created: String,
    updated: String,
    /// Skipped rather than null when there is no meeting, so a template sees an *undefined*
    /// `meeting` and `{% if meeting %}` is the whole of the guard.
    #[serde(skip_serializing_if = "Option::is_none")]
    meeting: Option<&'a Meeting>,
    turns: Vec<RenderTurn<'a>>,
    blocks: Vec<RenderBlock<'a>>,
}

impl<'a> RenderContext<'a> {
    fn new(transcript: &'a Transcript, ctx: &TranscriptContext<'a>) -> Self {
        RenderContext {
            session_id: &transcript.session_id,
            created: local_rfc3339(ctx.session.start_time),
            updated: local_rfc3339(ctx.rendered_at),
            meeting: ctx.session.meeting.as_ref(),
            turns: transcript.turns.iter().map(RenderTurn::new).collect(),
            blocks: RenderBlock::group(&transcript.turns),
        }
    }
}

#[derive(Serialize)]
struct RenderTurn<'a> {
    /// The turn's start as [`TranscriptTime`] spells it.
    time: String,
    speaker: &'a str,
    text: &'a str,
    start: f64,
    end: f64,
    source_track: SourceTrack,
    cluster: Option<u32>,
    speaker_id_confidence: Option<f32>,
}

impl<'a> RenderTurn<'a> {
    fn new(turn: &'a Turn) -> Self {
        RenderTurn {
            // The only place a transcript's timestamps are written, and it goes through the
            // type that also reads them back. See `TranscriptTime`.
            time: TranscriptTime::of(turn.start).to_string(),
            speaker: &turn.speaker,
            text: &turn.text,
            start: turn.start,
            end: turn.end,
            source_track: turn.source_track,
            cluster: turn.cluster,
            speaker_id_confidence: turn.speaker_id_confidence,
        }
    }
}

/// A run of consecutive turns by one speaker, rendered as a single timestamped paragraph.
///
/// The reading of the turns rather than the record of them: a reader does not need to know at
/// what second the same person started their second sentence, so the block is timestamped at
/// the moment that speaker took the floor and prints their name once. Field names mirror
/// [`RenderTurn`]'s so a template author's knowledge carries across, and `turns` keeps the
/// run's own turns reachable for anything a block cannot honestly claim -- see the context
/// documented on [`TranscriptTemplate`].
#[derive(Serialize)]
struct RenderBlock<'a> {
    /// The *first* turn's start as [`TranscriptTime`] spells it. This is the only timestamp the
    /// block prints, and it is one a turn really started at, so it still parses back and
    /// resolves to the voice speaking then.
    time: String,
    /// The label shared by every turn in the run.
    speaker: &'a str,
    /// The run's texts joined by a single space. Turn text arrives already trimmed and
    /// non-empty from transcription, so a single space is the whole of the separator; a
    /// one-turn block therefore reproduces that turn's line byte for byte.
    text: String,
    /// The first turn's start, in seconds.
    start: f64,
    /// The *last* turn's end, in seconds, so `start`..`end` spans the whole run.
    end: f64,
    turns: Vec<RenderTurn<'a>>,
}

impl<'a> RenderBlock<'a> {
    /// Collapses `turns` into blocks: a new block starts wherever the speaker label changes.
    ///
    /// Walks the slice in file order and never sorts. `Transcript` is a deserialized file and
    /// the rest of this module is careful not to assume its turns are sorted (see
    /// `candidates_at`); the blocks are the order the transcript is read in, whatever that is.
    ///
    /// Grouping is on the label text alone, deliberately -- the label is exactly the thing a
    /// collapsed block prints once, so grouping on anything finer (cluster, track) would let
    /// two adjacent blocks print the same name twice in a row. There is no gap threshold
    /// either: two same-speaker turns an hour apart still collapse.
    fn group(turns: &'a [Turn]) -> Vec<Self> {
        let mut blocks: Vec<Self> = Vec::new();
        for turn in turns {
            match blocks.last_mut() {
                Some(block) if block.speaker == turn.speaker => {
                    block.text.push(' ');
                    block.text.push_str(&turn.text);
                    block.end = turn.end;
                    block.turns.push(RenderTurn::new(turn));
                }
                _ => blocks.push(RenderBlock {
                    time: TranscriptTime::of(turn.start).to_string(),
                    speaker: &turn.speaker,
                    text: turn.text.clone(),
                    start: turn.start,
                    end: turn.end,
                    turns: vec![RenderTurn::new(turn)],
                }),
            }
        }
        blocks
    }
}

/// An instant as RFC 3339 in the machine's local zone, e.g. `2026-08-05T14:29:21-05:00`.
///
/// Local rather than UTC because these two keys are read beside a transcript of a meeting the
/// reader attended, and "2pm" is what they remember. `jiff::Zoned`'s own `Display` is not
/// usable here: it appends the IANA name in brackets, which is a jiff/RFC 9557 extension that
/// an ordinary YAML timestamp reader will reject.
fn local_rfc3339(at: Timestamp) -> String {
    at.to_zoned(jiff::tz::TimeZone::system())
        .strftime("%Y-%m-%dT%H:%M:%S%:z")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn two_turns() -> Transcript {
        Transcript::new(
            session_id(),
            vec![
                mic_turn(12.34, 14.0, "first"),
                mic_turn(5405.0, 5410.0, "much later"),
            ],
        )
    }

    /// The same two turns with another speaker between them, so the default renders three
    /// lines. `two_turns()` collapses into one under the shipped template, and several of the
    /// tests below want a body with more than one line in it.
    fn alternating() -> Transcript {
        Transcript::new(
            session_id(),
            vec![
                mic_turn(12.34, 14.0, "first"),
                speaker_turn(20.0, 25.0, Some(1)),
                mic_turn(5405.0, 5410.0, "much later"),
            ],
        )
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

    /// A meeting the session is claimed to have actually been -- the ordinary case, and the
    /// one the frontmatter tests below are written around.
    fn meeting(title: &str, notes: Option<&str>) -> crate::Meeting {
        meeting_with_fit(title, notes, crate::MeetingFit::Started)
    }

    fn meeting_with_fit(
        title: &str,
        notes: Option<&str>,
        fit: crate::MeetingFit,
    ) -> crate::Meeting {
        crate::Meeting::new(
            "event-1".to_string(),
            title.to_string(),
            "Work".to_string(),
            "2026-08-09T05:26:00Z".parse().unwrap(),
            "2026-08-09T06:26:00Z".parse().unwrap(),
        )
        .with_invite(None, None, notes.map(str::to_string))
        .with_fit(fit)
    }

    /// A fixed render instant, so two renderings of the same input are comparable.
    fn rendered_at() -> Timestamp {
        "2026-08-09T07:00:00Z".parse().unwrap()
    }

    fn render(
        transcript: &Transcript,
        template: &TranscriptTemplate,
        md: &SessionMetadata,
    ) -> String {
        transcript
            .render_markdown(template, &TranscriptContext::at(md, rendered_at()))
            .unwrap()
    }

    /// Splits a rendering into its YAML frontmatter lines and its body.
    fn frontmatter(rendered: &str) -> (Vec<&str>, &str) {
        let rest = rendered
            .strip_prefix("---\n")
            .unwrap_or_else(|| panic!("no opening frontmatter fence in {rendered:?}"));
        let (block, body) = rest
            .split_once("\n---\n")
            .unwrap_or_else(|| panic!("no closing frontmatter fence in {rendered:?}"));
        (block.lines().collect(), body)
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

    /// The body half of the shipped default: one line per *run* of consecutive same-speaker
    /// turns, in transcript order, minutes unwrapped past 60.
    ///
    /// `two_turns()` is one speaker twice, ninety minutes apart, and it collapses -- adjacency
    /// and the label are the whole of the rule, with no gap threshold. So this no longer
    /// asserts what it did before templates existed, when the body was one line per turn.
    #[test]
    fn the_default_template_renders_one_block_per_speaker_run_under_frontmatter() {
        let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &metadata());
        let (_, body) = frontmatter(&rendered);
        assert_eq!(
            body, "\n**[00:12] You:** first much later\n",
            "{rendered:?}"
        );
    }

    /// The multi-line body a collapsing transcript still has: alternating speakers give a line
    /// each, in transcript order, and the unwrapped minutes are pinned here now that
    /// `two_turns()` renders as one line.
    #[test]
    fn the_default_template_renders_alternating_speakers_one_line_each() {
        let rendered = render(&alternating(), &TranscriptTemplate::builtin(), &metadata());
        let (_, body) = frontmatter(&rendered);
        assert_eq!(
            body,
            "\n**[00:12] You:** first\n**[00:20] Unknown 1:** words\n\
             **[90:05] You:** much later\n",
            "{rendered:?}"
        );
    }

    /// Acceptance criterion #1: a run of three collapses to one line, timestamped at the first
    /// turn's start, with the label printed once and the texts joined by a single space.
    #[test]
    fn consecutive_turns_by_one_speaker_collapse_under_the_first_timestamp() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(12.34, 14.0, "So the thing about the merge threshold"),
                mic_turn(14.9, 17.0, "is that it only looks at centroids."),
                mic_turn(17.2, 21.0, "Which is why the second pass exists."),
            ],
        );
        let rendered = render(&transcript, &TranscriptTemplate::builtin(), &metadata());
        let (_, body) = frontmatter(&rendered);
        assert_eq!(
            body,
            "\n**[00:12] You:** So the thing about the merge threshold is that it only \
             looks at centroids. Which is why the second pass exists.\n",
            "{rendered:?}"
        );
    }

    /// Acceptance criterion #2: a turn between two different speakers renders the line it
    /// rendered before collapsing existed, asserted as a byte equality rather than by
    /// re-deriving it -- a one-turn block is meant to be an identity, not an approximation.
    #[test]
    fn a_turn_between_different_speakers_renders_exactly_as_it_did() {
        let rendered = render(&alternating(), &TranscriptTemplate::builtin(), &metadata());
        let (_, body) = frontmatter(&rendered);
        assert!(
            body.contains("\n**[00:20] Unknown 1:** words\n"),
            "{rendered:?}"
        );
    }

    /// Acceptance criterion #4, as a property rather than an example, in the spirit of
    /// TASK-033.01: every `[MM:SS]` the collapsed body prints parses back and resolves to the
    /// voice that opened that block. Collapsing removes timestamps; it must not invent one.
    ///
    /// The third turn is the case that matters -- it is swallowed into cluster 1's block, so
    /// the second it started at is no longer printed anywhere and the label that *is* printed
    /// still has to reach the right voice.
    #[test]
    fn every_timestamp_a_collapsed_transcript_prints_resolves_to_its_blocks_voice() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                speaker_turn(750.0, 754.0, Some(1)),
                speaker_turn(754.3, 758.0, Some(1)),
                speaker_turn(760.0, 764.0, Some(1)),
                speaker_turn(800.0, 809.0, Some(7)),
                speaker_turn(880.0, 884.0, Some(1)),
            ],
        );
        let rendered = render(&transcript, &TranscriptTemplate::builtin(), &metadata());
        let (_, body) = frontmatter(&rendered);

        let printed: Vec<TranscriptTime> = body
            .lines()
            .filter_map(|line| {
                line.split_once('[')
                    .and_then(|(_, rest)| rest.split_once(']'))
            })
            .map(|(label, _)| label)
            .map(|label| label.parse().expect(label))
            .collect();
        assert_eq!(printed, [at("12:30"), at("13:20"), at("14:40")], "{body:?}");
        assert_eq!(
            printed
                .iter()
                .map(|t| transcript.voice_at(*t))
                .collect::<Vec<_>>(),
            [
                VoiceAt::Cluster(1),
                VoiceAt::Cluster(7),
                VoiceAt::Cluster(1)
            ],
            "{body:?}"
        );
        // The second a swallowed turn started at is gone from the rendering, which is the
        // whole of what collapsing removes.
        assert!(!body.contains("12:34"), "{body:?}");
    }

    /// Acceptance criterion #5: `turns` is the record and is untouched by collapsing, so a
    /// template that loops it gets one line per turn even where the default would collapse
    /// them.
    #[test]
    fn a_user_template_looping_turns_still_gets_one_line_per_turn() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mine.jinja");
        std::fs::write(
            &path,
            "{% for t in turns %}{{ t.time }}|{{ t.speaker }}|{{ t.text }}\n{% endfor %}",
        )
        .unwrap();

        let rendered = render(
            &two_turns(),
            &TranscriptTemplate::load(&path).unwrap(),
            &metadata(),
        );
        assert_eq!(rendered, "00:12|You|first\n90:05|You|much later\n");
    }

    /// The per-turn provenance a block deliberately does not carry is still reachable through
    /// the block's own turns, which is what makes leaving it off the block honest rather than
    /// lossy.
    #[test]
    fn a_blocks_turns_carry_the_provenance_the_block_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mine.jinja");
        std::fs::write(
            &path,
            "{% for b in blocks %}{{ b.start }}-{{ b.end }}:\
             {% for t in b.turns %} {{ t.cluster }}/{{ t.source_track }}{% endfor %}\n\
             {% endfor %}",
        )
        .unwrap();

        // One label, two clusters -- the case a block-level `cluster` would have to lie about.
        let mut second = speaker_turn(30.0, 40.0, Some(2));
        second.speaker = unknown_speaker(1);
        let transcript = Transcript::new(
            session_id(),
            vec![speaker_turn(10.0, 20.0, Some(1)), second],
        );
        let rendered = render(
            &transcript,
            &TranscriptTemplate::load(&path).unwrap(),
            &metadata(),
        );
        assert_eq!(rendered, "10.0-40.0: 1/speaker 2/speaker\n");
    }

    /// The header half: `created` is the session start and `updated` the render instant, both
    /// RFC 3339. Asserted by parsing them back rather than by restating a formatted string,
    /// which would only assert the local zone of whichever machine ran the test.
    #[test]
    fn the_default_template_renders_created_and_updated_as_rfc_3339_instants() {
        let md = metadata();
        let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &md);
        let (lines, _) = frontmatter(&rendered);

        let created = lines[0].strip_prefix("created: ").expect(lines[0]);
        let updated = lines[1].strip_prefix("updated: ").expect(lines[1]);
        assert_eq!(created.parse::<Timestamp>().unwrap(), md.start_time);
        assert_eq!(updated.parse::<Timestamp>().unwrap(), rendered_at());
        // The RFC 9557 bracket `Zoned`'s own Display would append is what an ordinary YAML
        // timestamp reader chokes on, so its absence is the point rather than cosmetics.
        assert!(!created.contains('['), "{created}");
    }

    /// Acceptance criterion #3.
    #[test]
    fn the_meeting_title_and_notes_reach_the_frontmatter() {
        let md = metadata().with_meeting(Some(meeting("Weekly sync", Some("the agenda"))));
        let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &md);
        let (lines, _) = frontmatter(&rendered);

        assert_eq!(lines[2], r#"meeting_title: "Weekly sync""#, "{rendered:?}");
        assert_eq!(
            lines[3], r#"meeting_description: "the agenda""#,
            "{rendered:?}"
        );
    }

    /// A weak match is visibly tentative in the frontmatter, and a strong one adds nothing.
    ///
    /// Driven over every [`crate::MeetingFit`] variant rather than the ones somebody
    /// remembered, and asserted against `is_strong()` rather than against a second list: this
    /// is what stops the template's list of tentative fits and the Rust predicate from
    /// drifting apart when a variant is added.
    #[test]
    fn a_tentative_meeting_match_reaches_the_frontmatter_and_a_strong_one_does_not() {
        for fit in crate::MeetingFit::ALL {
            let md = metadata().with_meeting(Some(meeting_with_fit("Weekly sync", None, fit)));
            let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &md);
            let (lines, _) = frontmatter(&rendered);
            let matched: Vec<&&str> = lines
                .iter()
                .filter(|line| line.starts_with("meeting_match: "))
                .collect();

            if fit.is_strong() {
                assert!(matched.is_empty(), "{fit:?} must be silent: {rendered:?}");
                continue;
            }
            assert_eq!(matched.len(), 1, "{fit:?} must be marked: {rendered:?}");
            // The rendered value is the serde spelling, so the frontmatter and `session.json`
            // say the same word for the same fit.
            let spelling = serde_json::to_string(&fit).unwrap();
            assert_eq!(*matched[0], format!("meeting_match: {spelling}"));
        }
    }

    /// Acceptance criterion #4: most sessions are recorded outside any meeting, and those must
    /// not grow `meeting_title:` with nothing after the colon.
    ///
    /// Both halves are checked, because an empty value and a `null` one are different failures:
    /// no frontmatter key mentions a meeting at all, and every key that *is* there has a
    /// non-empty value that is not `null`.
    #[test]
    fn a_session_with_no_meeting_emits_no_meeting_keys_and_no_empty_values() {
        let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &metadata());
        let (lines, _) = frontmatter(&rendered);

        assert_eq!(lines.len(), 2, "{rendered:?}");
        for line in &lines {
            let (key, value) = line.split_once(": ").unwrap_or_else(|| panic!("{line:?}"));
            assert!(!key.contains("meeting"), "{rendered:?}");
            assert!(!value.trim().is_empty(), "{line:?}");
            assert_ne!(value.trim(), "null", "{line:?}");
        }
    }

    /// A meeting whose title has a colon in it and whose invite body is several lines long is
    /// the ordinary case, not a hostile one, and interpolated raw either would produce
    /// frontmatter no reader can parse.
    ///
    /// Checked by decoding each value back, which is the claim that matters: a JSON string is a
    /// valid YAML double-quoted scalar, so a value that round-trips through `serde_json` is a
    /// value a YAML reader recovers intact.
    #[test]
    fn a_colon_in_a_title_and_a_multi_line_body_survive_the_frontmatter() {
        let title = "Design & Review: Q3 #plans";
        let notes = "line one\nline two: with a colon\n# and a hash\n\"quoted\"";
        let md = metadata().with_meeting(Some(meeting(title, Some(notes))));
        let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &md);
        let (lines, body) = frontmatter(&rendered);

        // One line per key however many lines the invite body had: a raw newline here would
        // end the frontmatter block early and spill the agenda into the transcript.
        assert_eq!(lines.len(), 4, "{rendered:?}");
        let decode = |line: &str| {
            let (_, value) = line.split_once(": ").unwrap();
            serde_json::from_str::<String>(value).unwrap_or_else(|e| panic!("{line:?}: {e}"))
        };
        assert_eq!(decode(lines[2]), title);
        assert_eq!(decode(lines[3]), notes);
        // `tojson` would have written `Design & Review` here. See `yaml_filter`.
        assert!(lines[2].contains("Design & Review"), "{rendered:?}");
        assert!(body.starts_with("\n**[00:12] You:**"), "{rendered:?}");
    }

    /// Acceptance criterion #2: a user template is the whole of the output, frontmatter
    /// included -- nothing of the default is prepended, appended, or merged in.
    ///
    /// Its body loops `turns`, so it is also the guard that collapsing left that view alone:
    /// these two turns share a speaker and the default now renders them as one line, while
    /// this template's expected output has not moved.
    #[test]
    fn a_user_template_fully_replaces_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mine.jinja");
        std::fs::write(
            &path,
            "---\ntags: [meeting]\nid: {{ session_id }}\n---\n{% for t in turns %}\
             {{ t.time }}|{{ t.speaker }}|{{ t.text }}\n{% endfor %}",
        )
        .unwrap();

        let rendered = render(
            &two_turns(),
            &TranscriptTemplate::load(&path).unwrap(),
            &metadata(),
        );
        assert_eq!(
            rendered,
            "---\ntags: [meeting]\nid: 20260809-052600\n---\n\
             00:12|You|first\n90:05|You|much later\n"
        );
    }

    /// Precedence, and the reason a template lives at the root: `enroll` and `forget` resolve
    /// the same way `transcribe` did, so a re-render cannot revert a transcript to the default.
    #[test]
    fn resolution_prefers_an_explicit_template_then_the_root_then_the_builtin() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path().join("meethook"));
        std::fs::create_dir_all(paths.root()).unwrap();
        let named = dir.path().join("named.jinja");
        std::fs::write(&named, "named\n").unwrap();

        let builtin = TranscriptTemplate::resolve(&paths, None).unwrap();
        assert!(
            render(&two_turns(), &builtin, &metadata()).contains("**[00:12] You:** first"),
            "an absent template must fall back to the built-in default"
        );

        std::fs::write(paths.transcript_template(), "from the root\n").unwrap();
        let from_root = TranscriptTemplate::resolve(&paths, None).unwrap();
        assert_eq!(
            render(&two_turns(), &from_root, &metadata()),
            "from the root"
        );

        let explicit = TranscriptTemplate::resolve(&paths, Some(&named)).unwrap();
        assert_eq!(render(&two_turns(), &explicit, &metadata()), "named");
    }

    /// A user who named a template and silently got the built-in one has been lied to, and
    /// would find out by reading a transcript in the wrong shape -- so a named template that is
    /// not there is an error even though an *unnamed* one falls back.
    #[test]
    fn a_named_template_that_is_missing_is_an_error_rather_than_a_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let missing = dir.path().join("nope.jinja");

        let error = TranscriptTemplate::resolve(&paths, Some(&missing))
            .unwrap_err()
            .to_string();
        assert!(error.contains("nope.jinja"), "{error}");
    }

    /// Acceptance criterion #6, first half: a template that will not compile is refused when it
    /// is built, which is before any command has done any work.
    #[test]
    fn a_malformed_template_is_refused_at_construction_naming_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.jinja");
        std::fs::write(&path, "{% for turn in turns %}oops\n").unwrap();

        let error = TranscriptTemplate::load(&path).unwrap_err().to_string();
        assert!(error.contains("broken.jinja"), "{error}");
        assert!(error.contains("syntax error"), "{error}");
        // The line number is cited against the user's own file name, not against the name of
        // the file meethook was trying to write.
        assert!(error.contains("in broken.jinja:1"), "{error}");
    }

    /// Acceptance criterion #6, second half, and the part that is about privacy rather than
    /// diagnosis: a template that compiles but fails while rendering must leave the existing
    /// `transcript.md` exactly as it was, and its error must carry neither the transcript body
    /// nor the invite body.
    ///
    /// The failure used is a printed undefined value, which is the mistake a user actually
    /// makes -- a mistyped variable name -- and which is an error here only because the
    /// environment is semi-strict.
    #[test]
    fn a_render_failure_writes_nothing_and_prints_neither_the_transcript_nor_the_notes() {
        let dir = tempfile::tempdir().unwrap();
        let session = SessionPaths::new(dir.path());
        std::fs::write(session.transcript_md(), b"the previous rendering\n").unwrap();
        let path = dir.path().join("typo.jinja");
        std::fs::write(&path, "{{ meeting_titel }}\n").unwrap();

        let md =
            metadata().with_meeting(Some(meeting("Weekly sync", Some("dial-in PIN 4815162342"))));
        let template = TranscriptTemplate::load(&path).unwrap();
        let error = two_turns()
            .write(
                &session,
                &template,
                &TranscriptContext::at(&md, rendered_at()),
            )
            .unwrap_err()
            .to_string();

        assert!(error.contains("typo.jinja"), "{error}");
        assert!(error.contains("undefined"), "{error}");
        assert!(!error.contains("4815162342"), "{error}");
        assert!(!error.contains("much later"), "{error}");
        assert_eq!(
            std::fs::read(session.transcript_md()).unwrap(),
            b"the previous rendering\n",
            "a failed render must not touch the transcript it could not replace"
        );
        assert!(
            !session.transcript_json().exists(),
            "a failed render must not mark the session transcribed"
        );
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

    #[test]
    fn a_transcript_of_no_turns_still_renders_its_frontmatter() {
        let transcript = Transcript::new(session_id(), Vec::new());
        let rendered = render(&transcript, &TranscriptTemplate::builtin(), &metadata());
        let (lines, body) = frontmatter(&rendered);
        assert_eq!(lines.len(), 2, "{rendered:?}");
        assert_eq!(body, "\n", "{rendered:?}");
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
