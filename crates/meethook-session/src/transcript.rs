use std::path::{Path, PathBuf};

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
    /// Minutes in `turn.time` are not wrapped at 60, so a 90-minute meeting renders
    /// `[90:05]`. A single unambiguous format beats an `HH:MM:SS`/`MM:SS` switch that a
    /// reader has to detect, and computing it here rather than in the template means no
    /// template does clock arithmetic.
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
/// - `turns` -- each with `time` (the `MM:SS` label), `speaker`, `text`, `start`, `end`,
///   `source_track`, `cluster` and `speaker_id_confidence`.
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
    /// The shipped default: frontmatter, then one line per turn in the shape every transcript
    /// written before templates existed used.
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
}

impl<'a> RenderContext<'a> {
    fn new(transcript: &'a Transcript, ctx: &TranscriptContext<'a>) -> Self {
        RenderContext {
            session_id: &transcript.session_id,
            created: local_rfc3339(ctx.session.start_time),
            updated: local_rfc3339(ctx.rendered_at),
            meeting: ctx.session.meeting.as_ref(),
            turns: transcript.turns.iter().map(RenderTurn::new).collect(),
        }
    }
}

#[derive(Serialize)]
struct RenderTurn<'a> {
    /// `MM:SS` from session start, minutes unwrapped past 60.
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
        let total = turn.start.max(0.0);
        let minutes = (total / 60.0).floor() as u64;
        let seconds = (total - (minutes as f64) * 60.0).floor() as u64;
        RenderTurn {
            time: format!("{minutes:02}:{seconds:02}"),
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

    fn meeting(title: &str, notes: Option<&str>) -> crate::Meeting {
        crate::Meeting {
            title: title.to_string(),
            start: "2026-08-09T05:26:00Z".parse().unwrap(),
            end: "2026-08-09T06:26:00Z".parse().unwrap(),
            calendar: "Work".to_string(),
            organizer: None,
            attendees: Vec::new(),
            url: None,
            location: None,
            notes: notes.map(str::to_string),
            event_id: "event-1".to_string(),
        }
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

    /// The body half of the shipped default, unchanged from the format string it replaced: one
    /// line per turn, in transcript order, minutes unwrapped past 60. A pre-template
    /// `transcript.md` and a post-template one differ only by the header.
    #[test]
    fn the_default_template_renders_one_line_per_turn_under_frontmatter() {
        let rendered = render(&two_turns(), &TranscriptTemplate::builtin(), &metadata());
        let (_, body) = frontmatter(&rendered);
        assert_eq!(
            body, "\n**[00:12] You:** first\n**[90:05] You:** much later\n",
            "{rendered:?}"
        );
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
}
