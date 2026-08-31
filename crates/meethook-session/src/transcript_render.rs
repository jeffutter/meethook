//! Producing `transcript.md` and `transcript.vtt` from a [`Transcript`]'s turns.
//!
//! Owns the template engine and both output grammars: the user-facing markdown, rendered
//! through a minijinja template the user can replace, and the WebVTT captions, rendered
//! through the fixed format a player rejects when it is wrong. Everything here reads the
//! record through `turns` alone -- the record itself, its timestamp lookup, and its write
//! orchestration live in [`super::transcript`].

use std::path::{Path, PathBuf};

use jiff::Timestamp;
use minijinja::{Environment, UndefinedBehavior, Value};
use serde::Serialize;

use super::transcript::{SourceTrack, Transcript, TranscriptTime, Turn};
use crate::{Error, Meeting, Paths, Result, SessionId, SessionMetadata};

impl Transcript {
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

    /// Renders the same turns as WebVTT: the caption format media players, video tools and
    /// anything else that reads subtitles already understand.
    ///
    /// Not rendered through a template, deliberately. `transcript.md` is a document whose shape
    /// is the user's to choose; this is a machine format with a grammar a player rejects when it
    /// is wrong, so there is nothing here for a template to decide that would not be a way to
    /// produce a file nothing can read.
    ///
    /// Three differences from the markdown rendering, each of them the format's doing rather
    /// than a choice:
    ///
    /// - **A cue per turn, not per block.** The blocks a reader gets collapse a speaker's whole
    ///   run into one paragraph; as a cue that would hold the screen for minutes.
    /// - **`HH:MM:SS.mmm`, not `MM:SS`.** WebVTT wants minutes under 60 and reads milliseconds,
    ///   so [`TranscriptTime`]'s deliberately unwrapped minutes are not a legal cue timing. The
    ///   turns' own precision goes out, since a player seeks with it.
    /// - **Cues come out ordered by start.** The rest of this module never assumes the turns are
    ///   sorted (see `candidates_at`), but WebVTT requires non-decreasing cue starts, so this is
    ///   the one rendering that does not preserve the file's own order.
    ///
    /// Infallible: every turn is representable, so a caller has no error to handle and a
    /// transcript that writes at all writes its captions too.
    pub fn render_vtt(&self) -> String {
        let mut cues: Vec<&Turn> = self.turns.iter().collect();
        cues.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));

        let mut vtt = String::from("WEBVTT\n");
        for turn in cues {
            let start = cue_millis(turn.start);
            // A cue must end strictly after it begins. Clamping in milliseconds rather than in
            // seconds is what makes that true of the timings actually written: two instants a
            // tenth of a millisecond apart are one instant at this resolution.
            let end = cue_millis(turn.end).max(start + 1);
            vtt.push('\n');
            vtt.push_str(&cue_timestamp(start));
            vtt.push_str(" --> ");
            vtt.push_str(&cue_timestamp(end));
            // The `<v Name>` voice span rather than a "Name: " prefix: it is the format's own
            // way to say who is speaking, so a player can style or filter by speaker instead of
            // showing the label as words somebody said.
            vtt.push_str("\n<v ");
            vtt.push_str(&escape_cue_text(&turn.speaker));
            vtt.push('>');
            vtt.push_str(&escape_cue_text(&turn.text));
            vtt.push('\n');
        }
        vtt
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
/// - `meeting.fit` -- how strongly the session's start supports that being the meeting, as one
///   of `started`, `started_early`, `confirmed`, `joined_late`, `after_end` or `unknown`. The
///   first three are strong; the last three are tentative and are why the shipped default emits
///   a `meeting_match:` key for them and nothing for the others. A recording that *overran* its
///   meeting is not tentative -- the fit is a function of the start alone. `confirmed` is a
///   label a human chose with `meethook meeting`, which is the strongest of the six rather than
///   another degree of guess. `unknown` is what a `session.json` written before fits existed
///   reads as, so it is never evidence of a good match. See `MeetingFit`.
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

/// Seconds from session start as whole milliseconds, the resolution a WebVTT cue timing is
/// written at. Clamps a negative to zero, as [`TranscriptTime::of`] does.
fn cue_millis(seconds: f64) -> u64 {
    (seconds.max(0.0) * 1000.0).round() as u64
}

/// A cue timing: `HH:MM:SS.mmm`, hours always present.
///
/// Hours are not optional in the spelling even though WebVTT allows the short form, so every
/// timing in the file is the same width and a long meeting's cues do not change shape partway
/// through. Unlike [`TranscriptTime`], minutes and seconds wrap -- the format requires it.
fn cue_timestamp(millis: u64) -> String {
    let hours = millis / 3_600_000;
    let minutes = millis / 60_000 % 60;
    let seconds = millis / 1_000 % 60;
    let fraction = millis % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02}.{fraction:03}")
}

/// A speaker label or a turn's text as it can appear inside a cue.
///
/// `&`, `<` and `>` are WebVTT's own markup, so they are escaped -- which also means a `-->` a
/// speaker said cannot be read back as a cue timing, and a `>` in a name cannot end the voice
/// span early. Line breaks become spaces: a blank line ends a cue, so text carrying one would
/// truncate that cue and leave the remainder to be parsed as a malformed one.
fn escape_cue_text(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '\r' | '\n' => escaped.push(' '),
            _ => escaped.push(ch),
        }
    }
    escaped
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
    use crate::transcript::{YOU, unknown_speaker};
    use crate::{SessionPaths, VoiceAt};

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
        assert!(
            !session.transcript_vtt().exists(),
            "a failed render must not leave captions the markdown does not match"
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
    /// The whole file against a literal, because WebVTT is read by other people's parsers: the
    /// bytes *are* the contract, and "whatever the renderer currently emits" is not one.
    ///
    /// Note the last cue. The same turn is `[90:05]` in the markdown -- see
    /// [`TranscriptTime`]'s unwrapped minutes -- and `01:30:05.000` here, which is the one
    /// spelling a player accepts.
    #[test]
    fn the_captions_are_one_cue_per_turn_with_the_speaker_as_a_voice_span() {
        assert_eq!(
            alternating().render_vtt(),
            "WEBVTT\n\
             \n\
             00:00:12.340 --> 00:00:14.000\n\
             <v You>first\n\
             \n\
             00:00:20.000 --> 00:00:25.000\n\
             <v Unknown 1>words\n\
             \n\
             01:30:05.000 --> 01:30:10.000\n\
             <v You>much later\n"
        );
    }
    /// A file with no cues in it, rather than no file: a transcript of a silent session has been
    /// rendered, and a player opening it should find an empty caption track rather than an error.
    #[test]
    fn a_transcript_of_no_turns_is_still_a_well_formed_vtt_file() {
        assert_eq!(
            Transcript::new(session_id(), Vec::new()).render_vtt(),
            "WEBVTT\n"
        );
    }
    /// The rest of this module reads the turns in file order, whatever that is. WebVTT cannot:
    /// its cues have to be in non-decreasing start order.
    #[test]
    fn cues_are_ordered_by_start_however_the_turns_happen_to_be_stored() {
        let transcript = Transcript::new(
            session_id(),
            vec![
                mic_turn(30.0, 31.0, "third"),
                mic_turn(10.0, 11.0, "first"),
                mic_turn(20.0, 21.0, "second"),
            ],
        );
        let rendered = transcript.render_vtt();
        let cues: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("<v "))
            .collect();
        assert_eq!(
            cues,
            vec!["<v You>first", "<v You>second", "<v You>third"],
            "{rendered}"
        );
    }
    /// Everything a turn's own words could do to the format if they went out verbatim: end a
    /// voice span, open a tag, or -- the interesting one -- be read back as a cue timing.
    #[test]
    fn markup_a_speaker_said_cannot_escape_its_cue() {
        let mut turn = mic_turn(0.0, 1.0, "revenue --> costs, <b>always</b> & forever");
        turn.speaker = "Bo>b & co".to_string();
        let rendered = Transcript::new(session_id(), vec![turn]).render_vtt();

        assert_eq!(
            rendered,
            "WEBVTT\n\
             \n\
             00:00:00.000 --> 00:00:01.000\n\
             <v Bo&gt;b &amp; co>revenue --&gt; costs, &lt;b&gt;always&lt;/b&gt; &amp; forever\n"
        );
        // The cue holds three lines and the timing line is the only `-->` among them, which is
        // the whole point of escaping it: a parser splitting on `-->` still finds one cue.
        assert_eq!(rendered.matches("-->").count(), 1, "{rendered}");
    }
    /// A blank line is what ends a cue, so text carrying one would truncate its own cue and
    /// leave the remainder to be parsed as garbage.
    #[test]
    fn a_line_break_in_a_turns_text_does_not_split_its_cue() {
        let transcript = Transcript::new(
            session_id(),
            vec![mic_turn(0.0, 1.0, "first thought\n\nsecond thought")],
        );
        assert_eq!(
            transcript.render_vtt(),
            "WEBVTT\n\
             \n\
             00:00:00.000 --> 00:00:01.000\n\
             <v You>first thought  second thought\n"
        );
    }
    /// A cue must end strictly after it begins, so the degenerate turn a merge can produce has
    /// to be widened rather than written as the zero-length cue a strict parser rejects.
    #[test]
    fn a_turn_shorter_than_a_millisecond_still_ends_after_it_begins() {
        let transcript = Transcript::new(
            session_id(),
            vec![mic_turn(4.0, 4.0, "oh"), mic_turn(9.0, 9.0004, "ah")],
        );
        assert_eq!(
            transcript.render_vtt(),
            "WEBVTT\n\
             \n\
             00:00:04.000 --> 00:00:04.001\n\
             <v You>oh\n\
             \n\
             00:00:09.000 --> 00:00:09.001\n\
             <v You>ah\n"
        );
    }
    /// Written by the same call that writes the other two, which is what makes it impossible for
    /// a session to hold captions that disagree with its transcript: `enroll` and `forget`
    /// re-render through here after a rename without knowing this file exists.
    #[test]
    fn writing_a_transcript_puts_the_captions_beside_the_markdown() {
        let dir = tempfile::tempdir().unwrap();
        let session = SessionPaths::new(dir.path());
        let md = metadata();
        let transcript = alternating();
        transcript
            .write(
                &session,
                &TranscriptTemplate::builtin(),
                &TranscriptContext::at(&md, rendered_at()),
            )
            .unwrap();

        assert_eq!(
            std::fs::read_to_string(session.transcript_vtt()).unwrap(),
            transcript.render_vtt()
        );
    }
}
