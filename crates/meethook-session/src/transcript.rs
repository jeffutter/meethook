use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, write_atomic};

/// Bumped whenever `transcript.json`'s shape changes incompatibly.
///
/// Deliberately separate from [`crate::SESSION_SCHEMA_VERSION`]: `session.json` and
/// `transcript.json` are written by different commands at different times and evolve
/// independently, so one shared number would force a lie in one of the two files.
pub const TRANSCRIPT_SCHEMA_VERSION: u32 = 1;

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
    /// How confident the claim that `speaker` names the right *person* is: the cosine
    /// similarity between this voice and that person's enrolled reference, in `[-1, 1]` and
    /// in practice near 1.
    ///
    /// `null` wherever no such claim is being made, which is every turn that is not a matched
    /// enrolled speaker: mic-track turns, where the speaker is known by construction rather
    /// than inferred, and "Unknown N" turns, where the label names nobody. A number on either
    /// would be a number about a different question.
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

    /// Renders the human-readable `transcript.md` body: one line per turn, nothing else.
    ///
    /// No title, no header, no metadata. Being a pure function of the turns is what lets
    /// `enroll` rewrite the file in place after renaming speakers by calling this again
    /// rather than patching lines.
    ///
    /// Minutes are not wrapped at 60, so a 90-minute meeting renders `[90:05]`. A single
    /// unambiguous format beats an `HH:MM:SS`/`MM:SS` switch that a reader has to detect.
    pub fn render_markdown(&self) -> String {
        let mut out = String::new();
        for turn in &self.turns {
            let total = turn.start.max(0.0);
            let minutes = (total / 60.0).floor() as u64;
            let seconds = (total - (minutes as f64) * 60.0).floor() as u64;
            out.push_str(&format!(
                "**[{:02}:{:02}] {}:** {}\n",
                minutes, seconds, turn.speaker, turn.text
            ));
        }
        out
    }

    /// Writes both transcript files atomically.
    ///
    /// Markdown first, JSON second, and the order is load-bearing: `transcript.json`'s
    /// presence is the "already transcribed" marker, so a crash between the two writes
    /// leaves a session that still re-transcribes rather than one that is marked done but
    /// missing its readable rendering.
    pub fn write(&self, paths: &SessionPaths) -> Result<()> {
        let md = paths.transcript_md();
        write_atomic(&md, self.render_markdown().as_bytes())?;

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
            speaker_id_confidence: None,
        }
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

    #[test]
    fn markdown_renders_one_line_per_turn() {
        let transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                mic_turn(12.34, 14.0, "first"),
                mic_turn(5405.0, 5410.0, "much later"),
            ],
        );
        assert_eq!(
            transcript.render_markdown(),
            "**[00:12] You:** first\n**[90:05] You:** much later\n"
        );
    }

    #[test]
    fn markdown_of_no_turns_is_empty() {
        let transcript = Transcript::new(SessionId::parse("20260809-052600").unwrap(), Vec::new());
        assert_eq!(transcript.render_markdown(), "");
    }
}
