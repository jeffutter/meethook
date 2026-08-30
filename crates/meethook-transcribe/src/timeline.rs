//! Where each track starts on the session timeline: the two tracks' first samples as offsets
//! from session start, taken from the recorded host ticks in `session.json`.

use crate::{Error, Result};
use meethook_session::SessionMetadata;

/// Seconds from session start to the microphone track's first sample.
///
/// Public alongside [`speaker_offset_seconds`] so that a diagnostic can put the two tracks on
/// the timeline the transcript uses without re-deriving the tick arithmetic from
/// `session.json` -- see `examples/session-mixdown.rs`.
///
/// Session start is the earlier of the two tracks' first samples, not `session.json`'s
/// `start_time`: that field is a wall-clock instant captured when the directory was created,
/// with no recorded pairing to mach tick space, so it cannot be compared to either track's
/// `host_ticks`. Using the earliest track instead keeps every turn non-negative once
/// speaker-track turns join the same timeline.
pub fn mic_offset_seconds(metadata: &SessionMetadata) -> Result<f64> {
    Ok(mic_minus_speaker_seconds(metadata)?.max(0.0))
}

/// Seconds from session start to the speaker track's first sample: the mirror of
/// [`mic_offset_seconds`], so exactly one of the two is non-zero for any session.
///
/// Both come from `session.json`'s recorded ticks, and deliberately *not* from
/// [`crate::align::measure_reference_lag`]. That measurement is the acoustic path -- how long after
/// the system rendered a sample the microphone heard it come back out of a speaker in a room
/// -- and it bundles output latency and air propagation, neither of which has anything to do
/// with when the far end actually spoke. Applying it here would shift every participant turn
/// late by up to a few hundred milliseconds. The tick delta is honest to well under the
/// accuracy merge ordering needs.
pub fn speaker_offset_seconds(metadata: &SessionMetadata) -> Result<f64> {
    Ok((-mic_minus_speaker_seconds(metadata)?).max(0.0))
}

/// How much later the microphone track's first sample is than the speaker track's, negative
/// if the microphone started first.
///
/// The conversion is exact -- integer ticks scaled by the machine's rational timebase in
/// `i128`, rounded once at the end. Going through `f64` first would lose the low bits of a
/// mach tick count within a day of uptime.
///
/// This is metadata alignment only, and the sign is the whole reason it exists separately
/// from [`mic_offset_seconds`]: the echo canceller's delay search needs to know which track
/// actually started first, and clamping that to zero would centre the search in the wrong
/// place. Correcting the *acoustic* offset between the two capture APIs is a different
/// problem again, measured from the signals themselves in [`crate::align`].
pub(crate) fn mic_minus_speaker_seconds(metadata: &SessionMetadata) -> Result<f64> {
    let mic = metadata.mic;
    if mic.timebase_numer == 0 || mic.timebase_denom == 0 {
        return Err(Error::DegenerateTimebase {
            session: metadata.session_id.clone(),
            numer: mic.timebase_numer,
            denom: mic.timebase_denom,
        });
    }

    let delta = i128::from(mic.host_ticks) - i128::from(metadata.speaker.host_ticks);
    let nanos = delta * i128::from(mic.timebase_numer) / i128::from(mic.timebase_denom);
    Ok(nanos as f64 / 1e9)
}

#[cfg(test)]
pub(crate) mod tests {
    use jiff::Timestamp;
    use meethook_session::{SessionId, TrackSync};

    use super::*;

    /// Apple Silicon's timebase. 125/3 rather than Intel's 1/1 is exactly the ratio that
    /// makes an unscaled tick count look plausible while being 41x wrong.
    const NUMER: u32 = 125;
    const DENOM: u32 = 3;

    pub(crate) fn metadata(id: &SessionId, mic_ticks: u64, speaker_ticks: u64) -> SessionMetadata {
        let sync = |ticks| TrackSync {
            host_ticks: ticks,
            timebase_numer: NUMER,
            timebase_denom: DENOM,
        };
        SessionMetadata::new(
            id.clone(),
            Timestamp::from_second(1_770_000_000).unwrap(),
            sync(mic_ticks),
            sync(speaker_ticks),
        )
    }

    #[test]
    fn a_mic_track_that_started_later_is_offset_onto_the_session_timeline() {
        // 1_000_000 ticks at 125/3 ns per tick is 41.666... ms.
        let id = SessionId::parse("20260809-052600").unwrap();
        let offset =
            mic_offset_seconds(&metadata(&id, 900_000_001_000_000, 900_000_000_000_000)).unwrap();
        assert!(
            (offset - 0.041_666_666).abs() < 1e-9,
            "offset was {offset} s"
        );
    }

    #[test]
    fn a_mic_track_that_started_first_defines_time_zero() {
        let id = SessionId::parse("20260809-052600").unwrap();
        let offset =
            mic_offset_seconds(&metadata(&id, 900_000_000_000_000, 900_000_005_000_000)).unwrap();
        assert_eq!(offset, 0.0);
    }

    /// The two offsets are mirrors: whichever track started second is the one that gets
    /// pushed down the timeline, and exactly one of them is ever non-zero. A sign error here
    /// would put every participant turn on the wrong side of the meeting.
    #[test]
    fn exactly_one_track_is_offset_and_it_is_the_one_that_started_second() {
        let id = SessionId::parse("20260809-052600").unwrap();
        let base = 900_000_000_000_000u64;
        // 1_000_000 ticks at 125/3 ns per tick is 41.666... ms.
        let expected = 0.041_666_666;

        for (mic_ticks, speaker_ticks, mic_offset, speaker_offset) in [
            (base + 1_000_000, base, expected, 0.0),
            (base, base + 1_000_000, 0.0, expected),
            (base, base, 0.0, 0.0),
        ] {
            let metadata = metadata(&id, mic_ticks, speaker_ticks);
            let mic = mic_offset_seconds(&metadata).unwrap();
            let speaker = speaker_offset_seconds(&metadata).unwrap();
            assert!((mic - mic_offset).abs() < 1e-9, "mic offset was {mic} s");
            assert!(
                (speaker - speaker_offset).abs() < 1e-9,
                "speaker offset was {speaker} s"
            );
            assert!(mic == 0.0 || speaker == 0.0, "both tracks were offset");
        }
    }
}
