//! Host-clock plumbing: raw mach ticks, and the timebase needed to interpret them.
//!
//! Nothing here computes an offset between the two tracks. The recorder's whole
//! contribution to synchronization is recording, exactly and losslessly, *when* each track
//! delivered its first sample; turning two tick counts into a sample offset is
//! `transcribe`'s job and happens once, against both WAV headers.

use std::sync::OnceLock;

use mach2::mach_time::{mach_timebase_info, mach_timebase_info_data_t};
use meethook_session::TrackSync;
use objc2_core_media::{CMClock, CMTime};

/// `mach_timebase_info` as `(numer, denom)`; nanoseconds = `ticks * numer / denom`.
///
/// Read once per process and cached: the ratio is a property of the machine, not of a
/// moment. On Apple Silicon it is 125/3, not the 1/1 that Intel Macs report -- which is
/// precisely why the session contract stores raw ticks alongside this ratio instead of
/// pre-converted nanoseconds.
pub fn timebase() -> (u32, u32) {
    static TIMEBASE: OnceLock<(u32, u32)> = OnceLock::new();

    *TIMEBASE.get_or_init(|| {
        let mut info = mach_timebase_info_data_t { numer: 0, denom: 0 };
        // SAFETY: `info` is a live, uniquely borrowed `mach_timebase_info_data_t`.
        let status = unsafe { mach_timebase_info(&mut info) };

        // A failure here means the kernel could not report its own clock ratio. There is no
        // sensible fallback -- guessing 1/1 would silently corrupt every timestamp this
        // crate writes on Apple Silicon -- so this is a genuine "crash the process" case.
        assert_eq!(status, 0, "mach_timebase_info failed with status {status}");
        assert!(
            info.numer != 0 && info.denom != 0,
            "mach_timebase_info reported a degenerate ratio {}/{}",
            info.numer,
            info.denom
        );

        (info.numer, info.denom)
    })
}

/// Converts a Core Media host time into the units of `mach_absolute_time`.
///
/// `SCStream` hands out presentation timestamps on the host time clock, but with a
/// timescale of Core Media's choosing. This is a scale conversion, not a clock conversion,
/// and it is more accurate than doing the arithmetic by hand because the host clock's
/// timescale need not be an integer.
pub fn cmtime_to_host_ticks(host_time: CMTime) -> u64 {
    // SAFETY: `CMClockConvertHostTimeToSystemUnits` is a pure conversion over a by-value
    // CMTime; it has no pointer arguments and no preconditions beyond a host-clock input.
    unsafe { CMClock::convert_host_time_to_system_units(host_time) }
}

/// Pairs a track's first-buffer tick count with the timebase needed to read it.
///
/// Both tracks necessarily get the same ratio -- same process, same machine. The contract
/// keeps the field per-track anyway, and this slice does not change the contract.
pub fn track_sync(host_ticks: u64) -> TrackSync {
    let (numer, denom) = timebase();
    TrackSync {
        host_ticks,
        timebase_numer: numer,
        timebase_denom: denom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The conversion arithmetic is the part that can be silently wrong: a recorder that
    /// stores ticks as if they were nanoseconds still produces plausible-looking numbers.
    /// Timing a known sleep through the stored ratio is what catches that.
    #[test]
    fn ticks_convert_to_wall_time_through_the_stored_ratio() {
        let (numer, denom) = timebase();
        assert!(numer != 0 && denom != 0);

        // SAFETY: `mach_absolute_time` takes no arguments and has no preconditions.
        let start = unsafe { mach2::mach_time::mach_absolute_time() };
        std::thread::sleep(std::time::Duration::from_millis(50));
        let end = unsafe { mach2::mach_time::mach_absolute_time() };

        let elapsed_nanos = (end - start) as u128 * u128::from(numer) / u128::from(denom);
        let elapsed_millis = elapsed_nanos / 1_000_000;

        // Generous upper bound: this runs on a loaded CI-style machine, and the assertion
        // that matters is the lower one -- an unscaled tick count would read as ~1.2 ms.
        assert!(
            (45..500).contains(&elapsed_millis),
            "50 ms of sleep read back as {elapsed_millis} ms through timebase {numer}/{denom}"
        );
    }

    /// Guards against anyone "simplifying" the tick plumbing by routing it through `f64`,
    /// which silently loses the low bits of a mach tick count within hours of uptime.
    #[test]
    fn large_tick_counts_survive_a_json_round_trip() {
        let sync = track_sync(u64::MAX - 12_345);
        let json = serde_json::to_string(&sync).unwrap();
        let parsed: TrackSync = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, sync);
        assert_eq!(parsed.host_ticks, u64::MAX - 12_345);
    }
}
