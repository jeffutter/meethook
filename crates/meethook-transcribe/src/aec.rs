//! Acoustic echo cancellation over the recorded mic track.
//!
//! The mic track of a meeting contains the other participants twice: once through the
//! network as the speaker track, and once acoustically as whatever the laptop's speakers
//! played back into its own microphone. Whisper transcribes that leakage as if it were the
//! local user, so it has to be removed before the mic track is transcribed.
//!
//! The remover is WebRTC's AEC3, reached through the tonarino `webrtc-audio-processing`
//! crate. It is linked dynamically against the flake's `webrtc-audio-processing` package
//! rather than built here: the crate's `bundled` feature compiles the vendored C++ itself
//! and does not cross-compile on Apple (tonarino/webrtc-audio-processing#102).
//!
//! Only the linkage is settled here. Delay estimation, framing the two tracks against each
//! other, and writing `mic.cleaned.wav` belong to the pre-pass that will grow into this
//! module.

#[cfg(test)]
mod tests {
    use webrtc_audio_processing::{Config, Processor, config::EchoCanceller};

    /// AEC3's frame size is fixed at 10 ms, and feeding it any other length is a panic
    /// rather than an error, so the 16 kHz frame size is pinned here rather than left to be
    /// discovered by a crash in the pre-pass.
    const SAMPLES_PER_FRAME: usize = 160;

    /// Proves the C++ library is linked and reachable: a processor is constructed, a
    /// matched render/capture pair goes through it, and the capture frame comes back
    /// changed. It is a link and call-path check, not a measurement of cancellation
    /// quality -- one 10 ms frame is far too little for AEC3's filter to converge.
    #[test]
    fn one_frame_pair_goes_through_a_16_khz_echo_canceller() {
        let processor = Processor::new(16_000).expect("AEC3 processor at 16 kHz");
        assert_eq!(
            processor.num_samples_per_frame(),
            SAMPLES_PER_FRAME,
            "10 ms at 16 kHz"
        );

        processor.set_config(Config {
            echo_canceller: Some(EchoCanceller::default()),
            ..Default::default()
        });

        // Mono, which is what both recorded tracks are. The capture frame is the render
        // frame attenuated -- the echo -- plus a second tone standing in for the local
        // talker, so there is something for AEC3 to keep as well as something to remove.
        let render: Vec<f32> = (0..SAMPLES_PER_FRAME)
            .map(|i| (i as f32 / 40.0).cos() * 0.4)
            .collect();
        let capture: Vec<f32> = (0..SAMPLES_PER_FRAME)
            .map(|i| (i as f32 / 20.0).sin() * 0.4 + render[i] * 0.2)
            .collect();

        let mut render_out = vec![render.clone()];
        processor
            .process_render_frame(&mut render_out)
            .expect("render frame accepted");
        assert_eq!(
            render_out[0], render,
            "the playback frame is analyzed, not altered"
        );

        let mut capture_out = vec![capture.clone()];
        processor
            .process_capture_frame(&mut capture_out)
            .expect("capture frame accepted");
        assert_ne!(
            capture_out[0], capture,
            "the mic frame should come back altered by echo cancellation"
        );
    }
}
