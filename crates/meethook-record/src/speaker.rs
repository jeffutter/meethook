//! System audio capture via ScreenCaptureKit.
//!
//! The stream is configured system-wide -- a display filter excluding nothing -- rather than
//! scoped to particular applications, because a meeting's audio can come from a browser tab,
//! a native client, a notification, or a screen-shared video, and enumerating those in
//! advance is guesswork the user would pay for with a silent track.
//!
//! ScreenCaptureKit is fundamentally a *screen* capture API, so the stream still attaches to
//! a display even though no video output is ever added. The configuration shrinks that to a
//! 2x2 frame at one frame per second so the unused video path costs as close to nothing as
//! the API allows.

use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::Mutex;
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{AllocAnyThread, DefinedClass, define_class, msg_send};
use objc2_core_audio_types::AudioBufferList;
use objc2_core_foundation::CFRetained;
use objc2_core_media::{CMBlockBuffer, CMSampleBuffer, CMTime};
use objc2_foundation::{NSArray, NSError, NSObject, NSObjectProtocol};
use objc2_screen_capture_kit::{
    SCContentFilter, SCDisplay, SCShareableContent, SCStream, SCStreamConfiguration,
    SCStreamDelegate, SCStreamOutput, SCStreamOutputType,
};

use crate::clock;
use crate::track::{TrackSink, TrackSummary, TrackWriter};
use crate::{Error, Result};

/// ScreenCaptureKit's own native rate. Its only other options (8/16/24 kHz) are all lower,
/// so asking for anything else would be downsampling; asking for this is not.
const SPEAKER_SAMPLE_RATE: u32 = 48_000;

/// Every wait on a ScreenCaptureKit completion handler is bounded. A recorder that hangs is
/// worse than one that fails, because the user finds out about the hang after the meeting.
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(10);

/// Resolves the display whose system audio will be captured.
///
/// This runs before the session directory is created, so a machine with no usable display
/// fails without leaving an orphaned directory behind.
pub fn default_display() -> Result<Retained<SCDisplay>> {
    let (tx, rx) = mpsc::channel::<std::result::Result<Retained<SCShareableContent>, String>>();

    let handler = RcBlock::new(
        move |content: *mut SCShareableContent, error: *mut NSError| {
            let result = match NonNull::new(content) {
                // SAFETY: ScreenCaptureKit hands back an autoreleased, live object here.
                Some(content) => Ok(unsafe { Retained::retain(content.as_ptr()) }
                    .expect("a non-null SCShareableContent can always be retained")),
                None => Err(error_message(error)
                    .unwrap_or_else(|| "no shareable content was returned".to_owned())),
            };
            let _ = tx.send(result);
        },
    );

    // SAFETY: the handler is retained by the framework for the duration of the async call,
    // and `RcBlock` keeps it alive here until the wait below returns.
    crate::exception::catching("SCShareableContent.getShareableContent", || unsafe {
        SCShareableContent::getShareableContentWithCompletionHandler(&handler)
    })?;

    let content = wait(&rx)?.map_err(Error::ScreenCaptureKit)?;

    // SAFETY: `displays` is a plain property read on a live object.
    let displays = crate::exception::catching("SCShareableContent.displays", || unsafe {
        content.displays()
    })?;
    displays.firstObject().ok_or(Error::NoDisplay)
}

/// A live system-audio capture.
pub struct SpeakerCapture {
    stream: Retained<SCStream>,
    output: Retained<AudioOutput>,
    delegate: Retained<StreamDelegate>,
    /// The stream retains this internally, but keeping it here documents that the queue must
    /// outlive the stream and stops it being dropped early by a future refactor.
    _queue: DispatchRetained<DispatchQueue>,
    writer: TrackWriter,
}

impl SpeakerCapture {
    pub fn start(display: &SCDisplay, path: &Path) -> Result<SpeakerCapture> {
        let writer = TrackWriter::create(path, SPEAKER_SAMPLE_RATE)?;

        // The whole construction sequence under one `catching`, because a display that went
        // away between `default_display` resolving it and this line makes the filter
        // initializer raise -- and an uncaught raise aborts the process rather than failing
        // this session. The grouping matches the failure: every call in here is part of "build
        // a stream for this display", and there is nothing a caller could do differently
        // knowing which of them raised.
        let (stream, delegate, output, queue) =
            crate::exception::catching("SCStream.init", || {
                // Excluding an empty window list is what makes this system-wide. Any `including*`
                // initializer would scope capture to chosen windows or applications instead.
                let excluded: Retained<NSArray<_>> = NSArray::new();
                // SAFETY: both arguments are live objects of the required types.
                let filter = unsafe {
                    SCContentFilter::initWithDisplay_excludingWindows(
                        SCContentFilter::alloc(),
                        display,
                        &excluded,
                    )
                };

                // SAFETY: `new` has no preconditions beyond being called on a class that exists.
                let config = unsafe { SCStreamConfiguration::new() };
                // SAFETY: plain property writes on a freshly constructed configuration.
                unsafe {
                    config.setCapturesAudio(true);
                    config.setSampleRate(SPEAKER_SAMPLE_RATE as isize);
                    // Let ScreenCaptureKit downmix to mono. Doing it here would mean owning a
                    // channel-mixing policy the OS already implements.
                    config.setChannelCount(1);
                    // meethook renders no audio today, but `enroll` will play back clips, and a
                    // recorder that captures its own playback would poison the reference signal.
                    config.setExcludesCurrentProcessAudio(true);

                    // NOTE: `setCaptureMicrophone(true)` is deliberately never called. Combining
                    // ScreenCaptureKit's unified microphone output with recording is a known source
                    // of corrupted output, and a merged stream would also destroy the independent
                    // speaker reference that echo cancellation needs later. The microphone is
                    // captured by a separate AVAudioEngine tap; see `mic.rs`.

                    // No video output is ever added, but the stream still captures a display, so
                    // shrink that to nothing rather than paying for full-resolution frames.
                    config.setWidth(2);
                    config.setHeight(2);
                    config.setMinimumFrameInterval(CMTime {
                        value: 1,
                        timescale: 1,
                        flags: objc2_core_media::CMTimeFlags::Valid,
                        epoch: 0,
                    });
                    config.setQueueDepth(3);
                }

                let delegate = StreamDelegate::new();
                let output = AudioOutput::new(writer.sink());
                let queue = DispatchQueue::new("com.meethook.speaker", DispatchQueueAttr::SERIAL);

                // SAFETY: filter, config and delegate are live and of the required types.
                let stream: Retained<SCStream> = unsafe {
                    SCStream::initWithFilter_configuration_delegate(
                        SCStream::alloc(),
                        &filter,
                        &config,
                        Some(ProtocolObject::from_ref(&*delegate)),
                    )
                };

                (stream, delegate, output, queue)
            })?;

        // SAFETY: `output` conforms to SCStreamOutput, and `queue` is a live serial queue
        // owned by the returned struct.
        crate::exception::catching("SCStream.addStreamOutput", || unsafe {
            stream.addStreamOutput_type_sampleHandlerQueue_error(
                ProtocolObject::from_ref(&*output),
                SCStreamOutputType::Audio,
                Some(&queue),
            )
        })?
        .map_err(|e| Error::ScreenCaptureKit(describe(&e)))?;

        let (tx, rx) = mpsc::channel::<Option<String>>();
        let handler = RcBlock::new(move |error: *mut NSError| {
            let _ = tx.send(error_message(error));
        });
        // SAFETY: the framework retains the completion handler; the wait below keeps the
        // `RcBlock` alive until it has fired.
        crate::exception::catching("SCStream.startCapture", || unsafe {
            stream.startCaptureWithCompletionHandler(Some(&handler))
        })?;

        match wait(&rx) {
            Ok(None) => {}
            Ok(Some(message)) => return Err(Error::ScreenCaptureKit(message)),
            Err(e) => return Err(e),
        }

        Ok(SpeakerCapture {
            stream,
            output,
            delegate,
            _queue: queue,
            writer,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        SPEAKER_SAMPLE_RATE
    }

    /// Stops the stream and finalizes the WAV.
    ///
    /// A stream that died mid-meeting is reported here rather than swallowed: without this,
    /// a dropped stream would produce a short file and a cheerful success message.
    pub fn stop(self) -> Result<TrackSummary> {
        let SpeakerCapture {
            stream,
            output,
            delegate,
            _queue,
            writer,
        } = self;

        let (tx, rx) = mpsc::channel::<Option<String>>();
        let handler = RcBlock::new(move |error: *mut NSError| {
            let _ = tx.send(error_message(error));
        });
        // Both teardown calls are caught and their raises discarded, the same reading
        // `MicCapture::stop` takes: this is the finalize path, the audio is already on disk,
        // and a stream whose display has gone is precisely what raises here. `stop_result`
        // still carries a *returned* error, which is a stream that died mid-meeting and is
        // worth reporting; a raise on the way down is not.
        // SAFETY: as for `startCaptureWithCompletionHandler` above.
        let stop_result = match crate::exception::catching("SCStream.stopCapture", || unsafe {
            stream.stopCaptureWithCompletionHandler(Some(&handler))
        }) {
            Ok(()) => wait(&rx),
            // Nothing will complete the handler, so waiting out the full timeout here would
            // only delay the finalize by ten seconds to learn nothing.
            Err(_) => Ok(None),
        };

        // Detach the output before dropping anything, so no queued callback can fire into a
        // writer that is about to be finalized.
        // SAFETY: `output` is the same object that was added, with the same type.
        let _ = crate::exception::catching("SCStream.removeStreamOutput", || unsafe {
            stream.removeStreamOutput_type_error(
                ProtocolObject::from_ref(&*output),
                SCStreamOutputType::Audio,
            )
        });

        // Finalize before reporting any error: a half-written WAV helps nobody.
        let summary = writer.finish();

        if let Some(message) = delegate.take_error() {
            return Err(Error::ScreenCaptureKit(message));
        }
        match stop_result {
            Ok(None) => {}
            Ok(Some(message)) => return Err(Error::ScreenCaptureKit(message)),
            Err(e) => return Err(e),
        }
        summary
    }
}

/// Blocks on a completion-handler channel with a bounded wait.
fn wait<T>(rx: &Receiver<T>) -> Result<T> {
    rx.recv_timeout(CALLBACK_TIMEOUT)
        .map_err(|_| Error::ScreenCaptureKitTimeout(CALLBACK_TIMEOUT))
}

fn describe(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

fn error_message(error: *mut NSError) -> Option<String> {
    // SAFETY: ScreenCaptureKit passes either null or a live autoreleased NSError.
    NonNull::new(error).map(|e| describe(unsafe { e.as_ref() }))
}

/// Records the reason a stream stopped on its own.
struct DelegateIvars {
    error: Mutex<Option<String>>,
}

define_class!(
    // SAFETY:
    // - NSObject has no subclassing requirements.
    // - `StreamDelegate` does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[ivars = DelegateIvars]
    struct StreamDelegate;

    unsafe impl NSObjectProtocol for StreamDelegate {}

    unsafe impl SCStreamDelegate for StreamDelegate {
        #[unsafe(method(stream:didStopWithError:))]
        fn stream_did_stop_with_error(&self, _stream: &SCStream, error: &NSError) {
            *self
                .ivars()
                .error
                .lock()
                .expect("stream error mutex poisoned") = Some(describe(error));
        }
    }
);

impl StreamDelegate {
    fn new() -> Retained<Self> {
        let this = Self::alloc().set_ivars(DelegateIvars {
            error: Mutex::new(None),
        });
        unsafe { msg_send![super(this), init] }
    }

    fn take_error(&self) -> Option<String> {
        self.ivars()
            .error
            .lock()
            .expect("stream error mutex poisoned")
            .take()
    }
}

struct AudioOutputIvars {
    sink: TrackSink,
}

define_class!(
    // SAFETY:
    // - NSObject has no subclassing requirements.
    // - `AudioOutput` does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[ivars = AudioOutputIvars]
    struct AudioOutput;

    unsafe impl NSObjectProtocol for AudioOutput {}

    unsafe impl SCStreamOutput for AudioOutput {
        #[unsafe(method(stream:didOutputSampleBuffer:ofType:))]
        fn stream_did_output_sample_buffer_of_type(
            &self,
            _stream: &SCStream,
            sample_buffer: &CMSampleBuffer,
            r#type: SCStreamOutputType,
        ) {
            if r#type != SCStreamOutputType::Audio {
                return;
            }
            // SAFETY: ScreenCaptureKit delivers a live sample buffer for the duration of the
            // callback.
            unsafe { self.handle_audio(sample_buffer) };
        }
    }
);

impl AudioOutput {
    fn new(sink: TrackSink) -> Retained<Self> {
        let this = Self::alloc().set_ivars(AudioOutputIvars { sink });
        unsafe { msg_send![super(this), init] }
    }

    /// Copies one audio sample buffer into the track, timestamping the first one.
    ///
    /// # Safety
    ///
    /// `sample_buffer` must be a live audio `CMSampleBuffer`.
    unsafe fn handle_audio(&self, sample_buffer: &CMSampleBuffer) {
        // Read first, before any work, so it measures when this callback arrived rather
        // than how long it took.
        let delivered_ticks = clock::now_ticks();

        // SAFETY: plain accessors on a live sample buffer.
        let frames = unsafe { sample_buffer.num_samples() } as usize;
        if frames == 0 {
            return;
        }
        // This timestamp is internally consistent -- it stays linear in this stream's own
        // sample count to under 0.1 ms over 35 s -- but it is NOT the moment the sound
        // reached the air, and the gap is not recoverable from any API. Measured by click
        // test across built-in, USB and Bluetooth outputs:
        //
        // - It runs ~16 ms ahead of the mic's timeline by a fixed amount that originates
        //   inside ScreenCaptureKit. Driving the display between 72 Hz and 30 Hz moved it
        //   0.63 ms against a predicted 19.44 ms, so it is not the video frame clock.
        // - It excludes output latency entirely: swapping the output device moved the click
        //   residual by 423 ms while this timestamp's behaviour was unchanged.
        // - CoreAudio cannot supply that missing term. A probe reading device latency,
        //   stream latency and safety offset reported 186.688 ms for a Bluetooth path
        //   measured at ~426 ms. `AVAudioIONode.presentationLatency` is no better: it tracks
        //   the OUTPUT device (1.542 ms on USB, 203.197 ms on Bluetooth, same USB mic).
        //
        // So do not "correct" this with a constant. Doing so would fix ~16 ms of an error
        // that reached 410 ms while making `session.json` look authoritative. Acoustic
        // alignment has to be measured from the signals, which the AEC stage does anyway.
        // The CoreAudio probe was written, used to establish the above, and then removed.
        let host_ticks =
            clock::cmtime_to_host_ticks(unsafe { sample_buffer.presentation_time_stamp() });

        // One buffer is all a mono stream produces, so the list can live on the stack.
        let mut list = AudioBufferList {
            mNumberBuffers: 0,
            mBuffers: [objc2_core_audio_types::AudioBuffer {
                mNumberChannels: 0,
                mDataByteSize: 0,
                mData: ptr::null_mut(),
            }],
        };
        let mut block_buffer: *mut CMBlockBuffer = ptr::null_mut();

        // SAFETY: `list` is a valid, correctly sized AudioBufferList, and `block_buffer` is a
        // valid out-pointer.
        let status = unsafe {
            sample_buffer.audio_buffer_list_with_retained_block_buffer(
                ptr::null_mut(),
                &mut list,
                size_of::<AudioBufferList>(),
                None,
                None,
                0,
                &mut block_buffer,
            )
        };

        // The block buffer comes back +1 retained and owns the memory `list` points into.
        // Wrapping it here releases it on every path out of this function; leaking it
        // instead would grow without bound over an hour-long meeting.
        // SAFETY: a non-null out-pointer here is a live, +1-retained CMBlockBuffer.
        let _block_buffer = NonNull::new(block_buffer).map(|b| unsafe { CFRetained::from_raw(b) });

        if status != 0 || list.mNumberBuffers == 0 {
            return;
        }

        let buffer = list.mBuffers[0];
        if buffer.mData.is_null() {
            return;
        }

        // Trust the byte count over the frame count: a mismatch means the buffer is not the
        // mono float32 layout that was configured, and reading past it would be UB.
        let available = buffer.mDataByteSize as usize / size_of::<f32>();
        let channels = buffer.mNumberChannels.max(1) as usize;
        let usable = frames.min(available / channels);
        if usable == 0 {
            return;
        }

        let data = buffer.mData.cast::<f32>().cast_const();
        let channel_ptrs = [data];
        // SAFETY: `data` is valid for `available` f32s, and `usable * channels <= available`.
        let mono = unsafe { crate::track::channel_zero(channel_ptrs.as_ptr(), usable, channels) };
        self.ivars().sink.push(host_ticks, delivered_ticks, mono);
    }
}
