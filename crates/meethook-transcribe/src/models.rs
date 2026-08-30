//! The model catalog: the four checkpoints this tool downloads, each pinned by hash and size.

use meethook_models::ModelSpec;

/// The Whisper checkpoint this tool transcribes with.
///
/// large-v3-turbo: close to large-v3's accuracy at several times the speed, which is the
/// right trade for turning a finished meeting around rather than for streaming.
///
/// The URL pins an immutable revision, not `main`. `main` is a moving pointer, and a
/// republished checkpoint would turn a working install into a hash mismatch nobody asked
/// for. Both values below come from the git-LFS pointer Hugging Face serves at the `raw/`
/// path (`curl .../raw/<rev>/<file>` prints `oid sha256:...` and `size`), which is how to
/// get them again without downloading 1.6 GB.
///
/// If download size or memory ever becomes a problem, the quantized build of the same model
/// is `ggml-large-v3-turbo-q5_0.bin`, sha256
/// `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`, 574041195 bytes.
pub const WHISPER_MODEL: ModelSpec = ModelSpec {
    file_name: "ggml-large-v3-turbo.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/\
          5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin",
    sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    size_bytes: 1_624_555_275,
};

/// The speaker-segmentation graph diarization runs over the speaker track.
///
/// pyannote segmentation 3.0, exported to ONNX. It consumes a fixed-length window of raw
/// audio and emits, per frame, a distribution over the *powerset* of up to three concurrent
/// speakers -- which is why the output's last dimension is 7 (silence, three singles, three
/// pairs) rather than a speaker count.
///
/// Graph contract, asserted in the `onnx` module's smoke test:
/// input `input_values` f32 `[batch_size, num_channels, num_samples]`;
/// output `logits` f32 `[batch_size, num_frames, 7]`.
///
/// The file name is the repository's, not the repository's `model.onnx`: the models
/// directory is flat and shared, so a generic name would collide with the next ONNX model
/// added.
///
/// Like [`WHISPER_MODEL`], the URL pins an immutable revision and the hash and size come
/// from the git-LFS pointer Hugging Face serves at the `raw/` path, so bumping the revision
/// does not require downloading the weights to re-derive them.
pub const SEGMENTATION_MODEL: ModelSpec = ModelSpec {
    file_name: "pyannote-segmentation-3.0.onnx",
    url: "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/\
          733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx",
    sha256: "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25",
    size_bytes: 5_986_908,
};

/// The speaker-embedding graph that turns a segment of speech into a voice fingerprint.
///
/// WeSpeaker's VoxCeleb ResNet34-LM. It takes fbank features rather than raw audio -- 80
/// mel bins per frame -- and returns one 256-dimensional embedding per utterance, which is
/// what clustering and enrollment compare.
///
/// Graph contract, asserted in the `onnx` module's smoke test:
/// input `feats` f32 `[B, T, 80]`; output `embs` f32 `[B, 256]`.
pub const EMBEDDING_MODEL: ModelSpec = ModelSpec {
    file_name: "wespeaker-voxceleb-resnet34-LM.onnx",
    url: "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/\
          f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx",
    sha256: "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
    size_bytes: 26_530_309,
};

/// The voice-activity detector that says which stretches of a track hold speech.
///
/// Silero v5.1.2, in whisper.cpp's own ggml format. 885 KB, and no new crate dependency at
/// all: whisper.cpp 1.8.3 is already vendored by `whisper-rs-sys`, and `whisper-rs` 0.16
/// exposes a safe wrapper around its standalone VAD. See [`crate::SileroVad`] for why a separate
/// detector rather than the pyannote graph already installed.
///
/// v5.1.2 rather than the v6.2.0 that also sits in that repository: v5.1.2 is what
/// whisper.cpp's own documentation and default tooling use, so it is the version its
/// thresholds and post-processing were tuned against.
///
/// Like [`WHISPER_MODEL`], the URL pins an immutable revision rather than `main`, and the hash
/// and size come from the git-LFS pointer Hugging Face serves at the `raw/` path -- so bumping
/// the revision does not require downloading the weights to re-derive them.
pub const SILERO_VAD_MODEL: ModelSpec = ModelSpec {
    file_name: "ggml-silero-v5.1.2.bin",
    url: "https://huggingface.co/ggml-org/whisper-vad/resolve/\
          9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v5.1.2.bin",
    sha256: "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    size_bytes: 885_098,
};
