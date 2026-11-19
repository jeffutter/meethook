//! The model catalog: the checkpoints this tool downloads, each pinned by hash and size.

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
/// FireRedVAD's streaming model with an explicit cache input (`fireredvad_stream_vad_with_cache`),
/// 2.3 MB, Apache-2.0. Graph contract, asserted in the `vad` module's tests:
/// inputs `feat` f32 `[1, T, 80]` and `caches_in` f32 `[8, 1, 128, 19]`; outputs the speech
/// probability per frame `[1, T, 1]` and the updated cache `[8, 1, 128, >= 19]`. See
/// [`crate::FireRedVad`] for why this detector and why this variant.
///
/// The weights are not on Hugging Face's CDN but in the project's GitHub repository, so the URL
/// pins a commit and the hash was computed from the file at that commit. The input
/// normalisation that must accompany these exact weights (`cmvn.ark`, from the same commit) is
/// embedded in the crate rather than listed here: see `assets/firered-vad-cmvn.ark`. Changing
/// the commit means changing both.
pub const FIRERED_VAD_MODEL: ModelSpec = ModelSpec {
    file_name: "fireredvad-stream-vad-with-cache.onnx",
    url: "https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/\
          c30ec49e8cc69642b0ee65362eba11b9d11c6e54/pretrained_models/onnx_models/\
          fireredvad_stream_vad_with_cache.onnx",
    sha256: "b3c97836130dc34fc32d56fab551e88cf9454511de2b3c250a5e6578dee74b93",
    size_bytes: 2_306_042,
};

/// Directory under the models directory that holds the Nemotron pair. Unique to this model.
#[cfg(test)]
const NEMOTRON_DIR: &str = "nemotron-3-diarization-fp16";

/// The Nemotron-3-Diarization graph (fp16), half of a two-file model.
///
/// License: OpenMDW-1.1 (the model card says commercial use is allowed).
///
/// The graph is 352 kB and refers to its weights by the literal external-data name
/// `model_fp16.onnx_data` (seen with `strings` on the graph), so the data file cannot be
/// renamed to something collision-proof in the flat models directory. Instead both files keep
/// their original names inside the `nemotron-3-diarization-fp16` directory, a directory name no other model uses.
/// Install both together via [`NEMOTRON_DIARIZATION_MODELS`].
///
/// Revision, hash and size come from the git-LFS pointer at the `raw/` path, as for
/// [`WHISPER_MODEL`]. A q4f16 variant (73 MB) exists but was not evaluated for accuracy.
pub const NEMOTRON_DIARIZATION_GRAPH: ModelSpec = ModelSpec {
    file_name: "nemotron-3-diarization-fp16/model_fp16.onnx",
    url: "https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/resolve/\
          353b6f8ad2cac3580e982d7fbdf0a010786b0406/onnx/model_fp16.onnx",
    sha256: "286f56d64d80755eb0c84f9158c3620c9eccccb84ec53b48942c3795cf5a0c7b",
    size_bytes: 351_782,
};

/// The external weights for [`NEMOTRON_DIARIZATION_GRAPH`] (199 MB). Same revision and
/// license; must sit beside the graph under this exact name.
pub const NEMOTRON_DIARIZATION_DATA: ModelSpec = ModelSpec {
    file_name: "nemotron-3-diarization-fp16/model_fp16.onnx_data",
    url: "https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX/resolve/\
          353b6f8ad2cac3580e982d7fbdf0a010786b0406/onnx/model_fp16.onnx_data",
    sha256: "affecf841c462d78c86b56a6eee6216b119809285bcbdb4d5c4d541810e9d317",
    size_bytes: 199_287_808,
};

/// The voice isolator that strips room noise from the microphone track.
///
/// Inworld's voice isolator (Apache-2.0, a 64M-parameter denoiser) in the community ONNX
/// conversion `mrfakename/voice-isolator-onnx`. Graph contract, from that model card and
/// asserted in the `voice_isolator` module's ignored smoke test: input `spec` f32
/// `[1, frames, 1025, 2]`, the real/imaginary STFT of 48 kHz mono audio (n_fft 2048, hop 480,
/// periodic Hann, centred with reflect padding); output `correction` f32 `[1, frames, 1025, 2]`,
/// a complex correction to add back through the inverse STFT. See [`crate::VoiceIsolator`].
///
/// The file name is the repository's `voice_isolator_offline.onnx` with underscores turned to
/// hyphens, to match the other entries in this flat directory. Like [`SEGMENTATION_MODEL`], the
/// URL pins an immutable revision and the hash and size come from the git-LFS pointer served at
/// the `raw/` path.
pub const VOICE_ISOLATOR_MODEL: ModelSpec = ModelSpec {
    file_name: "voice-isolator-offline.onnx",
    url: "https://huggingface.co/mrfakename/voice-isolator-onnx/resolve/\
          f97841f53d911c6aa6f1c7778e2d1d214c741861/voice_isolator_offline.onnx",
    sha256: "11bd40a8089098d847a7c26aaf786b1af0383aa23ebe8a51c9d6877a5aa4ffd0",
    size_bytes: 260_588_988,
};

/// Both Nemotron files; install every one before opening the graph.
pub const NEMOTRON_DIARIZATION_MODELS: [ModelSpec; 2] =
    [NEMOTRON_DIARIZATION_GRAPH, NEMOTRON_DIARIZATION_DATA];

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const ALL: [&ModelSpec; 7] = [
        &WHISPER_MODEL,
        &SEGMENTATION_MODEL,
        &EMBEDDING_MODEL,
        &FIRERED_VAD_MODEL,
        &NEMOTRON_DIARIZATION_GRAPH,
        &NEMOTRON_DIARIZATION_DATA,
        &VOICE_ISOLATOR_MODEL,
    ];

    #[test]
    fn the_nemotron_pair_shares_one_directory_with_the_names_the_graph_expects() {
        let g = Path::new(NEMOTRON_DIARIZATION_GRAPH.file_name);
        let d = Path::new(NEMOTRON_DIARIZATION_DATA.file_name);
        assert_eq!(g.parent(), d.parent());
        assert_eq!(g.parent(), Some(Path::new(NEMOTRON_DIR)));
        assert_eq!(d.file_name().unwrap(), "model_fp16.onnx_data");
        assert_eq!(g.file_name().unwrap(), "model_fp16.onnx");
    }

    #[test]
    fn no_registry_path_collides_with_another() {
        for (i, a) in ALL.iter().enumerate() {
            for b in &ALL[i + 1..] {
                assert_ne!(a.file_name, b.file_name);
            }
        }
        for spec in ALL {
            if spec.file_name != NEMOTRON_DIARIZATION_GRAPH.file_name
                && spec.file_name != NEMOTRON_DIARIZATION_DATA.file_name
            {
                assert_ne!(spec.file_name, NEMOTRON_DIR);
            }
        }
    }

    #[test]
    fn every_url_pins_a_forty_hex_revision() {
        for spec in ALL {
            let has_rev = spec
                .url
                .split('/')
                .any(|seg| seg.len() == 40 && seg.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(has_rev, "{} is not pinned to a revision", spec.url);
            assert!(!spec.url.contains("/main/"), "{}", spec.url);
            assert_eq!(spec.sha256.len(), 64, "{}", spec.file_name);
        }
    }
}
