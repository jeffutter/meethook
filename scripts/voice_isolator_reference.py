#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["inworld-voice-isolation==1.0.0", "soundfile"]
# ///
"""Writes the parity fixtures for `voice_isolator.rs`'s `matches_the_official_python_package`.

    uv run scripts/voice_isolator_reference.py

Downloads the Apache-2.0 sample `demo/examples/sample_14.wav` from
https://github.com/inworld-ai/voice-isolator (pinned to a commit below), converts it to the
48 kHz mono the Rust engine takes, and runs it through the official `inworld-voice-isolation`
package pinned at 1.0.0 (model `voice-isolator-v1`, on the CPU for determinism). Writes, into
`crates/meethook-transcribe/testdata/voice-isolator/`:

- `sample_14.wav`           the 48 kHz mono input, 16-bit PCM;
- `sample_14.reference.wav` the package's output for exactly those samples, 32-bit float.

The input is re-read from the 16-bit file before isolating, so the package and the Rust engine
see the same samples. The first run downloads the model weights from Hugging Face. No meeting
audio is involved.
"""

import pathlib
import urllib.request

import numpy as np
import soundfile as sf
from inworld import voice_isolation

SAMPLE_URL = (
    "https://raw.githubusercontent.com/inworld-ai/voice-isolator/"
    "c73b3fab81a870410cbac50d2f05f25edc4d10aa/demo/examples/sample_14.wav"
)
OUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "crates/meethook-transcribe/testdata/voice-isolator"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "sample_14.source.wav"
    urllib.request.urlretrieve(SAMPLE_URL, source)
    wav, rate = voice_isolation.load_audio(str(source))
    # `load_audio` already mixes to mono; this only guards a future change to that.
    mono = voice_isolation.resample(wav.reshape(-1, wav.shape[-1]).mean(dim=0), rate, voice_isolation.SR)
    source.unlink()

    sf.write(OUT / "sample_14.wav", mono.numpy(), voice_isolation.SR, subtype="PCM_16")
    audio, rate = sf.read(OUT / "sample_14.wav", dtype="float32")
    assert rate == voice_isolation.SR and audio.ndim == 1

    isolator = voice_isolation.VoiceIsolator("voice-isolator-v1", device="cpu")
    voice = np.asarray(isolator.isolate(audio, rate), dtype=np.float32)
    assert voice.shape == audio.shape
    sf.write(OUT / "sample_14.reference.wav", voice, rate, subtype="FLOAT")
    print(f"wrote {OUT}: {len(audio) / rate:.2f} s")


if __name__ == "__main__":
    main()
