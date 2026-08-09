"""Build a `speaker-trials` manifest from LibriSpeech dev-clean.

`speaker-trials` (TASK-014.04.01) scores every cross-session pair in a
`speaker <TAB> session <TAB> wav` manifest. It had no population to point at.
This script produces one: fetch a corpus, decode it to WAV, and emit the
manifest plus the shape numbers that say what the manifest will actually
measure.

Why LibriSpeech dev-clean, and what its session ids honestly mean, is recorded
in TASK-014.04.02's implementation notes rather than repeated here. The two
facts that shape this code:

  * The corpus ships FLAC, and this machine has no `ffmpeg`, no `sox`, no
    `flac` and no `soundfile`. `afconvert` reads FLAC metadata but refuses to
    decode it (see `probe`). So the script builds a throwaway Rust decoder
    under the scratch root -- outside the meethook workspace, because this
    ticket adds no dependency to meethook -- from `claxon` and `hound`.

  * The network here is an allowlist that also goes silent for minutes at a
    time, and `curl --max-time` does not bound a blocked DNS lookup. Every
    fetch is therefore wrapped in an external timeout, and every stage is
    resumable: a run interrupted at 60% costs 40%, not 100%.

Nothing is written inside the meethook repository. The script refuses to run
if asked to.

Usage:
    python3 librispeech_trials_manifest.py [--root DIR] [--probe-only]
                                           [--seconds-per-session N]

Default root: $MEETHOOK_CALIB_ROOT, else $TMPDIR/meethook-calib.
"""

import argparse
import collections
import hashlib
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import wave
from pathlib import Path

# --------------------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------------------

# www.openslr.org and openslr.elda.org both answered during development; us.openslr.org
# did not resolve at all. The ELDA mirror is the fallback rather than a preference.
MIRRORS = (
    "https://www.openslr.org/resources/12",
    "https://openslr.elda.org/resources/12",
)
ARCHIVE = "dev-clean.tar.gz"
ARCHIVE_BYTES = 337926286  # Content-Length, checked before the md5 so a truncation is cheap
CHECKSUMS = "md5sum.txt"  # the corpus publishes its own; no hash is hardcoded here

# Budget, printed before anything is fetched so the outturn can be read against it.
BYTE_BUDGET = ARCHIVE_BYTES + 700 * 1024 * 1024  # archive, plus the WAV it decodes into
TIME_BUDGET_S = 20 * 60

# How much of each session is decoded. `speaker-trials` measures at most 120 s per item
# by default; the headroom covers the case where its own splicing trims differently.
DEFAULT_SECONDS_PER_SESSION = 150.0

# Every network call is wrapped in an external timeout, because `curl --max-time` does
# not interrupt a blocked `getaddrinfo` and a hung probe costs the whole run. The probe
# is generous and retries: the same URL alternated between a 0.5 s answer and total
# silence within one session, so a single tight attempt reports weather, not reachability.
# Worst case for the whole probe stage is two mirrors x PROBE_TIMEOUT_S.
PROBE_TIMEOUT_S = 45
FETCH_TIMEOUT_S = 15 * 60

FLAC2WAV_MAIN = '''\
// Throwaway FLAC -> WAV decoder, written and built by
// scratch/librispeech_trials_manifest.py under the scratch root.
//
// It lives outside the meethook workspace on purpose: TASK-014.04.02 adds no
// dependency to meethook, and `claxon` would be one.
//
// Prints "<sample_rate> <channels> <frames>" so the caller does not have to reopen
// the file it just wrote.
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: flac2wav <in.flac> <out.wav>");
        std::process::exit(2);
    }
    let input = PathBuf::from(&args[0]);
    let output = PathBuf::from(&args[1]);

    let mut reader = match claxon::FlacReader::open(&input) {
        Ok(reader) => reader,
        Err(e) => {
            eprintln!("{}: {e}", input.display());
            std::process::exit(1);
        }
    };
    let info = reader.streaminfo();
    if info.bits_per_sample != 16 {
        eprintln!(
            "{}: {} bits per sample, only 16 handled",
            input.display(),
            info.bits_per_sample
        );
        std::process::exit(1);
    }

    let spec = hound::WavSpec {
        channels: info.channels as u16,
        sample_rate: info.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&output, spec).expect("create wav");
    let mut frames: u64 = 0;
    for sample in reader.samples() {
        writer
            .write_sample(sample.expect("decode sample") as i16)
            .expect("write sample");
        frames += 1;
    }
    writer.finalize().expect("finalize wav");
    println!(
        "{} {} {}",
        info.sample_rate,
        info.channels,
        frames / u64::from(info.channels)
    );
}
'''

FLAC2WAV_CARGO_TOML = """\
[package]
name = "flac2wav"
version = "0.1.0"
edition = "2021"

[dependencies]
claxon = "0.4"
hound = "3.5"
"""


# --------------------------------------------------------------------------------------
# Shell helpers
# --------------------------------------------------------------------------------------


def run(argv, timeout, **kwargs):
    """Run `argv`, returning the CompletedProcess, or None if it timed out.

    Timing out is a normal outcome here, not an exception: the network stops
    answering for minutes at a time and the caller decides what to do about it.
    """
    try:
        return subprocess.run(argv, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        return None


def reachable(url):
    """(status, seconds) for a HEAD of `url`; status 0 means no answer at all."""
    started = time.monotonic()
    done = run(
        [
            "curl",
            "-sI",
            "-o",
            os.devnull,
            "-w",
            "%{http_code}",
            "--max-time",
            "12",
            "--retry",
            "2",
            "--retry-delay",
            "3",
            "--retry-connrefused",
            url,
        ],
        timeout=PROBE_TIMEOUT_S,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    if done is None or done.returncode != 0:
        return 0, elapsed
    return int(done.stdout.strip() or 0), elapsed


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_shape(path):
    """(rate, channels, frames, seconds) of a WAV, read back rather than assumed."""
    with wave.open(str(path)) as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return rate, handle.getnchannels(), frames, frames / rate


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


def pick_mirror():
    """The first mirror that answers, or None. Probing is the first thing the run does.

    A half-fetched corpus that yields a manifest of eleven speakers is worse than a
    clean stop, so a silent host ends the run here rather than partway through.
    """
    for base in MIRRORS:
        status, elapsed = reachable(f"{base}/{ARCHIVE}")
        print(f"  probe {base}/{ARCHIVE}  ->  {status or 'no answer'}  ({elapsed:.1f} s)")
        if status == 200:
            return base
    return None


def build_decoder(root):
    """Build the throwaway FLAC decoder under `root`; return the binary path.

    CARGO_HOME is redirected into the scratch root because the sandbox refuses writes
    to ~/.cargo, and a registry cache miss would otherwise fail the build.
    """
    crate = root / "flac2wav"
    binary = crate / "target" / "release" / "flac2wav"
    (crate / "src").mkdir(parents=True, exist_ok=True)
    (crate / "Cargo.toml").write_text(FLAC2WAV_CARGO_TOML)
    (crate / "src" / "main.rs").write_text(FLAC2WAV_MAIN)

    # Always invoked rather than skipped when the binary exists: cargo's own freshness
    # check costs a tenth of a second and it is the only thing that notices when the
    # source written just above differs from what the binary was built from.
    env = dict(os.environ, CARGO_HOME=str(root / "cargo"))
    done = run(
        ["cargo", "build", "--release", "--quiet"],
        timeout=FETCH_TIMEOUT_S,
        cwd=crate,
        env=env,
    )
    if done is None or done.returncode != 0:
        sys.exit(
            "could not build the FLAC decoder. Without it there is no route from this "
            "corpus to WAV on this machine: afconvert refuses FLAC with "
            "\"ExtAudioFileSetProperty ('cfmt') failed ('fmt?')\"."
        )
    print(f"  decoder built: {binary}")
    return binary


def probe_decode(root, base, binary):
    """Prove the decode path on one real corpus file before the bulk fetch starts.

    A ranged fetch of the archive's first megabytes yields the metadata files and the
    first chapter's FLACs; a truncated gzip stream still decompresses its complete
    members. This is what makes "prove the decoder, then fetch 337 MB" cheap.
    """
    probe = root / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    head = probe / "head.tar.gz"

    if not head.exists() or head.stat().st_size == 0:
        done = run(
            [
                "curl",
                "-s",
                "--retry",
                "2",
                "--max-time",
                "120",
                "-r",
                "0-3145727",
                "-o",
                str(head),
                f"{base}/{ARCHIVE}",
            ],
            timeout=300,
        )
        if done is None or done.returncode != 0 or not head.exists():
            sys.exit("ranged fetch of the archive head failed; stopping before the bulk fetch")
    print(f"  ranged fetch: {head} ({head.stat().st_size} bytes)")

    # Truncated stream: members before the cut extract, then it raises. That is the
    # expected outcome, not a failure.
    extracted = []
    try:
        with tarfile.open(head, "r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".flac"):
                    archive.extract(member, probe, filter="data")
                    extracted.append(probe / member.name)
                    if len(extracted) >= 1:
                        break
    except (tarfile.TarError, EOFError, OSError):
        pass
    if not extracted:
        sys.exit("the archive head yielded no FLAC file; cannot prove the decoder")

    source = extracted[0]
    target = probe / "probe.wav"
    done = run([str(binary), str(source), str(target)], timeout=120, capture_output=True, text=True)
    if done is None or done.returncode != 0:
        sys.exit(f"decoder failed on {source}: {(done.stderr if done else 'timed out')}")

    rate, channels, frames, seconds = wav_shape(target)
    print(f"  decoded {source.name} -> {target.name}")
    print(f"  read back: {rate} Hz, {channels} ch, {frames} frames, {seconds:.3f} s")
    if (rate, channels) != (16000, 1):
        sys.exit(f"expected 16000 Hz mono from LibriSpeech, read {rate} Hz {channels} ch")
    return rate, channels, seconds


def published_md5(root, base):
    """The corpus's own md5 for the archive, or None if the checksum file is unreachable.

    Fetched rather than hardcoded: a hash copied into this file is a claim about the
    corpus that nothing here can check, and it goes stale silently.
    """
    cached = root / CHECKSUMS
    if not cached.exists():
        done = run(
            ["curl", "-s", "--max-time", "30", "-o", str(cached), f"{base}/{CHECKSUMS}"],
            timeout=60,
        )
        if done is None or done.returncode != 0 or not cached.exists():
            return None
    for line in cached.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("*").endswith(ARCHIVE):
            return fields[0]
    return None


def fetch_archive(root, base):
    """Fetch the archive, resumably, and verify it by size and md5 before using it."""
    archive = root / ARCHIVE
    expected_md5 = published_md5(root, base)
    if expected_md5 is None:
        print("  checksum file unreachable; verifying by size only")
    else:
        print(f"  published md5: {expected_md5}")

    if archive.exists() and archive.stat().st_size == ARCHIVE_BYTES:
        if expected_md5 is None or md5_of(archive) == expected_md5:
            print(f"  already present and verified: {archive}")
            return archive
        print("  present but md5 mismatched; refetching from scratch")
        archive.unlink()

    done = run(
        [
            "curl",
            "-#",
            "-C",
            "-",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "--max-time",
            str(FETCH_TIMEOUT_S - 60),
            "-o",
            str(archive),
            f"{base}/{ARCHIVE}",
        ],
        timeout=FETCH_TIMEOUT_S,
    )
    if done is None or done.returncode != 0 or not archive.exists():
        sys.exit(
            "the archive did not arrive. It is resumable: rerun and the fetch continues "
            f"from {archive.stat().st_size if archive.exists() else 0} bytes."
        )
    if archive.stat().st_size != ARCHIVE_BYTES:
        sys.exit(f"{archive} is {archive.stat().st_size} bytes, expected {ARCHIVE_BYTES}")
    if expected_md5 is not None and md5_of(archive) != expected_md5:
        sys.exit(f"{archive} failed its md5 check against {expected_md5}")
    print(f"  fetched and verified: {archive}")
    return archive


def extract_archive(root, archive):
    corpus = root / "LibriSpeech"
    if (corpus / "dev-clean").is_dir() and (corpus / "CHAPTERS.TXT").is_file():
        print(f"  already extracted: {corpus}")
        return corpus
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")
    print(f"  extracted: {corpus}")
    return corpus


def read_chapters(corpus):
    """{speaker: {chapter: project}} for dev-clean, from the corpus's own metadata.

    Session ids come from the corpus's structure, never from a file counter: a
    synthesised session id manufactures exactly the fiction that cross-session
    measurement exists to avoid.
    """
    chapters = collections.defaultdict(dict)
    with open(corpus / "CHAPTERS.TXT", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(";"):
                continue
            fields = [field.strip() for field in line.split("|")]
            if len(fields) < 5 or fields[3] != "dev-clean":
                continue
            chapter, speaker, project = fields[0], fields[1], fields[4]
            chapters[speaker][chapter] = project
    return chapters


def decode_sessions(corpus, wav_root, binary, chapters, seconds_per_session):
    """Decode each chapter's leading utterances to WAV, capped at `seconds_per_session`.

    Returns {(speaker, chapter): [(wav_path, seconds), ...]} in utterance order.
    Utterance order is the corpus's own numbering, so the selection is deterministic
    and a rerun decodes nothing twice.
    """
    decoded = {}
    total = sum(len(chs) for chs in chapters.values())
    done_count = 0
    for speaker in sorted(chapters, key=int):
        for chapter in sorted(chapters[speaker], key=int):
            source_dir = corpus / "dev-clean" / speaker / chapter
            target_dir = wav_root / speaker / chapter
            target_dir.mkdir(parents=True, exist_ok=True)
            picked = []
            seconds = 0.0
            for flac in sorted(source_dir.glob("*.flac")):
                if seconds >= seconds_per_session:
                    break
                wav = target_dir / (flac.stem + ".wav")
                if wav.exists():
                    _, _, _, length = wav_shape(wav)
                else:
                    result = run(
                        [str(binary), str(flac), str(wav)],
                        timeout=120,
                        capture_output=True,
                        text=True,
                    )
                    if result is None or result.returncode != 0:
                        sys.exit(f"decode failed: {flac}")
                    rate, _, frames = (int(x) for x in result.stdout.split())
                    length = frames / rate
                picked.append((wav, length))
                seconds += length
            if picked:
                decoded[(speaker, chapter)] = picked
            done_count += 1
            if done_count % 20 == 0:
                print(f"  decoded {done_count}/{total} sessions")
    return decoded


# --------------------------------------------------------------------------------------
# Manifests and their shape
# --------------------------------------------------------------------------------------


def write_manifest(path, rows, header):
    """Write a `speaker <TAB> session <TAB> wav` manifest.

    Rows are emitted grouped and sorted, never in directory-walk order: `speaker-trials`
    enrols each speaker's *first* session in manifest order as their reference, so a
    nondeterministic order silently changes what the run measures.
    """
    with open(path, "w", encoding="utf-8") as handle:
        for line in header:
            handle.write(f"# {line}\n")
        handle.write("#\n")
        for speaker, session, wav in rows:
            handle.write(f"{speaker}\t{session}\t{wav}\n")


def shape(rows):
    """Speaker/item/session counts and the two cross-session pair counts a manifest yields.

    Counted the way `speaker-trials` counts: an item is one (speaker, session); a
    same-speaker trial is a pair of *distinct* sessions of one speaker, and every
    cross-speaker item pair is a different-speaker trial. Pairs inside one session are
    not trials at all -- that is `MERGE_DISTANCE`'s question, not this one.
    """
    sessions = collections.defaultdict(set)
    for speaker, session, _ in rows:
        sessions[speaker].add(session)
    counts = [len(s) for s in sessions.values()]
    items = sum(counts)
    same = sum(n * (n - 1) // 2 for n in counts)
    different = items * (items - 1) // 2 - same
    return {
        "speakers": len(sessions),
        "items": items,
        "sessions_per_speaker_min": min(counts),
        "sessions_per_speaker_median": statistics.median(counts),
        "sessions_per_speaker_max": max(counts),
        "speakers_with_multiple_sessions": sum(1 for n in counts if n > 1),
        "same_speaker_pairs": same,
        "different_speaker_pairs": different,
    }


def report(name, numbers):
    print(f"  {name}:")
    for key, value in numbers.items():
        print(f"    {key:<34} {value}")
    if numbers["same_speaker_pairs"] == 0 or numbers["different_speaker_pairs"] == 0:
        sys.exit(f"{name} yields an empty side; it would measure nothing")


# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="scratch root; must be outside the repo")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="stop after proving the decoder on one ranged-fetched corpus file",
    )
    parser.add_argument(
        "--seconds-per-session", type=float, default=DEFAULT_SECONDS_PER_SESSION
    )
    args = parser.parse_args()

    default_root = os.environ.get("MEETHOOK_CALIB_ROOT") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "meethook-calib"
    )
    root = Path(args.root or default_root).resolve()
    repo = Path(__file__).resolve().parent.parent
    if root == repo or repo in root.parents or root in repo.parents:
        sys.exit(f"refusing to write corpus data at {root}: it is inside {repo}")
    root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print("librispeech_trials_manifest")
    print("  corpus:       LibriSpeech dev-clean (OpenSLR SLR12), CC BY 4.0")
    print(f"  root:         {root}")
    print(f"  byte budget:  {BYTE_BUDGET / 1e9:.2f} GB "
          f"({ARCHIVE_BYTES / 1e6:.0f} MB archive + decoded WAV)")
    print(f"  time budget:  {TIME_BUDGET_S / 60:.0f} min")
    print()

    print("reachability")
    base = pick_mirror()
    if base is None:
        sys.exit(
            "no mirror answered. This is the expected weather here rather than an "
            "anomaly: rerun later, the fetch stages are resumable."
        )
    print(f"  using {base}")
    print()

    print("decoder")
    binary = build_decoder(root)
    print()

    print("decode proof (one real corpus file, before any bulk fetch)")
    probe_decode(root, base, binary)
    print()
    if args.probe_only:
        print(f"probe only; stopped after {time.monotonic() - started:.0f} s")
        return

    print("archive")
    archive = fetch_archive(root, base)
    corpus = extract_archive(root, archive)
    print()

    print("decode")
    chapters = read_chapters(corpus)
    wav_root = root / "wav"
    decoded = decode_sessions(corpus, wav_root, binary, chapters, args.seconds_per_session)
    print(f"  {len(decoded)} sessions decoded under {wav_root}")
    print()

    # Two manifests, because "chapter" and "different recording occasion" are not the
    # same claim and TASK-014.04 should be able to choose. A chapter is a separate take,
    # usually a separate day, but the same volunteer in the same room on the same
    # microphone. A LibriVox *project* is a different book entirely, so grouping by
    # project spends items to buy a cross-session claim that needs no caveat.
    by_chapter = []
    by_project = []
    for (speaker, chapter), picked in sorted(decoded.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        project = chapters[speaker][chapter]
        for wav, _ in picked:
            by_chapter.append((speaker, chapter, str(wav)))
            by_project.append((speaker, f"p{project}", str(wav)))
    by_project.sort(key=lambda row: (int(row[0]), row[1], row[2]))

    stamp = time.strftime("%Y-%m-%d")
    chapter_manifest = root / "librispeech-dev-clean.tsv"
    project_manifest = root / "librispeech-dev-clean-crossproject.tsv"
    write_manifest(
        chapter_manifest,
        by_chapter,
        [
            f"LibriSpeech dev-clean (OpenSLR SLR12), CC BY 4.0, built {stamp}",
            "speaker = LibriVox reader id; session = LibriVox chapter id.",
            "A chapter is a separate recording take, usually a separate day, but the",
            "same volunteer in the same room on the same microphone. It is a proxy for",
            "a separate occasion, not one. See librispeech-dev-clean-crossproject.tsv",
            f"for the stricter grouping. At most {args.seconds_per_session:.0f} s per session.",
        ],
    )
    write_manifest(
        project_manifest,
        by_project,
        [
            f"LibriSpeech dev-clean (OpenSLR SLR12), CC BY 4.0, built {stamp}",
            "speaker = LibriVox reader id; session = LibriVox project id (one book).",
            "Chapters of one book are spliced into a single item, so every same-speaker",
            "pair here spans two different books -- different recording occasions, weeks",
            "or months apart, though still the same volunteer, room and microphone.",
            "Fewer speakers qualify; the cross-session claim needs no caveat.",
        ],
    )

    print("manifest shape")
    report(f"{chapter_manifest.name} (session = chapter)", shape(by_chapter))
    print()
    report(f"{project_manifest.name} (session = LibriVox project)", shape(by_project))
    print()

    wav_bytes = sum(f.stat().st_size for f in wav_root.rglob("*.wav"))
    spent = ARCHIVE_BYTES + wav_bytes
    print("outturn against budget")
    print(f"  bytes: {spent / 1e9:.2f} GB of {BYTE_BUDGET / 1e9:.2f} GB "
          f"({ARCHIVE_BYTES / 1e6:.0f} MB fetched, {wav_bytes / 1e6:.0f} MB decoded)")
    print(f"  time:  {(time.monotonic() - started) / 60:.1f} min of {TIME_BUDGET_S / 60:.0f} min")
    print(f"  free:  {shutil.disk_usage(root).free / 1e9:.1f} GB left on the scratch volume")
    print()
    print(f"manifests: {chapter_manifest}")
    print(f"           {project_manifest}")


if __name__ == "__main__":
    main()
