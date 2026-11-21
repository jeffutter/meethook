//! Throwaway spike for MH-40.1: does rusty_flac 0.5.1 do what the recorder needs?
//!
//! Subcommands (all take a 16 kHz mono float32 WAV, e.g. a session's mic.cleaned.wav):
//!   exact <in.wav> <dir> [seconds]   criterion 1: streamed encode, header patch, decode compare
//!   killed <in.wav> <dir> [seconds]  criterion 2: placeholder + frames, no finish; truncation
//!   speed <in.wav> [minutes]         criterion 3: encode time for a long track
//!   cmp <a.wav> <b.wav>              compare the PCM of two 16-bit WAVs sample for sample

use rusty_flac::{DecodeError, Decoder, Encoder};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

fn read_f32_wav(path: &str) -> (u32, Vec<f32>) {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[..4], b"RIFF");
    let mut p = 12;
    let (mut rate, mut ch, mut fmt) = (0, 0, 0);
    while p + 8 <= b.len() {
        let id = &b[p..p + 4];
        let len = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
        let body = &b[p + 8..(p + 8 + len).min(b.len())];
        if id == b"fmt " {
            fmt = u16::from_le_bytes(body[0..2].try_into().unwrap());
            ch = u16::from_le_bytes(body[2..4].try_into().unwrap());
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            let bits = u16::from_le_bytes(body[14..16].try_into().unwrap());
            assert_eq!(bits, 32, "expected float32, fmt tag {fmt}");
        } else if id == b"data" {
            assert_eq!(ch, 1);
            let s = body
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            return (rate, s);
        }
        p += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

fn quantise(x: &[f32]) -> Vec<i32> {
    x.iter()
        .map(|v| (v * 32768.0).round().clamp(-32768.0, 32767.0) as i32)
        .collect()
}

/// The fixture the recorder would encode: real speech, plus edge cases.
fn fixture(path: &str, seconds: Option<usize>) -> Vec<i32> {
    let (rate, f) = read_f32_wav(path);
    assert_eq!(rate, 16000);
    let n = seconds.map_or(f.len(), |s| (s * 16000).min(f.len()));
    let mut s = quantise(&f[..n]);
    if seconds.is_some() && std::env::var("NOEDGE").is_err() {
        // edge cases appended (only for the short exactness runs)
        s.extend(std::iter::repeat(0).take(16000 * 20)); // long digital silence
        for i in 0..20000 {
            s.push(if (i / 7) % 2 == 0 {
                i16::MAX as i32
            } else {
                i16::MIN as i32
            }); // clipping square
        }
        let mut x: u32 = 12345;
        for _ in 0..50000 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            s.push(((x >> 16) as i16) as i32); // full-scale white noise
        }
        s.extend([i16::MIN as i32, i16::MAX as i32, 0, -1, 1]); // partial final block
    }
    s
}

/// Streamed encode the way the recorder will: placeholder header, frames as produced.
/// Returns (file bytes written, final streaminfo). `finish` false simulates a kill.
fn encode_to(
    path: &Path,
    samples: &[i32],
    chunk_sizes: &[usize],
    finish: bool,
) -> rusty_flac::StreamInfo {
    let mut enc = Encoder::new(16000, 1, 16).unwrap();
    let mut f = std::fs::File::create(path).unwrap();
    let placeholder = enc.stream_info_placeholder();
    f.write_all(&placeholder.stream_header()).unwrap();
    let (mut pos, mut k) = (0, 0);
    while pos < samples.len() {
        let n = chunk_sizes[k % chunk_sizes.len()].min(samples.len() - pos);
        k += 1;
        enc.push_interleaved(&samples[pos..pos + n]).unwrap();
        pos += n;
        for fr in enc.take_frames() {
            f.write_all(&fr.data).unwrap();
        }
    }
    if !finish {
        return placeholder; // encoder dropped: the buffered tail is lost
    }
    let (frames, info) = enc.finish_frames();
    for fr in frames {
        f.write_all(&fr.data).unwrap();
    }
    f.seek(SeekFrom::Start(8)).unwrap();
    f.write_all(&info.to_bytes()).unwrap();
    f.sync_all().unwrap();
    info
}

fn decode_all(bytes: &[u8]) -> (Vec<i32>, Result<(), DecodeError>) {
    let mut d = match Decoder::new(bytes) {
        Ok(d) => d,
        Err(e) => return (vec![], Err(e)),
    };
    let mut out = Vec::new();
    loop {
        match d.next_frame_interleaved::<i32>(&mut out) {
            Ok(Some(_)) => {}
            Ok(None) => return (out, Ok(())),
            Err(e) => return (out, Err(e)),
        }
    }
}

fn write_wav16(path: &Path, s: &[i32]) {
    let mut b = Vec::new();
    let data = (s.len() * 2) as u32;
    b.extend(b"RIFF");
    b.extend((36 + data).to_le_bytes());
    b.extend(b"WAVEfmt ");
    b.extend(16u32.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(1u16.to_le_bytes());
    b.extend(16000u32.to_le_bytes());
    b.extend(32000u32.to_le_bytes());
    b.extend(2u16.to_le_bytes());
    b.extend(16u16.to_le_bytes());
    b.extend(b"data");
    b.extend(data.to_le_bytes());
    for v in s {
        b.extend((*v as i16).to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

fn read_wav16(path: &str) -> Vec<i32> {
    let b = std::fs::read(path).unwrap();
    let mut p = 12;
    while p + 8 <= b.len() {
        let len = u32::from_le_bytes(b[p + 4..p + 8].try_into().unwrap()) as usize;
        if &b[p..p + 4] == b"data" {
            let end = (p + 8 + len).min(b.len());
            return b[p + 8..end]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
                .collect();
        }
        p += 8 + len + (len & 1);
    }
    panic!("no data");
}

fn first_diff(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "cmp" => {
            let (x, y) = (read_wav16(&a[2]), read_wav16(&a[3]));
            println!(
                "len {} vs {}, first diff {:?}, identical {}",
                x.len(),
                y.len(),
                first_diff(&x, &y),
                x == y
            );
        }
        "exact" => {
            let secs = a.get(4).map(|s| s.parse().unwrap());
            let s = fixture(&a[2], secs);
            let dir = Path::new(&a[3]);
            std::fs::create_dir_all(dir).unwrap();
            // odd, realistic callback sizes: 480 (30 ms), 441, 1, 4097, 7
            let info = encode_to(
                &dir.join("out.flac"),
                &s,
                &[480, 441, 1, 4097, 7, 1600],
                true,
            );
            write_wav16(&dir.join("in.wav"), &s);
            let bytes = std::fs::read(dir.join("out.flac")).unwrap();
            let (dec, r) = decode_all(&bytes);
            println!(
                "samples in {} ; flac bytes {} ; streaminfo total {} ; rusty decode {:?} len {} bit-exact {}",
                s.len(),
                bytes.len(),
                info.total_samples,
                r,
                dec.len(),
                dec == s
            );
            println!("streaminfo {info:?}");
            // md5 check: STREAMINFO md5 is md5 of the little-endian s16 PCM
            std::fs::write(
                dir.join("in.raw"),
                s.iter()
                    .flat_map(|v| (*v as i16).to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
            .unwrap();
            let hex: String = info.md5.iter().map(|b| format!("{b:02x}")).collect();
            println!("streaminfo md5 {hex}  (compare: md5 -q in.raw)");
        }
        "killed" => {
            let secs = a.get(4).map(|s| s.parse().unwrap());
            let s = fixture(&a[2], secs);
            let dir = Path::new(&a[3]);
            std::fs::create_dir_all(dir).unwrap();
            let p = dir.join("killed.flac");
            encode_to(&p, &s, &[480, 441, 1600], false);
            let bytes = std::fs::read(&p).unwrap();
            let (dec, r) = decode_all(&bytes);
            let hdr = Decoder::new(&bytes).unwrap().streaminfo().clone();
            let full_frames = s.len() / 4096 * 4096;
            println!(
                "killed: file {} B, streaminfo total_samples {}, md5 zero {}",
                bytes.len(),
                hdr.total_samples,
                hdr.md5 == [0; 16]
            );
            println!(
                "killed: rusty decode result {:?}, got {} samples; expected complete-frame samples {}; prefix exact {}",
                r,
                dec.len(),
                full_frames,
                dec == s[..dec.len()]
            );
            write_wav16(&dir.join("killed_expect.wav"), &s[..full_frames]);
            // cut mid last frame
            for cut in [1usize, 5, 40, 200] {
                let t = &bytes[..bytes.len() - cut];
                let (d, r) = decode_all(t);
                println!(
                    "cut -{cut} B: result {:?}, {} samples, prefix exact {}",
                    r,
                    d.len(),
                    d == s[..d.len()]
                );
            }
            std::fs::write(dir.join("killed_cut.flac"), &bytes[..bytes.len() - 40]).unwrap();
            // header-only (killed before the first frame completes)
            let (d, r) = decode_all(&bytes[..42]);
            println!("header only: {:?}, {} samples", r, d.len());
        }
        "speed" => {
            let mins: usize = a.get(3).map_or(79, |s| s.parse().unwrap());
            let (_, f) = read_f32_wav(&a[2]);
            let q = quantise(&f);
            let want = mins * 60 * 16000;
            let mut s = Vec::with_capacity(want);
            while s.len() < want {
                let n = (want - s.len()).min(q.len());
                s.extend_from_slice(&q[..n]);
            }
            let secs = want as f64 / 16000.0;
            let t = Instant::now();
            let mut enc = Encoder::new(16000, 1, 16).unwrap();
            let mut out = 0usize;
            for c in s.chunks(480) {
                enc.push_interleaved(c).unwrap();
                for fr in enc.take_frames() {
                    out += fr.data.len();
                }
            }
            let (frames, _) = enc.finish_frames();
            out += frames.iter().map(|f| f.data.len()).sum::<usize>();
            let el = t.elapsed().as_secs_f64();
            println!(
                "{mins} min ({} samples): encoded in {el:.2} s = {:.0}x real time, output {:.1} MB",
                want,
                secs / el,
                out as f64 / 1e6
            );
        }
        _ => panic!("unknown"),
    }
}
