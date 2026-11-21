// Spike for MH-40.1: reads /private/tmp/flacspike/in.raw (written by flac-spike exact), streams frames with flacenc, patches STREAMINFO.
use flacenc::component::{BitRepr, StreamInfo};
use flacenc::error::Verify;
use flacenc::source::{Fill, FrameBuf};
use std::io::{Seek, SeekFrom, Write};
fn bytes<T: BitRepr>(x: &T) -> Vec<u8> {
    let mut s = flacenc::bitsink::ByteSink::new();
    x.write(&mut s).unwrap();
    s.as_slice().to_vec()
}
fn main() {
    let finish = std::env::args().nth(1).as_deref() != Some("kill");
    let raw = std::fs::read("/private/tmp/flacspike/in.raw").unwrap();
    let s: Vec<i32> = raw
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as i32)
        .collect();
    let cfg = flacenc::config::Encoder::default().into_verified().unwrap();
    let mut info = StreamInfo::new(16000, 1, 16).unwrap();
    let mut f = std::fs::File::create(if finish {
        "/private/tmp/flacspike/fs.flac"
    } else {
        "/private/tmp/flacspike/fk.flac"
    })
    .unwrap();
    f.write_all(b"fLaC").unwrap();
    let mut ph = info.clone();
    ph.set_block_sizes(4096, 4096).unwrap();
    // placeholder header: last-metadata-block flag set, 34 bytes
    let mut hdr = vec![0x80, 0, 0, 34];
    hdr.extend(bytes(&ph));
    f.write_all(&hdr).unwrap();
    let t = std::time::Instant::now();
    let mut n = 0;
    for (i, blk) in s.chunks(4096).enumerate() {
        let mut fb = FrameBuf::with_size(1, blk.len().max(16)).unwrap();
        fb.fill_interleaved(blk).unwrap();
        let fr = flacenc::encode_fixed_size_frame(&cfg, &fb, i, &info).unwrap();
        info.update_frame_info(&fr);
        f.write_all(&bytes(&fr)).unwrap();
        n += 1;
    }
    println!("{n} frames in {:?}", t.elapsed());
    if finish {
        info.set_block_sizes(4096, 4096).unwrap();
        f.seek(SeekFrom::Start(8)).unwrap();
        f.write_all(&bytes(&info)).unwrap();
    }
}
