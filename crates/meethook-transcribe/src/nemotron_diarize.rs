//! Diarizing a speaker track with Nemotron-3-Diarization, producing the same [`Diarization`]
//! the pyannote pipeline does.
//!
//! Nemotron says who spoke when but gives no voice fingerprint, and a [`SpeakerCluster`] is
//! mostly fingerprint. So the slots' turns (see [`slot_turns`]) are embedded with the existing
//! WeSpeaker path and each slot becomes one cluster. There is no clustering step: a slot is
//! one speaker, and two slots active at the same moment are provably two people, which is what
//! fills [`SpeakerCluster::heard_at_once_with`].
//!
//! One real person can occupy two slots when the model's slot assignment churns; this does not
//! merge them. Turns too short to embed are absent from the result, as the [`Diarization`]
//! contract says, and so a slot with nothing embeddable produces no cluster.

use std::collections::BTreeMap;
use std::path::Path;

use meethook_session::SpeakerCluster;

use crate::audio::TARGET_RATE;
use crate::diarize::{Diarization, Diarize, SpeakerTurn};
use crate::nemotron_graph::NemotronGraph;
use crate::nemotron_mel::NemotronMel;
use crate::nemotron_stream::{SpeakerStep, speaker_probabilities};
use crate::nemotron_turns::{SlotTurn, slot_turns};
use crate::progress::Phase;
use crate::segmentation::LocalTurn;
use crate::speakers::{
    MIN_EMBEDDABLE_SECONDS, TurnEmbedder, WeSpeaker, reference_embedding, representatives, slice,
};
use crate::{Result, onnx};

/// Diarization by Nemotron's streaming speaker cache, with WeSpeaker voice fingerprints.
///
/// Both graphs are loaded once and reused across a batch, like [`crate::OnnxDiarizer`].
pub struct NemotronDiarizer {
    graph: NemotronGraph,
    mel: NemotronMel,
    embedder: WeSpeaker,
    accelerated: bool,
}

impl NemotronDiarizer {
    /// Loads the Nemotron graph from `models_dir` (where the registry installs it) and the
    /// WeSpeaker graph from `embedding_model`.
    pub fn load(models_dir: &Path, embedding_model: &Path) -> Result<NemotronDiarizer> {
        let graph = NemotronGraph::open(models_dir)?;
        let embedder = onnx::open_session(embedding_model)?;
        Ok(NemotronDiarizer {
            accelerated: graph.accelerated && embedder.accelerated,
            graph,
            mel: NemotronMel::new(),
            embedder: WeSpeaker::new(embedder.session),
        })
    }

    /// False when either graph fell back to CPU. See [`crate::OnnxDiarizer::accelerated`].
    pub fn accelerated(&self) -> bool {
        self.accelerated
    }
}

impl Diarize for NemotronDiarizer {
    fn diarize(&mut self, speaker_16k_mono: &[f32]) -> Result<Diarization> {
        diarize_with(
            &mut self.graph,
            &mut self.mel,
            &mut self.embedder,
            speaker_16k_mono,
        )
    }
}

/// All the logic, over seams so tests need no weights.
fn diarize_with(
    step: &mut impl SpeakerStep,
    mel: &mut NemotronMel,
    embedder: &mut impl TurnEmbedder,
    samples: &[f32],
) -> Result<Diarization> {
    if samples.is_empty() {
        return Ok(Diarization::default());
    }
    let probs = speaker_probabilities(step, mel, samples)?;
    assemble(&slot_turns(&probs), samples, embedder)
}

/// One embedded turn of one slot.
struct Embedded {
    turn: SlotTurn,
    embedding: Vec<f32>,
}

/// Embeds `turns` and builds one cluster per slot that has an embeddable turn.
fn assemble(
    turns: &[SlotTurn],
    samples: &[f32],
    embedder: &mut impl TurnEmbedder,
) -> Result<Diarization> {
    let track_end_s = samples.len() as f64 / TARGET_RATE as f64;

    let mut by_slot: BTreeMap<usize, Vec<Embedded>> = BTreeMap::new();
    let mut phase = Phase::start("diarize: embedding voices");
    for (index, &turn) in turns.iter().enumerate() {
        phase.at(index, turns.len());
        if turn.end_s - turn.start_s < MIN_EMBEDDABLE_SECONDS {
            continue;
        }
        let clip = LocalTurn {
            start_s: turn.start_s,
            end_s: turn.end_s,
            window: 0,
            local_speaker: turn.slot,
        };
        let Some(embedding) = embedder.embed(slice(samples, &clip))? else {
            continue;
        };
        by_slot
            .entry(turn.slot)
            .or_default()
            .push(Embedded { turn, embedding });
    }
    phase.done();

    let duration = |t: &SlotTurn| t.end_s - t.start_s;
    let mut slots: Vec<(usize, Vec<Embedded>)> = by_slot.into_iter().collect();
    // Most talkative first so cluster 0 is the person the meeting was mostly with. The sort is
    // stable over ascending slot, so ties go to the lower slot.
    let spoken = |m: &Vec<Embedded>| -> f64 { m.iter().map(|e| duration(&e.turn)).sum() };
    slots.sort_by(|a, b| spoken(&b.1).total_cmp(&spoken(&a.1)));

    let overlap = |a: &[Embedded], b: &[Embedded]| {
        a.iter().any(|x| {
            b.iter()
                .any(|y| x.turn.start_s < y.turn.end_s && y.turn.start_s < x.turn.end_s)
        })
    };

    let mut clusters = Vec::with_capacity(slots.len());
    let mut out_turns = Vec::new();
    for (id, (_, members)) in slots.iter().enumerate() {
        let embeddings: Vec<Vec<f32>> = members.iter().map(|m| m.embedding.clone()).collect();
        let indices: Vec<usize> = (0..embeddings.len()).collect();
        let local: Vec<LocalTurn> = members
            .iter()
            .map(|m| LocalTurn {
                start_s: m.turn.start_s,
                end_s: m.turn.end_s,
                window: 0,
                local_speaker: m.turn.slot,
            })
            .collect();
        let heard_at_once_with = slots
            .iter()
            .enumerate()
            .filter(|&(other, (_, theirs))| other != id && overlap(members, theirs))
            .map(|(other, _)| other as u32)
            .collect();
        clusters.push(SpeakerCluster {
            id: id as u32,
            embedding: reference_embedding(&indices, &embeddings),
            speech_seconds: spoken(members),
            first_spoke_seconds: members
                .iter()
                .map(|m| m.turn.start_s)
                .min_by(f64::total_cmp)
                .unwrap_or(0.0),
            heard_at_once_with,
            representatives: representatives(&local, track_end_s),
        });
        out_turns.extend(members.iter().map(|m| SpeakerTurn {
            start_s: m.turn.start_s,
            end_s: m.turn.end_s,
            cluster: id as u32,
        }));
    }
    out_turns.sort_by(|a, b| {
        a.start_s
            .total_cmp(&b.start_s)
            .then(a.cluster.cmp(&b.cluster))
    });

    Ok(Diarization {
        clusters,
        turns: out_turns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron_cache::config::{CHUNK, HIDDEN, SPEAKERS};
    use crate::nemotron_graph::{StepOutput, encoder_frames};
    use crate::nemotron_mel::{MEL_BINS, STACK};
    use meethook_session::MIN_REPRESENTATIVE_SECONDS;

    fn t(slot: usize, start_s: f64, end_s: f64) -> SlotTurn {
        SlotTurn {
            start_s,
            end_s,
            slot,
        }
    }

    /// An embedder that answers from a closure, called once per embeddable clip.
    struct Fake<F: FnMut(&[f32]) -> Option<Vec<f32>>>(F, usize);
    impl<F: FnMut(&[f32]) -> Option<Vec<f32>>> TurnEmbedder for Fake<F> {
        fn embed(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
            self.1 += 1;
            Ok((self.0)(samples))
        }
    }

    /// Two directions: slot 0 sounds like [1,0], any other slot like [0,1].
    fn by_slot_vector(samples: &[f32]) -> Option<Vec<f32>> {
        // The tests paint each slot's audio with its own constant: slot + 1.
        Some(if samples[0] < 1.5 {
            vec![1.0, 0.0]
        } else {
            vec![0.0, 1.0]
        })
    }

    /// 100 seconds of audio where every second carries the value `slot + 1` of whichever turn
    /// covers it, painted from `turns` (later turns overwrite earlier on overlap, which the
    /// by-slot embedder tolerates because tests keep overlaps short).
    fn painted(turns: &[SlotTurn]) -> Vec<f32> {
        let mut s = vec![0.0f32; 100 * TARGET_RATE as usize];
        for turn in turns {
            let a = (turn.start_s * TARGET_RATE as f64) as usize;
            let b = (turn.end_s * TARGET_RATE as f64) as usize;
            s[a..b].fill(turn.slot as f32 + 1.0);
        }
        s
    }

    fn run(turns: &[SlotTurn]) -> Diarization {
        assemble(turns, &painted(turns), &mut Fake(by_slot_vector, 0)).unwrap()
    }

    #[test]
    fn ids_follow_talk_time_most_talkative_first() {
        // Slot 5 speaks 3 s, slot 2 speaks 10 s: slot 2 is cluster 0.
        let d = run(&[t(5, 0.0, 3.0), t(2, 10.0, 20.0)]);
        assert_eq!(d.clusters.len(), 2);
        assert_eq!(d.clusters[0].id, 0);
        assert_eq!(d.clusters[0].speech_seconds, 10.0);
        assert_eq!(d.clusters[1].id, 1);
        assert_eq!(d.clusters[1].speech_seconds, 3.0);
        let ids: Vec<(f64, u32)> = d.turns.iter().map(|t| (t.start_s, t.cluster)).collect();
        assert_eq!(ids, vec![(0.0, 1), (10.0, 0)]);
    }

    #[test]
    fn the_embedding_is_the_mean_normalized_afterwards() {
        // Two turns of one slot with non-unit-sum, unequal vectors.
        let turns = [t(0, 0.0, 2.0), t(0, 5.0, 7.0)];
        let mut n = 0;
        let mut e = Fake(
            |_: &[f32]| {
                n += 1;
                Some(if n == 1 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.6, 0.8]
                })
            },
            0,
        );
        let d = assemble(&turns, &painted(&turns), &mut e).unwrap();
        let got = &d.clusters[0].embedding;
        let mean = [0.8f32, 0.4];
        let norm = (mean[0] * mean[0] + mean[1] * mean[1]).sqrt();
        assert!((got[0] - mean[0] / norm).abs() < 1e-6);
        assert!((got[1] - mean[1] / norm).abs() < 1e-6);
        let len: f32 = got.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn first_spoke_speech_seconds_and_representatives() {
        let d = run(&[t(0, 4.0, 5.0), t(0, 8.0, 12.0)]);
        let c = &d.clusters[0];
        assert_eq!(c.first_spoke_seconds, 4.0);
        assert_eq!(c.speech_seconds, 5.0);
        assert_eq!(c.representatives.len(), 2);
        assert!(c.representatives.iter().all(|r| {
            r.end - r.start >= MIN_REPRESENTATIVE_SECONDS - 1e-9 && r.start >= 0.0 && r.end <= 100.0
        }));
        // Longest first.
        assert_eq!(
            (c.representatives[0].start, c.representatives[0].end),
            (8.0, 12.0)
        );
    }

    #[test]
    fn slots_active_together_list_each_other_and_others_do_not() {
        // Slots 0 and 1 overlap between 10 and 12; slot 2 never overlaps either.
        let d = run(&[t(0, 0.0, 12.0), t(1, 10.0, 21.5), t(2, 30.0, 33.0)]);
        assert_eq!(d.clusters.len(), 3);
        // Cluster order by talk time: slot 1 (11.5) = 0, slot 0 (12.0) wins -> check by seconds.
        let by_secs = |s: f64| d.clusters.iter().find(|c| c.speech_seconds == s).unwrap();
        let (a, b, c) = (by_secs(12.0), by_secs(11.5), by_secs(3.0));
        assert_eq!(a.heard_at_once_with, vec![b.id]);
        assert_eq!(b.heard_at_once_with, vec![a.id]);
        assert!(c.heard_at_once_with.is_empty());
    }

    #[test]
    fn turns_that_only_touch_are_not_heard_at_once() {
        let d = run(&[t(0, 0.0, 5.0), t(1, 5.0, 10.0)]);
        assert!(d.clusters.iter().all(|c| c.heard_at_once_with.is_empty()));
    }

    #[test]
    fn a_slot_with_only_short_turns_has_no_cluster_and_no_turns() {
        let d = run(&[t(0, 0.0, 5.0), t(3, 10.0, 10.4), t(3, 20.0, 20.3)]);
        assert_eq!(d.clusters.len(), 1);
        assert_eq!(d.turns.len(), 1);
        assert_eq!(d.turns[0].cluster, 0);
    }

    #[test]
    fn a_turn_the_embedder_declines_is_absent() {
        let turns = [t(0, 0.0, 5.0), t(1, 10.0, 15.0)];
        let mut e = Fake(|s: &[f32]| (s[0] < 1.5).then(|| vec![1.0, 0.0]), 0);
        let d = assemble(&turns, &painted(&turns), &mut e).unwrap();
        assert_eq!(d.clusters.len(), 1);
        assert_eq!(d.turns.len(), 1);
    }

    #[test]
    fn no_turns_is_an_empty_diarization() {
        assert_eq!(run(&[]), Diarization::default());
    }

    // ---- through a fake graph ----

    /// A graph with no weights whose logits mark `active` slot ranges (in mel frames) as
    /// speaking, wherever the loop asks.
    struct Scripted {
        active: Vec<(usize, usize, usize)>,
        calls: usize,
    }

    impl SpeakerStep for Scripted {
        fn step(&mut self, f: &[f32], c: &[f32], _v: &[bool]) -> Result<StepOutput> {
            let mel = f.len() / MEL_BINS;
            let cached = c.len() / HIDDEN;
            let enc = encoder_frames(mel);
            let base = self.calls * CHUNK * STACK;
            self.calls += 1;
            let mut logits = vec![-10.0; (cached + enc) * STACK * SPEAKERS];
            for r in 0..enc * STACK {
                let frame = base + r;
                for &(slot, from, to) in &self.active {
                    if (from..to).contains(&frame) {
                        logits[(cached * STACK + r) * SPEAKERS + slot] = 10.0;
                    }
                }
            }
            Ok(StepOutput {
                logits,
                chunk_embeds: vec![0.0; enc * HIDDEN],
                silence: vec![0.0; HIDDEN],
            })
        }
    }

    fn run_graph(active: Vec<(usize, usize, usize)>, seconds: usize) -> Diarization {
        let mut samples = vec![0.0f32; seconds * TARGET_RATE as usize];
        // Slot-coloured audio: the by-slot embedder reads the first sample of each clip.
        for &(slot, from, to) in &active {
            let (a, b) = (from * 160, (to * 160).min(samples.len()));
            samples[a..b].fill(slot as f32 + 1.0);
        }
        // Mel features of this audio are irrelevant: the fake ignores them.
        diarize_with(
            &mut Scripted { active, calls: 0 },
            &mut NemotronMel::new(),
            &mut Fake(by_slot_vector, 0),
            &samples,
        )
        .unwrap()
    }

    #[test]
    fn a_fake_graph_drives_the_whole_pipeline() {
        // Frames are 10 ms: slot 0 speaks 1-6 s, slot 4 speaks 4-6 s (overlapping), then 8-9.
        let d = run_graph(vec![(0, 100, 600), (4, 400, 600), (4, 800, 900)], 12);
        assert_eq!(d.clusters.len(), 2);
        assert_eq!(d.clusters[0].speech_seconds, 5.0);
        assert_eq!(d.clusters[1].speech_seconds, 3.0);
        assert_eq!(d.clusters[0].heard_at_once_with, vec![1]);
        assert_eq!(d.clusters[1].heard_at_once_with, vec![0]);
        assert_eq!(d.turns.len(), 3);
        assert!(d.turns.windows(2).all(|w| w[0].start_s <= w[1].start_s));
    }

    #[test]
    fn a_silent_track_is_empty_and_ok() {
        assert_eq!(run_graph(vec![], 12), Diarization::default());
    }

    #[test]
    fn an_empty_track_never_touches_the_graph() {
        let mut g = Scripted {
            active: vec![],
            calls: 0,
        };
        let d = diarize_with(
            &mut g,
            &mut NemotronMel::new(),
            &mut Fake(by_slot_vector, 0),
            &[],
        )
        .unwrap();
        assert_eq!(d, Diarization::default());
        assert_eq!(g.calls, 0);
    }
}
