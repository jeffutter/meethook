---
id: decision-014
title: Diarization with Nemotron-3-Diarization and its streaming speaker cache
date: '2026-11-21T06:07:00Z'
status: accepted
---
## Context

<!-- SECTION:CONTEXT:BEGIN -->
decision-003 chose pyannote segmentation-3.0 plus WeSpeaker plus constrained agglomerative clustering, and passed over Sortformer because its ONNX export was broken. On a real 129 minute, three-speaker session that pipeline produced 364 clusters (doc-015). Every small cluster showed up as an extra "Unknown N" in transcripts and as an extra question in `enroll`. The clustering, the adoption pass and the talk-time floors all exist to repair that over-splitting. A working ONNX export of Nemotron-3-Diarization now exists.
<!-- SECTION:CONTEXT:END -->

## Decision

<!-- SECTION:DECISION:BEGIN -->
Diarize the speaker track with Nemotron-3-Diarization (onnx-community fp16 export), run in one pass over the whole track with a port of the model's streaming speaker cache (`NemotronDiarizer`). Slot probabilities are decoded into turns with hysteresis. Each slot becomes one cluster, and one WeSpeaker embedding per cluster keeps `speaker_clusters.json`, `enroll` and identification working. The pyannote segmentation graph, its window decoding, the agglomerative clustering and the adoption pass are removed.

Evidence: doc-015 compared three routes on one real session. On the 129 minute, three-speaker track the cache route used 3 slots, one per person, with 91-99% of each person's speech in one slot. Pyannote produced 364 clusters. The two agreed on 97% of frames where both named one of the three people, but that agreement is circular, since the names come from pyannote plus enrollment. The ground truth is the author's own listening comparison across real sessions (TASK-088.11): Nemotron was clearly better, so the switch went ahead. The per-disagreement tables and the 2 hour run-time and memory numbers were not recorded in that ticket.

What still applies from decision-003: speaker identity across sessions runs on a WeSpeaker ResNet34 embedding per cluster, matched by cosine distance, with the measured identification thresholds. The bias toward visible errors (an extra "Unknown" is better than a silent misattribution) also stands. The rationale for average linkage, the merge threshold and the adoption pass no longer applies, because that code is gone.
<!-- SECTION:DECISION:END -->

## Consequences

<!-- SECTION:CONSEQUENCES:BEGIN -->
- Eight speaker slots is a hard limit. A session with more than eight people shares slots, and TASK-088.11 did not record what the user sees in that case.
- First use downloads 199 MB of weights (the fp16 graph is 352 kB plus a 199 MB data file), up from about 6 MB for segmentation. Total diarization weights grow from about 32 MB to about 230 MB.
- Weights license is OpenMDW-1.1; the model card says commercial and non-commercial use are allowed.
- The graph uses `com.microsoft` contrib ops. Whether CoreML takes the whole graph varies, so the session report says whether diarization ran accelerated or on CPU. The Python reference port ran the 129 minute track on CPU in 149 s.
- One real person can occupy two slots when slot assignment churns. Slots active at the same moment are provably two people.
- Identification and enrollment thresholds were measured on pyannote clusters. TASK-088.09 re-measures them on Nemotron clusters.
- Quantized variants (q4f16, 73 MB) exist and were not evaluated for accuracy.
<!-- SECTION:CONSEQUENCES:END -->

## Alternatives

<!-- SECTION:ALTERNATIVES:BEGIN -->
- Keep pyannote: 364 clusters for three speakers on the reference session, and the repair machinery that goes with it.
- Nemotron in independent 30 s chunks, linked by WeSpeaker clustering: 146 clusters on the same session; the top 3 held 82% of speech. It keeps a clustering step and its tuning.
- Nemotron with the streaming cache (chosen): 3 slots, no clustering step.
- Sortformer: passed over in decision-003 for its broken ONNX export; the Nemotron export fixes that.
<!-- SECTION:ALTERNATIVES:END -->

