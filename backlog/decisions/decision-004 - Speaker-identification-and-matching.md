---
id: decision-004
title: Speaker identification and matching
date: '2026-08-27 04:39'
status: accepted
---
A diarized cluster is matched against the enrolled speaker database by cosine similarity — an argmax over every stored reference — accepted only if the best match clears a single threshold (`IDENTIFY_DISTANCE`), with no three-way "maybe" outcome, because there was no UI capable of resolving ambiguity at the time this shipped. The bias is the same one clustering uses: reject rather than guess, since a false accept silently mislabels a transcript nobody re-reads, while a false reject is a visible unnamed voice fixed in seconds through enrollment. `IDENTIFY_DISTANCE` is deliberately tracked as a separate constant from clustering's `MERGE_DISTANCE`, even though both started at 0.45, because cross-session identification variation (different meeting, different room or headset, weeks apart) is structurally larger than within-session clustering variation — recalibrating one must never silently move the other. Calibration data comes from LibriSpeech's `dev-clean` corpus, chosen because it's disjoint from the embedding model's own training data and licensed for publishing derived numbers — understood explicitly as a complement to, not a replacement for, measurement against real meethook recordings, since a public corpus has no ScreenCaptureKit tap or meeting codec in the chain.

A real production bug was traced to these two thresholds being conflated in practice: clustering's average-linkage distance and identification's centroid distance are structurally different comparisons, related by the same size-bias identity discovered in the clustering work, and a session existed where the two metrics disagreed enough that identification re-joined two people clustering had correctly kept apart, misattributing nine percent of that session's speech under one person's name. The fix documents the relationship between the two metrics explicitly in both places they're used and re-derives each threshold independently rather than assuming they should track each other.

Separately, identification must honor the same categorical "these are different people" constraint clustering already enforces from segmentation's own evidence. When multiple clusters' best match lands on the same enrolled name, contenders are ranked by similarity and awarded the name greedily, but any contender that was heard in the same window as a cluster already awarded that name is vetoed outright — checked against the whole set of already-awarded clusters, not just the winner, since the exclusion isn't transitive and a naive winner-only check can leave two mutually incompatible losers sharing one name. A vetoed cluster stays unidentified rather than falling back to its second-best match, deliberately, because a runner-up fallback is the exact mechanism that caused the original threshold-confusion bug.

## Considered options

- A three-way "ambiguous" match tier requiring human disambiguation — no interface existed to resolve it at the time.
- Sharing one threshold between clustering and identification — proven unsafe; they measure structurally different things.
- Falling back to a runner-up reference when a heard-at-once veto fires — rejected; this is the mechanism that caused the metric-confusion bug in the first place.
- Rejecting both contenders on a conflict instead of resolving greedily — reinvents the ambiguous-tier problem the single-threshold design was built to avoid.

