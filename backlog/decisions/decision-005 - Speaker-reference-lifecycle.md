---
id: decision-005
title: Speaker reference lifecycle
date: '2026-08-27 04:39'
status: accepted
---
Naming a second voice with a name that's already enrolled adds a new reference to that person's set rather than overwriting the one stored reference, reversing an earlier design that replaced references on re-enrollment. The earlier design silently discarded a person's previously confirmed audio and, in the same run, could un-name a voice that reference had been correctly identifying — with no error shown. The reference-set redesign was settled by measurement against a LibriSpeech-derived trial set: keeping every reference and matching against the nearest one beat newest-wins replacement outright, and beat averaging the references together on safety grounds — averaging measurably narrowed the margin between a correct match and an impostor, and would break exact-embedding matching used elsewhere to correct a stored assignment.

A name is only allowed to become a permanent reference in `speakers.json` once it's backed by at least 5.0 seconds of that voice's embedded speech — a floor set from a direct measurement sweep: references built from less than roughly 2.4 seconds of audio measured further from their own owner's held-out speech than a stranger's reference would, while references above roughly 5.2 seconds showed no further benefit. Below that floor, a name is still accepted and still labels that one session's transcript, but is written into a separate per-session file rather than `speakers.json`, and matched back to the clustering it was assigned against by exact embedding equality rather than by cluster ID — cluster IDs are only stable within one clustering run, and exact-embedding matching means a name from a stale run is dropped rather than silently reattached to the wrong voice after a re-diarization. A key finding justified accepting a floor at all rather than rejecting weak names outright: a reference built from too little audio fails closed (the voice goes unidentified later) rather than actively misattributing speech, so keeping it session-scoped costs nothing while still letting the transcript show the name the user actually typed.

Because a reference can't be regenerated once its source audio is gone, removing one is its own command (`meethook forget`) that computes and prints every downstream consequence — a voice reverting to "Unknown," switching to a different stored name, or newly winning a name that had been blocked — before any write, and refuses to write without explicit confirmation. The same before/after-diff check runs on every accepted enrollment answer generally: an answer that would cost an earlier answer its name in the same run is refused outright, after a third, previously unenumerated path to silent un-naming (a below-floor session assignment beating a stored identification) turned up during testing and made clear that one general safety check was needed rather than several situation-specific guards.

## Considered options

- Averaging two references into one blended embedding instead of keeping both — measurably less safe against impostors, and breaks exact-embedding correction.
- Writing a below-floor reference anyway with a "low confidence" flag — still puts a vector on disk further from its owner than a stranger's reference.
- Rejecting below-floor names outright rather than accepting them session-scoped — throws away information the user directly gave the tool for no safety benefit, since a weak reference fails closed rather than misattributing.
- Patching only the reference-overwrite un-naming case rather than a general before/after check — abandoned once a third, structurally different un-naming path was found.

