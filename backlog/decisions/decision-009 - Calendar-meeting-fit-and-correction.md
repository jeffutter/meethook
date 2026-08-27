---
id: decision-009
title: Calendar meeting-fit and correction
date: '2026-08-27 04:40'
status: accepted
---
How well a session's start time actually fits the meeting it was labelled with is tracked as its own classification — contained and begun on time, contained but joined late, matched only by proximity before the meeting started, matched only by proximity after it ended, or (for sessions predating this classification) unknown — computed from the session's start time against the meeting interval alone, deliberately never from the session's end or duration. A session that simply runs long is the ordinary case and must never be scored as a worse match for overrunning its meeting; the start time is the only point in a session where real information about intent exists, since a spontaneous incident call that happens to land inside a scheduled standup's window is otherwise indistinguishable from someone joining that standup late. Weak fits are surfaced to the user — a caveat on the `record` command's finish line and in the rendered transcript's frontmatter — rather than hidden, and only a strong fit unlocks using the meeting's attendee list as a seed for future speaker identification, gated structurally rather than by convention, because using a merely-proximate meeting's roster to seed speaker recognition would reintroduce the exact cross-session identity contamination a prior tool in this space was found to have caused by doing exactly that.

Because the automatic matching rule can be wrong — or absent, when there's no calendar match at all — a session's meeting label can be corrected or cleared by hand through a dedicated command, implemented behind a narrow, single-method trait boundary specifically so its logic never has to depend on the macOS-only, hardware-requiring calendar-access crate to be tested. A correction made by hand is marked with its own strong-fit classification and is structurally immune to ever being overwritten by a future automatic pass — enforced by making the automatic labelling function itself a no-op once a session has been settled by hand, rather than trusting a future engineer to remember to check. Clearing a label requires no calendar access at all, checked by a test that fails if the correction path so much as consults the calendar — saying "this wasn't a meeting" should never depend on a permission grant existing.

## Considered options

- Scoring fit by how much of the meeting's duration the session covered — penalizes the ordinary case of a session running past its scheduled end.
- Letting the meeting-fit classification change which meeting gets selected — kept strictly separate; conflating the two would regress a selection rule already proven correct on its own.
- Leaving the attendee roster available regardless of fit strength, relying on future callers to check fit themselves — a convention, not a guarantee, against a contamination failure mode with real prior-art evidence.

