---
name: grilling
description: Grill the user relentlessly about a plan, decision, or idea. Use when the user wants to stress-test their thinking, or uses any 'grill' trigger phrases.
---

Interview the user relentlessly until you reach a shared understanding. Map this as a **design tree**: every decision branches into the decisions that hang off it.

Work the tree in **rounds**. The **frontier** is every decision whose prerequisites are already settled — the questions you can ask _now_ without guessing at answers you haven't heard yet. Ask the whole frontier in one round through the **AskUserQuestion** tool — never as free-text prose — so it renders cleanly and the user can answer in one click.

Turn each frontier question into an AskUserQuestion entry:

- `header` — a short (≤12 char) label for the question.
- `question` — the full question; multiple paragraphs are fine.
- `options` — your recommended answer first, its label ending in "(Recommended)", plus any other real alternatives worth naming (2-4 total). The tool always adds a free-text "Other", so you don't need to enumerate every possibility — just the live contenders.

AskUserQuestion takes at most 4 questions per call. When a round's frontier has more than 4, split it across consecutive calls rather than dropping questions or falling back to prose.

Each round the user answers reshapes the tree — settled decisions push the frontier outward and unblock questions that depended on them. Recompute the frontier and ask the next round. A question whose answer depends on another question still open in this round belongs to a _later_ round, not this one.

Finding _facts_ is your job, never the user's. When a frontier question needs a fact from the environment (filesystem, tools, etc.), dispatch a sub-agent to find it — don't ask the user for anything you could look up yourself. Don't block on it: a running exploration is an unsettled prerequisite, so only the questions downstream of it wait for the sub-agent to report — ask the rest of the frontier now via AskUserQuestion. The _decisions_ are the user's — put each to them and wait.

The session is done when the frontier is empty: every branch of the design tree visited, nothing left silently assumed. Do not act on it until the user confirms you have reached a shared understanding.
