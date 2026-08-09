---
name: wayfinder
description: Plan a huge chunk of work — more than one agent session can hold — as a shared map of decision tickets in Backlog.md, and resolve them one at a time until the way to the destination is clear.
disable-model-invocation: true
---

A loose idea has arrived — too big for one agent session, and wrapped in fog: the way from here to the **destination** isn't visible yet. Wayfinding is about finding that way, not charging at the destination. This skill charts the way as a **shared map** in Backlog.md, then works its **decision tickets** — questions whose resolution is a decision, not slices of a build to execute — one at a time until the route is clear.

The destination varies per effort, and naming it is the first act of charting — it shapes every ticket. It might be a spec to hand off and iterate on, a decision to lock before planning starts, or a change made in place like a data-structure migration. The map is domain-agnostic — engineering work, course content, whatever fits the shape.

## Plan, don't do

Wayfinder is **planning** by default: each ticket resolves a decision, and the map is done when the way is clear — nothing left to decide before someone goes and does the thing. The pull to just do the work is usually the signal you've reached the edge of the map and it's time to hand off. An effort can override this in its **Notes** — carrying execution into the map itself — but absent that, produce decisions, not deliverables.

## Refer by name

Every map and ticket is a task, so it has a **name** — its title. In everything the human reads — narration, the map's Decisions-so-far — refer to it by that name, never by a bare id. A wall of `TASK-4, TASK-5, TASK-6` is illegible; names read at a glance. The id doesn't vanish — a name carries it — but it rides _inside_ the name, never stands in for it:

> `- **Pick the storage engine** (\`TASK-4\`) — Postgres; the write path needs transactions.`

Never `- TASK-4 — Postgres`. `backlog task view TASK-4 --plain` is how a reader zooms in.

## The Map

The map is a single Backlog task labelled `wayfinder:map` — the canonical artifact. Its tickets are subtasks of the map, and take a dotted id beneath it — map `TASK-4`, tickets `TASK-4.1`, `TASK-4.2` — so a ticket visibly belongs to its map.

The map task holds `In Progress` for the life of the effort, and moves to `Done` when the way is clear and no tickets remain. It is never assigned — assignment is what claims a *ticket*, and the map is claimed by nobody. It never appears in its own frontier, which is scoped to its children.

The map is an **index**, not a store. It lists the decisions made and points at the tickets that hold their detail; a decision lives in exactly one place — its ticket — so the map never restates it, only gists it and links.

### The map body

The whole map at low resolution, loaded once per session. Open tickets are **not** listed — they are open subtasks, found by the frontier query.

The body is split across two Backlog fields, because they have different write semantics:

| Map section | Lives in | Written with |
|---|---|---|
| Destination, Notes, Not yet specified, Out of scope | task **description** | `-d` (read the current description first — `-d` replaces the whole field) |
| Decisions so far | task **Implementation Notes** | `--append-notes` (append-only, so concurrent sessions can't clobber each other) |

Decisions-so-far is append-only by nature — one line per closed ticket — so it belongs in the field that appends. The other four sections get rewritten as fog graduates and scope shifts, so they belong in the field that replaces.

The description:

```markdown
## Destination

<what reaching the end of this map looks like — the spec, decision, or change this effort is finding its way to. One or two lines; every session orients to it before choosing a ticket.>

## Notes

<domain; skills every session should consult; standing preferences for this effort>

## Not yet specified

<!-- see "Fog of war": in-scope fog you can't ticket yet; graduates as the frontier advances -->

## Out of scope

<!-- see "Out of scope": work ruled beyond the destination; closed, never graduates -->
```

And the Implementation Notes accumulate the index:

```markdown
- **<closed ticket title>** (`<id>`) — <one-line gist of the answer>
```

### Tickets

Each ticket is a **subtask** of the map; the Backlog task id is its identity. Its description is the question, sized to one 100K token agent session:

```markdown
## Question

<the decision or investigation this ticket resolves>
```

Each ticket carries a `wayfinder:<type>` label — one of `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, `wayfinder:task` (see [Ticket Types](#ticket-types)).

A session **claims** a ticket by assigning it to the dev driving the map and moving it to `In Progress`, **first**, before any work, so concurrent sessions skip it. That assignee _is_ the claim: a `To Do`, unassigned ticket is unclaimed. Use one stable handle per human across the whole map — derive it from `git config user.name` and reuse it — because concurrent sessions read each other only through this field.

Blocking uses Backlog's **native** `--depends-on` — essential because it renders the frontier _visually_ on `backlog board`, so the human sees what's takeable without opening the map. A ticket is **unblocked** when every ticket it depends on is `Done`; the **frontier** is the open, unblocked, unclaimed subtasks — the edge of the known.

The answer isn't part of the description — it's recorded on resolution (see [Work through the map](#work-through-the-map)). Assets created while resolving a ticket are linked from the task with `--ref` or `--doc`, not pasted in.

## Ticket Types

Every ticket is either **HITL** — human in the loop, worked _with_ a human who speaks for themselves — or **AFK**, driven by the agent alone. A HITL ticket only resolves through that live exchange; the agent never stands in for the human's side of it (a grilling agent that answers its own questions has broken this).

- **Research** (AFK): Reading documentation, third-party APIs, or local resources like knowledge bases to surface a fact a decision waits on. Resolved by a `/research` **subagent**. Use when knowledge outside the current working directory is required.
- **Prototype** (HITL): Raise the fidelity of the discussion by making a cheap, rough, concrete artifact to react to — an outline, a rough take, a stub, or UI/logic code via the /prototype skill. Links the prototype as an asset. Use when "how should it look" or "how should it behave" is the key question.
- **Grilling** (HITL): Conversation. The default case. Always invoke the /grilling and /domain-modeling skills.
- **Task** (HITL or AFK): Manual work that must happen before a _decision_ can be made — nothing to decide, prototype, or research, but the discussion is blocked until it's done. Signing up for a service so its API can be judged, provisioning access, moving data so its shape can be seen. This is the one type that _does_ rather than decides — and it earns its place by unblocking a decision, not by delivering the destination. The agent drives it alone where it can (AFK); otherwise it hands the human a precise checklist (HITL). Resolved when the work is done; the answer records what was done and any resulting facts (credentials location, new URLs, row counts) later tickets depend on.

## Fog of war

The map is _deliberately_ incomplete: don't chart what you can't yet see. Beyond the live tickets lies the **fog of war** — the dim view of decisions and investigations you can tell are coming but can't yet pin down, because they hang on questions still open. Resolving a ticket clears the fog ahead of it, graduating whatever's now specifiable into fresh tickets — one at a time, until the way to the destination is clear and no tickets remain.

The map's **Not yet specified** section is where that dim view is written down: the suspected question, the area to revisit later. It's the undiscovered frontier _toward_ the destination — everything here is in scope, just not sharp enough to ticket. Write as loosely or as fully as the view allows; it doubles as a signpost for collaborators reading where the effort is headed.

**Fog or ticket?** The test is whether you can state the question precisely now — _not_ whether you can answer it now.

- **Ticket when** the question is already sharp — even if it's blocked and you can't act on it yet.
- **Not yet specified when** you can't yet phrase it that sharply. Don't pre-slice the fog into ticket-sized pieces: it's coarser than a ticket, and one patch may graduate into several tickets, or none, once the frontier reaches it.

**Not yet specified** excludes what's already decided (Decisions so far), what's already a live ticket, and what's out of scope (the next section).

## Out of scope

Fog only ever gathers _toward_ the destination. The destination fixes the scope, so work beyond it is **out of scope** — it isn't fog, and it doesn't belong in **Not yet specified**. It gets its own **Out of scope** section on the map: work you've consciously ruled out of _this_ effort. Scope, not sharpness, lands it here.

Out-of-scope work never graduates — the frontier stops at the destination — so it returns only if the destination is redrawn, and then as a fresh effort, not a resumption.

Ruling something out of scope is a scoping act, not a step on the route. When a ticket that already exists turns out to sit past the destination — mis-scoped in while charting, or exposed by a resolution — **close it** (a `Done` ticket is unambiguously off the frontier) and leave one line in the **Out of scope** section: the gist plus why it's out of scope, naming the closed ticket. It stays out of **Decisions so far**, which records the route actually walked — a scope boundary isn't a step on it.

## Invocation

Two modes. Either way, **never resolve more than one ticket per session** — with the exception of research tickets.

### Chart the map

User invokes with a loose idea.

1. **Name the destination.** Run a `/grilling` and `/domain-modeling` session to pin down what this map is finding its way to — the spec, decision, or change. The destination fixes the scope, so it's settled first.
2. **Map the frontier.** Grill again, **breadth-first** this time: fan out across the whole space rather than deep on any one thread, surfacing the open decisions and the first steps takeable now. **If this surfaces no fog** — the way to the destination is already clear, the whole journey small enough for one session — you don't need a map. Stop and put it to the user via **AskUserQuestion**: proceed without a map, or chart one anyway.
3. **Create the map**: Destination and Notes filled in, Decisions-so-far empty, the fog sketched into **Not yet specified**.
4. **Create the tickets you can specify now** as subtasks of the map — then wire blocking edges in a **second pass** (tasks need ids before they can reference each other). Wiring sorts them into the frontier and the blocked; everything you can't yet specify stays in the fog — the **Not yet specified** section.
5. **Fire the research subagents.** For each `wayfinder:research` ticket you just created, spin up a `/research` subagent to resolve it in parallel, capturing its findings as a Backlog doc with a context pointer from the ticket.
6. Stop — charting is one session's work; it hand-resolves nothing.

### Work through the map

User invokes with a map (title or id). A ticket is **optional** — without one, you pick the next decision, not the user.

1. Load the **map** — the low-res view, not every ticket body.
2. Choose the ticket. If the user named one, use it. Otherwise take the first frontier ticket in order. **Claim it** before any work — assigned to the dev driving the session, not to the agent.
3. Resolve it — **zoom as needed**: fetch the full body of any related or closed ticket on demand; invoke the skills the `## Notes` block names. If in doubt, use `/grilling` and `/domain-modeling`.
4. Record the resolution: post the answer as the ticket's **final summary**, move it to `Done`, and **append a context pointer** to the map's Decisions-so-far.
5. Add newly-surfaced tickets (create-then-wire); graduate any fog the answer has made specifiable, clearing each graduated patch from **Not yet specified** so it lives only as its new ticket. If the answer reveals a ticket — this one or another — sits beyond the destination, **rule it out of scope** rather than resolving it on the route. If the decision invalidates other parts of the map, update those tickets, or **archive** the ones that should never have existed.

Ruling out of scope and archiving are different acts, and Backlog records them differently. A ticket ruled out of scope was a real question you consciously placed past the destination — it stays as a `Done` task so the map can point at it. An **invalidated** ticket is one the route never walked and never will: a question the resolution dissolved, so there's nothing to record. That one gets `backlog task archive`, which takes it out of every listing. Never archive a resolved or out-of-scope ticket.

The user may run unblocked tickets in parallel, so expect other sessions to be editing Backlog concurrently.

## Backlog operations

Every read and write goes through the `backlog` CLI — never edit the markdown under `backlog/` by hand, or the CLI's metadata, ids, filenames, and relationships drift. Run `backlog <command> --help` before an unfamiliar command; help prints the full input schema.

Never invent an id. Use only ids returned by `backlog task create`, `backlog task list`, or `backlog task view` — the CLI assigns and echoes them, accepts loose forms, and normalizes (`task-001` and `TASK-1` both resolve). `auto_commit` is on, so each write lands as its own git commit: prefer one command with several flags over several commands.

**Create the map** — then move it to `In Progress`, where it stays for the life of the effort:

```bash
backlog task create "<effort name>" -l wayfinder:map -s "In Progress" \
  -d "<Destination / Notes / Not yet specified / Out of scope>"
```

**Load the map** — by id when the user gave one, by label when they gave a title or nothing:

```bash
backlog task list --labels wayfinder:map --plain     # every map; pick by title
backlog task view <map-id> --plain                   # the low-res view: description + Decisions-so-far
```

Read the map and stop there. Do **not** pull every ticket body — that's the whole point of the map being low-resolution. Zoom into individual tickets with `backlog task view <ticket-id> --plain` only as a resolution needs them.

**Create a ticket** — `-p` is the map id from the step above; add no acceptance criteria (see [Decision tickets vs. build tasks](#decision-tickets-vs-build-tasks)):

```bash
backlog task create "<question title>" -p <map-id> -l wayfinder:grilling -d "## Question

<the decision this ticket resolves>"
```

**Wire blocking** — the second pass, once every ticket has an id:

```bash
backlog task edit <blocked-ticket-id> --depends-on <blocker-id>,<blocker-id>
```

**Frontier query** — each flag carries one clause of the definition: `--parent` = children of this map, `--status "To Do"` = open, `--ready` = every dependency completed, `--unassigned` = unclaimed. First in the list wins.

```bash
backlog task list --parent <map-id> --status "To Do" --ready --unassigned --sort id --plain
```

`--sort id` makes "first in the list" deterministic and equal to map order, since ids are assigned in creation order.

**Claim** — the session's first write:

```bash
backlog task edit <ticket-id> -s "In Progress" -a @<dev-handle>
```

`<dev-handle>` is the stable per-human handle from [Tickets](#tickets) — `git config user.name` — never an agent name.

**Resolve** — then append the context pointer to the map:

```bash
backlog task edit <ticket-id> \
  --final-summary "<the answer — the decision and its one-line why>" \
  --append-notes "<rationale, alternatives weighed, facts the decision rested on>" \
  -s "Done"
backlog task edit <map-id> --append-notes "- **<ticket title>** (\`<ticket-id>\`) — <one-line gist>"
```

Backlog has three prose fields; use them as its own finalization guide splits them:

- **Final Summary** — the answer. This is the canonical resolution; a resolved ticket without one is not resolved.
- **Implementation Notes** — the supporting rationale, when the answer needs more than a paragraph.
- **Comments** (`--comment`) — in-flight discussion and review questions only, *not* the answer.

**Rewrite the map description** — graduating fog out of **Not yet specified**, or adding a line to **Out of scope**. `-d` replaces the whole field, so this is always read-modify-write:

```bash
backlog task view <map-id> --plain          # read the current description
backlog task edit <map-id> -d "<the full four sections, edited>"
```

Never write a partial description — the three sections you aren't touching must be carried through verbatim, or they're lost. Decisions-so-far is untouched by this: it lives in Implementation Notes and only ever appends.

**Archive an invalidated ticket** — one the route will never walk, because the resolution dissolved the question:

```bash
backlog task archive <ticket-id>
```

This removes it from every listing. Use it *only* for invalidated tickets — never for a resolved one, and never for an out-of-scope one, which the map still points at.

**Rule a ticket out of scope** — Backlog has no `wontfix`, so close it and mark it. `Done` puts it off the frontier; the label keeps it distinguishable from a ticket that was actually walked. Don't archive it — Backlog's guidance is that terminal-status work stays put until `backlog cleanup`.

```bash
backlog task edit <ticket-id> -s "Done" --add-label wayfinder:out-of-scope \
  --final-summary "Out of scope: <why it sits past the destination>"
```

**Assets**

- Research findings go to `backlog doc create "<question>" -p research`, linked from the ticket with `backlog task edit <ticket-id> --doc <doc-id>`.
- Prototypes stay on a throwaway branch; link the branch or file with `backlog task edit <ticket-id> --add-ref <branch-or-path>`. Use `--add-ref`, not `--ref` — `--ref` replaces every existing reference.
- If a resolution is a hard-to-reverse architectural decision, also record it with `backlog decision create` — see `/domain-modeling`.

### Decision tickets vs. build tasks

`backlog instructions task-finalization` requires objective verification evidence before checking an acceptance criterion, and a final summary naming that evidence. That fits build work. A wayfinder ticket is a **question**, and its evidence is different in kind:

- **Give wayfinder tickets no acceptance criteria.** An AC on a decision ticket only restates the question — cost without value. With none defined, the finalization checklist's AC step is satisfied vacuously, and `definition_of_done` is empty in `backlog/config.yml`, so nothing is being skipped.
- **The final summary is the verification.** For a HITL ticket (`grilling`, `prototype`, `task`) the evidence is the human's recorded decision — the agent never stands in for it. For an AFK `research` ticket it's the cited primary source. Name that evidence the way build work names a passing test.
- **`backlog instructions task-execution` step 6 wants a `--plan` before implementation.** A decision ticket implements nothing; skip the plan. Record what you learn in `--append-notes` as you go.

Everything else in the Backlog guides — read before mutate, claim before work, one subtask at a time, no silent scope expansion, no unapproved follow-up tasks — applies to wayfinder tickets unchanged, and matches wayfinder's own rules.
