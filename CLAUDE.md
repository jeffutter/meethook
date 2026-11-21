# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project and architecture guidance lives in [`AGENTS.md`](./AGENTS.md) — read it before
working in this repository.

## Ticket tracking

This project tracks work with scree, project key `MH`
(tickets are `MH-12`, subtasks `MH-12.3`). Use the `scree` skill for the work loop, and
`scree list`, `scree show MH-12` and `scree add` for everything else.

- Create tickets as `draft` with at least one acceptance criterion; a human makes them `ready`.
- Skills that publish tickets (`to-spec`, `to-tickets`) and the issue tracker they ask about:
  the tracker is scree, not local files or GitHub. Publish with `scree add`, blockers first,
  using `--blocked-by` for dependencies and `--parent` for a subtask of a top-level ticket.
  scree has no labels, so skip any triage label.
- Tickets migrated from Backlog.md say "was TASK-n" in their description, and
  `backlog/TASK-ID-MAP.md` maps every old id to its scree id.
- Backlog.md is no longer used. Do not run the `backlog` CLI. `backlog/decisions/` and
  `backlog/docs/` are plain markdown that stay as the design record; `backlog/tasks/` and
  `backlog/archive/` are old local files, ignored by git, kept only until the user deletes them.
