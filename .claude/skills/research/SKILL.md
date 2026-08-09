---
name: research
description: Investigate a question against high-trust primary sources and capture the findings as a Backlog document. Use when the user wants a topic researched, docs or API facts gathered, or reading legwork delegated to a background agent.
---

Spin up a **background agent** to do the research, so you keep working while it reads.

Its job:

1. Investigate the question against **primary sources** — official docs, source code, specs, first-party APIs — not a secondary write-up of them. Follow every claim back to the source that owns it.
2. Write the findings as a Backlog document, citing each claim's source:
   ```bash
   backlog doc create "<the question>" -p research --plain
   backlog doc update <doc-id> --content "<markdown findings>"
   ```
3. Report the doc id back. If the research was commissioned by a task, link it: `backlog task edit <task-id> --doc <doc-id>`.

Never write to `backlog/docs/` by hand — the CLI owns the id, filename, and metadata.
