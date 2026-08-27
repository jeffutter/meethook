---
id: decision-010
title: Transcript rendering
date: '2026-08-27 04:40'
status: accepted
---
`transcript.md` is rendered through a template loaded at runtime — resolved from a `--template` flag or environment variable, then a file in the session root, then a built-in default — using the `minijinja` crate rather than a compile-time engine (ruled out outright, since a runtime-loaded template can't work with one) or the more feature-complete `tera` (whose default build pulls in a directory-glob template loader unneeded here, for one known template path, along with several dependencies that come with it). The template override is deliberately a global CLI setting rather than scoped to the `transcribe` command alone, because `enroll` and `forget` both re-render an existing transcript after a rename and must re-render it with the same template it was originally built from — a command-scoped override would let those commands silently revert a transcript to the default template the next time either ran. Meeting titles are interpolated into the rendered file's YAML frontmatter through a small hand-written filter rather than the templating engine's own JSON-safe filter, after the built-in filter's HTML-escaping was found to mangle an ampersand in a real meeting title into unreadable YAML.

Consecutive turns from the same speaker are visually collapsed under one timestamp by adding a second, render-only view of the transcript data alongside the existing one, rather than reshaping the underlying transcript record itself — preserving a stronger guarantee than the feature otherwise needed: a custom template that doesn't use the new view renders exactly as before, `transcript.json`'s on-disk shape can't move, and re-rendering after a name correction re-groups automatically without any additional code path.

## Considered options

- A compile-time templating engine — incompatible outright with a template resolved and loaded at runtime.
- `tera` instead of `minijinja` — its default feature set assumes a directory of templates to search, not one known path.
- Scoping the template override to `transcribe` alone — would let `enroll`/`forget` silently revert a transcript to the default on their next re-render.
- Reshaping the transcript's own data structure to carry speaker-turn groups — would move the on-disk JSON shape a separate guarantee depends on staying stable.

