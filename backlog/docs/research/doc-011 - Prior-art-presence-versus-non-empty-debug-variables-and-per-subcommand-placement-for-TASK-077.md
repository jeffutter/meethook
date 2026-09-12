---
id: doc-011
title: 'Prior art: presence-versus-non-empty debug variables, and where a variable honoured by two subcommands gets named, for TASK-077'
type: research
created_date: '2026-09-12 13:40'
updated_date: '2026-09-12 13:40'
---

## Purpose

Gathered for planning `task-077 - Name the record-side debug variables in meethook record --help`.
Clap placement was already measured in `doc-010` and is not re-derived here; this covers the two things
TASK-077 actually has to decide: how a variable that enables on *presence* is described anywhere in the
wild, and where a variable honoured by more than one subcommand gets named. Not a plan. Citations are
to sources fetched during research, not to memory.

## 1. There is still nothing to wait for from clap

- [clap-rs/clap#6039](https://github.com/clap-rs/clap/issues/6039) remains closed *not planned*
  (epage closed it same day it was opened: "a misuse to hide an argument for the sake of having the
  environment variable read"). Nothing newer in the 4.x line adds env-only help rendering; the recent
  env-related work is man-page output gated behind the `env` feature
  ([dd1fcd3](https://github.com/clap-rs/clap/commit/dd1fcd3d4be0fcc2d3eede07df4d9ec25663241c)), which
  meethook does not generate. The rust-lang users thread on it lands on the same conclusion: "I might
  have to defer to a custom text block"
  ([users.rust-lang.org](https://users.rust-lang.org/t/clap-only-show-arg-as-environment-variable/130912)).
  Text epilogue remains the only route, so TASK-076's placement table is the whole answer.
- One adjacent trap upstream confirms rather than resolves: when a *real* arg carries `env`, clap reads
  the variable before help would even be consulted in some paths
  ([#5113](https://github.com/clap-rs/clap/issues/5113), closed not planned). Keeping the debug
  variables out of the parser avoids that class entirely, as doc-010 argued for `--cpu`.

## 2. The split the ticket points at is a real split in the wild

Two incompatible traditions exist, both mainstream, and mature tools disagree with themselves. That is
the argument for a per-variable sentence rather than a house rule stated once.

**Presence-based, empty included:**

- The `--usage`/`xdev` argv grammar states the reasoning best: "An environment variable set to the
  empty string is set. Treating empty as unset would make `EX_JOBS=` mean something no other empty
  value in the grammar means" ([usage.jdx.dev/spec/argv](https://usage.jdx.dev/spec/argv)).
- npm config: "Any environment configurations that are not given a value will be given the value of
  `true`" ([npm docs](https://docs.npmjs.com/cli/v12/using-npm/config/)).
- The Google-CLI-style parser documents exactly the clause meethook needs: "If the environment variable
  for a boolean argument is set to any value, it will be interpreted as `true`"
  ([cloud-copilot/cli docs](https://github.com/cloud-copilot/cli/blob/main/docs/EnvironmentVariables.md)).

**Non-empty required (the emerging cross-tool spec):**

- [no-color.org](https://no-color.org/) / [force-color.org](https://force-color.org/) specify "when
  present and not an empty string (regardless of its value)" — the strongest cross-tool statement of
  the *other* rule, and the one TASK-076 quoted for `MEETHOOK_CPU`.
- Even with a published spec, tools get the empty case wrong: Cpython had an open bug where empty
  `NO_COLOR`/`FORCE_COLOR` were honored, fixed in
  [python/cpython#129140](https://github.com/python/cpython/pull/129140)
  ([issue 129061](https://github.com/python/cpython/issues/129061)). An empty-string edge case is worth
  one clause of help text, not a wiki page.
- Terraform's older reference said `TF_LOG` "[i]f set to any value, enables detailed logs to appear on
  stderr" *and* "[t]o disable, either unset it or set it to empty"
  ([archived copy](https://docs.w3cub.com/terraform/configuration/environment-variables)): prose saying
  "any value" sitting on top of a rule where empty is special. Current Terraform docs have dropped that
  framing and name a value instead, "To disable, either unset it, or set it to `off`"
  ([official](https://developer.hashicorp.com/terraform/cli/config/environment-variables)). Either
  version is the sentence shape TASK-077 should avoid copying: `MEETHOOK_ACTIVITY_DEBUG=` turns
  diagnostics **on**, so "any value" plus a silent exception is precisely the wrong direction.
- GitHub CLI is truthy-plus-modes: "`GH_DEBUG`: set to a truthy value to enable verbose output on
  standard error. Set to `api` to additionally log details of HTTP traffic" and "`NO_COLOR`: set to any
  value to avoid printing ANSI escape sequences"
  ([gh manual](https://cli.github.com/manual/gh_help_environment)) — one tool, two different sentences,
  because the two variables genuinely behave differently. Same posture TASK-077 should take.
- Strict parsing is the far end of the spectrum and is not what these knobs are: git's Boolean env
  variables "take their values the same way as Boolean valued configuration variables"
  ([git(1)](https://git-scm.com/docs/git)), and pip famously trace-backed on an invalid Boolean env
  value ([pypa/pip#5616](https://github.com/pypa/pip/issues/5616)). A diagnostics switch that parses
  nothing refuses nothing; the help line is then the only place the contract exists, which raises the
  price of it being accurate.

Consequences for wording, given the code (`activity.rs:333`, `calendar/mod.rs:563`, `lib.rs:496`,
`track.rs:314` all `var_os(...).is_some()`/`is_none()`):

- Say presence, and say it including empty, in so many words: "`MEETHOOK_ACTIVITY_DEBUG` ... Setting it
  at all turns it on, to the empty string included." Do not reuse "any non-empty value counts, `0`
  included", which is true of `MEETHOOK_CPU` and false here.
- meethook now ships two different rules under one prefix. Nothing in the ecosystem forces one house
  rule, so either is defensible; but the help text should not let a reader infer the second rule from
  the first. Two options worth weighing at planning time: document the asymmetry explicitly, or spend a
  small behaviour change aligning the debug variables with `opted_in`'s non-empty rule (which the
  NO_COLOR tradition supports) and lose the ability to reproduce someone's `VAR=` invocation. That is a
  scope decision the ticket should make out loud, not inherit silently.
- README's Global options rows for both variables (README.md:322, :324) state a default of `unset` and
  never say what counts as setting them — unlike the `MEETHOOK_CPU` row, which states the rule. So help
  would be the first place the rule is written down anywhere, and the existing rows are the natural
  companion edit if the wording settles.

## 3. Where a variable honoured by more than one subcommand gets named

No ecosystem consensus says "duplicate it into every subcommand"; the well-known tools mostly centralize
and leave subcommand help alone.

- **GitHub CLI centralizes**: root `gh help` gained an `ENVIRONMENT VARIABLES` section listing variables
  that affect many commands ([cli/cli#1370](https://github.com/cli/cli/pull/1370)), and there is a whole
  separate topic, `gh help environment`, rendered online as
  [`gh_help_environment`](https://cli.github.com/manual/gh_help_environment). Per-subcommand help says
  nothing about them.
- **Git centralizes the same way**: one `ENVIRONMENT` section in git(1) introduces the family with
  "Various Git commands pay attention to environment variables", and individual commands' pages stay
  quiet ([git(1)](https://git-scm.com/docs/git); Pro Git's dedicated
  [Environment Variables](https://git-scm.com/book/en/v2/Git-Internals-Environment-Variables) page under
  Internals). The README Global options table already plays that role for meethook.
- **ripgrep names the var where the behavior is met, in prose**, not in a dedicated block:
  `RIPGREP_CONFIG_PATH` appears in the `--no-config` description ("When this flag is present, ripgrep
  will not respect the `RIPGREP_CONFIG_PATH` environment variable") and in the man page's configuration
  section ([rg(1)](https://manpages.debian.org/bookworm/ripgrep/rg.1.en.html)). Useful as the argument
  for attaching `MEETHOOK_CALENDAR_DEBUG` to the command whose confusing output it explains.
- Section labels in the wild are `ENVIRONMENT` (Fuchsia/cmdliner, cited in doc-010) and `ENVIRONMENT
  VARIABLES` (gh). Staying with `ENVIRONMENT:` keeps meethook internally consistent with TASK-076;
  there is no external cost to either.

The mechanical constraint stays whatever the choice: clap propagates nothing but `propagate_version`, so
two placements mean two references to one generated string. TASK-076's execution already established the
shape (`fn cpu_env_help() -> String` interpolating the crate's constant, passed as `after_help =
cpu_env_help()`); a shared `fn …_env_help()` used by both the `Record` and `Meeting` attributes is the
only way duplication cannot drift.

Testability differs sharply from TASK-076, and it is decided by platform gates rather than by clap:

- `Record` is `#[cfg(target_os = "macos")]` (`main.rs:155`), so `meethook record --help` does not exist
  off macOS; the assertion that names it must be `#[cfg(target_os = "macos")]`, matching
  `tests/record_single_instance.rs:20` rather than TASK-076's ungated test. CI builds the matrix on
  `macos-26` and `ubuntu-latest` (`ci.yml:16`), and the Linux job is precisely what catches the
  dead-code warning for an epilogue const referenced only from a macOS-only variant (`ci.yml:39` records
  that Clippy and Docs type-check per target).
- `meeting` exists everywhere, but off macOS the calendar seam returns an empty list outright
  (`commands.rs:627-641`), so `MEETHOOK_CALENDAR_DEBUG` cannot print anything there. If the calendar
  block goes on `Meeting` as well, its sentence needs the off-macOS clause TASK-076 used for the CPU
  ("changes nothing there"), which is what makes a Linux-runnable assertion honest instead of
  advertising a no-op.

## 4. Several debug topics, several variables: that is the mainstream shape

- Per-subsystem debug variables are the norm, not an accident: the `GIT_TRACE*` family (`GIT_TRACE`,
  `GIT_TRACE_PERFORMANCE`, `GIT_SETUP`, keyed categories) is documented as "inactive unless explicitly
  enabled by setting `GIT_TRACE*` environment variables"
  ([api-trace](https://git-scm.com/docs/api-trace)); `RUST_LOG` reaches the same place through
  `EnvFilter` directives ([tracing_subscriber::filter::Builder](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.Builder.html),
  [env_logger](https://docs.rs/env_logger/latest/env_logger/index.html)); `GH_DEBUG` puts one mode value
  on a single variable instead. Keeping `MEETHOOK_ACTIVITY_DEBUG` and `MEETHOOK_CALENDAR_DEBUG`
  separate is therefore conventional and needs no justification beyond itself.
- Count entries at execution time, not from this ticket's text: the record side actually has three such
  variables, since `MEETHOOK_TIMING_DEBUG` is read at `crates/meethook-record/src/lib.rs:496` and
  `src/track.rs:314`. It has no README row yet because TASK-075 (confirm on real hardware, then
  document) is still open and assigned to a human, so whether TASK-077 names two variables or three is a
  status question, not a code question.
- If a destination for the output is ever wanted, git is the reference implementation and it is richer
  than a boolean: `GIT_TRACE` accepts `"1"`, `"2"`, `"true"` (case-insensitive) for stderr, an integer
  3..9 as an already-open file descriptor, or an absolute path to append to
  ([git(1) man](https://man.archlinux.org/man/git.1)). Relevant here mainly as a rejection: meethook's
  diagnostics go to stderr with `2>activity.log` already documented (README.md:177), so a path-valued
  debug variable would duplicate shell redirection.

## 5. Privacy in diagnostic output: precedents back the calendar wording

- git's HTTP tracing redacts by default, and the *un*redacted legacy switch was retired for the reason
  that matters here: `GIT_CURL_VERBOSE` was changed to behave as `GIT_TRACE_CURL=1
  GIT_TRACE_CURL_NO_DATA=1` because it "redact[s] neither the `Authorization` header nor any cookies",
  "[t]his is to prevent inadvertent revelation of sensitive data"
  ([PATCH v2 3/3](https://public-inbox.org/git/f5a29e8fa1c97a80237b2f1414536c75a9911060.1589394456.git.jonathantanmy@google.com/);
  `GIT_TRACE_CURL_NO_DATA` itself is
  [PATCH v2 2/2](http://public-inbox.org/git/42366886098655a6bf712f55c6c0246e31218f6f.1516321355.git.jonathantanmy@google.com/);
  redaction logic in [http.c](https://github.com/git/git/blob/9d530dc0024503ab4218fe6c4395b8a0aa245478/http.c)).
- Meethook already made the deliberately non-leaky choice, and made it in code: the calendar diagnostics count
  attendees and omit notes and location entirely, precisely because an invite body is "the single most
  likely field here to contain a dial-in PIN" (`crates/meethook-record/src/calendar/mod.rs`, `summarize`
  and its pinned test). Since a debug switch's whole purpose is that its output gets pasted into a bug
  report, that property is user-facing knowledge, and git's history is the case for saying it in the
  help line ("attendee counts only, never names") rather than leaving it as a source-tree promise.

## Repo-side facts the plan inherits

- Read sites and their rule: `activity.rs:333` (`is_some()`), `crates/meethook/src/record.rs:826`
  (`is_some()`, in the CLI crate rather than the capture crate), `calendar/mod.rs:563` (`is_some()`),
  `lib.rs:496` and `track.rs:314` for TIMING. No site treats empty as unset.
- Both variables live in `meethook-record`, which is excluded from the root workspace and compiles only
  on macOS, so a shared constant like `CPU_ENV_VAR` (TASK-076 AC#2) cannot be reached from
  `crates/meethook/src/main.rs` on a Linux build. Either the constants are declared in the CLI crate, or
  the help text that interpolates them is macOS-gated — the latter mirrors TASK-076 most closely, and the
  existing pattern is `pub use` from the crate root (`meethook-transcribe/src/lib.rs:61`).
- `MEETHOOK_CALENDAR_DEBUG` is reachable from `record` and from `meeting`; `MEETHOOK_ACTIVITY_DEBUG`
  from `record` alone. Neither is honoured off macOS at all.
- Help-output assertions are new enough that only one exists
  (`crates/meethook/tests/help_names_the_cpu_fallback.rs`), and it sets the idiom: spawn
  `env!("CARGO_BIN_EXE_meethook")` with `stdin(Stdio::null())`, assert against the crate's constant
  rather than a literal, and pin both `-h` and `--help` plus the negative (root help stays silent).
