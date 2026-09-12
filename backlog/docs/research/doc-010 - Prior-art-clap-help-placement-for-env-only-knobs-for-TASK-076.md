---
id: doc-010
title: 'Prior art: naming an env-only knob in --help (clap placement, value semantics, test pinning) for TASK-076'
type: research
created_date: '2026-09-12 13:20'
updated_date: '2026-09-12 13:20'
---

## Purpose

Gathered for planning `task-076 - Name MEETHOOK_CPU in meethook transcribe --help`. Prior art plus the
measured clap facts that decide between the ticket's option 1 (help text) and option 3 (visible
`--cpu`). Not a plan. The scratch crate used for the measurements lived in `/tmp/clapprobe`
(clap 4.6.6, features `derive` + `env`, i.e. what the workspace pins at `Cargo.toml:73`).

## clap cannot render an env-only variable; the maintainer closed the request

- [clap-rs/clap#6039](https://github.com/clap-rs/clap/issues/6039) asked for exactly this — an
  `Environment Variables:` help section for a variable with no flag (`RUST_LOG` was the example), with
  three proposed mechanisms. epage **closed it as wontfix**: "I would consider it a misuse to hide an
  argument for the sake of having the environment variable read", redirecting to the broader
  config-sources thread [#2763](https://github.com/clap-rs/clap/issues/2763). So there is no supported
  rendering to wait for, and nothing on the roadmap to wait on.
- The rust-lang users thread
  ["Clap: Only Show Arg as Environment Variable"](https://users.rust-lang.org/t/clap-only-show-arg-as-environment-variable/130912)
  tried `hide(true)` + `hide_env(false)` and concluded the same: "it hides the argument altogether and
  displays nothing … I might have to defer to a custom text block."

## Measured: where each placement actually lands

All four rows are runs of the scratch crate, not documentation reading.

| Placement | `transcribe -h` | `transcribe --help` | root `-h`/`--help` | `help transcribe` |
| --- | --- | --- | --- | --- |
| `after_help` on `Cli` (top level) | absent | absent | present | absent |
| Extra doc-comment paragraph on the `Transcribe` variant | absent | present | absent | present |
| `#[command(after_help = …)]` on the `Transcribe` variant | **present** | **present** | absent | **present** |
| `#[arg(hide = true, env = …)]` hidden carrier arg | absent | absent | absent | absent |

Two consequences for the ticket's framing:

- An epilogue on `Cli` does **not** satisfy "name it in `meethook transcribe --help`". clap has no
  propagation for `after_help` (only `propagate_version` exists), so a top-level epilogue is reachable
  only from `meethook --help`. Verified: root help printed the epilogue, subcommand help did not.
- A subcommand-level `after_help` is the only text placement that reaches both `-h` and `--help`. A
  doc-comment paragraph reaches only `--help`, because clap derive splits the first line into `about`
  and the rest into `long_about` — visible today in this repo: `meethook record --help` prints all
  three doc paragraphs, `meethook transcribe -h` prints one line plus "(see more with '--help')".
- Option 2 is worse than the ticket says, in one way better and one way worse than written. Better: it
  renders *nothing* anywhere (grep for the hidden var name across root `-h`, root `--help`,
  `transcribe -h` = zero hits), so the claim that it "gets clap's `[env: …]` rendering for free" is
  false — it buys no discoverability at all and is strictly a decoy. Worse: it still participates in
  error suggestions — `probe --cpu-hid` printed `tip: a similar argument exists: '--cpu-hidden'`, so
  the pseudo-flag leaks precisely where a user is already confused. Upstream agrees hiding is not a
  help tool: native completions needed new work to show hidden flags at all
  ([#5283](https://github.com/clap-rs/clap/pull/5283),
  [#5583](https://github.com/clap-rs/clap/pull/5583)). Note that meethook ships **no** shell
  completions (`clap_complete` appears nowhere in the workspace), so that half of the stated downside
  is moot here.

## Measured: clap's `env` binding contradicts MEETHOOK_CPU's shipped value semantics

This is the trap in option 3 if the flag carries `env = "MEETHOOK_CPU"` rather than leaving the
variable to `gpu.rs`. A derived `bool` + `env` makes the variable an *option value*, parsed against
`[possible values: true, false]`:

| `MEETHOOK_CPU` | today's `opted_in` (`gpu.rs:129-141`) | clap `bool` + `env` | clap `BoolishValueParser` + `ArgAction::Set` |
| --- | --- | --- | --- |
| unset | off | off | off |
| `1` | **on** | `error: invalid value '1'` | on |
| `0` | **on** (README.md:228 says "`0` counts") | `error: invalid value '0'` | **off** — inverted |
| `false` | **on** (any non-empty) | off | off |
| empty | off (unset) | `error: a value is required` | error |

No clap value parser reproduces the shipped contract, because "any non-empty value including `0`" and
"boolean-ish" disagree on `0`, `1`, `false` and empty simultaneously
([BoolishValueParser docs](https://docs.rs/clap/latest/clap/builder/struct.BoolishValueParser.html);
the upstream history of this exact confusion is
[#2539](https://github.com/clap-rs/clap/issues/2539) /
[PR #2664](https://github.com/clap-rs/clap/pull/2664) /
[#1649](https://github.com/clap-rs/clap/issues/1649)). A visible `--cpu` therefore has to stay
decoupled from the env var — flag presence OR a separately-read `MEETHOOK_CPU`, with the flag→env
precedence written by hand — or the release breaks documented behaviour. Setting `MEETHOOK_CPU=1` does
not disturb help rendering (`MEETHOOK_CPU=1 meethook transcribe --help` prints normally), so there is
no interaction risk in the text-only option.

## Prior art on the decision itself: env-only vs flag, and where the text goes

- **clig.dev keeps these separate by purpose**: "Environment variables are for behavior that varies
  with the context in which a command is run", while flags are for what "likely [varies] from one
  invocation to the next" ([clig.dev §Configuration](https://clig.dev/)). A sandbox-hides-the-GPU
  workaround is per-machine/per-environment context, which is the env-var side of that line. The same
  document's counterweight is discoverability: keep knobs discoverable, since "hidden knobs are
  untestable knobs" — consistent with keeping it env-only *and* naming it in help.
- **The value semantics already match a real convention.** [force-color.org](https://force-color.org/)
  specifies `FORCE_COLOR` as "present and not an empty string (regardless of its value)" — verbatim
  `opted_in`. Same convention as `NO_COLOR`. So "any non-empty value, `0` included" needs no apology in
  help text; it is the well-known env boolean, and precedent says say-so-in-one-line rather than add a
  flag.
- **Device-hiding is done env-only elsewhere in this exact domain.** `CUDA_VISIBLE_DEVICES=""` is the
  canonical way to hide every GPU from a CUDA program
  ([NVIDIA env-var reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)),
  and llama.cpp documents it as a peer mechanism to its own `--device` flag ("As for any CUDA program,
  the environment variable `CUDA_VISIBLE_DEVICES` can be used… Use `--device` for selecting GPUs from
  among those visible", [docs/multi-gpu.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md)).
  Useful both ways: the ML world accepts an env-only device veto, *and* the tools that offer both
  document them in one table row — llama.cpp renders `--device … (env: LLAMA_ARG_DEVICE)`
  ([tools/cli/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)) — which
  is the shape clap gives meethook's `--root`/`--template` today. Caveat worth carrying into wording if
  a flag ever appears later: `CUDA_VISIBLE_DEVICES` inverts meethook on the empty case (empty string =
  hide all GPUs, whereas empty `MEETHOOK_CPU` = unset).
- **Real Rust CLIs use the epilogue for exactly this.** rustup sets `before_help`/`after_help` on its
  top-level `Cli` and names `RUSTUP_LOG` inside the option help
  ([src/cli/rustup_mode.rs](https://github.com/rust-lang/rustup/blob/4b2c0919/src/cli/rustup_mode.rs));
  uv manages `after_help`/`after_long_help` deliberately when it took over its own help
  ([commit 5f20bdb](https://github.com/astral-sh/uv/commit/5f20bdb2ee1207fb7617360335fbfdd23aba6dd1));
  tractor injects `after_help` + `after_long_help` through a small `CommandExt` trait precisely because
  derive attributes cannot hold generated text
  ([tractor/src/cli/help.rs](https://github.com/boukeversteegh/tractor/blob/main/tractor/src/cli/help.rs)).
  Non-clap ecosystems formalize the placement too: cmdliner generates a dedicated `ENVIRONMENT` man
  section for variables that are not options
  ([cmdliner docs](https://erratique.ch/software/cmdliner/doc/cli)).
- **Naming it outside the Options list is sanctioned, burying it in prose is not.** Fuchsia's CLI help
  requirements list "environment variables used, other than those already listed in Options" under
  "What not to put in the Description section", and say to "provide this in Options or Notes"
  ([cli_help.md](https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/docs/development/api/cli_help.md)).
  Read against clap that favors a distinct labelled epilogue block (`ENVIRONMENT` + the name) over
  weaving the variable into existing descriptive prose — plus a pointer out of wherever the user meets
  the failure.
- **bat names its env-only knobs inside long help prose**, e.g. "BAT_STYLE environment variable", and
  states precedence explicitly ("The command-line arguments are the highest priority, followed by the
  BAT_STYLE environment variable") — the closest widely-used Rust precedent for text-only env
  documentation, and a reminder to say what wins when both exist.

## Pinning the string: fits the tests this repo already has

- Still true as of HEAD (`de48df1`): no test asserts help output; the four binary-spawning tests use
  real subcommands. The established idiom is plain `std::process::Command::new(env!("CARGO_BIN_EXE_meethook"))`
  with `Stdio::piped()` and `stdin(Stdio::null())`
  (`crates/meethook/tests/sessions_report.rs:98-110`), with **no** `assert_cmd`/`predicates`/`trycmd`
  dev-dependency in `crates/meethook/Cargo.toml`. A substring assertion in that idiom adds no
  dependency; `trycmd` snapshots (the usual recommendation for help drift, `TRYCMD=overwrite`,
  [docs](https://docs.rs/trycmd/latest/trycmd/)) would be a new dev-dep plus a whole snapshot corpus
  for one string.
- Width fragility is smaller than feared and asymmetric: with stdout piped, clap ignored `COLUMNS=40`
  in measurement (max line length identical at 40 and default), so captured help is deterministic —
  but raw `after_help`/`long_about` text is emitted **verbatim, unwrapped**, so a long epilogue line
  will not reflow on a narrow terminal. Hard-wrap the epilogue in source if it matters.

## Repo-side facts the plan inherits

- TASK-074 landed (`de48df1`), so the wording already exists twice in prose: README.md:228 (a
  `transcribe`-section paragraph mirroring the failure text) and the README.md:323 Global options row,
  plus LINUX.md:17. Adding help text makes five places that describe one knob, counting the error text
  at `gpu.rs:75` and the runtime confirmation at `commands.rs:70-76` — the drift argument for one
  shared constant rather than a fifth literal copy.
- Precedent status quo: nothing in the workspace uses `after_help`/`before_help`/`long_help`; `main.rs:88`
  sets `long_about = None`, and every multi-paragraph explanation in today's help comes from doc
  comments on variants and fields (see `--template`'s rendered rationale). Whichever placement wins here
  is the pattern for the next env-only knob — and `MEETHOOK_CALENDAR_DEBUG`, `MEETHOOK_ACTIVITY_DEBUG`,
  `MEETHOOK_TIMING_DEBUG` are the obvious next candidates, none of which belongs to `transcribe`.
