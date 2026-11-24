#!/usr/bin/env bash
set -euo pipefail

# cargo-release `pre-release-hook` for both workspaces (see [workspace.metadata.release] in
# Cargo.toml and [package.metadata.release] in crates/meethook-record/Cargo.toml).
#
# The repo has two Cargo.lock files that both record the version of the path crates they share:
# the root's lists meethook-record, and meethook-record's lists meethook-session. cargo-release
# only re-locks the workspace it is releasing, so after either bump the other lockfile is stale:
# the root release aborts on an uncommitted Cargo.lock, or the tag lands with a stale
# crates/meethook-record/Cargo.lock. Re-locking both here, before cargo-release commits, puts
# both lockfiles in the release commit. `-w` only refreshes the workspace members' own entries.

root="$(git rev-parse --show-toplevel)"
cargo update -w --manifest-path "$root/Cargo.toml"
cargo update -w --manifest-path "$root/crates/meethook-record/Cargo.toml"
