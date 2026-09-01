#!/usr/bin/env bash
set -euo pipefail

# Cuts a release across this repo's two Cargo workspaces.
#
# meethook-record roots its own workspace (crates/meethook-record/Cargo.toml explains why:
# its Apple-framework dependencies can't compile off macOS), so the root workspace's
# `cargo release` never reaches it. This runs meethook-record's release first, then the
# root's -- both to the same level/version, so meethook, its four library crates (which move
# together via `version.workspace = true`), and meethook-record all land on one version.
#
# meethook-record's [package.metadata.release] sets tag = false and push = false, so its
# version-bump commit just lands on `main` locally; the root's release is what tags (a single
# `vX.Y.Z`, consolidated across its five crates) and pushes, carrying meethook-record's commit
# along with it. That ordering is why this script exists rather than two ad hoc commands: run
# them the other way around, or independently, and you either push meethook-record's commit
# under no tag at all or tag a commit that doesn't yet include it.
#
# cargo-release's own dry-run-by-default is the safety net: this script forwards every
# argument after the level/version straight through, so nothing is committed, tagged, or
# pushed until you add --execute yourself.
#
# Usage: scripts/release.sh <patch|minor|major|VERSION> [cargo-release flags...]
# Example (preview):        scripts/release.sh minor
# Example (do it for real): scripts/release.sh minor --execute

if [ $# -lt 1 ]; then
  echo "usage: $0 <patch|minor|major|VERSION> [cargo-release flags...]" >&2
  exit 1
fi

level="$1"
shift

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cargo release "$level" --manifest-path "$root/crates/meethook-record/Cargo.toml" "$@"
cargo release "$level" --manifest-path "$root/Cargo.toml" "$@"
