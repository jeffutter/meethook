#!/usr/bin/env bash
# Swatinem/rust-cache restores target/ from a previous run even after this run's flake.lock
# has moved to a different nixpkgs revision -- GitHub Actions' cache restore-keys fall back to
# an older cache by prefix whenever the exact key misses, regardless of what changed. Native
# build scripts (whisper-rs-sys's cmake configure, webrtc-audio-processing-sys's ninja build)
# bake the *absolute* Nix store paths of whichever compiler/libraries were live when they last
# configured into their OUT_DIR (CMakeCache.txt, build.ninja), and neither cmake nor ninja
# rewrites those paths on their own. Once a flake update rotates the store, the restored
# OUT_DIR points at files that no longer exist on this fresh runner, and the build fails with
# errors like "The CMAKE_C_COMPILER: ... is not a full path to an existing compiler tool."
#
# This scans every such cache file already on disk and deletes the ones referencing a
# /nix/store path that isn't actually present, so the build script starts over from scratch
# instead of tripping over a dangling reference.
set -euo pipefail

# Search from the repo root rather than a hardcoded `target`: the meethook-record job caches
# crates/meethook-record/target instead, and `find` on a path that doesn't exist yet (e.g. a
# first-ever run) would exit non-zero and, under `set -e`, kill this script before it does
# anything.
find . -type f \( -name CMakeCache.txt -o -name build.ninja \) 2>/dev/null | while IFS= read -r cache_file; do
  stale_path=""
  for store_path in $(grep -oE '/nix/store/[a-z0-9]{32}-[^"'"'"'[:space:]]*' "$cache_file" | sort -u); do
    if [[ ! -e "$store_path" ]]; then
      stale_path="$store_path"
      break
    fi
  done

  if [[ -n "$stale_path" ]]; then
    stale_dir=$(dirname "$cache_file")
    echo "evicting stale native build dir (references missing $stale_path): $stale_dir"
    rm -rf "$stale_dir"
  fi
done
