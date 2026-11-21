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
#
# Run it inside `nix develop` (`nix develop -c bash <this script>`), never before it. On a fresh
# runner the store holds nothing until the dev shell is realized, so "is this path present?" asked
# earlier answers "no" for every path in the shell's own closure (cctools' install_name_tool, the
# compiler wrapper, ...) and the script evicts caches that were perfectly good. That is how
# mlx-sys's `out/build` (its libraries and mlx.metallib) was deleted from a warm cache while
# cargo still considered the crate built, which no later step can notice until it links. Inside
# the shell, a path is missing only if the cache really was made against an older store.
set -euo pipefail

# Search from the repo root rather than a hardcoded `target`: the meethook-record job caches
# crates/meethook-record/target instead, and `find` on a path that doesn't exist yet (e.g. a
# first-ever run) would exit non-zero and, under `set -e`, kill this script before it does
# anything.
find . -type f \( -name CMakeCache.txt -o -name build.ninja \) 2>/dev/null | while IFS= read -r cache_file; do
  # An earlier hit may have removed the whole target dir this file was in.
  [[ -f "$cache_file" ]] || continue
  stale_path=""
  for store_path in $(grep -oE '/nix/store/[a-z0-9]{32}-[^"'"'"'[:space:]]*' "$cache_file" | sort -u); do
    if [[ ! -e "$store_path" ]]; then
      stale_path="$store_path"
      break
    fi
  done

  if [[ -n "$stale_path" ]]; then
    if [[ "$cache_file" == */target/* ]]; then
      # Inside a cargo target/: deleting only the native build dir leaves cargo's fingerprints
      # saying the build script already ran, so it never reruns, and whatever it produced there
      # (mlx-sys links its libraries from out/build) is simply gone. Evict the whole target dir,
      # as for a stale build-script executable below.
      target_dir="${cache_file%%/target/*}/target"
      echo "evicting stale target dir (native build cache $cache_file references missing $stale_path): $target_dir"
      rm -rf "$target_dir"
    else
      stale_dir=$(dirname "$cache_file")
      echo "evicting stale native build dir (references missing $stale_path): $stale_dir"
      rm -rf "$stale_dir"
    fi
  fi
done

# The same rot hits compiled build-script executables (ring's `build-script-build`, say): their
# ELF interpreter and rpath are absolute store paths of the glibc they were linked with. Once
# the store rotates, cargo reuses the cached binary as "fresh" and exec fails with "No such
# file or directory" -- surfacing as `could not execute process build_script_build (never
# executed)`. Deleting one crate's directory is not enough, because cargo's fingerprints and
# the dependent rlibs still point at it, so a hit evicts the whole target directory.
# (Cargo names the file `build-script-build` or `build_script_build-<hash>` by version.)
find . -type f -path '*/target/*/build/*' \( -name 'build-script-build' -o -name 'build_script_build*' \) \
  ! -name '*.d' 2>/dev/null | while IFS= read -r script; do
  for store_path in $(tr '\0' '\n' <"$script" | grep -aoE '/nix/store/[a-z0-9]{32}-[^"'"'"'[:space:]:]*' | sort -u); do
    if [[ ! -e "$store_path" ]]; then
      target_dir="${script%%/target/*}/target"
      echo "evicting stale target dir (build script $script references missing $store_path): $target_dir"
      rm -rf "$target_dir"
      break 2
    fi
  done
done
