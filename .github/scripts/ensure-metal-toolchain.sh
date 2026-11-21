#!/usr/bin/env bash
# The voice isolator's GPU engine (the `mlx` feature of meethook-transcribe, on by default on
# macOS) builds MLX with cmake, and that compiles Metal shaders with Apple's Metal Toolchain.
# Xcode 26 and later no longer bundle it; it is a separate download. This installs it when it
# is missing, and does nothing on Linux or when it is already there. The dev shell's `xcrun`
# shim (nix/xcrun-metal-shim.sh) is what then lets the build find it.
set -euo pipefail

[ "$(uname -s)" = Darwin ] || exit 0

if /usr/bin/xcodebuild -showComponent metalToolchain 2>/dev/null | grep -q '^Status: installed'; then
  echo "Metal Toolchain already installed"
  exit 0
fi

echo "Installing the Metal Toolchain"
/usr/bin/xcodebuild -downloadComponent MetalToolchain
/usr/bin/xcodebuild -showComponent metalToolchain
