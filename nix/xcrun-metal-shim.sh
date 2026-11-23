#!/bin/bash
# An `xcrun` for the flake's dev shell and MLX package on macOS that finds Apple's Metal compiler.
#
# MLX's CMake build (the `mlx-sys` crate, for the voice isolator's GPU engine) runs
# `xcrun -sdk macosx metal` to compile its shaders. In the dev shell the first `xcrun` on PATH is
# nix's xcbuild stub, which has no Metal compiler, and `DEVELOPER_DIR` points at the nix SDK, which
# has no toolchains. This script sits in front of that stub: `metal` and `metallib` run from the
# host Xcode's Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain` installs it; the
# toolchain is selected by its identifier, which is version specific, so it is read from
# `xcodebuild -showComponent` and cached), and every other call goes to the next `xcrun` on PATH
# unchanged. A sandboxed `nix build` has no host Xcode, so the MLX package builds unsandboxed
# (see flake.nix).
shim_dir=$(cd "$(dirname "$0")" && pwd)
case " $* " in
  *" metal "* | *" metallib "*)
    cache="${TMPDIR:-/tmp}/meethook-metal-toolchain-id"
    if [ ! -s "$cache" ]; then
      env -u DEVELOPER_DIR -u SDKROOT /usr/bin/xcodebuild -showComponent metalToolchain 2>/dev/null \
        | sed -n 's/^Toolchain Identifier: //p' > "$cache"
    fi
    exec env -u DEVELOPER_DIR -u SDKROOT TOOLCHAINS="$(cat "$cache")" /usr/bin/xcrun "$@"
    ;;
esac
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vxF "$shim_dir" | paste -sd: -) exec xcrun "$@"
