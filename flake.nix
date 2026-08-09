{
  description = "meethook - local meeting recorder + transcriber (macOS, Apple Silicon)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  # devShell-only on purpose: meethook is a personal, non-distributed tool, so there is
  # no buildRustPackage/crane package output to maintain.
  #
  # Native inputs are added by the slice that actually needs them, never speculatively.
  #
  # Model weights are deliberately absent, and not merely unadded: they are fetched at
  # runtime into ~/meethook/models/ and verified against sha256 hashes embedded in source,
  # so no checkpoint -- ggml or ONNX -- is ever part of this closure.
  outputs = { self, nixpkgs, rust-overlay }:
    let
      # Apple Silicon only. Cross-platform support is explicitly out of scope, so the
      # flake names the one system directly instead of mapping over a system list.
      system = "aarch64-darwin";

      pkgs = import nixpkgs {
        inherit system;
        overlays = [ (import rust-overlay) ];
      };

      # Rolling stable, not a pinned version; flake.lock is what keeps builds reproducible
      # between deliberate `nix flake update` runs.
      rustToolchain = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
        targets = [ "aarch64-apple-darwin" ];
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "meethook";

        # apple-sdk_26 belongs in buildInputs (not nativeBuildInputs): that is where it
        # sets SDKROOT for the shell, which is how the Darwin frameworks
        # (ScreenCaptureKit, CoreAudio, AudioToolbox, AVFoundation, Metal) become linkable.
        #
        # webrtc-audio-processing is the AEC3 implementation the transcribe pre-pass links
        # against dynamically; the `webrtc-audio-processing` crate's `bundled` feature,
        # which would build the vendored C++ instead, is off because it does not
        # cross-compile on Apple (tonarino/webrtc-audio-processing#102).
        #
        # onnxruntime runs the diarization and speaker-embedding graphs. This build is
        # configured with onnxruntime_USE_COREML, which is what makes the CoreML execution
        # provider registrable; `ort-sys` finds it by probing pkg-config for the module
        # named `libonnxruntime`, which is exactly what the dev output's .pc file is called,
        # so nothing is ever downloaded at build time.
        buildInputs = [ pkgs.apple-sdk_26 pkgs.onnxruntime pkgs.webrtc-audio-processing ];

        # whisper-rs-sys compiles its own vendored whisper.cpp with the `cmake` crate.
        # pkgs.whisper-cpp is deliberately *not* here: linking a second, separately built
        # copy alongside the vendored one is a silent version skew waiting to happen.
        #
        # pkg-config is a build-time tool, so it belongs here rather than in buildInputs:
        # its setup hook is what puts each buildInput's dev output on PKG_CONFIG_PATH,
        # which is how webrtc-audio-processing-sys finds the library at all.
        nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];

        # Developer tooling, not build inputs: neither is linked against nor invoked by
        # any crate's build script, so `packages` is the right list for both.
        #
        # lefthook runs the gates in lefthook.yml; the shellHook below installs its git
        # hooks, so entering the shell is the only setup step a fresh clone needs.
        #
        # cargo-audit backs the pre-push advisory scan.
        packages = [ rustToolchain pkgs.lefthook pkgs.cargo-audit ];

        env = {
          # bindgen (via whisper-rs-sys) loads libclang at build time rather than linking
          # it, so it needs the path handed to it explicitly.
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          # whisper.cpp declares `cmake_minimum_required(VERSION 3.5)`, which CMake 4
          # rejects outright. whisper-rs-sys forwards every CMAKE_*/GGML_*/WHISPER_* env
          # var straight into the CMake configure, so this restores the old floor without
          # patching or vendoring whisper.cpp.
          CMAKE_POLICY_VERSION_MINIMUM = "3.5";
        };

        shellHook = ''
          # bindgen invokes libclang directly, outside the cc wrapper that would otherwise
          # supply the SDK, so the system headers have to be pointed at by hand.
          export BINDGEN_EXTRA_CLANG_ARGS="-isysroot $SDKROOT"

          # Rewrites .git/hooks/{pre-commit,pre-push} from lefthook.yml. It is idempotent,
          # so it is unguarded and safe on every direnv reload; only its chatter is
          # dropped, never its errors.
          lefthook install > /dev/null

          echo "meethook devShell: $(rustc --version)"
        '';
      };

      formatter.${system} = pkgs.nixpkgs-fmt;
    };
}
