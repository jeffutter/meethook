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
  # Native inputs are added by the slice that actually needs them. Deliberately absent:
  # onnxruntime, webrtc-audio-processing, and whisper deps -- nothing in this slice links
  # them, and carrying them speculatively makes the shell slower to build for no benefit.
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
        # (ScreenCaptureKit, CoreAudio, AudioToolbox, AVFoundation) become linkable.
        buildInputs = [ pkgs.apple-sdk_26 ];

        packages = [ rustToolchain ];

        shellHook = ''
          echo "meethook devShell: $(rustc --version)"
        '';
      };

      formatter.${system} = pkgs.nixpkgs-fmt;
    };
}
