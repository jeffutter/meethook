{
  description = "meethook - local meeting recorder + transcriber (record on macOS Apple Silicon; transcribe on macOS and Linux)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  # DevShells only, by design. This is a personal, non-distributed tool:
  #
  #   - there is no deployment story to serve, so no packages.outputs
  #   - the Whisper / speaker-diarization model weights are deliberately kept OUT
  #     of the Nix closure. They are large, license-restricted artifacts that
  #     the app downloads on first use into ~/.cache/meethook/models/. Baking
  #     them in would bloat every Nix store copy for zero benefit.
  #   - a devShell is the unit of work here: a single `nix develop` gives a
  #     shell whose PATH carries the exact toolchain the project builds with.
  #
  # Native inputs are added only where the project actually needs them at
  # build time (see per-system notes below).
  outputs = { self, nixpkgs, rust-overlay }:
    let
      # The two supported systems, named explicitly: Apple Silicon hosts the
      # record crate's Apple frameworks, and x86_64 Linux gets the transcribe-
      # only toolchain. Each shell builds for its own machine; cross-compiling
      # between the two stays out of scope. (A host-derived list is impossible:
      # builtins.currentSystem is unavailable in pure flake evaluation.)
      systems = [ "aarch64-darwin" "x86_64-linux" ];

      forSystem = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ (import rust-overlay) ];
          };

          isMacos = system == "aarch64-darwin";


          # rust-overlay names targets by triple; the two supported systems map
          # to different ones, and naming both keeps the toolchain honest about
          # where it runs. An unsupported host system fails loudly here rather
          # than getting a silently wrong target.
          rustTarget = nixpkgs.lib.getAttr system {
            "aarch64-darwin" = "aarch64-apple-darwin";
            "x86_64-linux" = "x86_64-unknown-linux-gnu";
          };

          rustToolchain = pkgs.rust-bin.stable.latest.default.override {
            extensions = [ "rust-src" "rust-analyzer" "clippy" "rustfmt" ];
            targets = [ rustTarget ];
          };
        in
        {
          devShell = pkgs.mkShell {
            name = "meethook";

            packages = [
              rustToolchain
              pkgs.lefthook
              pkgs.cargo-audit
              pkgs.cargo-outdated
            ];

            # whisper.cpp pins an old CMake policy minimum; without this every
            # configure step dies before compiling anything.
            env = {
              CMAKE_POLICY_VERSION_MINIMUM = "3.5";

              # bindgen invokes libclang directly, outside the cc wrapper that
              # would otherwise supply the SDK, so the system headers have to be
              # pointed at by hand.
              LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            }
            # nix develop does not set LD_LIBRARY_PATH, and off macOS the ELF
            # binaries need it twice over: the test binaries link libonnxruntime
            # dynamically, and the statically linked C++ libraries (AEC3,
            # whisper.cpp) pull in libstdc++.so.6, which meson's clang-built
            # sanity executables also need while the build itself is running.
            # On macOS dyld resolves the .dylibs by absolute install name, so
            # nothing is needed.
            // nixpkgs.lib.optionalAttrs (!isMacos) {
              # The hyphen in the package name keeps it out of plain attribute
              # syntax, so the reference is bound here rather than inlined into
              # the string below.
              LD_LIBRARY_PATH =
                let gccRuntime = pkgs."gcc-unwrapped".lib; in
                "${pkgs.onnxruntime}/lib:${gccRuntime}/lib";
            };

            # On macOS the AEC3 library links dynamically against the flake's
            # webrtc-audio-processing package, so nothing extra is needed. Off
            # macOS the crate builds AEC3 from source (the bundled feature):
            # meson + ninja drive the build, and the compiler must be clang
            # because AEC3's meson.build hard-codes it. The wrapped clang also
            # points CC/CXX at itself, so the Rust and whisper.cpp builds
            # compile and link through clang as well; the repo gates pass
            # under that.
            buildInputs =
              [ pkgs.onnxruntime ]
              ++ nixpkgs.lib.optionals isMacos [
                # Frameworks live outside the Nix store, so the SDK has to be on
                # the search path explicitly.
                pkgs.apple-sdk_26
                # Links the flake's libwebrtc_audio_processing.dylib via
                # pkg-config, which is how the dynamic AEC3 dependency reaches
                # the linker.
                pkgs.webrtc-audio-processing
              ];

            # ort-sys probes pkg-config for libonnxruntime at build time; the
            # pkg-config wrapper mkShell generates from buildInputs is what
            # makes the probe find it.
            nativeBuildInputs =
              [
                pkgs.cmake
                pkgs.pkg-config
              ]
              ++ nixpkgs.lib.optionals (!isMacos) [
                pkgs.meson
                pkgs.ninja
                pkgs.clang
              ];

            shellHook = ''
              ${nixpkgs.lib.optionalString isMacos ''
                # bindgen invokes libclang directly, outside the cc wrapper that
                # would otherwise supply the SDK, so the system headers have to
                # be pointed at by hand.
                export BINDGEN_EXTRA_CLANG_ARGS="-isysroot $SDKROOT"
              ''}
              # Rewrites .git/hooks/{pre-commit,pre-push} from lefthook.yml.
              # Idempotent and safe to re-run on every activation.
              lefthook install > /dev/null
              echo "meethook devShell: $(rustc --version)"
            '';
          };

          formatter = pkgs.nixpkgs-fmt;
        };

      envs = nixpkgs.lib.genAttrs systems forSystem;
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (system: { default = envs.${system}.devShell; });
      formatter = nixpkgs.lib.genAttrs systems (system: envs.${system}.formatter);
    };
}
