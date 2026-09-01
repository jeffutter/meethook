{
  description = "meethook - local meeting recorder + transcriber (record on macOS Apple Silicon; transcribe on macOS and Linux)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
  };

  # DevShells are the primary output, and the whole reason native inputs are wired up at
  # all: a single `nix develop` gives a shell whose PATH carries the exact toolchain the
  # project builds with. The `packages.meethook` output below rides on that same wiring --
  # it exists so the CD workflow (and `nix run`/`nix profile install`) has a build of the
  # binary whose dynamic libraries (onnxruntime, webrtc-audio-processing) actually resolve,
  # which a plain `cargo build` on a bare GitHub Actions runner cannot give it.
  #
  # The Whisper / speaker-diarization model weights are still deliberately kept OUT of the
  # Nix closure regardless: they are large, license-restricted artifacts that the app
  # downloads on first use into ~/.cache/meethook/models/, and that stays true whether the
  # binary itself came from `cargo build` or `nix build`.
  outputs = { self, nixpkgs, rust-overlay, crane }:
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

          # Shared between the devShell and the `meethook` package build below, so a build
          # library gains exactly the same native dependencies a `nix develop` shell does --
          # neither can silently drift out of step with the other.

          # whisper.cpp pins an old CMake policy minimum; without this every configure step
          # dies before compiling anything.
          commonEnv = {
            CMAKE_POLICY_VERSION_MINIMUM = "3.5";

            # bindgen invokes libclang directly, outside the cc wrapper that would otherwise
            # supply the SDK, so the system headers have to be pointed at by hand.
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          }
          # nix develop does not set LD_LIBRARY_PATH, and off macOS the ELF binaries need it
          # twice over: the test binaries link libonnxruntime dynamically, and the statically
          # linked C++ libraries (AEC3, whisper.cpp) pull in libstdc++.so.6, which meson's
          # clang-built sanity executables also need while the build itself is running. On
          # macOS dyld resolves the .dylibs by absolute install name, so nothing is needed.
          // nixpkgs.lib.optionalAttrs (!isMacos) {
            # The hyphen in the package name keeps it out of plain attribute syntax, so the
            # reference is bound here rather than inlined into the string below.
            LD_LIBRARY_PATH =
              let gccRuntime = pkgs."gcc-unwrapped".lib; in
              "${pkgs.onnxruntime}/lib:${gccRuntime}/lib";
          }
          // nixpkgs.lib.optionalAttrs isMacos {
            # `pkgs.llvmPackages.libclang` ships only the shared library, not a `clang`
            # executable next to it -- and libclang-sys, which every bindgen build script
            # here dlopens directly, normally derives its resource directory (the builtin
            # `stddef.h`/`math.h` etc. that a real `clang` binary would find relative to
            # itself) from exactly that missing binary. Without it, libc++ headers that
            # lean on those builtins -- `<string>`'s `using_if_exists size_t;`, `<cmath>`'s
            # `FP_NAN` family -- fail to parse, which is otherwise invisible: it only bites
            # a bindgen invocation that actually walks the C++ standard library (i.e.
            # webrtc-audio-processing-sys, not the plain-C headers the other bindgen
            # consumers here parse), so it can look like "bindgen works" if that crate's
            # build.rs output happens to already be cached from before this was noticed.
            BINDGEN_EXTRA_CLANG_ARGS = "-isysroot ${pkgs.apple-sdk_26} -isystem ${pkgs.llvmPackages.libclang.lib}/lib/clang/${nixpkgs.lib.versions.major pkgs.llvmPackages.libclang.version}/include";
          };

          # On macOS the AEC3 library links dynamically against the flake's
          # webrtc-audio-processing package, so nothing extra is needed. Off
          # macOS the crate builds AEC3 from source (the bundled feature):
          # meson + ninja drive the build, and the compiler must be clang
          # because AEC3's meson.build hard-codes it. The wrapped clang also
          # points CC/CXX at itself, so the Rust and whisper.cpp builds
          # compile and link through clang as well; the repo gates pass
          # under that.
          commonBuildInputs =
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

          # ort-sys probes pkg-config for libonnxruntime at build time; the pkg-config
          # wrapper mkShell (or crane's build) generates from buildInputs is what makes the
          # probe find it.
          commonNativeBuildInputs =
            [
              pkgs.cmake
              pkgs.pkg-config
            ]
            ++ nixpkgs.lib.optionals (!isMacos) [
              pkgs.meson
              pkgs.ninja
              pkgs.clang
            ];

          craneLib = crane.mkLib pkgs;

          # Single source of truth for the package version -- read from the meethook crate's
          # own Cargo.toml (the workspace's [workspace.package] table has no `version` key;
          # every crate here pins its own) so the flake never drifts out of sync with `cargo`.
          version = (builtins.fromTOML (builtins.readFile ./crates/meethook/Cargo.toml)).package.version;

          # The whole repository, not just crates/meethook: the `meethook` binary pulls in
          # meethook-record (macOS only, target-gated) via a path dependency that roots its
          # own workspace one directory over, and crane needs that source on disk to resolve
          # it exactly as a local `cargo build --workspace` already does.
          src = nixpkgs.lib.cleanSourceWith { src = craneLib.path ./.; };

          commonArgs = {
            inherit src;
            buildInputs = commonBuildInputs;
            nativeBuildInputs = commonNativeBuildInputs;
          } // commonEnv;

          # version is deliberately NOT the release version here: buildDepsOnly's output only
          # depends on Cargo.lock, but Nix hashes a derivation's name (pname-version) into its
          # store path, so tying this to the release version would invalidate the deps cache
          # on every release even when no dependency actually changed.
          cargoArtifacts = craneLib.buildDepsOnly (
            commonArgs // { pname = "meethook-workspace"; version = "deps"; }
          );

          # doCheck is off: CI's `test` job already runs the full suite via `cargo nextest`
          # on every push, including the AEC3/onnxruntime/whisper fixtures that a Nix build
          # sandbox has no useful way to exercise. Re-running it inside crane's checkPhase
          # would be redundant at best.
          meethook = craneLib.buildPackage (
            commonArgs
            // {
              inherit cargoArtifacts version;
              pname = "meethook";
              cargoExtraArgs = "--bin meethook";
              doCheck = false;
            }
          );
        in
        {
          devShell = pkgs.mkShell {
            name = "meethook";

            packages = [
              rustToolchain
              pkgs.lefthook
              pkgs.cargo-audit
              pkgs.cargo-outdated
              # The pre-push test gate runs the suite through nextest, which
              # isolates each test binary and reports failures without the
              # panic-aborts the rest of the suite that cargo test does.
              pkgs.cargo-nextest
            ];

            env = commonEnv;
            buildInputs = commonBuildInputs;
            nativeBuildInputs = commonNativeBuildInputs;

            shellHook = ''
              # Rewrites .git/hooks/{pre-commit,pre-push} from lefthook.yml.
              # Idempotent and safe to re-run on every activation.
              lefthook install > /dev/null
              echo "meethook devShell: $(rustc --version)"
            '';
          };

          inherit meethook;

          formatter = pkgs.nixpkgs-fmt;
        };

      envs = nixpkgs.lib.genAttrs systems forSystem;
    in
    {
      devShells = nixpkgs.lib.genAttrs systems (system: { default = envs.${system}.devShell; });
      packages = nixpkgs.lib.genAttrs systems (system: {
        default = envs.${system}.meethook;
        inherit (envs.${system}) meethook;
      });
      formatter = nixpkgs.lib.genAttrs systems (system: envs.${system}.formatter);
    };
}
