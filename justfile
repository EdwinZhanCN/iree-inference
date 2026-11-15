# iree-inference tasks. `just` lists them. CI runs the same recipes, so `just
# check` passing locally means the linux job should pass too.
#
# Recipes are one command per line so they run the same under sh (macOS,
# Linux) and PowerShell (Windows). Branching is done with just's own `if`
# expressions, not in the shell.

set windows-shell := ["pwsh", "-NoLogo", "-NoProfile", "-Command"]

root := justfile_directory()

default:
    @just --list --unsorted

# ---- Check, test --------------------------------------------------------------

# Everything the linux CI job checks, in the same order.
check: fmt-check clippy test doctest

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all --check

clippy:
    cargo clippy --workspace --all-features --all-targets -- -D warnings

# Unit and integration tests. `just test --profile ci` writes JUnit.
test *args:
    cargo nextest run --workspace --all-features {{ args }}

# The ignored GPU tests exercise the Metal module on macOS. Run on an idle
# machine: other GPU work can fail allocations.
[doc("GPU tests: IREE on Metal (macOS)")]
[macos]
test-gpu *args:
    cargo nextest run --workspace --all-features --run-ignored only -E "test(/^gpu_/)" {{ args }}

[linux]
[windows]
test-gpu *args:
    @{{ error("no GPU backend is tested on this platform yet: the GPU tests are Metal") }}

# Doctests (nextest doesn't run them; iree-runtime's compile_fail checks live here).
doctest:
    cargo test --workspace --all-features --doc

# ---- IREE ---------------------------------------------------------------------
# Pinned: the runtime, its headers and the pip compiler must be one version
# (vmfb is not ABI-stable). Keep in step with [workspace.metadata.iree] in
# Cargo.toml.

iree_version := "3.13.0"
iree_commit := "89ece87b89852100c0a4ac3d515e14842d090967"
iree_src := root / "third_party" / "iree"
iree_build := root / "third_party" / "iree-build"
# Runtime-only submodules: flatcc is required, spirv_cross is for Metal, the
# rest are pulled in by CMake even with tests off.
iree_submodules := "third_party/flatcc third_party/spirv_cross third_party/benchmark third_party/googletest"
iree_cmake_flags := "-DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_COMPILER=OFF -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_BINDINGS_TFLITE=OFF -DIREE_BUILD_BINDINGS_TFLITE_JAVA=OFF -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_HAL_DRIVER_LOCAL_TASK=ON" + if os() == "macos" { " -DIREE_HAL_DRIVER_METAL=ON" } else { "" }
# CI caches the IREE build on this; it changes only when what is built does.
iree_cache_key := sha256(iree_commit + " " + iree_submodules + " " + iree_cmake_flags)

# Extra CMake flags pass through, e.g. -DIREE_HAL_DRIVER_VULKAN=ON.
[doc("Build the IREE runtime static library in third_party/ (needs cmake and ninja)")]
iree *cmake_args: _iree-clone
    test "$(git -C {{ quote(iree_src) }} rev-parse HEAD)" = "{{ iree_commit }}"
    git -C {{ quote(iree_src) }} submodule update --init --depth 1 {{ iree_submodules }}
    cmake -G Ninja -S {{ quote(iree_src) }} -B {{ quote(iree_build) }} {{ iree_cmake_flags }} {{ cmake_args }}
    ninja -C {{ quote(iree_build) }} iree_runtime_unified
    cargo clean -p iree-sys

_iree-clone:
    @{{ if path_exists(iree_src / ".git") == "true" { "git -C " + quote(iree_src) + " describe --tags" } else { "git clone --depth 1 --branch v" + iree_version + " https://github.com/iree-org/iree " + quote(iree_src) } }}

venv := root / ".venv"
venv_bin := venv / if os_family() == "windows" { "Scripts" } else { "bin" }
iree_compile := venv_bin / "iree-compile" + if os_family() == "windows" { ".exe" } else { "" }
testdata_out := root / "testdata" / "build"

# One dependency per module in testdata/.
[doc("Compile the IREE test modules into testdata/build/ (needs uv)")]
testdata: _venv (_module "basic")

_venv:
    @{{ if path_exists(iree_compile) == "true" { "" } else { "uv venv --python 3.12 " + quote(venv) } }}
    @{{ if path_exists(iree_compile) == "true" { "" } else { "uv pip install --python " + quote(venv) + " iree-base-compiler==" + iree_version } }}

# Generic CPU on purpose: test artifacts must not depend on the host CPU.
_module name: (_module-metal name)
    cmake -E make_directory {{ quote(testdata_out) }}
    {{ quote(iree_compile) }} {{ quote(root / "testdata" / name + ".mlir") }} --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic -o {{ quote(testdata_out / name + ".llvm-cpu.vmfb") }}

# Best effort: needs Xcode's Metal Toolchain
# (xcodebuild -downloadComponent MetalToolchain); CI doesn't have it.
[macos]
_module-metal name:
    cmake -E make_directory {{ quote(testdata_out) }}
    -{{ quote(iree_compile) }} {{ quote(root / "testdata" / name + ".mlir") }} --iree-hal-target-device=metal -o {{ quote(testdata_out / name + ".metal.vmfb") }}

[linux]
[windows]
_module-metal name:

