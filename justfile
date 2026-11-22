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

# ---- ESP32-P4: PP-OCRv6 through ESP-PPQ and QDQ ONNX -----------------------
# ESP-PPQ quantizes with esp-dl's rules (esp32p4, w8a8, power-of-two scales),
# exports QDQ ONNX, and IREE compiles it. ESP-PPQ stays in its own venv;
# models and outputs go under the ignored third_party/.

esp_venv := root / ".venv-esp"
esp_bin := esp_venv / if os_family() == "windows" { "Scripts" } else { "bin" }
esp_exe := if os_family() == "windows" { ".exe" } else { "" }
esp_py := esp_bin / "python" + esp_exe
esp_compiler := root / "compiler" / "targets" / "esp32p4"
esp_backend := root / "backends" / "esp32p4"
esp_kernel_test := esp_backend / "apps" / "kernel_test"
esp_rec_app := esp_backend / "apps" / "bench_rec"
ppocr_models := root / "third_party" / "ppocr-v6-tiny"
esp_out := root / "third_party" / "esp-out"
# The old CMake build directories contain absolute paths to the old sources.
esp_plugin_build := root / "third_party" / "esp32p4-reference-build"
esp_runtime_build := root / "third_party" / "iree-build-esp32p4"
esp_plugin := esp_plugin_build / if os_family() == "windows" { "espdl_host_plugin.dll" } else { "libespdl_host_plugin.so" }
esp_ref_lib := esp_plugin_build / if os() == "macos" { "libespdl_ref.dylib" } else if os_family() == "windows" { "espdl_ref.dll" } else { "libespdl_ref.so" }
# rv32imafc with the single-precision ABI. IREE codegen here is scalar.
esp_p4_flags := "--iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv32-pc-linux-elf --iree-llvmcpu-target-cpu=generic-rv32 --iree-llvmcpu-target-abi=ilp32f --iree-llvmcpu-target-cpu-features=+m,+a,+f,+c"

[doc("Create the ESP tooling venv in .venv-esp (esp-ppq, torch, IREE compiler + runtime)")]
esp-venv:
    @{{ if path_exists(esp_py) == "true" { "" } else { "uv venv --python 3.12 " + quote(esp_venv) } }}
    uv pip install --quiet --python {{ quote(esp_venv) }} -r {{ quote(esp_compiler / "requirements.txt") }}

# inference.yml holds the preprocessing and, for rec, the CTC dictionary.
[doc("Download PP-OCRv6 tiny det/rec ONNX (PaddlePaddle on Hugging Face, Apache-2.0)")]
ppocr-fetch: (_ppocr-fetch "det" "inference.onnx") (_ppocr-fetch "det" "inference.yml") (_ppocr-fetch "rec" "inference.onnx") (_ppocr-fetch "rec" "inference.yml")

_ppocr-fetch m file:
    cmake -E make_directory {{ quote(ppocr_models / m) }}
    @{{ if path_exists(ppocr_models / m / file) == "true" { "" } else { "curl -sSfL -o " + quote(ppocr_models / m / file) + " https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_" + m + "_onnx/resolve/main/" + file } }}

# Host CPU is `generic`: IREE 3.13 crashes lowering i8mm contractions for
# Apple Silicon on the rec model.
[doc("Quantize a PP-OCRv6 model (det|rec) for esp32p4, compile it for the host, compare ESP-PPQ / ORT / IREE")]
esp-ppocr m: esp-venv ppocr-fetch
    {{ quote(esp_py) }} {{ quote(esp_compiler / "quant_export.py") }} {{ quote(ppocr_models / m / "inference.onnx") }} {{ quote(esp_out / m) }} {{ m }} {{ if m == "det" { "1 3 736 736" } else if m == "rec" { "1 3 48 320" } else { error("model must be det or rec") } }}
    {{ quote(esp_py) }} {{ quote(esp_compiler / "fix_qdq.py") }} {{ quote(esp_out / m / m + ".qdq.onnx") }} {{ quote(esp_out / m / m + ".qdq.std.onnx") }}
    {{ quote(esp_bin / "iree-import-onnx" + esp_exe) }} {{ quote(esp_out / m / m + ".qdq.std.onnx") }} --opset-version 17 -o {{ quote(esp_out / m / m + ".qdq.mlir") }}
    {{ quote(esp_bin / "iree-compile" + esp_exe) }} {{ quote(esp_out / m / m + ".qdq.mlir") }} --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic -o {{ quote(esp_out / m / m + ".host.vmfb") }}
    {{ quote(esp_py) }} {{ quote(esp_compiler / "compare_iree_ort.py") }} {{ quote(esp_out / m / m + ".host.vmfb") }} {{ quote(esp_out / m / m + ".qdq.std.onnx") }} {{ quote(esp_out / m / m + ".fp32.onnx") }} {{ quote(esp_out / m / m + ".ppq_sim.npy") }} {{ m }} {{ quote(ppocr_models / m / "inference.yml") }}

[doc("Build the host plugin with the esp-dl kernel reference implementations (needs cmake and ninja)")]
esp-plugin: _iree-clone
    cmake -G Ninja -S {{ quote(esp_backend / "reference") }} -B {{ quote(esp_plugin_build) }} -DCMAKE_BUILD_TYPE=Release -DIREE_SOURCE_DIR={{ quote(iree_src) }}
    ninja -C {{ quote(esp_plugin_build) }}

[doc("Compile a quantized PP-OCRv6 model (after esp-ppocr) with esp-dl kernel dispatches for the host; compare against ORT with the reference plugin")]
esp-ppocr-espdl m: esp-venv esp-plugin (_esp-rewrite m)
    {{ quote(esp_bin / "iree-compile" + esp_exe) }} {{ quote(esp_out / m / m + ".espdl.mlir") }} --compile-from=input --iree-hal-target-device=local --iree-hal-local-target-device-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic -o {{ quote(esp_out / m / m + ".espdl.host.vmfb") }}
    {{ quote(esp_py) }} {{ quote(esp_compiler / "compare_iree_ort.py") }} --plugin {{ quote(esp_plugin) }} {{ quote(esp_out / m / m + ".espdl.host.vmfb") }} {{ quote(esp_out / m / m + ".qdq.std.onnx") }} {{ quote(esp_out / m / m + ".fp32.onnx") }} {{ quote(esp_out / m / m + ".ppq_sim.npy") }} {{ m }} {{ quote(ppocr_models / m / "inference.yml") }}

# Input-phase IR with esp-dl kernel dispatches in place of quantized
# convolutions and matrix products (target independent).
_esp-rewrite m: esp-venv
    {{ quote(esp_bin / "iree-compile" + esp_exe) }} {{ quote(esp_out / m / m + ".qdq.mlir") }} --compile-to=input -o {{ quote(esp_out / m / m + ".input.mlir") }}
    {{ quote(esp_py) }} {{ quote(esp_compiler / "espdl_rewrite.py") }} {{ quote(esp_out / m / m + ".input.mlir") }} {{ quote(esp_out / m / m + ".espdl.mlir") }}

# Static library: one LLVM module, compiled serially (det takes ~10 minutes
# without esp-dl kernels). Unresolved libgcc helpers (e.g. __fixsfdi) come
# from the firmware link; esp-dl kernel imports from the firmware's import
# provider. `kernels="iree"` compiles everything with IREE's own codegen.
[doc("Compile a quantized PP-OCRv6 model (after esp-ppocr) for ESP32-P4 as a static library; report code size")]
esp-ppocr-p4 m kernels="espdl": (_esp-rewrite m)
    {{ quote(esp_bin / "iree-compile" + esp_exe) }} {{ quote(esp_out / m / m + (if kernels == "espdl" { ".espdl.mlir" } else if kernels == "iree" { ".input.mlir" } else { error("kernels must be espdl or iree") })) }} --compile-from=input {{ esp_p4_flags }} --iree-llvmcpu-link-embedded=false --iree-llvmcpu-link-static --iree-llvmcpu-static-library-output-path={{ quote(esp_out / m / m + "." + kernels + ".p4.o") }} -o {{ quote(esp_out / m / m + "." + kernels + ".p4.vmfb") }}
    {{ quote(esp_py) }} {{ quote(esp_compiler / "p4_size.py") }} {{ quote(esp_out / m / m + "." + kernels + ".p4.o") }}

# Test vectors from the kernel configurations of both models (run esp-ppocr
# det and rec first); expected outputs from the reference kernels. The
# firmware compares the esp-dl backend's bytes on the board.
[doc("Check the esp-dl kernel backend on an ESP32-P4 board against the reference kernels (build, flash, monitor)")]
esp-p4-kernel-test: esp-venv esp-plugin
    {{ quote(esp_py) }} {{ quote(esp_compiler / "gen_vectors.py") }} {{ quote(esp_ref_lib) }} {{ quote(esp_kernel_test / "src" / "vectors.c") }} {{ quote(esp_out / "rec" / "rec.input.mlir") }} {{ quote(esp_out / "det" / "det.input.mlir") }}
    idf -C {{ quote(esp_kernel_test) }} run

# IREE runtime for ESP32-P4 (generic platform, local-sync, static library
# loader), built with ESP-IDF's GCC and picolibc. Uses the host build's
# flatcc tool, so run `just iree` first.
[doc("Cross-compile the IREE runtime for ESP32-P4 into third_party/iree-build-esp32p4")]
esp-iree-p4:
    idf_env="$(idf -C {{ quote(esp_rec_app) }} env --shell)" && eval "$idf_env" && \
      cmake -G Ninja -S {{ quote(iree_src) }} -B {{ quote(esp_runtime_build) }} -DCMAKE_TOOLCHAIN_FILE={{ quote(esp_backend / "cmake" / "p4_toolchain.cmake") }} -DCMAKE_BUILD_TYPE=Release -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF -DIREE_HOST_BIN_DIR={{ quote(iree_build / "tools") }} && \
      ninja -C {{ quote(esp_runtime_build) }} iree_runtime_unified iree_hal_local_loaders_static_library_loader iree_hal_local_executable_library_util flatcc_parsing

# Whole rec on the board: compares the output bytes with the host's and
# times each run (run esp-ppocr-espdl rec and esp-ppocr-p4 rec first).
[doc("Run PP-OCRv6 rec through IREE + esp-dl on an ESP32-P4 board (build, flash, monitor)")]
esp-p4-rec: esp-venv esp-plugin esp-iree-p4
    {{ quote(esp_py) }} {{ quote(esp_compiler / "gen_model_io.py") }} {{ quote(esp_plugin) }} {{ quote(esp_out / "rec" / "rec.espdl.host.vmfb") }} rec {{ quote(esp_out / "rec" / "model_io") }}
    idf -C {{ quote(esp_rec_app) }} run
