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

# ---- Metal: BioCLIP 2 image encoder ------------------------------------------
# Upstream weights (hf-hub:imageomics/bioclip-2, MIT) and outputs live in
# third_party/bioclip-2/. The TreeOfLife index is Lumen-Hub's; point
# BIOCLIP_DATASETS at its datasets/ directory. Recipes run one heavy job at a
# time: the ViT-L conversions peak at ~5 GB each.

metal_venv := root / ".venv-metal"
metal_bin := metal_venv / if os_family() == "windows" { "Scripts" } else { "bin" }
metal_py := metal_bin / "python" + esp_exe
metal_compiler := root / "compiler" / "targets" / "metal"
bioclip := root / "third_party" / "bioclip-2"
bioclip_out := bioclip / "out"
bioclip_datasets := env("BIOCLIP_DATASETS", "/Volumes/CodeBase/Projects/Lumen-Hub/models/bioclip-2/datasets")

metal-venv:
    @{{ if path_exists(metal_py) == "true" { "" } else { "uv venv --python 3.12 " + quote(metal_venv) } }}
    uv pip install --quiet --python {{ quote(metal_venv) }} -r {{ quote(metal_compiler / "requirements.txt") }}

[doc("Download BioCLIP 2 (open_clip weights) and the iNaturalist sample photos into third_party/bioclip-2")]
bioclip-fetch: metal-venv
    {{ quote(metal_py) }} -c "from huggingface_hub import hf_hub_download as d; [d('imageomics/bioclip-2', f, local_dir='{{ bioclip / "upstream" }}') for f in ('open_clip_config.json', 'open_clip_model.safetensors', 'README.md')]"
    {{ quote(metal_py) }} {{ quote(metal_compiler / "fetch_samples.py") }} {{ quote(bioclip / "samples") }}

[doc("Export the BioCLIP 2 fp32 reference, the f16 IREE MLIR (SDPA and decomposed) and the Core ML tier-2 package")]
bioclip-export: bioclip-fetch
    {{ quote(metal_py) }} {{ quote(metal_compiler / "export.py") }} reference {{ quote(bioclip / "upstream") }} {{ quote(bioclip / "samples") }} {{ quote(bioclip_datasets) }} {{ quote(bioclip_out) }}
    {{ quote(metal_py) }} {{ quote(metal_compiler / "export.py") }} iree {{ quote(bioclip / "upstream") }} {{ quote(bioclip_out) }}
    {{ quote(metal_py) }} {{ quote(metal_compiler / "export.py") }} iree {{ quote(bioclip / "upstream") }} {{ quote(bioclip_out) }} --attention decomposed
    {{ quote(metal_py) }} {{ quote(metal_compiler / "export.py") }} coreml {{ quote(bioclip / "upstream") }} {{ quote(bioclip_out) }}

# IREE 3.13's metal-spirv cannot compile iree_linalg_ext.attention, so the
# default-path baseline uses the decomposed export.
[doc("Compile the decomposed BioCLIP 2 encoder with IREE's default Metal path and compare it against the fp32 reference")]
[macos]
bioclip-metal-baseline:
    {{ quote(metal_bin / "iree-compile" + esp_exe) }} {{ quote(bioclip_out / "vision.b1.r224.f16.decomposed.mlirbc") }} --iree-hal-target-device=metal -o {{ quote(bioclip_out / "vision.b1.r224.f16.decomposed.metal.vmfb") }}
    {{ quote(metal_py) }} {{ quote(metal_compiler / "compare.py") }} {{ quote(bioclip_out) }} {{ quote(bioclip_datasets) }} {{ quote(bioclip_out / "vision.b1.r224.f16.decomposed.metal.vmfb") }}

# Metal HAL has no device profiling, so each executable is benchmarked on its
# own and weighted by its dispatch count. Takes about 10 minutes.
[doc("Per-dispatch GPU time breakdown of the decomposed BioCLIP 2 encoder on IREE's default Metal path")]
[macos]
bioclip-metal-profile:
    {{ quote(metal_py) }} {{ quote(metal_compiler / "dispatch_profile.py") }} {{ quote(bioclip_out / "vision.b1.r224.f16.decomposed.mlirbc") }} {{ quote(bioclip_out / "profile.decomposed") }} --iree-hal-target-device=metal

# ---- Metal: MLX kernels through hal.dispatch.extern --------------------------
# The pip compiler has no external Metal executables, so iree-compile is built
# from the pinned IREE source with patches/iree-metal-external-executables.patch
# (Metal target only, no torch input). Torch modules go through the pip
# compiler to the input phase first; both are the same commit.

metal_iree_build := root / "third_party" / "iree-build-compiler"
metal_iree_compile := metal_iree_build / "tools" / "iree-compile"
# Compiler patches: external Metal executables, and SIMD-group functions for
# iOS (SPIRV-Cross otherwise rejects reductions there).
metal_iree_patches := quote(metal_compiler / "patches" / "iree-metal-external-executables.patch") + " " + quote(metal_compiler / "patches" / "iree-metal-ios-simdgroup.patch")
metal_iree_cmake_flags := "-DCMAKE_BUILD_TYPE=Release -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF -DIREE_BUILD_COMPILER=ON -DIREE_BUILD_TESTS=OFF -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=OFF -DIREE_BUILD_BUNDLED_LLVM=ON -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TOSA=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_METAL_SPIRV=ON -DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_HAL_DRIVER_LOCAL_SYNC=ON -DIREE_ENABLE_ASSERTIONS=OFF -DLLVM_PARALLEL_LINK_JOBS=1"
mlx_commit := "c23a35c9e6f4b26a2b851dda8c9e0630d13069ad"
mlx_src := root / "third_party" / "mlx"
mlx_kernels_out := bioclip_out / "kernels"
mlx_lib := mlx_kernels_out / "mlx_kernels.metallib"
# Metal GPU architecture MLX tiles are picked for, e.g. applegpu_g17p for a
# cross-compile; empty picks them for this machine's GPU, as MLX does.
metal_gpu_arch := ""
mlx_arch_flag := if metal_gpu_arch == "" { "" } else { "--arch " + metal_gpu_arch }
# MLX rules for the BioCLIP 2 encoder. Linears (MxNxK: QKV/out projections,
# MLP up and down) read their weights untransposed. The SDPA export's
# attention (BxLxHxD) goes to MLX steel attention as a whole; the decomposed
# export's attention matmuls (BxMxNxK: scores, attention x V) go to GEMMs.
bioclip_linears := "--linear 257x1024x1024 --linear 257x4096x1024 --linear 257x1024x4096"
bioclip_mlx_rules := bioclip_linears + " --attention 1x257x16x64"
bioclip_mlx_gemm_rules := bioclip_linears + " --gemm 16x257x257x64 --gemm 16x257x64x257"

# Builds LLVM and MLIR from source: about 30 minutes on an M2 Pro. 4 jobs keep
# peak memory near 6 GB. A rebuild after a patch change takes a few minutes.
[doc("Build iree-compile with the Metal patches into third_party/iree-build-compiler")]
[macos]
metal-iree-compiler: _iree-clone
    test "$(git -C {{ quote(iree_src) }} rev-parse HEAD)" = "{{ iree_commit }}"
    git -C {{ quote(iree_src) }} submodule update --init --depth 1 third_party/llvm-project {{ iree_submodules }}
    for p in {{ metal_iree_patches }}; do git -C {{ quote(iree_src) }} apply --reverse --check "$p" 2>/dev/null || git -C {{ quote(iree_src) }} apply "$p"; done
    cmake -G Ninja -S {{ quote(iree_src) }} -B {{ quote(metal_iree_build) }} {{ metal_iree_cmake_flags }}
    ninja -C {{ quote(metal_iree_build) }} -j4 iree-compile

_mlx-clone:
    @{{ if path_exists(mlx_src / ".git") == "true" { "git -C " + quote(mlx_src) + " rev-parse --short HEAD" } else { "git init -q " + quote(mlx_src) + " && git -C " + quote(mlx_src) + " fetch -q --depth 1 https://github.com/ml-explore/mlx " + mlx_commit + " && git -C " + quote(mlx_src) + " checkout -q FETCH_HEAD" } }}
    test "$(git -C {{ quote(mlx_src) }} rev-parse HEAD)" = "{{ mlx_commit }}"

# Compiles the MLX GEMM and attention wrappers into one metallib for an SDK
# (macosx or iphoneos).
_mlx-metallib sdk out: _mlx-clone
    {{ quote(metal_py) }} {{ quote(metal_compiler / "kernels" / "gen_mlx_attention.py") }} {{ quote(mlx_src) }} {{ quote(mlx_kernels_out / "gen") }}
    xcrun -sdk {{ sdk }} metal -std=metal3.1 -O3 -Wno-c++17-extensions -I {{ quote(mlx_src) }} -c {{ quote(metal_compiler / "kernels" / "mlx_gemm.metal") }} -o {{ quote(mlx_kernels_out / "mlx_gemm." + sdk + ".air") }}
    xcrun -sdk {{ sdk }} metal -std=metal3.1 -O3 -Wno-c++17-extensions -I {{ quote(mlx_src) }} -I {{ quote(mlx_kernels_out / "gen") }} -c {{ quote(metal_compiler / "kernels" / "mlx_attention.metal") }} -o {{ quote(mlx_kernels_out / "mlx_attention." + sdk + ".air") }}
    xcrun -sdk {{ sdk }} metallib {{ quote(mlx_kernels_out / "mlx_gemm." + sdk + ".air") }} {{ quote(mlx_kernels_out / "mlx_attention." + sdk + ".air") }} -o {{ quote(out) }}

[doc("Compile the MLX steel GEMM and attention wrappers (kernels/) into one metallib")]
[macos]
mlx-kernels: (_mlx-metallib "macosx" mlx_lib)

[doc("Check the MLX GEMM and attention kernels against NumPy and IREE's own codegen on the BioCLIP 2 shapes")]
[macos]
mlx-kernel-test: mlx-kernels
    {{ quote(metal_py) }} {{ quote(metal_compiler / "test_mlx_kernels.py") }} {{ quote(metal_iree_compile) }} {{ quote(mlx_lib) }} {{ quote(mlx_kernels_out / "test") }} {{ mlx_arch_flag }}

# Lowers an export to the input phase with the pip compiler, then compiles it
# with the patched one and the MLX rules; the rest stays on IREE's
# metal-spirv codegen.
_bioclip-mlx stem rules: mlx-kernels
    {{ quote(metal_bin / "iree-compile" + esp_exe) }} {{ quote(bioclip_out / stem + ".mlirbc") }} --iree-hal-target-device=metal --compile-to=input --emit-mlir-bytecode -o {{ quote(bioclip_out / stem + ".input.mlirbc") }}
    {{ quote(metal_py) }} {{ quote(metal_compiler / "mlx_kernels.py") }} {{ quote(mlx_lib) }} {{ quote(bioclip_out / stem + ".mlx.spec.mlir") }} {{ rules }} {{ mlx_arch_flag }}
    {{ quote(metal_iree_compile) }} {{ quote(bioclip_out / stem + ".input.mlirbc") }} --compile-from=input --iree-hal-target-device=metal --iree-preprocessing-transform-spec-filename={{ quote(bioclip_out / stem + ".mlx.spec.mlir") }} -o {{ quote(bioclip_out / stem + ".mlx.metal.vmfb") }}
    {{ quote(metal_py) }} {{ quote(metal_compiler / "compare.py") }} {{ quote(bioclip_out) }} {{ quote(bioclip_datasets) }} {{ quote(bioclip_out / stem + ".mlx.metal.vmfb") }}

_bioclip-mlx-profile stem:
    {{ quote(metal_py) }} {{ quote(metal_compiler / "dispatch_profile.py") }} {{ quote(bioclip_out / stem + ".input.mlirbc") }} {{ quote(bioclip_out / "profile." + stem) }} --compiler {{ quote(metal_iree_compile) }} --compile-from=input --iree-hal-target-device=metal --iree-preprocessing-transform-spec-filename={{ quote(bioclip_out / stem + ".mlx.spec.mlir") }}

# Needs `just metal-iree-compiler` first.
[doc("Compile the BioCLIP 2 encoder with MLX linears and attention on Metal and compare it against the fp32 reference")]
[macos]
bioclip-metal-mlx: (_bioclip-mlx "vision.b1.r224.f16" bioclip_mlx_rules)

[doc("Same with the decomposed export: attention as MLX GEMMs, softmax on IREE codegen")]
[macos]
bioclip-metal-mlx-gemm: (_bioclip-mlx "vision.b1.r224.f16.decomposed" bioclip_mlx_gemm_rules)

[doc("Per-dispatch GPU time breakdown of the MLX build (after bioclip-metal-mlx)")]
[macos]
bioclip-metal-mlx-profile: (_bioclip-mlx-profile "vision.b1.r224.f16")

# --- iPhone -----------------------------------------------------------------
# The encoder runs in backends/ios/apps/bench_bioclip, a small app that times
# it and writes the embeddings, which compare.py then scores on the Mac. The
# device needs Developer Mode, and the app's signing team trusted on it
# (Settings > General > VPN & Device Management) after the first install.

ios_runtime_build := root / "third_party" / "iree-build-ios"
ios_runtime_patch := metal_compiler / "patches" / "iree-ios-runtime.patch"
ios_app := root / "backends" / "ios" / "apps" / "bench_bioclip"
ios_bundle_id := "dev.iree-inference.bench-bioclip"
# Signing team ID (Xcode > Settings > Accounts) and device name or UDID
# (`xcrun devicectl list devices`).
ios_team := env("IREE_IOS_TEAM", "")
ios_device := env("IREE_IOS_DEVICE", "")
# The app prints the GPU's architecture name; applegpu_g18p is the A19 Pro.
ios_gpu_arch := "applegpu_g18p"
ios_lib := mlx_kernels_out / "mlx_kernels.ios.metallib"
ios_results := bioclip_out / "ios"

# The posix async platform is gated to Darwin, Linux and Android, and iOS
# reports CMAKE_SYSTEM_NAME iOS; iree-ios-runtime.patch adds it. Local-task
# stays on: drivers/init.c includes the Metal driver header inside its #if.
[doc("Cross-compile the IREE runtime for iOS arm64 into third_party/iree-build-ios")]
[macos]
ios-iree-runtime: _iree-clone
    test "$(git -C {{ quote(iree_src) }} rev-parse HEAD)" = "{{ iree_commit }}"
    git -C {{ quote(iree_src) }} submodule update --init --depth 1 {{ iree_submodules }}
    git -C {{ quote(iree_src) }} apply --reverse --check {{ quote(ios_runtime_patch) }} 2>/dev/null || git -C {{ quote(iree_src) }} apply {{ quote(ios_runtime_patch) }}
    cmake -G Ninja -S {{ quote(iree_src) }} -B {{ quote(ios_runtime_build) }} -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=18.0 {{ iree_cmake_flags }}
    ninja -C {{ quote(ios_runtime_build) }} -j4 iree_runtime_unified flatcc_parsing

# Needs `just metal-iree-compiler`, `just ios-iree-runtime` and the SDPA
# export's input phase from `just bioclip-metal-mlx`.
[doc("Run the BioCLIP 2 encoder with MLX kernels on an iPhone and compare it against the fp32 reference")]
[macos]
bioclip-ios-mlx: (_mlx-metallib "iphoneos" ios_lib)
    test -n "{{ ios_team }}" -a -n "{{ ios_device }}" || { echo "set IREE_IOS_TEAM and IREE_IOS_DEVICE"; exit 1; }
    {{ quote(metal_py) }} {{ quote(metal_compiler / "mlx_kernels.py") }} {{ quote(ios_lib) }} {{ quote(bioclip_out / "vision.b1.r224.f16.mlx.ios.spec.mlir") }} {{ bioclip_mlx_rules }} --arch {{ ios_gpu_arch }}
    mkdir -p {{ quote(ios_app / "Model") }} {{ quote(ios_results) }}
    {{ quote(metal_iree_compile) }} {{ quote(bioclip_out / "vision.b1.r224.f16.input.mlirbc") }} --compile-from=input --iree-hal-target-device=metal --iree-metal-target-platform=ios --iree-preprocessing-transform-spec-filename={{ quote(bioclip_out / "vision.b1.r224.f16.mlx.ios.spec.mlir") }} -o {{ quote(ios_app / "Model" / "model.vmfb") }}
    {{ quote(metal_py) }} -c "import numpy as np, sys; np.load(sys.argv[1]).astype(np.float16).tofile(sys.argv[2])" {{ quote(bioclip_out / "pixels.npy") }} {{ quote(ios_app / "Model" / "pixels.f16") }}
    cd {{ quote(ios_app) }} && BENCH_TEAM={{ ios_team }} BENCH_BUNDLE_ID={{ ios_bundle_id }} xcodegen generate --quiet
    xcodebuild -project {{ quote(ios_app / "BenchBioclip.xcodeproj") }} -scheme BenchBioclip -configuration Release -destination "id=$(xcrun devicectl list devices | awk -v d={{ quote(ios_device) }} 'index($0, d) { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9A-F]{8}-[0-9A-F]{16}$/) print $i }' | head -1)" -derivedDataPath {{ quote(ios_app / "build") }} -allowProvisioningUpdates -quiet build
    xcrun devicectl device install app --device {{ quote(ios_device) }} {{ quote(ios_app / "build" / "Build" / "Products" / "Release-iphoneos" / "BenchBioclip.app") }} > /dev/null
    xcrun devicectl device process launch --device {{ quote(ios_device) }} --console --terminate-existing {{ ios_bundle_id }} | grep BENCH
    for f in embeddings.bin result.json; do xcrun devicectl device copy from --device {{ quote(ios_device) }} --domain-type appDataContainer --domain-identifier {{ ios_bundle_id }} --source "Documents/$f" --destination {{ quote(ios_results) }}/"$f" > /dev/null; done
    {{ quote(metal_py) }} {{ quote(metal_compiler / "compare.py") }} {{ quote(bioclip_out) }} {{ quote(bioclip_datasets) }} {{ quote(ios_results / "result.json") }}
