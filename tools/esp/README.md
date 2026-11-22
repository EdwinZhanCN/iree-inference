# ESP32-P4 model tooling

Scripts for running PP-OCRv6 through IREE on ESP32-P4 with esp-dl's
quantization rules. ESP-PPQ quantizes each model (esp32p4, w8a8, symmetric,
power-of-two scales) and exports it as QDQ ONNX. IREE then imports and
compiles it. They need `uv`, and a network connection on the first run.

```sh
just esp-ppocr det      # quantize, export QDQ, compile for host, compare
just esp-ppocr rec
just esp-ppocr-p4 rec   # ESP32-P4 static library + code-size report
```

Tools are installed into `.venv-esp`. Models and outputs go to
`third_party/ppocr-v6-tiny/` and `third_party/esp-out/<det|rec>/`.

| Script | What it does |
| --- | --- |
| `quant_export.py` | Rewrites the graph like Espressif's released models, quantizes with ESP-PPQ, writes `.espdl`, QDQ ONNX and ESP-PPQ's simulated outputs |
| `espdl/espdl_rewrite.py` | Replaces quantized convolutions and matrix products with esp-dl kernel dispatches (see below) |
| `fix_qdq.py` | Lowers what ESP-PPQ's QDQ export leaves non-standard (`InsertZeros`, `Gelu`, `HardSwish`) to opset 13 |
| `compare_iree_ort.py` | Compares ESP-PPQ simulation, ONNX Runtime (QDQ) and IREE on the same inputs; for rec, decodes text and measures line accuracy |
| `inspect_qdq.py` | Checks a QDQ graph: Q/DQ dtypes, power-of-two scales, unquantized ops, ORT load |
| `p4_size.py` | Code-size report of a static-library `.o`: sections, undefined symbols, largest dispatches |
| `synth.py` | Synthetic text images, preprocessed like PP-OCR |

Calibration and comparison use synthetic text images, preprocessed as
PaddleOCR does (det: ImageNet mean/std; rec: `x/127.5 - 1`, text line resized
to height 48, zero-padded). They check the pipeline; they do not replace a
real dataset.

## Graph rewrites

The Hugging Face ONNX models are Paddle2ONNX opset 11 exports. Before
quantization, `quant_export.py` rewrites them so the `.espdl` has the same
operators as Espressif's released `pp_ocr_v6_*_s8.espdl` (all int8):

- Decomposed GELU (Div, Erf, Add, Mul, Mul) becomes one `Gelu`, and
  `x * HardSigmoid(x)` one `HardSwish`. ESP-PPQ quantizes each as an int8
  LUT; decomposed, the Erf runs in float on esp-dl.
- rec: Paddle's Conv1D (Unsqueeze, 1×k Conv, Squeeze) becomes a 1D Conv, so
  the BatchNormalization after it folds into the Conv.
- rec: the output Softmax is removed; the model outputs logits, and CTC
  decoding takes the argmax.

Activations are calibrated with the `percentile` method on 256 images. On
synthetic rec lines it gave 88.5% exact lines (fp32: 91%), against 77.5% for
ESP-PPQ's default `kl`.

## esp-dl kernels

`espdl/espdl_rewrite.py` replaces quantized ops with dispatches that call
esp-dl kernel imports. It runs on IREE's input-phase IR
(`iree-compile --compile-to=input`), reads the int8 weights, bias and scales
from the constants, and checks them against esp-dl's scheme: zero point 0,
power-of-two scales, bias in the accumulator's scale. Ops that fail a check
stay with IREE. Five imports cover what it replaces:

- `espdl_conv2d_s8`: 1×1 and k×k convolutions (stride, padding, dilation,
  groups), depthwise, 1D convolutions, matrix products with constant
  weights, optional ReLU. Weights are passed channels-last with a parameter
  array.
- `espdl_lut_s8`: any chain of elementwise ops from one int8 tensor to a
  quantize (GELU, HardSwish, HardSigmoid, Sigmoid, requantization), as
  esp-dl's 256-entry LUT. The script fills each table by running the chain
  on -128..127 with IREE on the host, so the table holds what IREE computed
  for those ops before.
- `espdl_binary_s8`: Add and Mul of two int8 tensors (residual Add, SE
  channel Mul, bias Add), the second one repeating along trailing dims.
  With power-of-two scales the exact integer result rounded half to even
  equals the float QDQ result.
- `espdl_quant_s8`: boundary `f32 -> i8` quantization (power-of-two scale,
  round half to even, zero point 0) fused with optional `NCHW -> NHWC`
  transpose (`[outer, c, hw] -> [outer, hw, c]`).
- `espdl_dequant_s8_f32`: boundary `i8 -> f32` dequantization
  (`(float)x[i] * 2^x_exp`) for large model outputs.

The kernels are channels-last (NHWC); the imported model is channels-first.
The script does not transpose around each kernel. It carries each kernel's
NHWC result forward: elementwise ops, reductions, quantize/dequantize,
Resize, Concat, Pad and pooling that read it are rewritten to NHWC (the
iteration space stays, only indexing maps, dims and pads are permuted). A
transpose is left only where an op needs NCHW. IREE's own transpose
propagation does not do this: it stops at `quantize_affine` and
`dequantize_affine`.

A transform spec (`--iree-preprocessing-transform-spec-filename`) was tried
first. Its matcher compares attributes exactly, so every stride, padding and
kernel size needed its own pattern, and it cannot check scale values.

The import is resolved at run time:

- On the host, `espdl/host_plugin.c` is an IREE executable plugin over C
  reference implementations (`espdl/espdl_ref.c`). `espdl/espdl_ref.h` is the
  contract: layouts, parameters, round half to even.
- On ESP32-P4, `espdl/espdl_p4` (an ESP-IDF component) implements the same
  imports over esp-dl's `dl::base` kernels and RV32F helpers, single core. It
  wraps IREE's buffers without copying (with a 128-slot `ConvArgsType`
  setup cache for `conv2d_s8` and stack-built args for `binary_s8`): the
  rewriter already packs weights in esp-dl's ESP32-P4 filter layout (blocks
  of 16 output channels, as ESP-PPQ exports them), and `espdl/espdl_imports.h`
  holds the argument structs both backends unpack.

```sh
just esp-ppocr-espdl rec   # host: IREE + reference kernels vs ORT
just esp-ppocr-p4 rec      # P4 static library with esp-dl dispatches
just esp-ppocr-p4 rec iree # P4 static library, IREE codegen only
```

| Model | Ops to esp-dl | Transposes left | Host result | P4 `.text` |
| --- | --- | --- | --- | --- |
| rec | 74: 38 conv/MatMul, 22 LUT, 12 Add/Mul, 1 quant_nhwc, 1 dequant_f32 | 2 | bit-exact with ORT and ESP-PPQ | 16 KiB (IREE only: 7,763 KiB) |
| det | 162: 85 conv, 42 LUT, 35 Add/Mul | 7 | bit-exact with ORT and ESP-PPQ | 58 KiB |

What stays with IREE: the global average pools of the SE blocks and
AveragePool (reductions), Resize, Concat, MaxPool, the two ConvTranspose of
det (lowered `InsertZeros`), and rec's float depthwise Conv1D after the
AveragePool (its input is not quantized in the export). On rec, the model
input `quantize_affine + to_nhwc` and model output `dequantize_affine` are
handled by `espdl_quant_s8` and `espdl_dequant_s8_f32`; the 2 remaining
transposes wrap the unquantized float depthwise Conv1D.

`just esp-p4-kernel-test` checks the ESP32-P4 backend on a board. It takes
every kernel configuration of both models (spatial sizes cropped), fills it
with random data, computes the expected bytes with the reference kernels and
builds `espdl/espdl_p4_test`, which runs the esp-dl backend and compares. On
an ESP32-P4 rev 1.3 all 76 cases (55 convolutions, 17 Add/Mul, 4 LUT) are
bit-exact. The firmware is an idf-next project: `idf.toml` sets the chip
revision below 3.0 and a custom partition table, whose path is given
relative to the project idf-next generates under `build/`.

Known workarounds:

- The host build uses `--iree-llvmcpu-target-cpu=generic`. IREE 3.13
  crashes on Apple Silicon when it lowers i8mm contractions for rec.
- ESP32-P4 uses the static-library mode. The embedded-ELF mode cannot
  resolve `__fixsfdi` for det, which needs it for Resize coordinates.

## ESP32-P4 `rec` optimization log

Measured on ESP32-P4 rev 1.3 @ 360 MHz (`firmware/rec_p4`), warm runs (run 1–2),
`0 of 276240` output values differing from host across all stages:

| Stage | Flash XIP | PSRAM Copy | Key change |
| --- | --- | --- | --- |
| Baseline (`-Og`) | ~456 ms | 267.8 ms | Initial `esp-dl` integration |
| Compiler `-O2` | ~456 ms | 265.9 ms | `COMPILER_OPTIMIZATION_PERF = true` in `rec_p4/idf.toml` |
| Slab `calloc` bypass + setup elimination | 390.2 ms | 199.95 ms | `alloc` `19.64 -> 0.16 ms`; `conv/bin` setup `27.77 -> 1.56 ms` |
| Boundary `quant_s8` + `dequant_s8_f32` | **376.3 ms** | **185.4 ms** | `quant` `10.84 -> 4.26 ms`; `dequant` `23.54 -> 15.49 ms` |

Total PSRAM-copy speedup: **-82.4 ms (-30.8%)**.

Details of the three optimizations:

1. **Skip PSRAM payload zero-fill in HAL heap slab allocator (`firmware/rec_p4/src/main.c`)**:
   `iree_hal_allocator_create_heap` zeroes ~1.88 MiB of temporary and output PSRAM buffers per run
   via `IREE_ALLOCATOR_COMMAND_CALLOC`. Passing the custom `device_slab_alloc_ctl` as both
   `data_allocator` and `host_allocator` (single-allocator `SLAB` mode, avoiding upstream
   `buffer_heap.c`'s `SPLIT`-mode aligned-alloc / unaligned-free mismatch) lets allocations `> 256` B
   use `malloc` and only `memset` the first 256-byte metadata header. This cut `alloc/free` from
   `19.64 ms` to `0.17 ms` and reduced L2 cache pollution across subsequent dispatches.
2. **Eliminate per-layer heap/setup overhead in `espdl_p4` (`espdl/espdl_p4/espdl_p4.cpp`)**:
   - `espdl_p4_binary_s8`: constructs `dl::base::elemwiseArgsType<int8_t>` directly on the stack
     with zero `TensorBase` or `std::vector` heap allocations (`4.60 -> 0.57 ms` setup).
   - `espdl_p4_conv2d_s8`: caches resolved `ConvArgsType` + kernel pointers in a 128-slot open-address
     table keyed by `(w, bias, w_exp, p)` and `params` (`23.17 -> 0.50 ms` warm setup). On cache hit,
     `base_args` is copied to the stack before updating `input_element`/`output_element` because
     `esp-dl`'s border/grid loops mutate `ConvArgsType` in place.
3. **Move boundary `quantize_affine` and `dequantize_affine` to `espdl` kernels**:
   - `espdl_quant_s8`: fuses the `1x3x48x320` input `quantize_affine` and `NCHW -> NHWC` transpose,
     using RV32F's single-instruction `fcvt.w.s ..., rne` (exact match for MLIR `math.roundeven`)
     and packing 4 RGB pixels into 3 aligned `uint32_t` stores (`10.84 -> 4.26 ms`).
   - `espdl_dequant_s8_f32`: precomputes a 256-entry `float` LUT (`1 KiB`, L1-resident) on the stack
     and writes the `1x40x6906` (`1.05 MiB`) output via 4x unrolled table lookups (`23.54 -> 15.49 ms`,
     bound by PSRAM write-allocate + writeback bandwidth).

Warm PSRAM-copy breakdown (`186.19 ms`, run 2):
- `espdl` dispatches (74 calls): `156.84 ms` (`conv2d_s8` 100.76 + 0.51 setup, `lut_s8` 19.74,
  `dequant_s8_f32` 15.48, `binary_s8` 12.64 + 0.61 setup, `quant_s8` 4.29, glue 2.82)
- Remaining IREE dispatches (6 calls): `10.04 ms` (3 SE reductions 2.56, `pooling_nhwc_sum` 4.28,
  float `depthwise_conv1d` 3.11, `transpose` 0.09)
- IREE VM + HAL scheduling: `19.31 ms` (`alloc` `0.17 ms`)
