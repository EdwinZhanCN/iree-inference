# ESP32-P4 backend

Scripts for running PP-OCRv6 through IREE on ESP32-P4 with esp-dl's
quantization rules. ESP-PPQ quantizes each model (esp32p4, w8a8, symmetric,
power-of-two scales) and exports it as QDQ ONNX. IREE then imports and
compiles it. They need `uv`, and a network connection on the first run.

Run from the repository root.

```sh
just esp-ppocr det      # quantize, export QDQ, compile for host, compare
just esp-ppocr rec
```

The [ESP32-P4 compiler scripts](../../compiler/targets/esp32p4/) use `.venv-esp`.
Models and outputs go to `third_party/ppocr-v6-tiny/` and
`third_party/esp-out/<det|rec>/`.

| Script | What it does |
| --- | --- |
| `quant_export.py` | Rewrites the graph like Espressif's released models, quantizes with ESP-PPQ, writes `.espdl`, QDQ ONNX and ESP-PPQ's simulated outputs |
| `espdl_rewrite.py` | Replaces supported operators with esp-dl kernel dispatches |
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

`espdl_rewrite.py` replaces quantized ops with dispatches that call
esp-dl kernel imports. It runs on IREE's input-phase IR
(`iree-compile --compile-to=input`), reads the int8 weights, bias and scales
from the constants, and checks them against esp-dl's scheme: zero point 0,
power-of-two scales, bias in the accumulator's scale. Ops that fail a check
stay with IREE. Six imports cover what it replaces:

- `espdl_conv2d_s8`: 1×1 and k×k convolutions (stride, padding, dilation,
  groups), depthwise, 1D convolutions (including unit-dim `collapse_shape` /
  `expand_shape` wrappers around depthwise 1D convs), matrix products with
  constant weights, optional ReLU. Weights are passed channels-last with a
  parameter array.
- `espdl_lut_s8`: any chain of elementwise ops from one int8 tensor to a
  quantize (GELU, HardSwish, HardSigmoid, Sigmoid, requantization), as
  esp-dl's 256-entry LUT. The script fills each table by running the chain
  on -128..127 with IREE on the host, so the table holds what IREE computed
  for those ops before.
- `espdl_binary_s8`: Add and Mul of two int8 tensors (residual Add, SE
  channel Mul, bias Add), the second one repeating along trailing dims.
  With power-of-two scales the exact integer result rounded half to even
  equals the float QDQ result.
- `espdl_avg_pool2d_s8`: valid-padded 2D average pooling and global spatial
  average reductions (`H×W -> 1×1`) with power-of-two input/output scales.
  On ESP32-P4, 16-aligned channel counts use esp-dl's PIE
  `dl_esp32p4_s8_avg_pool2d_hwc_sum` for the exact `int32` sum, followed by
  float division by `area` and RV32F `fcvt.w.s ..., rne` to match ORT-QDQ /
  MLIR rounding bit-for-bit.
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

- On the host, [`reference/host_plugin.c`](reference/host_plugin.c) is an IREE
  executable plugin over C reference implementations. [`abi/espdl_ref.h`](abi/espdl_ref.h)
  specifies the layouts, parameters and rounding.
- On ESP32-P4, [`components/espdl_adapter/`](components/espdl_adapter/) implements the same
  imports over esp-dl's `dl::base` kernels and RV32F helpers, single core. It
  wraps IREE's buffers without copying (with a 128-slot `ConvArgsType`
  setup cache for `conv2d_s8` and stack-built args for `binary_s8`): the
  rewriter already packs weights in esp-dl's ESP32-P4 filter layout (blocks
  of 16 output channels, as ESP-PPQ exports them), and [`abi/espdl_imports.h`](abi/espdl_imports.h)
  holds the argument structs both implementations unpack.

## Firmware and regression tests

Use `idf-next` for the firmware projects. The kernel test needs both models'
input-phase MLIR, and the rec firmware also needs rec's P4 dispatch library.

```sh
just esp-ppocr-espdl rec   # host: IREE + reference kernels vs ORT
just esp-ppocr-espdl det
just esp-ppocr-p4 rec      # P4 static library with esp-dl dispatches
just iree                  # host build supplies iree-flatcc-cli
just esp-p4-kernel-test    # build, flash and compare kernel vectors
just esp-p4-rec            # build, flash, compare bytes and report timings
```

[`apps/kernel_test/`](apps/kernel_test/) compares esp-dl kernels against
generated C reference vectors. [`apps/bench_rec/`](apps/bench_rec/) embeds
the rec VMFB, static dispatch library, and a host-generated input/output pair.
The `idf.toml` files select the ESP32-P4 revision and partition tables.
`just esp-p4-rec` also cross-compiles the runtime under
`third_party/iree-build-esp32p4-3.14/`. The reference plugin builds under
`third_party/esp32p4-reference-build/`.

Use `just esp-ppocr-p4 rec iree` to build the IREE-only dispatch baseline.

The [PP-OCRv6 benchmark record](../../docs/benchmarks/esp32p4_ppocr.md)
contains dispatch coverage, precision results and the latency breakdown.

Known workarounds:

- The host build uses `--iree-llvmcpu-target-cpu=generic`. IREE 3.13
  crashes on Apple Silicon when it lowers i8mm contractions for rec.
- ESP32-P4 uses the static-library mode. The embedded-ELF mode cannot
  resolve `__fixsfdi` for det, which needs it for Resize coordinates.
