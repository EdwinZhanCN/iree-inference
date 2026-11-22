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
stay with IREE. Three imports cover what it replaces:

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
  imports over esp-dl's `dl::base` kernels, single core. It wraps IREE's
  buffers in `TensorBase` views without copying: the rewriter already packs
  weights in esp-dl's ESP32-P4 filter layout (blocks of 16 output channels,
  as ESP-PPQ exports them), and `espdl/espdl_imports.h` holds the argument
  structs both backends unpack.

```sh
just esp-ppocr-espdl rec   # host: IREE + reference kernels vs ORT
just esp-ppocr-p4 rec      # P4 static library with esp-dl dispatches
just esp-ppocr-p4 rec iree # P4 static library, IREE codegen only
```

| Model | Ops to esp-dl | Transposes left | Host result | P4 `.text` |
| --- | --- | --- | --- | --- |
| rec | 72: 38 conv/MatMul, 22 LUT, 12 Add/Mul | 3 | bit-exact with ORT and ESP-PPQ | 18 KiB (IREE only: 7,763 KiB) |
| det | 162: 85 conv, 42 LUT, 35 Add/Mul | 7 | bit-exact with ORT and ESP-PPQ | 58 KiB |

What stays with IREE: the global average pools of the SE blocks and
AveragePool (reductions), Resize, Concat, MaxPool, the two ConvTranspose of
det (lowered `InsertZeros`), and rec's float depthwise Conv1D after the
AveragePool (its input is not quantized in the export). The transposes left
are the model input and output, and around those last two.

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
