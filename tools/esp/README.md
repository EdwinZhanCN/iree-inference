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
stay with IREE. Each replacement gets channels-last weights and a parameter
array for one import, `espdl_conv2d_s8`. That import covers 1×1 and k×k
convolutions (stride, padding, dilation, groups), depthwise, 1D convolutions,
matrix products with constant weights, and an optional ReLU.

A transform spec (`--iree-preprocessing-transform-spec-filename`) was tried
first. Its matcher compares attributes exactly, so every stride, padding and
kernel size needed its own pattern, and it cannot check scale values.

The import is resolved at run time:

- On the host, `espdl/host_plugin.c` is an IREE executable plugin over C
  reference implementations (`espdl/espdl_ref.c`). `espdl/espdl_ref.h` is the
  contract: layouts, parameters, round half to even.
- On ESP32-P4, the same import name will resolve to an adapter over esp-dl.

```sh
just esp-ppocr-espdl rec   # host: IREE + reference kernels vs ORT
just esp-ppocr-p4 rec      # P4 static library with esp-dl dispatches
just esp-ppocr-p4 rec iree # P4 static library, IREE codegen only
```

| Model | Ops to esp-dl | Host result | P4 `.text` (IREE only → esp-dl) |
| --- | --- | --- | --- |
| rec | 35 (21 1×1, 2 3×3/s2, 9 depthwise, 1 Conv1D, 2 MatMul) | bit-exact with ORT and ESP-PPQ | 7,763 KiB → 161 KiB |
| det | 65 (48 conv, 17 depthwise 3×3/5×5) | bit-exact with ORT and ESP-PPQ | not measured |

The rest of the P4 code is elementwise ops (activations, Add, Mul, the
NCHW↔NHWC transposes around each dispatch) and small matrix-vector products.

Known workarounds:

- The host build uses `--iree-llvmcpu-target-cpu=generic`. IREE 3.13
  crashes on Apple Silicon when it lowers i8mm contractions for rec.
- ESP32-P4 uses the static-library mode. The embedded-ELF mode cannot
  resolve `__fixsfdi` for det, which needs it for Resize coordinates.
