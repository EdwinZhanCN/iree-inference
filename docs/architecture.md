# Inference architecture

`crates/` provides the Rust host API over the IREE C runtime. The
[ESP32-P4 compiler tooling](../compiler/targets/esp32p4/) runs offline with
Python. The [ESP32-P4 firmware](../backends/esp32p4/apps/) links the IREE C runtime
directly and consumes the compiler's VMFB and static dispatch library.
The host builds provide local CPU drivers and enable the upstream Metal driver
on macOS. The [root justfile](../justfile) drives both build paths.

## Compilation and execution stages

| Stage | ESP32-P4 path |
| --- | --- |
| L1 · Model and quantization | `quant_export.py` prepares PP-OCRv6 for ESP-PPQ and exports W8A8 QDQ ONNX plus simulated outputs for verification. |
| L2 · Import | `fix_qdq.py` lowers ESP-PPQ-specific operators. IREE imports the ONNX graph and emits input-phase MLIR. |
| L3 · Dispatch selection | `espdl_rewrite.py` matches compatible quantized operations and emits calls to esp-dl imports. IREE lowers the remaining operations. |
| L4 · Layout | The rewriter propagates NHWC where supported and packs constant filters for the ESP32-P4 kernels. |
| L5 · Memory planning | The rewriter ties dispatch results to input buffers when the kernel supports aliasing and the operand has one use. IREE Stream plans the resulting resource lifetimes and transient storage. |
| L6 · Execution | The IREE VM runs the VMFB and the local-sync HAL loads the static dispatch library. The P4 runtime component supplies a proactor, and the rec firmware routes transient allocations between SRAM and PSRAM. |
| L7 · Kernels | The firmware adapter calls esp-dl. On the host, an IREE executable plugin resolves the same imports against C reference kernels. |

## Build and verification boundaries

- The [ESP32-P4 ABI headers](../backends/esp32p4/abi/) specify the dispatch
  argument structs and parameter layouts consumed by the adapter and the host
  reference. Changes to these layouts also need matching compiler emission.
- IREE's pinned compiler and runtime build together. A P4 deployment pairs
  the VMFB with the static library from the same compilation. The host
  build of a rewritten model uses the same imports and the host reference plugin.
- The PP-OCRv6 tools compare host outputs with ONNX Runtime and ESP-PPQ.
  Firmware checks compare kernel vectors and the complete rec output against
  bytes generated from the host reference path. See the [accuracy and latency
  record](benchmarks/esp32p4_ppocr.md) for hardware and measurement details.
