# PP-OCRv6 on ESP32-P4

See the [ESP32-P4 build and validation guide](../../backends/esp32p4/README.md)
for the commands used to produce the model, plugin, and firmware.

## Dispatch coverage and correctness

| Model | Ops to esp-dl | Transposes left | Host result | P4 `.text` |
| --- | --- | --- | --- | --- |
| rec | 79: 39 conv/MatMul, 22 LUT, 12 Add/Mul, 4 avg_pool, 1 quant_nhwc, 1 dequant_f32 | 0 | bit-exact with ORT and ESP-PPQ | 2 KiB (IREE only: 7,763 KiB) |
| det | 178: 85 conv, 42 LUT, 35 Add/Mul, 13 avg_pool, 3 quant_nhwc | 4 | bit-exact with ORT and ESP-PPQ | 32 KiB |

What stays with IREE: on rec, **nothing** (all 79 compute dispatches go to
`esp-dl`, with 0 transposes and 0 IREE codegen dispatches). On det, Resize,
Concat, MaxPool, and the two float `ConvTranspose` layers (lowered
`InsertZeros`).

`just esp-p4-kernel-test` checks the ESP32-P4 backend on a board. It takes
every kernel configuration of both models (spatial sizes cropped), fills it
with random data, computes the expected bytes with the reference kernels and
builds [`kernel_test`](../../backends/esp32p4/apps/kernel_test/), which runs
the esp-dl backend and compares. On an ESP32-P4 rev 1.3 all 93 cases
(56 convolutions, 16 average pools,
17 Add/Mul, 4 LUT) are bit-exact. The firmware is an idf-next project:
`idf.toml` sets the chip revision below 3.0 and a custom partition table,
whose path is given relative to the project idf-next generates under
`build/`.

## `rec` optimization log

Measured on ESP32-P4 rev 1.3 @ 360 MHz ([`bench_rec`](../../backends/esp32p4/apps/bench_rec/)), warm runs (run 1–2),
`0 of 276240` output values differing from host across all stages:

| Stage | Flash XIP | PSRAM Copy | Key change |
| --- | --- | --- | --- |
| Baseline (`-Og`) | ~456 ms | 267.8 ms | Initial `esp-dl` integration |
| Compiler `-O2` | ~456 ms | 265.9 ms | `COMPILER_OPTIMIZATION_PERF = true` in [`bench_rec/idf.toml`](../../backends/esp32p4/apps/bench_rec/idf.toml) |
| Slab `calloc` bypass + setup elimination | 390.2 ms | 199.95 ms | `alloc` `19.64 -> 0.16 ms`; `conv/bin` setup `27.77 -> 1.56 ms` |
| Boundary `quant_s8` + `dequant_s8_f32` | 376.3 ms | 185.4 ms | `quant` `10.84 -> 4.26 ms`; `dequant` `23.54 -> 15.49 ms` |
| In-place LUT/binary + SRAM transient slab | 312.5 ms | 132.5 ms | Tied `lut_s8_ip`/`binary_s8_ip`; transient slab `596.9 -> 314.2 KB` in internal SRAM |
| Tail-op takeover (`avg_pool2d_s8` + `Conv.35__1d`) | 303.9 ms | 124.0 ms | `iree_disp` `9.93 -> 0.00 ms` (`79/79` esp-dl, `0` transposes, `.text` `16 -> 2 KiB`) |
| `esp-iree-p4` `-O3` (`Release`) | **303.9 ms** | **123.7 ms** | `justfile`: `-DCMAKE_BUILD_TYPE=Release`; `vm/hal` `9.99 -> 9.74 ms` |

Total PSRAM-copy speedup: **-144.1 ms (-53.8%)**.

Details of the optimizations:

1. **Skip PSRAM payload zero-fill in HAL heap slab allocator ([`bench_rec/src/main.c`](../../backends/esp32p4/apps/bench_rec/src/main.c))**:
   `iree_hal_allocator_create_heap` zeroes ~1.88 MiB of temporary and output PSRAM buffers per run
   via `IREE_ALLOCATOR_COMMAND_CALLOC`. Passing the custom `device_slab_alloc_ctl` as both
   `data_allocator` and `host_allocator` (single-allocator `SLAB` mode, avoiding upstream
   `buffer_heap.c`'s `SPLIT`-mode aligned-alloc / unaligned-free mismatch) lets allocations `> 256` B
   use `malloc` and only `memset` the first 256-byte metadata header. This cut `alloc/free` from
   `19.64 ms` to `0.17 ms` and reduced L2 cache pollution across subsequent dispatches.
2. **Eliminate per-layer heap/setup overhead in the [esp-dl adapter](../../backends/esp32p4/components/espdl_adapter/espdl_p4.cpp)**:
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
4. **In-place `lut_s8`/`binary_s8` and internal-SRAM transient slab ([`espdl_rewrite.py`](../../compiler/targets/esp32p4/espdl_rewrite.py), [`bench_rec/src/main.c`](../../backends/esp32p4/apps/bench_rec/src/main.c))**:
   - Previously, IREE packed all intermediate activations into a `596,916 B` `transient` slab
     (too large for ESP32-P4's 384 KiB contiguous internal SRAM block `0x4FF40000..0x4FFA0000`)
     because `MatMul.1` (`276,240 B`) and its untied bias `Add.106` (`276,240 B`) could not alias.
   - `espdl_rewrite.py` now emits tied `flow.dispatch ... -> %a0` variants (`lut_s8_ip` for 15/22 LUT
     calls, `binary_s8_ip` for 9/12 Add/Mul calls) whenever operand 0 has a single use, shrinking
     `stream.resource<transient>` from `596,736 B` to `314,048 B` (`314,228 B` allocated) with
     zero `stream.cmd.clone` copies.
   - `device_slab_alloc_ctl` routes invoke-time slabs into internal SRAM while preserving at least
     `64 KiB` (`SRAM_RESERVE_BYTES`) of internal heap headroom, placing the entire `314,228 B`
     transient slab in internal SRAM (`178 KiB` free after alloc, `144 KiB` min free) while
     keeping the `1,105,140 B` external output buffer in PSRAM.
5. **Take over remaining 6 IREE dispatches (`espdl_avg_pool2d_s8` + `Conv.35__1d` reshape unwrapping)**:
   - `Conv.35__1d` (`3.09 ms`) was actually already `int8`-quantized (`i8 x i8 -> i32 -> i8`),
     hidden behind unit-dim `tensor.collapse_shape` / `tensor.expand_shape` reshapes from the
     `Conv1D` lowering; unwrapping those unit reshapes in `espdl_rewrite.py` routes it (and drops
     the 2 surrounding `transpose` ops, `0.08 ms`) to `espdl_conv2d_s8`.
   - Added `espdl_avg_pool2d_s8` to match both `linalg.pooling_nchw_sum + divf` (`AveragePool.0`,
     `4.24 ms`) and global H/W `linalg.generic(addf) + divf` (3 SE reductions, `2.52 ms`), running
     in `1.37 ms` via PIE `dl_esp32p4_s8_avg_pool2d_hwc_sum` + RV32F `fcvt.w.s ..., rne`.
   - All 79 dispatches now run via `esp-dl` (`0.00 ms` IREE codegen, `rec.espdl.p4.o` `.text`
     `16 -> 2 KiB`), placing the `314,676 B` transient slab in internal SRAM (`178 KiB` free,
     `145 KiB` min free).
6. **Compile IREE runtime with `-O3` (`justfile`)**:
   - Switched `esp-iree-p4` from `-DCMAKE_BUILD_TYPE=MinSizeRel` (`-Os`) to `Release` (`-O3`),
     trimming `vm/hal` from `9.99 ms` to `9.74 ms` (`rec_p4.bin` `3.94 -> 3.99 MiB`, 50% partition free).

Warm PSRAM-copy breakdown (`123.7 ms`, run 2):
- `espdl` dispatches (79 calls): `113.91 ms` (`conv2d_s8` 82.02 incl. 0.19 setup, `dequant_s8_f32`
  13.32, `lut_s8_ip` 7.45 + `lut_s8` 2.93, `quant_s8` 3.97, `binary_s8_ip` 2.30 + `binary_s8` 0.56,
  `avg_pool2d_s8` 1.37)
- Remaining IREE dispatches (0 calls): `0.00 ms`
- IREE VM + HAL scheduling: `9.74 ms` (`alloc` `0.21 ms`, `12 alloc [1 sram, 1 psram]`)

For reference on the same ESP32-P4 rev 1.3 @ 360 MHz, Espressif's standalone `esp-dl 3.3.15`
`dl::Model` running `rec.espdl` (`param_copy=true`, `1-core`) measures `131.4 ms` with `sram=0`
and `102.3 ms` with `sram=200K` for pure `int8` inference (excluding `f32` boundary quant/dequant,
which takes `~17.3 ms` here).
