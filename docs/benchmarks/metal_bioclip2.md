# BioCLIP 2 on Metal

The BioCLIP 2 image encoder (open_clip ViT-L/14, f16 with f32 LayerNorm,
batch 1, 224 px) runs on IREE's Metal HAL. Its GEMMs and attention run as
MLX steel kernels through `hal.dispatch.extern`; everything else uses IREE's
metal-spirv codegen. The [Metal compiler tooling](../../compiler/targets/metal/)
and the [root justfile](../../justfile) produce every number below.

## Pipeline

1. `export.py` exports the encoder from upstream safetensors to IREE MLIR.
   It writes an SDPA export and a decomposed export.
2. The pip `iree-compile` lowers the export to the input phase.
3. `mlx_kernels.py` writes a preprocessing transform spec. Each rule matches
   one op chain and replaces it with an extern dispatch into
   `mlx_kernels.metallib`:
   - `--linear MxNxK`: transpose(W) + batch-1 broadcast + fill + batch_matmul
     → the nt GEMM on the original W.
   - `--attention BxLxHxD`: head split + `iree_linalg_ext.attention` + head
     merge → MLX steel attention, which reads `[B, L, H, D]` through strides.
   - `--gemm BxMxNxK`: fill + batch_matmul → the nn GEMM. Used only for the
     decomposed export.
4. The compiler from the pinned IREE fork compiles from the input phase with
   that spec. The fork adds external executables to the MetalSPIRV target.
   `just metal-iree-compiler` builds it.

The measurements below were taken with IREE 3.13.0 and local patches.
The build recipes now use the fork snapshot after 3.14.0.

The kernel sources are MLX's own code, vendored at a pinned commit.
[`mlx_gemm.metal`](../../compiler/targets/metal/kernels/mlx_gemm.metal) keeps the
steel GEMM body. [`gen_mlx_kernels.py`](../../compiler/targets/metal/kernels/gen_mlx_kernels.py)
extracts three kernel bodies unchanged: steel attention for
[`mlx_attention.metal`](../../compiler/targets/metal/kernels/mlx_attention.metal),
its NAX version for the same file built with `-DMLX_NAX`, and the NAX steel
GEMM for [`mlx_gemm_nax.metal`](../../compiler/targets/metal/kernels/mlx_gemm_nax.metal).
MLX's function constants become template parameters or fixed locals, because
IREE cannot set them. Kernels are compiled with MLX's flags, including
`-fno-fast-math`.

NAX kernels use the Metal 4 tensor ops of GPUs with neural accelerators.
They need iOS 26.2 and are built only into the iOS metallib. As in MLX's
`is_nax_available()`, they are used when the architecture generation is at
least 18 on a phone (A19) or 17 elsewhere (M5). `just ios_nax=off` forces the
regular kernels.

Tiles and swizzle follow MLX's own selection (`GEMM_TPARAM_MACRO` and
`sdpa_full`) by device class, transposition and shape. MLX picks at run time
from the GPU architecture name. IREE compiles ahead of time, so the spec is
written for this machine's GPU, or for `just metal_gpu_arch=<name>` on a
cross-compile. Every tile MLX can pick is instantiated.

| Command | What it does |
| --- | --- |
| `just bioclip-metal-baseline` | Decomposed export on IREE's default Metal path |
| `just bioclip-metal-mlx` | SDPA export with MLX linears and attention |
| `just bioclip-metal-mlx-gemm` | Decomposed export with attention as MLX GEMMs |
| `just bioclip-metal-mlx-profile` | Per-dispatch breakdown of the MLX build |
| `just mlx-kernel-test` | MLX kernels against NumPy and IREE's matmul |
| `just ios-iree-runtime` | Cross-compiles the IREE runtime for iOS |
| `just bioclip-ios-mlx` | MLX build on an iPhone (NAX where MLX would use it) |
| `just bioclip-ios-coreml` | The Core ML package on the same iPhone |

The iPhone recipes need `IREE_IOS_TEAM` (signing team) and
`IREE_IOS_DEVICE` (device name or id). They build
[`backends/ios/apps/bench_bioclip`](../../backends/ios/apps/bench_bioclip/),
run it with the phone unlocked, copy the embeddings back and score them with
`compare.py`. The app records the iOS thermal state at the start and end of
each run.

## Correctness

`compare.py` runs 40 iNaturalist photos and checks two things against an
fp32 PyTorch reference: the embedding cosine, and zero-shot retrieval
against the TreeOfLife Core index (669,871 entries).

| Build | Cosine min | Top-1 same |
| --- | --- | --- |
| IREE default (decomposed) | 0.99999 | 40/40 |
| MLX linears + attention | 0.99999 | 40/40 |

On the MLX build, the top-5 set matches on 39/40 photos. The one swap is
between two candidates with equal scores (gap 0.0000).

`just mlx-kernel-test` checks each kernel against NumPy:

- Attention at 1x257x16x64 and 2x64x4x64: max relative error 2.2e-4 and 2.4e-4.
- GEMMs on the BioCLIP 2 shapes plus one unaligned shape: max relative error
  at most 4.0e-4, the same as IREE's own matmul.
- With `--arch` forced to the small (g), medium (s) and large (d) classes,
  each of the four GEMM tiles MLX can pick was selected at least once and
  passed.

## Optimization log

Measured on an Apple M2 Pro (macOS 27.2) with the call median from
`compare.py`. The profile column is the dispatch total from
`dispatch_profile.py`: each executable is benchmarked alone and weighted by
its dispatch count, because Metal HAL has no device profiling.

| Stage | Call median | Profile | Key change |
| --- | --- | --- | --- |
| IREE default (decomposed) | 283 ms | 286 ms | metal-spirv cannot compile `iree_linalg_ext.attention`, so this build uses the decomposed export |
| MLX dense GEMMs | 163 ms | | Linears on MLX nn GEMMs; attention left on IREE |
| + attention batched GEMMs | 76 ms | | QKᵀ and PV on MLX nn GEMMs too |
| Linear weights untransposed | 67.6 ms | | `--linear` rule + nt GEMM; load `0.1-0.5 s -> ~30 ms`, about 576 MB less memory |
| Hand-tuned nt tile 32x32x32 | 60.6 ms | 62.9 ms | Fastest on this M2 Pro; dropped later |
| MLX steel attention (SDPA export) | 52.1 ms | 56.1 ms | Attention `~15 -> 3.0 ms`; head transposes gone |
| MLX tile selection | **57.4 ms** | **58.8 ms** | Tiles and swizzle as MLX picks them, for any GPU |

Total speedup over IREE's default path: **-225.6 ms (-79.7%)**.

Details:

1. **External Metal executables (IREE patch).** An external variant's object
   is a `.metallib` or MSL source. Each export names a kernel and gives its
   threadgroup size. Bindings use the generated-code ABI: an argument buffer
   at `buffer(0)`, constants at `buffer(3)`.
2. **MLX GEMMs.** On the default path, dense GEMMs took 62% of the time at
   about 0.86 TFLOPS, and attention took 35%, mostly QKᵀ at 0.04 TFLOPS. With
   MLX steel GEMMs the dense layers run at 3.0-3.4 TFLOPS. IREE still fuses
   the truncf, bias and residual epilogues after the extern dispatch.
3. **Untransposed weights.** A torch linear reaches preprocessing as
   transpose(W) + broadcast + batch_matmul. IREE folds the weight transpose at
   compile time, but not when the consumer is an extern dispatch. So all 144
   weights were transposed at module load. The `--linear` rule matches the
   whole chain and calls the nt kernel on W as stored, leaving no initializer
   dispatches.
4. **Per-shape profiling.** IREE deduplicates identical executables, so GEMM
   shapes that share one kernel shared one benchmark. Each extern dispatch
   now carries its problem size as the workload, which IREE keys dumped
   benchmarks on.
5. **MLX steel attention.** One dispatch per layer replaces softmax (7.7 ms),
   the attention GEMMs (4.2 ms) and the head transposes (about 3 ms). It runs
   24 x 0.13 ms.
6. **MLX tile selection.** The hand-tuned 32x32x32 nt tile was 5.3 ms faster
   on this M2 Pro but would need tuning on every GPU. Tile choice now follows
   MLX exactly, including its regular path's `swizzle_log = 0`.

Current breakdown (58.8 ms, 416 dispatches):

- MLX nt GEMMs (144 calls): `47.9 ms`
  - 1024x1024 (QKV and out projections, 96 calls): `17.3 ms`, 2.99 TFLOPS
  - MLP down 1024x4096 (24 calls): `15.5 ms`, 3.33 TFLOPS
  - MLP up 4096x1024 (24 calls): `15.1 ms`, 3.43 TFLOPS
- MLX attention (24 calls): `3.0 ms`
- IREE codegen (248 calls): `7.9 ms`, of which LayerNorm reductions are
  `2.7 ms`, elementwise epilogues `4.4 ms` and patch-embedding conv `0.5 ms`

GEMMs are 81% of the time. Folding bias and residual into the GEMM is
estimated to save only about 2-3 ms. For reference, the Core ML export runs
in about 36 ms on the same M2 Pro's Neural Engine.

After the switch to `-fno-fast-math` the M2 Pro build measured 60.8 ms, with
top-5 40/40. The 3 ms against 57.4 ms is within run-to-run variation here.

## iPhone

iPhone Air, A19 Pro GPU (`applegpu_g18p`), iOS 27.0.1. Each row is 30 timed
calls after 5 warmup calls, thermal state nominal at the start and end.

| Build | Call median | Load | Cosine min | Top-1 same | Top-5 same |
| --- | --- | --- | --- | --- | --- |
| MLX build, regular kernels (`ios_nax=off`) | 129.4 ms | 0.27 s | 0.99999 | 40/40 | 38/40 |
| MLX build, NAX GEMM + attention | **41.9 ms** | 0.31 s | 0.99999 | 40/40 | 40/40 |
| Core ML package, compute units all | 32.0 ms | 8.8 s first, 0.09 s cached | 0.99960 | 40/40 | 36/40 |

- NAX makes the build 3.1x faster, and it then beats the M2 Pro's 57.4 ms.
  MLX picks the 128x128x512 tile with 4x4 simdgroups for all three GEMM
  shapes on a phone, and bq 64 for attention.
- Core ML on the Neural Engine is about 10 ms faster, but less accurate. Its
  first load compiles for the Neural Engine; iOS caches the result.
- The phone throttles quickly. Back-to-back runs on a warm phone measured
  141-320 ms for the regular build, so compare only runs that start and end
  nominal.

iOS build notes:

- `iree-compile --iree-metal-target-platform=ios`. The MLX metallib is built
  with `xcrun -sdk iphoneos`, and all its objects target iOS 26.2: a metallib
  cannot link objects built for different OS versions.
- The fork enables SPIRV-Cross's `ios_use_simdgroup_functions` so LayerNorm
  reductions translate to MSL, and builds the POSIX async runtime for iOS.
