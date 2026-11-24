// MLX's NAX (Metal 4 tensor ops) steel GEMM behind IREE's Metal binding ABI,
// for hal.dispatch.extern on GPUs with neural accelerators (A19 / M5 on).
//
// The kernel body is MLX's (mlx/backend/metal/kernels/steel/gemm/kernels/
// steel_gemm_fused_nax.h, MIT, Copyright 2025 Apple Inc.), extracted by
// gen_mlx_kernels.py into steel_gemm_fused_nax_body.inc. MLX's function
// constants are template parameters (alignment) or constexpr false (batch
// broadcasting, addmm epilogue) under the same names.
//
// IREE ABI: bindings in an argument buffer at [[buffer(0)]], [[id(N)]] being
// the binding ordinal; 32-bit constants at [[buffer(3)]], which this kernel
// reads as MLX's GEMMParams (int64 fields as two little-endian words).
//
//   D[z] = op(A[z]) * op(B[z]),  op = transpose when the kernel name says t
//   bindings: 0 = A, 1 = B, 2 = D
//   threadgroup: (32, WN, WM); grid: (tiles_n << swizzle_log,
//   ceil(tiles_m >> swizzle_log), batch)

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"

using namespace metal;

// Function constants the extracted code reads at file scope.
constant constexpr bool has_batch = false;
constant constexpr bool use_out_source = false;
constant constexpr bool do_axpby = false;

#include "steel_gemm_fused_nax_ops.h"

template <typename T>
struct GemmBindings {
  const device T* a [[id(0)]];
  const device T* b [[id(1)]];
  device T* d [[id(2)]];
};

template <
    typename T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose_a,
    bool transpose_b,
    bool align_M,
    bool align_N,
    bool align_K>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm_nax(
    constant GemmBindings<T>& bindings [[buffer(0)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using AccumType = float;
  const device T* A = bindings.a;
  const device T* B = bindings.b;
  device T* D = bindings.d;
  // Unused: has_batch and use_out_source are false.
  const device T* C = nullptr;
  const constant GEMMAddMMParams* addmm_params = nullptr;
  const constant int* batch_shape = nullptr;
  const constant int64_t* batch_strides = nullptr;

#include "steel_gemm_fused_nax_body.inc"
}

// Names: mlx_gemm_nax_<nn|nt>_f16_bm<>_bn<>_bk<>_wm<>_wn<>_<M><N><K>, where
// each of M N K is a (aligned to the tile) or u.
#define instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, am, an, ak, aname) \
  template [[host_name("mlx_gemm_nax_" #tname "_f16_bm" #bm "_bn" #bn "_bk" #bk   \
                       "_wm" #wm "_wn" #wn "_" aname)]] [[kernel]] void            \
  gemm_nax<half, bm, bn, bk, wm, wn, ta, tb, am, an, ak>(                          \
      constant GemmBindings<half>&, const constant GEMMParams*, uint, uint3);

#define instantiate_gemm_nax_alignments(tname, ta, tb, bm, bn, bk, wm, wn)            \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, true, true, true, "aaa")    \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, false, true, true, "uaa")   \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, true, false, true, "aua")   \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, false, false, true, "uua")  \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, true, true, false, "aau")   \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, false, true, false, "uau")  \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, true, false, false, "auu")  \
  instantiate_gemm_nax(tname, ta, tb, bm, bn, bk, wm, wn, false, false, false, "uuu")

// Every tile MLX's steel_matmul_regular_axpby_nax picks
// (mlx_kernels.gemm_nax_tile), for both layouts.
#define instantiate_gemm_nax_tiles(tname, ta, tb)                    \
  instantiate_gemm_nax_alignments(tname, ta, tb, 128, 128, 512, 4, 4) \
  instantiate_gemm_nax_alignments(tname, ta, tb, 64, 128, 256, 2, 4)  \
  instantiate_gemm_nax_alignments(tname, ta, tb, 64, 128, 64, 2, 4)   \
  instantiate_gemm_nax_alignments(tname, ta, tb, 128, 64, 512, 4, 2)  \
  instantiate_gemm_nax_alignments(tname, ta, tb, 64, 128, 512, 2, 4)

instantiate_gemm_nax_tiles(nn, false, false)
instantiate_gemm_nax_tiles(nt, false, true)
