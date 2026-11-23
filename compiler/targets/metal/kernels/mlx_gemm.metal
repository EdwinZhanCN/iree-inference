// MLX steel GEMM behind IREE's Metal binding ABI, for hal.dispatch.extern.
//
// The kernel body follows mlx/backend/metal/kernels/steel/gemm/kernels/
// steel_gemm_fused.h (MIT, Copyright 2024 Apple Inc.) without the addmm
// epilogue and broadcast batching. MLX selects alignment with function
// constants; IREE cannot specialize those, so they are template parameters
// and each combination is its own kernel.
//
// IREE ABI: bindings in an argument buffer at [[buffer(0)]], [[id(N)]] being
// the binding ordinal; 32-bit constants at [[buffer(3)]].
//
//   D[z] = op(A[z]) * op(B[z]),  op = transpose when the kernel name says t
//   bindings: 0 = A, 1 = B, 2 = D
//   constants: M, N, K, lda, ldb, ldd, batch_stride_a, batch_stride_b,
//              batch_stride_d (elements; batch stride 0 broadcasts)
//   threadgroup: (32, WN, WM); grid: (tiles_n << swizzle, ceil(tiles_m >>
//   swizzle), batch)

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

using namespace metal;
using namespace mlx::steel;

template <typename T>
struct GemmBindings {
  const device T* a [[id(0)]];
  const device T* b [[id(1)]];
  device T* d [[id(2)]];
};

struct GemmConstants {
  int M, N, K;
  int lda, ldb, ldd;
  int batch_stride_a, batch_stride_b, batch_stride_d;
};

constant constexpr int kSwizzleLog = 2;

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
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void gemm(
    constant GemmBindings<T>& bindings [[buffer(0)]],
    constant GemmConstants& c [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) {
  using gemm_kernel = GEMMKernel<
      T, T, BM, BN, BK, WM, WN, transpose_a, transpose_b, true, true, float>;
  using loader_a_t = typename gemm_kernel::loader_a_t;
  using loader_b_t = typename gemm_kernel::loader_b_t;
  using mma_t = typename gemm_kernel::mma_t;

  const int tiles_n = (c.N + BN - 1) / BN;
  const int tiles_m = (c.M + BM - 1) / BM;
  const int tid_y = (int(tid.y) << kSwizzleLog) +
      (int(tid.x) & ((1 << kSwizzleLog) - 1));
  const int tid_x = int(tid.x) >> kSwizzleLog;
  if (tiles_n <= tid_x || tiles_m <= tid_y) {
    return;
  }

  const device T* A = bindings.a + size_t(c.batch_stride_a) * tid.z;
  const device T* B = bindings.b + size_t(c.batch_stride_b) * tid.z;
  device T* D = bindings.d + size_t(c.batch_stride_d) * tid.z;

  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  threadgroup_barrier(mem_flags::mem_none);

  const size_t c_row = size_t(tid_y * BM);
  const size_t c_col = size_t(tid_x * BN);
  A += transpose_a ? c_row : c_row * c.lda;
  B += transpose_b ? c_col * c.ldb : c_col;
  D += c_row * c.ldd + c_col;

  thread mma_t mma_op(simd_group_id, simd_lane_id);
  thread loader_a_t loader_a(A, c.lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, c.ldb, Bs, simd_group_id, simd_lane_id);

  const short tgp_bm = align_M ? BM : short(min(BM, c.M - int(c_row)));
  const short tgp_bn = align_N ? BN : short(min(BN, c.N - int(c_col)));
  const int gemm_k_iterations = c.K / BK;

  // Unaligned K remainder first, as in MLX.
  if (!align_K) {
    const int k_last = gemm_k_iterations * BK;
    const int k_remain = c.K - k_last;
    const size_t k_jump_a = transpose_a ? c.lda * size_t(k_last) : size_t(k_last);
    const size_t k_jump_b = transpose_b ? size_t(k_last) : c.ldb * size_t(k_last);
    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;
    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);
    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  if (align_M && align_N) {
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);
    return mma_op.store_result(D, c.ldd);
  }

  const int leftover_bk = 0;
  if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
    gemm_kernel::gemm_loop(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn,
        leftover_bk, LoopAlignment<true, true, true>{});
    return mma_op.store_result(D, c.ldd);
  } else if (align_N || tgp_bn == BN) {
    gemm_kernel::gemm_loop(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn,
        leftover_bk, LoopAlignment<false, true, true>{});
    return mma_op.store_result_safe(D, c.ldd, short2(tgp_bn, tgp_bm));
  } else if (align_M || tgp_bm == BM) {
    gemm_kernel::gemm_loop(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn,
        leftover_bk, LoopAlignment<true, false, true>{});
    return mma_op.store_result_safe(D, c.ldd, short2(tgp_bn, tgp_bm));
  } else {
    gemm_kernel::gemm_loop(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn,
        leftover_bk, LoopAlignment<false, false, true>{});
    return mma_op.store_result_safe(D, c.ldd, short2(tgp_bn, tgp_bm));
  }
}

// Names: mlx_gemm_<nn|nt>_f16_bm<>_bn<>_bk<>_wm<>_wn<>_<M><N><K>, where each
// of M N K is a (aligned to the tile) or u (unaligned).
#define instantiate_gemm(tname, ta, tb, bm, bn, bk, wm, wn, am, an, ak, aname) \
  template [[host_name("mlx_gemm_" #tname "_f16_bm" #bm "_bn" #bn "_bk" #bk   \
                       "_wm" #wm "_wn" #wn "_" aname)]] [[kernel]] void        \
  gemm<half, bm, bn, bk, wm, wn, ta, tb, am, an, ak>(                          \
      constant GemmBindings<half>&, constant GemmConstants&, uint, uint, uint3);

#define instantiate_gemm_alignments(tname, ta, tb, bm, bn, bk, wm, wn)           \
  instantiate_gemm(tname, ta, tb, bm, bn, bk, wm, wn, true, true, true, "aaa")   \
  instantiate_gemm(tname, ta, tb, bm, bn, bk, wm, wn, false, true, true, "uaa")  \
  instantiate_gemm(tname, ta, tb, bm, bn, bk, wm, wn, true, false, true, "aua")  \
  instantiate_gemm(tname, ta, tb, bm, bn, bk, wm, wn, false, false, true, "uua") \
  instantiate_gemm(tname, ta, tb, bm, bn, bk, wm, wn, false, false, false, "uuu")

// nn: MLX's choice for medium devices (M1/M2/M3 Pro and Max: applegpu_g1Xs).
instantiate_gemm_alignments(nn, false, false, 64, 64, 16, 2, 2)
// nt: fastest on an M2 Pro for the BioCLIP 2 linears (M = 257) among MLX's
// presets and a few BM = 32 tiles; see mlx_kernels.TILES.
instantiate_gemm_alignments(nt, false, true, 32, 32, 32, 2, 2)
