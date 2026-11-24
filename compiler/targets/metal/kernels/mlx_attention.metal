// MLX steel attention behind IREE's Metal binding ABI, for hal.dispatch.extern.
//
// The kernel body is MLX's (mlx/backend/metal/kernels/steel/attn/kernels/
// steel_attention.h, or steel_attention_nax.h, its Metal 4 tensor-op version
// for GPUs with neural accelerators, when compiled with -DMLX_NAX; MIT,
// Copyright 2024-26 Apple Inc.), extracted by gen_mlx_kernels.py into
// steel_attention[_nax]_body.inc. MLX's function
// constants are template parameters (alignment) or fixed locals (no mask,
// causal, sinks, logsumexp or window) under the same names.
//
// IREE ABI: bindings in an argument buffer at [[buffer(0)]], [[id(N)]] being
// the binding ordinal; 32-bit constants at [[buffer(3)]].
//
//   O = softmax(scale * Q K^T) V per batch and head, Q/K/V/O strided
//   bindings: 0 = Q, 1 = K, 2 = V, 3 = O
//   constants: B, H, D, qL, kL, scale (f32 bits), then (batch, head, row)
//              strides in elements of Q, K, V and O
//   threadgroup: (32, WM, WN); grid: (ceil(qL / BQ), H, B)

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#ifdef MLX_NAX
#include "steel_attention_nax_ops.h"
#define MLX_ATTENTION_BODY "steel_attention_nax_body.inc"
#define MLX_ATTENTION_PREFIX "mlx_attention_nax_f16"
#else
#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "steel_attention_ops.h"
#define MLX_ATTENTION_BODY "steel_attention_body.inc"
#define MLX_ATTENTION_PREFIX "mlx_attention_f16"
#endif

using namespace metal;
using namespace mlx::steel;

template <typename T>
struct AttnBindings {
  const device T* q [[id(0)]];
  const device T* k [[id(1)]];
  const device T* v [[id(2)]];
  device T* o [[id(3)]];
};

struct AttnConstants {
  int B, H, D, qL, kL;
  float scale;
  int q_strides[3], k_strides[3], v_strides[3], o_strides[3];
};

template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BV,
    int WM,
    int WN,
    bool align_Q,
    bool align_K>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention(
    constant AttnBindings<T>& bindings [[buffer(0)]],
    constant AttnConstants& c [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using MaskType = float;
  using AccumType = float;
  constexpr bool has_mask = false;
  constexpr bool do_causal = false;
  constexpr bool has_sinks = false;
  constexpr bool save_lse = false;
  constexpr int window_size = 0;
  constexpr bool has_window = false;

  const int NQ_aligned = c.qL / BQ;
  const int NK_aligned = c.kL / BK;
  const AttnParams p = {
      /* B = */ c.B,
      /* H = */ c.H,
      /* D = */ c.D,
      /* qL = */ c.qL,
      /* kL = */ c.kL,
      /* gqa_factor = */ 1,
      /* scale = */ c.scale,
      /* NQ = */ (c.qL + BQ - 1) / BQ,
      /* NK = */ (c.kL + BK - 1) / BK,
      /* NQ_aligned = */ NQ_aligned,
      /* NK_aligned = */ NK_aligned,
      /* qL_rem = */ c.qL - NQ_aligned * BQ,
      /* kL_rem = */ c.kL - NK_aligned * BK,
      /* qL_off = */ c.kL - c.qL,
      /* Q_strides = */ {c.q_strides[0], c.q_strides[1], c.q_strides[2]},
      /* K_strides = */ {c.k_strides[0], c.k_strides[1], c.k_strides[2]},
      /* V_strides = */ {c.v_strides[0], c.v_strides[1], c.v_strides[2]},
      /* O_strides = */ {c.o_strides[0], c.o_strides[1], c.o_strides[2]},
  };
  thread const AttnParams* params = &p;

  const device T* Q = bindings.q;
  const device T* K = bindings.k;
  const device T* V = bindings.v;
  device T* O = bindings.o;
  // Unused: has_mask, has_sinks and save_lse are false.
  const constant AttnMaskParams* mask_params = nullptr;
  const device MaskType* mask = nullptr;
  const device T* sinks = nullptr;
  device float* lse = nullptr;

#include MLX_ATTENTION_BODY
}

// Names: mlx_attention[_nax]_f16_bq<>_bk<>_bd<>_bv<>_wm<>_wn<>_<Q><K>, where Q and K
// are a (sequence length a multiple of the block) or u.
#define instantiate_attention(bq, bk, bd, bv, wm, wn, aq, ak, aname)          \
  template [[host_name(MLX_ATTENTION_PREFIX "_bq" #bq "_bk" #bk "_bd" #bd "_bv" #bv \
                       "_wm" #wm "_wn" #wn "_" aname)]] [[kernel]] void         \
  attention<half, bq, bk, bd, bv, wm, wn, aq, ak>(                              \
      constant AttnBindings<half>&, constant AttnConstants&, uint, uint, uint3, uint3);

#define instantiate_attention_alignments(bq, bk, bd, bv, wm, wn)            \
  instantiate_attention(bq, bk, bd, bv, wm, wn, true, true, "aa")           \
  instantiate_attention(bq, bk, bd, bv, wm, wn, false, true, "ua")          \
  instantiate_attention(bq, bk, bd, bv, wm, wn, true, false, "au")          \
  instantiate_attention(bq, bk, bd, bv, wm, wn, false, false, "uu")

// MLX's choice for head dim 64 (mlx_kernels.attention_tile).
#ifdef MLX_NAX
instantiate_attention_alignments(64, 32, 64, 64, 4, 1)
#else
instantiate_attention_alignments(32, 32, 64, 64, 4, 1)
#endif
