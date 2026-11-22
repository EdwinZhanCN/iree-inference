// esp-dl backend of the kernel imports: wraps the IREE buffers in esp-dl
// TensorBase views (no copies) and calls dl::base, single core.
//
// The arguments follow espdl_ref.h. Weights arrive already in esp-dl's
// ESP32-P4 filter layout and biases in the accumulator's scale, as ESP-PPQ
// exports them, so nothing is repacked here.

#include "espdl_p4.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "dl_base_add.hpp"
#include "dl_base_conv2d.hpp"
#include "dl_base_conv_args.hpp"
#include "dl_base_conv_select.hpp"
#include "dl_base_depthwise_conv2d.hpp"
#include "dl_base_elemwise.hpp"
#include "dl_base_lut.hpp"
#include "dl_base_mul.hpp"
#include "dl_tensor_base.hpp"
#include "dl_tool.hpp"
#include "esp_heap_caps.h"
#include "esp_timer.h"

using namespace dl;  // quant_type_t, runtime_mode_t, activation_type_t, TensorBase

namespace {

constexpr quant_type_t kQuant = QUANT_TYPE_SYMM_8BIT;
constexpr runtime_mode_t kMode = RUNTIME_MODE_SINGLE_CORE;
espdl_p4_stats_t g_stats = {};

// A TensorBase over memory IREE owns (deep = false: no copy, no free).
TensorBase view(std::vector<int> shape, const void* data, int exponent, dtype_t dtype) {
  return TensorBase(std::move(shape), data, exponent, dtype, /*deep=*/false);
}

struct ConvCacheEntry {
  const int8_t* w;
  const int32_t* bias;
  const int8_t* w_exp;
  const int32_t* p;
  int32_t params[ESPDL_CONV_PARAM_COUNT];
  dl::base::ConvArgsType base_args;
  dl_kernel_erased_t fn[3];
  bool valid;
};

constexpr size_t kConvCacheCap = 128;
ConvCacheEntry g_conv_cache[kConvCacheCap] = {};

size_t hash_conv_key(const int8_t* w, const int32_t* p) {
  uintptr_t h = (uintptr_t)w * 2654435761u ^ (uintptr_t)p * 2246822519u;
  h ^= h >> 16;
  return (size_t)h & (kConvCacheCap - 1);
}

ConvCacheEntry* find_conv_cache(const int8_t* w, const int32_t* bias, const int8_t* w_exp,
                                const int32_t* p, bool* hit) {
  const size_t start = hash_conv_key(w, p);
  for (size_t probe = 0; probe < kConvCacheCap; ++probe) {
    ConvCacheEntry* e = &g_conv_cache[(start + probe) & (kConvCacheCap - 1)];
    if (!e->valid) {
      *hit = false;
      return e;
    }
    if (e->w == w && e->bias == bias && e->w_exp == w_exp && e->p == p &&
        std::memcmp(e->params, p, sizeof(e->params)) == 0) {
      *hit = true;
      return e;
    }
  }
  // Full table fallback: evict start slot.
  ConvCacheEntry* e = &g_conv_cache[start];
  if (e->valid && e->base_args.mac_shift == INT_MIN) {
    heap_caps_free(e->base_args.tie_filter_channel_factor);
  }
  e->valid = false;
  *hit = false;
  return e;
}

}  // namespace

extern "C" void espdl_p4_reset_stats(void) { g_stats = {}; }
extern "C" void espdl_p4_get_stats(espdl_p4_stats_t* out) { *out = g_stats; }

extern "C" int espdl_p4_conv2d_s8(const int8_t* x, const int8_t* w, const int32_t* bias,
                                  const int8_t* w_exp, const int32_t* p, int8_t* y) {
  const int64_t t0 = esp_timer_get_time();
  const int H = p[ESPDL_CONV_H], W = p[ESPDL_CONV_W], C = p[ESPDL_CONV_C];
  const int O = p[ESPDL_CONV_O], KH = p[ESPDL_CONV_KH], KW = p[ESPDL_CONV_KW];
  const int G = p[ESPDL_CONV_G], OH = p[ESPDL_CONV_OH], OW = p[ESPDL_CONV_OW];
  const bool depthwise = G != 1;
  if (depthwise && !(G == C && G == O)) {
    return ESPDL_ERR_UNSUPPORTED;  // esp-dl: grouped conv only as depthwise
  }

  bool hit = false;
  ConvCacheEntry* entry = find_conv_cache(w, bias, w_exp, p, &hit);
  if (!hit) {
    TensorBase input = view({1, H, W, C}, x, p[ESPDL_CONV_X_EXP], dl::DATA_TYPE_INT8);
    TensorBase output = view({1, OH, OW, O}, y, p[ESPDL_CONV_Y_EXP], dl::DATA_TYPE_INT8);
    // Filter shapes as ESP-PPQ exports them: conv [KH, KW, C, O], depthwise
    // [KH, KW, O, 1]. Per-channel exponents select esp-dl's per-channel path.
    std::vector<int> exponents(w_exp, w_exp + O);
    TensorBase filter(depthwise ? std::vector<int>{KH, KW, O, 1} : std::vector<int>{KH, KW, C, O},
                      w, exponents, dl::DATA_TYPE_INT8, /*deep=*/false);
    if (!filter.data) {
      return ESPDL_ERR_UNSUPPORTED;  // exponent allocation failed
    }
    TensorBase bias_t = view({O}, bias, 0, dl::DATA_TYPE_INT32);

    // esp-dl padding order: top, bottom, left, right.
    std::vector<int> pads = {p[ESPDL_CONV_PT], p[ESPDL_CONV_PB], p[ESPDL_CONV_PL], p[ESPDL_CONV_PR]};
    dl::base::ConvOpArgs op_args(&output, &input, pads, &filter, {p[ESPDL_CONV_SH], p[ESPDL_CONV_SW]},
                                 {p[ESPDL_CONV_DH], p[ESPDL_CONV_DW]}, G, &bias_t,
                                 p[ESPDL_CONV_RELU] ? ReLU : Linear, nullptr, kMode, kQuant);
    if (op_args.size() != 1) {
      return ESPDL_ERR_UNSUPPORTED;
    }
    dl_kernel_erased_t fn[3] = {};
    if (!dl::base::dl_conv_ensure_kernels(fn, op_args.get_args(0), depthwise, kQuant, "espdl_iree")) {
      return ESPDL_ERR_UNSUPPORTED;
    }

    dl::base::ConvArgsType base = op_args.get_args(0);
    if (base.mac_shift == INT_MIN && base.tie_filter_channel_factor) {
      const size_t tie_sz = dl::tool::get_aligned_size(O);
      void* tie_copy = dl::tool::calloc_aligned(tie_sz, 1, MALLOC_CAP_DEFAULT);
      if (!tie_copy) {
        return ESPDL_ERR_UNSUPPORTED;
      }
      std::memcpy(tie_copy, base.tie_filter_channel_factor, tie_sz);
      base.tie_filter_channel_factor = tie_copy;
    }
    entry->w = w;
    entry->bias = bias;
    entry->w_exp = w_exp;
    entry->p = p;
    std::memcpy(entry->params, p, sizeof(entry->params));
    entry->base_args = base;
    entry->fn[0] = fn[0];
    entry->fn[1] = fn[1];
    entry->fn[2] = fn[2];
    entry->valid = true;
  }

  // Copy to stack because esp-dl's border/grid loops mutate ConvArgsType in place.
  dl::base::ConvArgsType args = entry->base_args;
  args.input_element = const_cast<int8_t*>(x);
  args.output_element = y;

  const int64_t t1 = esp_timer_get_time();
  if (depthwise) {
    dl::base::depthwise_conv2d(&args, kQuant, entry->fn[0], entry->fn[1], entry->fn[2]);
  } else {
    dl::base::conv2d(&args, kQuant, entry->fn[0], entry->fn[1], entry->fn[2]);
  }
  const int64_t t2 = esp_timer_get_time();
  ++g_stats.conv_calls;
  g_stats.conv_setup_us += (t1 - t0);
  g_stats.conv_kernel_us += (t2 - t1);
  return ESPDL_OK;
}

extern "C" int espdl_p4_lut_s8(const int8_t* x, const int8_t* table, const int32_t* p, int8_t* y) {
  if (p[0] < 0) return ESPDL_ERR_SHAPE;
  const int64_t t0 = esp_timer_get_time();
  dl::base::lut_s8(y, x, p[0], table);
  const int64_t t1 = esp_timer_get_time();
  ++g_stats.lut_calls;
  g_stats.lut_us += (t1 - t0);
  return ESPDL_OK;
}

extern "C" int espdl_p4_binary_s8(const int8_t* a, const int8_t* b, const int32_t* p, int8_t* y) {
  const int64_t t0 = esp_timer_get_time();
  const int n = p[ESPDL_BINARY_N], bl = p[ESPDL_BINARY_B_LEN], op = p[ESPDL_BINARY_OP];
  const int ea = p[ESPDL_BINARY_A_EXP], eb = p[ESPDL_BINARY_B_EXP], ey = p[ESPDL_BINARY_Y_EXP];
  if (n <= 0 || bl <= 0 || n % bl) return ESPDL_ERR_SHAPE;
  // esp-dl's int8 Add adds the integers: it needs one exponent for all three.
  if (op == ESPDL_BINARY_ADD && !(ea == eb && eb == ey)) return ESPDL_ERR_UNSUPPORTED;
  if (op != ESPDL_BINARY_ADD && op != ESPDL_BINARY_MUL) return ESPDL_ERR_SHAPE;

  // Populate elemwiseArgsType<int8_t> directly on stack (equivalent to [n/bl, bl] + [1, bl]).
  dl::base::elemwiseArgsType<int8_t> args = {};
  args.input0_element = const_cast<int8_t*>(a);
  args.input1_element = const_cast<int8_t*>(b);
  args.output_element = y;
  const int rows = n / bl;
  if (rows == 1) {
    args.dims = 1;
    args.output_d0 = n;
    args.input0_d0 = n;
    args.input1_d0 = n;
  } else {
    args.dims = 2;
    args.output_d0 = bl;
    args.output_d1 = rows;
    args.input0_d0 = bl;
    args.input1_d0 = bl;
    args.input0_d1_stride = bl;
    args.input1_d1_stride = 0;
  }
  constexpr int u = 16;
  const int c_div_x = args.output_d0 / u;
  args.c_remainder = args.output_d0 % u;
  args.c_div_x_1 = c_div_x - 1;
  args.c_div_2x_1 = std::max(c_div_x / 2 - 1, 0);
  args.c_left_x_1 = c_div_x - 2 * args.c_div_2x_1 - 1;
  args.mul_shift = ey - ea - eb;
  args.input0_scale = DL_SCALE(ea);
  args.input1_scale = DL_SCALE(eb);
  args.output_rescale = DL_RESCALE(ey);
  args.compare_mask = 1;

  const int64_t t1 = esp_timer_get_time();
  if (op == ESPDL_BINARY_ADD) {
    dl::base::elemwise_add(&args);
  } else {
    dl::base::elemwise_mul(&args);
  }
  const int64_t t2 = esp_timer_get_time();
  ++g_stats.binary_calls;
  g_stats.binary_setup_us += (t1 - t0);
  g_stats.binary_kernel_us += (t2 - t1);
  return ESPDL_OK;
}

namespace {

inline int8_t quant_rne_s8(float v, float inv_scale) {
  float s = v * inv_scale;
  if (s < -128.0f) s = -128.0f;
  if (s > 127.0f) s = 127.0f;
  int32_t q;
  __asm__("fcvt.w.s %0, %1, rne" : "=r"(q) : "f"(s));
  return static_cast<int8_t>(q);
}

}  // namespace

extern "C" int espdl_p4_quant_s8(const float* x, const int32_t* p, int8_t* y) {
  const int64_t t0 = esp_timer_get_time();
  const int32_t outer = p[ESPDL_QUANT_OUTER];
  const int32_t c = p[ESPDL_QUANT_C];
  const int32_t hw = p[ESPDL_QUANT_HW];
  const int x_exp = p[ESPDL_QUANT_X_EXP];
  if (outer <= 0 || c <= 0 || hw <= 0) return ESPDL_ERR_SHAPE;
  const float inv_scale = ldexpf(1.0f, -x_exp);

  for (int32_t o = 0; o < outer; ++o) {
    const float* xo = x + static_cast<size_t>(o) * c * hw;
    int8_t* yo = y + static_cast<size_t>(o) * hw * c;
    if (c == 3) {
      const float* p0 = xo;
      const float* p1 = xo + hw;
      const float* p2 = xo + 2 * hw;
      int32_t i = 0;
      if ((reinterpret_cast<uintptr_t>(yo) & 3u) == 0 && hw >= 4) {
        uint32_t* out_w = reinterpret_cast<uint32_t*>(yo);
        const int32_t hw4 = hw & ~3;
        for (; i < hw4; i += 4) {
          const uint32_t q00 = static_cast<uint8_t>(quant_rne_s8(p0[i + 0], inv_scale));
          const uint32_t q01 = static_cast<uint8_t>(quant_rne_s8(p1[i + 0], inv_scale));
          const uint32_t q02 = static_cast<uint8_t>(quant_rne_s8(p2[i + 0], inv_scale));
          const uint32_t q10 = static_cast<uint8_t>(quant_rne_s8(p0[i + 1], inv_scale));
          const uint32_t q11 = static_cast<uint8_t>(quant_rne_s8(p1[i + 1], inv_scale));
          const uint32_t q12 = static_cast<uint8_t>(quant_rne_s8(p2[i + 1], inv_scale));
          const uint32_t q20 = static_cast<uint8_t>(quant_rne_s8(p0[i + 2], inv_scale));
          const uint32_t q21 = static_cast<uint8_t>(quant_rne_s8(p1[i + 2], inv_scale));
          const uint32_t q22 = static_cast<uint8_t>(quant_rne_s8(p2[i + 2], inv_scale));
          const uint32_t q30 = static_cast<uint8_t>(quant_rne_s8(p0[i + 3], inv_scale));
          const uint32_t q31 = static_cast<uint8_t>(quant_rne_s8(p1[i + 3], inv_scale));
          const uint32_t q32 = static_cast<uint8_t>(quant_rne_s8(p2[i + 3], inv_scale));
          out_w[0] = q00 | (q01 << 8) | (q02 << 16) | (q10 << 24);
          out_w[1] = q11 | (q12 << 8) | (q20 << 16) | (q21 << 24);
          out_w[2] = q22 | (q30 << 8) | (q31 << 16) | (q32 << 24);
          out_w += 3;
        }
      }
      int8_t* out = yo + static_cast<size_t>(i) * 3;
      for (; i < hw; ++i) {
        out[0] = quant_rne_s8(p0[i], inv_scale);
        out[1] = quant_rne_s8(p1[i], inv_scale);
        out[2] = quant_rne_s8(p2[i], inv_scale);
        out += 3;
      }
    } else if (c == 1 || hw == 1) {
      const int32_t n = c * hw;
      int32_t i = 0;
      if ((reinterpret_cast<uintptr_t>(yo) & 3u) == 0 && n >= 4) {
        uint32_t* out_w = reinterpret_cast<uint32_t*>(yo);
        const int32_t n4 = n & ~3;
        for (; i < n4; i += 4) {
          const uint32_t q0 = static_cast<uint8_t>(quant_rne_s8(xo[i + 0], inv_scale));
          const uint32_t q1 = static_cast<uint8_t>(quant_rne_s8(xo[i + 1], inv_scale));
          const uint32_t q2 = static_cast<uint8_t>(quant_rne_s8(xo[i + 2], inv_scale));
          const uint32_t q3 = static_cast<uint8_t>(quant_rne_s8(xo[i + 3], inv_scale));
          *out_w++ = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        }
      }
      for (; i < n; ++i) {
        yo[i] = quant_rne_s8(xo[i], inv_scale);
      }
    } else {
      for (int32_t i = 0; i < hw; ++i) {
        int8_t* out = yo + static_cast<size_t>(i) * c;
        for (int32_t ch = 0; ch < c; ++ch) {
          out[ch] = quant_rne_s8(xo[static_cast<size_t>(ch) * hw + i], inv_scale);
        }
      }
    }
  }
  const int64_t t1 = esp_timer_get_time();
  ++g_stats.quant_calls;
  g_stats.quant_us += (t1 - t0);
  return ESPDL_OK;
}

extern "C" int espdl_p4_dequant_s8_f32(const int8_t* x, const int32_t* p, float* y) {
  const int64_t t0 = esp_timer_get_time();
  const int32_t n = p[ESPDL_DEQUANT_N];
  const int x_exp = p[ESPDL_DEQUANT_X_EXP];
  if (n < 0) return ESPDL_ERR_SHAPE;

  float lut[256];
  const float scale = ldexpf(1.0f, x_exp);
  for (int v = -128; v <= 127; ++v) {
    lut[static_cast<uint8_t>(static_cast<int8_t>(v))] = static_cast<float>(v) * scale;
  }

  int32_t i = 0;
  if ((reinterpret_cast<uintptr_t>(x) & 3u) == 0 && n >= 4) {
    const uint32_t* xp = reinterpret_cast<const uint32_t*>(x);
    const int32_t n4 = n & ~3;
    for (; i < n4; i += 4) {
      const uint32_t pack = *xp++;
      y[i + 0] = lut[pack & 0xFFu];
      y[i + 1] = lut[(pack >> 8) & 0xFFu];
      y[i + 2] = lut[(pack >> 16) & 0xFFu];
      y[i + 3] = lut[pack >> 24];
    }
  }
  for (; i < n; ++i) {
    y[i] = lut[static_cast<uint8_t>(x[i])];
  }
  const int64_t t1 = esp_timer_get_time();
  ++g_stats.dequant_calls;
  g_stats.dequant_us += (t1 - t0);
  return ESPDL_OK;
}
