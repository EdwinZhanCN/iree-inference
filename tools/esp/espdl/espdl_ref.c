#include "espdl_ref.h"

#include <math.h>
#include <stddef.h>

// v * 2^-shift rounded half to even (shift > 0), or v * 2^-shift (shift <= 0).
static int64_t requantize(int64_t v, int shift) {
  if (shift <= 0) return v * ((int64_t)1 << -shift);
  int64_t q = v >> shift;  // floor
  int64_t rem = v - q * ((int64_t)1 << shift);
  int64_t half = (int64_t)1 << (shift - 1);
  if (rem > half || (rem == half && (q & 1))) q += 1;
  return q;
}

static int8_t saturate_s8(int64_t v) {
  return (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
}

// Index of weight (o, kh, kw, c) in esp-dl's packed filter layout.
static int64_t filter_index(int32_t o, int32_t kh, int32_t kw, int32_t c, int32_t O,
                            int32_t KH, int32_t KW, int32_t cpg, int depthwise) {
  const int32_t al = O / 16 * 16;
  const int64_t taps = (int64_t)KH * KW * cpg;
  if (o < al) return ((o / 16) * taps + ((int64_t)kh * KW + kw) * cpg + c) * 16 + o % 16;
  if (depthwise) return al * taps + ((int64_t)kh * KW + kw) * (O - al) + (o - al);
  return al * taps + (o - al) * taps + ((int64_t)kh * KW + kw) * cpg + c;
}

int espdl_ref_conv2d_s8(const int8_t* x, const int8_t* w, const int32_t* bias,
                        const int8_t* w_exp, const int32_t* p, int8_t* y) {
  const int32_t H = p[ESPDL_CONV_H], W = p[ESPDL_CONV_W], C = p[ESPDL_CONV_C];
  const int32_t O = p[ESPDL_CONV_O], KH = p[ESPDL_CONV_KH], KW = p[ESPDL_CONV_KW];
  const int32_t SH = p[ESPDL_CONV_SH], SW = p[ESPDL_CONV_SW];
  const int32_t PT = p[ESPDL_CONV_PT], PL = p[ESPDL_CONV_PL];
  const int32_t PB = p[ESPDL_CONV_PB], PR = p[ESPDL_CONV_PR];
  const int32_t DH = p[ESPDL_CONV_DH], DW = p[ESPDL_CONV_DW], G = p[ESPDL_CONV_G];
  const int32_t OH = p[ESPDL_CONV_OH], OW = p[ESPDL_CONV_OW];
  const int relu = p[ESPDL_CONV_RELU] != 0;
  const int x_exp = p[ESPDL_CONV_X_EXP], y_exp = p[ESPDL_CONV_Y_EXP];

  if (H <= 0 || W <= 0 || C <= 0 || O <= 0 || KH <= 0 || KW <= 0 || SH <= 0 ||
      SW <= 0 || DH <= 0 || DW <= 0 || G <= 0 || C % G || O % G ||
      PT < 0 || PL < 0 || PB < 0 || PR < 0 ||
      OH != (H + PT + PB - DH * (KH - 1) - 1) / SH + 1 ||
      OW != (W + PL + PR - DW * (KW - 1) - 1) / SW + 1) {
    return ESPDL_ERR_SHAPE;
  }
  const int32_t cpg = C / G, opg = O / G;

  for (int32_t oh = 0; oh < OH; ++oh) {
    for (int32_t ow = 0; ow < OW; ++ow) {
      int8_t* yp = y + ((int64_t)oh * OW + ow) * O;
      for (int32_t o = 0; o < O; ++o) {
        const int32_t g = o / opg;
        int64_t acc = bias[o];
        for (int32_t kh = 0; kh < KH; ++kh) {
          const int32_t ih = oh * SH - PT + kh * DH;
          if (ih < 0 || ih >= H) continue;  // zero padding: zero point is 0
          for (int32_t kw = 0; kw < KW; ++kw) {
            const int32_t iw = ow * SW - PL + kw * DW;
            if (iw < 0 || iw >= W) continue;
            const int8_t* xs = x + ((int64_t)ih * W + iw) * C + (int64_t)g * cpg;
            for (int32_t c = 0; c < cpg; ++c) {
              acc += (int32_t)xs[c] * w[filter_index(o, kh, kw, c, O, KH, KW, cpg, G > 1)];
            }
          }
        }
        if (relu && acc < 0) acc = 0;
        yp[o] = saturate_s8(requantize(acc, y_exp - x_exp - w_exp[o]));
      }
    }
  }
  return ESPDL_OK;
}

int espdl_ref_lut_s8(const int8_t* x, const int8_t* table, const int32_t* p,
                     int8_t* y) {
  const int32_t n = p[0];
  if (n < 0) return ESPDL_ERR_SHAPE;
  for (int32_t i = 0; i < n; ++i) y[i] = table[(uint8_t)x[i] ^ 0x80u];
  return ESPDL_OK;
}

int espdl_ref_binary_s8(const int8_t* a, const int8_t* b, const int32_t* p,
                        int8_t* y) {
  const int32_t n = p[ESPDL_BINARY_N], bl = p[ESPDL_BINARY_B_LEN];
  const int32_t op = p[ESPDL_BINARY_OP];
  const int ea = p[ESPDL_BINARY_A_EXP], eb = p[ESPDL_BINARY_B_EXP];
  const int ey = p[ESPDL_BINARY_Y_EXP];
  if (n < 0 || bl <= 0 || n % bl || (op != ESPDL_BINARY_ADD && op != ESPDL_BINARY_MUL)) {
    return ESPDL_ERR_SHAPE;
  }
  if (op == ESPDL_BINARY_ADD) {
    // a * 2^ea + b * 2^eb = (a * 2^(ea-m) + b * 2^(eb-m)) * 2^m, m = min.
    const int m = ea < eb ? ea : eb;
    const int64_t fa = (int64_t)1 << (ea - m), fb = (int64_t)1 << (eb - m);
    for (int32_t i = 0; i < n; ++i) {
      y[i] = saturate_s8(requantize(a[i] * fa + b[i % bl] * fb, ey - m));
    }
  } else {
    for (int32_t i = 0; i < n; ++i) {
      y[i] = saturate_s8(requantize((int64_t)a[i] * b[i % bl], ey - ea - eb));
    }
  }
  return ESPDL_OK;
}

static int8_t quant_rne_s8(float v, float inv_scale) {
  float s = v * inv_scale;
  if (s < -128.0f) s = -128.0f;
  if (s > 127.0f) s = 127.0f;
  return (int8_t)lrintf(s);
}

int espdl_ref_quant_s8(const float* x, const int32_t* p, int8_t* y) {
  const int32_t outer = p[ESPDL_QUANT_OUTER];
  const int32_t c = p[ESPDL_QUANT_C];
  const int32_t hw = p[ESPDL_QUANT_HW];
  const int x_exp = p[ESPDL_QUANT_X_EXP];
  if (outer <= 0 || c <= 0 || hw <= 0) return ESPDL_ERR_SHAPE;
  const float inv_scale = ldexpf(1.0f, -x_exp);
  for (int32_t o = 0; o < outer; ++o) {
    const float* xo = x + (size_t)o * c * hw;
    int8_t* yo = y + (size_t)o * hw * c;
    for (int32_t i = 0; i < hw; ++i) {
      for (int32_t ch = 0; ch < c; ++ch) {
        yo[(size_t)i * c + ch] = quant_rne_s8(xo[(size_t)ch * hw + i], inv_scale);
      }
    }
  }
  return ESPDL_OK;
}

int espdl_ref_dequant_s8_f32(const int8_t* x, const int32_t* p, float* y) {
  const int32_t n = p[ESPDL_DEQUANT_N];
  const int x_exp = p[ESPDL_DEQUANT_X_EXP];
  if (n < 0) return ESPDL_ERR_SHAPE;
  const float scale = ldexpf(1.0f, x_exp);
  for (int32_t i = 0; i < n; ++i) {
    y[i] = (float)x[i] * scale;
  }
  return ESPDL_OK;
}

int espdl_ref_avg_pool2d_s8(const int8_t* x, const int32_t* p, int8_t* y) {
  const int32_t H = p[ESPDL_POOL_H], W = p[ESPDL_POOL_W], C = p[ESPDL_POOL_C];
  const int32_t KH = p[ESPDL_POOL_KH], KW = p[ESPDL_POOL_KW];
  const int32_t SH = p[ESPDL_POOL_SH], SW = p[ESPDL_POOL_SW];
  const int32_t OH = p[ESPDL_POOL_OH], OW = p[ESPDL_POOL_OW];
  const int x_exp = p[ESPDL_POOL_X_EXP], y_exp = p[ESPDL_POOL_Y_EXP];
  if (H <= 0 || W <= 0 || C <= 0 || KH <= 0 || KW <= 0 || SH <= 0 || SW <= 0 ||
      KH > H || KW > W || OH != (H - KH) / SH + 1 || OW != (W - KW) / SW + 1) {
    return ESPDL_ERR_SHAPE;
  }
  const float area = (float)((int64_t)KH * KW);
  const float scale = ldexpf(1.0f, x_exp - y_exp);
  for (int32_t oh = 0; oh < OH; ++oh) {
    for (int32_t ow = 0; ow < OW; ++ow) {
      int8_t* yp = y + ((int64_t)oh * OW + ow) * C;
      for (int32_t c = 0; c < C; ++c) {
        int32_t sum = 0;
        for (int32_t kh = 0; kh < KH; ++kh) {
          const int32_t ih = oh * SH + kh;
          for (int32_t kw = 0; kw < KW; ++kw) {
            const int32_t iw = ow * SW + kw;
            sum += x[((int64_t)ih * W + iw) * C + c];
          }
        }
        yp[c] = quant_rne_s8((float)sum / area, scale);
      }
    }
  }
  return ESPDL_OK;
}
