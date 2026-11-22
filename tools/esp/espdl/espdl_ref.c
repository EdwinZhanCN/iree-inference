#include "espdl_ref.h"

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
            const int8_t* ws = w + (((int64_t)o * KH + kh) * KW + kw) * cpg;
            for (int32_t c = 0; c < cpg; ++c) acc += (int32_t)xs[c] * ws[c];
          }
        }
        if (relu && acc < 0) acc = 0;
        yp[o] = saturate_s8(requantize(acc, y_exp - x_exp - w_exp[o]));
      }
    }
  }
  return ESPDL_OK;
}
