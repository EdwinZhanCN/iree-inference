// Reference implementations of the esp-dl kernels that IREE dispatches call.
// Each takes its tensors, then an int32 parameter array.
//
// Quantization follows esp-dl on ESP32-P4: int8 values, symmetric (zero point
// 0), every scale a power of two (real = q * 2^exp), per-channel weight
// exponents, int32 bias already in the accumulator's scale, round half to
// even. Tensors are channels-last (NHWC), as esp-dl's.
//
// These functions are the contract: the ESP32-P4 adapter must produce the same
// bytes. They return 0 on success and a nonzero code for arguments they do not
// handle, so a mismatch fails the dispatch instead of computing something else.

#ifndef ESPDL_REF_H_
#define ESPDL_REF_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  ESPDL_OK = 0,
  ESPDL_ERR_SHAPE = 1,
  ESPDL_ERR_UNSUPPORTED = 2,  // valid arguments the backend has no kernel for
};

// Parameters of espdl_ref_conv2d_s8, an int32 array in this order. Written by
// espdl_rewrite.py; append fields, never reorder.
enum {
  ESPDL_CONV_H = 0,   // input height (1 for 1D)
  ESPDL_CONV_W,       // input width
  ESPDL_CONV_C,       // input channels
  ESPDL_CONV_O,       // output channels
  ESPDL_CONV_KH,
  ESPDL_CONV_KW,
  ESPDL_CONV_SH,      // strides
  ESPDL_CONV_SW,
  ESPDL_CONV_PT,      // zero padding: top, left, bottom, right
  ESPDL_CONV_PL,
  ESPDL_CONV_PB,
  ESPDL_CONV_PR,
  ESPDL_CONV_DH,      // dilations
  ESPDL_CONV_DW,
  ESPDL_CONV_G,       // groups (C for depthwise)
  ESPDL_CONV_OH,      // output height and width
  ESPDL_CONV_OW,
  ESPDL_CONV_RELU,    // 1: clamp at 0 before quantizing
  ESPDL_CONV_X_EXP,   // input scale 2^x_exp
  ESPDL_CONV_Y_EXP,   // output scale 2^y_exp
  ESPDL_CONV_PARAM_COUNT,
};

// Quantized 2D convolution, N = 1, NHWC. Covers 1x1 and kxk convolutions,
// depthwise (G = C = O), 1D (H = 1) and matrix products (1x1, H = 1).
//
//   x:     int8  [H][W][C]             scale 2^x_exp
//   w:     int8  esp-dl's ESP32-P4 filter layout, scale 2^w_exp[o]: output
//          channels in blocks of 16, each [KH][KW][C/G][16]; the last O % 16
//          channels after them as [O % 16][KH][KW][C/G], for depthwise
//          (G = C = O) as [KH][KW][O % 16]
//   bias:  int32 [O]                   scale 2^(x_exp + w_exp[o])
//   y:     int8  [OH][OW][O]           scale 2^y_exp
int espdl_ref_conv2d_s8(const int8_t* x, const int8_t* w, const int32_t* bias,
                        const int8_t* w_exp, const int32_t* params, int8_t* y);

// Table lookup, as esp-dl's LUT: y[i] = table[x[i] + 128]. The table holds
// a whole int8 -> int8 elementwise chain (activation, requantization).
//
//   params: { n }
int espdl_ref_lut_s8(const int8_t* x, const int8_t* table, const int32_t* params,
                     int8_t* y);

// Parameters of espdl_ref_binary_s8.
enum {
  ESPDL_BINARY_N = 0,   // elements of a and y
  ESPDL_BINARY_B_LEN,   // elements of b: b repeats every B_LEN elements of a
  ESPDL_BINARY_OP,      // ESPDL_BINARY_ADD or ESPDL_BINARY_MUL
  ESPDL_BINARY_A_EXP,   // scales 2^a_exp, 2^b_exp, 2^y_exp
  ESPDL_BINARY_B_EXP,
  ESPDL_BINARY_Y_EXP,
  ESPDL_BINARY_PARAM_COUNT,
};
enum { ESPDL_BINARY_ADD = 0, ESPDL_BINARY_MUL = 1 };

// y[i] = a[i] (+|*) b[i % B_LEN], computed exactly and rounded half to even
// into y's scale. With power-of-two scales this is what float QDQ computes.
int espdl_ref_binary_s8(const int8_t* a, const int8_t* b, const int32_t* params,
                        int8_t* y);

// Parameters of espdl_ref_quant_s8: quantize [outer, c, hw] f32 -> [outer, hw, c] i8.
// When c == 1 or hw == 1, this is a flat elementwise quantization.
enum {
  ESPDL_QUANT_OUTER = 0,
  ESPDL_QUANT_C,
  ESPDL_QUANT_HW,
  ESPDL_QUANT_X_EXP,  // output scale 2^x_exp (quantizes by * 2^-x_exp)
  ESPDL_QUANT_PARAM_COUNT,
};

int espdl_ref_quant_s8(const float* x, const int32_t* params, int8_t* y);

// Parameters of espdl_ref_dequant_s8_f32: y[i] = (float)x[i] * 2^x_exp.
enum {
  ESPDL_DEQUANT_N = 0,
  ESPDL_DEQUANT_X_EXP,
  ESPDL_DEQUANT_PARAM_COUNT,
};

int espdl_ref_dequant_s8_f32(const int8_t* x, const int32_t* params, float* y);

// Parameters of espdl_ref_avg_pool2d_s8, N = 1, NHWC, no padding.
enum {
  ESPDL_POOL_H = 0,
  ESPDL_POOL_W,
  ESPDL_POOL_C,
  ESPDL_POOL_KH,
  ESPDL_POOL_KW,
  ESPDL_POOL_SH,
  ESPDL_POOL_SW,
  ESPDL_POOL_OH,
  ESPDL_POOL_OW,
  ESPDL_POOL_X_EXP,
  ESPDL_POOL_Y_EXP,
  ESPDL_POOL_PARAM_COUNT,
};

// Quantized 2D average pooling, N = 1, NHWC, unpadded:
//   x: int8 [H][W][C]   scale 2^x_exp
//   y: int8 [OH][OW][C] scale 2^y_exp
int espdl_ref_avg_pool2d_s8(const int8_t* x, const int32_t* params, int8_t* y);

#ifdef __cplusplus
}
#endif

#endif  // ESPDL_REF_H_
