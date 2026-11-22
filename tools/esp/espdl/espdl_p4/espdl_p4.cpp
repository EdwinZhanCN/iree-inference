// esp-dl backend of the kernel imports: wraps the IREE buffers in esp-dl
// TensorBase views (no copies) and calls dl::base, single core.
//
// The arguments follow espdl_ref.h. Weights arrive already in esp-dl's
// ESP32-P4 filter layout and biases in the accumulator's scale, as ESP-PPQ
// exports them, so nothing is repacked here.

#include "espdl_p4.h"

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

using namespace dl;  // quant_type_t, runtime_mode_t, activation_type_t, TensorBase

namespace {

constexpr quant_type_t kQuant = QUANT_TYPE_SYMM_8BIT;
constexpr runtime_mode_t kMode = RUNTIME_MODE_SINGLE_CORE;

// A TensorBase over memory IREE owns (deep = false: no copy, no free).
TensorBase view(std::vector<int> shape, const void* data, int exponent, dtype_t dtype) {
  return TensorBase(std::move(shape), data, exponent, dtype, /*deep=*/false);
}

}  // namespace

extern "C" int espdl_p4_conv2d_s8(const int8_t* x, const int8_t* w, const int32_t* bias,
                                  const int8_t* w_exp, const int32_t* p, int8_t* y) {
  const int H = p[ESPDL_CONV_H], W = p[ESPDL_CONV_W], C = p[ESPDL_CONV_C];
  const int O = p[ESPDL_CONV_O], KH = p[ESPDL_CONV_KH], KW = p[ESPDL_CONV_KW];
  const int G = p[ESPDL_CONV_G], OH = p[ESPDL_CONV_OH], OW = p[ESPDL_CONV_OW];
  const bool depthwise = G != 1;
  if (depthwise && !(G == C && G == O)) {
    return ESPDL_ERR_UNSUPPORTED;  // esp-dl: grouped conv only as depthwise
  }

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
  dl::base::ConvOpArgs args(&output, &input, pads, &filter, {p[ESPDL_CONV_SH], p[ESPDL_CONV_SW]},
                            {p[ESPDL_CONV_DH], p[ESPDL_CONV_DW]}, G, &bias_t,
                            p[ESPDL_CONV_RELU] ? ReLU : Linear, nullptr, kMode, kQuant);
  if (args.size() != 1) {
    return ESPDL_ERR_UNSUPPORTED;
  }
  dl_kernel_erased_t fn[3] = {};
  if (!dl::base::dl_conv_ensure_kernels(fn, args.get_args(0), depthwise, kQuant, "espdl_iree")) {
    return ESPDL_ERR_UNSUPPORTED;
  }
  if (depthwise) {
    dl::base::depthwise_conv2d(&args.get_args(0), kQuant, fn[0], fn[1], fn[2]);
  } else {
    dl::base::conv2d(&args.get_args(0), kQuant, fn[0], fn[1], fn[2]);
  }
  return ESPDL_OK;
}

extern "C" int espdl_p4_lut_s8(const int8_t* x, const int8_t* table, const int32_t* p, int8_t* y) {
  if (p[0] < 0) return ESPDL_ERR_SHAPE;
  dl::base::lut_s8(y, x, p[0], table);
  return ESPDL_OK;
}

extern "C" int espdl_p4_binary_s8(const int8_t* a, const int8_t* b, const int32_t* p, int8_t* y) {
  const int n = p[ESPDL_BINARY_N], bl = p[ESPDL_BINARY_B_LEN], op = p[ESPDL_BINARY_OP];
  const int ea = p[ESPDL_BINARY_A_EXP], eb = p[ESPDL_BINARY_B_EXP], ey = p[ESPDL_BINARY_Y_EXP];
  if (n <= 0 || bl <= 0 || n % bl) return ESPDL_ERR_SHAPE;
  // esp-dl's int8 Add adds the integers: it needs one exponent for all three.
  if (op == ESPDL_BINARY_ADD && !(ea == eb && eb == ey)) return ESPDL_ERR_UNSUPPORTED;
  if (op != ESPDL_BINARY_ADD && op != ESPDL_BINARY_MUL) return ESPDL_ERR_SHAPE;

  // b repeats every bl elements: a is [n / bl, bl], b is [1, bl].
  TensorBase ta = view({n / bl, bl}, a, ea, dl::DATA_TYPE_INT8);
  TensorBase tb = view({1, bl}, b, eb, dl::DATA_TYPE_INT8);
  TensorBase ty = view({n / bl, bl}, y, ey, dl::DATA_TYPE_INT8);
  std::vector<dl::base::elemwiseArgsType<int8_t>> args =
      dl::base::get_elemwise_operation_args<int8_t>(&ty, &ta, &tb, kMode);
  if (args.size() != 1) return ESPDL_ERR_UNSUPPORTED;
  if (op == ESPDL_BINARY_ADD) {
    dl::base::elemwise_add(&args[0]);
  } else {
    dl::base::elemwise_mul(&args[0]);
  }
  return ESPDL_OK;
}
