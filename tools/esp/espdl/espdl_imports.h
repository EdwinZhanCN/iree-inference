// Argument structs of the esp-dl kernel imports, as IREE passes them.
//
// espdl_rewrite.py declares each import with llvm.bareptr memref arguments;
// each memref<T> arrives as a base pointer followed by an index offset in
// elements. An import is called as fn(args, context, reserved) and returns 0
// on success. host_plugin.c (reference kernels) and espdl_p4/espdl_p4_imports.c
// (esp-dl) unpack the same structs.

#ifndef ESPDL_IMPORTS_H_
#define ESPDL_IMPORTS_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Signature of an import (iree_hal_executable_import_v0_t).
typedef int (*espdl_import_fn_t)(void* args, void* context, void* reserved);

// espdl_conv2d_s8: x, w, bias, w_exp, params, y.
typedef struct {
  const int8_t* x;
  size_t x_offset;
  const int8_t* w;
  size_t w_offset;
  const int32_t* bias;
  size_t bias_offset;
  const int8_t* w_exp;
  size_t w_exp_offset;
  const int32_t* params;
  size_t params_offset;
  int8_t* y;
  size_t y_offset;
} espdl_conv2d_s8_args_t;

// espdl_lut_s8: x, table, params, y.
typedef struct {
  const int8_t* x;
  size_t x_offset;
  const int8_t* table;
  size_t table_offset;
  const int32_t* params;
  size_t params_offset;
  int8_t* y;
  size_t y_offset;
} espdl_lut_s8_args_t;

// espdl_binary_s8: a, b, params, y.
typedef struct {
  const int8_t* a;
  size_t a_offset;
  const int8_t* b;
  size_t b_offset;
  const int32_t* params;
  size_t params_offset;
  int8_t* y;
  size_t y_offset;
} espdl_binary_s8_args_t;

#ifdef __cplusplus
}
#endif

#endif  // ESPDL_IMPORTS_H_
