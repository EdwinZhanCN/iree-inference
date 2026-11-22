// Test vectors written by gen_vectors.py (vectors.c).
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { ESPDL_CASE_CONV, ESPDL_CASE_BINARY, ESPDL_CASE_LUT, ESPDL_CASE_POOL };

typedef struct {
  const char* name;
  int kind;
  // conv: x, w, bias, w_exp, params. binary: a, b, params. lut: x, table, params.
  const void* in[5];
  const int8_t* expected;  // reference kernel output
  int n_out;
} espdl_case_t;

extern const espdl_case_t espdl_cases[];
extern const int espdl_case_count;

#ifdef __cplusplus
}
#endif
