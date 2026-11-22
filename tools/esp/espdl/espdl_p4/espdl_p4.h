// esp-dl backend of the kernel imports. Same arguments and results as the
// reference implementations in espdl_ref.h, which define the contract.

#ifndef ESPDL_P4_H_
#define ESPDL_P4_H_

#include <stdint.h>

#include "espdl_imports.h"
#include "espdl_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

int espdl_p4_conv2d_s8(const int8_t* x, const int8_t* w, const int32_t* bias,
                       const int8_t* w_exp, const int32_t* params, int8_t* y);
int espdl_p4_lut_s8(const int8_t* x, const int8_t* table, const int32_t* params,
                    int8_t* y);
int espdl_p4_binary_s8(const int8_t* a, const int8_t* b, const int32_t* params,
                       int8_t* y);

// The import named name (e.g. "espdl_conv2d_s8"), or NULL.
espdl_import_fn_t espdl_p4_lookup(const char* name);

#ifdef __cplusplus
}
#endif

#endif  // ESPDL_P4_H_
