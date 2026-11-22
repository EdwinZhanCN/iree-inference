// Import entry points for IREE: unpack the argument structs and call the
// esp-dl backend.

#include <string.h>

#include "espdl_p4.h"

static int conv2d_s8(void* args, void* context, void* reserved) {
  (void)context;
  (void)reserved;
  const espdl_conv2d_s8_args_t* a = (const espdl_conv2d_s8_args_t*)args;
  return espdl_p4_conv2d_s8(a->x + a->x_offset, a->w + a->w_offset,
                            a->bias + a->bias_offset, a->w_exp + a->w_exp_offset,
                            a->params + a->params_offset, a->y + a->y_offset);
}

static int lut_s8(void* args, void* context, void* reserved) {
  (void)context;
  (void)reserved;
  const espdl_lut_s8_args_t* a = (const espdl_lut_s8_args_t*)args;
  return espdl_p4_lut_s8(a->x + a->x_offset, a->table + a->table_offset,
                         a->params + a->params_offset, a->y + a->y_offset);
}

static int binary_s8(void* args, void* context, void* reserved) {
  (void)context;
  (void)reserved;
  const espdl_binary_s8_args_t* a = (const espdl_binary_s8_args_t*)args;
  return espdl_p4_binary_s8(a->a + a->a_offset, a->b + a->b_offset,
                            a->params + a->params_offset, a->y + a->y_offset);
}

static int quant_s8(void* args, void* context, void* reserved) {
  (void)context;
  (void)reserved;
  const espdl_quant_s8_args_t* a = (const espdl_quant_s8_args_t*)args;
  return espdl_p4_quant_s8(a->x + a->x_offset, a->params + a->params_offset,
                           a->y + a->y_offset);
}

static int dequant_s8_f32(void* args, void* context, void* reserved) {
  (void)context;
  (void)reserved;
  const espdl_dequant_s8_f32_args_t* a = (const espdl_dequant_s8_f32_args_t*)args;
  return espdl_p4_dequant_s8_f32(a->x + a->x_offset, a->params + a->params_offset,
                                 a->y + a->y_offset);
}

espdl_import_fn_t espdl_p4_lookup(const char* name) {
  static const struct {
    const char* name;
    espdl_import_fn_t fn;
  } imports[] = {
      {"espdl_conv2d_s8", conv2d_s8},
      {"espdl_lut_s8", lut_s8},
      {"espdl_binary_s8", binary_s8},
      {"espdl_quant_s8", quant_s8},
      {"espdl_dequant_s8_f32", dequant_s8_f32},
  };
  for (size_t i = 0; i < sizeof(imports) / sizeof(imports[0]); ++i) {
    if (strcmp(name, imports[i].name) == 0) return imports[i].fn;
  }
  return NULL;
}
