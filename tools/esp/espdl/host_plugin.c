// IREE executable plugin (system library) for the host: resolves the esp-dl
// kernel imports of dispatches inserted by espdl_rewrite.py to the C reference
// implementations in espdl_ref.c.
//
// Load with --executable_plugin=<path to this library>. On ESP32-P4 the same
// import names resolve to the esp-dl adapter instead.

#include <stddef.h>

#include "espdl_ref.h"
#include "iree/hal/local/executable_plugin.h"

// Argument struct of an import, as declared in espdl_rewrite.py: each
// memref<T> becomes a base pointer followed by an index offset (in elements).
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

static int espdl_conv2d_s8(void* args_ptr, void* context, void* reserved) {
  (void)context;
  (void)reserved;
  const espdl_conv2d_s8_args_t* a = (const espdl_conv2d_s8_args_t*)args_ptr;
  return espdl_ref_conv2d_s8(a->x + a->x_offset, a->w + a->w_offset,
                             a->bias + a->bias_offset, a->w_exp + a->w_exp_offset,
                             a->params + a->params_offset, a->y + a->y_offset);
}

// Signature of an import as IREE calls it (executable_library.h,
// iree_hal_executable_import_v0_t).
typedef int (*espdl_import_fn_t)(void* params, void* context, void* reserved);

typedef struct {
  const char* name;
  espdl_import_fn_t fn;
} espdl_import_t;

static const espdl_import_t kImports[] = {
    {"espdl_conv2d_s8", espdl_conv2d_s8},
};

static iree_hal_executable_plugin_status_t espdl_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  (void)environment;
  (void)param_count;
  (void)params;
  *out_self = NULL;  // stateless
  return iree_hal_executable_plugin_ok_status();
}

static void espdl_plugin_unload(void* self) { (void)self; }

static iree_hal_executable_plugin_status_t espdl_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  (void)self;
  *out_resolution = 0;
  int missing_required = 0;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* name = params->symbol_names[i];
    const int optional = iree_hal_executable_plugin_import_is_optional(name);
    if (optional) ++name;
    int found = 0;
    for (size_t j = 0; j < sizeof(kImports) / sizeof(kImports[0]); ++j) {
      if (iree_hal_executable_plugin_strcmp(name, kImports[j].name) == 0) {
        params->out_fn_ptrs[i] = (void*)kImports[j].fn;
        params->out_fn_contexts[i] = NULL;
        found = 1;
        break;
      }
    }
    if (found) continue;
    if (optional) {
      *out_resolution |= IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
    } else {
      missing_required = 1;
    }
  }
  return missing_required ? iree_hal_executable_plugin_status_from_code(
                                IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
                          : iree_hal_executable_plugin_ok_status();
}

IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t**
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  (void)reserved;
  static const iree_hal_executable_plugin_header_t header = {
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      .name = "espdl_ref",
      .description = "esp-dl kernel reference implementations (host)",
      .features = 0,
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = espdl_plugin_load,
      .unload = espdl_plugin_unload,
      .resolve = espdl_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}
