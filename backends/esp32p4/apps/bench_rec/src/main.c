// PP-OCRv6 tiny rec on ESP32-P4 through the IREE runtime.
//
// The model's dispatches are a static library (rec.espdl.p4.o) whose esp-dl
// kernel imports are resolved by the esp-dl adapter (espdl_p4_lookup). The
// firmware runs the embedded input, compares the output bytes with the
// host's output for the same input, and times each run.

#include <stdio.h>
#include <string.h>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "espdl_p4.h"
#include "io.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/runtime/api.h"
#include "rec.espdl.p4.h"

extern const uint8_t vmfb_start[] asm("rec_model_vmfb_start");
extern const uint8_t vmfb_end[] asm("rec_model_vmfb_end");
extern const uint8_t input_start[] asm("_binary_input_bin_start");
extern const uint8_t input_end[] asm("_binary_input_bin_end");
extern const uint8_t expected_start[] asm("_binary_expected_bin_start");
extern const uint8_t expected_end[] asm("_binary_expected_bin_end");

#define ENTRY "module.PaddlePaddle Graph in PIR mode"
#define RUNS 3
#define MAX_PROF_EXPORTS 16

static const iree_hal_executable_library_v0_t* g_orig_lib = NULL;
static iree_hal_executable_library_v0_t g_prof_lib;
static iree_hal_executable_dispatch_v0_t g_prof_ptrs[MAX_PROF_EXPORTS];
static uint64_t g_disp_cycles[MAX_PROF_EXPORTS];
static uint32_t g_disp_wg[MAX_PROF_EXPORTS];
static uint32_t g_disp_calls[MAX_PROF_EXPORTS];

#define DEFINE_TRAMPOLINE(IDX)                                                            \
  static int prof_dispatch_##IDX(const iree_hal_executable_environment_v0_t* env,         \
                                 const iree_hal_executable_dispatch_state_v0_t* ds,       \
                                 const iree_hal_executable_workgroup_state_v0_t* ws) {    \
    const uint32_t c0 = esp_cpu_get_cycle_count();                                        \
    const int ret = g_orig_lib->exports.ptrs[IDX](env, ds, ws);                           \
    const uint32_t dc = esp_cpu_get_cycle_count() - c0;                                   \
    g_disp_cycles[IDX] += dc;                                                             \
    ++g_disp_wg[IDX];                                                                     \
    if (ws->workgroup_id_x == 0 && ws->workgroup_id_y == 0 && ws->workgroup_id_z == 0) {  \
      ++g_disp_calls[IDX];                                                                \
    }                                                                                     \
    return ret;                                                                           \
  }

DEFINE_TRAMPOLINE(0)
DEFINE_TRAMPOLINE(1)
DEFINE_TRAMPOLINE(2)
DEFINE_TRAMPOLINE(3)
DEFINE_TRAMPOLINE(4)
DEFINE_TRAMPOLINE(5)
DEFINE_TRAMPOLINE(6)
DEFINE_TRAMPOLINE(7)
DEFINE_TRAMPOLINE(8)
DEFINE_TRAMPOLINE(9)
DEFINE_TRAMPOLINE(10)
DEFINE_TRAMPOLINE(11)
DEFINE_TRAMPOLINE(12)
DEFINE_TRAMPOLINE(13)
DEFINE_TRAMPOLINE(14)
DEFINE_TRAMPOLINE(15)

static const iree_hal_executable_dispatch_v0_t kTrampolines[MAX_PROF_EXPORTS] = {
    prof_dispatch_0,  prof_dispatch_1,  prof_dispatch_2,  prof_dispatch_3,
    prof_dispatch_4,  prof_dispatch_5,  prof_dispatch_6,  prof_dispatch_7,
    prof_dispatch_8,  prof_dispatch_9,  prof_dispatch_10, prof_dispatch_11,
    prof_dispatch_12, prof_dispatch_13, prof_dispatch_14, prof_dispatch_15,
};

static const iree_hal_executable_library_header_t** prof_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {
  const iree_hal_executable_library_header_t** hdr =
#ifdef IREE_GENERATED_STATIC_EXECUTABLE_LIBRARY_ESPDL_EXECUTABLE_
      espdl_executable_library_query(max_version, environment);
#else
      rec_espdl_linked_library_query(max_version, environment);
#endif
  if (!hdr) return NULL;
  g_orig_lib = (const iree_hal_executable_library_v0_t*)hdr;
  g_prof_lib = *g_orig_lib;
  const uint32_t n = g_orig_lib->exports.count;
  if (n <= MAX_PROF_EXPORTS) {
    for (uint32_t i = 0; i < n; ++i) g_prof_ptrs[i] = kTrampolines[i];
    g_prof_lib.exports.ptrs = g_prof_ptrs;
  }
  return &g_prof_lib.header;
}

static uint32_t g_alloc_calls = 0;
static uint32_t g_free_calls = 0;
static uint64_t g_alloc_cycles = 0;
static uint32_t g_sram_allocs = 0;
static uint32_t g_psram_allocs = 0;
static bool g_in_invoke = false;
static bool g_log_slabs = false;

static iree_status_t prof_alloc_ctl(void* self, iree_allocator_command_t command,
                                    const void* params, void** inout_ptr) {
  (void)self;
  const uint32_t c0 = esp_cpu_get_cycle_count();
  const iree_status_t st = iree_allocator_libc_ctl(NULL, command, params, inout_ptr);
  const uint32_t dc = esp_cpu_get_cycle_count() - c0;
  g_alloc_cycles += dc;
  if (command == IREE_ALLOCATOR_COMMAND_MALLOC ||
      command == IREE_ALLOCATOR_COMMAND_CALLOC ||
      command == IREE_ALLOCATOR_COMMAND_REALLOC) {
    ++g_alloc_calls;
  } else if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    ++g_free_calls;
  }
  return st;
}

static void reset_dispatch_stats(void) {
  memset(g_disp_cycles, 0, sizeof(g_disp_cycles));
  memset(g_disp_wg, 0, sizeof(g_disp_wg));
  memset(g_disp_calls, 0, sizeof(g_disp_calls));
  g_alloc_calls = 0;
  g_free_calls = 0;
  g_alloc_cycles = 0;
  g_sram_allocs = 0;
  g_psram_allocs = 0;
}

// Import provider: esp-dl kernels by name.
static iree_status_t resolve_imports(void* self, iree_host_size_t count,
                                     const char* const* symbol_names,
                                     void** out_fn_ptrs, void** out_fn_contexts,
                                     iree_hal_executable_import_resolution_t* out_resolution) {
  (void)self;
  // IREE may pass NULL when the caller does not need the resolution bits.
  iree_hal_executable_import_resolution_t resolution = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    if (out_fn_ptrs[i]) continue;
    const char* name = symbol_names[i];
    const bool optional = iree_hal_executable_import_is_optional(name);
    if (optional) ++name;
    espdl_import_fn_t fn = espdl_p4_lookup(name);
    if (fn) {
      out_fn_ptrs[i] = (void*)fn;
      out_fn_contexts[i] = NULL;
    } else if (optional) {
      resolution |= IREE_HAL_EXECUTABLE_IMPORT_RESOLUTION_MISSING_OPTIONAL;
    } else {
      return iree_make_status(IREE_STATUS_NOT_FOUND, "import %s not provided", name);
    }
  }
  if (out_resolution) *out_resolution = resolution;
  return iree_ok_status();
}

#define SRAM_RESERVE_BYTES (64 * 1024)

static iree_status_t device_slab_alloc_ctl(void* self, iree_allocator_command_t command,
                                           const void* params, void** inout_ptr) {
  (void)self;
  if (command == IREE_ALLOCATOR_COMMAND_CALLOC) {
    const iree_allocator_alloc_params_t* ap = (const iree_allocator_alloc_params_t*)params;
    const size_t n = ap->byte_length;
    if (n == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "allocations must be >0 bytes");
    }
    if (n > 256) {
      // iree_hal_heap_buffer_allocate_slab allocates [aligned header | tensor payload].
      // Route transient activations into internal SRAM when headroom permits, otherwise PSRAM.
      // Zero only the metadata prefix; skip zero-filling large tensor payloads.
      const uint32_t c0 = esp_cpu_get_cycle_count();
      void* p = NULL;
      bool in_sram = false;
      if (g_in_invoke &&
          heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) >= n + SRAM_RESERVE_BYTES) {
        p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        in_sram = (p != NULL);
      }
      if (!p) {
        p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
      }
      if (!p) {
        p = malloc(n);
      }
      if (p) memset(p, 0, 256);
      g_alloc_cycles += esp_cpu_get_cycle_count() - c0;
      ++g_alloc_calls;
      if (in_sram) ++g_sram_allocs;
      else ++g_psram_allocs;
      if (g_log_slabs) {
        printf("  slab alloc %7u B -> %s (invoke=%d, sram_free=%u)\n",
               (unsigned)n, in_sram ? "SRAM" : "PSRAM", (int)g_in_invoke,
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
      }
      if (!p) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "slab malloc failed");
      }
      *inout_ptr = p;
      return iree_ok_status();
    }
  }
  return prof_alloc_ctl(self, command, params, inout_ptr);
}

static iree_status_t create_device(iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  const iree_hal_executable_library_query_fn_t libraries[] = {prof_library_query};
  const iree_hal_executable_import_provider_t imports = {.self = NULL, .resolve = resolve_imports};
  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries, imports, host_allocator, &loader));

  iree_string_view_t identifier = iree_make_cstring_view("local-sync");
  iree_hal_allocator_t* device_allocator = NULL;
  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_allocator_t slab_allocator = {.self = NULL, .ctl = device_slab_alloc_ctl};
  iree_status_t status = iree_hal_allocator_create_heap(identifier, slab_allocator, slab_allocator,
                                                        &device_allocator);
  if (iree_status_is_ok(status)) {
    // No poll-thread runner: one thread, and local-sync works inline.
    iree_async_proactor_pool_options_t pool_options = iree_async_proactor_pool_options_default();
    memset(&pool_options.runner, 0, sizeof(pool_options.runner));
    status = iree_async_proactor_pool_create(1, /*node_ids=*/NULL, pool_options, host_allocator,
                                             &proactor_pool);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_sync_device_params_t params;
    iree_hal_sync_device_params_initialize(&params);
    iree_hal_device_create_params_t create_params = iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    status = iree_hal_sync_device_create(identifier, &params, &create_params, 1, &loader,
                                         device_allocator, host_allocator, out_device);
  }
  iree_async_proactor_pool_release(proactor_pool);
  iree_hal_allocator_release(device_allocator);
  iree_hal_executable_loader_release(loader);
  return status;
}

static void print_heap(const char* when) {
  printf("heap %-10s internal %6u KiB free (min %6u), psram %6u KiB free (min %6u)\n", when,
         (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024),
         (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) / 1024),
         (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
         (unsigned)(heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

// Compares the output with the host's: bytes, then the argmax of each step.
static int compare(const float* y, const float* expected, int steps, int classes) {
  const size_t n = (size_t)steps * classes;
  size_t differing = 0;
  for (size_t i = 0; i < n; ++i) differing += memcmp(&y[i], &expected[i], sizeof(float)) != 0;
  int argmax_same = 0;
  for (int t = 0; t < steps; ++t) {
    int a = 0, b = 0;
    for (int c = 1; c < classes; ++c) {
      if (y[t * classes + c] > y[t * classes + a]) a = c;
      if (expected[t * classes + c] > expected[t * classes + b]) b = c;
    }
    argmax_same += a == b;
  }
  printf("output: %u of %u values differ from the host, argmax same at %d/%d steps\n",
         (unsigned)differing, (unsigned)n, argmax_same, steps);
  return differing == 0;
}

static iree_status_t run_case(const char* label, const uint8_t* vmfb_data, size_t vmfb_size) {
  iree_allocator_t host_allocator = {.self = NULL, .ctl = prof_alloc_ctl};
  const iree_hal_dim_t in_dims[] = MODEL_INPUT_DIMS;
  const iree_hal_dim_t out_dims[] = MODEL_OUTPUT_DIMS;
  const int steps = (int)out_dims[1], classes = (int)out_dims[2];
  const size_t out_bytes = (size_t)steps * classes * sizeof(float);
  if ((size_t)(expected_end - expected_start) != out_bytes) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "expected.bin size");
  }

  printf("\n=== %s ===\n", label);
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_create(&instance_options, host_allocator, &instance));

  iree_hal_device_t* device = NULL;
  iree_runtime_session_t* session = NULL;
  iree_status_t status = create_device(host_allocator, &device);
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(instance, &session_options, device,
                                                     host_allocator, &session);
  }
  if (iree_status_is_ok(status)) {
    const int64_t t0 = esp_timer_get_time();
    status = iree_runtime_session_append_bytecode_module_from_memory(
        session, iree_make_const_byte_span(vmfb_data, vmfb_size), iree_allocator_null());
    if (iree_status_is_ok(status)) {
      printf("module loaded in %lld ms (%u bytes at %p)\n", (esp_timer_get_time() - t0) / 1000,
             (unsigned)vmfb_size, (const void*)vmfb_data);
    }
  }
  print_heap("loaded");

  float* y = heap_caps_malloc(out_bytes, MALLOC_CAP_SPIRAM);
  if (iree_status_is_ok(status) && !y) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "output buffer");
  }
  int ok_runs = 0;
  for (int r = 0; r < RUNS && iree_status_is_ok(status); ++r) {
    iree_runtime_call_t call;
    memset(&call, 0, sizeof(call));
    status = iree_runtime_call_initialize_by_name(session, iree_make_cstring_view(ENTRY), &call);
    iree_hal_buffer_view_t* input = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, iree_hal_device_allocator(device), MODEL_INPUT_RANK, in_dims,
          IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                                     .usage = IREE_HAL_BUFFER_USAGE_DEFAULT},
          iree_make_const_byte_span(input_start, input_end - input_start), &input);
    }
    if (iree_status_is_ok(status)) {
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, input);
    }
    iree_hal_buffer_view_release(input);

    espdl_p4_reset_stats();
    reset_dispatch_stats();
    g_log_slabs = (r == 0);
    g_in_invoke = true;
    const int64_t t0 = esp_timer_get_time();
    if (iree_status_is_ok(status)) status = iree_runtime_call_invoke(&call, 0);
    const int64_t t1 = esp_timer_get_time();
    g_in_invoke = false;
    g_log_slabs = false;
    espdl_p4_stats_t st;
    espdl_p4_get_stats(&st);

    iree_hal_buffer_view_t* output = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &output);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_transfer_d2h(device, iree_hal_buffer_view_buffer(output), 0, y,
                                            out_bytes, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                                            iree_infinite_timeout());
    }
    iree_hal_buffer_view_release(output);
    iree_runtime_call_deinitialize(&call);
    if (iree_status_is_ok(status)) {
      const int64_t total_us = t1 - t0;
      const int64_t espdl_us = st.conv_setup_us + st.conv_kernel_us + st.lut_us +
                               st.binary_setup_us + st.binary_kernel_us +
                               st.quant_us + st.dequant_us + st.pool_us;
      printf("run %d: total %lld ms (%lld us) | conv %lu (setup %.2f + kernel %.2f ms) | "
             "lut %lu (%.2f ms) | bin %lu (setup %.2f + kernel %.2f ms) | "
             "pool %lu (%.2f ms) | q %lu (%.2f ms) | dq %lu (%.2f ms) | iree/other %.2f ms\n",
             r, total_us / 1000, total_us,
             (unsigned long)st.conv_calls, st.conv_setup_us / 1000.0, st.conv_kernel_us / 1000.0,
             (unsigned long)st.lut_calls, st.lut_us / 1000.0,
             (unsigned long)st.binary_calls, st.binary_setup_us / 1000.0, st.binary_kernel_us / 1000.0,
             (unsigned long)st.pool_calls, st.pool_us / 1000.0,
             (unsigned long)st.quant_calls, st.quant_us / 1000.0,
             (unsigned long)st.dequant_calls, st.dequant_us / 1000.0,
             (total_us - espdl_us) / 1000.0);
      if (r == RUNS - 1 && g_orig_lib) {
        double sum_iree_disp_ms = 0.0;
        double sum_espdl_disp_ms = 0.0;
        for (uint32_t i = 0; i < g_orig_lib->exports.count && i < MAX_PROF_EXPORTS; ++i) {
          const char* name = g_orig_lib->exports.names ? g_orig_lib->exports.names[i] : "?";
          const double ms = g_disp_cycles[i] / 360000.0;
          const bool is_espdl = (strcmp(name, "conv2d_s8") == 0 ||
                                 strcmp(name, "lut_s8") == 0 ||
                                 strcmp(name, "lut_s8_ip") == 0 ||
                                 strcmp(name, "binary_s8") == 0 ||
                                 strcmp(name, "binary_s8_ip") == 0 ||
                                 strcmp(name, "quant_s8") == 0 ||
                                 strcmp(name, "dequant_s8_f32") == 0 ||
                                 strcmp(name, "avg_pool2d_s8") == 0);
          if (is_espdl) sum_espdl_disp_ms += ms;
          else sum_iree_disp_ms += ms;
          printf("  dispatch[%2lu] %7.2f ms (%3lu calls, %5lu wgs) %s\n",
                 (unsigned long)i, ms, (unsigned long)g_disp_calls[i],
                 (unsigned long)g_disp_wg[i], name);
        }
        const double vm_hal_ms = (total_us / 1000.0) - sum_iree_disp_ms - sum_espdl_disp_ms;
        const double alloc_ms = g_alloc_cycles / 360000.0;
        printf("  summary: espdl_disp %.2f ms (inner %.2f + teardown/glue %.2f) | iree_disp %.2f ms | vm/hal %.2f ms (alloc %.2f ms: %lu alloc [%lu sram, %lu psram], %lu free)\n",
               sum_espdl_disp_ms, espdl_us / 1000.0, sum_espdl_disp_ms - espdl_us / 1000.0,
               sum_iree_disp_ms, vm_hal_ms, alloc_ms,
               (unsigned long)g_alloc_calls, (unsigned long)g_sram_allocs,
               (unsigned long)g_psram_allocs, (unsigned long)g_free_calls);
      }
      ok_runs += compare(y, (const float*)expected_start, steps, classes);
    }
  }
  if (iree_status_is_ok(status)) {
    printf("%d/%d runs bit-exact with the host (input text \"%s\")\n", ok_runs, RUNS,
           MODEL_INPUT_TEXT);
  }

  heap_caps_free(y);
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
  return status;
}

static iree_status_t run(void) {
  const size_t vmfb_size = (size_t)(vmfb_end - vmfb_start);
  IREE_RETURN_IF_ERROR(run_case("flash xip (.rodata)", vmfb_start, vmfb_size));

  uint8_t* psram_vmfb = heap_caps_aligned_alloc(64, vmfb_size, MALLOC_CAP_SPIRAM);
  if (!psram_vmfb) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "psram vmfb alloc");
  }
  const int64_t tc0 = esp_timer_get_time();
  memcpy(psram_vmfb, vmfb_start, vmfb_size);
  printf("\ncopied vmfb (%u bytes) flash->psram in %.2f ms\n", (unsigned)vmfb_size,
         (esp_timer_get_time() - tc0) / 1000.0);
  iree_status_t status = run_case("psram copy", psram_vmfb, vmfb_size);
  heap_caps_free(psram_vmfb);
  return status;
}

void app_main(void) {
  print_heap("start");
  iree_status_t status = run();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stdout, status);
    iree_status_free(status);
    printf("FAILED\n");
  }
  printf("done\n");
}
