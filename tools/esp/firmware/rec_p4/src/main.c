// PP-OCRv6 tiny rec on ESP32-P4 through the IREE runtime.
//
// The model's dispatches are a static library (rec.espdl.p4.o) whose esp-dl
// kernel imports are resolved by the esp-dl adapter (espdl_p4_lookup). The
// firmware runs the embedded input, compares the output bytes with the
// host's output for the same input, and times each run.

#include <stdio.h>
#include <string.h>

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

static iree_status_t create_device(iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  const iree_hal_executable_library_query_fn_t libraries[] = {rec_espdl_linked_library_query};
  const iree_hal_executable_import_provider_t imports = {.self = NULL, .resolve = resolve_imports};
  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_static_library_loader_create(
      IREE_ARRAYSIZE(libraries), libraries, imports, host_allocator, &loader));

  iree_string_view_t identifier = iree_make_cstring_view("local-sync");
  iree_hal_allocator_t* device_allocator = NULL;
  iree_async_proactor_pool_t* proactor_pool = NULL;
  iree_status_t status = iree_hal_allocator_create_heap(identifier, host_allocator, host_allocator,
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
  iree_allocator_t host_allocator = iree_allocator_system();
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
    const int64_t t0 = esp_timer_get_time();
    if (iree_status_is_ok(status)) status = iree_runtime_call_invoke(&call, 0);
    const int64_t t1 = esp_timer_get_time();
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
                               st.binary_setup_us + st.binary_kernel_us;
      printf("run %d: total %lld ms (%lld us) | conv %lu (setup %.2f + kernel %.2f ms) | "
             "lut %lu (%.2f ms) | bin %lu (setup %.2f + kernel %.2f ms) | iree/other %.2f ms\n",
             r, total_us / 1000, total_us,
             (unsigned long)st.conv_calls, st.conv_setup_us / 1000.0, st.conv_kernel_us / 1000.0,
             (unsigned long)st.lut_calls, st.lut_us / 1000.0,
             (unsigned long)st.binary_calls, st.binary_setup_us / 1000.0, st.binary_kernel_us / 1000.0,
             (total_us - espdl_us) / 1000.0);
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
