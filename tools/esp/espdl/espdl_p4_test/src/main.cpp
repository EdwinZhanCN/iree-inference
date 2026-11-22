// Runs each test vector through the esp-dl backend (espdl_p4) and compares
// the output with the reference kernels' bytes, computed on the host.

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "espdl_p4.h"
#include "vectors.h"

static int run(const espdl_case_t* c, int8_t* y) {
  const int8_t* i8[5];
  const int32_t* i32[5];
  for (int k = 0; k < 5; ++k) {
    i8[k] = (const int8_t*)c->in[k];
    i32[k] = (const int32_t*)c->in[k];
  }
  switch (c->kind) {
    case ESPDL_CASE_CONV:
      return espdl_p4_conv2d_s8(i8[0], i8[1], i32[2], i8[3], i32[4], y);
    case ESPDL_CASE_BINARY:
      return espdl_p4_binary_s8(i8[0], i8[1], i32[2], y);
    default:
      return espdl_p4_lut_s8(i8[0], i8[1], i32[2], y);
  }
}

extern "C" void app_main(void) {
  int failed = 0;
  for (int i = 0; i < espdl_case_count; ++i) {
    const espdl_case_t* c = &espdl_cases[i];
    int8_t* y = (int8_t*)heap_caps_aligned_alloc(16, c->n_out + 16, MALLOC_CAP_8BIT);
    if (!y) {
      printf("FAIL %-48s out of memory (%d bytes)\n", c->name, c->n_out);
      ++failed;
      continue;
    }
    memset(y, 0x55, c->n_out + 16);
    int64_t t0 = esp_timer_get_time();
    int rc = run(c, y);
    int64_t us = esp_timer_get_time() - t0;
    int diff = 0, first = -1;
    for (int k = 0; k < c->n_out; ++k) {
      if (y[k] != c->expected[k]) {
        if (first < 0) first = k;
        ++diff;
      }
    }
    int overrun = 0;
    for (int k = 0; k < 16; ++k) overrun |= y[c->n_out + k] != 0x55;
    if (rc != 0 || diff || overrun) {
      ++failed;
      printf("FAIL %-48s rc %d, %d/%d differ", c->name, rc, diff, c->n_out);
      if (first >= 0) printf(" (first at %d: got %d, want %d)", first, y[first], c->expected[first]);
      if (overrun) printf(", wrote past the output");
      printf("\n");
    } else {
      printf("ok   %-48s %6lld us\n", c->name, (long long)us);
    }
    heap_caps_free(y);
  }
  printf("%d/%d cases passed\n", espdl_case_count - failed, espdl_case_count);
}
