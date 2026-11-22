// Force-included (-include) into every IREE runtime source for ESP32-P4.
#ifndef IREE_P4_CONFIG_H_
#define IREE_P4_CONFIG_H_

// esp_timer: microseconds since boot, as nanoseconds.
long long esp_timer_get_time(void);
#define IREE_TIME_NOW_FN { return esp_timer_get_time() * 1000; }

#endif  // IREE_P4_CONFIG_H_
