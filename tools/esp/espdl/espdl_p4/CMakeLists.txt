# esp-dl backend of the IREE kernel imports (espdl_imports.h) for ESP32-P4.
idf_component_register(
    SRCS "espdl_p4.cpp" "espdl_p4_imports.c"
    INCLUDE_DIRS "." ".."
    REQUIRES espressif__esp-dl esp_timer)
