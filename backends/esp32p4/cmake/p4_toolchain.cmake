# CMake toolchain for the IREE runtime on ESP32-P4 (bare metal under ESP-IDF).
#
# Uses ESP-IDF's GCC (riscv32-esp-elf) with the firmware's ABI (ilp32f). The
# runtime is built for IREE's generic platform: no threads, no file I/O, the
# local-sync driver and the static library loader only. Time comes from
# esp_timer.
#
#   cmake -G Ninja -S third_party/iree -B third_party/iree-build-esp32p4 \
#     -DCMAKE_TOOLCHAIN_FILE=backends/esp32p4/cmake/p4_toolchain.cmake \
#     -DESP_GCC_ROOT=<dir containing bin/riscv32-esp-elf-gcc>

cmake_minimum_required(VERSION 3.21)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)
set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(NOT ESP_GCC_ROOT AND DEFINED ENV{ESP_GCC_ROOT})
  set(ESP_GCC_ROOT "$ENV{ESP_GCC_ROOT}")
endif()
if(ESP_GCC_ROOT)
  set(_prefix "${ESP_GCC_ROOT}/bin/riscv32-esp-elf-")
else()
  set(_prefix "riscv32-esp-elf-")
endif()
set(CMAKE_C_COMPILER "${_prefix}gcc")
set(CMAKE_CXX_COMPILER "${_prefix}g++")
set(CMAKE_ASM_COMPILER "${_prefix}gcc")
set(CMAKE_AR "${_prefix}ar" CACHE FILEPATH "")
set(CMAKE_RANLIB "${_prefix}ranlib" CACHE FILEPATH "")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)

# Runtime only: local-sync driver, static library loader, no plugins.
set(IREE_BUILD_COMPILER OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_POSIX OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(IREE_SYNCHRONIZATION_DISABLE_UNSAFE ON CACHE BOOL "" FORCE)
# 32-bit newlib: uint32_t is long, so printf formats warn.
set(IREE_ENABLE_WERROR_FLAG OFF CACHE BOOL "" FORCE)

set(_flags
  # As ESP-IDF 6 builds the firmware (without the Espressif extensions), and
  # against the same libc: picolibc.
  "-march=rv32imafc_zicsr_zifencei_zaamo_zalrsc -mabi=ilp32f -specs=picolibc.specs"
  "-ffunction-sections -fdata-sections"
  "-DIREE_PLATFORM_GENERIC=1 -DIREE_FILE_IO_ENABLE=0"
  "-DIREE_DEVICE_SIZE_T=uint32_t -DPRIdsz=PRIu32"
  # IREE_TIME_NOW_FN from esp_timer.
  "-include ${CMAKE_CURRENT_LIST_DIR}/iree_p4_config.h"
)
string(JOIN " " _flags ${_flags})
set(CMAKE_C_FLAGS_INIT "${_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_flags}")
set(CMAKE_ASM_FLAGS_INIT "${_flags}")
