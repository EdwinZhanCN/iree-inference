# Host executable plugin with the esp-dl kernel reference implementations.
# Only IREE's plugin header is needed (IREE_SOURCE_DIR/runtime/src).
cmake_minimum_required(VERSION 3.21)
project(espdl_host_plugin C)

set(IREE_SOURCE_DIR "" CACHE PATH "IREE source checkout")
if(NOT EXISTS "${IREE_SOURCE_DIR}/runtime/src/iree/hal/local/executable_plugin.h")
  message(FATAL_ERROR "Set IREE_SOURCE_DIR to an IREE checkout (just _iree-clone).")
endif()

add_library(espdl_host_plugin MODULE host_plugin.c espdl_ref.c)
target_include_directories(espdl_host_plugin PRIVATE "${IREE_SOURCE_DIR}/runtime/src")
set_target_properties(espdl_host_plugin PROPERTIES C_STANDARD 11 C_VISIBILITY_PRESET hidden)
if(NOT MSVC)
  target_compile_options(espdl_host_plugin PRIVATE -Wall -Wextra)
  target_link_libraries(espdl_host_plugin PRIVATE m)
endif()

# The reference kernels with visible symbols, for gen_vectors.py (ctypes).
add_library(espdl_ref SHARED espdl_ref.c)
set_target_properties(espdl_ref PROPERTIES C_STANDARD 11)
if(NOT MSVC)
  target_compile_options(espdl_ref PRIVATE -Wall -Wextra)
  target_link_libraries(espdl_ref PRIVATE m)
endif()
