# IREE runtime for ESP32-P4, prebuilt by `just esp-iree-p4` with
# p4_toolchain.cmake (generic platform, local-sync, static library loader).
set(repo "${CMAKE_CURRENT_LIST_DIR}/../../../..")
set(src "${repo}/third_party/iree/runtime/src")
set(build "${repo}/third_party/iree-build-esp32p4")

# The proactor replaces the runtime's iree_async_proactor_create_platform
# (see the file); WHOLE_ARCHIVE keeps it ahead of the runtime archive.
idf_component_register(
    SRCS "iree_p4_proactor.c"
    REQUIRES esp_timer
    WHOLE_ARCHIVE)

set(libs
    "${build}/runtime/src/iree/runtime/libiree_runtime_unified.a"
    "${build}/runtime/src/iree/hal/local/loaders/libiree_hal_local_loaders_static_library_loader.a"
    "${build}/runtime/src/iree/hal/local/libiree_hal_local_executable_library_util.a"
    "${build}/build_tools/third_party/flatcc/libflatcc_parsing.a")
foreach(lib ${libs})
  if(NOT EXISTS "${lib}")
    message(FATAL_ERROR "${lib} not found: run `just esp-iree-p4` first")
  endif()
endforeach()

# SYSTEM: IREE's headers warn on 32-bit targets and IDF builds with -Werror.
target_include_directories(${COMPONENT_LIB} SYSTEM PUBLIC "${src}" "${build}/runtime/src")
# The same warnings from IREE macros expanded in our code.
target_compile_options(${COMPONENT_LIB} PUBLIC -Wno-pointer-sign -Wno-type-limits)

# Headers must see the configuration the runtime was built with.
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    IREE_PLATFORM_GENERIC=1 IREE_FILE_IO_ENABLE=0 IREE_DEVICE_SIZE_T=uint32_t
    PRIdsz=PRIu32 IREE_SYNCHRONIZATION_DISABLE_UNSAFE=1
    IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)
target_link_libraries(${COMPONENT_LIB} INTERFACE
    "-Wl,--start-group" ${libs} "-Wl,--end-group" m)
