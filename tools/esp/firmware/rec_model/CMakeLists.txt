# PP-OCRv6 tiny rec compiled for ESP32-P4 (`just esp-ppocr-p4 rec`): the
# static library of its dispatches, its bytecode module, and one input with
# the host's output for it (`gen_model_io.py`).
set(out "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/esp-out/rec")
foreach(f rec.espdl.p4.o rec.espdl.p4.vmfb model_io/input.bin model_io/expected.bin model_io/io.h)
  if(NOT EXISTS "${out}/${f}")
    message(FATAL_ERROR "${out}/${f} not found: run `just esp-p4-rec` first")
  endif()
endforeach()

set(VMFB "${out}/rec.espdl.p4.vmfb")
configure_file("${CMAKE_CURRENT_LIST_DIR}/model_data.S.in" "${CMAKE_CURRENT_BINARY_DIR}/model_data.S" @ONLY)

idf_component_register(
    SRCS "${CMAKE_CURRENT_BINARY_DIR}/model_data.S"
    INCLUDE_DIRS "${out}" "${out}/model_io"
    EMBED_FILES "${out}/model_io/input.bin" "${out}/model_io/expected.bin"
    REQUIRES iree_runtime)
# .incbin is not a tracked dependency: rebuild when the module changes.
set_property(SOURCE "${CMAKE_CURRENT_BINARY_DIR}/model_data.S" APPEND PROPERTY OBJECT_DEPENDS "${VMFB}")
target_link_libraries(${COMPONENT_LIB} INTERFACE "${out}/rec.espdl.p4.o")
