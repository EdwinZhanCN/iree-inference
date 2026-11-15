// Exposes sizeof/offsetof/constant values from the real IREE headers so the
// Rust unit tests can verify the hand-written FFI declarations in src/lib.rs.
//
// Only compiled against headers; does not require the IREE runtime library.

#include <stddef.h>
#include <stdint.h>

#include "iree/runtime/api.h"
#include "iree/vm/bytecode/archive.h"

typedef struct iree_sys_layout_entry_t {
  const char* name;
  uint64_t value;
} iree_sys_layout_entry_t;

#define SIZE(T) {"size:" #T, (uint64_t)sizeof(T)}, {"align:" #T, (uint64_t)_Alignof(T)}
#define OFF(T, F) {"offset:" #T "." #F, (uint64_t)offsetof(T, F)}
#define CONST(C) {"const:" #C, (uint64_t)(C)}

static const iree_sys_layout_entry_t kEntries[] = {
    // Scalar typedefs.
    SIZE(iree_host_size_t),
    SIZE(iree_device_size_t),
    SIZE(iree_hal_dim_t),
    SIZE(iree_status_t),
    SIZE(iree_allocator_command_t),
    SIZE(iree_timeout_type_t),
    SIZE(iree_vm_function_linkage_t),
    SIZE(iree_hal_element_type_t),
    SIZE(iree_hal_encoding_type_t),
    SIZE(iree_hal_buffer_usage_t),
    SIZE(iree_hal_memory_access_t),
    SIZE(iree_hal_memory_type_t),
    SIZE(iree_hal_queue_affinity_t),
    SIZE(iree_vm_context_flags_t),
    SIZE(iree_runtime_session_builtins_t),

    // base
    SIZE(iree_allocator_t),
    OFF(iree_allocator_t, self),
    OFF(iree_allocator_t, ctl),
    SIZE(iree_string_view_t),
    OFF(iree_string_view_t, data),
    OFF(iree_string_view_t, size),
    SIZE(iree_const_byte_span_t),
    OFF(iree_const_byte_span_t, data),
    OFF(iree_const_byte_span_t, data_length),
    SIZE(iree_timeout_t),
    OFF(iree_timeout_t, type),
    OFF(iree_timeout_t, nanos),

    // runtime
    SIZE(iree_runtime_instance_options_t),
    OFF(iree_runtime_instance_options_t, driver_registry),
    SIZE(iree_runtime_session_options_t),
    OFF(iree_runtime_session_options_t, context_flags),
    OFF(iree_runtime_session_options_t, builtin_modules),
    SIZE(iree_runtime_call_t),
    OFF(iree_runtime_call_t, session),
    OFF(iree_runtime_call_t, function),
    OFF(iree_runtime_call_t, inputs),
    OFF(iree_runtime_call_t, outputs),

    // vm
    SIZE(iree_vm_function_t),
    OFF(iree_vm_function_t, module),
    OFF(iree_vm_function_t, linkage),
    OFF(iree_vm_function_t, ordinal),
    SIZE(iree_vm_function_signature_t),
    OFF(iree_vm_function_signature_t, calling_convention),
    SIZE(iree_vm_module_signature_t),
    OFF(iree_vm_module_signature_t, version),
    OFF(iree_vm_module_signature_t, attr_count),
    OFF(iree_vm_module_signature_t, import_function_count),
    OFF(iree_vm_module_signature_t, export_function_count),
    OFF(iree_vm_module_signature_t, internal_function_count),

    // hal
    SIZE(iree_hal_buffer_params_t),
    OFF(iree_hal_buffer_params_t, usage),
    OFF(iree_hal_buffer_params_t, access),
    OFF(iree_hal_buffer_params_t, type),
    OFF(iree_hal_buffer_params_t, queue_affinity),
    OFF(iree_hal_buffer_params_t, min_alignment),

    // Constants.
    CONST(IREE_STATUS_OK),
    CONST(IREE_STATUS_CANCELLED),
    CONST(IREE_STATUS_UNKNOWN),
    CONST(IREE_STATUS_INVALID_ARGUMENT),
    CONST(IREE_STATUS_DEADLINE_EXCEEDED),
    CONST(IREE_STATUS_NOT_FOUND),
    CONST(IREE_STATUS_ALREADY_EXISTS),
    CONST(IREE_STATUS_PERMISSION_DENIED),
    CONST(IREE_STATUS_RESOURCE_EXHAUSTED),
    CONST(IREE_STATUS_FAILED_PRECONDITION),
    CONST(IREE_STATUS_ABORTED),
    CONST(IREE_STATUS_OUT_OF_RANGE),
    CONST(IREE_STATUS_UNIMPLEMENTED),
    CONST(IREE_STATUS_INTERNAL),
    CONST(IREE_STATUS_UNAVAILABLE),
    CONST(IREE_STATUS_DATA_LOSS),
    CONST(IREE_STATUS_UNAUTHENTICATED),
    CONST(IREE_STATUS_DEFERRED),
    CONST(IREE_STATUS_INCOMPATIBLE),
    CONST(IREE_STATUS_CODE_MASK),
    CONST(IREE_ALLOCATOR_COMMAND_MALLOC),
    CONST(IREE_ALLOCATOR_COMMAND_CALLOC),
    CONST(IREE_ALLOCATOR_COMMAND_REALLOC),
    CONST(IREE_ALLOCATOR_COMMAND_FREE),
    CONST(IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT),
    CONST(IREE_TIMEOUT_ABSOLUTE),
    CONST(IREE_TIMEOUT_RELATIVE),
    CONST(IREE_TIME_INFINITE_FUTURE),
    CONST(IREE_VM_FUNCTION_LINKAGE_INTERNAL),
    CONST(IREE_VM_FUNCTION_LINKAGE_IMPORT),
    CONST(IREE_VM_FUNCTION_LINKAGE_EXPORT),
    CONST(IREE_VM_CONTEXT_FLAG_NONE),
    CONST(IREE_RUNTIME_SESSION_BUILTIN_ALL),
    CONST(IREE_RUNTIME_CALL_FLAG_RESERVED),
    CONST(IREE_HAL_ENCODING_TYPE_OPAQUE),
    CONST(IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR),
    CONST(IREE_HAL_BUFFER_USAGE_DEFAULT),
    CONST(IREE_HAL_MEMORY_ACCESS_ALL),
    CONST(IREE_HAL_MEMORY_TYPE_OPTIMAL),
    CONST(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL),
    CONST(IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE),
    CONST(IREE_HAL_QUEUE_AFFINITY_ANY),
    CONST(IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT),
    CONST(IREE_HAL_ELEMENT_TYPE_NONE),
    CONST(IREE_HAL_ELEMENT_TYPE_BOOL_8),
    CONST(IREE_HAL_ELEMENT_TYPE_INT_8),
    CONST(IREE_HAL_ELEMENT_TYPE_SINT_8),
    CONST(IREE_HAL_ELEMENT_TYPE_UINT_8),
    CONST(IREE_HAL_ELEMENT_TYPE_INT_16),
    CONST(IREE_HAL_ELEMENT_TYPE_SINT_16),
    CONST(IREE_HAL_ELEMENT_TYPE_UINT_16),
    CONST(IREE_HAL_ELEMENT_TYPE_INT_32),
    CONST(IREE_HAL_ELEMENT_TYPE_SINT_32),
    CONST(IREE_HAL_ELEMENT_TYPE_UINT_32),
    CONST(IREE_HAL_ELEMENT_TYPE_INT_64),
    CONST(IREE_HAL_ELEMENT_TYPE_SINT_64),
    CONST(IREE_HAL_ELEMENT_TYPE_UINT_64),
    CONST(IREE_HAL_ELEMENT_TYPE_FLOAT_16),
    CONST(IREE_HAL_ELEMENT_TYPE_FLOAT_32),
    CONST(IREE_HAL_ELEMENT_TYPE_FLOAT_64),
    CONST(IREE_HAL_ELEMENT_TYPE_BFLOAT_16),
};

const iree_sys_layout_entry_t* iree_sys_layout_entries(size_t* out_count) {
  *out_count = sizeof(kEntries) / sizeof(kEntries[0]);
  return kEntries;
}
