//! Hand-written raw FFI bindings to the IREE runtime C API.
//!
//! Scope: the inference subset of `iree/runtime/api.h` plus the HAL buffer-view /
//! device-transfer and VM reflection functions needed to load `.vmfb` modules,
//! pass tensors in, invoke exported functions and read tensors back.
//!
//! Targets the IREE 3.14-based fork pinned in the workspace manifest.
//! Every `#[repr(C)]` struct and constant below is
//! checked against the real C headers by the `layout` unit test (see
//! `csrc/layout_check.c`); run `cargo test -p iree-sys` after bumping IREE.
//!
//! Things that are `static inline` / macros in C (and therefore have no symbol to
//! link against) are re-implemented here as Rust `const fn`s / `fn`s:
//! `iree_status_code`, `iree_status_is_ok`, `iree_allocator_system`,
//! `iree_allocator_null`, `iree_make_string_view`, `iree_infinite_timeout`,
//! `iree_hal_make_element_type`, ...
//!
//! Everything here is `unsafe`; use the `iree-runtime` crate for a safe API.

#![no_std]
#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    clippy::missing_safety_doc
)]

use core::ffi::{c_char, c_void};
use core::marker::{PhantomData, PhantomPinned};

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

macro_rules! opaque {
    ($($name:ident),* $(,)?) => {$(
        #[repr(C)]
        pub struct $name {
            _data: [u8; 0],
            _marker: PhantomData<(*mut u8, PhantomPinned)>,
        }
    )*};
}

opaque!(
    iree_status_handle_t,
    iree_runtime_instance_t,
    iree_runtime_session_t,
    iree_hal_driver_registry_t,
    iree_hal_device_t,
    iree_hal_allocator_t,
    iree_hal_buffer_t,
    iree_hal_buffer_view_t,
    iree_vm_instance_t,
    iree_vm_context_t,
    iree_vm_module_t,
    iree_vm_list_t,
);

// ---------------------------------------------------------------------------
// base: scalar typedefs
// ---------------------------------------------------------------------------

/// `IREE_HOST_SIZE_T` defaults to `size_t`.
pub type iree_host_size_t = usize;
/// `IREE_DEVICE_SIZE_T` defaults to `size_t`.
pub type iree_device_size_t = usize;
pub type iree_time_t = i64;

// ---------------------------------------------------------------------------
// base: status
// ---------------------------------------------------------------------------

/// Pointer-sized status. `NULL` means OK; the low 5 bits carry the status code
/// and the remaining bits (if any) point at heap storage that must be freed with
/// [`iree_status_free`] / [`iree_status_ignore`].
pub type iree_status_t = *mut iree_status_handle_t;

pub type iree_status_code_t = u32;
pub const IREE_STATUS_OK: iree_status_code_t = 0;
pub const IREE_STATUS_CANCELLED: iree_status_code_t = 1;
pub const IREE_STATUS_UNKNOWN: iree_status_code_t = 2;
pub const IREE_STATUS_INVALID_ARGUMENT: iree_status_code_t = 3;
pub const IREE_STATUS_DEADLINE_EXCEEDED: iree_status_code_t = 4;
pub const IREE_STATUS_NOT_FOUND: iree_status_code_t = 5;
pub const IREE_STATUS_ALREADY_EXISTS: iree_status_code_t = 6;
pub const IREE_STATUS_PERMISSION_DENIED: iree_status_code_t = 7;
pub const IREE_STATUS_RESOURCE_EXHAUSTED: iree_status_code_t = 8;
pub const IREE_STATUS_FAILED_PRECONDITION: iree_status_code_t = 9;
pub const IREE_STATUS_ABORTED: iree_status_code_t = 10;
pub const IREE_STATUS_OUT_OF_RANGE: iree_status_code_t = 11;
pub const IREE_STATUS_UNIMPLEMENTED: iree_status_code_t = 12;
pub const IREE_STATUS_INTERNAL: iree_status_code_t = 13;
pub const IREE_STATUS_UNAVAILABLE: iree_status_code_t = 14;
pub const IREE_STATUS_DATA_LOSS: iree_status_code_t = 15;
pub const IREE_STATUS_UNAUTHENTICATED: iree_status_code_t = 16;
pub const IREE_STATUS_DEFERRED: iree_status_code_t = 17;
pub const IREE_STATUS_INCOMPATIBLE: iree_status_code_t = 18;
pub const IREE_STATUS_CODE_MASK: iree_status_code_t = 0x1F;

/// `#define iree_status_code(value)`
#[inline]
pub fn iree_status_code(status: iree_status_t) -> iree_status_code_t {
    ((status as usize) & (IREE_STATUS_CODE_MASK as usize)) as iree_status_code_t
}

/// `#define iree_status_is_ok(value)`
#[inline]
pub fn iree_status_is_ok(status: iree_status_t) -> bool {
    status.is_null()
}

// ---------------------------------------------------------------------------
// base: allocator / spans / strings / time
// ---------------------------------------------------------------------------

/// `iree_allocator_command_t` (C enum, `int`-sized).
pub type iree_allocator_command_t = u32;
pub const IREE_ALLOCATOR_COMMAND_MALLOC: iree_allocator_command_t = 0;
pub const IREE_ALLOCATOR_COMMAND_CALLOC: iree_allocator_command_t = 1;
pub const IREE_ALLOCATOR_COMMAND_REALLOC: iree_allocator_command_t = 2;
pub const IREE_ALLOCATOR_COMMAND_FREE: iree_allocator_command_t = 3;

/// `IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT`: rodata segments inside a `.vmfb` are
/// aligned to this many bytes *relative to the start of the archive*, so
/// in-memory modules should start on (at least) this alignment.
pub const IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT: usize = 64;

/// `iree_status_from_code`: a code-only status (no heap payload).
#[inline]
pub fn iree_status_from_code(code: iree_status_code_t) -> iree_status_t {
    ((code & IREE_STATUS_CODE_MASK) as usize) as iree_status_t
}

pub type iree_allocator_ctl_fn_t = Option<
    unsafe extern "C" fn(
        self_: *mut c_void,
        command: iree_allocator_command_t,
        params: *const c_void,
        inout_ptr: *mut *mut c_void,
    ) -> iree_status_t,
>;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_allocator_t {
    pub self_: *mut c_void,
    pub ctl: iree_allocator_ctl_fn_t,
}

/// `static inline iree_allocator_system()` — only valid with the default
/// `IREE_ALLOCATOR_SYSTEM=libc` CMake configuration.
#[inline]
pub fn iree_allocator_system() -> iree_allocator_t {
    iree_allocator_t {
        self_: core::ptr::null_mut(),
        ctl: Some(iree_allocator_libc_ctl),
    }
}

/// `static inline iree_allocator_null()`
#[inline]
pub const fn iree_allocator_null() -> iree_allocator_t {
    iree_allocator_t {
        self_: core::ptr::null_mut(),
        ctl: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_string_view_t {
    pub data: *const c_char,
    pub size: iree_host_size_t,
}

impl iree_string_view_t {
    /// Borrowing view over `s` (not NUL-terminated; IREE uses the explicit size).
    #[inline]
    pub fn new(s: &str) -> Self {
        Self {
            data: s.as_ptr().cast(),
            size: s.len(),
        }
    }

    #[inline]
    pub const fn empty() -> Self {
        Self {
            data: core::ptr::null(),
            size: 0,
        }
    }

    /// # Safety
    /// `data` must point to `size` readable bytes that outlive the returned slice.
    #[inline]
    pub unsafe fn as_bytes<'a>(&self) -> &'a [u8] {
        if self.data.is_null() || self.size == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.data.cast(), self.size) }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_const_byte_span_t {
    pub data: *const u8,
    pub data_length: iree_host_size_t,
}

/// `iree_timeout_type_t` (C enum, `int`-sized).
pub type iree_timeout_type_t = u32;
pub const IREE_TIMEOUT_ABSOLUTE: iree_timeout_type_t = 0;
pub const IREE_TIMEOUT_RELATIVE: iree_timeout_type_t = 1;
pub const IREE_TIME_INFINITE_FUTURE: iree_time_t = i64::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_timeout_t {
    pub type_: iree_timeout_type_t,
    pub nanos: iree_time_t,
}

/// `static inline iree_infinite_timeout()`
#[inline]
pub const fn iree_infinite_timeout() -> iree_timeout_t {
    iree_timeout_t {
        type_: IREE_TIMEOUT_ABSOLUTE,
        nanos: IREE_TIME_INFINITE_FUTURE,
    }
}

// ---------------------------------------------------------------------------
// runtime: instance / session / call
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_runtime_instance_options_t {
    pub driver_registry: *mut iree_hal_driver_registry_t,
}

pub type iree_vm_context_flags_t = u32;
pub const IREE_VM_CONTEXT_FLAG_NONE: iree_vm_context_flags_t = 0;

pub type iree_runtime_session_builtins_t = u64;
pub const IREE_RUNTIME_SESSION_BUILTIN_ALL: iree_runtime_session_builtins_t = u64::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_runtime_session_options_t {
    pub context_flags: iree_vm_context_flags_t,
    pub builtin_modules: iree_runtime_session_builtins_t,
}

pub type iree_runtime_call_flags_t = u32;
pub const IREE_RUNTIME_CALL_FLAG_RESERVED: iree_runtime_call_flags_t = 0;

/// Stateful call builder. Plain data (no self-references) so it may be moved
/// after [`iree_runtime_call_initialize`]; must be released with
/// [`iree_runtime_call_deinitialize`].
#[repr(C)]
#[derive(Debug)]
pub struct iree_runtime_call_t {
    pub session: *mut iree_runtime_session_t,
    pub function: iree_vm_function_t,
    pub inputs: *mut iree_vm_list_t,
    pub outputs: *mut iree_vm_list_t,
}

// ---------------------------------------------------------------------------
// vm: functions / modules
// ---------------------------------------------------------------------------

/// `iree_vm_function_linkage_t` (C enum, `int`-sized).
pub type iree_vm_function_linkage_t = u32;
pub const IREE_VM_FUNCTION_LINKAGE_INTERNAL: iree_vm_function_linkage_t = 0;
pub const IREE_VM_FUNCTION_LINKAGE_IMPORT: iree_vm_function_linkage_t = 1;
pub const IREE_VM_FUNCTION_LINKAGE_EXPORT: iree_vm_function_linkage_t = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_vm_function_t {
    pub module: *mut iree_vm_module_t,
    pub linkage: u16,
    pub ordinal: u16,
}

impl iree_vm_function_t {
    pub const fn null() -> Self {
        Self {
            module: core::ptr::null_mut(),
            linkage: 0,
            ordinal: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_vm_function_signature_t {
    /// e.g. `0rr_r` (version 0; two refs in, one ref out).
    pub calling_convention: iree_string_view_t,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_vm_module_signature_t {
    pub version: u32,
    pub attr_count: iree_host_size_t,
    pub import_function_count: iree_host_size_t,
    pub export_function_count: iree_host_size_t,
    pub internal_function_count: iree_host_size_t,
}

// ---------------------------------------------------------------------------
// hal: element types / buffers
// ---------------------------------------------------------------------------

pub type iree_hal_dim_t = iree_device_size_t;
pub type iree_hal_element_type_t = u32;
pub type iree_hal_encoding_type_t = u32;
pub type iree_hal_numerical_type_t = u8;

pub const IREE_HAL_NUMERICAL_TYPE_UNKNOWN: iree_hal_numerical_type_t = 0x00;
pub const IREE_HAL_NUMERICAL_TYPE_INTEGER: iree_hal_numerical_type_t = 0x10;
pub const IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED: iree_hal_numerical_type_t = 0x11;
pub const IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED: iree_hal_numerical_type_t = 0x12;
pub const IREE_HAL_NUMERICAL_TYPE_BOOLEAN: iree_hal_numerical_type_t = 0x13;
pub const IREE_HAL_NUMERICAL_TYPE_FLOAT: iree_hal_numerical_type_t = 0x20;
pub const IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE: iree_hal_numerical_type_t = 0x21;
pub const IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN: iree_hal_numerical_type_t = 0x22;
pub const IREE_HAL_NUMERICAL_TYPE_FLOAT_COMPLEX: iree_hal_numerical_type_t = 0x23;

/// `#define iree_hal_make_element_type(numerical_type, bit_count)`
#[inline]
pub const fn iree_hal_make_element_type(
    numerical_type: iree_hal_numerical_type_t,
    bit_count: u8,
) -> iree_hal_element_type_t {
    ((numerical_type as u32) << 24) | bit_count as u32
}

/// `#define iree_hal_element_bit_count(element_type)`
#[inline]
pub const fn iree_hal_element_bit_count(element_type: iree_hal_element_type_t) -> u32 {
    element_type & 0xFF
}

/// `#define iree_hal_element_numerical_type(element_type)`
#[inline]
pub const fn iree_hal_element_numerical_type(
    element_type: iree_hal_element_type_t,
) -> iree_hal_numerical_type_t {
    (element_type >> 24) as u8
}

use iree_hal_make_element_type as et;
pub const IREE_HAL_ELEMENT_TYPE_NONE: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_UNKNOWN, 0);
pub const IREE_HAL_ELEMENT_TYPE_BOOL_8: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_BOOLEAN, 8);
pub const IREE_HAL_ELEMENT_TYPE_INT_8: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER, 8);
pub const IREE_HAL_ELEMENT_TYPE_SINT_8: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED, 8);
pub const IREE_HAL_ELEMENT_TYPE_UINT_8: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED, 8);
pub const IREE_HAL_ELEMENT_TYPE_INT_16: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER, 16);
pub const IREE_HAL_ELEMENT_TYPE_SINT_16: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED, 16);
pub const IREE_HAL_ELEMENT_TYPE_UINT_16: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED, 16);
pub const IREE_HAL_ELEMENT_TYPE_INT_32: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER, 32);
pub const IREE_HAL_ELEMENT_TYPE_SINT_32: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED, 32);
pub const IREE_HAL_ELEMENT_TYPE_UINT_32: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED, 32);
pub const IREE_HAL_ELEMENT_TYPE_INT_64: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER, 64);
pub const IREE_HAL_ELEMENT_TYPE_SINT_64: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_SIGNED, 64);
pub const IREE_HAL_ELEMENT_TYPE_UINT_64: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_INTEGER_UNSIGNED, 64);
pub const IREE_HAL_ELEMENT_TYPE_FLOAT_16: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE, 16);
pub const IREE_HAL_ELEMENT_TYPE_FLOAT_32: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE, 32);
pub const IREE_HAL_ELEMENT_TYPE_FLOAT_64: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_FLOAT_IEEE, 64);
pub const IREE_HAL_ELEMENT_TYPE_BFLOAT_16: iree_hal_element_type_t =
    et(IREE_HAL_NUMERICAL_TYPE_FLOAT_BRAIN, 16);

pub const IREE_HAL_ENCODING_TYPE_OPAQUE: iree_hal_encoding_type_t = 0;
pub const IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR: iree_hal_encoding_type_t = 1;

pub type iree_hal_buffer_usage_t = u32;
pub const IREE_HAL_BUFFER_USAGE_NONE: iree_hal_buffer_usage_t = 0;
pub const IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE: iree_hal_buffer_usage_t = 1 << 0;
pub const IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET: iree_hal_buffer_usage_t = 1 << 1;
pub const IREE_HAL_BUFFER_USAGE_TRANSFER: iree_hal_buffer_usage_t =
    IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
pub const IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ: iree_hal_buffer_usage_t = 1 << 10;
pub const IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE: iree_hal_buffer_usage_t = 1 << 11;
pub const IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE: iree_hal_buffer_usage_t =
    IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_READ | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE_WRITE;
pub const IREE_HAL_BUFFER_USAGE_DEFAULT: iree_hal_buffer_usage_t =
    IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

pub type iree_hal_memory_access_t = u16;
pub const IREE_HAL_MEMORY_ACCESS_NONE: iree_hal_memory_access_t = 0;
pub const IREE_HAL_MEMORY_ACCESS_READ: iree_hal_memory_access_t = 1 << 0;
pub const IREE_HAL_MEMORY_ACCESS_WRITE: iree_hal_memory_access_t = 1 << 1;
pub const IREE_HAL_MEMORY_ACCESS_DISCARD: iree_hal_memory_access_t = 1 << 2;
pub const IREE_HAL_MEMORY_ACCESS_ALL: iree_hal_memory_access_t =
    IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE | IREE_HAL_MEMORY_ACCESS_DISCARD;

pub type iree_hal_memory_type_t = u32;
pub const IREE_HAL_MEMORY_TYPE_NONE: iree_hal_memory_type_t = 0;
pub const IREE_HAL_MEMORY_TYPE_OPTIMAL: iree_hal_memory_type_t = 1 << 0;
pub const IREE_HAL_MEMORY_TYPE_HOST_VISIBLE: iree_hal_memory_type_t = 1 << 1;
pub const IREE_HAL_MEMORY_TYPE_HOST_COHERENT: iree_hal_memory_type_t = 1 << 2;
pub const IREE_HAL_MEMORY_TYPE_HOST_CACHED: iree_hal_memory_type_t = 1 << 3;
pub const IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE: iree_hal_memory_type_t = 1 << 4;
pub const IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL: iree_hal_memory_type_t =
    IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE | (1 << 5);
pub const IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE: iree_hal_memory_type_t =
    IREE_HAL_MEMORY_TYPE_OPTIMAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

pub type iree_hal_queue_affinity_t = u64;
pub const IREE_HAL_QUEUE_AFFINITY_ANY: iree_hal_queue_affinity_t = u64::MAX;

pub type iree_hal_transfer_buffer_flags_t = u32;
pub const IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT: iree_hal_transfer_buffer_flags_t = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct iree_hal_buffer_params_t {
    pub usage: iree_hal_buffer_usage_t,
    pub access: iree_hal_memory_access_t,
    pub type_: iree_hal_memory_type_t,
    pub queue_affinity: iree_hal_queue_affinity_t,
    pub min_alignment: iree_device_size_t,
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

unsafe extern "C" {
    // --- base ---
    pub fn iree_allocator_libc_ctl(
        self_: *mut c_void,
        command: iree_allocator_command_t,
        params: *const c_void,
        inout_ptr: *mut *mut c_void,
    ) -> iree_status_t;
    pub fn iree_allocator_malloc(
        allocator: iree_allocator_t,
        byte_length: iree_host_size_t,
        out_ptr: *mut *mut c_void,
    ) -> iree_status_t;
    pub fn iree_allocator_free(allocator: iree_allocator_t, ptr: *mut c_void);

    pub fn iree_status_free(status: iree_status_t);
    pub fn iree_status_ignore(status: iree_status_t) -> iree_status_t;
    pub fn iree_status_code_string(code: iree_status_code_t) -> *const c_char;
    pub fn iree_status_to_string(
        status: iree_status_t,
        allocator: *const iree_allocator_t,
        out_buffer: *mut *mut c_char,
        out_buffer_length: *mut iree_host_size_t,
    ) -> bool;

    // --- runtime: instance ---
    pub fn iree_runtime_instance_options_initialize(
        out_options: *mut iree_runtime_instance_options_t,
    );
    pub fn iree_runtime_instance_options_use_all_available_drivers(
        options: *mut iree_runtime_instance_options_t,
    );
    pub fn iree_runtime_instance_create(
        options: *const iree_runtime_instance_options_t,
        host_allocator: iree_allocator_t,
        out_instance: *mut *mut iree_runtime_instance_t,
    ) -> iree_status_t;
    pub fn iree_runtime_instance_retain(instance: *mut iree_runtime_instance_t);
    pub fn iree_runtime_instance_release(instance: *mut iree_runtime_instance_t);
    pub fn iree_runtime_instance_host_allocator(
        instance: *const iree_runtime_instance_t,
    ) -> iree_allocator_t;
    pub fn iree_runtime_instance_try_create_default_device(
        instance: *mut iree_runtime_instance_t,
        driver_name: iree_string_view_t,
        out_device: *mut *mut iree_hal_device_t,
    ) -> iree_status_t;

    // --- runtime: session ---
    pub fn iree_runtime_session_options_initialize(
        out_options: *mut iree_runtime_session_options_t,
    );
    pub fn iree_runtime_session_create_with_device(
        instance: *mut iree_runtime_instance_t,
        options: *const iree_runtime_session_options_t,
        device: *mut iree_hal_device_t,
        host_allocator: iree_allocator_t,
        out_session: *mut *mut iree_runtime_session_t,
    ) -> iree_status_t;
    pub fn iree_runtime_session_retain(session: *mut iree_runtime_session_t);
    pub fn iree_runtime_session_release(session: *mut iree_runtime_session_t);
    pub fn iree_runtime_session_host_allocator(
        session: *const iree_runtime_session_t,
    ) -> iree_allocator_t;
    pub fn iree_runtime_session_context(
        session: *const iree_runtime_session_t,
    ) -> *mut iree_vm_context_t;
    pub fn iree_runtime_session_device(
        session: *const iree_runtime_session_t,
    ) -> *mut iree_hal_device_t;
    pub fn iree_runtime_session_device_allocator(
        session: *const iree_runtime_session_t,
    ) -> *mut iree_hal_allocator_t;
    pub fn iree_runtime_session_trim(session: *mut iree_runtime_session_t) -> iree_status_t;
    pub fn iree_runtime_session_append_bytecode_module_from_memory(
        session: *mut iree_runtime_session_t,
        flatbuffer_data: iree_const_byte_span_t,
        flatbuffer_allocator: iree_allocator_t,
    ) -> iree_status_t;
    pub fn iree_runtime_session_append_bytecode_module_from_file(
        session: *mut iree_runtime_session_t,
        file_path: *const c_char,
    ) -> iree_status_t;
    pub fn iree_runtime_session_lookup_function(
        session: *const iree_runtime_session_t,
        full_name: iree_string_view_t,
        out_function: *mut iree_vm_function_t,
    ) -> iree_status_t;

    // --- runtime: call ---
    pub fn iree_runtime_call_initialize(
        session: *mut iree_runtime_session_t,
        function: iree_vm_function_t,
        out_call: *mut iree_runtime_call_t,
    ) -> iree_status_t;
    pub fn iree_runtime_call_initialize_by_name(
        session: *mut iree_runtime_session_t,
        full_name: iree_string_view_t,
        out_call: *mut iree_runtime_call_t,
    ) -> iree_status_t;
    pub fn iree_runtime_call_deinitialize(call: *mut iree_runtime_call_t);
    pub fn iree_runtime_call_reset(call: *mut iree_runtime_call_t);
    pub fn iree_runtime_call_inputs(call: *const iree_runtime_call_t) -> *mut iree_vm_list_t;
    pub fn iree_runtime_call_outputs(call: *const iree_runtime_call_t) -> *mut iree_vm_list_t;
    pub fn iree_runtime_call_invoke(
        call: *mut iree_runtime_call_t,
        flags: iree_runtime_call_flags_t,
    ) -> iree_status_t;
    pub fn iree_runtime_call_inputs_push_back_buffer_view(
        call: *mut iree_runtime_call_t,
        buffer_view: *mut iree_hal_buffer_view_t,
    ) -> iree_status_t;
    pub fn iree_runtime_call_outputs_pop_front_buffer_view(
        call: *mut iree_runtime_call_t,
        out_buffer_view: *mut *mut iree_hal_buffer_view_t,
    ) -> iree_status_t;

    // --- vm: reflection ---
    pub fn iree_vm_list_size(list: *const iree_vm_list_t) -> iree_host_size_t;
    pub fn iree_vm_context_module_count(context: *const iree_vm_context_t) -> iree_host_size_t;
    pub fn iree_vm_context_module_at(
        context: *const iree_vm_context_t,
        i: iree_host_size_t,
    ) -> *mut iree_vm_module_t;
    pub fn iree_vm_module_name(module: *const iree_vm_module_t) -> iree_string_view_t;
    pub fn iree_vm_module_signature(module: *const iree_vm_module_t) -> iree_vm_module_signature_t;
    pub fn iree_vm_module_lookup_function_by_ordinal(
        module: *const iree_vm_module_t,
        linkage: iree_vm_function_linkage_t,
        ordinal: iree_host_size_t,
        out_function: *mut iree_vm_function_t,
    ) -> iree_status_t;
    pub fn iree_vm_function_name(function: *const iree_vm_function_t) -> iree_string_view_t;
    pub fn iree_vm_function_signature(
        function: *const iree_vm_function_t,
    ) -> iree_vm_function_signature_t;
    pub fn iree_vm_function_lookup_attr_by_name(
        function: *const iree_vm_function_t,
        key: iree_string_view_t,
    ) -> iree_string_view_t;

    // --- hal: device ---
    pub fn iree_hal_device_retain(device: *mut iree_hal_device_t);
    pub fn iree_hal_device_release(device: *mut iree_hal_device_t);
    pub fn iree_hal_device_id(device: *mut iree_hal_device_t) -> iree_string_view_t;
    pub fn iree_hal_device_allocator(device: *mut iree_hal_device_t) -> *mut iree_hal_allocator_t;
    pub fn iree_hal_device_trim(device: *mut iree_hal_device_t) -> iree_status_t;
    pub fn iree_hal_device_transfer_d2h(
        device: *mut iree_hal_device_t,
        source: *mut iree_hal_buffer_t,
        source_offset: iree_device_size_t,
        target: *mut c_void,
        data_length: iree_device_size_t,
        flags: iree_hal_transfer_buffer_flags_t,
        timeout: iree_timeout_t,
    ) -> iree_status_t;

    // --- hal: buffer view ---
    pub fn iree_hal_buffer_view_allocate_buffer_copy(
        device: *mut iree_hal_device_t,
        allocator: *mut iree_hal_allocator_t,
        shape_rank: iree_host_size_t,
        shape: *const iree_hal_dim_t,
        element_type: iree_hal_element_type_t,
        encoding_type: iree_hal_encoding_type_t,
        buffer_params: iree_hal_buffer_params_t,
        initial_data: iree_const_byte_span_t,
        out_buffer_view: *mut *mut iree_hal_buffer_view_t,
    ) -> iree_status_t;
    pub fn iree_hal_buffer_view_retain(buffer_view: *mut iree_hal_buffer_view_t);
    pub fn iree_hal_buffer_view_release(buffer_view: *mut iree_hal_buffer_view_t);
    pub fn iree_hal_buffer_view_buffer(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> *mut iree_hal_buffer_t;
    pub fn iree_hal_buffer_view_shape_rank(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> iree_host_size_t;
    pub fn iree_hal_buffer_view_shape_dims(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> *const iree_hal_dim_t;
    pub fn iree_hal_buffer_view_shape_dim(
        buffer_view: *const iree_hal_buffer_view_t,
        index: iree_host_size_t,
    ) -> iree_hal_dim_t;
    pub fn iree_hal_buffer_view_element_count(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> iree_device_size_t;
    pub fn iree_hal_buffer_view_element_type(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> iree_hal_element_type_t;
    pub fn iree_hal_buffer_view_encoding_type(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> iree_hal_encoding_type_t;
    pub fn iree_hal_buffer_view_byte_length(
        buffer_view: *const iree_hal_buffer_view_t,
    ) -> iree_device_size_t;
}

// ---------------------------------------------------------------------------
// Layout verification against the real C headers.
// ---------------------------------------------------------------------------

#[cfg(all(test, iree_sys_layout_check))]
mod layout {
    extern crate std;

    use super::*;
    use core::mem::{align_of, offset_of, size_of};
    use std::collections::BTreeMap;
    use std::ffi::CStr;
    use std::string::{String, ToString};
    use std::vec::Vec;

    #[repr(C)]
    struct Entry {
        name: *const c_char,
        value: u64,
    }

    unsafe extern "C" {
        fn iree_sys_layout_entries(out_count: *mut usize) -> *const Entry;
    }

    fn c_entries() -> BTreeMap<String, u64> {
        let mut count = 0usize;
        let ptr = unsafe { iree_sys_layout_entries(&mut count) };
        let entries = unsafe { core::slice::from_raw_parts(ptr, count) };
        entries
            .iter()
            .map(|e| {
                let name = unsafe { CStr::from_ptr(e.name) }
                    .to_str()
                    .unwrap()
                    .to_string();
                (name, e.value)
            })
            .collect()
    }

    macro_rules! size {
        ($v:ident, $c:literal, $t:ty) => {
            $v.push((concat!("size:", $c), size_of::<$t>() as u64));
            $v.push((concat!("align:", $c), align_of::<$t>() as u64));
        };
    }
    macro_rules! off {
        ($v:ident, $c:literal, $t:ty, $rf:ident, $cf:literal) => {
            $v.push((concat!("offset:", $c, ".", $cf), offset_of!($t, $rf) as u64));
        };
    }
    macro_rules! konst {
        ($v:ident, $c:ident) => {
            $v.push((concat!("const:", stringify!($c)), $c as u64));
        };
    }

    fn rust_entries() -> Vec<(&'static str, u64)> {
        let mut v = Vec::new();
        size!(v, "iree_host_size_t", iree_host_size_t);
        size!(v, "iree_device_size_t", iree_device_size_t);
        size!(v, "iree_hal_dim_t", iree_hal_dim_t);
        size!(v, "iree_status_t", iree_status_t);
        size!(v, "iree_allocator_command_t", iree_allocator_command_t);
        size!(v, "iree_timeout_type_t", iree_timeout_type_t);
        size!(v, "iree_vm_function_linkage_t", iree_vm_function_linkage_t);
        size!(v, "iree_hal_element_type_t", iree_hal_element_type_t);
        size!(v, "iree_hal_encoding_type_t", iree_hal_encoding_type_t);
        size!(v, "iree_hal_buffer_usage_t", iree_hal_buffer_usage_t);
        size!(v, "iree_hal_memory_access_t", iree_hal_memory_access_t);
        size!(v, "iree_hal_memory_type_t", iree_hal_memory_type_t);
        size!(v, "iree_hal_queue_affinity_t", iree_hal_queue_affinity_t);
        size!(v, "iree_vm_context_flags_t", iree_vm_context_flags_t);
        size!(
            v,
            "iree_runtime_session_builtins_t",
            iree_runtime_session_builtins_t
        );

        size!(v, "iree_allocator_t", iree_allocator_t);
        off!(v, "iree_allocator_t", iree_allocator_t, self_, "self");
        off!(v, "iree_allocator_t", iree_allocator_t, ctl, "ctl");
        size!(v, "iree_string_view_t", iree_string_view_t);
        off!(v, "iree_string_view_t", iree_string_view_t, data, "data");
        off!(v, "iree_string_view_t", iree_string_view_t, size, "size");
        size!(v, "iree_const_byte_span_t", iree_const_byte_span_t);
        off!(
            v,
            "iree_const_byte_span_t",
            iree_const_byte_span_t,
            data,
            "data"
        );
        off!(
            v,
            "iree_const_byte_span_t",
            iree_const_byte_span_t,
            data_length,
            "data_length"
        );
        size!(v, "iree_timeout_t", iree_timeout_t);
        off!(v, "iree_timeout_t", iree_timeout_t, type_, "type");
        off!(v, "iree_timeout_t", iree_timeout_t, nanos, "nanos");

        size!(
            v,
            "iree_runtime_instance_options_t",
            iree_runtime_instance_options_t
        );
        off!(
            v,
            "iree_runtime_instance_options_t",
            iree_runtime_instance_options_t,
            driver_registry,
            "driver_registry"
        );
        size!(
            v,
            "iree_runtime_session_options_t",
            iree_runtime_session_options_t
        );
        off!(
            v,
            "iree_runtime_session_options_t",
            iree_runtime_session_options_t,
            context_flags,
            "context_flags"
        );
        off!(
            v,
            "iree_runtime_session_options_t",
            iree_runtime_session_options_t,
            builtin_modules,
            "builtin_modules"
        );
        size!(v, "iree_runtime_call_t", iree_runtime_call_t);
        off!(
            v,
            "iree_runtime_call_t",
            iree_runtime_call_t,
            session,
            "session"
        );
        off!(
            v,
            "iree_runtime_call_t",
            iree_runtime_call_t,
            function,
            "function"
        );
        off!(
            v,
            "iree_runtime_call_t",
            iree_runtime_call_t,
            inputs,
            "inputs"
        );
        off!(
            v,
            "iree_runtime_call_t",
            iree_runtime_call_t,
            outputs,
            "outputs"
        );

        size!(v, "iree_vm_function_t", iree_vm_function_t);
        off!(
            v,
            "iree_vm_function_t",
            iree_vm_function_t,
            module,
            "module"
        );
        off!(
            v,
            "iree_vm_function_t",
            iree_vm_function_t,
            linkage,
            "linkage"
        );
        off!(
            v,
            "iree_vm_function_t",
            iree_vm_function_t,
            ordinal,
            "ordinal"
        );
        size!(
            v,
            "iree_vm_function_signature_t",
            iree_vm_function_signature_t
        );
        off!(
            v,
            "iree_vm_function_signature_t",
            iree_vm_function_signature_t,
            calling_convention,
            "calling_convention"
        );
        size!(v, "iree_vm_module_signature_t", iree_vm_module_signature_t);
        off!(
            v,
            "iree_vm_module_signature_t",
            iree_vm_module_signature_t,
            version,
            "version"
        );
        off!(
            v,
            "iree_vm_module_signature_t",
            iree_vm_module_signature_t,
            attr_count,
            "attr_count"
        );
        off!(
            v,
            "iree_vm_module_signature_t",
            iree_vm_module_signature_t,
            import_function_count,
            "import_function_count"
        );
        off!(
            v,
            "iree_vm_module_signature_t",
            iree_vm_module_signature_t,
            export_function_count,
            "export_function_count"
        );
        off!(
            v,
            "iree_vm_module_signature_t",
            iree_vm_module_signature_t,
            internal_function_count,
            "internal_function_count"
        );

        size!(v, "iree_hal_buffer_params_t", iree_hal_buffer_params_t);
        off!(
            v,
            "iree_hal_buffer_params_t",
            iree_hal_buffer_params_t,
            usage,
            "usage"
        );
        off!(
            v,
            "iree_hal_buffer_params_t",
            iree_hal_buffer_params_t,
            access,
            "access"
        );
        off!(
            v,
            "iree_hal_buffer_params_t",
            iree_hal_buffer_params_t,
            type_,
            "type"
        );
        off!(
            v,
            "iree_hal_buffer_params_t",
            iree_hal_buffer_params_t,
            queue_affinity,
            "queue_affinity"
        );
        off!(
            v,
            "iree_hal_buffer_params_t",
            iree_hal_buffer_params_t,
            min_alignment,
            "min_alignment"
        );

        konst!(v, IREE_STATUS_OK);
        konst!(v, IREE_STATUS_CANCELLED);
        konst!(v, IREE_STATUS_UNKNOWN);
        konst!(v, IREE_STATUS_INVALID_ARGUMENT);
        konst!(v, IREE_STATUS_DEADLINE_EXCEEDED);
        konst!(v, IREE_STATUS_NOT_FOUND);
        konst!(v, IREE_STATUS_ALREADY_EXISTS);
        konst!(v, IREE_STATUS_PERMISSION_DENIED);
        konst!(v, IREE_STATUS_RESOURCE_EXHAUSTED);
        konst!(v, IREE_STATUS_FAILED_PRECONDITION);
        konst!(v, IREE_STATUS_ABORTED);
        konst!(v, IREE_STATUS_OUT_OF_RANGE);
        konst!(v, IREE_STATUS_UNIMPLEMENTED);
        konst!(v, IREE_STATUS_INTERNAL);
        konst!(v, IREE_STATUS_UNAVAILABLE);
        konst!(v, IREE_STATUS_DATA_LOSS);
        konst!(v, IREE_STATUS_UNAUTHENTICATED);
        konst!(v, IREE_STATUS_DEFERRED);
        konst!(v, IREE_STATUS_INCOMPATIBLE);
        konst!(v, IREE_STATUS_CODE_MASK);
        konst!(v, IREE_ALLOCATOR_COMMAND_MALLOC);
        konst!(v, IREE_ALLOCATOR_COMMAND_CALLOC);
        konst!(v, IREE_ALLOCATOR_COMMAND_REALLOC);
        konst!(v, IREE_ALLOCATOR_COMMAND_FREE);
        konst!(v, IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT);
        konst!(v, IREE_TIMEOUT_ABSOLUTE);
        konst!(v, IREE_TIMEOUT_RELATIVE);
        konst!(v, IREE_TIME_INFINITE_FUTURE);
        konst!(v, IREE_VM_FUNCTION_LINKAGE_INTERNAL);
        konst!(v, IREE_VM_FUNCTION_LINKAGE_IMPORT);
        konst!(v, IREE_VM_FUNCTION_LINKAGE_EXPORT);
        konst!(v, IREE_VM_CONTEXT_FLAG_NONE);
        konst!(v, IREE_RUNTIME_SESSION_BUILTIN_ALL);
        konst!(v, IREE_RUNTIME_CALL_FLAG_RESERVED);
        konst!(v, IREE_HAL_ENCODING_TYPE_OPAQUE);
        konst!(v, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR);
        konst!(v, IREE_HAL_BUFFER_USAGE_DEFAULT);
        konst!(v, IREE_HAL_MEMORY_ACCESS_ALL);
        konst!(v, IREE_HAL_MEMORY_TYPE_OPTIMAL);
        konst!(v, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL);
        konst!(v, IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE);
        konst!(v, IREE_HAL_QUEUE_AFFINITY_ANY);
        konst!(v, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT);
        konst!(v, IREE_HAL_ELEMENT_TYPE_NONE);
        konst!(v, IREE_HAL_ELEMENT_TYPE_BOOL_8);
        konst!(v, IREE_HAL_ELEMENT_TYPE_INT_8);
        konst!(v, IREE_HAL_ELEMENT_TYPE_SINT_8);
        konst!(v, IREE_HAL_ELEMENT_TYPE_UINT_8);
        konst!(v, IREE_HAL_ELEMENT_TYPE_INT_16);
        konst!(v, IREE_HAL_ELEMENT_TYPE_SINT_16);
        konst!(v, IREE_HAL_ELEMENT_TYPE_UINT_16);
        konst!(v, IREE_HAL_ELEMENT_TYPE_INT_32);
        konst!(v, IREE_HAL_ELEMENT_TYPE_SINT_32);
        konst!(v, IREE_HAL_ELEMENT_TYPE_UINT_32);
        konst!(v, IREE_HAL_ELEMENT_TYPE_INT_64);
        konst!(v, IREE_HAL_ELEMENT_TYPE_SINT_64);
        konst!(v, IREE_HAL_ELEMENT_TYPE_UINT_64);
        konst!(v, IREE_HAL_ELEMENT_TYPE_FLOAT_16);
        konst!(v, IREE_HAL_ELEMENT_TYPE_FLOAT_32);
        konst!(v, IREE_HAL_ELEMENT_TYPE_FLOAT_64);
        konst!(v, IREE_HAL_ELEMENT_TYPE_BFLOAT_16);
        v
    }

    #[test]
    fn rust_declarations_match_c_headers() {
        let c = c_entries();
        let rust = rust_entries();
        let mut mismatches = Vec::new();
        for (name, rust_value) in &rust {
            match c.get(*name) {
                Some(c_value) if c_value == rust_value => {}
                Some(c_value) => {
                    mismatches.push(std::format!("{name}: rust={rust_value} c={c_value}"))
                }
                None => mismatches.push(std::format!("{name}: missing from csrc/layout_check.c")),
            }
        }
        for name in c.keys() {
            if !rust.iter().any(|(n, _)| n == name) {
                mismatches.push(std::format!("{name}: checked in C but not in Rust"));
            }
        }
        assert!(
            mismatches.is_empty(),
            "FFI layout mismatches:\n{}",
            mismatches.join("\n")
        );
    }

    #[test]
    fn status_code_helpers() {
        assert!(iree_status_is_ok(core::ptr::null_mut()));
        let fake = IREE_STATUS_NOT_FOUND as usize as iree_status_t;
        assert_eq!(iree_status_code(fake), IREE_STATUS_NOT_FOUND);
    }
}
