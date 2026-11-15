use std::cell::Cell;
use std::ffi::CString;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::NonNull;

use iree_sys as sys;

use crate::call::Call;
use crate::error::{Error, Result, check, string_from_view};
use crate::instance::Device;

/// A set of loaded VM modules bound to one [`Device`] — think "process".
///
/// IREE sessions are *thread-compatible*: only one thread may use a session (or
/// anything derived from it, such as an in-flight [`Call`]) at a time. This is
/// encoded as `Send + !Sync`: a session can be moved to a worker thread, but
/// `&Session` cannot be shared across threads. Different sessions may run
/// concurrently, even on the same device.
pub struct Session {
    raw: NonNull<sys::iree_runtime_session_t>,
    device: Device,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: a session may be used from any thread as long as access is not
// concurrent; `!Sync` (via the `Cell` marker) prevents shared cross-thread use.
unsafe impl Send for Session {}

impl Session {
    /// Creates a session on `device` with all built-in runtime modules (HAL, ...).
    pub fn new(device: &Device) -> Result<Self> {
        let mut options = sys::iree_runtime_session_options_t {
            context_flags: sys::IREE_VM_CONTEXT_FLAG_NONE,
            builtin_modules: sys::IREE_RUNTIME_SESSION_BUILTIN_ALL,
        };
        let instance = device.instance();
        let mut raw = std::ptr::null_mut();
        // SAFETY: all pointers are live for the duration of the call; the session
        // retains the instance and device internally; `raw` receives an owned ref.
        unsafe {
            sys::iree_runtime_session_options_initialize(&mut options);
            check(sys::iree_runtime_session_create_with_device(
                instance.as_ptr(),
                &options,
                device.as_ptr(),
                sys::iree_runtime_instance_host_allocator(instance.as_ptr()),
                &mut raw,
            ))?;
        }
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::invalid_argument("session creation returned NULL"))?;
        Ok(Self {
            raw,
            device: device.clone(),
            _not_sync: PhantomData,
        })
    }

    /// Creates a session and loads one bytecode module (`.vmfb`) from a file.
    pub fn from_file(device: &Device, vmfb: impl AsRef<Path>) -> Result<Self> {
        let mut session = Self::new(device)?;
        session.load_module_file(vmfb)?;
        Ok(session)
    }

    /// Appends a `.vmfb` bytecode module loaded from disk (IREE reads/maps the
    /// file itself and keeps the contents alive for the session's lifetime).
    ///
    /// Modules must be appended before the first call (the VM context freezes).
    pub fn load_module_file(&mut self, vmfb: impl AsRef<Path>) -> Result<()> {
        let path = vmfb.as_ref();
        let c_path = path_to_cstring(path)?;
        // SAFETY: `c_path` is a valid NUL-terminated string for the call.
        unsafe {
            check(sys::iree_runtime_session_append_bytecode_module_from_file(
                self.raw.as_ptr(),
                c_path.as_ptr(),
            ))
        }
    }

    /// Appends a `.vmfb` bytecode module from memory.
    ///
    /// The bytes are copied into a 64-byte-aligned allocation
    /// (`IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT`; rodata such as weights and embedded
    /// executables is aligned relative to the archive start). The runtime keeps
    /// the copy alive for the session's lifetime and frees it itself, so `vmfb`
    /// can be dropped right after this returns. Prefer
    /// [`load_module_file`](Self::load_module_file) for large models (mmap).
    pub fn load_module_bytes(&mut self, vmfb: &[u8]) -> Result<()> {
        let (data, allocator) = aligned_copy(vmfb)?;
        // SAFETY: `data` + `allocator` form an owned allocation that IREE frees
        // through `aligned_free_ctl`, even if loading fails.
        unsafe {
            check(
                sys::iree_runtime_session_append_bytecode_module_from_memory(
                    self.raw.as_ptr(),
                    data,
                    allocator,
                ),
            )
        }
    }

    /// Looks up an exported function by fully-qualified name, e.g. `"module.main"`
    /// (`<mlir module name>.<func name>`; the default module name is `module`).
    pub fn lookup_function(&self, full_name: &str) -> Result<Function<'_>> {
        let mut raw = sys::iree_vm_function_t::null();
        // SAFETY: `self.raw` is live; `raw` is a valid out-pointer.
        unsafe {
            check(sys::iree_runtime_session_lookup_function(
                self.raw.as_ptr(),
                sys::iree_string_view_t::new(full_name),
                &mut raw,
            ))?;
        }
        if raw.module.is_null() {
            return Err(Error::invalid_argument(format!(
                "function `{full_name}` not found"
            )));
        }
        Ok(Function {
            raw,
            full_name: full_name.to_owned(),
            session: self,
        })
    }

    /// Convenience: `lookup_function(name)?.call()`.
    pub fn call(&self, full_name: &str) -> Result<Call<'_>> {
        self.lookup_function(full_name)?.call()
    }

    /// Lists exported functions of every module in the session (including
    /// runtime built-ins such as `hal.*`), with their reflection metadata.
    pub fn exported_functions(&self) -> Vec<FunctionInfo> {
        let mut out = Vec::new();
        // SAFETY: the context and its modules are owned by the session and stay
        // valid while `&self` is borrowed; all returned views are copied.
        unsafe {
            let context = sys::iree_runtime_session_context(self.raw.as_ptr());
            if context.is_null() {
                return out;
            }
            for i in 0..sys::iree_vm_context_module_count(context) {
                let module = sys::iree_vm_context_module_at(context, i);
                if module.is_null() {
                    continue;
                }
                let module_name = string_from_view(sys::iree_vm_module_name(module));
                let signature = sys::iree_vm_module_signature(module);
                for ordinal in 0..signature.export_function_count {
                    let mut function = sys::iree_vm_function_t::null();
                    let status = sys::iree_vm_module_lookup_function_by_ordinal(
                        module,
                        sys::IREE_VM_FUNCTION_LINKAGE_EXPORT,
                        ordinal,
                        &mut function,
                    );
                    if check(status).is_err() || function.module.is_null() {
                        continue;
                    }
                    let name = string_from_view(sys::iree_vm_function_name(&function));
                    out.push(FunctionInfo {
                        full_name: format!("{module_name}.{name}"),
                        calling_convention: calling_convention(&function),
                        declaration: attr(&function, ABI_DECLARATION_ATTR),
                    });
                }
            }
        }
        out
    }

    /// Trims transient/cached resources of the session (and its device pools).
    /// Takes `&mut self` so no [`Call`] or [`Function`] borrowed from this session
    /// can be in use.
    pub fn trim(&mut self) -> Result<()> {
        // SAFETY: exclusive access to the session.
        unsafe { check(sys::iree_runtime_session_trim(self.raw.as_ptr())) }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::iree_runtime_session_t {
        self.raw.as_ptr()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: we own one reference; no borrowed Call/Function can outlive us.
        unsafe { sys::iree_runtime_session_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("raw", &self.raw)
            .field("device", &self.device)
            .finish()
    }
}

/// Reflection attribute the IREE compiler attaches to every public entry point,
/// e.g. `sync func @main(%input0: tensor<1x3x224x224xf32>) -> (%output0: tensor<1x768xf32>)`.
pub const ABI_DECLARATION_ATTR: &str = "iree.abi.declaration";

/// An exported function handle; valid while its [`Session`] is borrowed.
#[derive(Clone)]
pub struct Function<'s> {
    raw: sys::iree_vm_function_t,
    full_name: String,
    session: &'s Session,
}

impl<'s> Function<'s> {
    pub fn full_name(&self) -> &str {
        &self.full_name
    }

    /// VM calling convention, e.g. `0rr_r` (2 refs in, 1 ref out).
    pub fn calling_convention(&self) -> String {
        // SAFETY: the function's module is retained by the borrowed session.
        unsafe { calling_convention(&self.raw) }
    }

    /// Value of a reflection attribute, if present.
    pub fn attr(&self, key: &str) -> Option<String> {
        // SAFETY: as above.
        unsafe { attr(&self.raw, key) }
    }

    /// The `iree.abi.declaration` string (input/output tensor types), if present.
    /// Use it to assert at load time that a vmfb's I/O dtypes and shapes match
    /// what the caller will feed.
    pub fn declaration(&self) -> Option<String> {
        self.attr(ABI_DECLARATION_ATTR)
    }

    /// Number of (arguments, results) according to the calling convention.
    /// Returns `None` for variadic or unparseable conventions.
    pub fn arity(&self) -> Option<(usize, usize)> {
        parse_arity(&self.calling_convention())
    }

    /// Creates a reusable call builder for this function.
    pub fn call(&self) -> Result<Call<'s>> {
        Call::new(self.session, self.raw)
    }
}

impl std::fmt::Debug for Function<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Function")
            .field("full_name", &self.full_name)
            .field("calling_convention", &self.calling_convention())
            .finish()
    }
}

/// Owned description of an exported function (see [`Session::exported_functions`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionInfo {
    pub full_name: String,
    pub calling_convention: String,
    pub declaration: Option<String>,
}

unsafe fn calling_convention(function: &sys::iree_vm_function_t) -> String {
    unsafe {
        let signature = sys::iree_vm_function_signature(function);
        string_from_view(signature.calling_convention)
    }
}

unsafe fn attr(function: &sys::iree_vm_function_t, key: &str) -> Option<String> {
    unsafe {
        let view =
            sys::iree_vm_function_lookup_attr_by_name(function, sys::iree_string_view_t::new(key));
        if view.data.is_null() {
            None
        } else {
            Some(string_from_view(view))
        }
    }
}

/// Parses `0<args>_<results>` where each char is one of `i I f F r` (no variadics).
pub(crate) fn parse_arity(cconv: &str) -> Option<(usize, usize)> {
    let body = cconv.strip_prefix('0')?;
    if body.contains(['C', 'D']) {
        return None;
    }
    let (args, results) = body.split_once('_').unwrap_or((body, ""));
    let valid = |s: &str| {
        s.chars()
            .all(|c| matches!(c, 'i' | 'I' | 'f' | 'F' | 'r' | 'v'))
    };
    if !valid(args) || !valid(results) {
        return None;
    }
    let count = |s: &str| s.chars().filter(|&c| c != 'v').count();
    Some((count(args), count(results)))
}

const MODULE_ALIGN: usize = sys::IREE_VM_ARCHIVE_SEGMENT_ALIGNMENT;

/// Copies `bytes` into a Rust allocation aligned to [`MODULE_ALIGN`] and returns
/// it with an `iree_allocator_t` whose ctl frees exactly that allocation. The
/// allocation size is carried in the allocator's `self` pointer, so no header is
/// needed.
fn aligned_copy(bytes: &[u8]) -> Result<(sys::iree_const_byte_span_t, sys::iree_allocator_t)> {
    if bytes.is_empty() {
        return Err(Error::invalid_argument("empty vmfb"));
    }
    let layout = std::alloc::Layout::from_size_align(bytes.len(), MODULE_ALIGN)
        .map_err(|_| Error::invalid_argument("vmfb too large"))?;
    // SAFETY: non-zero size layout.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: `ptr` is valid for `bytes.len()` writes and does not overlap `bytes`.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    let span = sys::iree_const_byte_span_t {
        data: ptr,
        data_length: bytes.len(),
    };
    let allocator = sys::iree_allocator_t {
        self_: bytes.len() as *mut std::ffi::c_void,
        ctl: Some(aligned_free_ctl),
    };
    Ok((span, allocator))
}

/// Allocator ctl that only supports FREE of a block made by [`aligned_copy`].
/// Never panics (it is called from C).
unsafe extern "C" fn aligned_free_ctl(
    self_: *mut std::ffi::c_void,
    command: sys::iree_allocator_command_t,
    _params: *const std::ffi::c_void,
    inout_ptr: *mut *mut std::ffi::c_void,
) -> sys::iree_status_t {
    if command != sys::IREE_ALLOCATOR_COMMAND_FREE {
        return sys::iree_status_from_code(sys::IREE_STATUS_UNIMPLEMENTED);
    }
    if inout_ptr.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: IREE passes back the exact pointer from `aligned_copy`; `self_`
    // carries its size, and the layout was valid when allocated.
    unsafe {
        let ptr = *inout_ptr;
        if !ptr.is_null() {
            let layout =
                std::alloc::Layout::from_size_align_unchecked(self_ as usize, MODULE_ALIGN);
            std::alloc::dealloc(ptr.cast(), layout);
            *inout_ptr = std::ptr::null_mut();
        }
    }
    std::ptr::null_mut()
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    let bytes = {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().to_vec()
    };
    #[cfg(not(unix))]
    let bytes = path
        .to_str()
        .ok_or_else(|| Error::invalid_argument(format!("non-UTF-8 path: {}", path.display())))?
        .as_bytes()
        .to_vec();
    CString::new(bytes)
        .map_err(|_| Error::invalid_argument(format!("path contains NUL: {}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::{MODULE_ALIGN, aligned_copy, parse_arity};
    use iree_sys as sys;

    #[test]
    fn aligned_copy_roundtrip_and_free() {
        let data: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        let (span, allocator) = aligned_copy(&data).unwrap();
        assert_eq!(span.data as usize % MODULE_ALIGN, 0);
        assert_eq!(span.data_length, data.len());
        let copied = unsafe { std::slice::from_raw_parts(span.data, span.data_length) };
        assert_eq!(copied, &data[..]);

        // Free exactly the way iree_allocator_free does.
        let ctl = allocator.ctl.unwrap();
        let mut ptr = span.data as *mut std::ffi::c_void;
        let status = unsafe {
            ctl(
                allocator.self_,
                sys::IREE_ALLOCATOR_COMMAND_FREE,
                std::ptr::null(),
                &mut ptr,
            )
        };
        assert!(sys::iree_status_is_ok(status));
        assert!(ptr.is_null());

        // Other commands are refused with a code-only status (no heap payload).
        let mut dummy = std::ptr::null_mut();
        let status = unsafe {
            ctl(
                allocator.self_,
                sys::IREE_ALLOCATOR_COMMAND_MALLOC,
                std::ptr::null(),
                &mut dummy,
            )
        };
        assert_eq!(
            sys::iree_status_code(status),
            sys::IREE_STATUS_UNIMPLEMENTED
        );

        assert!(aligned_copy(&[]).is_err());
    }

    #[test]
    fn arity() {
        assert_eq!(parse_arity("0rr_r"), Some((2, 1)));
        assert_eq!(parse_arity("0r_rrrrrrrrr"), Some((1, 9)));
        assert_eq!(parse_arity("0v_v"), Some((0, 0)));
        assert_eq!(parse_arity("0"), Some((0, 0)));
        assert_eq!(parse_arity("0iCiD_r"), None);
        assert_eq!(parse_arity("1r_r"), None);
    }
}
