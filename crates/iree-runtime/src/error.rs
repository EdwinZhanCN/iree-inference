use std::ffi::CStr;
use std::fmt;

use iree_sys as sys;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// IREE status code (`iree_status_code_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StatusCode {
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
    Deferred,
    Incompatible,
    /// A code not known to this crate version.
    Other(u32),
}

impl StatusCode {
    fn from_raw(code: sys::iree_status_code_t) -> Self {
        use StatusCode::*;
        match code {
            sys::IREE_STATUS_CANCELLED => Cancelled,
            sys::IREE_STATUS_UNKNOWN => Unknown,
            sys::IREE_STATUS_INVALID_ARGUMENT => InvalidArgument,
            sys::IREE_STATUS_DEADLINE_EXCEEDED => DeadlineExceeded,
            sys::IREE_STATUS_NOT_FOUND => NotFound,
            sys::IREE_STATUS_ALREADY_EXISTS => AlreadyExists,
            sys::IREE_STATUS_PERMISSION_DENIED => PermissionDenied,
            sys::IREE_STATUS_RESOURCE_EXHAUSTED => ResourceExhausted,
            sys::IREE_STATUS_FAILED_PRECONDITION => FailedPrecondition,
            sys::IREE_STATUS_ABORTED => Aborted,
            sys::IREE_STATUS_OUT_OF_RANGE => OutOfRange,
            sys::IREE_STATUS_UNIMPLEMENTED => Unimplemented,
            sys::IREE_STATUS_INTERNAL => Internal,
            sys::IREE_STATUS_UNAVAILABLE => Unavailable,
            sys::IREE_STATUS_DATA_LOSS => DataLoss,
            sys::IREE_STATUS_UNAUTHENTICATED => Unauthenticated,
            sys::IREE_STATUS_DEFERRED => Deferred,
            sys::IREE_STATUS_INCOMPATIBLE => Incompatible,
            other => Other(other),
        }
    }
}

/// An error returned by the IREE runtime, or by argument validation in this crate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    code: StatusCode,
    message: String,
}

impl Error {
    /// A validation error raised on the Rust side before calling into IREE.
    pub(crate) fn invalid_argument(message: impl Into<String>) -> Self {
        Self {
            code: StatusCode::InvalidArgument,
            message: message.into(),
        }
    }

    /// Converts a non-OK `iree_status_t` into an `Error`, taking ownership of
    /// (and freeing) the status.
    ///
    /// # Safety
    /// `status` must be a non-OK status that has not been freed and is not used
    /// afterwards.
    pub(crate) unsafe fn from_status(status: sys::iree_status_t) -> Self {
        debug_assert!(!sys::iree_status_is_ok(status));
        let code = StatusCode::from_raw(sys::iree_status_code(status));
        let allocator = sys::iree_allocator_system();
        let mut buffer: *mut std::ffi::c_char = std::ptr::null_mut();
        let mut length: usize = 0;
        // SAFETY: `status` is live; `allocator` outlives the call; the returned
        // buffer is owned by us and freed with the same allocator.
        let message = unsafe {
            if sys::iree_status_to_string(status, &allocator, &mut buffer, &mut length)
                && !buffer.is_null()
            {
                let bytes = std::slice::from_raw_parts(buffer.cast::<u8>(), length);
                let message = String::from_utf8_lossy(bytes).into_owned();
                sys::iree_allocator_free(allocator, buffer.cast());
                message
            } else {
                let name = sys::iree_status_code_string(sys::iree_status_code(status));
                if name.is_null() {
                    format!("IREE status code {}", sys::iree_status_code(status))
                } else {
                    CStr::from_ptr(name).to_string_lossy().into_owned()
                }
            }
        };
        // SAFETY: we own `status`; it is not used after this.
        unsafe { sys::iree_status_free(status) };
        Self { code, message }
    }

    pub fn code(&self) -> StatusCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IREE error ({:?}): {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Converts an `iree_status_t` into a `Result`, consuming the status.
///
/// # Safety
/// `status` must be a status returned by an IREE API that has not been freed.
#[inline]
pub(crate) unsafe fn check(status: sys::iree_status_t) -> Result<()> {
    if sys::iree_status_is_ok(status) {
        Ok(())
    } else {
        Err(unsafe { Error::from_status(status) })
    }
}

/// Copies an IREE string view into an owned `String` (lossy UTF-8).
///
/// # Safety
/// The view must point at valid memory for its size (or be empty/null).
pub(crate) unsafe fn string_from_view(view: sys::iree_string_view_t) -> String {
    String::from_utf8_lossy(unsafe { view.as_bytes() }).into_owned()
}
