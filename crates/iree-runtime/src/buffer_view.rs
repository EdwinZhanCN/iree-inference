use std::mem::size_of;
use std::ptr::NonNull;

use iree_sys as sys;

use crate::element::{Element, ElementType};
use crate::error::{Error, Result, check};
use crate::instance::Device;

/// A dense, row-major tensor in device memory (`iree_hal_buffer_view_t`):
/// buffer + shape + element type.
///
/// Created from host data with [`BufferView::from_slice`] / [`BufferView::from_bytes`]
/// (the data is copied, so the host slice can be dropped immediately), or returned
/// from [`crate::Call`] outputs. Read back with [`BufferView::to_vec`].
///
/// `Send` but not `Sync`: buffer views are ref-counted with atomic retain/release,
/// but outputs are resources derived from a (thread-compatible) session, so shared
/// concurrent access is not offered.
pub struct BufferView {
    raw: NonNull<sys::iree_hal_buffer_view_t>,
    device: Device,
}

// SAFETY: atomic ref-counting; the view is only accessed through `&self`/`&mut self`
// on one thread at a time (no `Sync`).
unsafe impl Send for BufferView {}

impl BufferView {
    /// Copies `data` into a new device-local buffer with the given `shape`.
    ///
    /// The element type is taken from `T` (e.g. `f32` → f32, `i64` → si64; signed
    /// integers match signless `iN` function arguments at the ABI boundary).
    pub fn from_slice<T: Element>(device: &Device, shape: &[usize], data: &[T]) -> Result<Self> {
        let numel = element_count(shape)?;
        if numel != data.len() {
            return Err(Error::invalid_argument(format!(
                "shape {shape:?} has {numel} elements but data has {}",
                data.len()
            )));
        }
        // SAFETY: `T: Element` is POD, so viewing it as bytes is sound.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data))
        };
        // SAFETY: byte length validated above.
        unsafe { Self::allocate_copy(device, shape, T::ELEMENT_TYPE, bytes) }
    }

    /// Copies raw little-endian `bytes` into a new buffer of `element_type`.
    /// Use this for types without a Rust [`Element`] impl (e.g. f16 without the
    /// `half` feature).
    pub fn from_bytes(
        device: &Device,
        shape: &[usize],
        element_type: ElementType,
        bytes: &[u8],
    ) -> Result<Self> {
        let elem_size = element_type.byte_size().ok_or_else(|| {
            Error::invalid_argument(format!("element type {element_type} is not byte-aligned"))
        })?;
        let numel = element_count(shape)?;
        let expected = numel
            .checked_mul(elem_size)
            .ok_or_else(|| Error::invalid_argument("tensor byte size overflows usize"))?;
        if expected != bytes.len() {
            return Err(Error::invalid_argument(format!(
                "shape {shape:?} x {element_type} needs {expected} bytes but got {}",
                bytes.len()
            )));
        }
        // SAFETY: byte length validated above.
        unsafe { Self::allocate_copy(device, shape, element_type, bytes) }
    }

    /// # Safety
    /// `bytes.len()` must equal `product(shape) * element_type.byte_size()`.
    unsafe fn allocate_copy(
        device: &Device,
        shape: &[usize],
        element_type: ElementType,
        bytes: &[u8],
    ) -> Result<Self> {
        let params = sys::iree_hal_buffer_params_t {
            usage: sys::IREE_HAL_BUFFER_USAGE_DEFAULT,
            access: sys::IREE_HAL_MEMORY_ACCESS_ALL,
            type_: sys::IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            queue_affinity: sys::IREE_HAL_QUEUE_AFFINITY_ANY,
            min_alignment: 0,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: device/allocator are live; `shape` and `bytes` are valid for the
        // call and IREE copies them; `raw` receives an owned reference.
        unsafe {
            check(sys::iree_hal_buffer_view_allocate_buffer_copy(
                device.as_ptr(),
                device.allocator_ptr(),
                shape.len(),
                shape.as_ptr(),
                element_type.raw(),
                sys::IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                params,
                sys::iree_const_byte_span_t {
                    data: bytes.as_ptr(),
                    data_length: bytes.len(),
                },
                &mut raw,
            ))?;
            Self::from_owned_raw(raw, device.clone())
                .ok_or_else(|| Error::invalid_argument("buffer view allocation returned NULL"))
        }
    }

    /// # Safety
    /// `raw` must be NULL or an owned reference to a buffer view whose memory is
    /// usable with `device`.
    pub(crate) unsafe fn from_owned_raw(
        raw: *mut sys::iree_hal_buffer_view_t,
        device: Device,
    ) -> Option<Self> {
        NonNull::new(raw).map(|raw| Self { raw, device })
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::iree_hal_buffer_view_t {
        self.raw.as_ptr()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn shape(&self) -> Vec<usize> {
        // SAFETY: dims pointer is valid for `rank` elements while the view lives.
        unsafe {
            let rank = sys::iree_hal_buffer_view_shape_rank(self.raw.as_ptr());
            let dims = sys::iree_hal_buffer_view_shape_dims(self.raw.as_ptr());
            if rank == 0 || dims.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(dims, rank).to_vec()
            }
        }
    }

    pub fn rank(&self) -> usize {
        // SAFETY: live view.
        unsafe { sys::iree_hal_buffer_view_shape_rank(self.raw.as_ptr()) }
    }

    pub fn element_type(&self) -> ElementType {
        // SAFETY: live view.
        ElementType::from_raw(unsafe { sys::iree_hal_buffer_view_element_type(self.raw.as_ptr()) })
    }

    pub fn element_count(&self) -> usize {
        // SAFETY: live view.
        unsafe { sys::iree_hal_buffer_view_element_count(self.raw.as_ptr()) }
    }

    pub fn byte_length(&self) -> usize {
        // SAFETY: live view.
        unsafe { sys::iree_hal_buffer_view_byte_length(self.raw.as_ptr()) }
    }

    fn is_dense_row_major(&self) -> bool {
        // SAFETY: live view.
        let encoding = unsafe { sys::iree_hal_buffer_view_encoding_type(self.raw.as_ptr()) };
        encoding == sys::IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR
    }

    /// Copies the tensor contents to the host.
    ///
    /// Fails if `T`'s element type is not compatible with the view's element type
    /// (same-width integers are interchangeable, floats must match exactly — a
    /// f16 output can't be read as `f32`; use `half::f16` or [`to_bytes`](Self::to_bytes)).
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        let actual = self.element_type();
        if !T::ELEMENT_TYPE.is_compatible_with(actual) {
            return Err(Error::invalid_argument(format!(
                "cannot read {actual} tensor as {}",
                T::ELEMENT_TYPE
            )));
        }
        let numel = self.element_count();
        let byte_len = numel
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid_argument("tensor byte size overflows usize"))?;
        let mut out = Vec::<T>::with_capacity(numel);
        // SAFETY: `out` has capacity for `byte_len` bytes; on success every byte is
        // written by the transfer and `T` is POD, so `set_len` is sound.
        unsafe {
            self.read_into(out.as_mut_ptr().cast(), byte_len)?;
            out.set_len(numel);
        }
        Ok(out)
    }

    /// Copies the tensor contents into `dst` (must have exactly `element_count()`
    /// elements). Avoids an allocation when reusing host buffers.
    pub fn copy_to_slice<T: Element>(&self, dst: &mut [T]) -> Result<()> {
        let actual = self.element_type();
        if !T::ELEMENT_TYPE.is_compatible_with(actual) {
            return Err(Error::invalid_argument(format!(
                "cannot read {actual} tensor as {}",
                T::ELEMENT_TYPE
            )));
        }
        if dst.len() != self.element_count() {
            return Err(Error::invalid_argument(format!(
                "destination has {} elements, tensor has {}",
                dst.len(),
                self.element_count()
            )));
        }
        // SAFETY: `dst` is exactly the right size in bytes; `T` is POD.
        unsafe { self.read_into(dst.as_mut_ptr().cast(), std::mem::size_of_val(dst)) }
    }

    /// Copies the raw tensor bytes to the host (any byte-aligned element type).
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let len = self.expected_dense_bytes()?;
        let mut out = Vec::<u8>::with_capacity(len);
        // SAFETY: capacity `len`, fully written on success.
        unsafe {
            self.read_into(out.as_mut_ptr(), len)?;
            out.set_len(len);
        }
        Ok(out)
    }

    fn expected_dense_bytes(&self) -> Result<usize> {
        let elem_type = self.element_type();
        let elem_size = elem_type.byte_size().ok_or_else(|| {
            Error::invalid_argument(format!("element type {elem_type} is not byte-aligned"))
        })?;
        self.element_count()
            .checked_mul(elem_size)
            .ok_or_else(|| Error::invalid_argument("tensor byte size overflows usize"))
    }

    /// # Safety
    /// `dst` must be valid for writes of `len` bytes.
    unsafe fn read_into(&self, dst: *mut u8, len: usize) -> Result<()> {
        if !self.is_dense_row_major() {
            return Err(Error::invalid_argument(
                "buffer view is not dense row-major",
            ));
        }
        let dense = self.expected_dense_bytes()?;
        if len != dense || len > self.byte_length() {
            return Err(Error::invalid_argument(format!(
                "read of {len} bytes does not match tensor size {dense} (buffer {} bytes)",
                self.byte_length()
            )));
        }
        if len == 0 {
            return Ok(());
        }
        // SAFETY: the buffer is owned by the live view; `dst` valid for `len` bytes.
        unsafe {
            let buffer = sys::iree_hal_buffer_view_buffer(self.raw.as_ptr());
            check(sys::iree_hal_device_transfer_d2h(
                self.device.as_ptr(),
                buffer,
                0,
                dst.cast(),
                len,
                sys::IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                sys::iree_infinite_timeout(),
            ))
        }
    }
}

impl Clone for BufferView {
    fn clone(&self) -> Self {
        // SAFETY: live reference.
        unsafe { sys::iree_hal_buffer_view_retain(self.raw.as_ptr()) };
        Self {
            raw: self.raw,
            device: self.device.clone(),
        }
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        // SAFETY: we own one reference.
        unsafe { sys::iree_hal_buffer_view_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for BufferView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferView")
            .field("shape", &self.shape())
            .field("element_type", &self.element_type())
            .finish()
    }
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| Error::invalid_argument(format!("shape {shape:?} overflows usize")))
}
