use std::cell::Cell;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::time::Duration;

use iree_sys as sys;

use crate::error::{Error, Result, StatusCode, check};
use crate::instance::Device;

/// A timeline semaphore on a [`Device`] (`iree_hal_semaphore_t`).
///
/// Semaphores carry a monotonically increasing `u64` payload value. Queue
/// submissions and host callers synchronize by waiting on or signaling
/// timepoints through [`Fence`]s or directly on the semaphore.
///
/// `Send + Sync`: HAL semaphores are thread-safe and ref-counted.
pub struct Semaphore {
    raw: NonNull<sys::iree_hal_semaphore_t>,
    device: Device,
}

// SAFETY: `iree_hal_semaphore_t` uses atomic ref-counting and thread-safe
// query/signal/wait operations across threads.
unsafe impl Send for Semaphore {}
unsafe impl Sync for Semaphore {}

impl Semaphore {
    /// Creates a new timeline semaphore on `device` initialized to `initial_value`.
    pub fn new(device: &Device, initial_value: u64) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `device` is live; `raw` receives an owned ref on success.
        unsafe {
            check(sys::iree_hal_semaphore_create(
                device.as_ptr(),
                sys::IREE_HAL_QUEUE_AFFINITY_ANY,
                initial_value,
                sys::IREE_HAL_SEMAPHORE_FLAG_NONE,
                &mut raw,
            ))?;
        }
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::invalid_argument("iree_hal_semaphore_create returned NULL"))?;
        Ok(Self {
            raw,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Queries the current payload value without blocking.
    pub fn query(&self) -> Result<u64> {
        let mut value = 0u64;
        // SAFETY: `self.raw` is live; `value` is a valid out-pointer.
        unsafe {
            check(sys::iree_hal_semaphore_query(self.raw.as_ptr(), &mut value))?;
        }
        Ok(value)
    }

    /// Advances the semaphore payload to `value` (ignored if already past `value`).
    pub fn signal(&self, value: u64) -> Result<()> {
        // SAFETY: `self.raw` is live; `frontier` is optional (NULL for local signals).
        unsafe {
            check(sys::iree_hal_semaphore_signal(
                self.raw.as_ptr(),
                value,
                std::ptr::null(),
            ))
        }
    }

    /// Blocks the calling thread until the semaphore reaches or exceeds `value`.
    pub fn wait(&self, value: u64) -> Result<()> {
        // SAFETY: `self.raw` is live.
        unsafe {
            check(sys::iree_hal_semaphore_wait(
                self.raw.as_ptr(),
                value,
                sys::iree_infinite_timeout(),
                sys::IREE_ASYNC_WAIT_FLAG_NONE,
            ))
        }
    }

    /// Blocks up to `timeout` until the semaphore reaches or exceeds `value`.
    /// Returns `Ok(true)` if signaled, `Ok(false)` if the timeout expired.
    pub fn wait_timeout(&self, value: u64, timeout: Duration) -> Result<bool> {
        let iree_timeout = relative_timeout(timeout);
        // SAFETY: `self.raw` is live.
        let result = unsafe {
            check(sys::iree_hal_semaphore_wait(
                self.raw.as_ptr(),
                value,
                iree_timeout,
                sys::IREE_ASYNC_WAIT_FLAG_NONE,
            ))
        };
        match result {
            Ok(()) => Ok(true),
            Err(err) if err.code() == StatusCode::DeadlineExceeded => Ok(false),
            Err(err) => Err(err),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::iree_hal_semaphore_t {
        self.raw.as_ptr()
    }
}

impl Clone for Semaphore {
    fn clone(&self) -> Self {
        // SAFETY: `self.raw` is a live reference.
        unsafe { sys::iree_hal_semaphore_retain(self.raw.as_ptr()) };
        Self {
            raw: self.raw,
            device: self.device.clone(),
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        // SAFETY: we own one reference.
        unsafe { sys::iree_hal_semaphore_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for Semaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Semaphore").field("raw", &self.raw).finish()
    }
}

/// A set of `(Semaphore, value)` timepoints used to synchronize `$async` calls
/// (`iree_hal_fence_t`).
///
/// `Send` but `!Sync`: fences must not be mutated concurrently or while
/// referenced by an in-flight queue operation.
pub struct Fence {
    raw: NonNull<sys::iree_hal_fence_t>,
    // Keep the owning device (and therefore its instance/driver) alive.
    device: Device,
    _not_sync: PhantomData<Cell<()>>,
}

// SAFETY: atomic ref-counting; `!Sync` prevents concurrent mutation via `&Fence`.
unsafe impl Send for Fence {}

impl Fence {
    /// Creates an empty fence (wait completes immediately; signal is a no-op).
    pub fn empty(device: &Device) -> Result<Self> {
        Self::with_capacity(device, 0)
    }

    /// Creates an empty fence pre-sized for up to `capacity` distinct semaphores.
    pub fn with_capacity(device: &Device, capacity: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; fence retains its own host allocator.
        unsafe {
            check(sys::iree_hal_fence_create(
                capacity,
                sys::iree_allocator_system(),
                &mut raw,
            ))?;
        }
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::invalid_argument("iree_hal_fence_create returned NULL"))?;
        Ok(Self {
            raw,
            device: device.clone(),
            _not_sync: PhantomData,
        })
    }

    /// Creates a fence with capacity 1 containing the single `(semaphore, value)`
    /// timepoint. Use [`Fence::with_capacity`] when inserting multiple semaphores.
    pub fn at(semaphore: &Semaphore, value: u64) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `semaphore` is live; `iree_hal_fence_create_at` retains it.
        unsafe {
            check(sys::iree_hal_fence_create_at(
                semaphore.as_ptr(),
                value,
                sys::iree_allocator_system(),
                &mut raw,
            ))?;
        }
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::invalid_argument("iree_hal_fence_create_at returned NULL"))?;
        Ok(Self {
            raw,
            device: semaphore.device().clone(),
            _not_sync: PhantomData,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Adds `(semaphore, value)` (or updates to `max` if already present).
    pub fn insert(&mut self, semaphore: &Semaphore, value: u64) -> Result<()> {
        // SAFETY: exclusive `&mut self` ensures no concurrent mutation; `semaphore` is live.
        unsafe {
            check(sys::iree_hal_fence_insert(
                self.raw.as_ptr(),
                semaphore.as_ptr(),
                value,
            ))
        }
    }

    /// Extends `self` with all `(semaphore, value)` timepoints from `other`.
    pub fn extend(&mut self, other: &Fence) -> Result<()> {
        // SAFETY: exclusive `&mut self`; `other` is live.
        unsafe {
            check(sys::iree_hal_fence_extend(
                self.raw.as_ptr(),
                other.raw.as_ptr(),
            ))
        }
    }

    /// Non-blocking check: returns `Ok(true)` if all timepoints are reached,
    /// `Ok(false)` if still pending (`Deferred`), or `Err` on failure.
    pub fn is_signaled(&self) -> Result<bool> {
        // SAFETY: `self.raw` is live.
        let result = unsafe { check(sys::iree_hal_fence_query(self.raw.as_ptr())) };
        match result {
            Ok(()) => Ok(true),
            Err(err) if err.code() == StatusCode::Deferred => Ok(false),
            Err(err) => Err(err),
        }
    }

    /// Signals all timepoints in the fence from the host.
    pub fn signal(&self) -> Result<()> {
        // SAFETY: `self.raw` is live.
        unsafe { check(sys::iree_hal_fence_signal(self.raw.as_ptr())) }
    }

    /// Blocks until every timepoint in the fence has been signaled.
    pub fn wait(&self) -> Result<()> {
        // SAFETY: `self.raw` is live.
        unsafe {
            check(sys::iree_hal_fence_wait(
                self.raw.as_ptr(),
                sys::iree_infinite_timeout(),
                sys::IREE_ASYNC_WAIT_FLAG_NONE,
            ))
        }
    }

    /// Blocks up to `timeout` until every timepoint in the fence is signaled.
    /// Returns `Ok(true)` if signaled, `Ok(false)` if the timeout expired.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<bool> {
        let iree_timeout = relative_timeout(timeout);
        // SAFETY: `self.raw` is live.
        let result = unsafe {
            check(sys::iree_hal_fence_wait(
                self.raw.as_ptr(),
                iree_timeout,
                sys::IREE_ASYNC_WAIT_FLAG_NONE,
            ))
        };
        match result {
            Ok(()) => Ok(true),
            Err(err) if err.code() == StatusCode::DeadlineExceeded => Ok(false),
            Err(err) => Err(err),
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::iree_hal_fence_t {
        self.raw.as_ptr()
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // SAFETY: we own one reference.
        unsafe { sys::iree_hal_fence_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fence").field("raw", &self.raw).finish()
    }
}

fn relative_timeout(timeout: Duration) -> sys::iree_timeout_t {
    let nanos = i64::try_from(timeout.as_nanos()).unwrap_or(sys::IREE_TIME_INFINITE_FUTURE);
    sys::iree_make_timeout_ns(nanos)
}
