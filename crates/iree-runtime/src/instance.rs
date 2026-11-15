use std::ptr::NonNull;

use iree_sys as sys;

use crate::error::{Error, Result, check, string_from_view};

/// Process-wide IREE runtime instance (driver registry, VM instance, shared caches).
///
/// IREE documents `iree_runtime_instance_t` as thread-safe, so `Instance` is
/// `Send + Sync`. Cloning retains the underlying ref-counted handle.
pub struct Instance {
    raw: NonNull<sys::iree_runtime_instance_t>,
}

// SAFETY: iree_runtime_instance_t is documented as thread-safe and ref-counted
// with atomic retain/release.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl Instance {
    /// Creates an instance with every HAL driver compiled into the linked runtime
    /// (`IREE_HAL_DRIVER_*` CMake options), using the system allocator.
    pub fn new() -> Result<Self> {
        let mut options = sys::iree_runtime_instance_options_t {
            driver_registry: std::ptr::null_mut(),
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: `options` is a valid out-pointer and initialized before use;
        // `raw` receives an owned reference on success.
        unsafe {
            sys::iree_runtime_instance_options_initialize(&mut options);
            sys::iree_runtime_instance_options_use_all_available_drivers(&mut options);
            check(sys::iree_runtime_instance_create(
                &options,
                sys::iree_allocator_system(),
                &mut raw,
            ))?;
        }
        let raw = NonNull::new(raw)
            .ok_or_else(|| Error::invalid_argument("iree_runtime_instance_create returned NULL"))?;
        Ok(Self { raw })
    }

    /// Creates the default device for a HAL driver, e.g. `"local-task"`,
    /// `"local-sync"`, `"metal"`, `"vulkan"`, `"cuda"`, `"hip"`.
    ///
    /// The driver must be compiled into the linked runtime, and `.vmfb` modules
    /// loaded on the device must have been compiled for a matching target.
    pub fn create_device(&self, driver: &str) -> Result<Device> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: `self.raw` is live; `driver` outlives the call (IREE copies what
        // it needs); `raw` receives an owned reference on success.
        unsafe {
            check(sys::iree_runtime_instance_try_create_default_device(
                self.raw.as_ptr(),
                sys::iree_string_view_t::new(driver),
                &mut raw,
            ))?;
        }
        let raw = NonNull::new(raw).ok_or_else(|| {
            Error::invalid_argument(format!("driver `{driver}` returned a NULL device"))
        })?;
        Ok(Device {
            raw,
            instance: self.clone(),
        })
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::iree_runtime_instance_t {
        self.raw.as_ptr()
    }
}

impl Clone for Instance {
    fn clone(&self) -> Self {
        // SAFETY: `self.raw` is a live reference.
        unsafe { sys::iree_runtime_instance_retain(self.raw.as_ptr()) };
        Self { raw: self.raw }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: we own one reference.
        unsafe { sys::iree_runtime_instance_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance").field("raw", &self.raw).finish()
    }
}

/// A HAL device created from an [`Instance`].
///
/// HAL devices are designed to be shared by multiple sessions that execute
/// concurrently, so `Device` is `Send + Sync`. Cloning retains the handle.
/// The device keeps its instance alive.
pub struct Device {
    raw: NonNull<sys::iree_hal_device_t>,
    instance: Instance,
}

// SAFETY: HAL devices are ref-counted with atomic retain/release and support
// concurrent use by multiple sessions. The non-thread-safe device APIs
// (replace_allocator / replace_channel_provider) are not exposed, and `trim` is
// exposed only as an `unsafe fn`.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

impl Device {
    /// Driver-specific device identifier (e.g. `local-task`, `metal`).
    pub fn id(&self) -> String {
        // SAFETY: the returned view is valid for the device's lifetime; copied.
        unsafe { string_from_view(sys::iree_hal_device_id(self.raw.as_ptr())) }
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    /// Trims HAL pools and caches down to what live allocations need.
    ///
    /// # Safety
    /// Must not run concurrently with any other work on this device (no session
    /// using the device may be executing). Prefer [`crate::Session::trim`].
    pub unsafe fn trim(&self) -> Result<()> {
        unsafe { check(sys::iree_hal_device_trim(self.raw.as_ptr())) }
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::iree_hal_device_t {
        self.raw.as_ptr()
    }

    pub(crate) fn allocator_ptr(&self) -> *mut sys::iree_hal_allocator_t {
        // SAFETY: the device always owns an allocator for its lifetime.
        unsafe { sys::iree_hal_device_allocator(self.raw.as_ptr()) }
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        // SAFETY: `self.raw` is a live reference.
        unsafe { sys::iree_hal_device_retain(self.raw.as_ptr()) };
        Self {
            raw: self.raw,
            instance: self.instance.clone(),
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: we own one reference. `instance` is dropped afterwards (field order).
        unsafe { sys::iree_hal_device_release(self.raw.as_ptr()) };
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device").field("raw", &self.raw).finish()
    }
}
