//! Safe wrapper over the IREE runtime C API for inference.
//!
//! Compilation, export and quantization happen elsewhere (Python: iree-turbine /
//! iree-import-onnx / iree-compile). This crate loads the resulting `.vmfb`,
//! feeds tensors in, invokes exported functions and reads tensors back.
//!
//! ```no_run
//! use iree_runtime::{BufferView, Instance, Session};
//!
//! # fn main() -> iree_runtime::Result<()> {
//! let instance = Instance::new()?;
//! let device = instance.create_device("local-task")?;
//! let session = Session::from_file(&device, "model.vmfb")?;
//!
//! let main = session.lookup_function("module.main")?;
//! println!("{:?}", main.declaration()); // tensor<1x3x224x224xf32> -> ...
//!
//! let pixels = vec![0.0f32; 3 * 224 * 224];
//! let input = BufferView::from_slice(&device, &[1, 3, 224, 224], &pixels)?;
//! let mut call = main.call()?;
//! let outputs = call.invoke_with(&[&input])?;
//! let embedding: Vec<f32> = outputs[0].to_vec()?;
//! # let _ = embedding; Ok(()) }
//! ```
//!
//! # Threading model (mirrors IREE's documented contracts)
//!
//! | Type | Send | Sync | Why |
//! |---|---|---|---|
//! | [`Instance`] | ✓ | ✓ | `iree_runtime_instance_t` is thread-safe |
//! | [`Device`] | ✓ | ✓ | HAL devices are shared by concurrently running sessions |
//! | [`Session`] | ✓ | ✗ | thread-compatible: one thread at a time |
//! | [`Function`], [`Call`] | ✗ | ✗ | borrow `&Session` |
//! | [`BufferView`] | ✓ | ✗ | ref-counted, derived from a session |
//!
//! These are enforced at compile time:
//!
//! ```compile_fail,E0277
//! fn is_sync<T: Sync>() {}
//! is_sync::<iree_runtime::Session>(); // a session cannot be shared across threads
//! ```
//!
//! ```compile_fail,E0277
//! fn is_send<T: Send>() {}
//! is_send::<iree_runtime::Call<'static>>(); // a call cannot leave its session's thread
//! ```
//!
//! ```compile_fail,E0277
//! fn is_sync<T: Sync>() {}
//! is_sync::<iree_runtime::BufferView>();
//! ```
//!
//! # Precision
//!
//! Precision is a compile-time property of the `.vmfb`; the runtime only sees the
//! exported function signature. Use [`Function::declaration`] to assert the I/O
//! dtypes/shapes of a loaded module before feeding it.

mod buffer_view;
mod call;
mod element;
mod error;
mod instance;
mod session;

pub use buffer_view::BufferView;
pub use call::Call;
pub use element::{Element, ElementType};
pub use error::{Error, Result, StatusCode};
pub use instance::{Device, Instance};
pub use session::{ABI_DECLARATION_ATTR, Function, FunctionInfo, Session};

/// Raw FFI bindings, for anything the safe API does not cover.
pub use iree_sys as sys;

#[cfg(test)]
mod static_assertions {
    use super::*;

    fn is_send<T: Send>() {}
    fn is_sync<T: Sync>() {}

    #[test]
    fn auto_traits() {
        is_send::<Instance>();
        is_sync::<Instance>();
        is_send::<Device>();
        is_sync::<Device>();
        is_send::<Session>();
        is_send::<BufferView>();
    }
}
