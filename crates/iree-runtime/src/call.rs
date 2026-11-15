use iree_sys as sys;

use crate::buffer_view::BufferView;
use crate::error::{Error, Result, check};
use crate::session::Session;

/// A reusable call builder (`iree_runtime_call_t`) for one exported function.
///
/// Borrows its [`Session`]; because `Session` is `!Sync`, a `Call` is `!Send`
/// and can only be used on the thread that owns the session borrow, which
/// satisfies IREE's thread-compatibility contract.
///
/// Typical use: [`Call::invoke_with`] per request. For lower-level control use
/// [`push_input`](Self::push_input) / [`invoke`](Self::invoke) /
/// [`pop_output`](Self::pop_output) and [`reset`](Self::reset) between calls.
pub struct Call<'s> {
    raw: sys::iree_runtime_call_t,
    session: &'s Session,
}

impl<'s> Call<'s> {
    pub(crate) fn new(session: &'s Session, function: sys::iree_vm_function_t) -> Result<Self> {
        let mut raw = sys::iree_runtime_call_t {
            session: std::ptr::null_mut(),
            function: sys::iree_vm_function_t::null(),
            inputs: std::ptr::null_mut(),
            outputs: std::ptr::null_mut(),
        };
        // SAFETY: `session` is live and borrowed for 's; `function` came from it.
        // On failure `iree_runtime_call_initialize` already deinitializes the
        // struct itself (releasing the session ref and lists without nulling the
        // pointers), so we must NOT deinitialize again — just drop `raw`.
        unsafe {
            check(sys::iree_runtime_call_initialize(
                session.as_ptr(),
                function,
                &mut raw,
            ))?
        };
        Ok(Self { raw, session })
    }

    pub fn session(&self) -> &'s Session {
        self.session
    }

    /// Appends an input. The list retains the buffer view, so `input` may be
    /// dropped afterwards.
    pub fn push_input(&mut self, input: &BufferView) -> Result<()> {
        // Buffers from another device may live in memory this device cannot
        // address (e.g. a Metal buffer passed to a local-task session).
        if input.device().as_ptr() != self.session.device().as_ptr() {
            return Err(Error::invalid_argument(
                "input buffer view was allocated on a different device than the session",
            ));
        }
        // SAFETY: `self.raw` is initialized; the buffer view is live and is
        // retained by the input list.
        unsafe {
            check(sys::iree_runtime_call_inputs_push_back_buffer_view(
                &mut self.raw,
                input.as_ptr(),
            ))
        }
    }

    /// Number of inputs currently pushed.
    pub fn input_count(&self) -> usize {
        // SAFETY: lists are owned by the initialized call.
        unsafe { list_size(sys::iree_runtime_call_inputs(&self.raw)) }
    }

    /// Number of outputs available to [`pop_output`](Self::pop_output).
    pub fn output_count(&self) -> usize {
        // SAFETY: as above.
        unsafe { list_size(sys::iree_runtime_call_outputs(&self.raw)) }
    }

    /// Synchronously invokes the function with the pushed inputs. Inputs remain
    /// in place for reuse; outputs are appended to the output list.
    pub fn invoke(&mut self) -> Result<()> {
        // SAFETY: exclusive access to the call; the session is borrowed and (being
        // !Sync) not used concurrently from another thread.
        unsafe {
            check(sys::iree_runtime_call_invoke(
                &mut self.raw,
                sys::IREE_RUNTIME_CALL_FLAG_RESERVED,
            ))
        }
    }

    /// Pops the next output as a buffer view (ownership transfers to the caller).
    /// Fails if there are no more outputs or the output is not a buffer view.
    pub fn pop_output(&mut self) -> Result<BufferView> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out-pointer; on success we receive an owned reference.
        unsafe {
            check(sys::iree_runtime_call_outputs_pop_front_buffer_view(
                &mut self.raw,
                &mut raw,
            ))?;
            BufferView::from_owned_raw(raw, self.session.device().clone())
                .ok_or_else(|| Error::invalid_argument("call output is NULL"))
        }
    }

    /// Clears inputs and outputs so the call can be reused.
    pub fn reset(&mut self) {
        // SAFETY: the call is initialized.
        unsafe { sys::iree_runtime_call_reset(&mut self.raw) };
    }

    /// Resets, pushes `inputs`, invokes, and returns every output in order.
    pub fn invoke_with(&mut self, inputs: &[&BufferView]) -> Result<Vec<BufferView>> {
        self.reset();
        for input in inputs {
            self.push_input(input)?;
        }
        self.invoke()?;
        let count = self.output_count();
        let mut outputs = Vec::with_capacity(count);
        for _ in 0..count {
            outputs.push(self.pop_output()?);
        }
        // Drop the (now empty) output list contents and the retained inputs so
        // input buffers are not kept alive until the next call.
        self.reset();
        Ok(outputs)
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        // SAFETY: releases the lists and the session reference taken at init.
        unsafe { sys::iree_runtime_call_deinitialize(&mut self.raw) };
    }
}

impl std::fmt::Debug for Call<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Call")
            .field("inputs", &self.input_count())
            .field("outputs", &self.output_count())
            .finish()
    }
}

unsafe fn list_size(list: *mut sys::iree_vm_list_t) -> usize {
    if list.is_null() {
        0
    } else {
        unsafe { sys::iree_vm_list_size(list) }
    }
}
