//! End-to-end tests against the real IREE runtime.
//!
//! Requires:
//!   1. just iree       (otherwise this file is compiled out)
//!   2. just testdata   (produces testdata/build/basic.*.vmfb)
#![cfg(iree_linked)]

use std::path::PathBuf;
use std::time::Duration;

use iree_runtime::{
    BufferView, Device, ElementType, Fence, Instance, Semaphore, Session, StatusCode,
};

/// `IREE_TESTDATA_DIR`, else `<workspace>/testdata/build`.
fn testdata_dir() -> PathBuf {
    std::env::var_os("IREE_TESTDATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/build"))
}

fn testdata(name: &str) -> PathBuf {
    let path = testdata_dir().join(name);
    assert!(
        path.is_file(),
        "missing {}; run just testdata first",
        path.display()
    );
    path
}

fn cpu() -> (Instance, Device) {
    let instance = Instance::new().expect("instance");
    let device = instance
        .create_device("local-task")
        .expect("local-task device");
    (instance, device)
}

fn basic_session(device: &Device) -> Session {
    Session::from_file(device, testdata("basic.llvm-cpu.vmfb")).expect("load basic.vmfb")
}

#[test]
fn unknown_driver_is_an_error() {
    let instance = Instance::new().unwrap();
    let err = instance.create_device("no-such-driver").unwrap_err();
    assert!(!err.message().is_empty(), "{err}");
}

#[test]
fn reflection_reports_declarations() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);

    let exported = session.exported_functions();
    let abs = exported
        .iter()
        .find(|f| f.full_name == "basic.abs")
        .unwrap_or_else(|| panic!("basic.abs not exported: {exported:#?}"));
    assert_eq!(abs.calling_convention, "0r_r");
    let decl = abs.declaration.as_deref().expect("iree.abi.declaration");
    assert!(decl.contains("tensor<4xf32>"), "{decl}");

    let f = session.lookup_function("basic.sum_and_product").unwrap();
    assert_eq!(f.arity(), Some((2, 2)));

    let missing = session.lookup_function("basic.nope").unwrap_err();
    assert_eq!(missing.code(), StatusCode::NotFound);
}

#[test]
fn abs_f32() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);
    let input = BufferView::from_slice(&device, &[4], &[-1.0f32, 2.0, -3.5, 0.0]).unwrap();
    let mut call = session.call("basic.abs").unwrap();
    let out = call.invoke_with(&[&input]).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), vec![4]);
    assert_eq!(out[0].element_type(), ElementType::FLOAT32);
    assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.5, 0.0]);
}

#[test]
fn module_from_bytes() {
    let (_instance, device) = cpu();
    let bytes = std::fs::read(testdata("basic.llvm-cpu.vmfb")).unwrap();
    let mut session = Session::new(&device).unwrap();
    session.load_module_bytes(&bytes).unwrap();
    drop(bytes); // IREE owns its own copy.
    let input = BufferView::from_slice(&device, &[4], &[-4.0f32; 4]).unwrap();
    let out = session
        .call("basic.abs")
        .unwrap()
        .invoke_with(&[&input])
        .unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![4.0; 4]);
}

#[test]
fn i64_inputs_match_signless_abi() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);
    let a = BufferView::from_slice(&device, &[2, 3], &[1i64, 2, 3, 4, 5, 6]).unwrap();
    let b = BufferView::from_slice(&device, &[2, 3], &[10i64; 6]).unwrap();
    let out = session
        .call("basic.add_i64")
        .unwrap()
        .invoke_with(&[&a, &b])
        .unwrap();
    assert_eq!(out[0].shape(), vec![2, 3]);
    assert_eq!(
        out[0].to_vec::<i64>().unwrap(),
        vec![11, 12, 13, 14, 15, 16]
    );
}

#[test]
fn multiple_results_in_order() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);
    let a = BufferView::from_slice(&device, &[4], &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let b = BufferView::from_slice(&device, &[4], &[2.0f32; 4]).unwrap();
    let out = session
        .call("basic.sum_and_product")
        .unwrap()
        .invoke_with(&[&a, &b])
        .unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(out[1].to_vec::<f32>().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn dynamic_shapes() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);
    let mut call = session.call("basic.negate_dynamic").unwrap();
    for (h, w) in [(1usize, 1usize), (3, 5), (32, 64)] {
        let data: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
        let input = BufferView::from_slice(&device, &[h, w], &data).unwrap();
        let out = call.invoke_with(&[&input]).unwrap();
        assert_eq!(out[0].shape(), vec![h, w]);
        let expected: Vec<f32> = data.iter().map(|v| -v).collect();
        assert_eq!(out[0].to_vec::<f32>().unwrap(), expected);
    }
}

#[test]
fn f16_output_requires_matching_type() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);
    let input = BufferView::from_slice(&device, &[4], &[1.0f32, -2.0, 0.5, 65504.0]).unwrap();
    let out = session
        .call("basic.to_f16")
        .unwrap()
        .invoke_with(&[&input])
        .unwrap();
    assert_eq!(out[0].element_type(), ElementType::FLOAT16);
    assert!(
        out[0].to_vec::<f32>().is_err(),
        "f16 must not be read as f32"
    );
    let bytes = out[0].to_bytes().unwrap();
    assert_eq!(bytes.len(), 8);
    // 1.0 in IEEE half = 0x3C00 (little endian).
    assert_eq!(&bytes[0..2], &[0x00, 0x3C]);
    #[cfg(feature = "half")]
    {
        let halfs = out[0].to_vec::<half::f16>().unwrap();
        assert_eq!(halfs[1].to_f32(), -2.0);
    }
}

#[test]
fn validation_errors_do_not_crash() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);

    // Shape / data length mismatch is caught before calling IREE.
    let err = BufferView::from_slice(&device, &[2, 2], &[1.0f32; 3]).unwrap_err();
    assert_eq!(err.code(), StatusCode::InvalidArgument);

    // Wrong arity: IREE rejects the call with a status, no UB.
    let x = BufferView::from_slice(&device, &[4], &[1.0f32; 4]).unwrap();
    let mut call = session.call("basic.sum_and_product").unwrap();
    assert!(call.invoke_with(&[&x]).is_err());

    // Wrong dtype (f32 into an i64 function) and wrong static shape.
    let mut call = session.call("basic.add_i64").unwrap();
    let f = BufferView::from_slice(&device, &[2, 3], &[0.0f32; 6]).unwrap();
    assert!(call.invoke_with(&[&f, &f]).is_err());
    let mut call = session.call("basic.abs").unwrap();
    let wrong = BufferView::from_slice(&device, &[5], &[0.0f32; 5]).unwrap();
    assert!(call.invoke_with(&[&wrong]).is_err());

    // The call is still usable after an error.
    assert_eq!(
        call.invoke_with(&[&x]).unwrap()[0].to_vec::<f32>().unwrap(),
        vec![1.0; 4]
    );
}

#[test]
fn inputs_from_another_device_are_rejected() {
    let instance = Instance::new().unwrap();
    let device_a = instance.create_device("local-task").unwrap();
    let device_b = instance.create_device("local-sync").unwrap();
    let session = basic_session(&device_a);
    let foreign = BufferView::from_slice(&device_b, &[4], &[1.0f32; 4]).unwrap();
    let err = session
        .call("basic.abs")
        .unwrap()
        .push_input(&foreign)
        .unwrap_err();
    assert_eq!(err.code(), StatusCode::InvalidArgument);
}

#[test]
fn sessions_run_concurrently_on_a_shared_device() {
    let (_instance, device) = cpu();
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let device = device.clone();
            std::thread::spawn(move || {
                // Each thread owns its session (Send, !Sync).
                let session = basic_session(&device);
                let mut call = session.call("basic.abs").unwrap();
                for i in 0..50 {
                    let v = -((t * 100 + i) as f32);
                    let x = BufferView::from_slice(&device, &[4], &[v; 4]).unwrap();
                    let out = call.invoke_with(&[&x]).unwrap();
                    assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![-v; 4]);
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn session_can_move_to_a_worker_thread() {
    let (_instance, device) = cpu();
    let session = basic_session(&device);
    let result = std::thread::spawn(move || {
        let x = BufferView::from_slice(session.device(), &[4], &[-1.0f32; 4]).unwrap();
        let out = session
            .call("basic.abs")
            .unwrap()
            .invoke_with(&[&x])
            .unwrap();
        out[0].to_vec::<f32>().unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(result, vec![1.0; 4]);
}

/// Metal: ignored by default, run by `just test-gpu`, where a module that
/// is not compiled or a driver that is unavailable fails.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "GPU: run with just test-gpu"]
fn gpu_metal_abs() {
    let vmfb = testdata_dir().join("basic.metal.vmfb");
    assert!(
        vmfb.is_file(),
        "{} not compiled: just testdata, with Xcode's Metal Toolchain",
        vmfb.display()
    );
    let instance = Instance::new().unwrap();
    let device = instance
        .create_device("metal")
        .unwrap_or_else(|err| panic!("metal driver unavailable: {err}"));
    let session = Session::from_file(&device, &vmfb).unwrap();
    let x = BufferView::from_slice(&device, &[4], &[-1.0f32, 2.0, -3.0, 4.0]).unwrap();
    let out = session
        .call("basic.abs")
        .unwrap()
        .invoke_with(&[&x])
        .unwrap();
    assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

fn basic_async_session(device: &Device) -> Session {
    Session::from_file(device, testdata("basic_async.llvm-cpu.vmfb"))
        .expect("load basic_async.vmfb")
}

#[test]
fn semaphore_and_fence_lifecycle() {
    let (_instance, device) = cpu();
    let sem = Semaphore::new(&device, 0).unwrap();
    assert_eq!(sem.query().unwrap(), 0);
    assert!(!sem.wait_timeout(1, Duration::from_millis(1)).unwrap());

    let mut fence = Fence::with_capacity(&device, 2).unwrap();
    fence.insert(&sem, 1).unwrap();
    assert!(!fence.is_signaled().unwrap());
    assert!(!fence.wait_timeout(Duration::from_millis(1)).unwrap());

    sem.signal(1).unwrap();
    assert_eq!(sem.query().unwrap(), 1);
    assert!(fence.is_signaled().unwrap());
    fence.wait().unwrap();
    assert!(fence.wait_timeout(Duration::from_millis(1)).unwrap());

    let sem2 = Semaphore::new(&device, 5).unwrap();
    fence.insert(&sem2, 7).unwrap();
    assert!(!fence.is_signaled().unwrap());
    sem2.signal(7).unwrap();
    assert!(fence.is_signaled().unwrap());
}

#[test]
fn async_call_pipeline_cpu() {
    let (_instance, device) = cpu();
    let session = basic_async_session(&device);
    let func = session.lookup_async_function("basic_async.abs").unwrap();
    assert!(func.is_async());
    let mut call = func.call().unwrap();

    // Independent per-slot semaphores avoid backward timeline signaling if
    // CPU worker threads complete out of submission order.
    let slots = 8usize;
    let semaphores: Vec<_> = (0..slots)
        .map(|_| Semaphore::new(&device, 0).unwrap())
        .collect();
    let mut pending = Vec::with_capacity(slots);
    for (i, sem) in semaphores.iter().enumerate() {
        let v = -((i + 1) as f32);
        let x = BufferView::from_slice(&device, &[4], &[v; 4]).unwrap();
        let sig = Fence::at(sem, 1).unwrap();
        pending.push((i, call.invoke_async(&[&x], None, sig).unwrap()));
    }

    for (i, p) in pending.into_iter().rev() {
        let out = p.wait().unwrap();
        assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![(i + 1) as f32; 4]);
    }

    // Chained dependency on device: abs(sum_and_product(a, b).0)
    let mut sum_prod = session.call_async("basic_async.sum_and_product").unwrap();
    let a = BufferView::from_slice(&device, &[4], &[-5.0f32, 1.0, -3.0, 2.0]).unwrap();
    let b = BufferView::from_slice(&device, &[4], &[2.0f32, -4.0, 1.0, -6.0]).unwrap();
    let sig1 = Fence::at(&semaphores[0], 2).unwrap();
    let first = sum_prod.invoke_async(&[&a, &b], None, sig1).unwrap();
    let sig2 = Fence::at(&semaphores[1], 2).unwrap();
    let second = {
        let sum_view = BufferView::from_slice(&device, &[4], &[-3.0f32, -3.0, -2.0, -4.0]).unwrap();
        call.invoke_async(&[&sum_view], Some(first.signal_fence()), sig2)
            .unwrap()
    };
    let out1 = first.wait().unwrap();
    let out2 = second.wait().unwrap();
    assert_eq!(
        out1[0].to_vec::<f32>().unwrap(),
        vec![-3.0, -3.0, -2.0, -4.0]
    );
    assert_eq!(
        out1[1].to_vec::<f32>().unwrap(),
        vec![-10.0, -4.0, -3.0, -12.0]
    );
    assert_eq!(out2[0].to_vec::<f32>().unwrap(), vec![3.0, 3.0, 2.0, 4.0]);
}

#[test]
fn async_call_cross_thread_wait() {
    let (_instance, device) = cpu();
    let session = basic_async_session(&device);
    let mut call = session.call_async("basic_async.abs").unwrap();
    let sem = Semaphore::new(&device, 0).unwrap();
    let x = BufferView::from_slice(&device, &[4], &[-9.0f32, 4.0, -2.5, 0.0]).unwrap();
    let pending = call
        .invoke_async(&[&x], None, Fence::at(&sem, 1).unwrap())
        .unwrap();
    let values = std::thread::spawn(move || pending.wait().unwrap()[0].to_vec::<f32>().unwrap())
        .join()
        .unwrap();
    assert_eq!(values, vec![9.0, 4.0, 2.5, 0.0]);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "GPU: run with just test-gpu"]
fn gpu_metal_async_pipeline() {
    let vmfb = testdata_dir().join("basic_async.metal.vmfb");
    assert!(
        vmfb.is_file(),
        "{} not compiled: just testdata, with Xcode's Metal Toolchain",
        vmfb.display()
    );
    let instance = Instance::new().unwrap();
    let device = instance
        .create_device("metal")
        .unwrap_or_else(|err| panic!("metal driver unavailable: {err}"));
    let session = Session::from_file(&device, &vmfb).unwrap();
    let mut call = session.call_async("basic_async.abs").unwrap();

    let slots = 16usize;
    let semaphores: Vec<_> = (0..slots)
        .map(|_| Semaphore::new(&device, 0).unwrap())
        .collect();
    let mut pending = Vec::with_capacity(slots);
    for (i, sem) in semaphores.iter().enumerate() {
        let v = -((i + 1) as f32);
        let x = BufferView::from_slice(&device, &[4], &[v; 4]).unwrap();
        let sig = Fence::at(sem, 1).unwrap();
        pending.push((i, call.invoke_async(&[&x], None, sig).unwrap()));
    }
    for (i, p) in pending {
        let out = p.wait().unwrap();
        assert_eq!(out[0].to_vec::<f32>().unwrap(), vec![(i + 1) as f32; 4]);
    }
}
