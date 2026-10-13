//! `force_mut` on imported tensors: writes through to the non-rstsr-owned producer buffer.
//!
//! Asserts that mutating an imported `TensorDlpack` through `force_mut` changes the
//! *producer's* own buffer in place (zero-copy), not a copy.

use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dlpark::ffi::{DLDataType, DLDevice, DLManagedTensorVersioned, DLPackVersion, DLTensor};
use dlpark::DlpackFlags;
use rstsr_core::prelude::*;
use rstsr_cpu_dlpack::{from_dlpack_versioned_f, TensorDlpack};

/// Producer context: raw boxes owned by the (leaked) allocation, freed by `deleter`.
struct Ctx {
    data: *mut [f64],
    shape: *mut [i64],
    strides: *mut [i64],
    calls: Arc<AtomicUsize>,
}

unsafe extern "C" fn deleter(ptr: *mut DLManagedTensorVersioned) {
    let managed = unsafe { Box::from_raw(ptr) };
    let ctx = unsafe { Box::from_raw(managed.manager_ctx as *mut Ctx) };
    unsafe {
        if !ctx.data.is_null() {
            drop(Box::from_raw(ctx.data));
        }
        if !ctx.shape.is_null() {
            drop(Box::from_raw(ctx.shape));
        }
        if !ctx.strides.is_null() {
            drop(Box::from_raw(ctx.strides));
        }
    }
    ctx.calls.fetch_add(1, Ordering::SeqCst);
}

/// Builds a versioned producer tensor; data/shape/strides live in raw boxes that are
/// never moved, and the data pointer is taken with mutable provenance.
fn make(
    values: Vec<f64>,
    shape: Vec<i64>,
    strides: Option<Vec<i64>>,
    data_elem_offset: usize,
) -> (*mut DLManagedTensorVersioned, *const f64, Arc<AtomicUsize>) {
    let ndim = i32::try_from(shape.len()).unwrap();
    let data: *mut [f64] = Box::into_raw(values.into_boxed_slice());
    let shape: *mut [i64] = Box::into_raw(shape.into_boxed_slice());
    let strides: *mut [i64] = match strides {
        Some(s) => Box::into_raw(s.into_boxed_slice()),
        None => ptr::slice_from_raw_parts_mut(ptr::null_mut::<i64>(), 0),
    };
    let base = data as *const f64;
    let data_ptr = unsafe { (data as *mut f64).add(data_elem_offset) } as *mut c_void;
    let calls = Arc::new(AtomicUsize::new(0));
    let ctx = Box::into_raw(Box::new(Ctx { data, shape, strides, calls: Arc::clone(&calls) }));
    let dl_tensor = DLTensor {
        data: data_ptr,
        device: DLDevice::CPU,
        ndim,
        dtype: DLDataType::of::<f64>(),
        shape: shape as *mut i64,
        strides: strides as *mut i64,
        byte_offset: 0,
    };
    let managed = Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: ctx as *mut c_void,
        deleter: Some(deleter),
        flags: DlpackFlags::empty(),
        dl_tensor,
    });
    (Box::into_raw(managed), base, calls)
}

#[test]
fn force_mut_writes_through_to_foreign_memory_zero_copy() {
    let (ptr, base, calls) = make(vec![1.0, 2.0, 3.0, 4.0], vec![4], None, 0);
    let tensor: TensorDlpack<f64, DeviceCpuSerial, IxD> = unsafe { from_dlpack_versioned_f(ptr) }.unwrap();

    // mutate through the `force_mut` escape hatch (no other access to the buffer)
    {
        let mut tm = unsafe { tensor.force_mut() };
        tm *= 2.0;
    }

    // the producer's own buffer now holds the doubled values
    unsafe {
        assert_eq!(*base, 2.0);
        assert_eq!(*base.add(1), 4.0);
        assert_eq!(*base.add(2), 6.0);
        assert_eq!(*base.add(3), 8.0);
    }

    // and the deleter still fires exactly once on drop
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

/// Same, but the imported layout is strided/offset — writes must land at the
/// addressed elements, not at address `0`.
#[test]
fn force_mut_respects_the_imported_layout() {
    // buffer 0..8, tensor = elements 1,3,5,7 (stride 2, offset 1)
    let (ptr, base, _calls) = make((0..8).map(|v| v as f64).collect(), vec![4], Some(vec![2]), 1);
    let tensor: TensorDlpack<f64, DeviceCpuSerial, IxD> = unsafe { from_dlpack_versioned_f(ptr) }.unwrap();
    assert_eq!(tensor.layout().stride().to_vec(), vec![2]);

    {
        let mut tm = unsafe { tensor.force_mut() };
        tm += 10.0;
    }

    // addressed elements bumped by 10; the interleaved ones keep their value
    unsafe {
        assert_eq!(*base, 0.0);
        assert_eq!(*base.add(1), 11.0);
        assert_eq!(*base.add(2), 2.0);
        assert_eq!(*base.add(3), 13.0);
        assert_eq!(*base.add(4), 4.0);
        assert_eq!(*base.add(5), 15.0);
        assert_eq!(*base.add(6), 6.0);
        assert_eq!(*base.add(7), 17.0);
    }
}

/// Alternative route: `dl_tensor.view().force_mut()`. The view borrows the repr's
/// span as a `DataRef`, which core already implements `DataForceMutAPI` for — so this
/// needs no bridge-crate impl at all.
#[test]
fn force_mut_through_view_writes_through_to_foreign_memory() {
    let (ptr, base, calls) = make(vec![1.0, 2.0, 3.0, 4.0], vec![4], None, 0);
    let tensor: TensorDlpack<f64, DeviceCpuSerial, IxD> = unsafe { from_dlpack_versioned_f(ptr) }.unwrap();

    {
        let view = tensor.view();
        let mut tm = unsafe { view.force_mut() };
        tm += 100.0;
    }

    // the producer's own buffer was mutated in place
    unsafe {
        assert_eq!(*base, 101.0);
        assert_eq!(*base.add(1), 102.0);
        assert_eq!(*base.add(2), 103.0);
        assert_eq!(*base.add(3), 104.0);
    }
    drop(tensor);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}
