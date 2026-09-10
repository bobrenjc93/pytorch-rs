//! Native backend contracts: these tests run without Python bindings.
use pytorch_rs::{DType, Device, Tensor, TensorError, cuda};

#[test]
fn cuda_zeros_checks_rank_device_and_layout_before_runtime() {
    for shape in [vec![], vec![1, 2, 3], vec![0, 2, 3]] {
        assert_eq!(
            Tensor::cuda_zeros_float32(shape, Device::Cuda(0)).err(),
            Some(TensorError::UnsupportedCudaZeroTensor {
                reason: "shape rank is not 1 or 2",
            })
        );
    }
    for shape in [vec![2], vec![2, 3], vec![0, 3]] {
        assert_eq!(
            Tensor::cuda_zeros_float32(shape, Device::Cpu).err(),
            Some(TensorError::UnsupportedDevice {
                operation: "zeros",
                device: Device::Cpu,
            })
        );
    }
    let too_many = isize::MAX.unsigned_abs() / size_of::<f32>() + 1;
    for (shape, error) in [
        ([usize::MAX, 2], TensorError::ElementCountOverflow),
        ([0, usize::MAX], TensorError::StrideCalculationOverflow),
        (
            [too_many, 1],
            TensorError::StorageCapacityOverflow { elements: too_many },
        ),
    ] {
        assert_eq!(
            Tensor::cuda_zeros_float32(shape, Device::Cuda(0)).err(),
            Some(error)
        );
    }
}

#[test]
fn native_cuda_matrix_zeros_preserve_layout_values_and_independent_storage() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA matrix zeros: no CUDA runtime/device");
        return;
    }
    let shapes = [[0, 0], [0, 7], [5, 0], [1, 9], [11, 1]]
        .into_iter()
        .chain((1..=16).map(|i| [i * 3, i * 5 + 1]));
    for shape in shapes {
        let tensor = Tensor::cuda_zeros_float32(shape, Device::Cuda(0)).unwrap();
        let second = Tensor::cuda_zeros_float32(shape, Device::Cuda(0)).unwrap();
        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.stride(), [shape[1].max(1), 1]);
        assert_eq!(tensor.storage_offset(), 0);
        assert_eq!(tensor.numel(), shape[0] * shape[1]);
        assert_eq!(tensor.dtype(), DType::Float32);
        assert_eq!(tensor.device(), Device::Cuda(0));
        assert!(!tensor.requires_grad());
        if tensor.numel() == 0 {
            assert_eq!(tensor.data_ptr(), 0);
            assert_eq!(second.data_ptr(), 0);
        } else {
            assert_ne!(tensor.data_ptr(), 0);
            assert_ne!(tensor.data_ptr(), second.data_ptr());
        }
        let cpu = tensor.try_copy_cuda_to_cpu().unwrap();
        assert_eq!(cpu.shape(), shape);
        assert_eq!(cpu.stride(), tensor.stride());
        assert_eq!(cpu.device(), Device::Cpu);
        assert!(
            cpu.try_to_vec()
                .unwrap()
                .iter()
                .all(|value| value.to_bits() == 0)
        );
        drop(tensor);
        assert_eq!(second.try_copy_cuda_to_cpu().unwrap(), cpu);
    }
    for shape in [[0, 3], [3, 0], [2, 3]] {
        assert!(matches!(
            Tensor::cuda_zeros_float32(shape, Device::Cuda(cuda::device_count())),
            Err(TensorError::CudaRuntimeError { .. })
        ));
    }
}

fn unsupported<T>(result: Result<T, TensorError>, operation: &'static str) {
    assert_eq!(
        result.err(),
        Some(TensorError::UnsupportedDevice {
            operation,
            device: Device::Cuda(0),
        })
    );
}

fn rejection_at_boundary(call: impl FnOnce()) {
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(call))
        .expect_err("infallible CPU convenience API must reject CUDA at entry");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap();
    assert!(message.contains("unsupported device"), "{message}");
    assert!(
        !message.contains("storage"),
        "reached storage internals: {message}"
    );
}

#[test]
fn fallible_cpu_apis_preserve_values_and_gradients() {
    let x = Tensor::from_vec(vec![1.0, 2.0], [2])
        .unwrap()
        .try_with_requires_grad(true)
        .unwrap();
    let y = x.try_sum().unwrap();
    assert_eq!(y.try_to_vec().unwrap(), [3.0]);
    assert!(
        x.try_equal(&Tensor::from_vec(vec![1.0, 2.0], [2]).unwrap())
            .unwrap()
    );
    assert!(!x.try_equal(&Tensor::zeros([2]).unwrap()).unwrap());
    y.backward().unwrap();
    assert_eq!(x.grad().unwrap().unwrap().try_to_vec().unwrap(), [1.0, 1.0]);
}

#[test]
fn standalone_cuda_operations_reject_before_cpu_storage_access() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA rejection tests: no CUDA runtime/device");
        return;
    }
    for size in [0, 1, 2] {
        let mut x = Tensor::cuda_zeros_float32(vec![size], Device::Cuda(0)).unwrap();
        let cpu = Tensor::zeros([size]).unwrap();
        for inputs in [[&x, &x], [&x, &cpu], [&cpu, &x]] {
            unsupported(Tensor::cat(&inputs, 0), "cat");
            unsupported(Tensor::stack(&inputs, 0), "stack");
        }
        unsupported(x.try_sum(), "sum");
        unsupported(x.try_equal(&x), "equal");
        unsupported(x.try_equal(&cpu), "equal");
        unsupported(cpu.try_equal(&x), "equal");
        unsupported(x.backward(), "backward");
        unsupported(x.requires_grad_(true), "requires_grad_");
        unsupported(
            x.detach().unwrap().try_with_requires_grad(true),
            "with_requires_grad",
        );
        assert!(!x.requires_grad());
        assert!(
            !x.detach()
                .unwrap()
                .try_with_requires_grad(false)
                .unwrap()
                .requires_grad()
        );
        assert!(format!("{x:?}").contains("Cuda(0)"));
        rejection_at_boundary(|| {
            let _ = x.sum();
        });
        // An alias shares the leaf flag. Rejection must precede its mutation.
        let alias = x.reshape([i64::try_from(size).unwrap()]).unwrap();
        rejection_at_boundary(|| {
            let _ = x == alias;
        });
        rejection_at_boundary(|| {
            let _ = x.with_requires_grad(true);
        });
        assert!(!alias.requires_grad());
        assert_eq!(
            alias.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![0.0; size]
        );
    }
}

#[test]
fn host_to_device_rejects_unsupported_inputs_without_runtime() {
    let source = Tensor::from_vec(vec![3.25], []).unwrap();
    assert_eq!(
        source.try_copy_cpu_to_cuda(Device::Cpu).err(),
        Some(TensorError::UnsupportedCudaTransfer {
            reason: "destination must be CUDA",
        })
    );
    let source = source.with_requires_grad(true);
    assert_eq!(
        source.try_copy_cpu_to_cuda(Device::Cuda(0)).err(),
        Some(TensorError::UnsupportedCudaTransfer {
            reason: "requires_grad is true",
        })
    );
}

#[test]
fn native_nonzero_cpu_cuda_roundtrips_preserve_layout_and_lifetimes() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native H2D tests: no CUDA runtime/device");
        return;
    }
    let base = Tensor::from_vec(
        (0..60_i16).map(|i| f32::from(i) * 0.25 - 7.0).collect(),
        [3, 4, 5],
    )
    .unwrap();
    let inputs = [
        Tensor::from_vec(vec![-3.25], []).unwrap(),
        Tensor::zeros([3, 0, 2]).unwrap().transpose(0, 2).unwrap(),
        base.clone(),
        base.transpose(0, 2).unwrap(),
        base.select_dimension(2, 2)
            .unwrap()
            .transpose(0, 1)
            .unwrap(),
    ];
    for source in inputs {
        let expected = source.try_clone().unwrap();
        let copied = source.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap();
        assert_eq!(copied.device(), Device::Cuda(0));
        assert_eq!(copied.shape(), source.shape());
        assert_eq!(copied.stride(), expected.stride());
        assert_eq!(copied.dtype(), source.dtype());
        assert_eq!(copied.storage_offset(), 0);
        assert!(!copied.requires_grad());
        assert!(matches!(
            copied.try_copy_cpu_to_cuda(Device::Cuda(0)),
            Err(TensorError::UnsupportedCudaTransfer { .. })
        ));
        drop(source);
        assert_eq!(copied.try_copy_cuda_to_cpu().unwrap(), expected);
    }
    for source in [Tensor::zeros([0]).unwrap(), Tensor::zeros([1]).unwrap()] {
        for device in [Device::Cuda(cuda::device_count()), Device::Cuda(usize::MAX)] {
            assert!(matches!(
                source.try_copy_cpu_to_cuda(device),
                Err(TensorError::CudaRuntimeError { .. })
            ));
        }
    }
}
