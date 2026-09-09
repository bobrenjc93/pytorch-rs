//! Native backend contracts: these tests run without Python bindings.
use pytorch_rs::{Device, Tensor, TensorError, cuda};

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
