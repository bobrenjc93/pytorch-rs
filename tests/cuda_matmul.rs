//! Eager native CUDA matrix products, independent of Python and the scoring corpus.
use pytorch_rs::{DType, Device, Tensor, TensorError, cuda};

fn available() -> bool {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA matmul: no CUDA runtime/device");
        false
    } else {
        true
    }
}

#[test]
fn generated_rectangles_offsets_metadata_and_lifetimes() {
    if !available() {
        return;
    }
    let shapes = [
        [0, 0, 0],
        [0, 7, 5],
        [3, 7, 0],
        [5, 0, 9],
        [1, 1, 1],
        [3, 257, 17],
        [1025, 3, 1027],
    ]
    .into_iter()
    .chain((1..=16).map(|i| [i * 3, i * 7 + 1, i * 2 + 1]));
    for [m, k, n] in shapes {
        let make = |rows, columns, salt| {
            let values = (0..2 * rows * columns)
                .map(|i| f32::from(i16::try_from((i * 13 + salt) % 127).unwrap()) / 16.0 - 4.0)
                .collect();
            Tensor::from_vec(values, [2, rows, columns]).unwrap()
        };
        // Binary fractions make the independent CPU oracle exact for these sizes.
        let left_cpu = make(m, k, 3);
        let right_cpu = make(k, n, 17);
        let left_base = left_cpu.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap();
        let right_base = right_cpu.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap();
        let left = left_base.select_dimension(0, 1).unwrap();
        let right = right_base.select_dimension(0, 1).unwrap();
        let expected = left_cpu
            .select_dimension(0, 1)
            .unwrap()
            .matmul(&right_cpu.select_dimension(0, 1).unwrap())
            .unwrap();
        let output = left.matmul(&right).unwrap();
        let second = left.matmul(&right).unwrap();
        assert_eq!(output.shape(), [m, n]);
        assert_eq!(output.stride(), [n.max(1), 1]);
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.dtype(), DType::Float32);
        assert_eq!(output.device(), Device::Cuda(0));
        assert!(output.is_contiguous());
        assert!(!output.requires_grad());
        if output.numel() != 0 {
            assert_ne!(output.data_ptr(), left.data_ptr());
            assert_ne!(output.data_ptr(), right.data_ptr());
            assert_ne!(output.data_ptr(), second.data_ptr());
        }
        assert_eq!(left_base.try_copy_cuda_to_cpu().unwrap(), left_cpu);
        assert_eq!(right_base.try_copy_cuda_to_cpu().unwrap(), right_cpu);
        drop((left, right, left_base, right_base, second));
        // Keep a view alive after dropping the original result; churn the cache.
        let retained = output.detach().unwrap();
        drop(output);
        for _ in 0..4 {
            drop(Tensor::cuda_zeros_float32([m, n], Device::Cuda(0)).unwrap());
        }
        assert_eq!(retained.try_copy_cuda_to_cpu().unwrap(), expected);
    }
}

#[test]
fn invalid_inputs_remain_rejected() {
    if !available() {
        return;
    }
    let x = Tensor::cuda_zeros_float32([3, 5], Device::Cuda(0)).unwrap();
    let y = Tensor::cuda_zeros_float32([5, 3], Device::Cuda(0)).unwrap();
    for (left, right) in [
        (&x, Tensor::zeros([5, 3]).unwrap()),
        (&x, x.transpose(0, 1).unwrap()),
        (&y.transpose(0, 1).unwrap(), y.detach().unwrap()),
        (&x, y.reshape([5, 1, 3]).unwrap()),
        (&x, y.select_dimension(1, 0).unwrap()),
    ] {
        assert!(matches!(
            left.matmul(&right),
            Err(TensorError::UnsupportedCudaMatmul { .. })
        ));
    }
    assert!(matches!(
        Tensor::zeros([3, 5]).unwrap().matmul(&y),
        Err(TensorError::UnsupportedCudaMatmul { .. })
    ));
    assert!(matches!(
        x.matmul(&x),
        Err(TensorError::MatmulInnerDimensionMismatch { .. })
    ));
    assert!(x.detach().unwrap().try_with_requires_grad(true).is_err());
    assert!(x.matmul(&y).unwrap().backward().is_err());
    if cuda::device_count() >= 2 {
        let other = Tensor::cuda_zeros_float32([5, 3], Device::Cuda(1)).unwrap();
        assert!(matches!(
            x.matmul(&other),
            Err(TensorError::UnsupportedCudaMatmul { .. })
        ));
    }
}
