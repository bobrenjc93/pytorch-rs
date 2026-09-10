//! Public Rust row sums execute on native CUDA storage without Python.
use pytorch_rs::{DType, Device, Tensor, TensorError, cuda};

#[test]
fn generated_rectangles_offsets_empty_axes_and_storage() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA row sums: no CUDA runtime/device");
        return;
    }
    let shapes = [
        [0, 0],
        [0, 17],
        [7, 0],
        [1, 1],
        [3, 31],
        [9, 33],
        [17, 257],
        [2, 65_539],
        [32_777, 1],
    ]
    .into_iter()
    .chain((1..=18).map(|i| [i * 3 + 1, i * 17 + 3]));
    for [rows, columns] in shapes {
        // Selecting the second matrix tests contiguous, nonzero storage offsets.
        let values: Vec<_> = (0..2 * rows * columns)
            .map(|i| f32::from(i16::try_from(i % 127).unwrap()) / 16.0 - 4.0)
            .collect();
        let cpu = Tensor::from_vec(values, [2, rows, columns]).unwrap();
        let base = cpu.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap();
        let input = base.select_dimension(0, 1).unwrap();
        let reference = cpu.select_dimension(0, 1).unwrap();
        for keepdim in [false, true] {
            let output = input.sum_rank_two_dimension(1, keepdim).unwrap();
            let expected = reference.sum_rank_two_dimension(1, keepdim).unwrap();
            assert_eq!(output.shape(), expected.shape());
            assert_eq!(output.stride(), expected.stride());
            assert_eq!(output.storage_offset(), 0);
            assert_eq!(output.dtype(), DType::Float32);
            assert_eq!(output.device(), Device::Cuda(0));
            assert!(!output.requires_grad());
            assert!(output.is_contiguous());
            let second = input.sum_rank_two_dimension(1, keepdim).unwrap();
            if rows != 0 {
                assert_ne!(output.data_ptr(), input.data_ptr());
                assert_ne!(output.data_ptr(), second.data_ptr());
            }
            assert_eq!(output.try_copy_cuda_to_cpu().unwrap(), expected);
        }
        assert_eq!(base.try_copy_cuda_to_cpu().unwrap(), cpu);
        assert!(matches!(
            input.sum_rank_two_dimension(0, false),
            Err(TensorError::UnsupportedCudaSum { .. })
        ));
        assert!(input.try_sum().is_err());
        assert!(input.mean_rank_two_dimension(1, false).is_err());
        assert!(input.try_with_requires_grad(true).is_err());
    }
    let input = Tensor::cuda_zeros_float32([3, 5], Device::Cuda(0)).unwrap();
    assert!(matches!(
        input
            .transpose(0, 1)
            .unwrap()
            .sum_rank_two_dimension(1, false),
        Err(TensorError::UnsupportedCudaSum { .. })
    ));
    assert!(input.sum_rank_two_dimension(2, false).is_err());
    assert!(
        input
            .select_dimension(0, 0)
            .unwrap()
            .sum_rank_two_dimension(1, false)
            .is_err()
    );
}

#[test]
fn cancellation_nonfinite_values_and_output_lifetime() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA row sum numerics: no CUDA runtime/device");
        return;
    }
    let values = vec![
        8192.,
        -8192.,
        0.125,
        -0.0625,
        f32::INFINITY,
        1.,
        2.,
        3.,
        f32::NEG_INFINITY,
        1.,
        2.,
        3.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.,
        2.,
        f32::NAN,
        1.,
        2.,
        3.,
        -0.,
        -0.,
        -0.,
        -0.,
    ];
    let source = Tensor::from_vec(values, [6, 4])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let output = source.sum_rank_two_dimension(1, false).unwrap();
    drop(source);
    let replacement = Tensor::cuda_zeros_float32([6, 4], Device::Cuda(0)).unwrap();
    let actual = output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
    assert_eq!(actual[..3], [0.0625, f32::INFINITY, f32::NEG_INFINITY]);
    assert!(actual[3].is_nan());
    assert!(actual[4].is_nan());
    assert_eq!(actual[5].to_bits(), 0);
    assert_eq!(
        replacement
            .try_copy_cuda_to_cpu()
            .unwrap()
            .try_to_vec()
            .unwrap(),
        vec![0.; 24]
    );
}
