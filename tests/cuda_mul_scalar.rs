//! Public Rust scalar multiplication uses native device storage without Python.
use pytorch_rs::{Device, Tensor, TensorError, cuda};

#[test]
fn generated_scalar_multiplication_shapes_views_and_boundaries() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA scalar multiplication: no CUDA device/runtime");
        return;
    }
    for n in [0_usize, 1, 255, 256, 257, 65_539, 1_048_589] {
        let values: Vec<_> = (0..n)
            .map(|i| f32::from(i16::try_from(i % 997).unwrap()) / 31.0 - 17.0)
            .collect();
        let x = Tensor::from_vec(values.clone(), [n])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        for scalar in [1.375_f32, -2.719, 0.0, -0.0, f32::INFINITY, f32::NAN] {
            let result = x.mul_scalar(scalar).unwrap();
            assert_eq!(result.device(), Device::Cuda(0));
            assert_eq!(result.shape(), x.shape());
            assert_eq!(result.storage_offset(), 0);
            assert!(!result.requires_grad());
            let actual = result.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
            for (&input, output) in values.iter().zip(actual) {
                let expected = input * scalar;
                if expected.is_nan() {
                    assert!(output.is_nan());
                } else {
                    assert_eq!(output.to_bits(), expected.to_bits());
                }
            }
        }
        assert_eq!(
            x.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            values
        );
    }
    let base = Tensor::from_vec(vec![7.0, -1.25, 3.5, -2.0], [2, 2])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let scalar = base
        .select_dimension(0, 1)
        .unwrap()
        .select_dimension(0, 1)
        .unwrap();
    assert_eq!(
        scalar
            .mul_scalar(1.5)
            .unwrap()
            .try_copy_cuda_to_cpu()
            .unwrap()
            .try_to_vec()
            .unwrap(),
        [-3.0]
    );
    assert!(matches!(
        base.transpose(0, 1).unwrap().mul_scalar(1.0),
        Err(TensorError::UnsupportedCudaScalarMultiplication { .. })
    ));
    assert!(base.mul(&base).is_err());
    assert!(base.try_with_requires_grad(true).is_err());
}
