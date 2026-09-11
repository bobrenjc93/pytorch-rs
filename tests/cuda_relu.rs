//! Rectified linear activation uses native device storage with the same rank and offset contract as negation.
use pytorch_rs::{Device, Tensor, TensorError, cuda};

#[test]
fn cuda_relu_bits_shapes_offsets_and_boundaries() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA ReLU: no CUDA runtime/device");
        return;
    }
    let bits: [u32; 22] = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3fa0_0000,
        0xbfa0_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0xffc1_2345,
        0x7f80_0001,
        0xff80_0001,
        0x7fa1_2345,
        0xffa1_2345,
        0x7fff_ffff,
        0xffff_ffff,
    ];
    for shape in [
        vec![],
        vec![0],
        vec![2, 0, 3],
        vec![1],
        vec![257],
        vec![65_539],
        vec![1_048_589],
        vec![2, 3, 1, 5, 2],
    ] {
        let count: usize = shape.iter().product();
        let data: Vec<_> = (0..count)
            .map(|i| f32::from_bits(bits[i % bits.len()]))
            .collect();
        let x = Tensor::from_vec(data.clone(), shape.clone())
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let out = x.relu().unwrap();
        assert_eq!(out.device(), Device::Cuda(0));
        assert_eq!(out.shape(), shape);
        assert_eq!(out.storage_offset(), 0);
        assert!(out.is_contiguous());
        assert!(!out.requires_grad());
        let actual = out.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
        for (a, input) in actual.iter().zip(&data) {
            let expected = if *input <= 0. { 0 } else { input.to_bits() };
            assert_eq!(a.to_bits(), expected);
        }
        let unchanged = x.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
        assert_eq!(
            unchanged.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    let base = Tensor::from_vec(vec![9., -2., 3., -4., 5., -6.], [3, 2])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let row = base.select_dimension(0, 1).unwrap();
    assert_eq!(row.storage_offset(), 2);
    assert_eq!(
        row.relu()
            .unwrap()
            .try_copy_cuda_to_cpu()
            .unwrap()
            .try_to_vec()
            .unwrap(),
        [3., 0.]
    );
    assert!(matches!(
        base.transpose(0, 1).unwrap().relu(),
        Err(TensorError::UnsupportedCudaRelu { .. })
    ));
    assert!(base.try_with_requires_grad(true).is_err());
}
