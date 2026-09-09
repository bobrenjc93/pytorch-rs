//! Ordinary native CUDA addition, with no Python or installed `PyTorch` involved.
use pytorch_rs::{Device, Tensor, TensorError, cuda};

fn available() -> bool {
    if cuda::device_count() == 0 {
        eprintln!("skipping CUDA addition: no CUDA runtime/device");
        return false;
    }
    true
}

#[test]
fn shapes_offsets_aliases_and_completion() {
    if !available() {
        return;
    }
    // Deterministic, nonzero pseudo-random data independent of kernel geometry.
    let mut state = 0x6e3b_8241_u32;
    for shape in [
        vec![],
        vec![0],
        vec![2, 0, 5],
        vec![1],
        vec![259],
        vec![7, 13, 17],
        vec![1_048_589],
    ] {
        let n = shape.iter().product::<usize>();
        let values: Vec<f32> = (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                f32::from((state % 8192) as u16) / 128.0 - 32.0
            })
            .collect();
        let cpu = Tensor::from_vec([vec![99.0; n], values.clone()].concat(), [2, n]).unwrap();
        let base = cpu.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap();
        let dimensions: Vec<i64> = shape.iter().map(|&x| i64::try_from(x).unwrap()).collect();
        let x = base
            .select_dimension(0, 1)
            .unwrap()
            .reshape(&dimensions)
            .unwrap();
        let y = x.detach().unwrap();
        drop(base);
        let out = x.add(&y).unwrap();
        assert_eq!(out.shape(), shape);
        assert_eq!(out.storage_offset(), 0);
        assert!(out.is_contiguous());
        assert_eq!(out.device(), Device::Cuda(0));
        assert!(!out.requires_grad());
        drop(x);
        drop(y);
        // Chained launches, immediate drops and cached allocation reuse.
        let out = out.add(&out).unwrap();
        let zeros = Tensor::cuda_zeros_float32(vec![n], Device::Cuda(0)).unwrap();
        let actual = out.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap();
        assert_eq!(actual, values.iter().map(|x| x * 4.0).collect::<Vec<_>>());
        assert_eq!(
            zeros.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![0.0; n]
        );
    }
}

#[test]
fn ieee_edges_and_rejections() {
    if !available() {
        return;
    }
    let a = [
        0.0_f32,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        16_777_216.0,
    ];
    let b = [
        -0.0_f32,
        -0.0,
        f32::NEG_INFINITY,
        -2.0,
        3.0,
        f32::MAX,
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0,
    ];
    let upload = |data: &[f32]| {
        Tensor::from_vec(data.to_vec(), [data.len()])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap()
    };
    let x = upload(&a);
    let y = upload(&b);
    let actual = x
        .add(&y)
        .unwrap()
        .try_copy_cuda_to_cpu()
        .unwrap()
        .try_to_vec()
        .unwrap();
    for ((x, y), z) in a.iter().zip(&b).zip(actual) {
        let expected = x + y;
        if expected.is_nan() {
            assert!(z.is_nan());
        } else {
            assert_eq!(z.to_bits(), expected.to_bits());
        }
    }
    let cpu = Tensor::zeros([9]).unwrap();
    for result in [
        x.add(&cpu),
        cpu.add(&x),
        x.add(&upload(&[1.0])),
        x.add(&x.reshape([3, 3]).unwrap()),
    ] {
        assert!(matches!(
            result,
            Err(TensorError::UnsupportedCudaAddition { .. })
        ));
    }
    let transposed = x.reshape([3, 3]).unwrap().transpose(0, 1).unwrap();
    assert!(matches!(
        transposed.add(&transposed),
        Err(TensorError::UnsupportedCudaAddition { .. })
    ));
    assert!(x.add_scalar(1.0).is_err());
    assert!(x.mul(&y).is_err());
    assert!(x.try_with_requires_grad(true).is_err());
}

#[test]
fn cross_thread_output_ownership_and_reuse() {
    if !available() {
        return;
    }
    let handles: Vec<_> = (0..4_u16)
        .map(|worker| {
            std::thread::spawn(move || {
                let values = vec![f32::from(worker); 4099];
                let x = Tensor::from_vec(values, [4099])
                    .unwrap()
                    .try_copy_cpu_to_cuda(Device::Cuda(0))
                    .unwrap();
                for _ in 0..20 {
                    let result = x.add(&x).unwrap();
                    assert_eq!(
                        result.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
                        vec![f32::from(worker) * 2.0; 4099]
                    );
                }
                x.add(&x).unwrap()
            })
        })
        .collect();
    for (worker, handle) in handles.into_iter().enumerate() {
        let output = handle.join().unwrap();
        // A fresh thread has not initialized a CUDA context. Force it to use
        // a cached output allocation and consume another thread's tensors.
        let output = std::thread::spawn(move || output.add(&output).unwrap())
            .join()
            .unwrap();
        assert_eq!(
            output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            vec![f32::from(u16::try_from(worker).unwrap()) * 4.0; 4099]
        );
    }
}
