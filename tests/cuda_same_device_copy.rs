//! Public native CUDA vector cloning without Python or host staging.
use pytorch_rs::{Device, MemoryFormat, Tensor, TensorError, cuda};

#[test]
fn generated_vector_copies_preserve_values_layout_and_live_storage() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA copies: no CUDA runtime/device");
        return;
    }
    let mut seed = 0x5eed_u32;
    let mut lengths = vec![0, 1, 255, 256, 257, 65_539];
    for _ in 0..16 {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        lengths.push(usize::try_from(seed % 8192).unwrap());
    }
    for n in lengths {
        let values: Vec<_> = (0..2 * n)
            .map(|i| f32::from(i16::try_from(i % 997).unwrap()) * 0.25 - 7.0)
            .collect();
        let base = Tensor::from_vec(values.clone(), [2, n])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let source = base.select_dimension(0, 1).unwrap();
        let original_pointer = source.data_ptr();
        let original_offset = source.storage_offset();
        let output = source.try_clone().unwrap();
        let second = source
            .try_clone_with_memory_format(MemoryFormat::Preserve)
            .unwrap();
        assert_eq!(output.shape(), [n]);
        assert_eq!(output.stride(), [1]);
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.device(), Device::Cuda(0));
        assert!(!output.requires_grad());
        assert_eq!(source.data_ptr(), original_pointer);
        assert_eq!(source.storage_offset(), original_offset);
        if n != 0 {
            assert_ne!(output.data_ptr(), source.data_ptr());
            assert_ne!(output.data_ptr(), second.data_ptr());
        }
        assert_eq!(
            base.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            values
        );
        drop(base);
        drop(source);
        let mut pointers = std::collections::HashSet::new();
        for _ in 0..40 {
            let recycled = output.try_clone().unwrap();
            pointers.insert(recycled.data_ptr());
            assert_eq!(
                recycled
                    .try_copy_cuda_to_cpu()
                    .unwrap()
                    .try_to_vec()
                    .unwrap(),
                values[n..]
            );
        }
        assert!(pointers.len() < 40);
        assert_eq!(
            second.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            values[n..]
        );
    }
}

#[test]
fn vector_copy_keeps_layout_and_autograd_boundaries() {
    if cuda::device_count() == 0 {
        eprintln!("skipping native CUDA copy boundaries: no CUDA runtime/device");
        return;
    }
    let matrix = Tensor::zeros([3, 4])
        .unwrap()
        .try_copy_cpu_to_cuda(Device::Cuda(0))
        .unwrap();
    let vector = matrix.select_dimension(0, 1).unwrap();
    for source in [
        &matrix,
        &matrix.select_dimension(1, 1).unwrap(),
        &vector.select_dimension(0, 1).unwrap(),
    ] {
        assert!(matches!(
            source.try_clone(),
            Err(TensorError::UnsupportedCudaTransfer { .. })
        ));
    }
    for format in [
        MemoryFormat::Contiguous,
        MemoryFormat::ChannelsLast,
        MemoryFormat::ChannelsLast3d,
    ] {
        assert!(matches!(
            vector.try_clone_with_memory_format(format),
            Err(TensorError::UnsupportedCudaTransfer { .. })
        ));
    }
    assert!(vector.try_with_requires_grad(true).is_err());
}
