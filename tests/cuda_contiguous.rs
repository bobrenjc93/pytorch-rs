//! Public native packing and reshape contracts without Python.
use pytorch_rs::{DType, Device, MemoryFormat, Tensor, TensorError, cuda};

#[test]
fn cuda_contiguous_and_reshape_layout_contract() {
    if cuda::device_count() == 0 {
        eprintln!("skipping CUDA contiguous: no CUDA runtime/device");
        return;
    }
    for (rows, columns) in [(2, 7), (17, 31), (257, 263)] {
        let values: Vec<_> = (0..rows * columns)
            .map(|i| f32::from(u16::try_from(i % 997).unwrap()))
            .collect();
        let base = Tensor::from_vec(values.clone(), [rows, columns])
            .unwrap()
            .try_copy_cpu_to_cuda(Device::Cuda(0))
            .unwrap();
        let view = base.transpose(0, 1).unwrap();
        let packed = view.try_contiguous(MemoryFormat::Contiguous).unwrap();
        let reshape = view.reshape([-1]).unwrap();
        let expected: Vec<_> = (0..columns)
            .flat_map(|c| {
                let values = &values;
                (0..rows).map(move |r| values[r * columns + c])
            })
            .collect();
        assert_eq!(packed.shape(), [columns, rows]);
        assert_eq!(packed.stride(), [rows, 1]);
        assert_eq!(packed.storage_offset(), 0);
        assert_eq!(packed.device(), Device::Cuda(0));
        assert_eq!(packed.dtype(), DType::Float32);
        assert!(!packed.requires_grad());
        assert_ne!(packed.data_ptr(), view.data_ptr());
        assert_ne!(reshape.data_ptr(), view.data_ptr());
        assert_ne!(reshape.data_ptr(), packed.data_ptr());
        assert_eq!(
            base.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            values
        );
        let alias = base.try_contiguous(MemoryFormat::Contiguous).unwrap();
        assert!(alias.is_set_to(&base));
        assert!(
            view.reshape([
                i64::try_from(columns).unwrap(),
                1,
                i64::try_from(rows).unwrap()
            ])
            .is_ok()
        );
        assert!(matches!(
            view.try_clone(),
            Err(TensorError::UnsupportedCudaTransfer { .. })
        ));
        drop(base);
        drop(view);
        assert_eq!(
            packed.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
            expected
        );
        assert_eq!(
            reshape
                .try_copy_cuda_to_cpu()
                .unwrap()
                .try_to_vec()
                .unwrap(),
            expected
        );
    }
}
