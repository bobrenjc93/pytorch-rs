use std::mem::size_of;

use pytorch_rs::{DType, Scalar, Tensor, TensorDataRef, TensorError};

#[test]
fn int64_storage_is_physical_typed_and_shared_by_views() {
    let tensor = Tensor::from_i64_vec(vec![1, -2, i64::MAX, i64::MIN], [2, 2]).unwrap();
    assert_eq!(tensor.dtype(), DType::Int64);
    assert_eq!(DType::Int64.to_string(), "int64");
    assert_eq!(
        tensor.as_i64_slice(),
        Some(&[1, -2, i64::MAX, i64::MIN][..])
    );
    assert_eq!(tensor.as_f32_slice(), None);
    assert_eq!(
        tensor.data(),
        TensorDataRef::Int64(&[1, -2, i64::MAX, i64::MIN])
    );

    let view = tensor.reshape([4]).unwrap();
    assert_eq!(view.dtype(), DType::Int64);
    assert_eq!(
        view.as_i64_slice().unwrap().as_ptr(),
        tensor.as_i64_slice().unwrap().as_ptr()
    );
    assert_eq!(
        view.item(),
        Err(TensorError::ItemRequiresOneElement { elements: 4 })
    );
    assert_eq!(
        Tensor::from_i64_vec(vec![i64::MAX], [])
            .unwrap()
            .item_scalar(),
        Ok(Scalar::Int64(i64::MAX))
    );
}

#[test]
fn int64_creation_covers_constants_empties_and_capacity_checks() {
    assert_eq!(
        Tensor::full_i64([2, 2], -7).unwrap().as_i64_slice(),
        Some(&[-7; 4][..])
    );
    assert_eq!(Tensor::full_i64([2, 0, 3], 9).unwrap().shape(), [2, 0, 3]);
    assert_eq!(Tensor::full_i64([2, 0, 3], 9).unwrap().numel(), 0);

    let elements = isize::MAX.unsigned_abs() / size_of::<i64>() + 1;
    assert_eq!(
        Tensor::full_i64([elements], 1),
        Err(TensorError::StorageCapacityOverflow { elements })
    );
}

#[test]
fn dtype_promotion_and_broadcasting_are_centralized() {
    let integers = Tensor::from_i64_vec(vec![1, 2], [2, 1]).unwrap();
    let row = Tensor::from_i64_vec(vec![10, 20, 30], [1, 3]).unwrap();
    let floats = Tensor::from_vec(vec![0.5, 1.5, 2.5], [1, 3]).unwrap();

    let integer_output = integers.add(&row).unwrap();
    assert_eq!(integer_output.dtype(), DType::Int64);
    assert_eq!(
        integer_output.as_i64_slice(),
        Some(&[11, 21, 31, 12, 22, 32][..])
    );

    for output in [
        integers.add(&floats).unwrap(),
        integers.sub(&floats).unwrap(),
        integers.mul(&floats).unwrap(),
        integers.div(&row).unwrap(),
    ] {
        assert_eq!(output.dtype(), DType::Float32);
        assert_eq!(output.shape(), [2, 3]);
    }
    assert_eq!(
        integers.add(&floats).unwrap().as_f32_slice(),
        Some(&[1.5, 2.5, 3.5, 2.5, 3.5, 4.5][..])
    );

    let empty = Tensor::from_i64_vec(Vec::new(), [2, 0, 3]).unwrap();
    let empty_output = empty.add(&Tensor::full_i64([1, 1, 3], 1).unwrap()).unwrap();
    assert_eq!(empty_output.dtype(), DType::Int64);
    assert_eq!(empty_output.shape(), [2, 0, 3]);
    assert!(empty_output.as_i64_slice().unwrap().is_empty());
}

#[test]
fn integer_scalars_preserve_dtype_and_float_scalars_promote() {
    let tensor = Tensor::from_i64_vec(vec![-2, 0, 3], [3]).unwrap();
    let integer = tensor.add_typed_scalar(Scalar::Int64(4)).unwrap();
    assert_eq!(integer.dtype(), DType::Int64);
    assert_eq!(integer.as_i64_slice(), Some(&[2, 4, 7][..]));

    let float = tensor.add_typed_scalar(Scalar::Float32(0.5)).unwrap();
    assert_eq!(float.dtype(), DType::Float32);
    assert_eq!(float.as_f32_slice(), Some(&[-1.5, 0.5, 3.5][..]));

    let divided = tensor.div_typed_scalar(Scalar::Int64(2), false).unwrap();
    assert_eq!(divided.dtype(), DType::Float32);
    assert_eq!(divided.as_f32_slice(), Some(&[-1.0, 0.0, 1.5][..]));
    assert_eq!(
        tensor
            .sub_typed_scalar(Scalar::Int64(1), true)
            .unwrap()
            .as_i64_slice(),
        Some(&[3, 1, -2][..])
    );
}

#[test]
fn integer_kernels_wrap_and_preserve_dtype() {
    let extremes = Tensor::from_i64_vec(vec![i64::MAX, i64::MIN], [2]).unwrap();
    assert_eq!(
        extremes
            .add_typed_scalar(Scalar::Int64(1))
            .unwrap()
            .as_i64_slice(),
        Some(&[i64::MIN, i64::MIN + 1][..])
    );
    assert_eq!(
        extremes
            .mul_typed_scalar(Scalar::Int64(2))
            .unwrap()
            .as_i64_slice(),
        Some(&[-2, 0][..])
    );

    let sum = Tensor::from_i64_vec(vec![i64::MAX, 1], [2]).unwrap().sum();
    assert_eq!(sum.item_scalar(), Ok(Scalar::Int64(i64::MIN)));
    assert_eq!(
        Tensor::from_i64_vec(vec![-4, 0, 5], [3])
            .unwrap()
            .relu()
            .unwrap()
            .as_i64_slice(),
        Some(&[0, 0, 5][..])
    );
}

#[test]
fn integer_matmul_preserves_dtype_and_mixed_matmul_is_rejected() {
    let left = Tensor::from_i64_vec(vec![1, 2, 3, 4, 5, 6], [2, 3]).unwrap();
    let right = Tensor::from_i64_vec(vec![7, 8, 9, 10, 11, 12], [3, 2]).unwrap();
    let integer = left.matmul(&right).unwrap();
    assert_eq!(integer.dtype(), DType::Int64);
    assert_eq!(integer.as_i64_slice(), Some(&[58, 64, 139, 154][..]));

    let float_right = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [3, 2]).unwrap();
    assert_eq!(
        left.matmul(&float_right),
        Err(TensorError::MatmulDTypeMismatch {
            left: DType::Int64,
            right: DType::Float32,
        })
    );

    let incompatible_float = Tensor::ones([4, 5]).unwrap();
    assert_eq!(
        left.matmul(&incompatible_float),
        Err(TensorError::MatmulInnerDimensionMismatch {
            left: vec![2, 3],
            right: vec![4, 5],
        })
    );

    let overflowing = Tensor::from_i64_vec(vec![i64::MAX], [1, 1])
        .unwrap()
        .matmul(&Tensor::from_i64_vec(vec![2], [1, 1]).unwrap())
        .unwrap();
    assert_eq!(overflowing.as_i64_slice(), Some(&[-2][..]));
}

#[test]
fn dtype_promoting_empty_scalar_operations_use_common_dtype_strides() {
    let source = Tensor::from_i64_vec(Vec::new(), [0])
        .unwrap()
        .reshape([0, 1, 2, 1_i64 << 61])
        .unwrap();

    let added = source.add_typed_scalar(Scalar::Float32(1.0)).unwrap();
    assert_eq!(added.dtype(), DType::Float32);
    assert_eq!(added.stride(), [0, 0, 1, 2]);

    let divided = source.div_typed_scalar(Scalar::Int64(2), false).unwrap();
    assert_eq!(divided.dtype(), DType::Float32);
    assert_eq!(divided.stride(), [0, 0, 1, 2]);
}
