use pytorch_rs::{Tensor, TensorError};

#[test]
fn construction_validates_shape() {
    let error = Tensor::from_vec(vec![1.0, 2.0], [3]).unwrap_err();
    assert_eq!(
        error,
        TensorError::ShapeDataMismatch {
            shape: vec![3],
            elements: 2,
        }
    );
}

#[test]
fn empty_dimensions_have_zero_elements() {
    let tensor = Tensor::zeros([3, 0, 4]).unwrap();
    assert_eq!(tensor.shape(), [3, 0, 4]);
    assert_eq!(tensor.numel(), 0);
}

#[test]
fn elementwise_operations_preserve_shape() {
    let left = Tensor::from_vec(vec![-1.0, 2.0, 3.0, -4.0], [2, 2]).unwrap();
    let right = Tensor::ones([2, 2]).unwrap();
    assert_eq!(left.add(&right).unwrap().as_slice(), [0.0, 3.0, 4.0, -3.0]);
    assert_eq!(left.mul(&right).unwrap(), left);
    assert_eq!(left.relu().as_slice(), [0.0, 2.0, 3.0, 0.0]);
}

#[test]
fn reductions_handle_ordinary_and_empty_tensors() {
    let sum = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3])
        .unwrap()
        .sum()
        .item()
        .unwrap();
    assert!((sum - 6.0).abs() < f32::EPSILON);
    assert!(Tensor::zeros([0]).unwrap().sum().item().unwrap().abs() < f32::EPSILON);
}

#[test]
fn matrix_multiplication_matches_known_result() {
    let left = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let right = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]).unwrap();
    let output = left.matmul(&right).unwrap();
    assert_eq!(output.shape(), [2, 2]);
    assert_eq!(output.as_slice(), [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matrix_multiplication_rejects_incompatible_shapes() {
    let left = Tensor::zeros([2, 3]).unwrap();
    let right = Tensor::zeros([4, 2]).unwrap();
    assert!(matches!(
        left.matmul(&right),
        Err(TensorError::MatmulInnerDimensionMismatch { .. })
    ));
}
