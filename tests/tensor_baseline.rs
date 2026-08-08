use pytorch_rs::{Tensor, TensorError};
use std::mem::size_of;

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
fn full_handles_scalar_empty_and_multidimensional_shapes() {
    let scalar = Tensor::full([], -2.5).unwrap();
    assert!(scalar.shape().is_empty());
    assert_eq!(scalar.numel(), 1);
    assert!((scalar.item().unwrap() + 2.5).abs() < f32::EPSILON);

    let empty = Tensor::full([2, 0, 3], 7.0).unwrap();
    assert_eq!(empty.shape(), [2, 0, 3]);
    assert_eq!(empty.numel(), 0);
    assert!(empty.as_slice().is_empty());

    let matrix = Tensor::full([2, 3], 1.25).unwrap();
    assert_eq!(matrix.shape(), [2, 3]);
    assert_eq!(matrix.as_slice(), [1.25; 6]);
}

#[test]
fn full_preserves_non_finite_values() {
    assert!(
        Tensor::full([2], f32::NAN)
            .unwrap()
            .as_slice()
            .iter()
            .all(|value| value.is_nan())
    );
    assert_eq!(
        Tensor::full([2], f32::INFINITY).unwrap().as_slice(),
        [f32::INFINITY; 2]
    );
    assert_eq!(
        Tensor::full([2], f32::NEG_INFINITY).unwrap().as_slice(),
        [f32::NEG_INFINITY; 2]
    );
}

#[test]
fn full_preserves_signed_zero() {
    let positive = Tensor::full([2], 0.0).unwrap();
    let negative = Tensor::full([2], -0.0).unwrap();

    assert!(positive.as_slice().iter().all(|value| value.to_bits() == 0));
    assert!(
        negative
            .as_slice()
            .iter()
            .all(|value| value.to_bits() == (-0.0_f32).to_bits())
    );
}

#[test]
fn full_rejects_storage_capacity_overflow_without_allocating() {
    let elements = isize::MAX.unsigned_abs() / size_of::<f32>() + 1;
    assert_eq!(
        Tensor::full([elements], 1.0),
        Err(TensorError::StorageCapacityOverflow { elements })
    );
}

#[test]
fn full_rejects_empty_shapes_with_unrepresentable_contiguous_strides() {
    let large = 1_usize << 62;
    for shape in [
        vec![0, large, 2],
        vec![2, 0, large, 2],
        vec![1, large, 2, 0],
    ] {
        assert_eq!(
            Tensor::full(shape, 1.0),
            Err(TensorError::StrideCalculationOverflow)
        );
    }
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
fn binary_arithmetic_broadcasts_mixed_ranks_and_singleton_dimensions() {
    let left = Tensor::from_vec(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [2, 1, 3]).unwrap();
    let right = Tensor::from_vec(vec![1.0, 2.0, 4.0], [3, 1]).unwrap();

    let add = left.add(&right).unwrap();
    assert_eq!(add.shape(), [2, 3, 3]);
    assert_eq!(
        add.as_slice(),
        [
            2.0, 3.0, 5.0, 3.0, 4.0, 6.0, 5.0, 6.0, 8.0, 9.0, 17.0, 33.0, 10.0, 18.0, 34.0, 12.0,
            20.0, 36.0,
        ]
    );
    assert_eq!(
        left.sub(&right).unwrap().as_slice(),
        [
            0.0, 1.0, 3.0, -1.0, 0.0, 2.0, -3.0, -2.0, 0.0, 7.0, 15.0, 31.0, 6.0, 14.0, 30.0, 4.0,
            12.0, 28.0,
        ]
    );
    assert_eq!(
        left.mul(&right).unwrap().as_slice(),
        [
            1.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 8.0, 16.0, 8.0, 16.0, 32.0, 16.0, 32.0, 64.0, 32.0,
            64.0, 128.0,
        ]
    );
    assert_eq!(
        left.div(&right).unwrap().as_slice(),
        [
            1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 8.0, 16.0, 32.0, 4.0, 8.0, 16.0, 2.0,
            4.0, 8.0,
        ]
    );
}

#[test]
fn binary_arithmetic_broadcasts_rank_zero_and_zero_sized_tensors() {
    let scalar = Tensor::from_vec(vec![2.0], []).unwrap();
    let matrix = Tensor::from_vec(vec![1.0, 3.0, 5.0, 7.0], [2, 2]).unwrap();
    assert_eq!(
        matrix.add(&scalar).unwrap().as_slice(),
        [3.0, 5.0, 7.0, 9.0]
    );
    assert_eq!(
        scalar.sub(&matrix).unwrap().as_slice(),
        [1.0, -1.0, -3.0, -5.0]
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap();
    let row = Tensor::from_vec(vec![1.0, 2.0, 3.0], [1, 1, 3]).unwrap();
    for output in [
        empty.add(&row).unwrap(),
        empty.sub(&row).unwrap(),
        empty.mul(&row).unwrap(),
        empty.div(&row).unwrap(),
    ] {
        assert_eq!(output.shape(), [2, 0, 3]);
        assert!(output.as_slice().is_empty());
    }

    let no_values = Tensor::zeros([0]).unwrap();
    assert_eq!(
        no_values.add(&Tensor::ones([1]).unwrap()).unwrap().shape(),
        [0]
    );
    assert!(matches!(
        no_values.add(&Tensor::ones([2]).unwrap()),
        Err(TensorError::ShapeMismatch { .. })
    ));

    let large_empty = Tensor::from_vec(Vec::new(), [isize::MAX.unsigned_abs(), 0, 1]).unwrap();
    let output = large_empty.add(&scalar).unwrap();
    assert_eq!(output.shape(), [isize::MAX.unsigned_abs(), 0, 1]);
    assert_eq!(output.numel(), 0);
}

#[test]
fn empty_broadcast_rejects_unrepresentable_result_strides() {
    let large = isize::MAX.unsigned_abs() / 2 + 1;
    let left = Tensor::from_vec(Vec::new(), [0, large, 1]).unwrap();
    let right = Tensor::from_vec(vec![1.0, 2.0], [1, 1, 2]).unwrap();

    for result in [
        left.add(&right),
        left.sub(&right),
        left.mul(&right),
        left.div(&right),
    ] {
        assert_eq!(result, Err(TensorError::StrideCalculationOverflow));
    }
}

#[test]
fn scalar_arithmetic_supports_both_operand_orders_and_signed_zero() {
    let tensor = Tensor::from_vec(vec![1.0, -1.0, 0.0, -0.0], [4]).unwrap();
    assert_eq!(
        tensor.add_scalar(2.0).unwrap().as_slice(),
        [3.0, 1.0, 2.0, 2.0]
    );
    assert_eq!(
        tensor.sub_scalar(2.0).unwrap().as_slice(),
        [-1.0, -3.0, -2.0, -2.0]
    );
    assert_eq!(
        tensor.scalar_sub(2.0).unwrap().as_slice(),
        [1.0, 3.0, 2.0, 2.0]
    );
    assert_eq!(
        tensor.mul_scalar(2.0).unwrap().as_slice(),
        [2.0, -2.0, 0.0, -0.0]
    );

    let divided = tensor.div_scalar(-0.0).unwrap();
    assert_eq!(divided.as_slice()[0].to_bits(), f32::NEG_INFINITY.to_bits());
    assert_eq!(divided.as_slice()[1].to_bits(), f32::INFINITY.to_bits());
    assert!(divided.as_slice()[2].is_nan());
    assert!(divided.as_slice()[3].is_nan());

    let reverse = tensor.scalar_div(-0.0).unwrap();
    assert_eq!(reverse.as_slice()[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(reverse.as_slice()[1].to_bits(), 0.0_f32.to_bits());
    assert!(reverse.as_slice()[2].is_nan());
    assert!(reverse.as_slice()[3].is_nan());
}

#[test]
fn reflected_scalar_division_uses_float32_reciprocal_multiplication() {
    let ordinary_denominator = Tensor::from_vec(vec![f32::from_bits(0xc27c_80a7)], [1]).unwrap();
    let ordinary = ordinary_denominator
        .scalar_div(f32::from_bits(0xc25f_b64c))
        .unwrap();
    assert_eq!(ordinary.as_slice()[0].to_bits(), 0x3f62_cf8f);

    let subnormal = Tensor::from_vec(vec![1.0e-39_f32], [1]).unwrap();
    assert!(subnormal.scalar_div(1.0e-38).unwrap().as_slice()[0].is_infinite());
    assert!(subnormal.scalar_div(0.0).unwrap().as_slice()[0].is_nan());
}

#[test]
fn subtraction_and_division_support_scalar_empty_and_multidimensional_tensors() {
    let scalar_left = Tensor::from_vec(vec![7.0], []).unwrap();
    let scalar_right = Tensor::from_vec(vec![2.0], []).unwrap();
    assert_eq!(
        scalar_left
            .sub(&scalar_right)
            .unwrap()
            .item()
            .unwrap()
            .to_bits(),
        5.0_f32.to_bits()
    );
    assert_eq!(
        scalar_left
            .div(&scalar_right)
            .unwrap()
            .item()
            .unwrap()
            .to_bits(),
        3.5_f32.to_bits()
    );

    let empty_left = Tensor::zeros([2, 0, 3]).unwrap();
    let empty_right = Tensor::ones([2, 0, 3]).unwrap();
    for output in [
        empty_left.sub(&empty_right).unwrap(),
        empty_left.div(&empty_right).unwrap(),
    ] {
        assert_eq!(output.shape(), [2, 0, 3]);
        assert!(output.as_slice().is_empty());
    }

    let left = Tensor::from_vec(vec![12.0, -8.0, 3.0, 0.5], [2, 1, 2]).unwrap();
    let right = Tensor::from_vec(vec![3.0, 2.0, -1.5, 0.25], [2, 1, 2]).unwrap();
    assert_eq!(
        left.sub(&right).unwrap().as_slice(),
        [9.0, -10.0, 4.5, 0.25]
    );
    assert_eq!(left.div(&right).unwrap().as_slice(), [4.0, -4.0, -2.0, 2.0]);
}

#[test]
fn subtraction_matches_pytorch_non_finite_and_signed_zero_semantics() {
    let left = Tensor::from_vec(
        vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
        ],
        [7],
    )
    .unwrap();
    let right = Tensor::from_vec(
        vec![
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            0.0,
            -0.0,
        ],
        [7],
    )
    .unwrap();
    let output = left.sub(&right).unwrap();

    assert!(output.as_slice()[..3].iter().all(|value| value.is_nan()));
    assert_eq!(output.as_slice()[3].to_bits(), f32::INFINITY.to_bits());
    assert_eq!(output.as_slice()[4].to_bits(), f32::NEG_INFINITY.to_bits());
    assert_eq!(output.as_slice()[5].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(output.as_slice()[6].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn division_matches_pytorch_zero_non_finite_and_signed_zero_semantics() {
    let left = Tensor::from_vec(
        vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1.0,
            -1.0,
            1.0,
            -1.0,
            0.0,
            -0.0,
            0.0,
            -0.0,
        ],
        [13],
    )
    .unwrap();
    let right = Tensor::from_vec(
        vec![
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0,
            2.0,
            0.0,
            0.0,
            -0.0,
            -0.0,
            2.0,
            2.0,
            -2.0,
            -2.0,
        ],
        [13],
    )
    .unwrap();
    let output = left.div(&right).unwrap();

    assert!(output.as_slice()[..3].iter().all(|value| value.is_nan()));
    assert_eq!(output.as_slice()[3].to_bits(), f32::INFINITY.to_bits());
    assert_eq!(output.as_slice()[4].to_bits(), f32::NEG_INFINITY.to_bits());
    assert_eq!(output.as_slice()[5].to_bits(), f32::INFINITY.to_bits());
    assert_eq!(output.as_slice()[6].to_bits(), f32::NEG_INFINITY.to_bits());
    assert_eq!(output.as_slice()[7].to_bits(), f32::NEG_INFINITY.to_bits());
    assert_eq!(output.as_slice()[8].to_bits(), f32::INFINITY.to_bits());
    assert_eq!(output.as_slice()[9].to_bits(), 0.0_f32.to_bits());
    assert_eq!(output.as_slice()[10].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(output.as_slice()[11].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(output.as_slice()[12].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn subtraction_and_division_reject_incompatible_shapes() {
    let left = Tensor::zeros([2, 2]).unwrap();
    let right = Tensor::zeros([4]).unwrap();
    let expected = TensorError::ShapeMismatch {
        left: vec![2, 2],
        right: vec![4],
    };

    assert_eq!(
        expected.to_string(),
        "The size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 1"
    );

    assert_eq!(left.sub(&right), Err(expected.clone()));
    assert_eq!(left.div(&right), Err(expected));
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
