use pytorch_rs::{DType, Device, Tensor, TensorError};
use std::mem::size_of;

#[test]
fn native_metadata_describes_all_supported_storage_shapes() {
    assert_eq!(DType::default(), DType::Float32);
    assert_eq!(Device::default(), Device::Cpu);
    assert_eq!(DType::Float32.to_string(), "float32");
    assert_eq!(Device::Cpu.to_string(), "cpu");

    for tensor in [
        Tensor::from_vec(vec![2.5], []).unwrap(),
        Tensor::zeros([2, 0, 3]).unwrap(),
        Tensor::ones([2, 3]).unwrap(),
        Tensor::eye(2, None).unwrap(),
        Tensor::full([4], -1.25).unwrap(),
    ] {
        assert_eq!(tensor.dtype(), DType::Float32);
        assert_eq!(tensor.device(), Device::Cpu);
    }
}

#[test]
fn native_metadata_survives_views_kernels_and_reductions() {
    let source = Tensor::from_vec(vec![-1.0, 2.0, 3.0, -4.0], [2, 2]).unwrap();
    let matrix = Tensor::ones([2, 2]).unwrap();
    let outputs = [
        source.reshape([4]).unwrap(),
        source.add(&matrix).unwrap(),
        source.mul_scalar(2.0).unwrap(),
        source.relu().unwrap(),
        source.matmul(&matrix).unwrap(),
        source.sum(),
    ];

    for output in outputs {
        assert_eq!(output.dtype(), source.dtype());
        assert_eq!(output.device(), source.device());
    }
}

#[test]
fn native_metadata_introspection_covers_all_constructible_layout_kinds() {
    let ordinary = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let reshaped = ordinary.reshape([4, 6]).unwrap();
    let indexed = ordinary.index([1, 2]).unwrap();
    let cases = [
        (Tensor::from_vec(vec![2.5], []).unwrap(), 0, 1),
        (Tensor::zeros([2, 0, 3]).unwrap(), 3, 0),
        (ordinary, 3, 24),
        (reshaped, 2, 24),
        (indexed, 1, 4),
    ];

    for (tensor, dimensions, elements) in cases {
        assert_eq!(tensor.dim(), dimensions);
        assert_eq!(tensor.ndimension(), dimensions);
        assert_eq!(tensor.numel(), elements);
        assert!(tensor.is_contiguous());
    }

    let unusual_empty_layout = Tensor::zeros([0, 1]).unwrap().add_scalar(1.0).unwrap();
    assert_eq!(unusual_empty_layout.stride(), [1, 0]);
    assert!(unusual_empty_layout.is_contiguous());
}

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
fn eye_creates_square_wide_and_tall_matrices() {
    let square = Tensor::eye(3, None).unwrap();
    assert_eq!(square.shape(), [3, 3]);
    assert_eq!(square.stride(), [3, 1]);
    assert_eq!(
        square.as_slice(),
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    );

    let wide = Tensor::eye(2, 4).unwrap();
    assert_eq!(wide.shape(), [2, 4]);
    assert_eq!(wide.stride(), [4, 1]);
    assert_eq!(wide.as_slice(), [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let tall = Tensor::eye(4, Some(2)).unwrap();
    assert_eq!(tall.shape(), [4, 2]);
    assert_eq!(tall.stride(), [2, 1]);
    assert_eq!(tall.as_slice(), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(tall.dtype(), DType::Float32);
    assert_eq!(tall.device(), Device::Cpu);
}

#[test]
fn eye_handles_zero_dimensions_without_allocating_storage() {
    let square = Tensor::eye(0, None).unwrap();
    assert_eq!(square.shape(), [0, 0]);
    assert_eq!(square.stride(), [1, 1]);
    assert!(square.as_slice().is_empty());

    let maximum = isize::MAX.unsigned_abs();
    let no_rows = Tensor::eye(0, maximum).unwrap();
    assert_eq!(no_rows.shape(), [0, maximum]);
    assert_eq!(no_rows.stride(), [maximum, 1]);
    assert_eq!(no_rows.numel(), 0);

    let no_columns = Tensor::eye(maximum, 0).unwrap();
    assert_eq!(no_columns.shape(), [maximum, 0]);
    assert_eq!(no_columns.stride(), [1, 1]);
    assert_eq!(no_columns.numel(), 0);
}

#[test]
fn eye_rejects_shape_and_storage_overflow_before_allocation() {
    assert_eq!(
        Tensor::eye(usize::MAX, 2),
        Err(TensorError::ElementCountOverflow)
    );

    let elements = isize::MAX.unsigned_abs() / size_of::<f32>() + 1;
    assert_eq!(
        Tensor::eye(elements, 1),
        Err(TensorError::StorageCapacityOverflow { elements })
    );
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
    assert_eq!(left.relu().unwrap().as_slice(), [0.0, 2.0, 3.0, 0.0]);
}

#[test]
fn sine_matches_pytorch_float32_values_and_special_cases() {
    const ATOL: f32 = 1.0e-6;
    const RTOL: f32 = 1.0e-6;

    let input = Tensor::from_vec(
        vec![
            0.25,
            -0.5,
            1.0,
            -2.0,
            std::f32::consts::PI,
            1.0e10,
            -1.0e10,
            f32::MAX,
        ],
        [2, 2, 2],
    )
    .unwrap();
    // PyTorch 2.x CPU float32 reference values. Transcendental kernels are
    // compared using the same small mixed absolute/relative tolerance as the
    // differential suite rather than requiring backend-specific bit identity.
    let pytorch_reference = [
        0.247_403_96,
        -0.479_425_55,
        0.841_470_96,
        -0.909_297_4,
        -8.742_278e-8,
        -0.487_506_03,
        0.487_506_03,
        -0.521_876_5,
    ];
    let output = input.sin().unwrap();

    assert_eq!(output.shape(), input.shape());
    assert_eq!(output.stride(), input.stride());
    assert_eq!(output.dtype(), input.dtype());
    assert_eq!(output.device(), input.device());
    for (actual, expected) in output.as_slice().iter().zip(pytorch_reference) {
        assert!((actual - expected).abs() <= ATOL + RTOL * expected.abs());
    }

    let special = Tensor::from_vec(
        vec![0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        [5],
    )
    .unwrap()
    .sin()
    .unwrap();
    assert_eq!(special.as_slice()[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(special.as_slice()[1].to_bits(), (-0.0_f32).to_bits());
    assert!(special.as_slice()[2..].iter().all(|value| value.is_nan()));
}

#[test]
fn sine_handles_scalar_and_empty_tensors_with_pytorch_layouts() {
    let scalar = Tensor::from_vec(vec![0.5], []).unwrap();
    let scalar_output = scalar.sin().unwrap();
    assert!(scalar_output.shape().is_empty());
    assert!(scalar_output.stride().is_empty());
    assert!((scalar_output.item().unwrap() - 0.479_425_55).abs() <= 1.0e-6);

    let empty = Tensor::zeros([2, 0, 3]).unwrap();
    let empty_output = empty.sin().unwrap();
    assert_eq!(empty_output.shape(), empty.shape());
    assert_eq!(empty_output.stride(), empty.stride());
    assert!(empty_output.as_slice().is_empty());

    let unusual_layout = Tensor::zeros([0, 1]).unwrap().add_scalar(1.0).unwrap();
    assert_eq!(unusual_layout.stride(), [1, 0]);
    assert_eq!(unusual_layout.sin().unwrap().stride(), [1, 1]);
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

#[test]
fn contiguous_strides_cover_scalars_zero_dimensions_and_singletons() {
    assert!(Tensor::from_vec(vec![1.0], []).unwrap().stride().is_empty());
    assert_eq!(Tensor::zeros([2, 3, 4]).unwrap().stride(), [12, 4, 1]);
    assert_eq!(Tensor::zeros([2, 0, 3]).unwrap().stride(), [3, 3, 1]);
    assert_eq!(Tensor::zeros([1, 0, 1]).unwrap().stride(), [1, 1, 1]);
}

#[test]
fn empty_elementwise_results_match_pytorch_strides() {
    for (shape, expected) in [([1, 0, 1], [0, 1, 0]), ([2, 0, 3], [3, 3, 1])] {
        assert_eq!(
            Tensor::zeros(shape)
                .unwrap()
                .add_scalar(1.0)
                .unwrap()
                .stride(),
            expected
        );
    }
    assert_eq!(
        Tensor::zeros([1, 0])
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .stride(),
        [1, 1]
    );
    assert_eq!(
        Tensor::zeros([0, 1])
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .stride(),
        [1, 0]
    );

    let empty = Tensor::zeros([1, 0, 1]).unwrap();
    assert_eq!(
        empty
            .add(&Tensor::ones([1, 0, 1]).unwrap())
            .unwrap()
            .stride(),
        [1, 1, 1]
    );

    let broadcast = empty.add(&Tensor::ones([2, 1, 3]).unwrap()).unwrap();
    assert_eq!(broadcast.shape(), [2, 0, 3]);
    assert_eq!(broadcast.stride(), [3, 3, 1]);

    let compatible = Tensor::zeros([0, 1])
        .unwrap()
        .add(&Tensor::ones([1, 1]).unwrap())
        .unwrap();
    assert_eq!(compatible.stride(), [1, 0]);

    let chained = Tensor::zeros([0, 1]).unwrap().add_scalar(1.0).unwrap();
    assert_eq!(chained.stride(), [1, 0]);
    assert_eq!(chained.relu().unwrap().stride(), [1, 1]);
}

#[test]
fn extreme_empty_pointwise_outputs_match_pytorch_stride_boundaries() {
    let maximum = i64::MAX;
    let tensor = Tensor::zeros([0])
        .unwrap()
        .reshape([0, maximum, 3])
        .unwrap();

    let scalar_output = tensor.add_scalar(1.0).unwrap();
    assert_eq!(scalar_output.shape(), [0, usize::MAX / 2, 3]);
    assert_eq!(scalar_output.stride(), [1, 0, 0]);
    assert_eq!(tensor.relu(), Err(TensorError::StrideCalculationOverflow));
    assert_eq!(tensor.sin(), Err(TensorError::StrideCalculationOverflow));

    let wrapped_shape = Tensor::zeros([0])
        .unwrap()
        .reshape([0, 2, maximum, maximum])
        .unwrap();
    let wrapped_output = wrapped_shape.add_scalar(1.0).unwrap();
    assert_eq!(
        wrapped_output.shape(),
        [0, 2, usize::MAX / 2, usize::MAX / 2]
    );
    assert_eq!(wrapped_output.stride(), [2, usize::MAX / 2, 1, 1]);
    assert_eq!(
        wrapped_shape.sin(),
        Err(TensorError::StrideCalculationOverflow)
    );

    let zeroed_byte_stride = Tensor::zeros([0])
        .unwrap()
        .reshape([0, 1, 2, 1_i64 << 61])
        .unwrap();
    assert_eq!(
        zeroed_byte_stride.add_scalar(1.0).unwrap().stride(),
        [0, 0, 1, 2]
    );
}

#[test]
fn empty_reshape_preserves_compatible_source_strides() {
    let source = Tensor::zeros([0, 1]).unwrap().add_scalar(1.0).unwrap();
    let view = source.reshape([0, 1]).unwrap();

    assert_eq!(source.stride(), [1, 0]);
    assert_eq!(view.stride(), source.stride());
    assert_eq!(view.shape(), source.shape());
    assert_eq!(view.as_slice(), source.as_slice());
}

#[test]
fn reshape_is_a_contiguous_shared_storage_view() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let view = tensor.reshape([3, 2]).unwrap();

    assert_eq!(view.shape(), [3, 2]);
    assert_eq!(view.stride(), [2, 1]);
    assert_eq!(view.storage_offset(), tensor.storage_offset());
    assert_eq!(view.as_slice(), tensor.as_slice());
    assert!(std::ptr::eq(
        view.as_slice().as_ptr(),
        tensor.as_slice().as_ptr()
    ));

    let chained = view.reshape([1, 6, 1]).unwrap();
    assert_eq!(chained.stride(), [6, 1, 1]);
    assert!(std::ptr::eq(
        chained.as_slice().as_ptr(),
        tensor.as_slice().as_ptr()
    ));

    assert_eq!(
        tensor.add_scalar(1.0).unwrap().as_slice(),
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(view.into_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn integer_indexing_returns_shared_storage_views_with_pytorch_layouts() {
    let tensor = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();

    let row = tensor.index_integer(-1).unwrap();
    assert_eq!(row.shape(), [3, 4]);
    assert_eq!(row.stride(), [4, 1]);
    assert_eq!(row.storage_offset(), 12);
    assert!(row.shares_storage_with(&tensor));
    assert_eq!(
        row.as_slice(),
        (12_u8..24).map(f32::from).collect::<Vec<_>>()
    );
    assert_eq!(row.dtype(), tensor.dtype());
    assert_eq!(row.device(), tensor.device());
    assert!(std::ptr::eq(
        row.as_slice().as_ptr(),
        tensor.as_slice()[12..].as_ptr()
    ));

    let partial = tensor.index([-1, 1]).unwrap();
    assert_eq!(partial.shape(), [4]);
    assert_eq!(partial.stride(), [1]);
    assert_eq!(partial.storage_offset(), 16);
    assert_eq!(partial.as_slice(), [16.0, 17.0, 18.0, 19.0]);

    let scalar = tensor.index([1, -1, -2]).unwrap();
    assert!(scalar.shape().is_empty());
    assert!(scalar.stride().is_empty());
    assert_eq!(scalar.storage_offset(), 22);
    assert_eq!(scalar.item().unwrap().to_bits(), 22.0_f32.to_bits());

    let alias = tensor.index([]).unwrap();
    assert_eq!(alias.shape(), tensor.shape());
    assert_eq!(alias.stride(), tensor.stride());
    assert_eq!(alias.storage_offset(), tensor.storage_offset());
    assert!(alias.shares_storage_with(&tensor));
    assert!(std::ptr::eq(
        alias.as_slice().as_ptr(),
        tensor.as_slice().as_ptr()
    ));
}

#[test]
fn integer_indexing_reports_pytorch_compatible_errors() {
    let tensor = Tensor::zeros([2, 3, 4]).unwrap();
    for (indices, expected) in [
        (
            vec![2],
            TensorError::IndexOutOfBounds {
                index: 2,
                dimension: 0,
                size: 2,
            },
        ),
        (
            vec![-4, 0],
            TensorError::IndexOutOfBounds {
                index: -4,
                dimension: 0,
                size: 2,
            },
        ),
        (
            vec![0, 3],
            TensorError::IndexOutOfBounds {
                index: 3,
                dimension: 1,
                size: 3,
            },
        ),
    ] {
        assert_eq!(tensor.index(indices), Err(expected));
    }
    assert_eq!(
        tensor.index([0, 0, 0, 0]),
        Err(TensorError::TooManyIndices { dimensions: 3 })
    );

    let scalar = Tensor::from_vec(vec![5.0], []).unwrap();
    assert_eq!(
        scalar.index_integer(0),
        Err(TensorError::InvalidScalarIndex)
    );
    assert_eq!(
        scalar.index_integer(-1),
        Err(TensorError::IndexOutOfBounds {
            index: -1,
            dimension: 0,
            size: 0,
        })
    );
    assert_eq!(
        scalar.index([0]),
        Err(TensorError::TooManyIndices { dimensions: 0 })
    );
}

#[test]
fn integer_indexing_empty_dimensions_preserves_offsets_without_storage_access() {
    let empty = Tensor::zeros([2, 0, 3]).unwrap();
    let view = empty.index_integer(1).unwrap();

    assert_eq!(view.shape(), [0, 3]);
    assert_eq!(view.stride(), [3, 1]);
    assert_eq!(view.storage_offset(), 3);
    assert!(view.shares_storage_with(&empty));
    assert_eq!(view.dtype(), empty.dtype());
    assert_eq!(view.device(), empty.device());
    assert!(view.as_slice().is_empty());
    assert!(view.clone().into_vec().is_empty());
    assert_eq!(
        empty.index([1, 0]),
        Err(TensorError::IndexOutOfBounds {
            index: 0,
            dimension: 1,
            size: 0,
        })
    );
}

#[test]
fn integer_indexing_uses_checked_offset_arithmetic() {
    let maximum = usize::try_from(i64::MAX).unwrap();
    let empty = Tensor::zeros([maximum, 0]).unwrap();
    let once = empty.index_integer(i64::MAX - 1).unwrap();
    assert_eq!(once.storage_offset(), maximum - 1);
    let error = once
        .reshape([i64::MAX, 0])
        .unwrap()
        .index_integer(i64::MAX - 1);

    assert_eq!(error, Err(TensorError::InvalidStorageOffset { offset: -4 }));
    assert_eq!(
        TensorError::InvalidStorageOffset { offset: -4 }.to_string(),
        "Tensor: invalid storage offset -4"
    );
}

#[test]
fn reshape_infers_one_dimension_and_handles_scalars_and_empty_tensors() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [6]).unwrap();
    assert_eq!(tensor.reshape([2, -1]).unwrap().shape(), [2, 3]);
    assert_eq!(tensor.reshape([-1]).unwrap().shape(), [6]);

    let scalar = Tensor::from_vec(vec![7.0], [1])
        .unwrap()
        .reshape([])
        .unwrap();
    assert!(scalar.shape().is_empty());
    assert!(scalar.stride().is_empty());
    assert_eq!(scalar.item().unwrap().to_bits(), 7.0_f32.to_bits());
    assert_eq!(scalar.reshape([1]).unwrap().shape(), [1]);

    let empty = Tensor::zeros([0]).unwrap();
    assert_eq!(empty.reshape([2, -1, 3]).unwrap().shape(), [2, 0, 3]);
    assert_eq!(empty.reshape([2, -1, 3]).unwrap().stride(), [3, 3, 1]);
    assert_eq!(empty.reshape([0, 2]).unwrap().shape(), [0, 2]);

    let large = 1_i64 << 32;
    let large_empty = empty.reshape([0, large, large]).unwrap();
    assert_eq!(large_empty.shape(), [0, 1_usize << 32, 1_usize << 32]);
    assert_eq!(large_empty.stride(), [0, 1_usize << 32, 1]);
    assert_eq!(large_empty.numel(), 0);

    let maximum = i64::MAX;
    let wrapped_inference = empty.reshape([-1, maximum, maximum]).unwrap();
    assert_eq!(
        wrapped_inference.shape(),
        [0, usize::MAX / 2, usize::MAX / 2]
    );
    assert_eq!(wrapped_inference.stride(), [1, usize::MAX / 2, 1]);

    let one = Tensor::from_vec(vec![1.0], [1]).unwrap();
    assert_eq!(
        one.reshape([maximum, maximum, -1]),
        Err(TensorError::ElementCountOverflow)
    );
    assert_eq!(
        empty.reshape([3, maximum, -1]),
        Err(TensorError::ElementCountOverflow)
    );

    assert_eq!(
        empty.reshape([2, -1, 1_i64 << 62]),
        Err(TensorError::ReshapeElementCountMismatch {
            shape: vec![2, -1, 1_i64 << 62],
            elements: 0,
        })
    );

    assert_eq!(
        empty.reshape([0, 1_i64 << 62, 3]),
        Err(TensorError::StrideCalculationOverflow)
    );
}

#[test]
fn reshape_reports_pytorch_compatible_invalid_shape_errors() {
    let tensor = Tensor::zeros([6]).unwrap();

    assert_eq!(
        tensor.reshape([4, 2]).unwrap_err().to_string(),
        "shape '[4, 2]' is invalid for input of size 6"
    );
    assert_eq!(
        tensor.reshape([-1, -1]).unwrap_err().to_string(),
        "only one dimension can be inferred"
    );
    assert_eq!(
        tensor.reshape([-2, 3]).unwrap_err().to_string(),
        "invalid shape dimension -2 at index 0 of shape [-2, 3]"
    );
    assert_eq!(
        Tensor::zeros([0])
            .unwrap()
            .reshape([0, -1])
            .unwrap_err()
            .to_string(),
        "cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous"
    );

    let large = 1_i64 << 62;
    assert_eq!(
        tensor.reshape([large, 4, -2]).unwrap_err().to_string(),
        "invalid shape dimension -2 at index 2 of shape [4611686018427387904, 4, -2]"
    );
    assert_eq!(
        tensor.reshape([large, 4, -1, -1]).unwrap_err(),
        TensorError::ReshapeMultipleInferredDimensions
    );
}
