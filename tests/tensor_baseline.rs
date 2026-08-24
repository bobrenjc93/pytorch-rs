use pytorch_rs::{DType, Device, MemoryFormat, Tensor, TensorError};
use std::mem::size_of;

#[test]
fn native_metadata_describes_all_supported_storage_shapes() {
    assert_eq!(DType::default(), DType::Float32);
    assert_eq!(Device::default(), Device::Cpu);
    assert_eq!(MemoryFormat::default(), MemoryFormat::Preserve);
    assert_eq!(DType::Float32.to_string(), "float32");
    assert_eq!(DType::Float32.abbr(), "f32");
    assert_eq!(DType::Float32.element_size(), size_of::<f32>());
    assert_eq!(DType::Float32.element_size(), 4);
    assert!(DType::Float32.is_floating_point());
    assert!(!DType::Float32.is_complex());
    assert!(!DType::Float32.is_quantized());
    assert!(DType::Float32.is_signed());
    assert_eq!(DType::Float32.to_real(), DType::Float32);
    assert_eq!(Device::Cpu.to_string(), "cpu");
    assert_eq!(Device::Cpu.index(), None);
    assert!(Device::Cpu.is_cpu());
    assert!(!Device::Cpu.is_cuda());
    assert!(!Device::Cpu.is_ipu());
    assert!(!Device::Cpu.is_mtia());
    assert!(!Device::Cpu.is_maia());
    assert!(!Device::Cpu.is_xpu());
    assert!(!Device::Cpu.is_xla());
    assert!(!Device::Cpu.is_mps());
    assert!(!Device::Cpu.is_vulkan());
    assert!(!Device::Cpu.is_meta());
    assert_eq!(MemoryFormat::Preserve.to_string(), "preserve_format");
    assert_eq!(MemoryFormat::Contiguous.to_string(), "contiguous_format");

    for tensor in [
        Tensor::from_vec(vec![2.5], []).unwrap(),
        Tensor::zeros([2, 0, 3]).unwrap(),
        Tensor::ones([2, 3]).unwrap(),
        Tensor::eye(2, None).unwrap(),
        Tensor::full([4], -1.25).unwrap(),
    ] {
        assert_eq!(tensor.dtype(), DType::Float32);
        assert_eq!(tensor.element_size(), tensor.dtype().element_size());
        assert_eq!(tensor.dense_dim(), tensor.shape().len());
        assert_eq!(tensor.sparse_dim(), 0);
        assert_eq!(tensor.device(), Device::Cpu);
        assert_eq!(tensor.device().index(), None);
        assert!(tensor.is_cpu());
        assert!(!tensor.is_cuda());
        assert!(!tensor.is_ipu());
        assert!(!tensor.is_mtia());
        assert!(!tensor.is_maia());
        assert!(!tensor.is_xpu());
        assert!(!tensor.is_xla());
        assert!(!tensor.is_mps());
        assert!(!tensor.is_vulkan());
        assert!(!tensor.is_meta());
        assert!(tensor.is_floating_point());
        assert!(!tensor.is_complex());
        assert!(!tensor.is_quantized());
        assert!(!tensor.is_mkldnn());
        assert!(!tensor.is_nested());
        assert!(!tensor.is_sparse());
        assert!(!tensor.is_sparse_csr());
        assert!(!tensor.is_inference());
        assert!(!tensor.retains_grad());
        assert!(tensor.is_signed());
        assert!(!tensor.is_pinned());
    }
}

#[test]
fn native_metadata_survives_views_kernels_and_reductions() {
    let source = Tensor::from_vec(vec![-1.0, 2.0, 3.0, -4.0], [2, 2]).unwrap();
    let matrix = Tensor::ones([2, 2]).unwrap();
    let outputs = [
        source.reshape([4]).unwrap(),
        source.transpose(0, 1).unwrap(),
        source.add(&matrix).unwrap(),
        source.mul_scalar(2.0).unwrap(),
        source.relu().unwrap(),
        source.matmul(&matrix).unwrap(),
        source.sum(),
    ];

    for output in outputs {
        assert_eq!(output.dtype(), source.dtype());
        assert_eq!(output.element_size(), source.element_size());
        assert_eq!(output.dense_dim(), output.shape().len());
        assert_eq!(output.sparse_dim(), 0);
        assert_eq!(output.device(), source.device());
        assert!(output.is_cpu());
        assert!(!output.is_cuda());
        assert!(!output.is_ipu());
        assert!(!output.is_mtia());
        assert!(!output.is_maia());
        assert!(!output.is_xpu());
        assert!(!output.is_xla());
        assert!(!output.is_mps());
        assert!(!output.is_vulkan());
        assert!(!output.is_meta());
        assert!(output.is_floating_point());
        assert!(!output.is_complex());
        assert!(!output.is_quantized());
        assert!(!output.is_mkldnn());
        assert!(!output.is_nested());
        assert!(!output.is_sparse());
        assert!(!output.is_sparse_csr());
        assert!(!output.is_inference());
        assert!(!output.retains_grad());
        assert!(output.is_signed());
        assert!(!output.is_pinned());
    }
}

#[test]
fn data_ptr_tracks_empty_offset_alias_and_materialized_storage() {
    let source = Tensor::from_vec((0_u8..12).map(f32::from).collect(), [3, 4]).unwrap();
    let source_ptr = source.data_ptr();
    assert_ne!(source_ptr, 0);
    assert_eq!(source.const_data_ptr(), source_ptr);

    let row = source.index_integer(2).unwrap();
    assert_eq!(row.storage_offset(), 8);
    assert_eq!(
        row.data_ptr(),
        source_ptr + row.storage_offset() * source.element_size()
    );
    assert_eq!(row.const_data_ptr(), row.data_ptr());

    let transposed = source.transpose(0, 1).unwrap();
    assert_eq!(transposed.data_ptr(), source_ptr);
    assert_eq!(transposed.const_data_ptr(), transposed.data_ptr());
    let strided_row = transposed.index_integer(1).unwrap();
    assert_eq!(strided_row.storage_offset(), 1);
    assert_eq!(strided_row.data_ptr(), source_ptr + source.element_size());
    assert_eq!(strided_row.const_data_ptr(), strided_row.data_ptr());
    assert_eq!(
        strided_row.detach().unwrap().data_ptr(),
        strided_row.data_ptr()
    );
    assert_eq!(
        strided_row.detach().unwrap().const_data_ptr(),
        strided_row.data_ptr()
    );

    let cloned = strided_row.try_clone().unwrap();
    assert_ne!(cloned.data_ptr(), strided_row.data_ptr());
    assert_eq!(cloned.const_data_ptr(), cloned.data_ptr());
    assert!(!cloned.shares_storage_with(&strided_row));

    let packed = transposed.try_contiguous(MemoryFormat::Contiguous).unwrap();
    assert_ne!(packed.data_ptr(), transposed.data_ptr());
    assert_eq!(packed.const_data_ptr(), packed.data_ptr());
    assert!(!packed.shares_storage_with(&transposed));

    let empty = Tensor::zeros([3, 0, 4]).unwrap();
    assert_eq!(empty.data_ptr(), 0);
    assert_eq!(empty.const_data_ptr(), 0);
    let offset_empty = empty.index_integer(2).unwrap();
    assert_ne!(offset_empty.storage_offset(), 0);
    assert_eq!(offset_empty.data_ptr(), 0);
    assert_eq!(offset_empty.const_data_ptr(), 0);
    assert_eq!(offset_empty.detach().unwrap().data_ptr(), 0);
    assert_eq!(offset_empty.detach().unwrap().const_data_ptr(), 0);
    assert_eq!(offset_empty.try_clone().unwrap().data_ptr(), 0);
    assert_eq!(offset_empty.try_clone().unwrap().const_data_ptr(), 0);
}

#[test]
fn const_data_ptr_matches_data_ptr_for_ordinary_storage() {
    let scalar = Tensor::from_vec(vec![2.5], []).unwrap();
    let source = Tensor::from_vec((0_u8..12).map(f32::from).collect(), [3, 4]).unwrap();
    let offset = source.index_integer(2).unwrap();
    let strided = source.transpose(0, 1).unwrap().index_integer(1).unwrap();
    let detached = strided.detach().unwrap();
    let empty = Tensor::zeros([3, 0, 4]).unwrap().index_integer(2).unwrap();
    let leaf = Tensor::ones([2, 2]).unwrap().with_requires_grad(true);
    let autograd_output = leaf.mul_scalar(3.0).unwrap().transpose(0, 1).unwrap();

    for tensor in [
        &scalar,
        &source,
        &offset,
        &strided,
        &detached,
        &empty,
        &leaf,
        &autograd_output,
    ] {
        assert_eq!(tensor.const_data_ptr(), tensor.data_ptr());
        assert_eq!(tensor.const_data_ptr(), tensor.const_data_ptr());
    }
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
fn item_preserves_exact_values_across_scalar_offset_and_strided_layouts() {
    let bits = [
        0x0000_0000,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let source = Tensor::from_vec(bits.map(f32::from_bits).to_vec(), [1, bits.len()]).unwrap();
    let transposed = source.transpose(0, 1).unwrap();

    for (index, expected) in bits.into_iter().enumerate() {
        let scalar = Tensor::from_vec(vec![f32::from_bits(expected)], []).unwrap();
        let offset = source.index([0, i64::try_from(index).unwrap()]).unwrap();
        let strided = transposed.index([i64::try_from(index).unwrap()]).unwrap();

        assert!(scalar.shape().is_empty());
        assert_eq!(scalar.storage_offset(), 0);
        assert!(offset.shape().is_empty());
        assert_eq!(offset.storage_offset(), index);
        assert_eq!(strided.shape(), [1]);
        assert_eq!(strided.stride(), [bits.len()]);
        assert_eq!(strided.storage_offset(), index);
        for tensor in [&scalar, &offset, &strided] {
            assert_eq!(tensor.item().unwrap().to_bits(), expected);
        }
    }
}

#[test]
fn item_cardinality_failures_remain_typed_rust_errors() {
    for (tensor, elements) in [
        (Tensor::zeros([0]).unwrap(), 0),
        (Tensor::zeros([2]).unwrap(), 2),
        (Tensor::zeros([2, 3]).unwrap().transpose(0, 1).unwrap(), 6),
    ] {
        let error = tensor.item().unwrap_err();
        assert_eq!(error, TensorError::ItemRequiresOneElement { elements });
        assert_eq!(
            error.to_string(),
            format!("item requires one element, got {elements}")
        );
    }
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
fn relu_preserves_signed_zero_and_materializes_scalar_offset_and_strided_inputs() {
    const INPUT_BITS: [u32; 12] = [
        0x8000_0000,
        0x0000_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xff80_0000,
        0x7f80_0000,
        0x8000_0001,
        0x0000_0001,
        0xff7f_ffff,
        0x7f7f_ffff,
        0xbf00_0000,
        0x3f00_0000,
    ];
    const EXPECTED_BITS: [u32; 12] = [
        0x8000_0000,
        0x0000_0000,
        0x0000_0000,
        0x3f80_0000,
        0x0000_0000,
        0x7f80_0000,
        0x0000_0000,
        0x0000_0001,
        0x0000_0000,
        0x7f7f_ffff,
        0x0000_0000,
        0x3f00_0000,
    ];
    const TRANSPOSED_EXPECTED_BITS: [u32; 12] = [
        0x8000_0000,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x7f80_0000,
        0x7f7f_ffff,
        0x0000_0000,
        0x0000_0000,
        0x0000_0000,
        0x3f80_0000,
        0x0000_0001,
        0x3f00_0000,
    ];

    for zero_bits in [0x8000_0000, 0x0000_0000] {
        let input = Tensor::from_vec(vec![f32::from_bits(zero_bits)], [])
            .unwrap()
            .with_requires_grad(true);
        let output = input.relu().unwrap();
        assert!(output.shape().is_empty());
        assert!(output.stride().is_empty());
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.item().unwrap().to_bits(), zero_bits);
        assert!(!output.shares_storage_with(&input));
        assert!(output.requires_grad());
        assert!(!output.is_leaf());
    }

    let mut storage = vec![1.0; INPUT_BITS.len()];
    storage.extend(INPUT_BITS.map(f32::from_bits));
    let base = Tensor::from_vec(storage, [2, 3, 4]).unwrap();
    let offset = base.index_integer(1).unwrap();
    assert_eq!(offset.storage_offset(), INPUT_BITS.len());

    let offset_output = offset.relu().unwrap();
    assert_eq!(offset_output.shape(), offset.shape());
    assert_eq!(offset_output.stride(), offset.stride());
    assert_eq!(offset_output.storage_offset(), 0);
    assert_eq!(
        offset_output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        EXPECTED_BITS
    );
    assert!(!offset_output.shares_storage_with(&offset));

    let strided = offset.transpose(0, 1).unwrap();
    assert_eq!(strided.shape(), [4, 3]);
    assert_eq!(strided.stride(), [1, 4]);
    assert_eq!(strided.storage_offset(), INPUT_BITS.len());

    let strided_output = strided.relu().unwrap();
    assert_eq!(strided_output.shape(), strided.shape());
    assert_eq!(strided_output.stride(), strided.stride());
    assert_eq!(strided_output.storage_offset(), 0);
    assert_eq!(
        strided_output
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        TRANSPOSED_EXPECTED_BITS
    );
    assert!(!strided_output.shares_storage_with(&strided));
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
fn reciprocal_matches_pytorch_float32_ieee_bits() {
    let input_bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0080_0000,
        0x8080_0000,
        0x3eaa_aaab,
        0xbeaa_aaab,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f81_2345,
        0xff81_2345,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let expected_bits = [
        0x7f80_0000,
        0xff80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7e80_0000,
        0xfe80_0000,
        0x4040_0000,
        0xc040_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x0020_0000,
        0x8020_0000,
        0x0000_0000,
        0x8000_0000,
        0x7fc1_2345,
        0xffc1_2345,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let input =
        Tensor::from_vec(input_bits.map(f32::from_bits).to_vec(), [input_bits.len()]).unwrap();
    let output = input.reciprocal().unwrap();

    assert!(output.logical_values().map(f32::to_bits).eq(expected_bits));
    assert!(!output.shares_storage_with(&input));
}

#[test]
fn reciprocal_preserves_unary_layouts_and_materializes_fresh_storage() {
    let base = Tensor::from_vec((1_u8..=24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let strided = base.transpose(0, 2).unwrap();
    let offset = strided.index_integer(1).unwrap();
    let cases: [(Tensor, Vec<usize>); 6] = [
        (Tensor::from_vec(vec![-0.0], []).unwrap(), Vec::new()),
        (Tensor::zeros([0, 1]).unwrap(), vec![1, 1]),
        (Tensor::zeros([0, 1, 2]).unwrap(), vec![2, 2, 1]),
        (Tensor::zeros([1, 0, 1]).unwrap(), vec![1, 1, 1]),
        (offset, vec![1, 3]),
        (strided, vec![1, 4, 12]),
    ];

    for (input, expected_strides) in cases {
        let output = input.reciprocal().unwrap();
        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.stride(), expected_strides);
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.dtype(), input.dtype());
        assert_eq!(output.device(), input.device());
        assert!(!output.shares_storage_with(&input));
    }
}

#[test]
fn hyperbolic_tangent_matches_pytorch_float32_values_and_ieee_special_cases() {
    const ATOL: f32 = f32::from_bits(1);
    const RTOL: f32 = 3.0 * f32::EPSILON;

    let ordinary = Tensor::from_vec(
        vec![
            -20.0, -10.0, -3.0, -2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0, 3.0, 10.0, 20.0,
        ],
        [2, 7],
    )
    .unwrap()
    .tanh()
    .unwrap();
    let pytorch_reference = [
        -1.0,
        -1.0,
        -0.995_054_8,
        -0.964_027_6,
        -0.761_594_2,
        -0.462_117_17,
        -0.244_918_66,
        0.244_918_66,
        0.462_117_17,
        0.761_594_2,
        0.964_027_6,
        0.995_054_8,
        1.0,
        1.0,
    ];
    for (actual, expected) in ordinary.as_slice().iter().zip(pytorch_reference) {
        assert!((actual - expected).abs() <= ATOL + RTOL * expected.abs());
    }

    let input_bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0080_0000,
        0x8080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7f81_2345,
        0xff81_2345,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let expected_bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7fc1_2345,
        0xffc1_2345,
        0x7fc1_2345,
        0xffc5_4321,
    ];
    let input =
        Tensor::from_vec(input_bits.map(f32::from_bits).to_vec(), [input_bits.len()]).unwrap();
    let output = input.tanh().unwrap();

    assert!(output.logical_values().map(f32::to_bits).eq(expected_bits));
    assert!(!output.shares_storage_with(&input));
}

#[test]
fn hyperbolic_tangent_preserves_unary_layouts_and_materializes_fresh_storage() {
    let base = Tensor::from_vec(
        (0_u8..24).map(|value| f32::from(value) - 12.0).collect(),
        [2, 3, 4],
    )
    .unwrap();
    let strided = base.transpose(0, 2).unwrap();
    let offset = strided.index_integer(1).unwrap();
    let channels_last = Tensor::from_vec(
        (0_u8..120).map(|value| f32::from(value) - 60.0).collect(),
        [2, 3, 4, 5],
    )
    .unwrap()
    .try_contiguous(MemoryFormat::ChannelsLast)
    .unwrap();
    let channels_last_3d = Tensor::from_vec(
        (0_u16..720).map(|value| f32::from(value) - 360.0).collect(),
        [2, 3, 4, 5, 6],
    )
    .unwrap()
    .try_contiguous(MemoryFormat::ChannelsLast3d)
    .unwrap();
    let cases: [(Tensor, Vec<usize>); 8] = [
        (Tensor::from_vec(vec![-0.0], []).unwrap(), Vec::new()),
        (Tensor::zeros([0, 1]).unwrap(), vec![1, 1]),
        (Tensor::zeros([0, 1, 2]).unwrap(), vec![2, 2, 1]),
        (Tensor::zeros([1, 0, 1]).unwrap(), vec![1, 1, 1]),
        (offset, vec![1, 3]),
        (strided, vec![1, 4, 12]),
        (channels_last, vec![60, 1, 15, 3]),
        (channels_last_3d, vec![360, 1, 90, 18, 3]),
    ];

    for (input, expected_strides) in cases {
        let output = input.tanh().unwrap();
        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.stride(), expected_strides);
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.dtype(), input.dtype());
        assert_eq!(output.device(), input.device());
        assert!(!output.shares_storage_with(&input));
    }
}

#[test]
fn exponential_matches_pytorch_float32_values_and_ieee_special_cases() {
    const ATOL: f32 = f32::from_bits(1);
    const RTOL: f32 = 2.0e-6;

    let ordinary = Tensor::from_vec(
        vec![
            -80.0, -20.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0, 20.0, 80.0,
        ],
        [2, 2, 3],
    )
    .unwrap()
    .exp()
    .unwrap();
    let pytorch_reference = [
        1.804_851_3e-35,
        2.061_153_7e-9,
        0.135_335_28,
        0.367_879_45,
        0.606_530_67,
        1.0,
        1.648_721_2,
        2.718_281_7,
        7.389_056,
        22_026.465,
        4.851_652e8,
        5.540_622_5e34,
    ];
    for (actual, expected) in ordinary.as_slice().iter().zip(pytorch_reference) {
        assert!((actual - expected).abs() <= ATOL + RTOL * expected.abs());
    }

    let smallest_subnormal = f32::from_bits(1);
    let special = Tensor::from_vec(
        vec![
            0.0,
            -0.0,
            smallest_subnormal,
            -smallest_subnormal,
            -100.0,
            -104.0,
            88.0,
            89.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ],
        [11],
    )
    .unwrap()
    .exp()
    .unwrap();
    assert!(
        special.as_slice()[..4]
            .iter()
            .all(|value| value.to_bits() == 1.0_f32.to_bits())
    );
    assert!(special.as_slice()[4].is_subnormal());
    assert_eq!(special.as_slice()[5].to_bits(), 0.0_f32.to_bits());
    assert!(special.as_slice()[6].is_finite());
    assert_eq!(special.as_slice()[7].to_bits(), f32::INFINITY.to_bits());
    assert!(special.as_slice()[8].is_nan());
    assert_eq!(special.as_slice()[9].to_bits(), f32::INFINITY.to_bits());
    assert_eq!(special.as_slice()[10].to_bits(), 0.0_f32.to_bits());
}

#[test]
fn exponential_preserves_metadata_and_materializes_views_canonically() {
    let scalar = Tensor::from_vec(vec![1.0], []).unwrap().exp().unwrap();
    assert!(scalar.shape().is_empty());
    assert!(scalar.stride().is_empty());
    assert_eq!(scalar.dtype(), DType::Float32);
    assert_eq!(scalar.device(), Device::Cpu);

    let source = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3]).unwrap();
    let indexed = source.index_integer(1).unwrap();
    assert_eq!(indexed.storage_offset(), 3);
    let indexed_output = indexed.exp().unwrap();
    assert_eq!(indexed_output.shape(), [3]);
    assert_eq!(indexed_output.stride(), [1]);
    assert_eq!(indexed_output.storage_offset(), 0);
    assert!(!indexed_output.shares_storage_with(&source));
    for (actual, expected) in indexed_output
        .as_slice()
        .iter()
        .zip([20.085_537, 54.598_15, 148.413_16])
    {
        assert!((actual - expected).abs() <= 2.0e-6 * expected);
    }

    let reshaped = source.reshape([1, 2, 3]).unwrap();
    let reshaped_output = reshaped.exp().unwrap();
    assert_eq!(reshaped_output.shape(), [1, 2, 3]);
    assert_eq!(reshaped_output.stride(), [6, 3, 1]);
    assert_eq!(reshaped_output.dtype(), reshaped.dtype());
    assert_eq!(reshaped_output.device(), reshaped.device());
}

#[test]
fn exponential_handles_empty_shapes_and_reports_metadata_overflow() {
    let empty = Tensor::zeros([2, 0, 3]).unwrap().exp().unwrap();
    assert_eq!(empty.shape(), [2, 0, 3]);
    assert_eq!(empty.stride(), [3, 3, 1]);
    assert!(empty.as_slice().is_empty());

    let maximum = isize::MAX.unsigned_abs();
    let offset_view = Tensor::zeros([maximum, 0])
        .unwrap()
        .index_integer(i64::try_from(maximum - 1).unwrap())
        .unwrap();
    assert!(offset_view.storage_offset() > 0);
    let offset_output = offset_view.exp().unwrap();
    assert_eq!(offset_output.shape(), [0]);
    assert_eq!(offset_output.stride(), [1]);
    assert_eq!(offset_output.storage_offset(), 0);
    assert!(offset_output.as_slice().is_empty());

    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap();
    assert_eq!(extreme.exp(), Err(TensorError::StrideCalculationOverflow));
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
fn rank_zero_binary_arithmetic_matches_general_broadcast_bits_and_view_layout() {
    let mut source_values = vec![0.0; 6];
    source_values.extend(
        [
            0x0000_0000,
            0x8000_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc5_4321,
            0xffc6_789a,
        ]
        .map(f32::from_bits),
    );
    let view = Tensor::from_vec(source_values, [2, 2, 3])
        .unwrap()
        .index([1])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert_eq!(view.stride(), [1, 3]);
    assert_eq!(view.storage_offset(), 6);

    for scalar_bits in [0x7f81_2345, 0xffc1_2345] {
        let scalar_value = f32::from_bits(scalar_bits);
        let scalar = Tensor::from_vec(vec![0.0, scalar_value], [2])
            .unwrap()
            .index([1])
            .unwrap();
        let singleton = Tensor::from_vec(vec![scalar_value], [1, 1]).unwrap();
        assert!(scalar.shape().is_empty());
        assert_eq!(scalar.storage_offset(), 1);

        let cases = [
            (scalar.add(&view).unwrap(), singleton.add(&view).unwrap()),
            (view.add(&scalar).unwrap(), view.add(&singleton).unwrap()),
            (scalar.sub(&view).unwrap(), singleton.sub(&view).unwrap()),
            (view.sub(&scalar).unwrap(), view.sub(&singleton).unwrap()),
            (scalar.mul(&view).unwrap(), singleton.mul(&view).unwrap()),
            (view.mul(&scalar).unwrap(), view.mul(&singleton).unwrap()),
            (scalar.div(&view).unwrap(), singleton.div(&view).unwrap()),
            (view.div(&scalar).unwrap(), view.div(&singleton).unwrap()),
        ];
        for (actual, expected) in cases {
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(actual.stride(), expected.stride());
            assert_eq!(actual.stride(), [1, 3]);
            assert_eq!(actual.storage_offset(), 0);
            assert!(
                actual
                    .logical_values()
                    .map(f32::to_bits)
                    .eq(expected.logical_values().map(f32::to_bits))
            );
        }
    }
}

#[test]
fn rank_zero_binary_arithmetic_preserves_extreme_empty_metadata_and_errors() {
    let scalar = Tensor::from_vec(vec![1.0], []).unwrap();
    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap();
    for output in [
        scalar.add(&extreme).unwrap(),
        extreme.add(&scalar).unwrap(),
        scalar.sub(&extreme).unwrap(),
        extreme.sub(&scalar).unwrap(),
        scalar.mul(&extreme).unwrap(),
        extreme.mul(&scalar).unwrap(),
        scalar.div(&extreme).unwrap(),
        extreme.div(&scalar).unwrap(),
    ] {
        assert_eq!(output.shape(), [0, isize::MAX.unsigned_abs(), 3]);
        assert_eq!(output.stride(), [1, 0, 0]);
        assert_eq!(output.storage_offset(), 0);
        assert_eq!(output.numel(), 0);
    }

    let overflowing = Tensor::zeros([0])
        .unwrap()
        .reshape([0, 1, 1_i64 << 62, 1_i64 << 32])
        .unwrap();
    for result in [
        scalar.add(&overflowing),
        overflowing.add(&scalar),
        scalar.sub(&overflowing),
        overflowing.sub(&scalar),
        scalar.mul(&overflowing),
        overflowing.mul(&scalar),
        scalar.div(&overflowing),
        overflowing.div(&scalar),
    ] {
        assert_eq!(result, Err(TensorError::StrideCalculationOverflow));
    }
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
fn reflected_division_uses_unary_layout_planning_and_checks_stride_overflow() {
    let source = Tensor::from_vec(vec![1.0, 2.0], [1, 2]).unwrap();
    let transposed = source.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), [2, 1]);
    assert_eq!(transposed.stride(), [1, 2]);

    let ordinary = transposed.div_scalar(1.0).unwrap();
    assert_eq!(ordinary.stride(), [1, 2]);
    let reflected = transposed.scalar_div(1.0).unwrap();
    assert_eq!(reflected.stride(), [1, 1]);
    assert_eq!(reflected.logical_values().collect::<Vec<_>>(), [1.0, 0.5]);

    let empty_cases = [
        (
            Tensor::zeros([1, 0]).unwrap().transpose(0, 1).unwrap(),
            vec![1, 0],
        ),
        (Tensor::zeros([1, 0, 1]).unwrap(), vec![0, 1, 0]),
        (
            Tensor::zeros([2, 0, 3]).unwrap().transpose(0, 2).unwrap(),
            vec![2, 2, 1],
        ),
    ];
    for (empty, expected_strides) in empty_cases {
        let output = empty.scalar_div(1.0).unwrap();
        assert_eq!(output.shape(), empty.shape());
        assert_eq!(output.stride(), expected_strides);
        assert_eq!(output.numel(), 0);
    }

    let maximum = isize::MAX.unsigned_abs();
    let large = Tensor::zeros([2, 0, maximum]).unwrap();
    assert_eq!(large.scalar_div(1.0).unwrap().stride(), [0, 1, 0]);
    let extreme = large.transpose(0, 1).unwrap();
    assert_eq!(
        extreme.scalar_div(1.0),
        Err(TensorError::StrideCalculationOverflow)
    );
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
    let maximum = isize::MAX.unsigned_abs();
    for (left_shape, right_shape) in [
        ([2, 3], [4, 2]),
        ([0, 3], [4, 0]),
        ([maximum, 0], [1, 0]),
        ([0, maximum], [0, 0]),
    ] {
        let error = Tensor::zeros(left_shape)
            .unwrap()
            .matmul(&Tensor::zeros(right_shape).unwrap())
            .unwrap_err();
        assert_eq!(
            error,
            TensorError::MatmulInnerDimensionMismatch {
                left: left_shape.to_vec(),
                right: right_shape.to_vec(),
            }
        );
        assert_eq!(
            error.to_string(),
            format!(
                "mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})",
                left_shape[0], left_shape[1], right_shape[0], right_shape[1]
            )
        );
    }
}

#[test]
fn transpose_is_a_metadata_only_shared_storage_view() {
    let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let transposed = source.transpose(0, -1).unwrap();

    assert_eq!(transposed.shape(), [4, 3, 2]);
    assert_eq!(transposed.stride(), [1, 4, 12]);
    assert_eq!(transposed.storage_offset(), 0);
    assert_eq!(transposed.dtype(), source.dtype());
    assert_eq!(transposed.device(), source.device());
    assert!(transposed.shares_storage_with(&source));
    assert_eq!(
        transposed.logical_values().collect::<Vec<_>>(),
        [
            0.0, 12.0, 4.0, 16.0, 8.0, 20.0, 1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0,
            10.0, 22.0, 3.0, 15.0, 7.0, 19.0, 11.0, 23.0,
        ]
    );

    let restored = transposed.transpose(-1, 0).unwrap();
    assert_eq!(restored.shape(), source.shape());
    assert_eq!(restored.stride(), source.stride());
    assert_eq!(
        restored.logical_values().collect::<Vec<_>>(),
        source.as_slice()
    );
    assert!(restored.shares_storage_with(&source));

    let duplicate = source.transpose(1, -2).unwrap();
    assert_eq!(duplicate.shape(), source.shape());
    assert_eq!(duplicate.stride(), source.stride());
    assert!(duplicate.shares_storage_with(&source));
}

#[test]
fn is_set_to_requires_identical_storage_offset_shape_and_strides() {
    let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let detached = source.detach().unwrap();
    let identical_view = source.reshape([2, 3, 4]).unwrap();
    let clone = source.try_clone().unwrap();
    let reshaped = source.reshape([6, 4]).unwrap();
    let transposed = source.transpose(0, 2).unwrap();

    assert!(source.is_set_to(&source));
    assert!(source.is_set_to(&detached));
    assert!(source.is_set_to(&identical_view));
    assert!(!source.is_set_to(&clone));
    assert!(!source.is_set_to(&reshaped));
    assert!(!source.is_set_to(&transposed));

    let first_offset_view = source.transpose(0, 2).unwrap().index_integer(1).unwrap();
    let second_offset_view = source.transpose(0, 2).unwrap().index_integer(1).unwrap();
    let different_offset = source.transpose(0, 2).unwrap().index_integer(2).unwrap();
    assert!(first_offset_view.is_set_to(&second_offset_view));
    assert!(first_offset_view.is_set_to(&first_offset_view.detach().unwrap()));
    assert!(!first_offset_view.is_set_to(&different_offset));

    let scalar = Tensor::from_vec(vec![3.0], []).unwrap();
    assert!(scalar.is_set_to(&scalar.detach().unwrap()));
    assert!(!scalar.is_set_to(&scalar.try_clone().unwrap()));

    let empty = Tensor::zeros([2, 0, 3]).unwrap();
    assert!(empty.is_set_to(&empty.detach().unwrap()));
    assert!(!empty.is_set_to(&empty.try_clone().unwrap()));
    assert!(!empty.is_set_to(&empty.transpose(0, 2).unwrap()));
}

#[test]
fn is_same_size_compares_only_shape_metadata() {
    let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let clone = source.try_clone().unwrap();
    let independent = Tensor::zeros([2, 3, 4]).unwrap();
    let restored = source.transpose(0, 2).unwrap().transpose(0, 2).unwrap();

    assert!(source.is_same_size(&source));
    assert!(source.is_same_size(&source.detach().unwrap()));
    assert!(source.is_same_size(&clone));
    assert!(source.is_same_size(&independent));
    assert!(source.is_same_size(&restored));
    assert!(!source.is_same_size(&source.reshape([6, 4]).unwrap()));
    assert!(!source.is_same_size(&source.transpose(0, 2).unwrap()));

    let square = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
    let transposed = square.transpose(0, 1).unwrap();
    assert_ne!(square.stride(), transposed.stride());
    assert!(square.is_same_size(&transposed));

    let scalar = Tensor::from_vec(vec![3.0], []).unwrap();
    assert!(scalar.is_same_size(&Tensor::from_vec(vec![-8.0], []).unwrap()));
    assert!(!scalar.is_same_size(&Tensor::from_vec(vec![3.0], [1]).unwrap()));

    let extreme_empty = Tensor::zeros([2, 0, usize::MAX / 2]).unwrap();
    assert!(extreme_empty.is_same_size(&extreme_empty.try_clone().unwrap()));
    assert!(!extreme_empty.is_same_size(&extreme_empty.transpose(0, 2).unwrap()));
}

#[test]
fn arbitrary_dimension_permutations_power_t_and_mt_views() {
    let source = Tensor::from_vec((0_u8..120).map(f32::from).collect(), [2, 3, 4, 5]).unwrap();
    let offset_view = source.transpose(0, 3).unwrap().index_integer(1).unwrap();
    assert_eq!(offset_view.shape(), [3, 4, 2]);
    assert_eq!(offset_view.stride(), [20, 5, 60]);
    assert_eq!(offset_view.storage_offset(), 1);

    let permuted = offset_view.permute_axes([2, 0, 1]).unwrap();
    assert_eq!(permuted.shape(), [2, 3, 4]);
    assert_eq!(permuted.stride(), [60, 20, 5]);
    assert_eq!(permuted.storage_offset(), 1);
    assert_eq!(permuted.numel(), offset_view.numel());
    assert_eq!(permuted.dtype(), offset_view.dtype());
    assert_eq!(permuted.device(), offset_view.device());
    assert!(permuted.shares_storage_with(&source));

    let reversed = offset_view.reverse_dimensions().unwrap();
    assert_eq!(reversed.shape(), [2, 4, 3]);
    assert_eq!(reversed.stride(), [60, 5, 20]);
    assert_eq!(reversed.storage_offset(), 1);
    let restored = reversed.reverse_dimensions().unwrap();
    assert_eq!(restored.shape(), offset_view.shape());
    assert_eq!(restored.stride(), offset_view.stride());
    assert_eq!(restored.storage_offset(), offset_view.storage_offset());
    assert!(restored.shares_storage_with(&offset_view));
    assert_eq!(
        restored.try_to_vec().unwrap(),
        offset_view.try_to_vec().unwrap()
    );

    let matrix_transposed = offset_view.matrix_transpose().unwrap();
    let explicit = offset_view.transpose(-2, -1).unwrap();
    assert_eq!(matrix_transposed.shape(), explicit.shape());
    assert_eq!(matrix_transposed.stride(), explicit.stride());
    assert_eq!(
        matrix_transposed.try_to_vec().unwrap(),
        explicit.try_to_vec().unwrap()
    );
    assert!(matrix_transposed.shares_storage_with(&offset_view));
}

#[test]
fn dimension_permutation_validation_is_safe_and_rank_independent() {
    let tensor = Tensor::zeros([2, 3, 4]).unwrap();
    assert_eq!(
        tensor.permute_axes([0, 1]),
        Err(TensorError::PermutationRankMismatch {
            dimensions: 2,
            rank: 3,
        })
    );
    assert_eq!(
        tensor.permute_axes([0, 1, 3]),
        Err(TensorError::PermutationDimensionOutOfRange {
            dimension: 3,
            rank: 3,
        })
    );
    assert_eq!(
        tensor.permute_axes([0, 1, 1]),
        Err(TensorError::DuplicatePermutationDimension { dimension: 1 })
    );

    let maximum = isize::MAX.unsigned_abs();
    let extreme_empty = Tensor::zeros([3, 0, 1, maximum]).unwrap();
    assert_eq!(
        extreme_empty.permute_axes([3, 0, 1, 2]),
        Err(TensorError::ElementCountOverflow)
    );
    assert_eq!(
        extreme_empty.permute_axes([3, 0, 0, 1]),
        Err(TensorError::DuplicatePermutationDimension { dimension: 0 })
    );
    let zero_before_overflow = extreme_empty.permute_axes([3, 1, 0, 2]).unwrap();
    assert_eq!(zero_before_overflow.shape(), [maximum, 0, 3, 1]);
    assert!(zero_before_overflow.shares_storage_with(&extreme_empty));

    let scalar = Tensor::from_vec(vec![2.5], []).unwrap();
    let scalar_t = scalar.reverse_dimensions().unwrap();
    assert!(scalar_t.shape().is_empty());
    assert!(scalar_t.stride().is_empty());
    assert!(scalar_t.shares_storage_with(&scalar));
    assert!(
        scalar
            .matrix_transpose()
            .unwrap()
            .shares_storage_with(&scalar)
    );

    let vector = Tensor::from_vec(vec![1.0, 2.0], [2]).unwrap();
    let vector_t = vector.reverse_dimensions().unwrap();
    assert_eq!(vector_t.shape(), [2]);
    assert_eq!(vector_t.stride(), [1]);
    assert!(vector_t.shares_storage_with(&vector));
    assert_eq!(
        vector.matrix_transpose(),
        Err(TensorError::MatrixTransposeRequiresMatrix { rank: 1 })
    );
    assert_eq!(
        vector.matrix_transpose().unwrap_err().to_string(),
        "tensor.mT is only supported on matrices or batches of matrices. Got 1-D tensor."
    );

    let mut high_rank_shape = vec![1; 96];
    high_rank_shape[3] = 2;
    high_rank_shape[47] = 0;
    high_rank_shape[91] = 3;
    let high_rank = Tensor::zeros(high_rank_shape.clone()).unwrap();
    let high_rank_t = high_rank.reverse_dimensions().unwrap();
    high_rank_shape.reverse();
    assert_eq!(high_rank_t.shape(), high_rank_shape);
    assert_eq!(
        high_rank_t.stride(),
        high_rank.stride().iter().rev().copied().collect::<Vec<_>>()
    );
    assert!(high_rank_t.shares_storage_with(&high_rank));
    assert!(high_rank_t.logical_values().next().is_none());
}

#[test]
fn t_and_mt_preserve_extreme_empty_metadata_without_materializing() {
    let maximum = isize::MAX.unsigned_abs();
    let source = Tensor::zeros([maximum, 0, maximum]).unwrap();

    let reversed = source.reverse_dimensions().unwrap();
    assert_eq!(reversed.shape(), [maximum, 0, maximum]);
    assert_eq!(reversed.stride(), [1, maximum, maximum]);
    assert_eq!(reversed.storage_offset(), 0);
    assert_eq!(reversed.numel(), 0);
    assert!(reversed.shares_storage_with(&source));

    assert_eq!(
        source.matrix_transpose(),
        Err(TensorError::ElementCountOverflow)
    );
    assert_eq!(
        source.transpose(-2, -1),
        Err(TensorError::ElementCountOverflow)
    );

    let offset = Tensor::zeros([maximum, 0, 1])
        .unwrap()
        .index_integer(i64::MAX - 1)
        .unwrap();
    for view in [
        offset.reverse_dimensions().unwrap(),
        offset.matrix_transpose().unwrap(),
    ] {
        assert_eq!(view.storage_offset(), maximum - 1);
        assert_eq!(view.numel(), 0);
        assert!(view.shares_storage_with(&offset));
        assert!(view.logical_values().next().is_none());
    }
}

#[test]
fn transpose_dimension_normalization_matches_pytorch_for_scalars_and_ranks() {
    let scalar = Tensor::from_vec(vec![3.5], []).unwrap();
    for dimensions in [(0, 0), (-1, -1), (0, -1), (-1, 0)] {
        let view = scalar.transpose(dimensions.0, dimensions.1).unwrap();
        assert!(view.shape().is_empty());
        assert!(view.stride().is_empty());
        assert_eq!(view.item().unwrap().to_bits(), 3.5_f32.to_bits());
        assert!(view.shares_storage_with(&scalar));
    }
    for dimension in [-2, 1] {
        assert_eq!(
            scalar.transpose(dimension, 0),
            Err(TensorError::DimensionOutOfRange { dimension, rank: 0 })
        );
    }

    let tensor = Tensor::zeros([2, 3, 4]).unwrap();
    for dimension in [-4, 3] {
        assert_eq!(
            tensor.transpose(dimension, 0),
            Err(TensorError::DimensionOutOfRange { dimension, rank: 3 })
        );
    }
    assert_eq!(
        tensor.transpose(3, 0).unwrap_err().to_string(),
        "Dimension out of range (expected to be in range of [-3, 2], but got 3)"
    );
}

#[test]
fn transpose_defines_contiguous_and_non_overlapping_dense_invariants() {
    let contiguous = Tensor::zeros([2, 3, 4]).unwrap();
    let dense_transpose = contiguous.transpose(0, 2).unwrap();
    let non_dense = dense_transpose.index_integer(1).unwrap();
    let singleton = Tensor::zeros([2, 1, 3]).unwrap().transpose(0, 1).unwrap();
    let empty = Tensor::zeros([2, 0, 3]).unwrap().transpose(0, 2).unwrap();

    assert!(contiguous.is_contiguous());
    assert!(contiguous.is_non_overlapping_and_dense());
    assert!(!dense_transpose.is_contiguous());
    assert!(dense_transpose.is_non_overlapping_and_dense());
    assert!(!non_dense.is_contiguous());
    assert!(!non_dense.is_non_overlapping_and_dense());
    assert!(singleton.is_contiguous());
    assert!(singleton.is_non_overlapping_and_dense());
    assert!(empty.is_contiguous());
    assert!(empty.is_non_overlapping_and_dense());
}

#[test]
fn memory_format_contiguity_queries_match_layout_metadata() {
    let contiguous = Tensor::zeros([2, 3, 4, 5]).unwrap();
    assert!(contiguous.is_contiguous_with_memory_format(MemoryFormat::Preserve));
    assert!(contiguous.is_contiguous_with_memory_format(MemoryFormat::Contiguous));
    assert!(!contiguous.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));
    assert!(!contiguous.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast3d));

    let channels_last = Tensor::zeros([1, 1, 2, 2])
        .unwrap()
        .transpose(1, 3)
        .unwrap();
    assert!(!channels_last.is_contiguous());
    assert!(channels_last.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));

    let channels_last_3d = Tensor::zeros([2, 4, 5, 6, 3])
        .unwrap()
        .transpose(1, 4)
        .unwrap()
        .transpose(2, 4)
        .unwrap()
        .transpose(3, 4)
        .unwrap();
    assert_eq!(channels_last_3d.stride(), [360, 1, 90, 18, 3]);
    assert!(channels_last_3d.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast3d));
    assert!(!channels_last_3d.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));

    let empty_channels_last = Tensor::zeros([2, 4, 5, 0])
        .unwrap()
        .transpose(1, 3)
        .unwrap()
        .transpose(2, 3)
        .unwrap();
    assert_eq!(empty_channels_last.stride(), [20, 1, 5, 1]);
    assert!(!empty_channels_last.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));
}

#[test]
fn pointwise_outputs_canonicalize_singleton_channels_last_strides() {
    let source = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0], [1, 1, 2, 2]).unwrap();
    let view = source.transpose(1, 3).unwrap();
    assert_eq!(view.shape(), [1, 2, 2, 1]);
    assert_eq!(view.stride(), [4, 1, 2, 4]);

    for output in [
        view.relu().unwrap(),
        view.sin().unwrap(),
        view.add(&view).unwrap(),
    ] {
        assert_eq!(output.shape(), view.shape());
        assert_eq!(output.stride(), [4, 1, 2, 2]);
        assert_eq!(output.reshape([1, 2, 2, 1]).unwrap().stride(), [2, 1, 2, 2]);
        assert!(!output.shares_storage_with(&view));
    }
}

#[test]
fn stride_aware_consumers_handle_transposed_and_indexed_views() {
    let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let view = source.transpose(0, 2).unwrap();
    let indexed = view.index_integer(1).unwrap();
    assert_eq!(indexed.shape(), [3, 2]);
    assert_eq!(indexed.stride(), [4, 12]);
    assert_eq!(indexed.storage_offset(), 1);
    assert_eq!(
        indexed.index([2, 1]).unwrap().item().unwrap().to_bits(),
        21.0_f32.to_bits()
    );

    let clone = indexed.try_clone().unwrap();
    assert_eq!(clone.stride(), [1, 3]);
    assert_eq!(
        clone.logical_values().collect::<Vec<_>>(),
        [1.0, 13.0, 5.0, 17.0, 9.0, 21.0]
    );
    assert!(!clone.shares_storage_with(&source));

    let same_shape = view.reshape([4, 3, 2]).unwrap();
    assert_eq!(same_shape.stride(), view.stride());
    assert!(same_shape.shares_storage_with(&view));
    let flattened = view.reshape([24]).unwrap();
    assert_eq!(flattened.stride(), [1]);
    assert!(!flattened.shares_storage_with(&view));
    assert_eq!(
        flattened.as_slice(),
        view.logical_values().collect::<Vec<_>>()
    );

    for output in [
        view.relu().unwrap(),
        view.sin().unwrap(),
        view.exp().unwrap(),
    ] {
        assert_eq!(output.shape(), view.shape());
        assert_eq!(output.stride(), view.stride());
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.shares_storage_with(&view));
    }
    assert_eq!(view.sum().item().unwrap().to_bits(), 276.0_f32.to_bits());

    let broadcast_row = Tensor::from_vec(vec![10.0, 20.0], [2]).unwrap();
    let broadcast = view.add(&broadcast_row).unwrap();
    assert_eq!(broadcast.stride(), view.stride());
    assert_eq!(
        broadcast.logical_values().take(6).collect::<Vec<_>>(),
        [10.0, 32.0, 14.0, 36.0, 18.0, 40.0]
    );
}

#[test]
fn flatten_collapses_compatible_ranges_as_shared_storage_views() {
    let source = Tensor::from_vec((0_u8..120).map(f32::from).collect(), [2, 3, 4, 5]).unwrap();
    let view = source.transpose(0, 1).unwrap().index_integer(1).unwrap();
    assert_eq!(view.shape(), [2, 4, 5]);
    assert_eq!(view.stride(), [60, 5, 1]);
    assert_eq!(view.storage_offset(), 20);

    let flattened = view.flatten(1, -1).unwrap();
    assert_eq!(flattened.shape(), [2, 20]);
    assert_eq!(flattened.stride(), [60, 1]);
    assert_eq!(flattened.storage_offset(), 20);
    assert!(flattened.shares_storage_with(&source));
    assert_eq!(
        flattened.logical_values().collect::<Vec<_>>(),
        view.logical_values().collect::<Vec<_>>()
    );

    let unchanged = view.flatten(-1, -1).unwrap();
    assert_eq!(unchanged.shape(), view.shape());
    assert_eq!(unchanged.stride(), view.stride());
    assert!(unchanged.shares_storage_with(&view));
}

#[test]
fn flatten_incompatible_ranges_are_eager_independent_contiguous_copies() {
    let bits = [
        0x0000_0000_u32,
        0x8000_0000,
        0x7fc1_2345,
        0x7f80_0000,
        0xff80_0000,
        0x40a0_0000,
    ];
    let source = Tensor::from_vec(bits.map(f32::from_bits).to_vec(), [2, 3]).unwrap();
    let flattened = source.transpose(0, 1).unwrap().flatten(0, 1).unwrap();

    assert_eq!(flattened.shape(), [6]);
    assert_eq!(flattened.stride(), [1]);
    assert_eq!(flattened.storage_offset(), 0);
    assert!(!flattened.shares_storage_with(&source));
    assert_eq!(
        flattened
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]]
    );
    assert!(flattened.sum().shape().is_empty());
    assert_eq!(flattened.add_scalar(1.0).unwrap().stride(), [1]);
}

#[test]
fn flatten_handles_scalars_lifetimes_empty_shapes_and_wrapping_metadata() {
    let scalar = Tensor::from_vec(vec![3.5], []).unwrap();
    let scalar_view = scalar.flatten(0, -1).unwrap();
    assert_eq!(scalar_view.shape(), [1]);
    assert_eq!(scalar_view.stride(), [1]);
    assert!(scalar_view.shares_storage_with(&scalar));

    let surviving_view = {
        let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
        source.index_integer(1).unwrap().flatten(0, 1).unwrap()
    };
    assert_eq!(surviving_view.storage_offset(), 12);
    assert_eq!(
        surviving_view.as_slice(),
        (12_u8..24).map(f32::from).collect::<Vec<_>>()
    );

    let empty = Tensor::zeros([2, 0, 3]).unwrap().flatten(0, -1).unwrap();
    assert_eq!(empty.shape(), [0]);
    assert_eq!(empty.stride(), [1]);
    assert!(empty.logical_values().next().is_none());

    let maximum = i64::MAX;
    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([0, maximum, maximum])
        .unwrap();
    let wrapped = extreme.flatten(1, 2).unwrap();
    assert_eq!(wrapped.shape(), [0, 1]);
    assert_eq!(wrapped.stride(), [1, 1]);
    assert!(wrapped.shares_storage_with(&extreme));

    let wrapping_negative_one = Tensor::zeros([3, 0, 6_148_914_691_236_517_205])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert!(matches!(
        wrapping_negative_one.flatten(1, 2),
        Err(TensorError::ReshapeAmbiguousZeroElements { .. })
    ));

    let symbolic_boundary = Tensor::zeros([0])
        .unwrap()
        .reshape([1_i64 << 31, 1_i64 << 32, 0])
        .unwrap()
        .transpose(0, 2)
        .unwrap();
    assert_eq!(
        symbolic_boundary.flatten(1, 2),
        Err(TensorError::FlattenNonConcreteInteger)
    );
    assert_eq!(
        TensorError::FlattenNonConcreteInteger.to_string(),
        "SymIntArrayRef expected to contain only concrete integers"
    );

    let wrapping_negative_two = Tensor::zeros([isize::MAX.unsigned_abs(), 0, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert!(matches!(
        wrapping_negative_two.flatten(1, 2),
        Err(TensorError::ReshapeInvalidDimension { dimension: -2, .. })
    ));
}

#[test]
fn flatten_normalizes_dimensions_and_reports_pytorch_errors() {
    let tensor = Tensor::zeros([2, 3, 4]).unwrap();
    assert_eq!(tensor.flatten(-3, -1).unwrap().shape(), [24]);
    assert_eq!(tensor.flatten(2, 1), Err(TensorError::FlattenStartAfterEnd));
    assert_eq!(
        tensor.flatten(-4, -1),
        Err(TensorError::DimensionOutOfRange {
            dimension: -4,
            rank: 3,
        })
    );
    assert_eq!(
        tensor.flatten(2, 1).unwrap_err().to_string(),
        "flatten() has invalid args: start_dim cannot come after end_dim"
    );

    let scalar = Tensor::from_vec(vec![1.0], []).unwrap();
    assert_eq!(
        scalar.flatten(1, 0),
        Err(TensorError::DimensionOutOfRange {
            dimension: 1,
            rank: 0,
        })
    );
    assert_eq!(
        tensor.collapse_dimensions(3, 3),
        Err(TensorError::DimensionOutOfRange {
            dimension: 3,
            rank: 3,
        })
    );
}

#[test]
fn ravel_reuses_contiguous_storage_and_packs_strided_inputs() {
    let source = Tensor::from_vec((0_u8..12).map(f32::from).collect(), [3, 4]).unwrap();

    let scalar = source.index([1, 2]).unwrap();
    let scalar_ravel = scalar.ravel().unwrap();
    assert_eq!(scalar_ravel.shape(), [1]);
    assert_eq!(scalar_ravel.stride(), [1]);
    assert_eq!(scalar_ravel.storage_offset(), 6);
    assert!(scalar_ravel.shares_storage_with(&source));

    let row = source.index_integer(1).unwrap();
    let row_ravel = row.ravel().unwrap();
    assert_eq!(row_ravel.shape(), [4]);
    assert_eq!(row_ravel.stride(), [1]);
    assert_eq!(row_ravel.storage_offset(), 4);
    assert!(row_ravel.shares_storage_with(&source));

    let ordinary = source.ravel().unwrap();
    assert_eq!(ordinary.shape(), [12]);
    assert_eq!(ordinary.stride(), [1]);
    assert!(ordinary.shares_storage_with(&source));

    // Size-one dimensions are contiguous regardless of their stride, and
    // PyTorch preserves that stride through contiguous().view(-1).
    let singleton_source = Tensor::from_vec((0_u8..4).map(f32::from).collect(), [1, 4]).unwrap();
    let singleton = singleton_source
        .transpose(0, 1)
        .unwrap()
        .index_integer(2)
        .unwrap();
    assert_eq!(singleton.shape(), [1]);
    assert_eq!(singleton.stride(), [4]);
    let singleton_ravel = singleton.ravel().unwrap();
    assert_eq!(singleton_ravel.shape(), [1]);
    assert_eq!(singleton_ravel.stride(), [4]);
    assert_eq!(singleton_ravel.storage_offset(), 2);
    assert!(singleton_ravel.shares_storage_with(&singleton_source));

    let strided_vector = source.transpose(0, 1).unwrap().index_integer(0).unwrap();
    assert_eq!(strided_vector.shape(), [3]);
    assert_eq!(strided_vector.stride(), [4]);
    let packed_vector = strided_vector.ravel().unwrap();
    assert_eq!(packed_vector.shape(), [3]);
    assert_eq!(packed_vector.stride(), [1]);
    assert_eq!(packed_vector.storage_offset(), 0);
    assert!(!packed_vector.shares_storage_with(&source));
    assert_eq!(packed_vector.as_slice(), [0.0, 4.0, 8.0]);

    let transposed = source.transpose(0, 1).unwrap();
    let packed = transposed.ravel().unwrap();
    assert_eq!(packed.shape(), [12]);
    assert_eq!(packed.stride(), [1]);
    assert_eq!(packed.storage_offset(), 0);
    assert!(!packed.shares_storage_with(&source));
    assert_eq!(
        packed.as_slice(),
        [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    );

    let empty_source = Tensor::zeros([2, 0, 3]).unwrap();
    let empty = empty_source
        .transpose(0, 2)
        .unwrap()
        .index_integer(1)
        .unwrap();
    let empty_ravel = empty.ravel().unwrap();
    assert_eq!(empty_ravel.shape(), [0]);
    assert_eq!(empty_ravel.stride(), [1]);
    assert_eq!(empty_ravel.storage_offset(), 1);
    assert!(empty_ravel.shares_storage_with(&empty_source));
}

#[test]
fn rank_two_matmul_reads_transposed_strides() {
    let left = Tensor::from_vec((0_u8..6).map(f32::from).collect(), [2, 3])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let right = Tensor::from_vec((0_u8..8).map(f32::from).collect(), [4, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let output = left.matmul(&right).unwrap();

    assert_eq!(output.shape(), [3, 4]);
    assert_eq!(output.stride(), [4, 1]);
    assert_eq!(
        output.as_slice(),
        [
            3.0, 9.0, 15.0, 21.0, 4.0, 14.0, 24.0, 34.0, 5.0, 19.0, 33.0, 47.0
        ]
    );
}

#[test]
fn rank_two_matmul_reads_indexed_transposed_offsets() {
    let left = Tensor::from_vec((0_u8..12).map(f32::from).collect(), [2, 3, 2])
        .unwrap()
        .index_integer(1)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let right = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 4, 3])
        .unwrap()
        .index_integer(1)
        .unwrap()
        .transpose(0, 1)
        .unwrap();

    assert_eq!(left.shape(), [2, 3]);
    assert_eq!(left.stride(), [1, 2]);
    assert_eq!(left.storage_offset(), 6);
    assert_eq!(right.shape(), [3, 4]);
    assert_eq!(right.stride(), [1, 3]);
    assert_eq!(right.storage_offset(), 12);

    let output = left.matmul(&right).unwrap();
    assert_eq!(output.shape(), [2, 4]);
    assert_eq!(output.stride(), [4, 1]);
    assert_eq!(
        output.as_slice(),
        [316.0, 388.0, 460.0, 532.0, 355.0, 436.0, 517.0, 598.0]
    );
}

#[test]
fn rank_two_matmul_preserves_offset_empty_results() {
    let left = Tensor::from_vec((0_u8..12).map(f32::from).collect(), [2, 3, 2])
        .unwrap()
        .index_integer(1)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let right = Tensor::zeros([2, 0, 3])
        .unwrap()
        .index_integer(1)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    assert_eq!(right.shape(), [3, 0]);
    assert_eq!(right.storage_offset(), 3);

    let output = left.matmul(&right).unwrap();
    assert_eq!(output.shape(), [2, 0]);
    assert_eq!(output.storage_offset(), 0);
    assert!(output.as_slice().is_empty());
}

#[test]
fn transpose_preserves_exact_empty_and_extreme_strides() {
    let maximum = isize::MAX.unsigned_abs();
    let empty = Tensor::zeros([2, 0, maximum]).unwrap();
    assert_eq!(empty.stride(), [maximum, maximum, 1]);

    let transposed = empty.transpose(0, 2).unwrap();
    assert_eq!(transposed.shape(), [maximum, 0, 2]);
    assert_eq!(transposed.stride(), [1, maximum, maximum]);
    assert!(transposed.shares_storage_with(&empty));
    assert!(transposed.logical_values().next().is_none());

    let offset = Tensor::zeros([maximum, 0])
        .unwrap()
        .index_integer(i64::MAX - 1)
        .unwrap()
        .transpose(0, 0)
        .unwrap();
    assert_eq!(offset.storage_offset(), maximum - 1);
    assert!(offset.logical_values().next().is_none());
    assert_eq!(offset.try_clone().unwrap().storage_offset(), 0);

    let overflow_order = Tensor::zeros([maximum, 0, 2, 2]).unwrap();
    assert_eq!(
        overflow_order.transpose(1, 3),
        Err(TensorError::ElementCountOverflow)
    );
    assert_eq!(
        overflow_order.transpose(-3, -1),
        Err(TensorError::ElementCountOverflow)
    );
    assert_eq!(overflow_order.transpose(1, 1).unwrap().numel(), 0);
}

#[test]
fn squeeze_is_a_metadata_only_shared_storage_view() {
    let source = Tensor::from_vec((0_u8..6).map(f32::from).collect(), [1, 2, 1, 3, 1]).unwrap();

    let squeezed = source.squeeze().unwrap();
    assert_eq!(squeezed.shape(), [2, 3]);
    assert_eq!(squeezed.stride(), [3, 1]);
    assert_eq!(squeezed.storage_offset(), source.storage_offset());
    assert_eq!(squeezed.dtype(), source.dtype());
    assert_eq!(squeezed.device(), source.device());
    assert_eq!(
        squeezed.logical_values().collect::<Vec<_>>(),
        source.as_slice()
    );
    assert!(squeezed.shares_storage_with(&source));
    assert!(squeezed.is_contiguous());

    let leading = source.squeeze_dim(0).unwrap();
    assert_eq!(leading.shape(), [2, 1, 3, 1]);
    assert_eq!(leading.stride(), [3, 3, 1, 1]);
    assert!(leading.shares_storage_with(&source));

    let selected = source.squeeze_dims([0, -1, 2]).unwrap();
    assert_eq!(selected.shape(), [2, 3]);
    assert_eq!(selected.stride(), [3, 1]);
    assert!(selected.shares_storage_with(&source));

    let unchanged = source.squeeze_dim(1).unwrap();
    assert_eq!(unchanged.shape(), source.shape());
    assert_eq!(unchanged.stride(), source.stride());
    assert!(unchanged.shares_storage_with(&source));

    let empty_selection = source.squeeze_dims([]).unwrap();
    assert_eq!(empty_selection.shape(), source.shape());
    assert_eq!(empty_selection.stride(), source.stride());
    assert!(empty_selection.shares_storage_with(&source));
}

#[test]
fn squeeze_preserves_non_contiguous_layouts_offsets_lifetimes_and_consumers() {
    let view = {
        let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 1, 3, 4]).unwrap();
        source
            .transpose(0, 3)
            .unwrap()
            .index_integer(1)
            .unwrap()
            .squeeze()
            .unwrap()
    };

    assert_eq!(view.shape(), [3, 2]);
    assert_eq!(view.stride(), [4, 12]);
    assert_eq!(view.storage_offset(), 1);
    assert!(!view.is_contiguous());
    assert_eq!(
        view.logical_values().collect::<Vec<_>>(),
        [1.0, 13.0, 5.0, 17.0, 9.0, 21.0]
    );

    let transposed = view.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), [2, 3]);
    assert_eq!(transposed.stride(), [12, 4]);
    assert_eq!(
        transposed.logical_values().collect::<Vec<_>>(),
        [1.0, 5.0, 9.0, 13.0, 17.0, 21.0]
    );

    let clone = view.try_clone().unwrap();
    assert_eq!(clone.shape(), view.shape());
    assert_eq!(clone.storage_offset(), 0);
    assert!(!clone.shares_storage_with(&view));
    assert_eq!(
        view.add_scalar(1.0)
            .unwrap()
            .logical_values()
            .collect::<Vec<_>>(),
        [2.0, 14.0, 6.0, 18.0, 10.0, 22.0]
    );
    assert_eq!(view.sum().item().unwrap().to_bits(), 66.0_f32.to_bits());
    assert_eq!(
        view.reshape([6]).unwrap().as_slice(),
        [1.0, 13.0, 5.0, 17.0, 9.0, 21.0]
    );
}

#[test]
fn squeeze_handles_scalars_empty_tensors_dimensions_and_high_ranks() {
    let scalar = Tensor::from_vec(vec![3.5], []).unwrap();
    for squeezed in [
        scalar.squeeze().unwrap(),
        scalar.squeeze_dim(0).unwrap(),
        scalar.squeeze_dim(-1).unwrap(),
        scalar.squeeze_dims([0]).unwrap(),
        scalar.squeeze_dims([]).unwrap(),
    ] {
        assert!(squeezed.shape().is_empty());
        assert!(squeezed.stride().is_empty());
        assert_eq!(squeezed.item().unwrap().to_bits(), 3.5_f32.to_bits());
        assert!(squeezed.shares_storage_with(&scalar));
    }

    let empty = Tensor::zeros([1, 0, 1, 2]).unwrap();
    let squeezed = empty.squeeze().unwrap();
    assert_eq!(squeezed.shape(), [0, 2]);
    assert_eq!(squeezed.stride(), [2, 1]);
    assert_eq!(squeezed.numel(), 0);
    assert!(squeezed.shares_storage_with(&empty));
    assert!(squeezed.logical_values().next().is_none());

    let high_rank = Tensor::zeros(vec![1; 65]).unwrap();
    assert!(high_rank.squeeze().unwrap().shape().is_empty());
    assert_eq!(high_rank.squeeze_dim(0).unwrap().shape().len(), 64);
    assert_eq!(
        high_rank.squeeze_dims([]),
        Err(TensorError::SqueezeDimensionsRankLimit)
    );
}

#[test]
fn squeeze_reports_pytorch_compatible_dimension_errors() {
    let tensor = Tensor::zeros([1, 2, 1]).unwrap();
    assert_eq!(
        tensor.squeeze_dims([0, -3]),
        Err(TensorError::DuplicateDimension { dimension: 0 })
    );
    assert_eq!(
        tensor.squeeze_dim(3),
        Err(TensorError::DimensionOutOfRange {
            dimension: 3,
            rank: 3,
        })
    );
    assert_eq!(
        tensor.squeeze_dims([0, 3]),
        Err(TensorError::DimensionOutOfRange {
            dimension: 3,
            rank: 3,
        })
    );
    assert_eq!(
        tensor.squeeze_dims([0, -3]).unwrap_err().to_string(),
        "dim 0 appears multiple times in the list of dims"
    );

    let scalar = Tensor::from_vec(vec![1.0], []).unwrap();
    for dimension in [-2, 1] {
        assert_eq!(
            scalar.squeeze_dim(dimension),
            Err(TensorError::DimensionOutOfRange { dimension, rank: 0 })
        );
    }
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
fn view_reuses_reshape_stride_analysis_without_copying() {
    let scalar = Tensor::from_vec(vec![-0.0], []).unwrap();
    let scalar_view = scalar.view([]).unwrap();
    assert!(scalar_view.shape().is_empty());
    assert!(scalar_view.stride().is_empty());
    assert!(scalar_view.shares_storage_with(&scalar));

    let empty = Tensor::zeros([2, 0, 3])
        .unwrap()
        .transpose(0, 2)
        .unwrap()
        .index_integer(1)
        .unwrap();
    let empty_view = empty.view([2, 0]).unwrap();
    assert_eq!(empty_view.shape(), [2, 0]);
    assert_eq!(empty_view.stride(), [1, 1]);
    assert_eq!(empty_view.storage_offset(), 1);
    assert!(empty_view.shares_storage_with(&empty));

    let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
    let offset = source.index_integer(1).unwrap();
    let offset_view = offset.view([2, 6]).unwrap();
    assert_eq!(offset_view.stride(), [6, 1]);
    assert_eq!(offset_view.storage_offset(), 12);
    assert!(offset_view.shares_storage_with(&offset));

    let non_contiguous = source.transpose(0, 1).unwrap();
    let compatible = non_contiguous.view([3, 2, 2, 2]).unwrap();
    assert_eq!(compatible.stride(), [4, 12, 2, 1]);
    assert!(compatible.shares_storage_with(&non_contiguous));

    let error = non_contiguous.view([6, 4]).unwrap_err();
    assert_eq!(error, TensorError::ViewIncompatibleLayout);
    assert_eq!(
        error.to_string(),
        "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead."
    );
    assert!(
        !non_contiguous
            .reshape([6, 4])
            .unwrap()
            .shares_storage_with(&non_contiguous)
    );

    assert_eq!(
        Tensor::zeros([6]).unwrap().view([2, 2]),
        Err(TensorError::ReshapeElementCountMismatch {
            shape: vec![2, 2],
            elements: 6,
        })
    );
}

#[test]
fn clone_deep_copies_a_views_logical_range_and_preserves_float_bits() {
    let values = [
        0.0_f32,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        f32::from_bits(0x7fc1_2345),
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        10.0,
        11.0,
    ];
    let source = Tensor::from_vec(values.to_vec(), [2, 2, 3]).unwrap();
    let view = source.index_integer(1).unwrap().reshape([3, 2]).unwrap();

    let copied = view.try_clone().unwrap();
    assert_eq!(copied.shape(), [3, 2]);
    assert_eq!(copied.stride(), [2, 1]);
    assert_eq!(copied.storage_offset(), 0);
    assert_eq!(copied.dtype(), view.dtype());
    assert_eq!(copied.device(), view.device());
    assert!(!copied.shares_storage_with(&source));
    assert!(!copied.shares_storage_with(&view));
    assert_eq!(
        copied
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values[6..]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let cloned_via_trait = view.clone();
    assert!(!cloned_via_trait.shares_storage_with(&view));
    assert_eq!(cloned_via_trait.shape(), copied.shape());
    assert_eq!(cloned_via_trait.stride(), copied.stride());
    assert_eq!(
        cloned_via_trait
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        copied
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn clone_handles_scalars_and_extreme_empty_view_offsets() {
    let scalar = Tensor::from_vec(vec![-0.0], []).unwrap();
    let scalar_copy = scalar.try_clone().unwrap();
    assert!(scalar_copy.shape().is_empty());
    assert!(scalar_copy.stride().is_empty());
    assert_eq!(scalar_copy.storage_offset(), 0);
    assert_eq!(scalar_copy.item().unwrap().to_bits(), (-0.0_f32).to_bits());
    assert!(!scalar_copy.shares_storage_with(&scalar));

    let maximum = usize::try_from(i64::MAX).unwrap();
    let empty = Tensor::zeros([maximum, 0]).unwrap();
    let view = empty
        .index_integer(i64::MAX - 1)
        .unwrap()
        .reshape([2, 0, 3])
        .unwrap();
    assert_eq!(view.storage_offset(), maximum - 1);

    let copied = view.try_clone().unwrap();
    assert_eq!(copied.shape(), [2, 0, 3]);
    assert_eq!(copied.stride(), [3, 3, 1]);
    assert_eq!(copied.storage_offset(), 0);
    assert_eq!(copied.numel(), 0);
    assert!(copied.as_slice().is_empty());
    assert!(!copied.shares_storage_with(&empty));
    assert!(!copied.shares_storage_with(&view));

    let unusual = Tensor::zeros([0, 1]).unwrap().add_scalar(1.0).unwrap();
    assert_eq!(unusual.stride(), [1, 0]);
    assert_eq!(
        unusual
            .try_clone_with_memory_format(MemoryFormat::Preserve)
            .unwrap()
            .stride(),
        [1, 0]
    );
    assert_eq!(
        unusual
            .try_clone_with_memory_format(MemoryFormat::Contiguous)
            .unwrap()
            .stride(),
        [1, 1]
    );

    let extreme_shape = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3])
        .unwrap();
    let extreme_copy = extreme_shape.try_clone().unwrap();
    assert_eq!(extreme_copy.shape(), extreme_shape.shape());
    assert_eq!(extreme_copy.stride(), extreme_shape.stride());
    assert_eq!(extreme_copy.storage_offset(), 0);
    assert!(!extreme_copy.shares_storage_with(&extreme_shape));
    assert_eq!(
        extreme_shape.try_clone_with_memory_format(MemoryFormat::Contiguous),
        Err(TensorError::StrideCalculationOverflow)
    );
    assert_eq!(
        extreme_shape.try_clone_with_memory_format(MemoryFormat::ChannelsLast),
        Err(TensorError::StrideCalculationOverflow)
    );

    let extreme_rank_four = Tensor::zeros([0])
        .unwrap()
        .reshape([0, i64::MAX, 3, 1])
        .unwrap();
    assert_eq!(
        extreme_rank_four.try_clone_with_memory_format(MemoryFormat::ChannelsLast3d),
        Err(TensorError::StrideCalculationOverflow)
    );
}

#[test]
fn clone_materializes_explicit_channels_last_storage() {
    let bit_patterns = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
        0x3f80_0000,
    ];
    let values = (0..144)
        .map(|index| f32::from_bits(bit_patterns[index % bit_patterns.len()]))
        .collect::<Vec<_>>();
    let contiguous = Tensor::from_vec(values[..48].to_vec(), [2, 3, 2, 4]).unwrap();
    let offset = Tensor::from_vec(values, [3, 2, 3, 2, 4])
        .unwrap()
        .index_integer(1)
        .unwrap();
    let strided = Tensor::from_vec(
        (0..48)
            .map(|index| f32::from_bits(bit_patterns[index % bit_patterns.len()]))
            .collect(),
        [2, 3, 2, 4],
    )
    .unwrap()
    .transpose(0, 3)
    .unwrap();
    let empty = Tensor::zeros([2, 0, 4, 5]).unwrap();

    for (source, expected_strides) in [
        (contiguous, [24, 1, 12, 3]),
        (offset, [24, 1, 12, 3]),
        (strided, [12, 1, 6, 3]),
        (empty, [0, 1, 0, 0]),
    ] {
        let expected_bits = source
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let copied = source
            .try_clone_with_memory_format(MemoryFormat::ChannelsLast)
            .unwrap();

        assert_eq!(copied.shape(), source.shape());
        assert_eq!(copied.stride(), expected_strides);
        assert_eq!(copied.storage_offset(), 0);
        assert_eq!(copied.dtype(), source.dtype());
        assert_eq!(copied.device(), source.device());
        assert!(copied.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));
        assert!(!copied.shares_storage_with(&source));
        assert_eq!(
            copied
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            expected_bits
        );
    }
}

#[test]
fn clone_materializes_explicit_channels_last_3d_storage() {
    let bit_patterns = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc5_4321,
        0x3f80_0000,
    ];

    let volume_values = (0..720)
        .map(|index| f32::from_bits(bit_patterns[index % bit_patterns.len()]))
        .collect::<Vec<_>>();
    let contiguous = Tensor::from_vec(volume_values[..240].to_vec(), [2, 3, 2, 4, 5]).unwrap();
    let offset = Tensor::from_vec(volume_values, [3, 2, 3, 2, 4, 5])
        .unwrap()
        .index_integer(1)
        .unwrap();
    let strided = Tensor::from_vec(
        (0..240)
            .map(|index| f32::from_bits(bit_patterns[index % bit_patterns.len()]))
            .collect(),
        [2, 3, 2, 4, 5],
    )
    .unwrap()
    .transpose(0, 4)
    .unwrap();
    let empty = Tensor::zeros([2, 0, 4, 5, 6]).unwrap();

    for (source, expected_strides) in [
        (contiguous, [120, 1, 60, 15, 3]),
        (offset, [120, 1, 60, 15, 3]),
        (strided, [48, 1, 24, 6, 3]),
        (empty, [0, 1, 0, 0, 0]),
    ] {
        let expected_bits = source
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let copied = source
            .try_clone_with_memory_format(MemoryFormat::ChannelsLast3d)
            .unwrap();

        assert_eq!(copied.shape(), source.shape());
        assert_eq!(copied.stride(), expected_strides);
        assert_eq!(copied.storage_offset(), 0);
        assert_eq!(copied.dtype(), source.dtype());
        assert_eq!(copied.device(), source.device());
        assert!(copied.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast3d));
        assert!(!copied.shares_storage_with(&source));
        assert_eq!(
            copied
                .logical_values()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            expected_bits
        );
    }

    let rank_four = Tensor::zeros([1, 2, 3, 4]).unwrap();
    assert_eq!(
        rank_four.try_clone_with_memory_format(MemoryFormat::ChannelsLast3d),
        Err(TensorError::ContiguousMemoryFormatRankMismatch {
            memory_format: MemoryFormat::ChannelsLast3d,
            expected_rank: 5,
            actual_rank: 4,
        })
    );
}

#[test]
fn contiguous_reuses_matching_storage_and_materializes_arbitrary_views() {
    let source = Tensor::from_vec(
        vec![
            0.0,
            -0.0,
            f32::from_bits(0x7fc1_2345),
            f32::INFINITY,
            f32::NEG_INFINITY,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
        ],
        [2, 2, 3],
    )
    .unwrap();
    let offset_contiguous = source.index_integer(1).unwrap();
    for memory_format in [MemoryFormat::Contiguous, MemoryFormat::Preserve] {
        let unchanged = offset_contiguous.try_contiguous(memory_format).unwrap();
        assert!(unchanged.shares_storage_with(&offset_contiguous));
        assert_eq!(unchanged.stride(), offset_contiguous.stride());
        assert_eq!(
            unchanged.storage_offset(),
            offset_contiguous.storage_offset()
        );
    }

    let view = source.transpose(0, 2).unwrap().squeeze().unwrap();
    let expected_bits = view.logical_values().map(f32::to_bits).collect::<Vec<_>>();
    let packed = view.try_contiguous(MemoryFormat::Contiguous).unwrap();
    assert_eq!(packed.shape(), view.shape());
    assert_eq!(packed.stride(), [4, 2, 1]);
    assert_eq!(packed.storage_offset(), 0);
    assert!(packed.is_contiguous());
    assert!(!packed.shares_storage_with(&view));
    assert_eq!(
        packed
            .logical_values()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        expected_bits
    );

    let repeated = packed.try_contiguous(MemoryFormat::Contiguous).unwrap();
    assert!(repeated.shares_storage_with(&packed));
    assert_eq!(repeated.storage_offset(), 0);
    assert_eq!(
        repeated
            .reshape([4, 3])
            .unwrap()
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        packed
            .as_slice()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let surviving_copy = {
        let temporary = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 4]).unwrap();
        let offset_view = temporary
            .transpose(0, 2)
            .unwrap()
            .index_integer(1)
            .unwrap()
            .squeeze()
            .unwrap();
        assert_eq!(offset_view.storage_offset(), 1);
        let output = offset_view
            .try_contiguous(MemoryFormat::Contiguous)
            .unwrap();
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.shares_storage_with(&offset_view));
        output
    };
    assert_eq!(surviving_copy.as_slice(), [1.0, 13.0, 5.0, 17.0, 9.0, 21.0]);
}

#[test]
fn contiguous_materializes_channels_last_and_channels_last_3d_storage() {
    let source = Tensor::from_vec((0_u8..24).map(f32::from).collect(), [2, 3, 2, 2]).unwrap();
    let channels_last = source.try_contiguous(MemoryFormat::ChannelsLast).unwrap();
    assert_eq!(channels_last.shape(), source.shape());
    assert_eq!(channels_last.stride(), [12, 1, 6, 3]);
    assert_eq!(channels_last.storage_offset(), 0);
    assert!(channels_last.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));
    assert!(!channels_last.is_contiguous());
    assert!(!channels_last.shares_storage_with(&source));
    assert_eq!(channels_last.try_to_vec().unwrap(), source.as_slice());

    let repeated = channels_last
        .try_contiguous(MemoryFormat::ChannelsLast)
        .unwrap();
    assert!(repeated.shares_storage_with(&channels_last));
    let cloned = channels_last.try_clone().unwrap();
    assert_eq!(cloned.stride(), channels_last.stride());
    assert!(!cloned.shares_storage_with(&channels_last));
    assert_eq!(cloned.try_to_vec().unwrap(), source.as_slice());
    let row_major = channels_last
        .try_contiguous(MemoryFormat::Contiguous)
        .unwrap();
    assert_eq!(row_major.stride(), source.stride());
    assert_eq!(row_major.as_slice(), source.as_slice());
    assert!(!row_major.shares_storage_with(&channels_last));
    assert_eq!(
        channels_last.add_scalar(1.0).unwrap().stride(),
        [12, 1, 6, 3]
    );
    assert_eq!(
        channels_last.sum().item().unwrap().to_bits(),
        276.0_f32.to_bits()
    );

    let volume = Tensor::from_vec((0_u8..48).map(f32::from).collect(), [2, 3, 2, 2, 2]).unwrap();
    let channels_last_3d = volume.try_contiguous(MemoryFormat::ChannelsLast3d).unwrap();
    assert_eq!(channels_last_3d.stride(), [24, 1, 12, 6, 3]);
    assert!(channels_last_3d.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast3d));
    assert_eq!(channels_last_3d.try_to_vec().unwrap(), volume.as_slice());
    assert!(!channels_last_3d.shares_storage_with(&volume));
}

#[test]
fn contiguous_handles_singleton_zero_scalar_and_high_rank_layouts() {
    let singleton = Tensor::zeros([2, 1, 4, 5]).unwrap();
    let singleton_result = singleton
        .try_contiguous(MemoryFormat::ChannelsLast)
        .unwrap();
    assert!(singleton_result.shares_storage_with(&singleton));
    assert_eq!(singleton_result.stride(), [20, 20, 5, 1]);

    for (shape, expected_strides) in [
        (vec![2, 0, 4, 5], vec![0, 1, 0, 0]),
        (vec![2, 3, 0, 5], vec![0, 1, 15, 3]),
        (vec![2, 3, 4, 0], vec![0, 1, 0, 3]),
        (vec![0, 3, 4, 5], vec![60, 1, 15, 3]),
    ] {
        let source = Tensor::zeros(shape).unwrap();
        let output = source.try_contiguous(MemoryFormat::ChannelsLast).unwrap();
        assert_eq!(output.stride(), expected_strides);
        assert_eq!(output.storage_offset(), 0);
        assert!(!output.shares_storage_with(&source));
        assert!(output.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));
    }

    let empty_volume = Tensor::zeros([2, 3, 4, 0, 6]).unwrap();
    let empty_volume = empty_volume
        .try_contiguous(MemoryFormat::ChannelsLast3d)
        .unwrap();
    assert_eq!(empty_volume.stride(), [0, 1, 0, 18, 3]);

    let scalar = Tensor::from_vec(vec![-0.0], []).unwrap();
    let scalar_result = scalar.try_contiguous(MemoryFormat::Contiguous).unwrap();
    assert!(scalar_result.shares_storage_with(&scalar));
    assert_eq!(
        scalar_result.item().unwrap().to_bits(),
        (-0.0_f32).to_bits()
    );

    let mut high_rank_shape = vec![1; 128];
    high_rank_shape[3] = 2;
    high_rank_shape[117] = 2;
    let high_rank = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], high_rank_shape)
        .unwrap()
        .transpose(3, 117)
        .unwrap();
    let packed = high_rank.try_contiguous(MemoryFormat::Contiguous).unwrap();
    assert_eq!(packed.shape().len(), 128);
    assert!(packed.is_contiguous());
    assert_eq!(
        packed.try_to_vec().unwrap(),
        high_rank.try_to_vec().unwrap()
    );
}

#[test]
fn contiguous_validates_preserve_rank_and_stride_overflow_like_pytorch() {
    let non_contiguous = Tensor::zeros([2, 3]).unwrap().transpose(0, 1).unwrap();
    assert_eq!(
        non_contiguous.try_contiguous(MemoryFormat::Preserve),
        Err(TensorError::ContiguousPreserveFormatUnsupported)
    );
    assert_eq!(
        non_contiguous
            .try_contiguous(MemoryFormat::Preserve)
            .unwrap_err()
            .to_string(),
        "preserve memory format is unsupported by the contiguous operator"
    );

    for rank in 0..=6 {
        let tensor = Tensor::zeros(vec![2; rank]).unwrap();
        if rank != 4 {
            assert_eq!(
                tensor.try_contiguous(MemoryFormat::ChannelsLast),
                Err(TensorError::ContiguousMemoryFormatRankMismatch {
                    memory_format: MemoryFormat::ChannelsLast,
                    expected_rank: 4,
                    actual_rank: rank,
                })
            );
        }
        if rank != 5 {
            assert_eq!(
                tensor.try_contiguous(MemoryFormat::ChannelsLast3d),
                Err(TensorError::ContiguousMemoryFormatRankMismatch {
                    memory_format: MemoryFormat::ChannelsLast3d,
                    expected_rank: 5,
                    actual_rank: rank,
                })
            );
        }
    }
    assert_eq!(
        Tensor::zeros([2, 3, 4])
            .unwrap()
            .try_contiguous(MemoryFormat::ChannelsLast)
            .unwrap_err()
            .to_string(),
        "required rank 4 tensor to use channels_last format"
    );

    let maximum = i64::MAX;
    let extreme = Tensor::zeros([0])
        .unwrap()
        .reshape([2, 0, maximum, maximum])
        .unwrap();
    assert_eq!(
        extreme.try_contiguous(MemoryFormat::ChannelsLast),
        Err(TensorError::StrideCalculationOverflow)
    );

    let wrapping_identity = Tensor::zeros([0])
        .unwrap()
        .reshape([0, 1, 1_i64 << 62, 1_i64 << 32])
        .unwrap();
    assert_eq!(wrapping_identity.stride(), [0, 0, 1_usize << 32, 1]);
    assert!(wrapping_identity.is_contiguous_with_memory_format(MemoryFormat::ChannelsLast));
    let unchanged = wrapping_identity
        .try_contiguous(MemoryFormat::ChannelsLast)
        .unwrap();
    assert!(unchanged.shares_storage_with(&wrapping_identity));
    assert_eq!(unchanged.stride(), wrapping_identity.stride());
    assert_eq!(
        unchanged.storage_offset(),
        wrapping_identity.storage_offset()
    );
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
