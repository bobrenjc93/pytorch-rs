use pytorch_rs::{DType, Device, Tensor, TensorError};

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.logical_values().collect()
}

#[test]
fn matmul_supports_every_rank_one_and_rank_two_pair() {
    let vector = Tensor::from_vec(vec![1.0, 2.0, 3.0], [3]).unwrap();
    let other_vector = Tensor::from_vec(vec![4.0, 5.0, 6.0], [3]).unwrap();
    let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
    let right_matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]).unwrap();

    let dot = vector.matmul(&other_vector).unwrap();
    assert!(dot.shape().is_empty());
    assert!(dot.stride().is_empty());
    assert_eq!(dot.item().unwrap().to_bits(), 32.0_f32.to_bits());

    let matrix_vector = matrix.matmul(&other_vector).unwrap();
    assert_eq!(matrix_vector.shape(), [2]);
    assert_eq!(matrix_vector.stride(), [1]);
    assert_eq!(values(&matrix_vector), [32.0, 77.0]);

    let vector_matrix = vector.matmul(&right_matrix).unwrap();
    assert_eq!(vector_matrix.shape(), [2]);
    assert_eq!(vector_matrix.stride(), [1]);
    assert_eq!(values(&vector_matrix), [22.0, 28.0]);

    let matrix_matrix = matrix.matmul(&right_matrix).unwrap();
    assert_eq!(matrix_matrix.shape(), [2, 2]);
    assert_eq!(matrix_matrix.stride(), [2, 1]);
    assert_eq!(values(&matrix_matrix), [22.0, 28.0, 49.0, 64.0]);

    for output in [dot, matrix_vector, vector_matrix, matrix_matrix] {
        assert_eq!(output.dtype(), DType::Float32);
        assert_eq!(output.device(), Device::Cpu);
    }
}

#[test]
fn matmul_reads_rank_one_and_rank_two_view_strides_and_offsets() {
    let vectors = Tensor::from_vec(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], [3, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let left_vector = vectors.index_integer(0).unwrap();
    let right_vector = vectors.index_integer(1).unwrap();
    assert_eq!(left_vector.stride(), [2]);
    assert_eq!(right_vector.stride(), [2]);
    assert_eq!(right_vector.storage_offset(), 1);
    assert_eq!(left_vector.matmul(&right_vector).unwrap().item(), Ok(140.0));

    let left_matrix = Tensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [3, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let matrix_vector = left_matrix.matmul(&right_vector).unwrap();
    assert_eq!(values(&matrix_vector), [140.0, 320.0]);

    let right_matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let short_vector = Tensor::from_vec(vec![2.0, 99.0, 3.0, 99.0], [2, 2])
        .unwrap()
        .transpose(0, 1)
        .unwrap()
        .index_integer(0)
        .unwrap();
    assert_eq!(short_vector.stride(), [2]);
    assert_eq!(right_matrix.stride(), [1, 2]);
    assert_eq!(
        values(&short_vector.matmul(&right_matrix).unwrap()),
        [8.0, 18.0]
    );
}

#[test]
fn matmul_handles_empty_inner_and_output_dimensions() {
    let dot = Tensor::zeros([0])
        .unwrap()
        .matmul(&Tensor::zeros([0]).unwrap())
        .unwrap();
    assert!(dot.shape().is_empty());
    assert_eq!(dot.item().unwrap().to_bits(), 0.0_f32.to_bits());

    let matrix_vector = Tensor::zeros([2, 0])
        .unwrap()
        .matmul(&Tensor::zeros([0]).unwrap())
        .unwrap();
    assert_eq!(matrix_vector.shape(), [2]);
    assert_eq!(values(&matrix_vector), [0.0, 0.0]);

    let vector_matrix = Tensor::zeros([0])
        .unwrap()
        .matmul(&Tensor::zeros([0, 3]).unwrap())
        .unwrap();
    assert_eq!(vector_matrix.shape(), [3]);
    assert_eq!(values(&vector_matrix), [0.0, 0.0, 0.0]);

    let empty_output = Tensor::zeros([0, 3])
        .unwrap()
        .matmul(&Tensor::zeros([3]).unwrap())
        .unwrap();
    assert_eq!(empty_output.shape(), [0]);
    assert_eq!(empty_output.stride(), [1]);
    assert!(values(&empty_output).is_empty());

    let maximum = isize::MAX.unsigned_abs();
    let extreme_empty = Tensor::zeros([0, maximum])
        .unwrap()
        .matmul(&Tensor::zeros([maximum, 0]).unwrap())
        .unwrap();
    assert_eq!(extreme_empty.shape(), [0, 0]);
    assert_eq!(extreme_empty.stride(), [1, 1]);
}

#[test]
fn matmul_uses_ordered_float32_accumulation_and_ieee_values() {
    let cancellation = Tensor::from_vec(vec![1.0e20, 1.0, -1.0e20], [3])
        .unwrap()
        .matmul(&Tensor::ones([3]).unwrap())
        .unwrap();
    assert_eq!(cancellation.item().unwrap().to_bits(), 0.0_f32.to_bits());

    let signed_zero = Tensor::from_vec(vec![-0.0], [1])
        .unwrap()
        .matmul(&Tensor::ones([1]).unwrap())
        .unwrap();
    assert_eq!(signed_zero.item().unwrap().to_bits(), 0.0_f32.to_bits());

    for (left, right) in [
        (vec![f32::INFINITY], vec![0.0]),
        (vec![f32::NAN], vec![1.0]),
        (vec![f32::INFINITY, f32::NEG_INFINITY], vec![1.0, 1.0]),
    ] {
        assert!(
            Tensor::from_vec(left.clone(), [left.len()])
                .unwrap()
                .matmul(&Tensor::from_vec(right.clone(), [right.len()]).unwrap())
                .unwrap()
                .item()
                .unwrap()
                .is_nan()
        );
    }
}

#[test]
fn matmul_reports_pytorch_rank_and_pair_specific_shape_errors() {
    let cases = [
        (
            Tensor::zeros([]).unwrap(),
            Tensor::zeros([2]).unwrap(),
            "both arguments to matmul need to be at least 1D, but they are 0D and 1D",
        ),
        (
            Tensor::zeros([2]).unwrap(),
            Tensor::zeros([]).unwrap(),
            "both arguments to matmul need to be at least 1D, but they are 1D and 0D",
        ),
        (
            Tensor::zeros([2]).unwrap(),
            Tensor::zeros([3]).unwrap(),
            "inconsistent tensor size, expected tensor [2] and src [3] to have the same number of elements, but got 2 and 3 elements respectively",
        ),
        (
            Tensor::zeros([2, 3]).unwrap(),
            Tensor::zeros([4]).unwrap(),
            "size mismatch, got input (2), mat (2x3), vec (4)",
        ),
        (
            Tensor::zeros([3]).unwrap(),
            Tensor::zeros([4, 2]).unwrap(),
            "mat1 and mat2 shapes cannot be multiplied (1x3 and 4x2)",
        ),
        (
            Tensor::zeros([2, 3]).unwrap(),
            Tensor::zeros([4, 2]).unwrap(),
            "mat1 and mat2 shapes cannot be multiplied (2x3 and 4x2)",
        ),
    ];

    for (left, right, message) in cases {
        assert_eq!(left.matmul(&right).unwrap_err().to_string(), message);
    }

    assert!(matches!(
        Tensor::zeros([2])
            .unwrap()
            .matmul(&Tensor::zeros([3]).unwrap()),
        Err(TensorError::MatmulInnerDimensionMismatch { .. })
    ));
    assert!(matches!(
        Tensor::zeros([])
            .unwrap()
            .matmul(&Tensor::zeros([1]).unwrap()),
        Err(TensorError::MatmulRequiresMatrices { .. })
    ));
}

#[test]
fn matmul_checks_result_storage_capacity_before_allocation() {
    let maximum = isize::MAX.unsigned_abs();
    let error = Tensor::zeros([0])
        .unwrap()
        .matmul(&Tensor::zeros([0, maximum]).unwrap())
        .unwrap_err();
    assert_eq!(
        error,
        TensorError::StorageCapacityOverflow { elements: maximum }
    );

    let overflow = Tensor::zeros([usize::MAX, 0])
        .unwrap()
        .matmul(&Tensor::zeros([0, 2]).unwrap())
        .unwrap_err();
    assert_eq!(overflow, TensorError::ElementCountOverflow);
}
