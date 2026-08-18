use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::memory_format::MemoryFormat;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    ShapeDataMismatch {
        shape: Vec<usize>,
        elements: usize,
    },
    ShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    MatmulRequiresMatrices {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    MatmulInnerDimensionMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    ItemRequiresOneElement {
        elements: usize,
    },
    InvalidScalarIndex,
    TooManyIndices {
        dimensions: usize,
    },
    IndexOutOfBounds {
        index: i64,
        dimension: usize,
        size: usize,
    },
    InvalidStorageOffset {
        offset: i64,
    },
    IndexCalculationOverflow,
    DimensionOutOfRange {
        dimension: i64,
        rank: usize,
    },
    PermutationRankMismatch {
        dimensions: usize,
        rank: usize,
    },
    PermutationDimensionOutOfRange {
        dimension: usize,
        rank: usize,
    },
    DuplicatePermutationDimension {
        dimension: usize,
    },
    MatrixTransposeRequiresMatrix {
        rank: usize,
    },
    DuplicateDimension {
        dimension: usize,
    },
    SqueezeDimensionsRankLimit,
    FlattenStartAfterEnd,
    FlattenNonConcreteInteger,
    ReshapeMultipleInferredDimensions,
    ReshapeInvalidDimension {
        dimension: i64,
        index: usize,
        shape: Vec<i64>,
    },
    ReshapeAmbiguousZeroElements {
        shape: Vec<i64>,
    },
    ReshapeElementCountMismatch {
        shape: Vec<i64>,
        elements: usize,
    },
    ViewIncompatibleLayout,
    ElementCountOverflow,
    StrideCalculationOverflow,
    StorageCapacityOverflow {
        elements: usize,
    },
    AllocationFailed {
        elements: usize,
    },
    UnsupportedMemoryFormat {
        memory_format: MemoryFormat,
    },
    ContiguousPreserveFormatUnsupported,
    ContiguousMemoryFormatRankMismatch {
        memory_format: MemoryFormat,
        expected_rank: usize,
        actual_rank: usize,
    },
    BackwardRequiresScalar {
        elements: usize,
    },
    DoesNotRequireGrad,
    BackwardGraphFreed,
}

impl Display for TensorError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeDataMismatch { shape, elements } => write!(
                formatter,
                "shape {shape:?} does not describe {elements} elements"
            ),
            Self::ShapeMismatch { left, right } => format_shape_mismatch(formatter, left, right),
            Self::MatmulRequiresMatrices { left, right } => write!(
                formatter,
                "matmul currently requires two rank-2 tensors, got {left:?} and {right:?}"
            ),
            Self::MatmulInnerDimensionMismatch { left, right } => {
                format_matmul_inner_dimension_mismatch(formatter, left, right)
            }
            Self::ItemRequiresOneElement { elements } => {
                write!(formatter, "item requires one element, got {elements}")
            }
            Self::InvalidScalarIndex => write!(
                formatter,
                "invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number"
            ),
            Self::TooManyIndices { dimensions } => {
                write!(
                    formatter,
                    "too many indices for tensor of dimension {dimensions}"
                )
            }
            Self::IndexOutOfBounds {
                index,
                dimension,
                size,
            } => write!(
                formatter,
                "index {index} is out of bounds for dimension {dimension} with size {size}"
            ),
            Self::InvalidStorageOffset { offset } => {
                write!(formatter, "Tensor: invalid storage offset {offset}")
            }
            Self::IndexCalculationOverflow => {
                write!(formatter, "tensor index calculation overflowed usize")
            }
            Self::DimensionOutOfRange { dimension, rank } => {
                format_dimension_out_of_range(formatter, *dimension, *rank)
            }
            error @ (Self::PermutationRankMismatch { .. }
            | Self::PermutationDimensionOutOfRange { .. }
            | Self::DuplicatePermutationDimension { .. }
            | Self::MatrixTransposeRequiresMatrix { .. }) => {
                format_permutation_error(formatter, error)
            }
            error @ (Self::DuplicateDimension { .. } | Self::SqueezeDimensionsRankLimit) => {
                format_squeeze_error(formatter, error)
            }
            Self::FlattenStartAfterEnd => format_flatten_error(formatter),
            Self::FlattenNonConcreteInteger => format_flatten_non_concrete_error(formatter),
            Self::ReshapeMultipleInferredDimensions => format_reshape_inference_error(formatter),
            Self::ReshapeInvalidDimension {
                dimension,
                index,
                shape,
            } => write!(
                formatter,
                "invalid shape dimension {dimension} at index {index} of shape {shape:?}"
            ),
            Self::ReshapeAmbiguousZeroElements { shape } => write!(
                formatter,
                "cannot reshape tensor of 0 elements into shape {shape:?} because the unspecified dimension size -1 can be any value and is ambiguous"
            ),
            Self::ReshapeElementCountMismatch { shape, elements } => write!(
                formatter,
                "shape '{shape:?}' is invalid for input of size {elements}"
            ),
            Self::ViewIncompatibleLayout => formatter.write_str(
                "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.",
            ),
            Self::ElementCountOverflow => {
                formatter.write_str("tensor element count overflowed usize")
            }
            Self::StrideCalculationOverflow => formatter.write_str("Stride calculation overflowed"),
            Self::StorageCapacityOverflow { elements } => write!(
                formatter,
                "storage for a tensor with {elements} elements exceeds the platform capacity"
            ),
            Self::AllocationFailed { elements } => {
                write!(
                    formatter,
                    "failed to allocate storage for {elements} elements"
                )
            }
            error @ (Self::UnsupportedMemoryFormat { .. }
            | Self::ContiguousPreserveFormatUnsupported
            | Self::ContiguousMemoryFormatRankMismatch { .. }) => {
                format_memory_format_error(formatter, error)
            }
            error @ (Self::BackwardRequiresScalar { .. }
            | Self::DoesNotRequireGrad
            | Self::BackwardGraphFreed) => format_autograd_error(formatter, error),
        }
    }
}

impl Error for TensorError {}

fn format_shape_mismatch(
    formatter: &mut Formatter<'_>,
    left: &[usize],
    right: &[usize],
) -> std::fmt::Result {
    if let Some((left_dimension, right_dimension, axis)) = first_broadcast_mismatch(left, right) {
        write!(
            formatter,
            "The size of tensor a ({left_dimension}) must match the size of tensor b ({right_dimension}) at non-singleton dimension {axis}"
        )
    } else {
        write!(
            formatter,
            "tensor shapes are not broadcastable: {left:?} and {right:?}"
        )
    }
}

fn format_matmul_inner_dimension_mismatch(
    formatter: &mut Formatter<'_>,
    left: &[usize],
    right: &[usize],
) -> std::fmt::Result {
    match (left, right) {
        ([rows, inner], [other_inner, columns]) => write!(
            formatter,
            "mat1 and mat2 shapes cannot be multiplied ({rows}x{inner} and {other_inner}x{columns})"
        ),
        _ => write!(
            formatter,
            "matmul inner dimensions differ for {left:?} and {right:?}"
        ),
    }
}

fn format_dimension_out_of_range(
    formatter: &mut Formatter<'_>,
    dimension: i64,
    rank: usize,
) -> std::fmt::Result {
    let rank = rank.max(1);
    write!(
        formatter,
        "Dimension out of range (expected to be in range of [-{rank}, {}], but got {dimension})",
        rank - 1
    )
}

fn format_permutation_error(
    formatter: &mut Formatter<'_>,
    error: &TensorError,
) -> std::fmt::Result {
    match error {
        TensorError::PermutationRankMismatch { dimensions, rank } => write!(
            formatter,
            "number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = {rank} is not equal to len(dims) = {dimensions}"
        ),
        TensorError::PermutationDimensionOutOfRange { dimension, rank } => write!(
            formatter,
            "permutation dimension {dimension} is out of range for tensor rank {rank}"
        ),
        TensorError::DuplicatePermutationDimension { .. } => {
            formatter.write_str("permute(): duplicate dims are not allowed.")
        }
        TensorError::MatrixTransposeRequiresMatrix { rank } => write!(
            formatter,
            "tensor.mT is only supported on matrices or batches of matrices. Got {rank}-D tensor."
        ),
        _ => unreachable!("only dimension-permutation errors are formatted here"),
    }
}

fn format_squeeze_error(formatter: &mut Formatter<'_>, error: &TensorError) -> std::fmt::Result {
    match error {
        TensorError::DuplicateDimension { dimension } => {
            write!(
                formatter,
                "dim {dimension} appears multiple times in the list of dims"
            )
        }
        TensorError::SqueezeDimensionsRankLimit => {
            formatter.write_str("only tensors with up to 64 dims are supported")
        }
        _ => unreachable!("only squeeze-specific errors are formatted here"),
    }
}

fn format_flatten_error(formatter: &mut Formatter<'_>) -> std::fmt::Result {
    formatter.write_str("flatten() has invalid args: start_dim cannot come after end_dim")
}

fn format_flatten_non_concrete_error(formatter: &mut Formatter<'_>) -> std::fmt::Result {
    formatter.write_str("SymIntArrayRef expected to contain only concrete integers")
}

fn format_reshape_inference_error(formatter: &mut Formatter<'_>) -> std::fmt::Result {
    formatter.write_str("only one dimension can be inferred")
}

fn format_memory_format_error(
    formatter: &mut Formatter<'_>,
    error: &TensorError,
) -> std::fmt::Result {
    match error {
        TensorError::UnsupportedMemoryFormat { memory_format } => write!(
            formatter,
            "clone with memory format torch.{memory_format} is not supported"
        ),
        TensorError::ContiguousPreserveFormatUnsupported => {
            formatter.write_str("preserve memory format is unsupported by the contiguous operator")
        }
        TensorError::ContiguousMemoryFormatRankMismatch {
            memory_format,
            expected_rank,
            ..
        } => write!(
            formatter,
            "required rank {expected_rank} tensor to use {memory_format} format"
        ),
        _ => unreachable!("only memory-format errors are formatted here"),
    }
}

fn format_autograd_error(formatter: &mut Formatter<'_>, error: &TensorError) -> std::fmt::Result {
    match error {
        TensorError::BackwardRequiresScalar { elements } => write!(
            formatter,
            "grad can be implicitly created only for scalar outputs (output has {elements} elements)"
        ),
        TensorError::DoesNotRequireGrad => formatter.write_str(
            "element 0 of tensors does not require grad and does not have a grad_fn",
        ),
        TensorError::BackwardGraphFreed => formatter.write_str(
            "Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.",
        ),
        _ => unreachable!("only autograd errors are formatted here"),
    }
}

fn first_broadcast_mismatch(left: &[usize], right: &[usize]) -> Option<(usize, usize, usize)> {
    let rank = left.len().max(right.len());
    let mut left_dimensions = left.iter().rev().copied();
    let mut right_dimensions = right.iter().rev().copied();
    for trailing_axis in 0..rank {
        let axis = rank - trailing_axis - 1;
        let left_dimension = left_dimensions.next().unwrap_or(1);
        let right_dimension = right_dimensions.next().unwrap_or(1);
        if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
            return Some((left_dimension, right_dimension, axis));
        }
    }
    None
}
