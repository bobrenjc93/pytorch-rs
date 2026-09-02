//! Python exception translation for native tensor errors.

use pyo3::PyErr;
use pyo3::exceptions::{PyIndexError, PyRuntimeError};

use crate::TensorError;

pub(crate) fn tensor_error(error: &TensorError) -> PyErr {
    match error {
        TensorError::ShapeDataMismatch { .. }
        | TensorError::ShapeMismatch { .. }
        | TensorError::DotRequiresVectors { .. }
        | TensorError::DotElementCountMismatch { .. }
        | TensorError::MatmulRequiresMatrices { .. }
        | TensorError::MatmulInnerDimensionMismatch { .. }
        | TensorError::ItemRequiresOneElement { .. }
        | TensorError::InvalidStorageOffset { .. }
        | TensorError::IndexCalculationOverflow
        | TensorError::ReshapeMultipleInferredDimensions
        | TensorError::ReshapeInvalidDimension { .. }
        | TensorError::ReshapeAmbiguousZeroElements { .. }
        | TensorError::ReshapeElementCountMismatch { .. }
        | TensorError::ViewIncompatibleLayout
        | TensorError::StrideCalculationOverflow
        | TensorError::NegativeStrides { .. }
        | TensorError::StorageCapacityOverflow { .. }
        | TensorError::AllocationFailed { .. }
        | TensorError::UnsupportedMemoryFormat { .. }
        | TensorError::ContiguousPreserveFormatUnsupported
        | TensorError::ContiguousMemoryFormatRankMismatch { .. }
        | TensorError::PermutationRankMismatch { .. }
        | TensorError::PermutationDimensionOutOfRange { .. }
        | TensorError::DuplicatePermutationDimension { .. }
        | TensorError::MatrixTransposeRequiresMatrix { .. }
        | TensorError::DuplicateDimension { .. }
        | TensorError::SqueezeDimensionsRankLimit
        | TensorError::FlattenStartAfterEnd
        | TensorError::FlattenNonConcreteInteger
        | TensorError::NonConcreteInteger
        | TensorError::ElementCountOverflow
        | TensorError::BackwardRequiresScalar { .. }
        | TensorError::AutogradRecordingUnsupported { .. }
        | TensorError::DoesNotRequireGrad
        | TensorError::DoesNotRequireGradAt { .. }
        | TensorError::BackwardGraphFreed => PyRuntimeError::new_err(error.to_string()),
        TensorError::InvalidScalarIndex
        | TensorError::SliceCannotApplyToScalar
        | TensorError::TooManyIndices { .. }
        | TensorError::IndexOutOfBounds { .. }
        | TensorError::DimensionOutOfRange { .. } => PyIndexError::new_err(error.to_string()),
    }
}

pub(crate) fn item_error(error: &TensorError) -> PyErr {
    if let TensorError::ItemRequiresOneElement { elements } = error {
        PyRuntimeError::new_err(format!(
            "a Tensor with {elements} elements cannot be converted to Scalar"
        ))
    } else {
        tensor_error(error)
    }
}

pub(crate) fn transpose_error(error: &TensorError) -> PyErr {
    if matches!(error, TensorError::ElementCountOverflow) {
        PyRuntimeError::new_err("numel: integer multiplication overflow")
    } else {
        tensor_error(error)
    }
}

pub(crate) fn permute_error(error: &TensorError) -> PyErr {
    if matches!(error, TensorError::PermutationRankMismatch { .. }) {
        PyRuntimeError::new_err(format!("permute(sparse_coo): {error}"))
    } else if matches!(error, TensorError::ElementCountOverflow) {
        PyRuntimeError::new_err("numel: integer multiplication overflow")
    } else {
        tensor_error(error)
    }
}
