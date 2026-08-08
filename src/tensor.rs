use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::sync::{Arc, OnceLock};

/// Native scalar types implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE 754 single-precision floating point.
    #[default]
    Float32,
}

impl Display for DType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float32 => formatter.write_str("float32"),
        }
    }
}

/// Native execution devices implemented by tensor storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host CPU memory and kernels.
    #[default]
    Cpu,
}

impl Display for Device {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
        }
    }
}

struct Storage {
    data: Vec<f32>,
    dtype: DType,
    device: Device,
}

/// A row-major tensor view over immutable, reference-counted native storage.
///
/// Strides are measured in elements and may describe a positive-stride,
/// non-contiguous layout. Every non-empty layout is validated to remain inside
/// its storage span. Empty layouts may carry a virtual offset, matching
/// `PyTorch`'s metadata behavior without accessing storage.
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    elements: usize,
    logical_cache: OnceLock<Vec<f32>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            elements: self.elements,
            logical_cache: OnceLock::new(),
        }
    }
}

/// One dimension of a basic tensor index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorIndex {
    /// Removes a dimension by selecting one element. Negative values wrap.
    Integer(i64),
    /// Retains a dimension while selecting a positive-step range.
    Slice(Slice),
}

/// Bounds and step for one basic slice dimension.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Slice {
    pub start: Option<i64>,
    pub stop: Option<i64>,
    pub step: Option<i64>,
}

impl Slice {
    /// Creates a slice with optional bounds and step.
    #[must_use]
    pub const fn new(
        bound_start: Option<i64>,
        bound_end: Option<i64>,
        step_size: Option<i64>,
    ) -> Self {
        Self {
            start: bound_start,
            stop: bound_end,
            step: step_size,
        }
    }

    /// Creates the full-dimension slice `..`.
    #[must_use]
    pub const fn full() -> Self {
        Self::new(None, None, None)
    }

    fn normalize(self, size: usize) -> Result<NormalizedSlice, TensorError> {
        let size = i64::try_from(size).map_err(|_| TensorError::IndexCalculationOverflow)?;
        let step_size = self.step.unwrap_or(1);
        if step_size <= 0 {
            return Err(TensorError::SliceStepMustBePositive { step: step_size });
        }

        let first_index = normalize_slice_bound(self.start.unwrap_or(0), size);
        let end_index = normalize_slice_bound(self.stop.unwrap_or(size), size);
        let length = if end_index <= first_index {
            0
        } else {
            let distance = end_index
                .checked_sub(first_index)
                .and_then(|value| value.checked_sub(1))
                .ok_or(TensorError::IndexCalculationOverflow)?;
            distance
                .checked_div(step_size)
                .and_then(|value| value.checked_add(1))
                .ok_or(TensorError::IndexCalculationOverflow)?
        };

        Ok(NormalizedSlice {
            start: usize::try_from(first_index)
                .map_err(|_| TensorError::IndexCalculationOverflow)?,
            length: usize::try_from(length).map_err(|_| TensorError::IndexCalculationOverflow)?,
            step: usize::try_from(step_size).map_err(|_| TensorError::IndexCalculationOverflow)?,
        })
    }
}

#[derive(Clone, Copy)]
struct NormalizedSlice {
    start: usize,
    length: usize,
    step: usize,
}

pub(crate) struct CheckedSlice {
    pub(crate) offset: usize,
    length: usize,
    stride: usize,
}

fn normalize_slice_bound(bound: i64, size: i64) -> i64 {
    if bound < 0 {
        bound.saturating_add(size).max(0)
    } else {
        bound.min(size)
    }
}

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
    SliceStepMustBePositive {
        step: i64,
    },
    LayoutOutOfStorage {
        offset: usize,
        span: usize,
        storage_elements: usize,
    },
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
    ElementCountOverflow,
    StrideCalculationOverflow,
    StorageCapacityOverflow {
        elements: usize,
    },
    AllocationFailed {
        elements: usize,
    },
}

impl Display for TensorError {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeDataMismatch { shape, elements } => write!(
                formatter,
                "shape {shape:?} does not describe {elements} elements"
            ),
            Self::ShapeMismatch { left, right } => {
                if let Some((left_dimension, right_dimension, axis)) =
                    first_broadcast_mismatch(left, right)
                {
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
            Self::MatmulRequiresMatrices { left, right } => write!(
                formatter,
                "matmul currently requires two rank-2 tensors, got {left:?} and {right:?}"
            ),
            Self::MatmulInnerDimensionMismatch { left, right } => write!(
                formatter,
                "matmul inner dimensions differ for {left:?} and {right:?}"
            ),
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
            Self::SliceStepMustBePositive { step } => {
                write!(
                    formatter,
                    "slice step must be greater than zero, got {step}"
                )
            }
            Self::LayoutOutOfStorage {
                offset,
                span,
                storage_elements,
            } => write!(
                formatter,
                "tensor layout at offset {offset} with span {span} exceeds storage of {storage_elements} elements"
            ),
            Self::ReshapeMultipleInferredDimensions => {
                write!(formatter, "only one dimension can be inferred")
            }
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
            Self::ElementCountOverflow => {
                write!(formatter, "tensor element count overflowed usize")
            }
            Self::StrideCalculationOverflow => {
                write!(formatter, "Stride calculation overflowed")
            }
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
        }
    }
}

impl Error for TensorError {}

// Preserve the original value-oriented debug representation; storage identity
// and layout bookkeeping are implementation details.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("data", &self.as_slice())
            .field("shape", &self.shape)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.dtype() == other.dtype()
            && self.device() == other.device()
            && self.as_slice() == other.as_slice()
    }
}

fn first_broadcast_mismatch(left: &[usize], right: &[usize]) -> Option<(usize, usize, usize)> {
    let rank = left.len().max(right.len());
    for trailing_axis in 0..rank {
        let axis = rank - trailing_axis - 1;
        let left_dimension = aligned_dimension(left, rank, axis);
        let right_dimension = aligned_dimension(right, rank, axis);
        if broadcast_dimension(left_dimension, right_dimension).is_none() {
            return Some((left_dimension, right_dimension, axis));
        }
    }
    None
}

impl Tensor {
    /// Creates a tensor after validating that `shape` describes `data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the element count or contiguous stride overflows,
    /// or when the element count differs from the supplied data length.
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::from_vec_with_metadata(data, shape, DType::Float32, Device::Cpu)
    }

    pub(crate) fn from_vec_with_metadata(
        data: Vec<f32>,
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (expected, strides) = validated_layout(&shape)?;
        if data.len() != expected {
            return Err(TensorError::ShapeDataMismatch {
                shape,
                elements: data.len(),
            });
        }
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a zero-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::zeros_with_metadata(shape, DType::Float32, Device::Cpu)
    }

    pub(crate) fn zeros_with_metadata(
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        let data = filled_storage(elements, 0.0)?;
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a one-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::ones_with_metadata(shape, DType::Float32, Device::Cpu)
    }

    pub(crate) fn ones_with_metadata(
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        let data = filled_storage(elements, 1.0)?;
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a two-dimensional tensor with ones on the main diagonal and
    /// zeros elsewhere.
    ///
    /// Passing [`None`] for `m` creates a square `n` by `n` tensor. A column
    /// count may be passed directly or as [`Some`] to create a rectangular
    /// tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows, or when storage allocation fails.
    pub fn eye(n: usize, m: impl Into<Option<usize>>) -> Result<Self, TensorError> {
        Self::eye_with_metadata(n, m.into().unwrap_or(n), DType::Float32, Device::Cpu)
    }

    pub(crate) fn eye_with_metadata(
        n: usize,
        m: usize,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let mut shape = try_result_vector(2, 0)?;
        shape.push(n);
        shape.push(m);
        let (elements, strides) = validated_layout(&shape)?;
        let mut data = filled_storage(elements, 0.0)?;

        let diagonal = n.min(m);
        if diagonal > 0 {
            for (index, row) in data.chunks_exact_mut(m).take(diagonal).enumerate() {
                row[index] = 1.0;
            }
        }

        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    /// Creates a tensor filled with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows, or when storage allocation fails.
    pub fn full(shape: impl Into<Vec<usize>>, fill_value: f32) -> Result<Self, TensorError> {
        Self::full_with_metadata(shape, fill_value, DType::Float32, Device::Cpu)
    }

    pub(crate) fn full_with_metadata(
        shape: impl Into<Vec<usize>>,
        fill_value: f32,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        validate_storage_capacity(elements)?;
        let data = filled_storage(elements, fill_value)?;
        Ok(Self::from_owned_parts(data, shape, strides, dtype, device))
    }

    pub(crate) fn validate_full_shape(shape: &[usize]) -> Result<usize, TensorError> {
        let (elements, _) = validated_layout(shape)?;
        validate_storage_capacity(elements)?;
        Ok(elements)
    }

    fn from_owned_parts(
        data: Vec<f32>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let elements = data.len();
        debug_assert_eq!(validate_layout(&shape, &strides, 0, elements), Ok(elements));
        Self {
            storage: Arc::new(Storage {
                data,
                dtype,
                device,
            }),
            shape,
            strides,
            offset: 0,
            elements,
            logical_cache: OnceLock::new(),
        }
    }

    fn shared_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, TensorError> {
        let elements = validate_layout(&shape, &strides, offset, self.storage.data.len())?;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset,
            elements,
            logical_cache: OnceLock::new(),
        })
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor's row-major element strides.
    #[must_use]
    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the first element's offset into the shared storage.
    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.offset
    }

    /// Reports whether two tensors refer to the same underlying allocation.
    #[must_use]
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Returns the scalar type physically represented by this tensor's storage.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.storage.dtype
    }

    /// Returns the device owning this tensor's storage.
    #[must_use]
    pub fn device(&self) -> Device {
        self.storage.device
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.elements
    }

    /// Reports whether logical row-major iteration visits one dense storage span.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        if self.elements == 0 {
            return true;
        }

        let mut expected = 1_usize;
        for (&dimension, &stride) in self.shape.iter().zip(&self.strides).rev() {
            if dimension == 1 {
                continue;
            }
            if stride != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(dimension) else {
                return false;
            };
            expected = next;
        }
        true
    }

    fn dense_slice(&self) -> Option<&[f32]> {
        if !self.is_contiguous() {
            return None;
        }
        if self.elements == 0 {
            return Some(&self.storage.data[0..0]);
        }
        let end = self.offset.checked_add(self.elements)?;
        self.storage.data.get(self.offset..end)
    }

    /// Returns the tensor's values in logical row-major order.
    ///
    /// Dense tensors borrow their storage directly. A non-contiguous view is
    /// materialized lazily for this borrowed-slice compatibility API; tensor
    /// operations themselves traverse the shared strided storage directly.
    ///
    /// # Panics
    ///
    /// Panics if materializing a non-contiguous view cannot allocate memory, or
    /// if the tensor's private, validated layout invariant has been violated.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        if let Some(data) = self.dense_slice() {
            return data;
        }
        self.logical_cache
            .get_or_init(|| self.values().collect())
            .as_slice()
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        if self.elements == 0 {
            return Vec::new();
        }
        if self.is_contiguous() {
            let Self {
                storage,
                offset,
                elements,
                ..
            } = self;
            return match Arc::try_unwrap(storage) {
                Ok(storage) => {
                    if offset == 0 && elements == storage.data.len() {
                        storage.data
                    } else {
                        storage.data[offset..offset + elements].to_vec()
                    }
                }
                Err(storage) => storage.data[offset..offset + elements].to_vec(),
            };
        }
        self.values().collect()
    }

    fn values(&self) -> TensorValues<'_> {
        TensorValues::new(self)
    }

    /// Selects the leading dimension with one integer, returning a shared-storage view.
    ///
    /// Negative indices wrap from the end of the dimension. This entry point
    /// preserves the distinct `PyTorch` diagnostic for indexing a scalar with
    /// the integer zero; [`Self::index`] models tuple indexing instead.
    ///
    /// # Errors
    ///
    /// Returns an error when this tensor is scalar, the index is out of
    /// bounds, offset arithmetic overflows, or view metadata allocation fails.
    pub fn index_integer(&self, index: i64) -> Result<Self, TensorError> {
        if self.shape.is_empty() {
            return if index == 0 {
                Err(TensorError::InvalidScalarIndex)
            } else {
                Err(TensorError::IndexOutOfBounds {
                    index,
                    dimension: 0,
                    size: 0,
                })
            };
        }
        self.index_dimensions(&[index])
    }

    /// Selects consecutive leading dimensions with integer indices.
    ///
    /// The returned tensor is a metadata-only view that retains the remaining
    /// shape and strides and shares storage, dtype, and device with `self`.
    /// Passing no indices returns an alias with identical metadata.
    ///
    /// # Errors
    ///
    /// Returns an error for too many indices, an out-of-bounds index, checked
    /// arithmetic overflow, or view metadata allocation failure.
    pub fn index(&self, indices: impl AsRef<[i64]>) -> Result<Self, TensorError> {
        self.index_dimensions(indices.as_ref())
    }

    /// Applies integer and positive-step basic indices to leading dimensions.
    ///
    /// Missing trailing dimensions are retained. The result contains only new
    /// metadata and shares storage, dtype, and device with this tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for too many indices, invalid integers, zero or
    /// negative slice steps, metadata overflow, allocation failure, or a view
    /// whose non-empty storage span would escape the underlying allocation.
    pub fn slice(&self, indices: impl AsRef<[TensorIndex]>) -> Result<Self, TensorError> {
        let indices = indices.as_ref();
        if indices.len() > self.shape.len() {
            return Err(TensorError::TooManyIndices {
                dimensions: self.shape.len(),
            });
        }

        let result_rank = self
            .shape
            .len()
            .checked_sub(
                indices
                    .iter()
                    .filter(|index| matches!(index, TensorIndex::Integer(_)))
                    .count(),
            )
            .ok_or(TensorError::IndexCalculationOverflow)?;
        let mut shape = try_result_vector(result_rank, self.elements)?;
        let mut strides = try_result_vector(result_rank, self.elements)?;
        let mut offset = self.offset;

        for (dimension, index) in indices.iter().copied().enumerate() {
            match index {
                TensorIndex::Integer(index) => {
                    offset = self.checked_index_offset(offset, dimension, index)?;
                }
                TensorIndex::Slice(slice) => {
                    let checked = self.checked_slice_layout(offset, dimension, slice)?;
                    offset = checked.offset;
                    shape.push(checked.length);
                    strides.push(checked.stride);
                }
            }
        }

        shape.extend_from_slice(&self.shape[indices.len()..]);
        strides.extend_from_slice(&self.strides[indices.len()..]);
        self.shared_view(shape, strides, offset)
    }

    fn index_dimensions(&self, indices: &[i64]) -> Result<Self, TensorError> {
        if indices.len() > self.shape.len() {
            return Err(TensorError::TooManyIndices {
                dimensions: self.shape.len(),
            });
        }

        let mut offset = self.offset;
        for (dimension, index) in indices.iter().copied().enumerate() {
            offset = self.checked_index_offset(offset, dimension, index)?;
        }

        let shape = try_clone_result_shape(&self.shape[indices.len()..], self.elements)?;
        let strides = try_clone_result_shape(&self.strides[indices.len()..], self.elements)?;
        self.shared_view(shape, strides, offset)
    }

    pub(crate) fn checked_index_offset(
        &self,
        offset: usize,
        dimension: usize,
        index: i64,
    ) -> Result<usize, TensorError> {
        let size = self
            .shape
            .get(dimension)
            .copied()
            .ok_or(TensorError::TooManyIndices {
                dimensions: self.shape.len(),
            })?;
        let signed_size = i64::try_from(size).map_err(|_| TensorError::IndexCalculationOverflow)?;
        if index < -signed_size || index >= signed_size {
            return Err(TensorError::IndexOutOfBounds {
                index,
                dimension,
                size,
            });
        }
        let normalized = if index < 0 {
            signed_size
                .checked_add(index)
                .ok_or(TensorError::IndexCalculationOverflow)?
        } else {
            index
        };
        let normalized =
            usize::try_from(normalized).map_err(|_| TensorError::IndexCalculationOverflow)?;
        let contribution = normalized
            .checked_mul(self.strides[dimension])
            .ok_or(TensorError::IndexCalculationOverflow)?;
        checked_storage_offset_add(offset, contribution)
    }

    pub(crate) fn checked_slice_layout(
        &self,
        offset: usize,
        dimension: usize,
        slice: Slice,
    ) -> Result<CheckedSlice, TensorError> {
        let size = self
            .shape
            .get(dimension)
            .copied()
            .ok_or(TensorError::TooManyIndices {
                dimensions: self.shape.len(),
            })?;
        let source_stride = self
            .strides
            .get(dimension)
            .copied()
            .ok_or(TensorError::StrideCalculationOverflow)?;
        let normalized = slice.normalize(size)?;
        let contribution = normalized
            .start
            .checked_mul(source_stride)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        let offset = checked_storage_offset_add(offset, contribution)?;
        let stride = checked_stride_product(source_stride, normalized.step)?;
        Ok(CheckedSlice {
            offset,
            length: normalized.length,
            stride,
        })
    }

    /// Returns a tensor with a new shape and the same logical values.
    ///
    /// One dimension may be `-1`, in which case it is inferred from the
    /// tensor's element count. View-compatible layouts share immutable storage;
    /// other non-contiguous layouts are copied into a contiguous allocation,
    /// matching `PyTorch` reshape semantics.
    ///
    /// # Errors
    ///
    /// Returns an error for negative dimensions other than `-1`, multiple
    /// inferred dimensions, incompatible element counts, ambiguous inference
    /// for an empty tensor, arithmetic overflow, or metadata allocation
    /// failure.
    pub fn reshape(&self, shape: impl AsRef<[i64]>) -> Result<Self, TensorError> {
        let requested = shape.as_ref();
        let mut inferred_index = None;

        for (index, dimension) in requested.iter().copied().enumerate() {
            if dimension == -1 {
                if inferred_index.replace(index).is_some() {
                    return Err(TensorError::ReshapeMultipleInferredDimensions);
                }
                continue;
            }
            if dimension < 0 {
                return Err(TensorError::ReshapeInvalidDimension {
                    dimension,
                    index,
                    shape: try_clone_reshape_shape(requested, self.elements)?,
                });
            }
        }

        let mut resolved = try_result_vector(requested.len(), self.elements)?;
        for dimension in requested.iter().copied() {
            let dimension = if dimension == -1 {
                1
            } else {
                usize::try_from(dimension).map_err(|_| TensorError::ElementCountOverflow)?
            };
            resolved.push(dimension);
        }

        if let Some(index) = inferred_index {
            let specified_elements = requested
                .iter()
                .copied()
                .filter(|dimension| *dimension != -1)
                .fold(1_i64, i64::wrapping_mul);
            let elements =
                i64::try_from(self.elements).map_err(|_| TensorError::ElementCountOverflow)?;
            if !((specified_elements > 0 && elements % specified_elements == 0)
                || elements == specified_elements)
            {
                return Err(TensorError::ReshapeElementCountMismatch {
                    shape: try_clone_reshape_shape(requested, self.elements)?,
                    elements: self.elements,
                });
            }
            if specified_elements == 0 {
                if self.elements == 0 {
                    return Err(TensorError::ReshapeAmbiguousZeroElements {
                        shape: try_clone_reshape_shape(requested, self.elements)?,
                    });
                }
                return Err(TensorError::ReshapeElementCountMismatch {
                    shape: try_clone_reshape_shape(requested, self.elements)?,
                    elements: self.elements,
                });
            }
            resolved[index] = usize::try_from(elements / specified_elements)
                .map_err(|_| TensorError::ElementCountOverflow)?;
        }

        let resolved_elements = element_count(&resolved)?;
        if resolved_elements != self.elements {
            return Err(TensorError::ReshapeElementCountMismatch {
                shape: try_clone_reshape_shape(requested, self.elements)?,
                elements: self.elements,
            });
        }

        if self.elements == 0 {
            let strides = if resolved == self.shape {
                try_clone_result_shape(&self.strides, self.elements)?
            } else {
                reshape_strides(&resolved, self.elements)?
            };
            return self.shared_view(resolved, strides, self.offset);
        }

        if let Some(strides) = compute_reshape_view_strides(self, &resolved)? {
            return self.shared_view(resolved, strides, self.offset);
        }

        let strides = reshape_strides(&resolved, self.elements)?;
        let data = self.try_logical_vec()?;
        Ok(Self::from_owned_parts(
            data,
            resolved,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    /// Adds tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left + right)
    }

    /// Subtracts tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left - right)
    }

    /// Multiplies tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left * right)
    }

    /// Divides tensors element by element using IEEE 754 true division and
    /// trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left / right)
    }

    /// Adds a scalar to every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn add_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| value + scalar)
    }

    /// Subtracts a scalar from every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn sub_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| value - scalar)
    }

    /// Multiplies every element by a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| value * scalar)
    }

    /// Divides every element by a scalar using IEEE 754 true division.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn div_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| value / scalar)
    }

    /// Subtracts every element from a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_sub(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| scalar - value)
    }

    /// Divides a scalar by every element using `PyTorch`'s float32 reciprocal
    /// multiplication semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_div(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| scalar * value.recip())
    }

    /// Applies rectified linear activation element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn relu(&self) -> Result<Self, TensorError> {
        self.unary_map(|value| value.max(0.0))
    }

    /// Computes the sine of every element in radians.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn sin(&self) -> Result<Self, TensorError> {
        self.unary_map(f32::sin)
    }

    #[must_use]
    pub fn sum(&self) -> Self {
        let value = self
            .dense_slice()
            .map_or_else(|| self.values().sum(), |data| data.iter().sum());
        Self::from_owned_parts(
            vec![value],
            Vec::new(),
            Vec::new(),
            self.dtype(),
            self.device(),
        )
    }

    /// Extracts the value of a one-element tensor.
    ///
    /// # Errors
    ///
    /// Returns an error unless the tensor contains exactly one element.
    pub fn item(&self) -> Result<f32, TensorError> {
        if self.elements != 1 {
            return Err(TensorError::ItemRequiresOneElement {
                elements: self.elements,
            });
        }
        Ok(self.storage.data[self.offset])
    }

    /// Multiplies two rank-2 matrices.
    ///
    /// # Errors
    ///
    /// Returns an error unless both tensors are matrices with compatible inner
    /// dimensions.
    pub fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::MatmulRequiresMatrices {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        let (rows, inner) = (self.shape[0], self.shape[1]);
        let (other_inner, columns) = (other.shape[0], other.shape[1]);
        if inner != other_inner {
            return Err(TensorError::MatmulInnerDimensionMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }

        let mut output_shape = try_result_vector(2, 0)?;
        output_shape.push(rows);
        output_shape.push(columns);
        let (output_elements, output_strides) = validated_layout(&output_shape)?;
        let mut output = filled_storage(output_elements, 0.0)?;
        for row in 0..rows {
            for depth in 0..inner {
                let left_offset = self.offset + row * self.strides[0] + depth * self.strides[1];
                let left = self.storage.data[left_offset];
                for column in 0..columns {
                    let right_offset =
                        other.offset + depth * other.strides[0] + column * other.strides[1];
                    output[row * columns + column] += left * other.storage.data[right_offset];
                }
            }
        }
        Ok(Self::from_owned_parts(
            output,
            output_shape,
            output_strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn zip_map(
        &self,
        other: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        if self.shape == other.shape {
            return self.zip_map_same_shape(other, operation);
        }

        let plan = BroadcastPlan::new(self, other)?;
        let mut data = try_result_vector(plan.elements, plan.elements)?;
        if plan.elements == 0 {
            return Ok(Self::from_owned_parts(
                data,
                plan.shape,
                plan.strides,
                self.dtype(),
                self.device(),
            ));
        }

        let mut coordinates = try_result_vector(plan.shape.len(), plan.elements)?;
        coordinates.resize(plan.shape.len(), 0_usize);
        let mut left_offset = 0_usize;
        let mut right_offset = 0_usize;
        for output_offset in 0..plan.elements {
            data.push(operation(
                self.storage.data[self.offset + left_offset],
                other.storage.data[other.offset + right_offset],
            ));
            if output_offset + 1 == plan.elements {
                break;
            }

            for axis in (0..plan.shape.len()).rev() {
                coordinates[axis] = coordinates[axis]
                    .checked_add(1)
                    .ok_or(TensorError::StrideCalculationOverflow)?;
                if coordinates[axis] < plan.shape[axis] {
                    left_offset = left_offset
                        .checked_add(plan.dimensions[axis].left_step)
                        .ok_or(TensorError::StrideCalculationOverflow)?;
                    right_offset = right_offset
                        .checked_add(plan.dimensions[axis].right_step)
                        .ok_or(TensorError::StrideCalculationOverflow)?;
                    break;
                }

                coordinates[axis] = 0;
                left_offset = left_offset
                    .checked_sub(plan.dimensions[axis].left_rewind)
                    .ok_or(TensorError::StrideCalculationOverflow)?;
                right_offset = right_offset
                    .checked_sub(plan.dimensions[axis].right_rewind)
                    .ok_or(TensorError::StrideCalculationOverflow)?;
            }
        }

        Ok(Self::from_owned_parts(
            data,
            plan.shape,
            plan.strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn zip_map_same_shape(
        &self,
        other: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        let mut data = try_result_vector(elements, elements)?;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = contiguous_strides(&shape, elements)?;
        if let (Some(left), Some(right)) = (self.dense_slice(), other.dense_slice()) {
            data.extend(
                left.iter()
                    .copied()
                    .zip(right.iter().copied())
                    .map(|(left, right)| operation(left, right)),
            );
        } else {
            data.extend(
                self.values()
                    .zip(other.values())
                    .map(|(left, right)| operation(left, right)),
            );
        }
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn map_scalar(
        &self,
        scalar: f32,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        let mut data = try_result_vector(elements, elements)?;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = if elements == 0 {
            elementwise_output_strides(&shape, &[self], elements)?
        } else {
            contiguous_strides(&shape, elements)?
        };
        if let Some(values) = self.dense_slice() {
            data.extend(values.iter().copied().map(|value| operation(value, scalar)));
        } else {
            data.extend(self.values().map(|value| operation(value, scalar)));
        }
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn unary_map(&self, operation: impl Fn(f32) -> f32) -> Result<Self, TensorError> {
        let elements = self.elements;
        let mut data = try_result_vector(elements, elements)?;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = contiguous_strides(&shape, elements)?;
        if let Some(values) = self.dense_slice() {
            data.extend(values.iter().copied().map(operation));
        } else {
            data.extend(self.values().map(operation));
        }
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    fn try_logical_vec(&self) -> Result<Vec<f32>, TensorError> {
        let mut data = try_result_vector(self.elements, self.elements)?;
        if let Some(dense) = self.dense_slice() {
            data.extend_from_slice(dense);
        } else {
            data.extend(self.values());
        }
        Ok(data)
    }
}

struct TensorValues<'a> {
    tensor: &'a Tensor,
    coordinates: Vec<usize>,
    storage_index: usize,
    remaining: usize,
}

impl<'a> TensorValues<'a> {
    fn new(tensor: &'a Tensor) -> Self {
        Self {
            tensor,
            coordinates: vec![0; tensor.shape.len()],
            storage_index: tensor.offset,
            remaining: tensor.elements,
        }
    }
}

impl Iterator for TensorValues<'_> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let value = self.tensor.storage.data[self.storage_index];
        self.remaining -= 1;
        if self.remaining == 0 {
            return Some(value);
        }

        for axis in (0..self.tensor.shape.len()).rev() {
            self.coordinates[axis] += 1;
            if self.coordinates[axis] < self.tensor.shape[axis] {
                self.storage_index += self.tensor.strides[axis];
                break;
            }
            self.storage_index -=
                self.coordinates[axis].saturating_sub(1) * self.tensor.strides[axis];
            self.coordinates[axis] = 0;
        }
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for TensorValues<'_> {}

fn checked_storage_offset_add(offset: usize, contribution: usize) -> Result<usize, TensorError> {
    let offset = offset
        .checked_add(contribution)
        .ok_or(TensorError::IndexCalculationOverflow)?;
    if i64::try_from(offset).is_err() {
        let offset =
            i64::try_from(offset.cast_signed()).expect("an isize storage offset must fit in i64");
        return Err(TensorError::InvalidStorageOffset { offset });
    }
    Ok(offset)
}

fn validate_layout(
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    storage_elements: usize,
) -> Result<usize, TensorError> {
    if shape.len() != strides.len() {
        return Err(TensorError::StrideCalculationOverflow);
    }
    let elements = element_count(shape)?;
    if i64::try_from(offset).is_err() {
        let offset =
            i64::try_from(offset.cast_signed()).expect("an isize storage offset must fit in i64");
        return Err(TensorError::InvalidStorageOffset { offset });
    }
    if elements == 0 {
        return Ok(elements);
    }

    let max_displacement =
        shape
            .iter()
            .zip(strides)
            .try_fold(0_usize, |displacement, (&dimension, &stride)| {
                let axis_displacement = dimension
                    .saturating_sub(1)
                    .checked_mul(stride)
                    .ok_or(TensorError::IndexCalculationOverflow)?;
                displacement
                    .checked_add(axis_displacement)
                    .ok_or(TensorError::IndexCalculationOverflow)
            })?;
    let span = max_displacement
        .checked_add(1)
        .ok_or(TensorError::IndexCalculationOverflow)?;
    let end = offset
        .checked_add(span)
        .ok_or(TensorError::IndexCalculationOverflow)?;
    if end > storage_elements {
        return Err(TensorError::LayoutOutOfStorage {
            offset,
            span,
            storage_elements,
        });
    }
    Ok(elements)
}

fn compute_reshape_view_strides(
    tensor: &Tensor,
    new_shape: &[usize],
) -> Result<Option<Vec<usize>>, TensorError> {
    if tensor.is_contiguous() {
        return contiguous_strides(new_shape, tensor.elements).map(Some);
    }
    if tensor.shape.is_empty() {
        return contiguous_strides(new_shape, tensor.elements).map(Some);
    }

    let mut new_strides = try_result_vector(new_shape.len(), tensor.elements)?;
    new_strides.resize(new_shape.len(), 0);
    let mut view_axis = new_shape.len();
    let mut chunk_elements = 1_usize;
    let mut view_elements = 1_usize;
    let mut chunk_base_stride = *tensor
        .strides
        .last()
        .expect("a non-scalar tensor has a final stride");

    for tensor_axis in (0..tensor.shape.len()).rev() {
        chunk_elements = chunk_elements
            .checked_mul(tensor.shape[tensor_axis])
            .ok_or(TensorError::StrideCalculationOverflow)?;
        let chunk_boundary = tensor_axis == 0 || {
            let previous_dimension = tensor.shape[tensor_axis - 1];
            let expected_previous_stride = chunk_elements
                .checked_mul(chunk_base_stride)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            previous_dimension != 1 && tensor.strides[tensor_axis - 1] != expected_previous_stride
        };

        if chunk_boundary {
            while view_axis > 0 && (view_elements < chunk_elements || new_shape[view_axis - 1] == 1)
            {
                view_axis -= 1;
                new_strides[view_axis] = view_elements
                    .checked_mul(chunk_base_stride)
                    .ok_or(TensorError::StrideCalculationOverflow)?;
                view_elements = view_elements
                    .checked_mul(new_shape[view_axis])
                    .ok_or(TensorError::StrideCalculationOverflow)?;
            }
            if view_elements != chunk_elements {
                return Ok(None);
            }
            if tensor_axis > 0 {
                chunk_base_stride = tensor.strides[tensor_axis - 1];
                chunk_elements = 1;
                view_elements = 1;
            }
        }
    }

    if view_axis == 0 {
        Ok(Some(new_strides))
    } else {
        Ok(None)
    }
}

#[derive(Clone, Copy)]
struct BroadcastDimension {
    left_step: usize,
    right_step: usize,
    left_rewind: usize,
    right_rewind: usize,
}

struct BroadcastPlan {
    shape: Vec<usize>,
    strides: Vec<usize>,
    dimensions: Vec<BroadcastDimension>,
    elements: usize,
}

impl BroadcastPlan {
    fn new(left: &Tensor, right: &Tensor) -> Result<Self, TensorError> {
        let rank = left.shape.len().max(right.shape.len());
        for axis in 0..rank {
            let left_dimension = aligned_dimension(&left.shape, rank, axis);
            let right_dimension = aligned_dimension(&right.shape, rank, axis);
            if broadcast_dimension(left_dimension, right_dimension).is_none() {
                return Err(TensorError::ShapeMismatch {
                    left: try_clone_result_shape(&left.shape, left.elements)?,
                    right: try_clone_result_shape(&right.shape, right.elements)?,
                });
            }
        }

        let mut elements = 1_usize;
        for axis in 0..rank {
            let dimension = broadcast_dimension(
                aligned_dimension(&left.shape, rank, axis),
                aligned_dimension(&right.shape, rank, axis),
            )
            .expect("broadcast compatibility was checked above");
            elements = elements
                .checked_mul(dimension)
                .ok_or(TensorError::ElementCountOverflow)?;
        }
        validate_storage_capacity(elements)?;

        let mut shape = try_result_vector(rank, elements)?;
        for axis in 0..rank {
            shape.push(
                broadcast_dimension(
                    aligned_dimension(&left.shape, rank, axis),
                    aligned_dimension(&right.shape, rank, axis),
                )
                .expect("broadcast compatibility was checked above"),
            );
        }
        let strides = if elements == 0 {
            elementwise_output_strides(&shape, &[left, right], elements)?
        } else {
            contiguous_strides(&shape, elements)?
        };

        let mut dimensions = try_result_vector(rank, elements)?;
        if elements == 0 {
            dimensions.resize(
                rank,
                BroadcastDimension {
                    left_step: 0,
                    right_step: 0,
                    left_rewind: 0,
                    right_rewind: 0,
                },
            );
            return Ok(Self {
                shape,
                strides,
                dimensions,
                elements,
            });
        }

        for axis in (0..rank).rev() {
            let output_dimension = shape[axis];
            let left_step = aligned_broadcast_stride(left, rank, axis, output_dimension);
            let right_step = aligned_broadcast_stride(right, rank, axis, output_dimension);
            let repeats = output_dimension.saturating_sub(1);
            let left_rewind = left_step
                .checked_mul(repeats)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            let right_rewind = right_step
                .checked_mul(repeats)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            dimensions.push(BroadcastDimension {
                left_step,
                right_step,
                left_rewind,
                right_rewind,
            });
        }
        dimensions.reverse();

        Ok(Self {
            shape,
            strides,
            dimensions,
            elements,
        })
    }
}

fn aligned_dimension(shape: &[usize], output_rank: usize, output_axis: usize) -> usize {
    let leading_dimensions = output_rank - shape.len();
    if output_axis < leading_dimensions {
        1
    } else {
        shape[output_axis - leading_dimensions]
    }
}

fn broadcast_dimension(left: usize, right: usize) -> Option<usize> {
    if left == right {
        Some(left)
    } else if left == 1 {
        Some(right)
    } else if right == 1 {
        Some(left)
    } else {
        None
    }
}

fn aligned_broadcast_stride(
    tensor: &Tensor,
    output_rank: usize,
    output_axis: usize,
    output_dimension: usize,
) -> usize {
    let leading_dimensions = output_rank - tensor.shape.len();
    if output_axis < leading_dimensions {
        return 0;
    }

    let input_axis = output_axis - leading_dimensions;
    let input_dimension = tensor.shape[input_axis];
    if input_dimension == 1 && output_dimension != 1 {
        0
    } else {
        tensor.strides[input_axis]
    }
}

fn aligned_broadcast_stride_bytes(
    tensor: &Tensor,
    output_rank: usize,
    output_axis: usize,
    output_dimension: usize,
) -> i64 {
    // TensorIterator compares byte strides stored in signed 64-bit integers.
    // Preserve its wrapping conversion at this boundary: an extreme but valid
    // empty view can therefore change the recovered output permutation without
    // accessing any storage.
    let stride =
        aligned_broadcast_stride(tensor, output_rank, output_axis, output_dimension).cast_signed();
    let element_size =
        i64::try_from(size_of::<f32>()).expect("an f32 element size must fit in i64");
    i64::try_from(stride)
        .expect("an isize stride must fit in a signed 64-bit TensorIterator stride")
        .wrapping_mul(element_size)
}

fn try_result_vector<T>(capacity: usize, elements: usize) -> Result<Vec<T>, TensorError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| TensorError::AllocationFailed { elements })?;
    Ok(values)
}

fn try_clone_result_shape(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    let mut cloned = try_result_vector(shape.len(), elements)?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

fn try_clone_reshape_shape(shape: &[i64], elements: usize) -> Result<Vec<i64>, TensorError> {
    let mut cloned = try_result_vector(shape.len(), elements)?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(TensorError::ElementCountOverflow)
    })
}

fn validated_layout(shape: &[usize]) -> Result<(usize, Vec<usize>), TensorError> {
    let elements = element_count(shape)?;
    let strides = contiguous_strides(shape, elements)?;
    Ok((elements, strides))
}

fn contiguous_strides(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    let mut strides = try_result_vector(shape.len(), elements)?;
    strides.resize(shape.len(), 0);
    let mut stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        if axis > 0 {
            stride = checked_stride_product(stride, shape[axis])?;
        }
    }
    Ok(strides)
}

fn elementwise_output_strides(
    shape: &[usize],
    operands: &[&Tensor],
    elements: usize,
) -> Result<Vec<usize>, TensorError> {
    let rank = shape.len();
    let mut permutation = try_result_vector(rank, elements)?;
    permutation.extend((0..rank).rev());

    for index in 1..rank {
        let mut dimension_1 = index;
        for dimension_0 in (0..index).rev() {
            let comparison = compare_elementwise_dimensions(
                shape,
                operands,
                permutation[dimension_0],
                permutation[dimension_1],
            );
            if comparison > 0 {
                permutation.swap(dimension_0, dimension_1);
                dimension_1 = dimension_0;
            } else if comparison < 0 {
                break;
            }
        }
    }

    if permutation
        .iter()
        .enumerate()
        .all(|(index, axis)| *axis == rank - index - 1)
    {
        return contiguous_strides(shape, elements);
    }

    let element_size =
        i64::try_from(size_of::<f32>()).expect("an f32 element size must fit in i64");
    let mut byte_strides = try_result_vector(rank, elements)?;
    byte_strides.resize(rank, 0_i64);
    let mut next_byte_stride = element_size;
    for (position, axis) in permutation.into_iter().enumerate() {
        byte_strides[axis] = next_byte_stride;
        if position + 1 < rank {
            let dimension =
                i64::try_from(shape[axis]).map_err(|_| TensorError::StrideCalculationOverflow)?;
            next_byte_stride = next_byte_stride.wrapping_mul(dimension);
        }
    }

    let mut strides = try_result_vector(rank, elements)?;
    strides.resize(rank, 0);
    for axis in (0..rank).rev() {
        let stride = byte_strides[axis] / element_size;
        if stride >= 0 {
            strides[axis] =
                usize::try_from(stride).map_err(|_| TensorError::StrideCalculationOverflow)?;
        } else if axis + 1 == rank {
            strides[axis] = 1;
        } else {
            strides[axis] = checked_stride_product(strides[axis + 1], shape[axis + 1])?;
        }
    }
    Ok(strides)
}

fn compare_elementwise_dimensions(
    shape: &[usize],
    operands: &[&Tensor],
    dimension_0: usize,
    dimension_1: usize,
) -> i8 {
    for tensor in operands {
        let stride_0 =
            aligned_broadcast_stride_bytes(tensor, shape.len(), dimension_0, shape[dimension_0]);
        let stride_1 =
            aligned_broadcast_stride_bytes(tensor, shape.len(), dimension_1, shape[dimension_1]);
        if stride_0 == 0 || stride_1 == 0 {
            continue;
        }
        if stride_0 < stride_1 {
            return -1;
        }
        if stride_0 > stride_1 || shape[dimension_0] > shape[dimension_1] {
            return 1;
        }
    }
    0
}

fn reshape_strides(shape: &[usize], elements: usize) -> Result<Vec<usize>, TensorError> {
    if elements != 0 {
        return contiguous_strides(shape, elements);
    }

    let mut strides = try_result_vector(shape.len(), elements)?;
    strides.resize(shape.len(), 0);
    let mut stride = 1_usize;
    for axis in (0..shape.len()).rev() {
        strides[axis] = stride;
        if axis > 0 {
            // PyTorch treats empty-view strides as arbitrary metadata and its
            // resize-style calculation permits non-negative signed wrapping.
            stride = signed_wrapping_stride_product(stride, shape[axis].max(1))?;
        }
    }
    Ok(strides)
}

fn signed_wrapping_stride_product(stride: usize, dimension: usize) -> Result<usize, TensorError> {
    if stride == 0 || dimension == 0 {
        return Ok(0);
    }
    let stride = i64::try_from(stride).map_err(|_| TensorError::StrideCalculationOverflow)?;
    let dimension = i64::try_from(dimension).map_err(|_| TensorError::StrideCalculationOverflow)?;
    let product = stride.wrapping_mul(dimension);
    usize::try_from(product).map_err(|_| TensorError::StrideCalculationOverflow)
}

fn checked_stride_product(stride: usize, dimension: usize) -> Result<usize, TensorError> {
    stride
        .checked_mul(dimension.max(1))
        .filter(|product| *product <= isize::MAX.unsigned_abs())
        .ok_or(TensorError::StrideCalculationOverflow)
}

fn filled_storage(elements: usize, fill_value: f32) -> Result<Vec<f32>, TensorError> {
    validate_storage_capacity(elements)?;

    let mut data = Vec::new();
    data.try_reserve_exact(elements)
        .map_err(|_| TensorError::AllocationFailed { elements })?;
    data.resize(elements, fill_value);
    Ok(data)
}

fn validate_storage_capacity(elements: usize) -> Result<(), TensorError> {
    let maximum_elements = isize::MAX.unsigned_abs() / size_of::<f32>();
    if elements > maximum_elements {
        return Err(TensorError::StorageCapacityOverflow { elements });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{TensorError, try_result_vector};

    #[test]
    fn binary_result_reservation_failures_return_tensor_errors() {
        let elements = 17;
        let expected = Err(TensorError::AllocationFailed { elements });

        assert_eq!(try_result_vector::<f32>(usize::MAX, elements), expected);
        assert_eq!(
            try_result_vector::<usize>(usize::MAX, elements),
            Err(TensorError::AllocationFailed { elements })
        );
    }
}
