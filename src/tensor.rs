use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::sync::Arc;

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

/// Storage layouts accepted by tensor-copy operations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Retain the source tensor's supported layout metadata.
    #[default]
    Preserve,
    /// Produce canonical contiguous row-major strides.
    Contiguous,
    /// Four-dimensional channels-last layout, currently unsupported.
    ChannelsLast,
    /// Five-dimensional channels-last layout, currently unsupported.
    ChannelsLast3d,
}

impl Display for MemoryFormat {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preserve => formatter.write_str("preserve_format"),
            Self::Contiguous => formatter.write_str("contiguous_format"),
            Self::ChannelsLast => formatter.write_str("channels_last"),
            Self::ChannelsLast3d => formatter.write_str("channels_last_3d"),
        }
    }
}

struct Storage {
    data: Vec<f32>,
    dtype: DType,
    device: Device,
}

/// A contiguous, row-major tensor with native storage metadata.
///
/// This deliberately narrow representation is the campaign's baseline, not a
/// claim of `PyTorch` feature parity. Later iterations may generalize storage as
/// long as these observable semantics remain compatible.
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    elements: usize,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("cloning validated tensor storage should succeed")
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
    DimensionOutOfRange {
        dimension: i64,
        minimum: i64,
        maximum: i64,
    },
    NegativeStrides {
        strides: Vec<i64>,
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
    UnsupportedMemoryFormat {
        memory_format: MemoryFormat,
    },
}

impl Display for TensorError {
    // Keep the exhaustive compatibility diagnostics together with the error
    // variants they describe.
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
            Self::DimensionOutOfRange {
                dimension,
                minimum,
                maximum,
            } => write!(
                formatter,
                "Dimension out of range (expected to be in range of [{minimum}, {maximum}], but got {dimension})"
            ),
            Self::NegativeStrides { strides } => write!(
                formatter,
                "as_strided: Negative strides are not supported at the moment, got strides: {strides:?}"
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
            Self::UnsupportedMemoryFormat { memory_format } => write!(
                formatter,
                "clone with memory format torch.{memory_format} is not supported"
            ),
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
        }
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

    /// Creates an independent copy of this tensor's logical values.
    ///
    /// The returned tensor preserves this supported contiguous view's strides
    /// and has a storage offset of zero. Only the logical range of a view is
    /// copied; unused values in the view's backing allocation are not retained.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn try_clone(&self) -> Result<Self, TensorError> {
        self.try_clone_with_memory_format(MemoryFormat::Preserve)
    }

    /// Creates an independent copy using the requested storage layout.
    ///
    /// [`MemoryFormat::Preserve`] retains this supported contiguous view's
    /// strides. [`MemoryFormat::Contiguous`] recalculates canonical row-major
    /// strides. Channel-last formats are not implemented by the current tensor
    /// representation.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails,
    /// contiguous stride calculation overflows, or the requested format is not
    /// supported.
    pub fn try_clone_with_memory_format(
        &self,
        memory_format: MemoryFormat,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = match memory_format {
            MemoryFormat::Preserve => try_clone_result_shape(&self.strides, elements)?,
            MemoryFormat::Contiguous => contiguous_strides(&shape, elements)?,
            MemoryFormat::ChannelsLast | MemoryFormat::ChannelsLast3d => {
                return Err(TensorError::UnsupportedMemoryFormat { memory_format });
            }
        };
        let data = copied_storage(self.as_slice(), elements)?;
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
    }

    #[must_use]
    /// # Panics
    ///
    /// Panics only if the tensor's private, validated layout invariant has
    /// been violated.
    pub fn as_slice(&self) -> &[f32] {
        if self.elements == 0 {
            return &self.storage.data[0..0];
        }
        let end = self
            .offset
            .checked_add(self.elements)
            .expect("validated tensor view end must fit in usize");
        &self.storage.data[self.offset..end]
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        let Self {
            storage,
            offset,
            elements,
            ..
        } = self;
        if elements == 0 {
            return Vec::new();
        }
        match Arc::try_unwrap(storage) {
            Ok(storage) => {
                if offset == 0 && elements == storage.data.len() {
                    storage.data
                } else {
                    storage.data[offset..offset + elements].to_vec()
                }
            }
            Err(storage) => storage.data[offset..offset + elements].to_vec(),
        }
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
        let elements = element_count(&shape)?;
        offset
            .checked_add(elements)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset,
            elements,
        })
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
        let offset = offset
            .checked_add(contribution)
            .ok_or(TensorError::IndexCalculationOverflow)?;
        if i64::try_from(offset).is_err() {
            let offset = i64::try_from(offset.cast_signed())
                .expect("an isize storage offset must fit in i64");
            return Err(TensorError::InvalidStorageOffset { offset });
        }
        Ok(offset)
    }

    /// Inserts a size-one dimension and returns a shared-storage view.
    ///
    /// Negative dimensions are normalized over the output rank, so both `-1`
    /// and the current rank append a trailing dimension. The inserted stride
    /// follows `PyTorch`'s view geometry exactly: it is one when appending and
    /// otherwise the selected input size multiplied by its stride.
    ///
    /// # Errors
    ///
    /// Returns an error when `dimension` is outside `[-rank - 1, rank]`, when
    /// `PyTorch`'s signed stride calculation produces an unsupported negative
    /// stride, or when view metadata allocation fails.
    pub fn unsqueeze(&self, dimension: i64) -> Result<Self, TensorError> {
        let rank =
            i64::try_from(self.shape.len()).map_err(|_| TensorError::StrideCalculationOverflow)?;
        let output_rank = rank
            .checked_add(1)
            .ok_or(TensorError::StrideCalculationOverflow)?;
        let minimum = -output_rank;
        if dimension < minimum || dimension > rank {
            return Err(TensorError::DimensionOutOfRange {
                dimension,
                minimum,
                maximum: rank,
            });
        }
        let normalized = if dimension < 0 {
            dimension
                .checked_add(output_rank)
                .ok_or(TensorError::StrideCalculationOverflow)?
        } else {
            dimension
        };
        let axis =
            usize::try_from(normalized).map_err(|_| TensorError::StrideCalculationOverflow)?;

        let inserted_stride = if axis == self.shape.len() {
            1_i64
        } else {
            let size = i64::try_from(self.shape[axis])
                .map_err(|_| TensorError::StrideCalculationOverflow)?;
            let stride = i64::try_from(self.strides[axis])
                .map_err(|_| TensorError::StrideCalculationOverflow)?;
            size.wrapping_mul(stride)
        };
        if inserted_stride < 0 {
            return Err(TensorError::NegativeStrides {
                strides: try_insert_signed_stride(
                    &self.strides,
                    axis,
                    inserted_stride,
                    self.elements,
                )?,
            });
        }

        let output_rank = self
            .shape
            .len()
            .checked_add(1)
            .ok_or(TensorError::AllocationFailed {
                elements: self.elements,
            })?;
        let mut shape = try_result_vector(output_rank, self.elements)?;
        shape.extend_from_slice(&self.shape[..axis]);
        shape.push(1);
        shape.extend_from_slice(&self.shape[axis..]);

        let mut strides = try_result_vector(output_rank, self.elements)?;
        strides.extend_from_slice(&self.strides[..axis]);
        strides.push(
            usize::try_from(inserted_stride).map_err(|_| TensorError::StrideCalculationOverflow)?,
        );
        strides.extend_from_slice(&self.strides[axis..]);

        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape,
            strides,
            offset: self.offset,
            elements: self.elements,
        })
    }

    /// Returns a contiguous metadata-only view with a new shape.
    ///
    /// One dimension may be `-1`, in which case it is inferred from the
    /// tensor's element count. The returned tensor shares immutable storage
    /// with `self`.
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

        let strides = if self.elements == 0 && resolved == self.shape {
            try_clone_result_shape(&self.strides, self.elements)?
        } else {
            reshape_strides(&resolved, self.elements)?
        };
        Ok(Self {
            storage: Arc::clone(&self.storage),
            shape: resolved,
            strides,
            offset: self.offset,
            elements: self.elements,
        })
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
        Self::from_owned_parts(
            vec![self.as_slice().iter().sum()],
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
        Ok(self.as_slice()[0])
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
        let left_data = self.as_slice();
        let right_data = other.as_slice();
        for row in 0..rows {
            for depth in 0..inner {
                let left = left_data[row * inner + depth];
                for column in 0..columns {
                    output[row * columns + column] += left * right_data[depth * columns + column];
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
        let left_data = self.as_slice();
        let right_data = other.as_slice();

        for output_offset in 0..plan.elements {
            data.push(operation(left_data[left_offset], right_data[right_offset]));
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
        let strides = if elements == 0 {
            contiguous_strides(&shape, elements)?
        } else {
            try_clone_result_shape(&self.strides, elements)?
        };
        data.extend(
            self.as_slice()
                .iter()
                .copied()
                .zip(other.as_slice().iter().copied())
                .map(|(left, right)| operation(left, right)),
        );
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
            try_clone_result_shape(&self.strides, elements)?
        };
        data.extend(
            self.as_slice()
                .iter()
                .copied()
                .map(|value| operation(value, scalar)),
        );
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
        let strides = if elements == 0 {
            contiguous_strides(&shape, elements)?
        } else {
            try_clone_result_shape(&self.strides, elements)?
        };
        data.extend(self.as_slice().iter().copied().map(operation));
        Ok(Self::from_owned_parts(
            data,
            shape,
            strides,
            self.dtype(),
            self.device(),
        ))
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

fn try_insert_signed_stride(
    strides: &[usize],
    axis: usize,
    inserted: i64,
    elements: usize,
) -> Result<Vec<i64>, TensorError> {
    let output_rank = strides
        .len()
        .checked_add(1)
        .ok_or(TensorError::AllocationFailed { elements })?;
    let mut output = try_result_vector(output_rank, elements)?;
    for stride in &strides[..axis] {
        output.push(i64::try_from(*stride).map_err(|_| TensorError::StrideCalculationOverflow)?);
    }
    output.push(inserted);
    for stride in &strides[axis..] {
        output.push(i64::try_from(*stride).map_err(|_| TensorError::StrideCalculationOverflow)?);
    }
    Ok(output)
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

fn copied_storage(values: &[f32], elements: usize) -> Result<Vec<f32>, TensorError> {
    validate_storage_capacity(elements)?;

    let mut data = try_result_vector(elements, elements)?;
    data.extend_from_slice(values);
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
