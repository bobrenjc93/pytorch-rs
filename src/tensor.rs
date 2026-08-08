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
    /// Signed 64-bit integer.
    Int64,
}

impl Display for DType {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Float32 => formatter.write_str("float32"),
            Self::Int64 => formatter.write_str("int64"),
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

/// Owned, type-safe tensor payload.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorData {
    /// IEEE 754 single-precision values.
    Float32(Vec<f32>),
    /// Signed 64-bit integer values.
    Int64(Vec<i64>),
}

/// Borrowed, type-safe tensor payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TensorDataRef<'a> {
    /// IEEE 754 single-precision values.
    Float32(&'a [f32]),
    /// Signed 64-bit integer values.
    Int64(&'a [i64]),
}

impl TensorData {
    #[must_use]
    fn len(&self) -> usize {
        match self {
            Self::Float32(values) => values.len(),
            Self::Int64(values) => values.len(),
        }
    }

    #[must_use]
    fn dtype(&self) -> DType {
        match self {
            Self::Float32(_) => DType::Float32,
            Self::Int64(_) => DType::Int64,
        }
    }
}

/// A scalar value whose variant records its native dtype.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    /// IEEE 754 single-precision value.
    Float32(f32),
    /// Signed 64-bit integer value.
    Int64(i64),
}

impl Scalar {
    #[must_use]
    pub fn dtype(self) -> DType {
        match self {
            Self::Float32(_) => DType::Float32,
            Self::Int64(_) => DType::Int64,
        }
    }

    #[must_use]
    pub(crate) fn as_f32(self) -> f32 {
        match self {
            Self::Float32(value) => value,
            #[allow(clippy::cast_precision_loss)]
            Self::Int64(value) => value as f32,
        }
    }
}

#[derive(Clone, Copy)]
enum ArithmeticOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

fn promote_dtype(left: DType, right: DType, operation: ArithmeticOperation) -> DType {
    if matches!(operation, ArithmeticOperation::Divide)
        || matches!(left, DType::Float32)
        || matches!(right, DType::Float32)
    {
        DType::Float32
    } else {
        DType::Int64
    }
}

struct Storage {
    data: TensorData,
    device: Device,
}

/// A contiguous, row-major tensor with native storage metadata.
///
/// This deliberately narrow representation is the campaign's baseline, not a
/// claim of `PyTorch` feature parity. Later iterations may generalize storage as
/// long as these observable semantics remain compatible.
#[derive(Clone)]
pub struct Tensor {
    storage: Arc<Storage>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    elements: usize,
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
    MatmulDTypeMismatch {
        left: DType,
        right: DType,
    },
    ItemRequiresOneElement {
        elements: usize,
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
            Self::MatmulDTypeMismatch { left, right } => write!(
                formatter,
                "matmul requires tensors with the same dtype, but got {left} and {right}"
            ),
            Self::ItemRequiresOneElement { elements } => {
                write!(formatter, "item requires one element, got {elements}")
            }
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
            .field("data", &self.data())
            .field("shape", &self.shape)
            .finish()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.dtype() == other.dtype()
            && self.device() == other.device()
            && self.data() == other.data()
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
        Self::from_data_with_metadata(TensorData::Float32(data), shape, Device::Cpu)
    }

    /// Creates an int64 tensor after validating that `shape` describes `data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the element count or contiguous stride overflows,
    /// or when the element count differs from the supplied data length.
    pub fn from_i64_vec(data: Vec<i64>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::from_data_with_metadata(TensorData::Int64(data), shape, Device::Cpu)
    }

    pub(crate) fn from_data_with_metadata(
        data: TensorData,
        shape: impl Into<Vec<usize>>,
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
        validate_storage_capacity(expected, data.dtype())?;
        Ok(Self::from_owned_parts(data, shape, strides, device))
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
        let data = filled_storage(elements, Scalar::Int64(0), dtype)?;
        Ok(Self::from_owned_parts(data, shape, strides, device))
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
        let data = filled_storage(elements, Scalar::Int64(1), dtype)?;
        Ok(Self::from_owned_parts(data, shape, strides, device))
    }

    /// Creates a tensor filled with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows, or when storage allocation fails.
    pub fn full(shape: impl Into<Vec<usize>>, fill_value: f32) -> Result<Self, TensorError> {
        Self::full_with_metadata(
            shape,
            Scalar::Float32(fill_value),
            DType::Float32,
            Device::Cpu,
        )
    }

    /// Creates an int64 tensor filled with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error when layout or storage allocation fails.
    pub fn full_i64(shape: impl Into<Vec<usize>>, fill_value: i64) -> Result<Self, TensorError> {
        Self::full_with_metadata(shape, Scalar::Int64(fill_value), DType::Int64, Device::Cpu)
    }

    pub(crate) fn full_with_metadata(
        shape: impl Into<Vec<usize>>,
        fill_value: Scalar,
        dtype: DType,
        device: Device,
    ) -> Result<Self, TensorError> {
        let shape = shape.into();
        let (elements, strides) = validated_layout(&shape)?;
        let data = filled_storage(elements, fill_value, dtype)?;
        Ok(Self::from_owned_parts(data, shape, strides, device))
    }

    pub(crate) fn validate_full_shape(shape: &[usize], dtype: DType) -> Result<usize, TensorError> {
        let (elements, _) = validated_layout(shape)?;
        validate_storage_capacity(elements, dtype)?;
        Ok(elements)
    }

    fn from_owned_parts(
        data: TensorData,
        shape: Vec<usize>,
        strides: Vec<usize>,
        device: Device,
    ) -> Self {
        let elements = data.len();
        Self {
            storage: Arc::new(Storage { data, device }),
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

    /// Returns the scalar type physically represented by this tensor's storage.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.storage.data.dtype()
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

    #[must_use]
    /// # Panics
    ///
    /// Panics if this tensor is not float32, or if its private, validated
    /// layout invariant has been violated.
    pub fn as_slice(&self) -> &[f32] {
        self.as_f32_slice()
            .expect("as_slice is only available for float32 tensors")
    }

    /// Returns this view's float32 values, or `None` for another dtype.
    ///
    /// # Panics
    ///
    /// Panics only if the tensor's private, validated layout invariant has
    /// been violated.
    #[must_use]
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        let end = self
            .offset
            .checked_add(self.elements)
            .expect("validated tensor view end must fit in usize");
        match &self.storage.data {
            TensorData::Float32(values) => Some(&values[self.offset..end]),
            TensorData::Int64(_) => None,
        }
    }

    /// Returns this view's int64 values, or `None` for another dtype.
    ///
    /// # Panics
    ///
    /// Panics only if the tensor's private, validated layout invariant has
    /// been violated.
    #[must_use]
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        let end = self
            .offset
            .checked_add(self.elements)
            .expect("validated tensor view end must fit in usize");
        match &self.storage.data {
            TensorData::Float32(_) => None,
            TensorData::Int64(values) => Some(&values[self.offset..end]),
        }
    }

    /// Returns a borrowed, type-safe view of this tensor's payload.
    #[must_use]
    pub fn data(&self) -> TensorDataRef<'_> {
        match &self.storage.data {
            TensorData::Float32(values) => {
                TensorDataRef::Float32(&values[self.offset..self.offset + self.elements])
            }
            TensorData::Int64(values) => {
                TensorDataRef::Int64(&values[self.offset..self.offset + self.elements])
            }
        }
    }

    #[must_use]
    /// # Panics
    ///
    /// Panics if this tensor is not float32, or if its private, validated
    /// layout invariant has been violated.
    pub fn into_vec(self) -> Vec<f32> {
        let Self {
            storage,
            offset,
            elements,
            ..
        } = self;
        match Arc::try_unwrap(storage) {
            Ok(storage) => match storage.data {
                TensorData::Float32(values) => owned_view(values, offset, elements),
                TensorData::Int64(_) => panic!("into_vec is only available for float32 tensors"),
            },
            Err(storage) => match &storage.data {
                TensorData::Float32(values) => values[offset..offset + elements].to_vec(),
                TensorData::Int64(_) => panic!("into_vec is only available for float32 tensors"),
            },
        }
    }

    /// Consumes an int64 tensor and returns this view's values.
    ///
    /// # Panics
    ///
    /// Panics if this tensor is not int64, or if its private, validated layout
    /// invariant has been violated.
    #[must_use]
    pub fn into_i64_vec(self) -> Vec<i64> {
        let Self {
            storage,
            offset,
            elements,
            ..
        } = self;
        match Arc::try_unwrap(storage) {
            Ok(storage) => match storage.data {
                TensorData::Int64(values) => owned_view(values, offset, elements),
                TensorData::Float32(_) => {
                    panic!("into_i64_vec is only available for int64 tensors")
                }
            },
            Err(storage) => match &storage.data {
                TensorData::Int64(values) => values[offset..offset + elements].to_vec(),
                TensorData::Float32(_) => {
                    panic!("into_i64_vec is only available for int64 tensors")
                }
            },
        }
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
        self.binary_operation(
            other,
            ArithmeticOperation::Add,
            |left, right| left + right,
            i64::wrapping_add,
        )
    }

    /// Subtracts tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_operation(
            other,
            ArithmeticOperation::Subtract,
            |left, right| left - right,
            i64::wrapping_sub,
        )
    }

    /// Multiplies tensors element by element with trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_operation(
            other,
            ArithmeticOperation::Multiply,
            |left, right| left * right,
            i64::wrapping_mul,
        )
    }

    /// Divides tensors element by element using IEEE 754 true division and
    /// trailing-dimension broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes are not broadcastable or when result
    /// shape calculation or allocation fails.
    pub fn div(&self, other: &Self) -> Result<Self, TensorError> {
        self.binary_operation(
            other,
            ArithmeticOperation::Divide,
            |left, right| left / right,
            |_, _| unreachable!("true division is always promoted to float32"),
        )
    }

    /// Adds a scalar to every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn add_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.scalar_operation(
            Scalar::Float32(scalar),
            ArithmeticOperation::Add,
            false,
            |left, right| left + right,
            i64::wrapping_add,
        )
    }

    /// Subtracts a scalar from every element.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn sub_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.scalar_operation(
            Scalar::Float32(scalar),
            ArithmeticOperation::Subtract,
            false,
            |left, right| left - right,
            i64::wrapping_sub,
        )
    }

    /// Multiplies every element by a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.scalar_operation(
            Scalar::Float32(scalar),
            ArithmeticOperation::Multiply,
            false,
            |left, right| left * right,
            i64::wrapping_mul,
        )
    }

    /// Divides every element by a scalar using IEEE 754 true division.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn div_scalar(&self, scalar: f32) -> Result<Self, TensorError> {
        self.scalar_operation(
            Scalar::Float32(scalar),
            ArithmeticOperation::Divide,
            false,
            |left, right| left / right,
            |_, _| unreachable!("true division is always promoted to float32"),
        )
    }

    /// Subtracts every element from a scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_sub(&self, scalar: f32) -> Result<Self, TensorError> {
        self.scalar_operation(
            Scalar::Float32(scalar),
            ArithmeticOperation::Subtract,
            true,
            |left, right| left - right,
            i64::wrapping_sub,
        )
    }

    /// Divides a scalar by every element using `PyTorch`'s float32 reciprocal
    /// multiplication semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_div(&self, scalar: f32) -> Result<Self, TensorError> {
        self.scalar_operation(
            Scalar::Float32(scalar),
            ArithmeticOperation::Divide,
            true,
            |left, right| left / right,
            |_, _| unreachable!("true division is always promoted to float32"),
        )
    }

    /// Applies a typed scalar operation, preserving int64 when both operands
    /// are integral and promoting every true division to float32.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn add_typed_scalar(&self, scalar: Scalar) -> Result<Self, TensorError> {
        self.scalar_operation(
            scalar,
            ArithmeticOperation::Add,
            false,
            |left, right| left + right,
            i64::wrapping_add,
        )
    }

    /// See [`Tensor::add_typed_scalar`].
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn sub_typed_scalar(&self, scalar: Scalar, reverse: bool) -> Result<Self, TensorError> {
        self.scalar_operation(
            scalar,
            ArithmeticOperation::Subtract,
            reverse,
            |left, right| left - right,
            i64::wrapping_sub,
        )
    }

    /// See [`Tensor::add_typed_scalar`].
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn mul_typed_scalar(&self, scalar: Scalar) -> Result<Self, TensorError> {
        self.scalar_operation(
            scalar,
            ArithmeticOperation::Multiply,
            false,
            |left, right| left * right,
            i64::wrapping_mul,
        )
    }

    /// See [`Tensor::add_typed_scalar`].
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn div_typed_scalar(&self, scalar: Scalar, reverse: bool) -> Result<Self, TensorError> {
        self.scalar_operation(
            scalar,
            ArithmeticOperation::Divide,
            reverse,
            |left, right| left / right,
            |_, _| unreachable!("true division is always promoted to float32"),
        )
    }

    /// Applies rectified linear activation element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when result metadata or storage allocation fails.
    pub fn relu(&self) -> Result<Self, TensorError> {
        let elements = self.elements;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = if elements == 0 {
            contiguous_strides(&shape, elements)?
        } else {
            try_clone_result_shape(&self.strides, elements)?
        };
        let data = match self.data() {
            TensorDataRef::Float32(values) => {
                let mut output = try_result_vector(elements, elements)?;
                output.extend(
                    values
                        .iter()
                        .map(|value| if *value <= 0.0 { 0.0 } else { *value }),
                );
                TensorData::Float32(output)
            }
            TensorDataRef::Int64(values) => {
                let mut output = try_result_vector(elements, elements)?;
                output.extend(values.iter().map(|value| (*value).max(0)));
                TensorData::Int64(output)
            }
        };
        Ok(Self::from_owned_parts(data, shape, strides, self.device()))
    }

    #[must_use]
    pub fn sum(&self) -> Self {
        let data = match self.data() {
            TensorDataRef::Float32(values) => TensorData::Float32(vec![values.iter().sum()]),
            TensorDataRef::Int64(values) => {
                TensorData::Int64(vec![values.iter().copied().fold(0_i64, i64::wrapping_add)])
            }
        };
        Self::from_owned_parts(data, Vec::new(), Vec::new(), self.device())
    }

    /// Extracts the value of a one-element tensor.
    ///
    /// # Errors
    ///
    /// Returns an error unless the tensor contains exactly one element.
    pub fn item(&self) -> Result<f32, TensorError> {
        Ok(self.item_scalar()?.as_f32())
    }

    /// Extracts a one-element tensor without losing its dtype.
    ///
    /// # Errors
    ///
    /// Returns an error unless the tensor contains exactly one element.
    pub fn item_scalar(&self) -> Result<Scalar, TensorError> {
        if self.elements != 1 {
            return Err(TensorError::ItemRequiresOneElement {
                elements: self.elements,
            });
        }
        Ok(match self.data() {
            TensorDataRef::Float32(values) => Scalar::Float32(values[0]),
            TensorDataRef::Int64(values) => Scalar::Int64(values[0]),
        })
    }

    /// Multiplies two rank-2 matrices.
    ///
    /// # Errors
    ///
    /// Returns an error unless both tensors are matrices with compatible inner
    /// dimensions and matching dtypes. Shape validation takes precedence over
    /// dtype validation, matching `PyTorch` diagnostics.
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
        if self.dtype() != other.dtype() {
            return Err(TensorError::MatmulDTypeMismatch {
                left: self.dtype(),
                right: other.dtype(),
            });
        }

        let mut output_shape = try_result_vector(2, 0)?;
        output_shape.push(rows);
        output_shape.push(columns);
        let (output_elements, output_strides) = validated_layout(&output_shape)?;
        let output_dtype = self.dtype();
        validate_storage_capacity(output_elements, output_dtype)?;
        let output = match output_dtype {
            DType::Float32 => {
                let (TensorDataRef::Float32(left), TensorDataRef::Float32(right)) =
                    (self.data(), other.data())
                else {
                    unreachable!("matching float32 dtypes require two float32 tensors")
                };
                TensorData::Float32(matmul_kernel(
                    (rows, inner, columns),
                    output_elements,
                    0.0_f32,
                    |index| left[index],
                    |index| right[index],
                    |output, left, right| output + left * right,
                )?)
            }
            DType::Int64 => {
                let (TensorDataRef::Int64(left), TensorDataRef::Int64(right)) =
                    (self.data(), other.data())
                else {
                    unreachable!("int64 promotion requires two int64 tensors")
                };
                TensorData::Int64(matmul_kernel(
                    (rows, inner, columns),
                    output_elements,
                    0_i64,
                    |index| left[index],
                    |index| right[index],
                    |output, left, right| output.wrapping_add(left.wrapping_mul(right)),
                )?)
            }
        };
        Ok(Self::from_owned_parts(
            output,
            output_shape,
            output_strides,
            self.device(),
        ))
    }

    fn binary_operation(
        &self,
        other: &Self,
        operation: ArithmeticOperation,
        float_kernel: impl Fn(f32, f32) -> f32 + Copy,
        integer_kernel: impl Fn(i64, i64) -> i64 + Copy,
    ) -> Result<Self, TensorError> {
        let dtype = promote_dtype(self.dtype(), other.dtype(), operation);
        if self.shape == other.shape {
            return self.binary_operation_same_shape(other, dtype, float_kernel, integer_kernel);
        }
        let plan = BroadcastPlan::new(self, other, dtype)?;
        let data = match dtype {
            DType::Float32 => TensorData::Float32(match (self.data(), other.data()) {
                (TensorDataRef::Float32(left), TensorDataRef::Float32(right)) => {
                    Self::collect_binary(&plan, |left_offset, right_offset| {
                        float_kernel(left[left_offset], right[right_offset])
                    })?
                }
                (TensorDataRef::Float32(left), TensorDataRef::Int64(right)) => {
                    Self::collect_binary(&plan, |left_offset, right_offset| {
                        float_kernel(left[left_offset], i64_as_f32(right[right_offset]))
                    })?
                }
                (TensorDataRef::Int64(left), TensorDataRef::Float32(right)) => {
                    Self::collect_binary(&plan, |left_offset, right_offset| {
                        float_kernel(i64_as_f32(left[left_offset]), right[right_offset])
                    })?
                }
                (TensorDataRef::Int64(left), TensorDataRef::Int64(right)) => {
                    Self::collect_binary(&plan, |left_offset, right_offset| {
                        float_kernel(
                            i64_as_f32(left[left_offset]),
                            i64_as_f32(right[right_offset]),
                        )
                    })?
                }
            }),
            DType::Int64 => {
                let (TensorDataRef::Int64(left), TensorDataRef::Int64(right)) =
                    (self.data(), other.data())
                else {
                    unreachable!("int64 promotion requires two int64 tensors")
                };
                TensorData::Int64(Self::collect_binary(&plan, |left_offset, right_offset| {
                    integer_kernel(left[left_offset], right[right_offset])
                })?)
            }
        };
        Ok(Self::from_owned_parts(
            data,
            plan.shape,
            plan.strides,
            self.device(),
        ))
    }

    fn binary_operation_same_shape(
        &self,
        other: &Self,
        dtype: DType,
        float_kernel: impl Fn(f32, f32) -> f32,
        integer_kernel: impl Fn(i64, i64) -> i64,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        validate_storage_capacity(elements, dtype)?;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = if elements == 0 {
            contiguous_strides(&shape, elements)?
        } else {
            try_clone_result_shape(&self.strides, elements)?
        };
        let data = match dtype {
            DType::Float32 => {
                let mut output = try_result_vector(elements, elements)?;
                match (self.data(), other.data()) {
                    (TensorDataRef::Float32(left), TensorDataRef::Float32(right)) => output.extend(
                        left.iter()
                            .copied()
                            .zip(right.iter().copied())
                            .map(|(left, right)| float_kernel(left, right)),
                    ),
                    (TensorDataRef::Float32(left), TensorDataRef::Int64(right)) => output.extend(
                        left.iter()
                            .copied()
                            .zip(right.iter().copied())
                            .map(|(left, right)| float_kernel(left, i64_as_f32(right))),
                    ),
                    (TensorDataRef::Int64(left), TensorDataRef::Float32(right)) => output.extend(
                        left.iter()
                            .copied()
                            .zip(right.iter().copied())
                            .map(|(left, right)| float_kernel(i64_as_f32(left), right)),
                    ),
                    (TensorDataRef::Int64(left), TensorDataRef::Int64(right)) => {
                        output.extend(left.iter().copied().zip(right.iter().copied()).map(
                            |(left, right)| float_kernel(i64_as_f32(left), i64_as_f32(right)),
                        ));
                    }
                }
                TensorData::Float32(output)
            }
            DType::Int64 => {
                let (TensorDataRef::Int64(left), TensorDataRef::Int64(right)) =
                    (self.data(), other.data())
                else {
                    unreachable!("int64 promotion requires two int64 tensors")
                };
                let mut output = try_result_vector(elements, elements)?;
                output.extend(
                    left.iter()
                        .copied()
                        .zip(right.iter().copied())
                        .map(|(left, right)| integer_kernel(left, right)),
                );
                TensorData::Int64(output)
            }
        };
        Ok(Self::from_owned_parts(data, shape, strides, self.device()))
    }

    fn collect_binary<T>(
        plan: &BroadcastPlan,
        mut operation: impl FnMut(usize, usize) -> T,
    ) -> Result<Vec<T>, TensorError> {
        let mut data = try_result_vector(plan.elements, plan.elements)?;
        if plan.elements == 0 {
            return Ok(data);
        }
        let mut coordinates = try_result_vector(plan.shape.len(), plan.elements)?;
        coordinates.resize(plan.shape.len(), 0_usize);
        let mut left_offset = 0_usize;
        let mut right_offset = 0_usize;
        for output_offset in 0..plan.elements {
            data.push(operation(left_offset, right_offset));
            if output_offset + 1 == plan.elements {
                break;
            }
            advance_broadcast_offsets(plan, &mut coordinates, &mut left_offset, &mut right_offset)?;
        }
        Ok(data)
    }

    fn scalar_operation(
        &self,
        scalar: Scalar,
        operation: ArithmeticOperation,
        reverse: bool,
        float_kernel: impl Fn(f32, f32) -> f32 + Copy,
        integer_kernel: impl Fn(i64, i64) -> i64 + Copy,
    ) -> Result<Self, TensorError> {
        let elements = self.elements;
        let dtype = promote_dtype(self.dtype(), scalar.dtype(), operation);
        let shape = try_clone_result_shape(&self.shape, elements)?;
        let strides = if elements == 0 {
            elementwise_output_strides(&shape, &[self], elements, dtype)?
        } else {
            try_clone_result_shape(&self.strides, elements)?
        };
        validate_storage_capacity(elements, dtype)?;
        let data = match dtype {
            DType::Float32 => {
                let scalar = scalar.as_f32();
                let mut output = try_result_vector(elements, elements)?;
                let apply = |value: f32| match (reverse, operation) {
                    (true, ArithmeticOperation::Divide) => scalar * value.recip(),
                    (true, _) => float_kernel(scalar, value),
                    (false, _) => float_kernel(value, scalar),
                };
                match self.data() {
                    TensorDataRef::Float32(values) => {
                        output.extend(values.iter().copied().map(apply));
                    }
                    TensorDataRef::Int64(values) => {
                        output.extend(values.iter().copied().map(i64_as_f32).map(apply));
                    }
                }
                TensorData::Float32(output)
            }
            DType::Int64 => {
                let (TensorDataRef::Int64(values), Scalar::Int64(scalar)) = (self.data(), scalar)
                else {
                    unreachable!("int64 promotion requires int64 tensor and scalar storage")
                };
                let mut output = try_result_vector(elements, elements)?;
                output.extend(values.iter().map(|value| {
                    if reverse {
                        integer_kernel(scalar, *value)
                    } else {
                        integer_kernel(*value, scalar)
                    }
                }));
                TensorData::Int64(output)
            }
        };
        Ok(Self::from_owned_parts(data, shape, strides, self.device()))
    }
}

#[allow(clippy::cast_precision_loss)]
fn i64_as_f32(value: i64) -> f32 {
    value as f32
}

fn matmul_kernel<T: Clone + Copy>(
    dimensions: (usize, usize, usize),
    output_elements: usize,
    zero: T,
    mut left_value: impl FnMut(usize) -> T,
    mut right_value: impl FnMut(usize) -> T,
    mut multiply_add: impl FnMut(T, T, T) -> T,
) -> Result<Vec<T>, TensorError> {
    let (rows, inner, columns) = dimensions;
    let mut output = filled_vector(output_elements, zero)?;
    for row in 0..rows {
        for depth in 0..inner {
            let left = left_value(row * inner + depth);
            for column in 0..columns {
                let offset = row * columns + column;
                output[offset] =
                    multiply_add(output[offset], left, right_value(depth * columns + column));
            }
        }
    }
    Ok(output)
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
    fn new(left: &Tensor, right: &Tensor, output_dtype: DType) -> Result<Self, TensorError> {
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
        validate_storage_capacity(elements, output_dtype)?;

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
            elementwise_output_strides(&shape, &[left, right], elements, output_dtype)?
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

fn advance_broadcast_offsets(
    plan: &BroadcastPlan,
    coordinates: &mut [usize],
    left_offset: &mut usize,
    right_offset: &mut usize,
) -> Result<(), TensorError> {
    for axis in (0..plan.shape.len()).rev() {
        coordinates[axis] = coordinates[axis]
            .checked_add(1)
            .ok_or(TensorError::StrideCalculationOverflow)?;
        if coordinates[axis] < plan.shape[axis] {
            *left_offset = left_offset
                .checked_add(plan.dimensions[axis].left_step)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            *right_offset = right_offset
                .checked_add(plan.dimensions[axis].right_step)
                .ok_or(TensorError::StrideCalculationOverflow)?;
            return Ok(());
        }

        coordinates[axis] = 0;
        *left_offset = left_offset
            .checked_sub(plan.dimensions[axis].left_rewind)
            .ok_or(TensorError::StrideCalculationOverflow)?;
        *right_offset = right_offset
            .checked_sub(plan.dimensions[axis].right_rewind)
            .ok_or(TensorError::StrideCalculationOverflow)?;
    }
    Ok(())
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
    iteration_dtype: DType,
) -> i64 {
    // TensorIterator compares byte strides stored in signed 64-bit integers.
    // Dtype-promoting pointwise operations plan against the common iteration
    // dtype, including when an empty input needs no physical conversion.
    // Preserve its wrapping conversion at this boundary: an extreme but valid
    // empty view can therefore change the recovered output permutation without
    // accessing any storage.
    let stride =
        aligned_broadcast_stride(tensor, output_rank, output_axis, output_dimension).cast_signed();
    let element_size = i64::try_from(dtype_size(iteration_dtype))
        .expect("a supported element size must fit in i64");
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
    output_dtype: DType,
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
                output_dtype,
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
        i64::try_from(dtype_size(output_dtype)).expect("a supported element size must fit in i64");
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
    iteration_dtype: DType,
) -> i8 {
    for tensor in operands {
        let stride_0 = aligned_broadcast_stride_bytes(
            tensor,
            shape.len(),
            dimension_0,
            shape[dimension_0],
            iteration_dtype,
        );
        let stride_1 = aligned_broadcast_stride_bytes(
            tensor,
            shape.len(),
            dimension_1,
            shape[dimension_1],
            iteration_dtype,
        );
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

fn filled_storage(
    elements: usize,
    fill_value: Scalar,
    dtype: DType,
) -> Result<TensorData, TensorError> {
    validate_storage_capacity(elements, dtype)?;
    Ok(match dtype {
        DType::Float32 => TensorData::Float32(filled_vector(elements, fill_value.as_f32())?),
        DType::Int64 => {
            let value = match fill_value {
                Scalar::Int64(value) => value,
                #[allow(clippy::cast_possible_truncation)]
                Scalar::Float32(value) => value as i64,
            };
            TensorData::Int64(filled_vector(elements, value)?)
        }
    })
}

fn filled_vector<T: Clone>(elements: usize, fill_value: T) -> Result<Vec<T>, TensorError> {
    let mut data = Vec::new();
    data.try_reserve_exact(elements)
        .map_err(|_| TensorError::AllocationFailed { elements })?;
    data.resize(elements, fill_value);
    Ok(data)
}

fn validate_storage_capacity(elements: usize, dtype: DType) -> Result<(), TensorError> {
    let maximum_elements = isize::MAX.unsigned_abs() / dtype_size(dtype);
    if elements > maximum_elements {
        return Err(TensorError::StorageCapacityOverflow { elements });
    }
    Ok(())
}

const fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::Float32 => size_of::<f32>(),
        DType::Int64 => size_of::<i64>(),
    }
}

fn owned_view<T: Clone>(values: Vec<T>, offset: usize, elements: usize) -> Vec<T> {
    if offset == 0 && elements == values.len() {
        values
    } else {
        values[offset..offset + elements].to_vec()
    }
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
