use std::error::Error;
use std::fmt::{Display, Formatter};
use std::mem::size_of;

/// A contiguous, row-major, CPU `f32` tensor.
///
/// This deliberately narrow representation is the campaign's baseline, not a
/// claim of `PyTorch` feature parity. Later iterations may generalize storage as
/// long as these observable semantics remain compatible.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    ShapeDataMismatch { shape: Vec<usize>, elements: usize },
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    MatmulRequiresMatrices { left: Vec<usize>, right: Vec<usize> },
    MatmulInnerDimensionMismatch { left: Vec<usize>, right: Vec<usize> },
    ItemRequiresOneElement { elements: usize },
    ElementCountOverflow,
    StrideCalculationOverflow,
    StorageCapacityOverflow { elements: usize },
    AllocationFailed { elements: usize },
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
            Self::ItemRequiresOneElement { elements } => {
                write!(formatter, "item requires one element, got {elements}")
            }
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
        let shape = shape.into();
        let expected = validated_element_count(&shape)?;
        if data.len() != expected {
            return Err(TensorError::ShapeDataMismatch {
                shape,
                elements: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Creates a zero-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let elements = validated_element_count(&shape)?;
        validate_storage_capacity(elements)?;
        Ok(Self {
            data: vec![0.0; elements],
            shape,
        })
    }

    /// Creates a one-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let elements = validated_element_count(&shape)?;
        validate_storage_capacity(elements)?;
        Ok(Self {
            data: vec![1.0; elements],
            shape,
        })
    }

    /// Creates a tensor filled with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count, contiguous stride, or
    /// storage size overflows, or when storage allocation fails.
    pub fn full(shape: impl Into<Vec<usize>>, fill_value: f32) -> Result<Self, TensorError> {
        let shape = shape.into();
        let elements = Self::validate_full_shape(&shape)?;
        let data = filled_storage(elements, fill_value)?;
        Ok(Self { data, shape })
    }

    pub(crate) fn validate_full_shape(shape: &[usize]) -> Result<usize, TensorError> {
        let elements = validated_element_count(shape)?;
        validate_storage_capacity(elements)?;
        Ok(elements)
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.data
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

    /// Divides a scalar by every element using IEEE 754 true division.
    ///
    /// # Errors
    ///
    /// Returns an error when result allocation fails.
    pub fn scalar_div(&self, scalar: f32) -> Result<Self, TensorError> {
        self.map_scalar(scalar, |value, scalar| scalar / value)
    }

    #[must_use]
    pub fn relu(&self) -> Self {
        Self {
            data: self.data.iter().map(|value| value.max(0.0)).collect(),
            shape: self.shape.clone(),
        }
    }

    #[must_use]
    pub fn sum(&self) -> Self {
        Self {
            data: vec![self.data.iter().sum()],
            shape: vec![],
        }
    }

    /// Extracts the value of a one-element tensor.
    ///
    /// # Errors
    ///
    /// Returns an error unless the tensor contains exactly one element.
    pub fn item(&self) -> Result<f32, TensorError> {
        if self.data.len() != 1 {
            return Err(TensorError::ItemRequiresOneElement {
                elements: self.data.len(),
            });
        }
        Ok(self.data[0])
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

        let mut output = vec![0.0; rows * columns];
        for row in 0..rows {
            for depth in 0..inner {
                let left = self.data[row * inner + depth];
                for column in 0..columns {
                    output[row * columns + column] += left * other.data[depth * columns + column];
                }
            }
        }
        Ok(Self {
            data: output,
            shape: vec![rows, columns],
        })
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
            return Ok(Self {
                data,
                shape: plan.shape,
            });
        }

        let mut coordinates = try_result_vector(plan.shape.len(), plan.elements)?;
        coordinates.resize(plan.shape.len(), 0_usize);
        let mut left_offset = 0_usize;
        let mut right_offset = 0_usize;

        for output_offset in 0..plan.elements {
            data.push(operation(self.data[left_offset], other.data[right_offset]));
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

        Ok(Self {
            data,
            shape: plan.shape,
        })
    }

    fn zip_map_same_shape(
        &self,
        other: &Self,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let elements = self.data.len();
        let mut data = try_result_vector(elements, elements)?;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        data.extend(
            self.data
                .iter()
                .copied()
                .zip(other.data.iter().copied())
                .map(|(left, right)| operation(left, right)),
        );
        Ok(Self { data, shape })
    }

    fn map_scalar(
        &self,
        scalar: f32,
        operation: impl Fn(f32, f32) -> f32,
    ) -> Result<Self, TensorError> {
        let elements = self.data.len();
        let mut data = try_result_vector(elements, elements)?;
        let shape = try_clone_result_shape(&self.shape, elements)?;
        data.extend(
            self.data
                .iter()
                .copied()
                .map(|value| operation(value, scalar)),
        );
        Ok(Self { data, shape })
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
                    left: try_clone_result_shape(&left.shape, left.data.len())?,
                    right: try_clone_result_shape(&right.shape, right.data.len())?,
                });
            }
        }

        let mut elements = 1_usize;
        let mut output_stride = 1_usize;
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
        for axis in (1..rank).rev() {
            let dimension = broadcast_dimension(
                aligned_dimension(&left.shape, rank, axis),
                aligned_dimension(&right.shape, rank, axis),
            )
            .expect("broadcast compatibility was checked above");
            output_stride = checked_stride_product(output_stride, dimension)?;
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

        let mut dimensions = try_result_vector(rank, elements)?;
        let mut left_stride = 1_usize;
        let mut right_stride = 1_usize;
        for axis in (0..rank).rev() {
            let output_dimension = shape[axis];
            let left_step = aligned_broadcast_stride(
                &left.shape,
                rank,
                axis,
                output_dimension,
                &mut left_stride,
            )?;
            let right_step = aligned_broadcast_stride(
                &right.shape,
                rank,
                axis,
                output_dimension,
                &mut right_stride,
            )?;
            let repeats = output_dimension.saturating_sub(1);
            let left_rewind = if elements == 0 {
                0
            } else {
                left_step
                    .checked_mul(repeats)
                    .ok_or(TensorError::StrideCalculationOverflow)?
            };
            let right_rewind = if elements == 0 {
                0
            } else {
                right_step
                    .checked_mul(repeats)
                    .ok_or(TensorError::StrideCalculationOverflow)?
            };
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
    shape: &[usize],
    output_rank: usize,
    output_axis: usize,
    output_dimension: usize,
    contiguous_stride: &mut usize,
) -> Result<usize, TensorError> {
    let leading_dimensions = output_rank - shape.len();
    if output_axis < leading_dimensions {
        return Ok(0);
    }

    let input_axis = output_axis - leading_dimensions;
    let input_dimension = shape[input_axis];
    let step = if input_dimension == 1 && output_dimension != 1 {
        0
    } else {
        *contiguous_stride
    };
    if input_axis > 0 {
        *contiguous_stride = checked_stride_product(*contiguous_stride, input_dimension)?;
    }
    Ok(step)
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

fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(TensorError::ElementCountOverflow)
    })
}

fn validated_element_count(shape: &[usize]) -> Result<usize, TensorError> {
    let elements = element_count(shape)?;
    validate_contiguous_strides(shape)?;
    Ok(elements)
}

fn validate_contiguous_strides(shape: &[usize]) -> Result<(), TensorError> {
    shape
        .iter()
        .skip(1)
        .rev()
        .try_fold(1_usize, |stride, dimension| {
            checked_stride_product(stride, *dimension)
        })?;
    Ok(())
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
