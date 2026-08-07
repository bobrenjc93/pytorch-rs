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
                write!(formatter, "tensor shapes differ: {left:?} and {right:?}")
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

impl Tensor {
    /// Creates a tensor after validating that `shape` describes `data`.
    ///
    /// # Errors
    ///
    /// Returns an error when the element count overflows or differs from the
    /// supplied data length.
    pub fn from_vec(data: Vec<f32>, shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let expected = element_count(&shape)?;
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
    /// Returns an error when the shape's element count or storage size
    /// overflows, or when storage allocation fails.
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::full(shape, 0.0)
    }

    /// Creates a one-filled tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count or storage size
    /// overflows, or when storage allocation fails.
    pub fn ones(shape: impl Into<Vec<usize>>) -> Result<Self, TensorError> {
        Self::full(shape, 1.0)
    }

    /// Creates a tensor filled with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape's element count or storage size
    /// overflows, or when storage allocation fails.
    pub fn full(shape: impl Into<Vec<usize>>, fill_value: f32) -> Result<Self, TensorError> {
        let shape = shape.into();
        let elements = element_count(&shape)?;
        let data = filled_storage(elements, fill_value)?;
        Ok(Self { data, shape })
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

    /// Adds tensors element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes differ.
    pub fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left + right)
    }

    /// Multiplies tensors element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when the shapes differ.
    pub fn mul(&self, other: &Self) -> Result<Self, TensorError> {
        self.zip_map(other, |left, right| left * right)
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
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: other.shape.clone(),
            });
        }
        Ok(Self {
            data: self
                .data
                .iter()
                .copied()
                .zip(other.data.iter().copied())
                .map(|(left, right)| operation(left, right))
                .collect(),
            shape: self.shape.clone(),
        })
    }
}

fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or(TensorError::ElementCountOverflow)
    })
}

fn filled_storage(elements: usize, fill_value: f32) -> Result<Vec<f32>, TensorError> {
    let maximum_elements = isize::MAX.unsigned_abs() / size_of::<f32>();
    if elements > maximum_elements {
        return Err(TensorError::StorageCapacityOverflow { elements });
    }

    let mut data = Vec::new();
    data.try_reserve_exact(elements)
        .map_err(|_| TensorError::AllocationFailed { elements })?;
    data.resize(elements, fill_value);
    Ok(data)
}
