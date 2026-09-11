//! Metadata planning for the existing CUDA capture operations. No allocation
//! of tensor storage or kernel launch is allowed while constructing this plan.
use super::{ElementwiseLayout, Tensor, TensorError, elementwise_output_strides, validated_layout};

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    Neg(usize),
    MulScalar(usize, f32),
    Add(usize, usize),
    Matmul(usize, usize),
    SumRows(usize, bool),
}

pub(crate) struct Layout {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl Layout {
    pub(crate) fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            strides: tensor.stride().to_vec(),
        }
    }

    fn elementwise(&self) -> ElementwiseLayout<'_> {
        ElementwiseLayout {
            shape: &self.shape,
            strides: &self.strides,
        }
    }

    pub(crate) fn matches(&self, tensor: &Tensor) -> bool {
        self.shape == tensor.shape() && self.strides == tensor.stride()
    }
}

impl Operation {
    pub(crate) fn layout(self, values: &[Layout]) -> Result<Layout, TensorError> {
        let get = |index| {
            values
                .get(index)
                .ok_or(TensorError::IndexCalculationOverflow)
        };
        let shape = match self {
            Self::Neg(input) | Self::MulScalar(input, _) => get(input)?.shape.clone(),
            Self::Add(left, right) => {
                let (left, right) = (get(left)?, get(right)?);
                if left.shape == right.shape {
                    left.shape.clone()
                } else {
                    match (left.shape.as_slice(), right.shape.as_slice()) {
                        ([_, columns], [n]) if columns == n => left.shape.clone(),
                        ([n], [_, columns]) if columns == n => right.shape.clone(),
                        _ => {
                            return Err(TensorError::UnsupportedCudaAddition {
                                reason: "inputs must have the same shape or shapes (M, N) and (N,)",
                            });
                        }
                    }
                }
            }
            Self::SumRows(input, keepdim) => {
                let [rows, _] = get(input)?.shape.as_slice() else {
                    return Err(TensorError::UnsupportedCudaSum {
                        reason: "input must be rank-2",
                    });
                };
                if keepdim { vec![*rows, 1] } else { vec![*rows] }
            }
            Self::Matmul(left, right) => {
                let (left, right) = (get(left)?, get(right)?);
                let ([rows, inner], [other_inner, columns]) =
                    (left.shape.as_slice(), right.shape.as_slice())
                else {
                    return Err(TensorError::UnsupportedCudaMatmul {
                        reason: "operands must be rank-2 matrices",
                    });
                };
                if inner != other_inner {
                    return Err(TensorError::MatmulInnerDimensionMismatch {
                        left: left.shape.clone(),
                        right: right.shape.clone(),
                    });
                }
                if [*rows, *inner, *columns]
                    .iter()
                    .any(|&n| i64::try_from(n).is_err())
                {
                    return Err(TensorError::IndexCalculationOverflow);
                }
                vec![*rows, *columns]
            }
        };
        let (elements, canonical) = validated_layout(&shape)?;
        elements
            .checked_mul(4)
            .filter(|&bytes| isize::try_from(bytes).is_ok())
            .ok_or(TensorError::AllocationFailed { elements })?;
        let strides = match self {
            Self::MulScalar(input, _) => {
                elementwise_output_strides(&shape, &[get(input)?.elementwise()], elements)?
            }
            Self::Add(left, right) if get(left)?.shape != get(right)?.shape => {
                // Eager addition normalizes the matrix operand to the left.
                let (left, right) = (get(left)?, get(right)?);
                let operands = if left.shape.len() == 2 {
                    [left.elementwise(), right.elementwise()]
                } else {
                    [right.elementwise(), left.elementwise()]
                };
                elementwise_output_strides(&shape, &operands, elements)?
            }
            _ => canonical,
        };
        Ok(Layout { shape, strides })
    }

    pub(crate) fn execute(
        self,
        inputs: &[&Tensor],
        outputs: &[Tensor],
    ) -> Result<Tensor, TensorError> {
        // Indices were checked for the entire plan before the first launch.
        let get = |index: usize| {
            if index < inputs.len() {
                inputs[index]
            } else {
                &outputs[index - inputs.len()]
            }
        };
        match self {
            Self::Neg(input) => get(input).negate(),
            Self::MulScalar(input, scalar) => get(input).mul_scalar(scalar),
            Self::Add(left, right) => get(left).add(get(right)),
            Self::Matmul(left, right) => get(left).matmul(get(right)),
            Self::SumRows(input, keepdim) => get(input).sum_rank_two_dimension(1, keepdim),
        }
    }
}
