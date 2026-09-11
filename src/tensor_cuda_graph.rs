//! Metadata planning for the existing CUDA capture operations. No allocation
//! of tensor storage or kernel launch is allowed while constructing this plan.
use super::{
    ElementwiseLayout, MemoryFormat, Tensor, TensorError, contiguous_strides, element_count,
    elementwise_output_strides, layout_is_contiguous, normalize_transpose_dimension,
    reshape_view_strides, validated_layout,
};

#[derive(Clone, Copy)]
pub(crate) enum Operation {
    Contiguous(usize),
    T(usize),
    Transpose(usize, i64, i64),
    Reshape(usize, Shape),
    Neg(usize),
    MulScalar(usize, f32),
    Add(usize, usize),
    Matmul(usize, usize),
    SumRows(usize, bool),
}

/// Bounded requested dimensions, separate from scalar/reduction/axis payloads.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    dimensions: [i64; 2],
    rank: usize,
}

impl Shape {
    pub(crate) fn new(requested: &[i64]) -> Result<Self, TensorError> {
        if requested.len() > 2 {
            return Err(TensorError::UnsupportedCudaContiguous {
                reason: "compiled reshape requires output rank 0, 1 or 2",
            });
        }
        let mut dimensions = [0; 2];
        dimensions[..requested.len()].copy_from_slice(requested);
        Ok(Self {
            dimensions,
            rank: requested.len(),
        })
    }

    fn as_slice(&self) -> &[i64] {
        &self.dimensions[..self.rank]
    }
}

#[derive(Clone)]
pub(crate) struct Layout {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl Layout {
    pub(crate) fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            strides: tensor.stride().to_vec(),
            offset: tensor.storage_offset(),
        }
    }

    pub(crate) fn is_contiguous(&self) -> Result<bool, TensorError> {
        // Classify the existing layout without constructing materialized
        // strides: an empty alias can have no representable canonical layout.
        let elements = element_count(&self.shape)?;
        Ok(layout_is_contiguous(&self.shape, &self.strides, elements))
    }

    fn require_contiguous(&self) -> Result<&Self, TensorError> {
        if !self.is_contiguous()? {
            return Err(TensorError::UnsupportedCudaContiguous {
                reason: "CUDA graph arithmetic requires contiguous operands",
            });
        }
        Ok(self)
    }

    fn elementwise(&self) -> ElementwiseLayout<'_> {
        ElementwiseLayout {
            shape: &self.shape,
            strides: &self.strides,
        }
    }

    fn t(&self) -> Result<Self, TensorError> {
        if self.shape.len() > 2 {
            return Err(TensorError::UnsupportedCudaTranspose {
                reason: "Tensor.t requires rank 0, 1 or 2",
            });
        }
        // A checked view preserves storage/offset; no canonical
        // materialized strides or tensor allocation are needed.
        let mut output = self.clone();
        output.shape.reverse();
        output.strides.reverse();
        Ok(output)
    }

    fn transpose(&self, dim0: i64, dim1: i64) -> Result<Self, TensorError> {
        if self.shape.len() > 2 {
            return Err(TensorError::UnsupportedCudaTranspose {
                reason: "compiled Tensor.transpose requires rank 0, 1 or 2",
            });
        }
        let axis0 = normalize_transpose_dimension(dim0, self.shape.len())?;
        let axis1 = normalize_transpose_dimension(dim1, self.shape.len())?;
        let mut output = self.clone();
        if !output.shape.is_empty() {
            output.shape.swap(axis0, axis1);
            output.strides.swap(axis0, axis1);
        }
        Ok(output)
    }

    fn reshape(&self, requested: Shape) -> Result<Self, TensorError> {
        if self.shape.len() > 2 || self.shape.len() != self.strides.len() {
            return Err(TensorError::UnsupportedCudaContiguous {
                reason: "compiled reshape requires input rank 0, 1 or 2",
            });
        }
        let elements = element_count(&self.shape)?;
        let shape = Tensor::resolve_reshape_shape(requested.as_slice(), elements)?;
        if let Some(strides) = reshape_view_strides(&self.shape, &self.strides, &shape, elements)? {
            return Ok(Self {
                shape,
                strides,
                offset: self.offset,
            });
        }
        if !(1..=2).contains(&self.shape.len()) || self.strides.contains(&0) {
            return Err(TensorError::UnsupportedCudaContiguous {
                reason: "reshape packing requires positive-stride rank-1 or rank-2 inputs",
            });
        }
        // Validate both the pack allocation and the final output layout before
        // execution. The actual copy uses eager reshape's native CUDA pack.
        validated_layout(&self.shape)?;
        elements
            .checked_mul(4)
            .filter(|&bytes| isize::try_from(bytes).is_ok())
            .ok_or(TensorError::AllocationFailed { elements })?;
        let strides = contiguous_strides(&shape, elements)?;
        Ok(Self {
            shape,
            strides,
            offset: 0,
        })
    }

    pub(crate) fn matches(&self, tensor: &Tensor) -> bool {
        self.shape == tensor.shape()
            && self.strides == tensor.stride()
            && self.offset == tensor.storage_offset()
    }
}

impl Operation {
    pub(crate) fn layout(self, values: &[Layout]) -> Result<Layout, TensorError> {
        let get = |index: usize| {
            values
                .get(index)
                .ok_or(TensorError::IndexCalculationOverflow)
        };
        // Input admission allows views, but every arithmetic operand must
        // independently satisfy its contiguous-only contract during planning.
        let get_contiguous = |index| get(index)?.require_contiguous();
        let shape = match self {
            Self::T(input) => return get(input)?.t(),
            Self::Reshape(input, shape) => return get(input)?.reshape(shape),
            Self::Transpose(input, dim0, dim1) => return get(input)?.transpose(dim0, dim1),
            Self::Contiguous(input) => {
                let input = get(input)?;
                if input.is_contiguous()? {
                    return Ok(input.clone());
                }
                if !(1..=2).contains(&input.shape.len()) || input.strides.contains(&0) {
                    return Err(TensorError::UnsupportedCudaContiguous {
                        reason: "packing requires positive-stride rank-1 or rank-2 inputs",
                    });
                }
                input.shape.clone()
            }
            Self::Neg(input) | Self::MulScalar(input, _) => get_contiguous(input)?.shape.clone(),
            Self::Add(left, right) => {
                let (left, right) = (get_contiguous(left)?, get_contiguous(right)?);
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
                let [rows, _] = get_contiguous(input)?.shape.as_slice() else {
                    return Err(TensorError::UnsupportedCudaSum {
                        reason: "input must be rank-2",
                    });
                };
                if keepdim { vec![*rows, 1] } else { vec![*rows] }
            }
            Self::Matmul(left, right) => {
                let (left, right) = (get_contiguous(left)?, get_contiguous(right)?);
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
        Ok(Layout {
            shape,
            strides,
            offset: 0,
        })
    }

    pub(crate) fn execute(
        self,
        inputs: &[&Tensor],
        outputs: &[Tensor],
    ) -> Result<Tensor, TensorError> {
        #[cfg(test)]
        EXECUTIONS.with(|count| count.set(count.get() + 1));
        // Indices were checked for the entire plan before the first launch.
        let get = |index: usize| {
            if index < inputs.len() {
                inputs[index]
            } else {
                &outputs[index - inputs.len()]
            }
        };
        match self {
            Self::T(input) => get(input).t(),
            Self::Reshape(input, shape) => get(input).reshape(shape.as_slice()),
            Self::Transpose(input, dim0, dim1) => get(input).transpose(dim0, dim1),
            Self::Contiguous(input) => get(input).try_contiguous(MemoryFormat::Contiguous),
            Self::Neg(input) => get(input).negate(),
            Self::MulScalar(input, scalar) => get(input).mul_scalar(scalar),
            Self::Add(left, right) => get(left).add(get(right)),
            Self::Matmul(left, right) => get(left).matmul(get(right)),
            Self::SumRows(input, keepdim) => get(input).sum_rank_two_dimension(1, keepdim),
        }
    }
}

// Test-only, thread-local accounting: absent from release builds and unrelated
// to evaluator observers. Counts entry even if a native operation later fails.
#[cfg(test)]
thread_local! {
    pub(crate) static EXECUTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_plans_shared_views_and_device_packs() {
        use crate::{Device, cuda};
        use std::sync::Arc;
        let requests = [vec![-1], vec![1, -1], vec![3, 2]];
        let base = Tensor::from_vec((0_u16..6).map(f32::from).collect(), [2, 3]).unwrap();
        let transposed = base.t().unwrap();
        let column = base.select_dimension(1, 1).unwrap();
        let scalar = column.select_dimension(0, 1).unwrap();
        let empty = Tensor::from_vec(vec![], [0, 1_usize << 32]).unwrap();
        for source in [&base, &transposed, &column, &scalar, &empty] {
            let mut shapes = requests.to_vec();
            shapes.push(vec![]);
            shapes.push(
                source
                    .shape()
                    .iter()
                    .map(|&n| i64::try_from(n).unwrap())
                    .collect(),
            );
            for requested in shapes {
                let op = Operation::Reshape(0, Shape::new(&requested).unwrap());
                let planned = op.layout(&[Layout::from_tensor(source)]);
                let eager = source.reshape(&requested);
                assert_eq!(planned.is_ok(), eager.is_ok());
                let (Ok(plan), Ok(eager)) = (planned, eager) else {
                    continue;
                };
                assert!(plan.matches(&eager));
                if cuda::device_count() == 0 {
                    continue;
                }
                // Transfer the base first so the CUDA input retains the view.
                let gpu_base = base.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap();
                let gpu = if std::ptr::eq(source, &raw const base) {
                    gpu_base
                } else if std::ptr::eq(source, &raw const transposed) {
                    gpu_base.t().unwrap()
                } else if std::ptr::eq(source, &raw const column) {
                    gpu_base.select_dimension(1, 1).unwrap()
                } else if std::ptr::eq(source, &raw const scalar) {
                    gpu_base
                        .select_dimension(1, 1)
                        .unwrap()
                        .select_dimension(0, 1)
                        .unwrap()
                } else {
                    empty.try_copy_cpu_to_cuda(Device::Cuda(0)).unwrap()
                };
                let output = op.execute(&[&gpu], &[]).unwrap();
                assert!(
                    op.layout(&[Layout::from_tensor(&gpu)])
                        .unwrap()
                        .matches(&output)
                );
                assert_eq!(
                    Arc::ptr_eq(&source.storage, &eager.storage),
                    Arc::ptr_eq(&gpu.storage, &output.storage)
                );
                assert_eq!(
                    output.try_copy_cuda_to_cpu().unwrap().try_to_vec().unwrap(),
                    eager.try_to_vec().unwrap()
                );
            }
        }
        for requested in [vec![-2], vec![-1, -1], vec![7], vec![i64::MAX, 2]] {
            assert!(
                Operation::Reshape(0, Shape::new(&requested).unwrap())
                    .layout(&[Layout::from_tensor(&base)])
                    .is_err()
            );
        }
        assert!(Shape::new(&[1, 1, 1]).is_err());
        assert!(
            Operation::Reshape(1, Shape::new(&[-1]).unwrap())
                .layout(&[])
                .is_err()
        );
        assert!(
            Operation::Reshape(0, Shape::new(&[0, -1]).unwrap())
                .layout(&[Layout::from_tensor(&empty)])
                .is_err()
        );
    }

    #[test]
    fn t_plans_only_bounded_views_and_reuses_storage() {
        use crate::{Device, cuda};
        for (shape, strides) in [
            (vec![], vec![]),
            (vec![7], vec![3]),
            (vec![3, 7], vec![9, 1]),
            (vec![0, 1 << 32], vec![1 << 32, 1]),
        ] {
            let input = Layout {
                shape,
                strides,
                offset: 13,
            };
            let output = Operation::T(0)
                .layout(std::slice::from_ref(&input))
                .unwrap();
            assert_eq!(
                output.shape,
                input.shape.iter().rev().copied().collect::<Vec<_>>()
            );
            assert_eq!(
                output.strides,
                input.strides.iter().rev().copied().collect::<Vec<_>>()
            );
            assert_eq!(output.offset, 13);
        }
        assert!(
            Operation::T(0)
                .layout(&[Layout {
                    shape: vec![1, 2, 3],
                    strides: vec![6, 3, 1],
                    offset: 0,
                }])
                .is_err()
        );
        assert!(Operation::T(1).layout(&[]).is_err());
        if cuda::device_count() == 0 {
            eprintln!("skipping CUDA view storage check: no CUDA device");
            return;
        }
        for shape in [vec![], vec![6], vec![2, 3]] {
            let elements = element_count(&shape).unwrap();
            let input = Tensor::from_vec(vec![1.; elements], shape)
                .unwrap()
                .try_copy_cpu_to_cuda(Device::Cuda(0))
                .unwrap();
            let output = Operation::T(0).execute(&[&input], &[]).unwrap();
            assert!(std::sync::Arc::ptr_eq(&input.storage, &output.storage));
            assert_eq!(input.device(), output.device());
            assert_eq!(input.dtype(), output.dtype());
            assert_eq!(input.storage_offset(), output.storage_offset());
            assert!(
                Operation::T(0)
                    .layout(&[Layout::from_tensor(&input)])
                    .unwrap()
                    .matches(&output)
            );
        }
    }

    #[test]
    fn transpose_axes_plan_and_execute_shared_storage() {
        use crate::{Device, cuda};
        for shape in [vec![], vec![7], vec![3, 7], vec![0, 1 << 32]] {
            let (_, strides) = validated_layout(&shape).unwrap();
            let input = Layout {
                shape: shape.clone(),
                strides,
                offset: 3,
            };
            let rank = i64::try_from(shape.len().max(1)).unwrap();
            for dim0 in -rank..rank {
                for dim1 in -rank..rank {
                    let op = Operation::Transpose(0, dim0, dim1);
                    let output = op.layout(std::slice::from_ref(&input)).unwrap();
                    assert_eq!(output.offset, 3);
                    if cuda::device_count() != 0 {
                        let tensor = Tensor::from_vec(
                            vec![1.; element_count(&shape).unwrap()],
                            shape.clone(),
                        )
                        .unwrap()
                        .try_copy_cpu_to_cuda(Device::Cuda(0))
                        .unwrap();
                        let view = op.execute(&[&tensor], &[]).unwrap();
                        assert!(std::sync::Arc::ptr_eq(&tensor.storage, &view.storage));
                        assert_eq!(output.shape, view.shape());
                        assert_eq!(output.strides, view.stride());
                        assert_eq!(view.device(), tensor.device());
                    }
                }
            }
            for dim in [-rank - 1, rank, i64::MIN, i64::MAX] {
                assert!(
                    Operation::Transpose(0, dim, 0)
                        .layout(std::slice::from_ref(&input))
                        .is_err()
                );
                assert!(
                    Operation::Transpose(0, 0, dim)
                        .layout(std::slice::from_ref(&input))
                        .is_err()
                );
            }
        }
        if cuda::device_count() == 0 {
            eprintln!("skipping transpose CUDA storage assertions: no CUDA device");
        }
    }

    #[test]
    fn empty_alias_does_not_require_canonical_strides() {
        let dimension = 1_usize << (usize::BITS / 2);
        let input = Layout {
            shape: vec![0, dimension, dimension],
            strides: vec![dimension, dimension, 1],
            offset: 13,
        };
        assert!(validated_layout(&input.shape).is_err());
        assert!(input.is_contiguous().unwrap());
        let alias = Operation::Contiguous(0)
            .layout(std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(alias.shape, input.shape);
        assert_eq!(alias.strides, input.strides);
        assert_eq!(alias.offset, input.offset);
        // Materializing arithmetic must still reject the overflowing layout.
        assert!(Operation::Neg(0).layout(&[input]).is_err());
    }

    #[test]
    fn packing_and_arithmetic_plan_layouts_independently() {
        let mut values = vec![Layout {
            shape: vec![7, 3],
            strides: vec![1, 9],
            offset: 11,
        }];
        for op in [
            Operation::Neg(0),
            Operation::MulScalar(0, 2.),
            Operation::Add(0, 0),
            Operation::Matmul(0, 0),
            Operation::SumRows(0, false),
        ] {
            assert!(op.layout(&values).is_err());
        }
        let packed = Operation::Contiguous(0).layout(&values).unwrap();
        assert_eq!(packed.strides, [3, 1]);
        assert_eq!(packed.offset, 0);
        values.push(packed);
        assert!(Operation::Neg(1).layout(&values).is_ok());
        assert!(Operation::Contiguous(2).layout(&values).is_err());
        for (shape, strides) in [(vec![2, 3, 4], vec![4, 8, 1]), (vec![3], vec![0])] {
            assert!(
                Operation::Contiguous(0)
                    .layout(&[Layout {
                        shape,
                        strides,
                        offset: 0
                    }])
                    .is_err()
            );
        }
        for (shape, strides) in [
            (vec![], vec![]),
            (vec![0, 3], vec![1, 7]),
            (vec![1, 3, 1], vec![99, 1, 8]),
        ] {
            let input = Layout {
                shape: shape.clone(),
                strides: strides.clone(),
                offset: 13,
            };
            let alias = Operation::Contiguous(0).layout(&[input]).unwrap();
            assert_eq!(alias.shape, shape);
            assert_eq!(alias.strides, strides);
            assert_eq!(alias.offset, 13);
        }
    }
}
