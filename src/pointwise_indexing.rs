//! Checked pointwise shapes and contiguous-input broadcast addresses. No CUDA access.
use super::{Graph, Node, invalid};
use crate::tensor_error::TensorError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Address {
    Linear,
    // Sum of (output_index / divisor % dimension) * input_stride.
    Broadcast(Vec<(usize, usize, usize)>),
}

impl Address {
    pub(super) fn source(&self) -> String {
        match self {
            Self::Linear => "i".into(),
            Self::Broadcast(terms) if terms.is_empty() => "0".into(),
            Self::Broadcast(terms) => terms
                .iter()
                .map(|(divisor, dimension, stride)| {
                    format!("((i / {divisor}ull) % {dimension}ull) * {stride}ull")
                })
                .collect::<Vec<_>>()
                .join(" + "),
        }
    }
}

pub(crate) struct Layout {
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) elements: usize,
}

impl Layout {
    pub(crate) fn new(shape: &[usize]) -> Result<Self, TensorError> {
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| invalid("broadcast element count overflow"))?;
        // CUDA storage and address arithmetic must fit the host pointer range.
        if elements > isize::MAX as usize / size_of::<f32>() {
            return Err(invalid("broadcast byte count overflow"));
        }
        let mut strides = vec![1; shape.len()];
        let mut stride = 1usize;
        for axis in (0..shape.len()).rev() {
            strides[axis] = stride;
            if axis > 0 {
                stride = stride
                    .checked_mul(shape[axis].max(1))
                    .filter(|&s| isize::try_from(s).is_ok())
                    .ok_or_else(|| invalid("broadcast stride overflow"))?;
            }
        }
        Ok(Self {
            shape: shape.to_vec(),
            strides,
            elements,
        })
    }
}

fn broadcast(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorError> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for (axis, dimension) in shape.iter_mut().rev().enumerate() {
        let a = left.len().checked_sub(axis + 1).map_or(1, |i| left[i]);
        let b = right.len().checked_sub(axis + 1).map_or(1, |i| right[i]);
        *dimension = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(invalid("incompatible pointwise broadcast shapes"));
        };
    }
    Ok(shape)
}

pub(crate) struct Indexing {
    pub(crate) output: Layout,
    pub(crate) addresses: Vec<Address>,
    pub(crate) input_elements: Vec<usize>,
}

impl Graph {
    pub(crate) fn indexing(&self, shapes: &[&[usize]]) -> Result<Indexing, TensorError> {
        self.validate()?;
        if shapes.len() != self.inputs {
            return Err(invalid("pointwise shape arity mismatch"));
        }
        let inputs = shapes
            .iter()
            .map(|s| Layout::new(s))
            .collect::<Result<Vec<_>, _>>()?;
        // Validate even dead expressions before liveness/numerical rewriting.
        // Dependency masks also distinguish unused parameters from broadcast loads.
        let mut layouts: Vec<Layout> = Vec::with_capacity(self.nodes.len());
        let mut dependencies = Vec::with_capacity(self.nodes.len());
        // Original-IR capability, before identities, CSE or sign rewriting.
        // Propagating only through operands makes the returned node authoritative:
        // dead expressions still validate their shapes but do not restrict numerics.
        let mut depth: Vec<usize> = Vec::with_capacity(self.nodes.len());
        let mut transcendental: Vec<bool> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let (shape, mask) = match *node {
                Node::Input(i) => (shapes[i].to_vec(), 1u8 << i),
                Node::Constant(_)
                | Node::Integer(_)
                | Node::Boolean(_)
                | Node::RuntimeScalar(_, _) => (vec![], 0),
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => (
                    broadcast(&layouts[a].shape, &layouts[b].shape)?,
                    dependencies[a] | dependencies[b],
                ),
                Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a) => {
                    (layouts[a].shape.clone(), dependencies[a])
                }
            };
            layouts.push(Layout::new(&shape)?);
            dependencies.push(mask);
            let (stages, has_transcendental) = match *node {
                Node::Input(_)
                | Node::Constant(_)
                | Node::Integer(_)
                | Node::Boolean(_)
                | Node::RuntimeScalar(_, _) => (0, false),
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => (
                    1 + depth[a].max(depth[b]),
                    transcendental[a] || transcendental[b],
                ),
                Node::Neg(a) => (1 + depth[a], transcendental[a]),
                Node::Relu(a) => (depth[a], transcendental[a]),
                Node::Sin(a) | Node::Cos(a) => (depth[a], true),
            };
            depth.push(stages);
            transcendental.push(has_transcendental);
        }
        // Argument shape equality, not address equality: unequal singleton-only
        // reshapes and unused arguments must not bypass this admission rule.
        // Graph validation bounds depth by the 4096-node limit.
        if shapes.windows(2).any(|pair| pair[0] != pair[1])
            && (depth[self.output] > 1 || transcendental[self.output])
        {
            return Err(invalid(
                "unequal input shapes require at most one arithmetic stage and no live sin/cos",
            ));
        }
        let output = layouts.swap_remove(self.output);
        let addresses = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                // Adding/removing singleton axes without expanding elements
                // still has a linear address.
                if input.shape == output.shape
                    || (dependencies[self.output] & (1 << i) != 0
                        && input.elements == output.elements)
                {
                    return Address::Linear;
                }
                let mut terms = Vec::new();
                if dependencies[self.output] & (1 << i) != 0 && output.elements != 0 {
                    let leading = output.shape.len() - input.shape.len();
                    for (axis, &dimension) in input.shape.iter().enumerate() {
                        if dimension != 1 {
                            terms.push((
                                output.strides[leading + axis],
                                dimension,
                                input.strides[axis],
                            ));
                        }
                    }
                }
                Address::Broadcast(terms)
            })
            .collect();
        Ok(Indexing {
            output,
            addresses,
            input_elements: inputs.iter().map(|x| x.elements).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> Graph {
        Graph {
            inputs: 2,
            nodes: vec![Node::Input(0), Node::Input(1), Node::Mul(0, 1)],
            output: 2,
        }
    }

    #[test]
    fn broadcasts_scalar_empty_leading_and_interior_dimensions() {
        for (left, right, output) in [
            (vec![], vec![], vec![]),
            (vec![], vec![2, 1, 3], vec![2, 1, 3]),
            (vec![3, 1], vec![2, 1, 5], vec![2, 3, 5]),
            (vec![1, 0, 3], vec![2, 1, 1], vec![2, 0, 3]),
        ] {
            for shapes in [
                [left.as_slice(), right.as_slice()],
                [right.as_slice(), left.as_slice()],
            ] {
                let plan = graph().indexing(&shapes).unwrap();
                assert_eq!(plan.output.shape, output);
                for (input, address) in shapes.iter().zip(&plan.addresses) {
                    let layout = Layout::new(input).unwrap();
                    for linear in 0..plan.output.elements {
                        // Independent coordinate walk with singleton projection.
                        let mut remainder = linear;
                        let mut expected = 0;
                        let mut input_stride = 1;
                        for axis in (0..output.len()).rev() {
                            let coordinate = remainder % output[axis];
                            remainder /= output[axis];
                            if let Some(input_axis) = axis.checked_sub(output.len() - input.len()) {
                                if input[input_axis] != 1 {
                                    expected += coordinate * input_stride;
                                }
                                input_stride *= input[input_axis];
                            }
                        }
                        let actual = match address {
                            Address::Linear => linear,
                            Address::Broadcast(terms) => {
                                terms.iter().map(|&(d, n, s)| (linear / d % n) * s).sum()
                            }
                        };
                        assert_eq!(actual, expected);
                        assert!(actual < layout.elements);
                    }
                }
            }
        }
    }

    #[test]
    fn validates_dead_shapes_but_does_not_expand_unrelated_output() {
        let mut graph = graph();
        graph.nodes.push(Node::Neg(0));
        graph.output = 3;
        assert_eq!(
            graph.indexing(&[&[3, 1], &[2, 1, 5]]).unwrap().output.shape,
            [3, 1]
        );
        assert!(graph.indexing(&[&[3], &[5]]).is_err());
        graph.nodes[2] = Node::Neg(1);
        assert!(
            graph
                .indexing(&[&[], &[2, 3, 4]])
                .unwrap()
                .output
                .shape
                .is_empty()
        );
        assert!(graph.indexing(&[&[]]).is_err());
    }

    #[test]
    fn original_depth_rejects_live_stages_before_numeric_simplification() {
        for scalar in [
            Node::Constant(0f64.to_bits()),
            Node::Integer(1f64.to_bits()),
            Node::Boolean(true),
            Node::RuntimeScalar(0, true),
        ] {
            let mut g = graph();
            g.nodes = vec![Node::Input(0), Node::Input(1), scalar, Node::Mul(0, 2)];
            g.output = 3;
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
            for second in [Node::Add(3, 2), Node::Sub(3, 3), Node::Neg(3)] {
                g.nodes.truncate(4);
                g.nodes.push(second);
                g.output = 4;
                // Includes unused input 1, linear singleton reshapes, and empties.
                for shapes in [
                    (&[2, 1][..], &[1, 3][..]),
                    (&[2][..], &[1, 2][..]),
                    (&[0][..], &[1, 0][..]),
                ] {
                    assert!(g.indexing(&[shapes.0, shapes.1]).is_err());
                }
                assert!(g.indexing(&[&[2], &[2]]).is_ok());
            }
        }
    }

    #[test]
    fn relu_preserves_depth_and_only_live_transcendentals_restrict_admission() {
        let mut g = graph();
        g.nodes = vec![
            Node::Input(0),
            Node::Input(1),
            Node::Relu(0),
            Node::Relu(1),
            Node::Mul(2, 3),
            Node::Relu(4),
        ];
        g.output = 5;
        assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
        for op in [Node::Add(2, 3), Node::Sub(2, 3), Node::Neg(2)] {
            g.nodes[4] = op;
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
        }
        for op in [Node::Sin(0), Node::Cos(0)] {
            g.nodes[2] = op;
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_err());
            assert!(g.indexing(&[&[2], &[2]]).is_ok());
            g.output = 3; // All arithmetic and sin/cos are dead; ReLU(y) is live.
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
            g.output = 5;
        }
        g.nodes[4] = Node::Add(0, 1);
        g.output = 3;
        assert!(
            g.indexing(&[&[2], &[3]])
                .err()
                .unwrap()
                .to_string()
                .contains("incompatible")
        );
    }

    #[test]
    fn rejects_output_size_and_empty_stride_overflow_before_codegen() {
        assert!(Layout::new(&[usize::MAX, 2]).is_err());
        assert!(Layout::new(&[isize::MAX as usize]).is_err());
        assert!(Layout::new(&[0, usize::MAX, 2]).is_err());
        let large = 1usize << (usize::BITS / 2);
        assert!(graph().indexing(&[&[large, 1], &[1, large]]).is_err());
        assert!(graph().indexing(&[&[0], &[2]]).is_err());
        assert_eq!(graph().indexing(&[&[0], &[1]]).unwrap().output.elements, 0);
    }
}
