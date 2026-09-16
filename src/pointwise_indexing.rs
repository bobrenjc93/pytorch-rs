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
    fn tensor_leaf_madd(&self, root: usize) -> bool {
        let tensor_product = |id| match self.nodes[id] {
            Node::Mul(a, b) => {
                matches!(self.nodes[a], Node::Input(_)) && matches!(self.nodes[b], Node::Input(_))
            }
            _ => false,
        };
        match self.nodes[root] {
            Node::Add(a, b) => {
                (tensor_product(a) && matches!(self.nodes[b], Node::Input(_)))
                    || (matches!(self.nodes[a], Node::Input(_)) && tensor_product(b))
            }
            _ => false,
        }
    }

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
        // The sole two-stage exception is an original tensor-leaf product plus
        // an input, in either add order. Match nodes directly: identities, CSE,
        // scalar leaves and wrappers must not turn a different graph into it.
        // A live trig consumer excludes that exception for the whole graph:
        // p=x*y; return (p+x, p.sin()) shares a product across arithmetic and
        // trig consumers. Both output orders must stay outside this increment.
        let unequal_shapes = shapes.windows(2).any(|pair| pair[0] != pair[1]);
        let live_transcendental = self.outputs.iter().any(|&root| transcendental[root]);
        for &root in &self.outputs {
            if unequal_shapes
                && depth[root] > 1
                && (live_transcendental || !self.tensor_leaf_madd(root))
            {
                return Err(invalid(
                    "unequal input shapes require at most one arithmetic stage, or a tensor-leaf multiply-add with no live sin/cos in any output",
                ));
            }
        }
        let first = self.outputs[0];
        if self
            .outputs
            .iter()
            .any(|&root| layouts[root].shape != layouts[first].shape)
        {
            return Err(invalid("computed outputs require the same actual shape"));
        }
        let live_inputs = self
            .outputs
            .iter()
            .fold(0, |mask, &root| mask | dependencies[root]);
        let output = layouts.swap_remove(first);
        let addresses = inputs
            .iter()
            .enumerate()
            .map(|(i, input)| {
                // Adding/removing singleton axes without expanding elements
                // still has a linear address.
                if input.shape == output.shape
                    || (live_inputs & (1 << i) != 0 && input.elements == output.elements)
                {
                    return Address::Linear;
                }
                let mut terms = Vec::new();
                if live_inputs & (1 << i) != 0 && output.elements != 0 {
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
            outputs: vec![2],
        }
    }

    #[test]
    fn multiple_roots_union_dependencies_and_preserve_original_limits() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Relu(0),
                Node::Add(0, 1),
            ],
            outputs: vec![2, 3],
        };
        let plan = graph.indexing(&[&[2, 3], &[1, 3]]).unwrap();
        assert_eq!(plan.output.shape, [2, 3]);
        assert_eq!(plan.addresses[0], Address::Linear);
        assert_eq!(plan.addresses[1], Address::Broadcast(vec![(1, 3, 1)]));
        // Equal element counts alone cannot supply the shared output layout.
        graph.nodes[3] = Node::Relu(1);
        assert!(graph.indexing(&[&[2, 3], &[6]]).is_err());
        graph.nodes[3] = Node::Sin(1);
        assert!(graph.indexing(&[&[2, 3], &[1, 3]]).is_err());
        assert!(graph.indexing(&[&[2, 3], &[2, 3]]).is_ok());
        // The second root cannot launder a forbidden original two-stage graph.
        graph.nodes.extend([Node::Neg(0), Node::Neg(4)]);
        graph.outputs = vec![2, 5];
        assert!(graph.indexing(&[&[2, 3], &[1, 3]]).is_err());
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
        graph.outputs = vec![3];
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
            g.outputs = vec![3];
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
            for second in [Node::Add(3, 2), Node::Sub(3, 3), Node::Neg(3)] {
                g.nodes.truncate(4);
                g.nodes.push(second);
                g.outputs = vec![4];
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
    fn relu_and_trig_preserve_depth_and_dead_nodes_still_validate() {
        let mut g = graph();
        g.nodes = vec![
            Node::Input(0),
            Node::Input(1),
            Node::Relu(0),
            Node::Relu(1),
            Node::Mul(2, 3),
            Node::Relu(4),
        ];
        g.outputs = vec![5];
        assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
        for op in [Node::Add(2, 3), Node::Sub(2, 3), Node::Neg(2)] {
            g.nodes[4] = op;
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
        }
        for op in [Node::Sin(0), Node::Cos(0)] {
            g.nodes[2] = op;
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
            assert!(g.indexing(&[&[2], &[2]]).is_ok());
            g.outputs = vec![3]; // All arithmetic and sin/cos are dead; ReLU(y) is live.
            assert!(g.indexing(&[&[2, 1], &[1, 3]]).is_ok());
            g.outputs = vec![5];
        }
        g.nodes[4] = Node::Add(0, 1);
        g.outputs = vec![3];
        assert!(
            g.indexing(&[&[2], &[3]])
                .err()
                .unwrap()
                .to_string()
                .contains("incompatible")
        );
    }

    #[test]
    fn trig_before_and_after_one_stage_accepts_all_scalar_kinds_and_input_orders() {
        for leaf in [
            Node::Input(1),
            Node::Constant(0f64.to_bits()),
            Node::Integer(0f64.to_bits()),
            Node::Boolean(false),
            Node::RuntimeScalar(0, true),
        ] {
            for op in [
                Node::Add(2, 1),
                Node::Sub(1, 2),
                Node::Mul(2, 1),
                Node::Neg(2),
            ] {
                let g = Graph {
                    inputs: 2,
                    nodes: vec![
                        Node::Input(0),
                        leaf.clone(),
                        Node::Sin(0),
                        op,
                        Node::Cos(3),
                        Node::Relu(4),
                        Node::Sin(5),
                    ],
                    outputs: vec![4, 6],
                };
                for (a, b) in [
                    (&[2, 1][..], &[1, 3][..]),
                    (&[2][..], &[1, 2][..]),
                    (&[][..], &[1][..]),
                    (&[0, 1][..], &[1, 3][..]),
                ] {
                    for shapes in [[a, b], [b, a]] {
                        assert!(g.indexing(&shapes).is_ok(), "{g:?} {shapes:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn live_trig_excludes_madd_graph_wide_in_both_output_orders() {
        for trig in [Node::Sin(2), Node::Cos(2), Node::Sin(0), Node::Cos(0)] {
            for reversed in [false, true] {
                let mut g = graph();
                // Graph roots are canonical SSA order. Vary producer order so
                // either root can be the first one visited by admission.
                g.nodes.extend(if reversed {
                    [trig.clone(), Node::Add(2, 0)]
                } else {
                    [Node::Add(2, 0), trig.clone()]
                });
                g.outputs = vec![3, 4];
                for shapes in [
                    [&[2, 3][..], &[1, 3][..]],
                    [&[2][..], &[1, 2][..]],
                    [&[][..], &[1][..]],
                    [&[0, 3][..], &[1, 3][..]],
                ] {
                    let error = g.indexing(&shapes).err().unwrap().to_string();
                    assert!(
                        error.contains("arithmetic stage"),
                        "{g:?} {shapes:?}: {error}"
                    );
                }
                assert!(g.indexing(&[&[2], &[2]]).is_ok());
                // Dead trig does not take away the existing MAdd exception.
                g.outputs = vec![if reversed { 4 } else { 3 }];
                assert!(g.indexing(&[&[2, 3], &[1, 3]]).is_ok());
            }
        }
    }

    #[test]
    fn tensor_leaf_madd_accepts_both_orders_all_input_ids_and_broadcasts() {
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    for reversed in [false, true] {
                        let g = Graph {
                            inputs: 2,
                            // Separate input nodes deliberately include repeated IDs.
                            nodes: vec![
                                Node::Input(a),
                                Node::Input(b),
                                Node::Input(c),
                                Node::Mul(0, 1),
                                if reversed {
                                    Node::Add(2, 3)
                                } else {
                                    Node::Add(3, 2)
                                },
                            ],
                            outputs: vec![4],
                        };
                        for (left, right) in [
                            (&[3, 1][..], &[2, 1, 5][..]),
                            (&[2, 1, 5][..], &[3, 1][..]),
                            (&[7][..], &[1, 7][..]),
                            (&[][..], &[1][..]),
                            (&[0, 1][..], &[1, 3][..]),
                        ] {
                            let indexing = g.indexing(&[left, right]).unwrap();
                            let code = g.indexed_source(&indexing.addresses).unwrap();
                            assert_eq!(code.matches("fmaf(").count(), 1);
                            assert!(!code.contains("/ 0ull"));
                            assert!(!code.contains("% 0ull"));
                            if a == b && b == c {
                                assert_eq!(indexing.output.shape, [left, right][a]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_madd_does_not_admit_wrappers_other_leaves_or_products() {
        let mut g = graph();
        g.nodes.push(Node::Add(2, 0));
        g.outputs = vec![3];
        let original = g.clone();
        let mut negatives = Vec::new();
        for wrapper in [
            Node::Relu(3),
            Node::Neg(3),
            Node::Sin(3),
            Node::Cos(3),
            Node::Add(3, 0),
            Node::Mul(3, 0),
            Node::Sub(3, 0),
        ] {
            let mut g = original.clone();
            g.nodes.push(wrapper);
            g.outputs = vec![4];
            negatives.push(g);
        }
        for leaf in [
            Node::Constant(1f64.to_bits()),
            Node::Integer(1f64.to_bits()),
            Node::Boolean(true),
            Node::RuntimeScalar(0, false),
            Node::Neg(1),
            Node::Relu(1),
            Node::Sin(1),
            Node::Cos(1),
        ] {
            for replacement in [2, 3] {
                let mut g = graph();
                g.nodes.insert(2, leaf.clone());
                g.nodes[3] = if replacement == 2 {
                    Node::Mul(0, 2)
                } else {
                    Node::Mul(0, 1)
                };
                g.nodes.push(if replacement == 2 {
                    Node::Add(3, 0)
                } else {
                    Node::Add(3, 2)
                });
                g.outputs = vec![4];
                negatives.push(g);
            }
        }
        for output in [Node::Add(2, 2), Node::Sub(2, 0), Node::Sub(0, 2)] {
            let mut g = original.clone();
            g.nodes[3] = output;
            negatives.push(g);
        }
        // A dead qualifying expression must not admit the actual output.
        let mut g = original;
        g.nodes.extend([Node::Neg(0), Node::Neg(4)]);
        g.outputs = vec![5];
        negatives.push(g);
        for g in negatives {
            for shapes in [
                [&[2, 1][..], &[1, 3][..]],
                [&[2][..], &[1, 2][..]],
                [&[][..], &[1][..]],
                [&[0][..], &[1, 0][..]],
            ] {
                assert!(g.indexing(&shapes).is_err(), "{g:?}");
            }
            assert!(g.indexing(&[&[2], &[2]]).is_ok(), "{g:?}");
        }
    }

    #[test]
    fn tensor_madd_still_checks_dead_nodes_unused_inputs_and_overflow() {
        let mut g = graph();
        g.nodes[2] = Node::Mul(0, 0);
        g.nodes.push(Node::Add(2, 0));
        g.outputs = vec![3];
        assert_eq!(g.indexing(&[&[3], &[5]]).unwrap().output.shape, [3]);
        for unused in [
            &[usize::MAX, 2][..],
            &[isize::MAX as usize][..],
            &[0, usize::MAX, 2][..],
        ] {
            assert!(g.indexing(&[&[3], unused]).is_err());
        }
        g.nodes.push(Node::Add(0, 1));
        assert!(g.indexing(&[&[3], &[5]]).is_err());
        let large = 1usize << (usize::BITS / 2);
        assert!(g.indexing(&[&[large, 1], &[1, large]]).is_err());
        g.nodes.push(Node::Add(0, 99));
        assert!(g.indexing(&[&[3], &[1]]).is_err());
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
