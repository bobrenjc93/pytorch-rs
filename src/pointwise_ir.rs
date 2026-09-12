//! Typed, shape-independent pointwise SSA and CUDA C lowering. No device access.
use crate::tensor_error::TensorError;
#[path = "pointwise_lowering.rs"]
mod lowering;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Node {
    Input(usize),
    Constant(u64),
    Boolean(bool),
    // Python integer normalized to binary64, with its scalar kind retained.
    Integer(u64),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Neg(usize),
    Relu(usize),
    Sin(usize),
    Cos(usize),
}

#[derive(Clone, Debug)]
pub(crate) struct Graph {
    pub(crate) inputs: usize,
    pub(crate) nodes: Vec<Node>,
    pub(crate) output: usize,
}

pub(crate) fn invalid(message: impl Into<String>) -> TensorError {
    TensorError::CudaRuntimeError {
        operation: "pointwise JIT",
        message: message.into(),
    }
}

impl Graph {
    pub(crate) fn validate(&self) -> Result<(), TensorError> {
        if !(1..=2).contains(&self.inputs) || self.nodes.len() > 4096 {
            return Err(invalid("expected one or two inputs and at most 4096 nodes"));
        }
        let mut tensor: Vec<bool> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let operand = |id: usize| {
                tensor
                    .get(id)
                    .copied()
                    .filter(|_| id < index)
                    .ok_or_else(|| invalid("operand must refer to an earlier SSA node"))
            };
            tensor.push(match *node {
                Node::Input(id) if id < self.inputs => true,
                Node::Input(_) => return Err(invalid("invalid input index")),
                Node::Constant(_) | Node::Boolean(_) | Node::Integer(_) => false,
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    let (a, b) = (operand(a)?, operand(b)?);
                    if !a && !b {
                        return Err(invalid("scalar-only arithmetic is not tensor arithmetic"));
                    }
                    true
                }
                Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a) => {
                    if !operand(a)? {
                        return Err(invalid("unary operations require a tensor expression"));
                    }
                    true
                }
            });
        }
        if tensor.get(self.output) != Some(&true)
            || matches!(self.nodes.get(self.output), Some(Node::Input(_)))
        {
            return Err(invalid("output must be a computed tensor expression"));
        }
        Ok(())
    }

    pub(crate) fn source(&self) -> Result<String, TensorError> {
        self.validate()?;
        Ok(lowering::source(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constant_tensors_fold_before_float32_materialization() {
        for (left, increment, expected) in [
            (16_777_216.0_f64, 1.0_f64, 1.0_f32),
            (16_777_217.0, 0.0, 1.0),
        ] {
            let graph = Graph {
                inputs: 1,
                nodes: vec![
                    Node::Input(0),
                    Node::Integer(0),
                    Node::Mul(0, 1),
                    Node::Constant(left.to_bits()),
                    Node::Add(2, 3),
                    Node::Constant(increment.to_bits()),
                    Node::Add(4, 5),
                    Node::Constant(16_777_216.0_f64.to_bits()),
                    Node::Sub(6, 7),
                ],
                output: 8,
            };
            let source = graph.source().unwrap();
            assert!(source.contains(&format!("0x{:08x}u", expected.to_bits())));
            assert!(!source.contains("x0[i]"));
            assert!(!source.contains("__fadd_rn"));
        }
    }

    #[test]
    fn computed_zero_does_not_acquire_early_zero_identity() {
        let mut graph = Graph {
            inputs: 1,
            nodes: vec![
                Node::Input(0),
                Node::Integer(0),
                Node::Mul(0, 1),
                Node::Constant(0),
                Node::Add(2, 3),
                Node::Add(0, 4),
            ],
            output: 5,
        };
        assert!(graph.source().unwrap().contains("__fadd_rn"));
        graph.nodes[5] = Node::Add(0, 2);
        let source = graph.source().unwrap();
        assert!(source.contains("x0[i]"));
        assert!(!source.contains("__fadd_rn"));
    }

    #[test]
    fn subtraction_aliases_preserve_contraction_after_addition_pass() {
        let graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Integer(0),
                Node::Mul(0, 2),
                Node::Sub(3, 3),
                Node::Constant(2.0_f64.to_bits()),
                Node::Mul(0, 5),
                Node::Sub(6, 4),
                Node::Mul(1, 5),
                Node::Sub(7, 8),
            ],
            output: 9,
        };
        let source = graph.source().unwrap();
        assert_eq!(source.matches("fmaf(").count(), 1);
        assert!(!source.contains("__fsub_rn"));
    }

    #[test]
    fn boolean_multiplication_is_distinct_from_float_zero() {
        let mut graph = Graph {
            inputs: 1,
            nodes: vec![Node::Input(0), Node::Boolean(false), Node::Mul(0, 1)],
            output: 2,
        };
        let source = graph.source().unwrap();
        assert!(source.contains("0x00000000u"));
        assert!(!source.contains("x0[i]"));
        graph.nodes[1] = Node::Integer(0);
        assert_eq!(graph.source().unwrap(), source);
        graph.nodes[1] = Node::Constant(0);
        assert!(graph.source().unwrap().contains("__fmul_rn("));
        graph.nodes[1] = Node::Boolean(true);
        let source = graph.source().unwrap();
        assert!(source.contains("x0[i]"));
        assert!(!source.contains("__fmul_rn("));
    }

    #[test]
    fn constant_tensor_arithmetic_retains_signed_zero_for_negation() {
        for scalar in [Node::Integer(0), Node::Boolean(false)] {
            for operation in [Node::Add(2, 3), Node::Mul(2, 4), Node::Sub(2, 2)] {
                let graph = Graph {
                    inputs: 1,
                    nodes: vec![
                        Node::Input(0),
                        scalar.clone(),
                        Node::Mul(0, 1),
                        Node::Constant(0),
                        Node::Constant(2.0_f64.to_bits()),
                        operation,
                        Node::Neg(5),
                    ],
                    output: 6,
                };
                let source = graph.source().unwrap();
                assert!(source.contains("0x80000000u"));
                assert!(!source.contains("x0[i]"));
                assert!(!source.contains("fmaf("));
                assert!(!source.contains("__fsub_rn("));
            }
        }
    }

    #[test]
    fn sign_rewrites_do_not_acquire_shared_subtraction_semantics() {
        let mut graph = Graph {
            inputs: 1,
            nodes: vec![
                Node::Input(0),
                Node::Constant(2.0_f64.to_bits()),
                Node::Mul(0, 1),
                Node::Constant((-2.0_f64).to_bits()),
                Node::Mul(0, 3),
                Node::Add(2, 4),
            ],
            output: 5,
        };
        assert!(graph.source().unwrap().contains("fmaf("));
        graph.nodes[5] = Node::Sub(2, 2);
        let source = graph.source().unwrap();
        assert!(!source.contains("fmaf("));
        assert!(source.contains("__fsub_rn(v2, v2)"));
    }

    #[test]
    fn contracts_left_product_when_both_operands_are_products() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Constant(2.0_f64.to_bits()),
                Node::Mul(0, 2),
                Node::Mul(1, 2),
                Node::Sub(3, 4),
            ],
            output: 5,
        };
        assert!(graph.source().unwrap().contains("fmaf(v0, v2, -v4)"));
        graph.nodes[5] = Node::Add(3, 4);
        assert!(graph.source().unwrap().contains("fmaf(v0, v2, v4)"));
    }

    #[test]
    fn dead_consumers_do_not_prevent_live_contraction_or_bypass_validation() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Constant(2.0_f64.to_bits()),
                Node::Mul(0, 2),
                Node::Mul(1, 2),
                Node::Add(3, 3),
                Node::Sub(3, 4),
            ],
            output: 6,
        };
        let source = graph.source().unwrap();
        assert!(source.contains("fmaf(v0, v2, -v4)"));
        assert!(!source.contains("const float v5"));
        graph.nodes[5] = Node::Sin(2); // Invalid even though not returned.
        assert!(graph.source().is_err());
    }

    #[test]
    fn negation_contracts_products_but_retains_isolated_zero_semantics() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, 1),
                Node::Neg(2),
            ],
            output: 3,
        };
        assert!(graph.source().unwrap().contains("fmaf(-v0, v1, 0.0f)"));
        graph.nodes.push(Node::Add(3, 2));
        graph.output = 4;
        // The negation consumes factors directly; the remaining product use
        // can contract without losing the separately rounded negation result.
        assert!(graph.source().unwrap().contains("fmaf(v0, v1, v3)"));
        graph.nodes[3] = Node::Neg(0);
        assert!(graph.source().unwrap().contains("__fsub_rn(0.0f, v0)"));
    }

    #[test]
    fn validates_ssa_before_codegen() {
        for nodes in [
            vec![],
            vec![Node::Input(2)],
            vec![Node::Neg(0)],
            vec![Node::Constant(0), Node::Sin(0)],
            vec![Node::Constant(0), Node::Cos(0)],
            vec![Node::Constant(0), Node::Relu(0)],
            vec![Node::Constant(0), Node::Add(0, 0)],
        ] {
            assert!(
                Graph {
                    inputs: 1,
                    nodes,
                    output: 0
                }
                .source()
                .is_err()
            );
        }
        let graph = Graph {
            inputs: 1,
            nodes: vec![
                Node::Input(0),
                Node::Sin(0),
                Node::Mul(1, 1),
                Node::Constant(0x8000_0000_0000_0000),
                Node::Sub(2, 3),
            ],
            output: 4,
        };
        let code = graph.source().unwrap();
        assert_eq!(code.matches("sinf(").count(), 1);
        assert!(code.contains("__fmul_rn(v1, v1)"));
        assert!(code.contains("0x80000000u"));
        assert!(!code.contains("__sinf"));
    }

    #[test]
    fn equivalent_products_share_rounding_without_commuting_operands() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Mul(0, 1),
                Node::Mul(0, 1),
                Node::Sub(2, 3),
            ],
            output: 4,
        };
        let code = graph.source().unwrap();
        assert_eq!(code.matches("__fmul_rn(").count(), 1);
        assert!(code.contains("__fsub_rn(v2, v2)"));
        assert!(!code.contains("fmaf("));
        graph.nodes[3] = Node::Mul(1, 0);
        assert!(graph.source().unwrap().contains("fmaf("));
    }

    #[test]
    fn shared_products_can_contract_for_each_consumer() {
        let graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Constant(2.0_f64.to_bits()),
                Node::Mul(0, 2),
                Node::Sub(3, 1),
                Node::Add(4, 3),
            ],
            output: 5,
        };
        let code = graph.source().unwrap();
        assert!(code.contains("fmaf(v0, v2, -v1)"));
        assert!(code.contains("fmaf(v0, v2, v4)"));
    }

    #[test]
    fn negated_subtraction_reverses_contraction_before_rounding_zero() {
        let graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Constant(2.0_f64.to_bits()),
                Node::Mul(0, 2),
                Node::Mul(1, 2),
                Node::Sub(3, 4),
                Node::Neg(5),
            ],
            output: 6,
        };
        let code = graph.source().unwrap();
        assert!(code.contains("fmaf(v1, v2, -v3)"));
        assert!(code.contains("__fadd_rn("));
        assert!(!code.contains("fmaf(v0, v2, -v4)"));
    }
}
