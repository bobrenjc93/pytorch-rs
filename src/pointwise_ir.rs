//! Typed, shape-independent pointwise SSA and CUDA C lowering. No device access.
use crate::tensor_error::TensorError;
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Node {
    Input(usize),
    Constant(u32),
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
                Node::Constant(_) => false,
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
        // Only observable consumers constrain contraction. Validate the entire
        // graph first, then remove dead expressions (including unused locals).
        // Negated products consume their factors directly in the emitted FMA.
        let operands = |node: &Node| match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => Some((a, Some(b))),
            Node::Neg(a) => match self.nodes[a] {
                Node::Mul(left, right) => Some((left, Some(right))),
                _ => Some((a, None)),
            },
            Node::Relu(a) | Node::Sin(a) | Node::Cos(a) => Some((a, None)),
            Node::Input(_) | Node::Constant(_) => None,
        };
        let mut live = vec![false; self.nodes.len()];
        live[self.output] = true;
        for index in (0..self.nodes.len()).rev() {
            if live[index]
                && let Some((a, b)) = operands(&self.nodes[index])
            {
                live[a] = true;
                if let Some(b) = b {
                    live[b] = true;
                }
            }
        }
        let mut uses = vec![0; self.nodes.len()];
        uses[self.output] += 1;
        for (index, node) in self.nodes.iter().enumerate() {
            if live[index]
                && let Some((a, b)) = operands(node)
            {
                uses[a] += 1;
                if let Some(b) = b {
                    uses[b] += 1;
                }
            }
        }
        let mut source = String::from(
            "// torch_rs typed pointwise SSA v1; float32, no fast math\n\
             extern \"C\" __global__ void torch_rs_pointwise(\n\
             const float* x0, const float* x1, float* out, unsigned long long n) {\n\
             for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
             i < n; i += (unsigned long long)blockDim.x * gridDim.x) {\n",
        );
        for (index, node) in self.nodes.iter().enumerate() {
            if !live[index] {
                continue;
            }
            let expression = match *node {
                Node::Input(id) => format!("x{id}[i]"),
                Node::Constant(bits) => format!("__uint_as_float(0x{bits:08x}u)"),
                Node::Add(a, b) => self
                    .contract(a, b, false, &uses)
                    .unwrap_or_else(|| format!("(v{a} + v{b})")),
                Node::Sub(a, b) => self
                    .contract(a, b, true, &uses)
                    .unwrap_or_else(|| format!("(v{a} - v{b})")),
                Node::Mul(a, b) => format!("(v{a} * v{b})"),
                // Inductor lowers negation as 0 - x and contracts products into
                // that subtraction. FMA preserves negative underflowed zero,
                // while an exactly zero product still yields positive zero.
                Node::Neg(a) => match self.nodes[a] {
                    Node::Mul(left, right) => format!("fmaf(-v{left}, v{right}, 0.0f)"),
                    _ => format!("__fsub_rn(0.0f, v{a})"),
                },
                Node::Relu(a) => format!("(v{a} < 0.0f ? 0.0f : v{a})"),
                Node::Sin(a) => format!("sinf(v{a})"),
                Node::Cos(a) => format!("cosf(v{a})"),
            };
            writeln!(source, "const float v{index} = {expression};").unwrap();
        }
        writeln!(source, "out[i] = v{};\n}}\n}}", self.output).unwrap();
        Ok(source)
    }

    // Contract a single-use product explicitly. Otherwise NVRTC can strength-
    // reduce multiplication into addition before its FMA pass, changing overflow
    // compared with Inductor. This is a local SSA rule, independent of constants,
    // input shapes and program identity. If both operands are products, follow
    // Inductor's left-first contraction: the right product is rounded first.
    // Addition of a negative-coefficient product is normalized as subtraction
    // from the positive product before selecting that first operand.
    // Products with remaining shared uses remain SSA values.
    fn contract(
        &self,
        left: usize,
        right: usize,
        subtract: bool,
        uses: &[usize],
    ) -> Option<String> {
        let product = |id| match self.nodes[id] {
            Node::Mul(a, b) if uses[id] == 1 => Some((a, b)),
            _ => None,
        };
        let negative_coefficient = |a, b| {
            [a, b].iter().any(|&id| {
                matches!(self.nodes[id],
                Node::Constant(bits) if f32::from_bits(bits) < 0.0)
            })
        };
        match (product(left), product(right)) {
            (Some((a, b)), Some((c, d)))
                if !subtract && negative_coefficient(a, b) && !negative_coefficient(c, d) =>
            {
                Some(format!("fmaf(v{c}, v{d}, v{left})"))
            }
            (Some((a, b)), _) => Some(format!(
                "fmaf(v{a}, v{b}, {}v{right})",
                if subtract { "-" } else { "" }
            )),
            (None, Some((a, b))) => Some(format!(
                "fmaf({}v{a}, v{b}, v{left})",
                if subtract { "-" } else { "" }
            )),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contracts_left_product_when_both_operands_are_products() {
        let mut graph = Graph {
            inputs: 2,
            nodes: vec![
                Node::Input(0),
                Node::Input(1),
                Node::Constant(2.0_f32.to_bits()),
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
                Node::Constant(2.0_f32.to_bits()),
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
                Node::Constant(0x8000_0000),
                Node::Sub(2, 3),
            ],
            output: 4,
        };
        let code = graph.source().unwrap();
        assert_eq!(code.matches("sinf(").count(), 1);
        assert!(code.contains("(v1 * v1)"));
        assert!(code.contains("0x80000000u"));
        assert!(!code.contains("__sinf"));
    }
}
