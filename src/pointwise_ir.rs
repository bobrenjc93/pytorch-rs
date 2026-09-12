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
        let mut uses = vec![0; self.nodes.len()];
        uses[self.output] += 1;
        for node in &self.nodes {
            match *node {
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    uses[a] += 1;
                    uses[b] += 1;
                }
                Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a) => uses[a] += 1,
                Node::Input(_) | Node::Constant(_) => {}
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
                // Default Inductor lowers negation as 0 - x, including +0
                // for a positive-zero input (CUDA eager neg returns -0).
                Node::Neg(a) => format!("__fsub_rn(0.0f, v{a})"),
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
    // input shapes and program identity. Shared products remain SSA values.
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
        match (product(left), product(right)) {
            (Some((a, b)), None) => Some(format!(
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
    fn validates_ssa_before_codegen() {
        for nodes in [
            vec![],
            vec![Node::Input(2)],
            vec![Node::Neg(0)],
            vec![Node::Constant(0), Node::Sin(0)],
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
