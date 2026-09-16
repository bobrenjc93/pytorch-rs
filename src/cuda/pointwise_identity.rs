//! Versioned exact equality for an executable, independent of preparation shape.
use crate::pointwise_ir::{Graph, Node, indexing::Address, program::Program};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExecutableIdentity {
    pub(crate) bytes: Vec<u8>,
    pub(crate) direct: bool,
    pub(crate) context: usize,
}

impl ExecutableIdentity {
    pub(crate) fn new(
        graph: &Graph,
        addresses: &[Address],
        device: usize,
        context: usize,
        direct: Option<&Program>,
    ) -> Self {
        // v1 fixes the launch ABI, opcode semantics, precise compiler options,
        // float32 data and u64 indexing. Length prefixes make equality exact.
        let mut bytes = b"torch_rs.pointwise.executable.v1\0".to_vec();
        let mut word = |value: u64| bytes.extend_from_slice(&value.to_le_bytes());
        word(device as u64);
        word(context as u64);
        word(graph.inputs as u64);
        word(graph.scalar_count() as u64);
        word(graph.nodes.len() as u64);
        for node in &graph.nodes {
            let (tag, a, b, bits) = match *node {
                Node::Input(a) => (0, a, 0, 0),
                Node::RuntimeScalar(a, sign) => (1, a, usize::from(sign), 0),
                Node::Constant(bits) => (2, 0, 0, bits),
                Node::Boolean(value) => (3, 0, 0, u64::from(value)),
                Node::Integer(bits) => (4, 0, 0, bits),
                Node::Add(a, b) => (5, a, b, 0),
                Node::Sub(a, b) => (6, a, b, 0),
                Node::Mul(a, b) => (7, a, b, 0),
                Node::Neg(a) => (8, a, 0, 0),
                Node::Relu(a) => (9, a, 0, 0),
                Node::Sin(a) => (10, a, 0, 0),
                Node::Cos(a) => (11, a, 0, 0),
                Node::Erf(a) => (12, a, 0, 0),
            };
            for value in [tag, a as u64, b as u64, bits] {
                word(value);
            }
        }
        word(graph.outputs.len() as u64);
        for &output in &graph.outputs {
            word(output as u64);
        }
        word(addresses.len() as u64);
        for address in addresses {
            match address {
                Address::Linear => word(0),
                Address::Broadcast(terms) => {
                    word(1);
                    word(terms.len() as u64);
                    for &(divisor, dimension, stride) in terms {
                        for value in [divisor, dimension, stride] {
                            word(value as u64);
                        }
                    }
                }
            }
        }
        word(u64::from(direct.is_some()));
        if let Some(program) = direct {
            word(program.register_count() as u64);
            word(program.instruction_count() as u64);
            for instruction in program.instructions() {
                for &value in instruction {
                    word(u64::from(value));
                }
            }
        }
        Self {
            bytes,
            direct: direct.is_some(),
            context,
        }
    }
}
