//! Numerical lowering after whole-graph admission. Hash-consing precedes sign
//! normalization and per-consumer FMA selection; unused nodes are never emitted.
use super::{Graph, Node};
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Expr {
    Node(Node),
    // Equality established before sign rewriting requires one rounded value.
    // A subtraction introduced by normalization must not acquire this rule.
    SelfSub(usize),
    // An early integer/Boolean multiplication rewrite, eligible for tensor
    // zero identities. Later computed zeros must never acquire this provenance.
    Zero,
    // A constant tensor propagated in binary64, materialized as float32 at
    // runtime consumers or output. Shared constant consumers retain precision.
    Folded(u64),
    // Exact sign-bit inversion, unlike the language's positive-zero subtraction.
    Flip(usize),
}

#[derive(Default)]
struct Lowering {
    nodes: Vec<Expr>,
    interned: HashMap<Expr, usize>,
}

impl Lowering {
    fn intern(&mut self, expr: Expr) -> usize {
        if let Some(&id) = self.interned.get(&expr) {
            return id;
        }
        let id = self.nodes.len();
        self.interned.insert(expr.clone(), id);
        self.nodes.push(expr);
        id
    }

    fn node(&mut self, node: Node) -> usize {
        self.intern(Expr::Node(node))
    }

    fn constant(&self, id: usize) -> Option<f64> {
        match self.nodes[id] {
            Expr::Node(Node::Constant(bits) | Node::Integer(bits)) => Some(f64::from_bits(bits)),
            Expr::Node(Node::Boolean(value)) => Some(f64::from(u8::from(value))),
            _ => None,
        }
    }

    fn product(&self, id: usize) -> Option<(usize, usize)> {
        match self.nodes[id] {
            Expr::Node(Node::Mul(a, b)) => Some((a, b)),
            _ => None,
        }
    }

    fn float_operand(&mut self, id: usize) -> usize {
        // Runtime scalar uses are separately interned at float32 precision;
        // never replace the unrounded value used by another constant consumer.
        if let Some(value) = self.constant(id) {
            let rounded = f64::from(f32::from_bits(float32_bits(value.to_bits())));
            self.node(Node::Constant(rounded.to_bits()))
        } else {
            id
        }
    }

    fn float_operands(&mut self, node: Node) -> Node {
        match node {
            Node::Add(a, b) => Node::Add(self.float_operand(a), self.float_operand(b)),
            Node::Sub(a, b) => Node::Sub(self.float_operand(a), self.float_operand(b)),
            Node::Mul(a, b) => Node::Mul(self.float_operand(a), self.float_operand(b)),
            other => other,
        }
    }

    fn fold_tensor_arithmetic(&self, node: &Node) -> Option<u64> {
        let (Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) = *node else {
            return None;
        };
        if !matches!(self.nodes[a], Expr::Zero | Expr::Folded(_))
            && !matches!(self.nodes[b], Expr::Zero | Expr::Folded(_))
        {
            return None;
        }
        let value = |id| match self.nodes[id] {
            Expr::Zero => Some(0.0),
            Expr::Folded(bits) => Some(f64::from_bits(bits)),
            _ => self.constant(id),
        };
        let (left, right) = (value(a)?, value(b)?);
        Some(match node {
            Node::Add(_, _) => (left + right).to_bits(),
            Node::Sub(_, _) => (left - right).to_bits(),
            Node::Mul(_, _) => (left * right).to_bits(),
            _ => unreachable!(),
        })
    }

    // Extract a negative product's sign without changing operand order. A sign
    // extracted from an add becomes subtraction before choosing its FMA side.
    fn positive(&mut self, id: usize) -> Option<usize> {
        if let Expr::Flip(a) = self.nodes[id] {
            return Some(a);
        }
        let (mut a, mut b) = self.product(id)?;
        let coefficient = if self.constant(a).is_some_and(|x| x < 0.0) {
            &mut a
        } else if self.constant(b).is_some_and(|x| x < 0.0) {
            &mut b
        } else {
            return None;
        };
        let bits = self.constant(*coefficient).unwrap().to_bits() ^ 0x8000_0000_0000_0000;
        *coefficient = self.node(Node::Constant(bits));
        Some(self.node(Node::Mul(a, b)))
    }

    fn normalize(&mut self, node: Node) -> usize {
        // Keep arithmetic on known constant tensors distinct from scalar
        // nodes and runtime expressions. In particular, later negation must
        // invert the constant's sign, not emit positive-zero subtraction or
        // contract a multiplication with an added positive zero. Non-arithmetic
        // operations still use their native libdevice/intrinsic lowering.
        if let Some(bits) = self.fold_tensor_arithmetic(&node) {
            return self.intern(Expr::Folded(bits));
        }
        let node = self.float_operands(node);
        match node {
            // Only integer zero needs a different arithmetic rule after scalar
            // conversion. Other integers share the converted float expression.
            Node::Integer(bits) if bits != 0 => return self.node(Node::Constant(bits)),
            Node::Mul(a, b) => {
                let coefficient = self
                    .constant(b)
                    .map(|c| (a, b, c, false))
                    .or_else(|| self.constant(a).map(|c| (b, a, c, true)));
                if let Some((value, _, scalar, reversed)) = coefficient {
                    if scalar.to_bits() == 1.0_f64.to_bits() {
                        return value;
                    }
                    if scalar.to_bits() == (-1.0_f64).to_bits() {
                        return self.intern(Expr::Flip(value));
                    }
                    // Keep signed doubling's preferred rounded form, but retain
                    // its factors for consumers without another FMA candidate.
                    if scalar.to_bits() == (-2.0_f64).to_bits() {
                        let two = self.node(Node::Constant(2.0_f64.to_bits()));
                        let product = self.node(if reversed {
                            Node::Mul(two, value)
                        } else {
                            Node::Mul(value, two)
                        });
                        return self.intern(Expr::Flip(product));
                    }
                }
            }
            Node::Sub(a, b) if a == b => return self.intern(Expr::SelfSub(a)),
            Node::Add(a, b) => {
                if let Some(positive) = self.positive(b) {
                    return self.node(Node::Sub(a, positive));
                }
                if let Some(positive) = self.positive(a) {
                    return self.node(Node::Sub(b, positive));
                }
            }
            Node::Sub(a, b) if a != b => {
                if let Some(positive) = self.positive(b) {
                    // Preserve this add's operand orientation. Reapplying the
                    // add sign rewrite would choose a different rounded product.
                    return self.node(Node::Add(a, positive));
                }
            }
            Node::Neg(a) => {
                match self.nodes[a] {
                    Expr::Zero => return self.intern(Expr::Folded((-0.0_f64).to_bits())),
                    Expr::Folded(bits) => {
                        return self.intern(Expr::Folded(bits ^ 0x8000_0000_0000_0000));
                    }
                    _ => {}
                }
                let value = match self.nodes[a] {
                    Expr::Node(Node::Sub(left, right)) if left != right => {
                        Some(self.node(Node::Sub(right, left)))
                    }
                    Expr::Node(Node::Neg(value)) | Expr::Flip(value) => Some(value),
                    Expr::Node(_) | Expr::SelfSub(_) | Expr::Zero | Expr::Folded(_) => None,
                };
                if let Some(value) = value {
                    // 0-(a-b) = (b-a)+0, including IEEE zero signs.
                    let zero = self.node(Node::Constant(0));
                    return self.node(Node::Add(value, zero));
                }
            }
            _ => {}
        }
        self.node(node)
    }

    fn contract(&self, a: usize, b: usize, subtract: bool) -> Option<String> {
        // Prefer direct products over a sign-flipped rounded product. If no
        // direct product exists, the flipped product can still contract. This
        // keeps signed doubling from imposing rounding on every consumer.
        if let Some((x, y)) = self.product(a) {
            return Some(format!(
                "fmaf(v{x}, v{y}, {}v{b})",
                if subtract { "-" } else { "" }
            ));
        }
        if let Some((x, y)) = self.product(b) {
            return Some(format!(
                "fmaf({}v{x}, v{y}, v{a})",
                if subtract { "-" } else { "" }
            ));
        }
        if let Expr::Flip(product) = self.nodes[a]
            && let Some((x, y)) = self.product(product)
        {
            return Some(format!(
                "fmaf(-v{x}, v{y}, {}v{b})",
                if subtract { "-" } else { "" }
            ));
        }
        if let Expr::Flip(product) = self.nodes[b]
            && let Some((x, y)) = self.product(product)
        {
            return Some(format!(
                "fmaf({}v{x}, v{y}, v{a})",
                if subtract { "" } else { "-" }
            ));
        }
        None
    }

    fn operands(&self, id: usize) -> Vec<usize> {
        match self.nodes[id] {
            Expr::Node(Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) => vec![a, b],
            Expr::Node(Node::Neg(a)) => {
                self.product(a).map_or_else(|| vec![a], |(x, y)| vec![x, y])
            }
            Expr::Node(Node::Relu(a) | Node::Sin(a) | Node::Cos(a))
            | Expr::Flip(a)
            | Expr::SelfSub(a) => vec![a],
            Expr::Node(
                Node::Input(_) | Node::Constant(_) | Node::Boolean(_) | Node::Integer(_),
            )
            | Expr::Zero
            | Expr::Folded(_) => vec![],
        }
    }
}

// Reference graph identities run by operator phase, before numeric lowering.
// Resolving every alias here avoids exposing a subtraction's zero to an
// already-completed addition pass, while later subtractions can still use it.
fn early_aliases(graph: &Graph) -> (Vec<usize>, Vec<bool>) {
    fn resolve(aliases: &[usize], mut id: usize) -> usize {
        while aliases[id] != id {
            id = aliases[id];
        }
        id
    }
    let scalar_zero = |id| matches!(graph.nodes[id], Node::Integer(0) | Node::Boolean(false));
    let tensor = |id| {
        !matches!(
            graph.nodes[id],
            Node::Constant(_) | Node::Integer(_) | Node::Boolean(_)
        )
    };
    let zeros: Vec<_> = graph
        .nodes
        .iter()
        .map(|node| matches!(*node, Node::Mul(a, b) if scalar_zero(a) || scalar_zero(b)))
        .collect();
    let mut aliases: Vec<_> = (0..graph.nodes.len()).collect();
    for subtract in [false, true] {
        for (id, node) in graph.nodes.iter().enumerate() {
            let (a, b) = match *node {
                Node::Add(a, b) if !subtract => (a, b),
                Node::Sub(a, b) if subtract => (a, b),
                _ => continue,
            };
            let (a, b) = (resolve(&aliases, a), resolve(&aliases, b));
            if tensor(a) && tensor(b) {
                if !subtract && zeros[a] {
                    aliases[id] = b;
                } else if zeros[b] {
                    aliases[id] = a;
                }
            }
        }
    }
    for id in 0..aliases.len() {
        aliases[id] = resolve(&aliases, id);
    }
    (aliases, zeros)
}

#[allow(clippy::cast_possible_truncation)]
fn float32_bits(bits: u64) -> u32 {
    (f64::from_bits(bits) as f32).to_bits()
}

pub(super) fn source(graph: &Graph) -> String {
    let mut lower = Lowering::default();
    let mut mapped = Vec::with_capacity(graph.nodes.len());
    let (aliases, zeros) = early_aliases(graph);
    for (id, node) in graph.nodes.iter().enumerate() {
        if aliases[id] != id {
            mapped.push(mapped[aliases[id]]);
            continue;
        }
        if zeros[id] {
            mapped.push(lower.intern(Expr::Zero));
            continue;
        }
        let node = match *node {
            Node::Input(i) => Node::Input(i),
            Node::Constant(bits) => Node::Constant(bits),
            Node::Boolean(value) => Node::Boolean(value),
            Node::Integer(bits) => Node::Integer(bits),
            Node::Add(a, b) => Node::Add(mapped[a], mapped[b]),
            Node::Sub(a, b) => Node::Sub(mapped[a], mapped[b]),
            Node::Mul(a, b) => Node::Mul(mapped[a], mapped[b]),
            Node::Neg(a) => Node::Neg(mapped[a]),
            Node::Relu(a) => Node::Relu(mapped[a]),
            Node::Sin(a) => Node::Sin(mapped[a]),
            Node::Cos(a) => Node::Cos(mapped[a]),
        };
        mapped.push(lower.normalize(node));
    }
    let output = mapped[graph.output];
    let mut live = vec![false; lower.nodes.len()];
    live[output] = true;
    for id in (0..lower.nodes.len()).rev() {
        if live[id] {
            for operand in lower.operands(id) {
                live[operand] = true;
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
    for (id, node) in lower.nodes.iter().enumerate() {
        if !live[id] {
            continue;
        }
        let expression = match *node {
            Expr::Node(Node::Input(i)) => format!("x{i}[i]"),
            Expr::Node(Node::Constant(bits) | Node::Integer(bits)) | Expr::Folded(bits) => {
                format!("__uint_as_float(0x{:08x}u)", float32_bits(bits))
            }
            Expr::Zero => "__uint_as_float(0x00000000u)".into(),
            Expr::Node(Node::Boolean(value)) => if value { "1.0f" } else { "0.0f" }.into(),
            Expr::Node(Node::Add(a, b)) => lower
                .contract(a, b, false)
                .unwrap_or_else(|| format!("__fadd_rn(v{a}, v{b})")),
            Expr::Node(Node::Sub(a, b)) => lower
                .contract(a, b, true)
                .unwrap_or_else(|| format!("__fsub_rn(v{a}, v{b})")),
            Expr::Node(Node::Mul(a, b)) => format!("__fmul_rn(v{a}, v{b})"),
            Expr::Node(Node::Neg(a)) => lower.product(a).map_or_else(
                || format!("__fsub_rn(0.0f, v{a})"),
                |(x, y)| format!("fmaf(-v{x}, v{y}, 0.0f)"),
            ),
            Expr::Node(Node::Relu(a)) => format!("(v{a} < 0.0f ? 0.0f : v{a})"),
            Expr::Node(Node::Sin(a)) => format!("sinf(v{a})"),
            Expr::Node(Node::Cos(a)) => format!("cosf(v{a})"),
            Expr::Flip(a) => format!("(-v{a})"),
            Expr::SelfSub(a) => format!("__fsub_rn(v{a}, v{a})"),
        };
        writeln!(source, "const float v{id} = {expression};").unwrap();
    }
    writeln!(source, "out[i] = v{output};\n}}\n}}").unwrap();
    source
}
