//! Numerical lowering after whole-graph admission. Hash-consing precedes sign
//! normalization and per-consumer FMA selection; unused nodes are never emitted.
use super::{Graph, Node};
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Expr {
    Node(Node),
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

    fn constant(&self, id: usize) -> Option<f32> {
        match self.nodes[id] {
            Expr::Node(Node::Constant(bits)) => Some(f32::from_bits(bits)),
            _ => None,
        }
    }

    fn product(&self, id: usize) -> Option<(usize, usize)> {
        match self.nodes[id] {
            Expr::Node(Node::Mul(a, b)) => Some((a, b)),
            _ => None,
        }
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
        let bits = self.constant(*coefficient).unwrap().to_bits() ^ 0x8000_0000;
        *coefficient = self.node(Node::Constant(bits));
        Some(self.node(Node::Mul(a, b)))
    }

    fn normalize(&mut self, node: Node) -> usize {
        match node {
            Node::Mul(a, b) => {
                let coefficient = self
                    .constant(b)
                    .map(|c| (a, b, c, false))
                    .or_else(|| self.constant(a).map(|c| (b, a, c, true)));
                if let Some((value, _, scalar, reversed)) = coefficient {
                    if scalar.to_bits() == 1.0_f32.to_bits() {
                        return value;
                    }
                    if scalar.to_bits() == (-1.0_f32).to_bits() {
                        return self.intern(Expr::Flip(value));
                    }
                    // The target's signed doubling combines to -(x+x), which
                    // rounds before its consumers. Positive doubling remains a
                    // product eligible for FMA; don't let NVRTC decide this.
                    if scalar.to_bits() == (-2.0_f32).to_bits() {
                        let two = self.node(Node::Constant(2.0_f32.to_bits()));
                        let product = self.node(if reversed {
                            Node::Mul(two, value)
                        } else {
                            Node::Mul(value, two)
                        });
                        return self.intern(Expr::Flip(product));
                    }
                }
            }
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
                let value = match self.nodes[a] {
                    Expr::Node(Node::Sub(left, right)) if left != right => {
                        Some(self.node(Node::Sub(right, left)))
                    }
                    Expr::Node(Node::Neg(value)) | Expr::Flip(value) => Some(value),
                    Expr::Node(_) => None,
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
        // Equal expressions share a rounded value. A self-subtraction must not
        // become fma(x,y,-round(x*y)); infinity-infinity still produces NaN.
        if subtract && a == b {
            return None;
        }
        if let Some((x, y)) = self.product(a) {
            return Some(format!(
                "fmaf(v{x}, v{y}, {}v{b})",
                if subtract { "-" } else { "" }
            ));
        }
        self.product(b)
            .map(|(x, y)| format!("fmaf({}v{x}, v{y}, v{a})", if subtract { "-" } else { "" }))
    }

    fn operands(&self, id: usize) -> Vec<usize> {
        match self.nodes[id] {
            Expr::Node(Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) => vec![a, b],
            Expr::Node(Node::Neg(a)) => {
                self.product(a).map_or_else(|| vec![a], |(x, y)| vec![x, y])
            }
            Expr::Node(Node::Relu(a) | Node::Sin(a) | Node::Cos(a)) | Expr::Flip(a) => vec![a],
            Expr::Node(Node::Input(_) | Node::Constant(_)) => vec![],
        }
    }
}

pub(super) fn source(graph: &Graph) -> String {
    let mut lower = Lowering::default();
    let mut mapped = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let node = match *node {
            Node::Input(i) => Node::Input(i),
            Node::Constant(bits) => Node::Constant(bits),
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
            Expr::Node(Node::Constant(bits)) => format!("__uint_as_float(0x{bits:08x}u)"),
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
        };
        writeln!(source, "const float v{id} = {expression};").unwrap();
    }
    writeln!(source, "out[i] = v{output};\n}}\n}}").unwrap();
    source
}
