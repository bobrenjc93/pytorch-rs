//! Numerical lowering after whole-graph admission. Hash-consing precedes sign
//! normalization and per-consumer FMA selection; unused nodes are never emitted.
use super::{Graph, Node};
use std::collections::HashMap;
use std::fmt::Write;

const LOAD_RANK_BASE: usize = 1 << 16;

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
    // Keep the original negative double separate from its positive factors.
    // Shared raw uses round before addition; subtraction may cancel the sign.
    SignedDouble(usize),
}

#[derive(Clone, Copy)]
struct ContractionProduct {
    factors: (usize, usize),
    negative: bool,
    direct: bool,
}

#[derive(Default)]
struct Lowering {
    nodes: Vec<Expr>,
    interned: HashMap<Expr, usize>,
    constant_expressions: Vec<bool>,
}

impl Lowering {
    fn intern(&mut self, expr: Expr) -> usize {
        if let Some(&id) = self.interned.get(&expr) {
            return id;
        }
        let id = self.nodes.len();
        self.interned.insert(expr.clone(), id);
        self.nodes.push(expr);
        self.constant_expressions.push(
            !matches!(
                self.nodes[id],
                Expr::Node(Node::Input(_) | Node::RuntimeScalar(_, _))
            ) && self
                .operands(id)
                .iter()
                .all(|&operand| self.constant_expressions[operand]),
        );
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

    fn subtracts_scalar_zero(&self, id: usize) -> Option<usize> {
        match self.nodes[id] {
            Expr::Node(Node::Sub(a, b))
                if self.constant(b).is_some_and(|value| value.to_bits() == 0) =>
            {
                Some(a)
            }
            _ => None,
        }
    }

    fn contraction_product(&self, mut id: usize) -> Option<ContractionProduct> {
        // Decode only exact sign flips and scalar +0 subtraction, after
        // expression identity checks. Neither addition of zero nor tensor-zero
        // provenance is transparent here.
        let (mut negative, mut flipped) = (false, false);
        let (mut outer_zero, mut inner_zero) = (false, false);
        loop {
            if let Some(value) = self.subtracts_scalar_zero(id) {
                if flipped {
                    inner_zero = true;
                } else {
                    outer_zero = true;
                }
                id = value;
            } else if let Expr::Flip(value) = self.nodes[id] {
                negative = !negative;
                flipped = true;
                id = value;
            } else {
                break;
            }
        }
        self.product(id).map(|factors| ContractionProduct {
            factors,
            negative,
            // Cancelled flips expose the original positive product. A single
            // effective flip over a zero-subtracted expression retains its
            // fallback priority; leading zero subtraction alone exposes it.
            direct: !negative || (outer_zero && !inner_zero),
        })
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

    fn unit_product(&self, node: &Node) -> Option<usize> {
        let Node::Mul(a, b) = *node else {
            return None;
        };
        if self.constant(b) == Some(1.0) {
            Some(a)
        } else if self.constant(a) == Some(1.0) {
            Some(b)
        } else {
            None
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
    fn positive(&mut self, id: usize, single_use: bool) -> Option<usize> {
        if let Expr::Flip(a) = self.nodes[id] {
            return Some(a);
        }
        let (mut a, mut b) = self.product(id)?;
        // InstCombine extracts a negated factor at Add/Sub consumers only
        // when the signed multiply itself has one use. A shared signed result
        // keeps its identity even when its original positive product had one use.
        if single_use {
            if let Expr::Flip(value) = self.nodes[a] {
                return Some(self.node(Node::Mul(value, b)));
            }
            if let Expr::Flip(value) = self.nodes[b] {
                return Some(self.node(Node::Mul(a, value)));
            }
        }
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

    fn normalize(&mut self, node: Node, last_use: [bool; 2], uses: [usize; 2]) -> usize {
        // Keep arithmetic on known constant tensors distinct from scalar
        // nodes and runtime expressions. In particular, later negation must
        // invert the constant's sign, not emit positive-zero subtraction or
        // contract a multiplication with an added positive zero. Non-arithmetic
        // operations still use their native libdevice/intrinsic lowering.
        if let Some(bits) = self.fold_tensor_arithmetic(&node) {
            return self.intern(Expr::Folded(bits));
        }
        let node = self.float_operands(node);
        if let Some(value) = self.unit_product(&node) {
            return value;
        }
        let expose = |this: &mut Self, id, eligible| {
            if eligible && let Expr::SignedDouble(product) = this.nodes[id] {
                this.intern(Expr::Flip(product))
            } else {
                id
            }
        };
        // Earlier live additions consume the rounded signed double. Its last
        // consumer, and subtraction/negation consumers, may expose the factors.
        let node = match node {
            Node::Add(a, b) => {
                Node::Add(expose(self, a, last_use[0]), expose(self, b, last_use[1]))
            }
            Node::Sub(a, b) if a != b => Node::Sub(expose(self, a, true), expose(self, b, true)),
            Node::Neg(a) => Node::Neg(expose(self, a, true)),
            other => other,
        };
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
                    if scalar.to_bits() == (-1.0_f64).to_bits() {
                        // LLVM visitFNeg moves the sign into a single-use
                        // tensor multiply before rewriting its consumers.
                        // Count the original product's uses, not the signed
                        // result's uses or uses introduced by normalization.
                        if uses[usize::from(reversed)] == 1
                            && let Some((a, b)) = self.product(value)
                            && [a, b].iter().all(|&factor| {
                                !self.constant_expressions[factor]
                                    && !matches!(
                                        self.nodes[factor],
                                        Expr::Node(Node::RuntimeScalar(_, _))
                                    )
                            })
                        {
                            let negative = self.intern(Expr::Flip(b));
                            return self.node(Node::Mul(a, negative));
                        }
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
                        return self.intern(Expr::SignedDouble(product));
                    }
                }
            }
            Node::Sub(a, b) if a == b => return self.intern(Expr::SelfSub(a)),
            Node::Add(a, b) => {
                if let Some(positive) = self.positive(b, uses[1] == 1) {
                    return self.node(Node::Sub(a, positive));
                }
                if let Some(positive) = self.positive(a, uses[0] == 1) {
                    return self.node(Node::Sub(b, positive));
                }
            }
            Node::Sub(a, b) if a != b => {
                if let Some(positive) = self.positive(b, uses[1] == 1) {
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
                    Expr::Node(_)
                    | Expr::SelfSub(_)
                    | Expr::Zero
                    | Expr::Folded(_)
                    | Expr::SignedDouble(_) => None,
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

    fn contraction_ranks(&self, input_ranks: &[usize]) -> Vec<Option<usize>> {
        // LLVM Reassociate orders commutative operands by dependency rank
        // before NVPTX selects an FMA. Model arithmetic/select dependencies,
        // not expression size: shared operands do not increase the rank twice.
        // Libdevice's internal control flow is outside this rank model.
        let mut ranks: Vec<Option<usize>> = Vec::with_capacity(self.nodes.len());
        for (id, expr) in self.nodes.iter().enumerate() {
            let rank = match *expr {
                Expr::Node(Node::Input(i)) => Some(input_ranks[i]),
                // Reference runtime scalars are kernel arguments, below the
                // separately ranked, side-effecting tensor loads.
                Expr::Node(Node::RuntimeScalar(i, _)) => Some(i + 1),
                Expr::Node(Node::Constant(_) | Node::Integer(_) | Node::Boolean(_))
                | Expr::Zero
                | Expr::Folded(_) => Some(0),
                Expr::Node(Node::Sub(a, _)) if self.subtracts_scalar_zero(id).is_some() => ranks[a],
                Expr::Node(Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) => {
                    ranks[a].zip(ranks[b]).map(|(a, b)| a.max(b) + 1)
                }
                Expr::Node(Node::Relu(a)) => ranks[a].map(|rank| rank + 2), // compare, select
                Expr::Node(Node::Neg(a)) | Expr::SelfSub(a) => ranks[a].map(|rank| rank + 1),
                Expr::Flip(a) | Expr::SignedDouble(a) => ranks[a], // actual fneg
                Expr::Node(Node::Sin(_) | Node::Cos(_)) => None,
            };
            ranks.push(rank);
        }
        ranks
    }

    fn contract(
        &self,
        a: usize,
        b: usize,
        subtract: bool,
        ranks: &[Option<usize>],
    ) -> Option<String> {
        let left = self.contraction_product(a);
        let right = self.contraction_product(b);
        // Canonicalize only competing positive products of an addition. Signed
        // products have already undergone a different normalization phase, and
        // subtraction is not commutative. Ties retain the original orientation.
        if !subtract
            && left.is_some_and(|p| p.direct && !p.negative)
            && right.is_some_and(|p| p.direct && !p.negative)
            && ranks[a].zip(ranks[b]).is_some_and(|(a, b)| b < a)
        {
            return self.contract(b, a, false, ranks);
        }
        // Prefer direct products, preserving left-to-right order when both
        // candidates are sign-flipped even if only the right is zero-wrapped.
        let preferred_left = left.filter(|product| {
            product.direct || (product.negative && right.is_some_and(|other| other.negative))
        });
        let (product, on_right) = preferred_left
            .map(|product| (product, false))
            .or_else(|| {
                right
                    .filter(|product| product.direct)
                    .map(|product| (product, true))
            })
            .or_else(|| left.map(|product| (product, false)))
            .or_else(|| right.map(|product| (product, true)))?;
        let (x, y) = product.factors;
        let addend = if on_right { a } else { b };
        Some(format!(
            "fmaf({}v{x}, v{y}, {}v{addend})",
            if product.negative ^ (on_right && subtract) {
                "-"
            } else {
                ""
            },
            if subtract && !on_right { "-" } else { "" }
        ))
    }

    fn operands(&self, id: usize) -> Vec<usize> {
        match self.nodes[id] {
            Expr::Node(Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) => vec![a, b],
            Expr::Node(Node::Neg(a)) => {
                self.product(a).map_or_else(|| vec![a], |(x, y)| vec![x, y])
            }
            Expr::Node(Node::Relu(a) | Node::Sin(a) | Node::Cos(a))
            | Expr::Flip(a)
            | Expr::SelfSub(a)
            | Expr::SignedDouble(a) => vec![a],
            Expr::Node(
                Node::Input(_)
                | Node::RuntimeScalar(_, _)
                | Node::Constant(_)
                | Node::Boolean(_)
                | Node::Integer(_),
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
            Node::Constant(_) | Node::Integer(_) | Node::Boolean(_) | Node::RuntimeScalar(_, _)
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

fn input_load_ranks(graph: &Graph, aliases: &[usize], zeros: &[bool]) -> Vec<usize> {
    // Reference loads are emitted on first live use, not in positional-input
    // order. Traverse before sign normalization can reverse operand orientation.
    let mut ranks = vec![0; graph.inputs];
    let mut seen = vec![false; graph.nodes.len()];
    let mut pending = vec![graph.output];
    let mut rank = LOAD_RANK_BASE;
    while let Some(id) = pending.pop() {
        let id = aliases[id];
        if seen[id] || zeros[id] {
            continue;
        }
        seen[id] = true;
        match graph.nodes[id] {
            Node::Input(i) if ranks[i] == 0 => {
                rank += 1;
                ranks[i] = rank;
            }
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                pending.extend([b, a]);
            }
            Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a) => pending.push(a),
            _ => {}
        }
    }
    ranks
}

#[allow(clippy::cast_possible_truncation)]
fn float32_bits(bits: u64) -> u32 {
    (f64::from_bits(bits) as f32).to_bits()
}

fn remap(node: &Node, mapped: &[usize]) -> Node {
    match *node {
        Node::Input(i) => Node::Input(i),
        Node::RuntimeScalar(i, negative) => Node::RuntimeScalar(i, negative),
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
    }
}

// Find each original expression's last canonical live consumer before sign
// normalization. Dead and duplicate expressions cannot impose rounding boundaries.
fn consumer_analysis(
    graph: &Graph,
    aliases: &[usize],
    zeros: &[bool],
) -> (Vec<[bool; 2]>, Vec<[usize; 2]>) {
    let mut canonical = Lowering::default();
    let mut mapped = Vec::with_capacity(graph.nodes.len());
    for (id, node) in graph.nodes.iter().enumerate() {
        let value = if aliases[id] != id {
            mapped[aliases[id]]
        } else if zeros[id] {
            canonical.intern(Expr::Zero)
        } else {
            let node = remap(node, &mapped);
            if let Some(bits) = canonical.fold_tensor_arithmetic(&node) {
                canonical.intern(Expr::Folded(bits))
            } else {
                let node = canonical.float_operands(node);
                if let Some(value) = canonical.unit_product(&node) {
                    value
                } else {
                    canonical.node(node)
                }
            }
        };
        mapped.push(value);
    }
    let mut live = vec![false; canonical.nodes.len()];
    let mut last = vec![0; canonical.nodes.len()];
    let mut uses = vec![0; canonical.nodes.len()];
    live[mapped[graph.output]] = true;
    for id in (0..canonical.nodes.len()).rev() {
        if !live[id] {
            continue;
        }
        let operands = match canonical.nodes[id] {
            Expr::Node(Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) => vec![a, b],
            Expr::Node(Node::Neg(a) | Node::Relu(a) | Node::Sin(a) | Node::Cos(a)) => vec![a],
            _ => vec![],
        };
        for operand in operands {
            live[operand] = true;
            last[operand] = last[operand].max(id);
            uses[operand] += 1;
        }
    }
    let last = mapped
        .iter()
        .map(|&id| match canonical.nodes[id] {
            Expr::Node(Node::Add(a, b)) => [last[a] == id, last[b] == id],
            _ => [true, true],
        })
        .collect();
    let uses = mapped
        .iter()
        .map(|&id| match canonical.nodes[id] {
            Expr::Node(Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b)) => [uses[a], uses[b]],
            _ => [0, 0],
        })
        .collect();
    (last, uses)
}

pub(super) fn source(graph: &Graph) -> String {
    let mut lower = Lowering::default();
    let mut mapped = Vec::with_capacity(graph.nodes.len());
    let (aliases, zeros) = early_aliases(graph);
    let (last, uses) = consumer_analysis(graph, &aliases, &zeros);
    for (id, node) in graph.nodes.iter().enumerate() {
        if aliases[id] != id {
            mapped.push(mapped[aliases[id]]);
            continue;
        }
        if zeros[id] {
            mapped.push(lower.intern(Expr::Zero));
            continue;
        }
        let node = remap(node, &mapped);
        mapped.push(lower.normalize(node, last[id], uses[id]));
    }
    let output = mapped[graph.output];
    let ranks = lower.contraction_ranks(&input_load_ranks(graph, &aliases, &zeros));
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
         const float* x0, const float* x1, float* out, unsigned long long n",
    );
    for index in 0..graph.scalar_count() {
        write!(source, ", float s{index}").unwrap();
    }
    source.push_str(
        ") {\n\
         for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
         i < n; i += (unsigned long long)blockDim.x * gridDim.x) {\n",
    );
    for (id, node) in lower.nodes.iter().enumerate() {
        if !live[id] {
            continue;
        }
        let expression = match *node {
            Expr::Node(Node::Input(i)) => format!("x{i}[i]"),
            Expr::Node(Node::RuntimeScalar(i, negative)) => {
                format!("{}s{i}", if negative { "-" } else { "" })
            }
            Expr::Node(Node::Constant(bits) | Node::Integer(bits)) | Expr::Folded(bits) => {
                format!("__uint_as_float(0x{:08x}u)", float32_bits(bits))
            }
            Expr::Zero => "__uint_as_float(0x00000000u)".into(),
            Expr::Node(Node::Boolean(value)) => if value { "1.0f" } else { "0.0f" }.into(),
            Expr::Node(Node::Add(a, b)) => lower
                .contract(a, b, false, &ranks)
                .unwrap_or_else(|| format!("__fadd_rn(v{a}, v{b})")),
            Expr::Node(Node::Sub(a, _)) if lower.subtracts_scalar_zero(id).is_some() => {
                format!("v{a}")
            }
            Expr::Node(Node::Sub(a, b)) => lower
                .contract(a, b, true, &ranks)
                .unwrap_or_else(|| format!("__fsub_rn(v{a}, v{b})")),
            Expr::Node(Node::Mul(a, b)) => format!("__fmul_rn(v{a}, v{b})"),
            Expr::Node(Node::Neg(a)) => lower.product(a).map_or_else(
                || format!("__fsub_rn(0.0f, v{a})"),
                |(x, y)| format!("fmaf(-v{x}, v{y}, 0.0f)"),
            ),
            Expr::Node(Node::Relu(a)) => format!("(v{a} < 0.0f ? 0.0f : v{a})"),
            Expr::Node(Node::Sin(a)) => {
                if lower.constant_expressions[a] {
                    // Reference constant folding rounds a high-precision unary
                    // result to f32. Keep both input and output materialization
                    // boundaries; sinf's allowed ULP error can amplify later.
                    format!("(float)sin((double)v{a})")
                } else {
                    // Match runtime reference sine's input FTZ boundary without
                    // approximate range reduction or flushing other arithmetic.
                    format!(
                        "sinf((__float_as_uint(v{a}) & 0x7f800000u) == 0 ? __uint_as_float(__float_as_uint(v{a}) & 0x80000000u) : v{a})"
                    )
                }
            }
            Expr::Node(Node::Cos(a)) if lower.constant_expressions[a] => {
                format!("(float)cos((double)v{a})")
            }
            Expr::Node(Node::Cos(a)) => format!("cosf(v{a})"),
            Expr::Flip(a) | Expr::SignedDouble(a) => format!("(-v{a})"),
            Expr::SelfSub(a) => format!("__fsub_rn(v{a}, v{a})"),
        };
        writeln!(source, "const float v{id} = {expression};").unwrap();
    }
    writeln!(source, "out[i] = v{output};\n}}\n}}").unwrap();
    source
}
